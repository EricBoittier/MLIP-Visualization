"""Export a small LOREM model from metatrain's experimental port for MLIP-Visualization.

Run it with the interpreter that has that port installed (the metatrain ``lorem-tests`` tox env):

    /path/to/metatrain/.tox/lorem-tests/bin/python scripts/convert_lorem.py \\
        --out public/models/lorem-demo --ref tests/reference

The checkpoint is a seeded random initialization of the architecture in
``metatrain.experimental.lorem`` (short-range density + long-range Coulomb head),
not a trained potential: it exists so the visualiser can run the real forward.
Reference energies and forces for water, ethanol, NaCl, silicon and dimethyl sulfide
are written next to it, evaluated by that same PyTorch model in float32.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ase.build import bulk, molecule
from metatomic.torch import ModelOutput, System
from metatrain.experimental.lorem import LOREM
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists
def save_safetensors(tensors: dict, path: Path) -> None:
    """Float32 safetensors, the layout src/common/safetensors.ts reads. The lorem env has no safetensors package."""
    header, blobs, off = {}, [], 0
    for name, arr in tensors.items():
        raw = np.ascontiguousarray(arr, dtype=np.float32).tobytes()
        header[name] = {"dtype": "F32", "shape": list(arr.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    text = json.dumps(header, separators=(",", ":"))
    text += " " * ((8 - ((8 + len(text)) % 8)) % 8)
    body = text.encode()
    path.write_bytes(len(body).to_bytes(8, "little") + body + b"".join(blobs))

SEED = 0
ELEMENTS = [1, 6, 7, 8, 11, 14, 16, 17]  # H C N O Na Si S Cl — the presets, plus S


def kernel_of(mod) -> np.ndarray:
    """Learned tensor weight expanded onto (ℓ, m) and multiplied by the Clebsch–Gordan tensor."""
    weight = mod.tensor_weight.detach()
    gathered = weight.reshape(-1, weight.shape[-1])[mod.weight_index.reshape(-1)]
    gathered = gathered.reshape(*mod.weight_index.shape, weight.shape[-1])
    ker = gathered * mod.cg.to(dtype=gathered.dtype).unsqueeze(-1)
    return np.ascontiguousarray(ker.float().cpu().numpy())


def export_tensors(model) -> dict:
    skip, tensors = set(), {}
    for name, mod in model.named_modules():
        if type(mod).__name__ == "_DegreeWiseLinear":
            # [ℓ, in, out] -> [ℓ, out, in], the layout every other linear layer uses
            w = mod.weight.detach().transpose(-1, -2).contiguous().float().cpu().numpy()
            tensors[f"{name}.weight"] = np.ascontiguousarray(w)
            skip.add(f"{name}.weight")
            if mod.bias is not None:
                tensors[f"{name}.bias"] = np.ascontiguousarray(mod.bias.detach().float().cpu().numpy())
                skip.add(f"{name}.bias")
        if type(mod).__name__ in ("TensorDense", "TensorProduct"):
            tensors[f"{name}.kernel"] = kernel_of(mod)
            skip.add(f"{name}.tensor_weight")
    for name, param in model.named_parameters():
        if not name.startswith(("sr.", "lr.")) or name in skip:
            continue
        tensors[name] = np.ascontiguousarray(param.detach().float().cpu().numpy())
    return tensors


def system_of(atoms) -> System:
    return System(
        types=torch.tensor(atoms.numbers, dtype=torch.int32),
        positions=torch.tensor(np.asarray(atoms.positions), dtype=torch.float32),
        cell=torch.tensor(np.asarray(atoms.cell.array), dtype=torch.float32),
        pbc=torch.tensor(atoms.pbc.tolist()),
    )


def cases():
    rng = np.random.default_rng(11)
    out = {}
    for name, atoms in dict(
        water=molecule("H2O"), ethanol=molecule("CH3CH2OH"),
        nacl=bulk("NaCl", "rocksalt", a=5.6), silicon=bulk("Si", "diamond", a=5.43),
        sulfur=molecule("CH3SCH3"),
    ).items():
        atoms.positions += rng.normal(0, 0.06, atoms.positions.shape)
        out[name] = atoms
    return out


def evaluate(model, atoms):
    system = get_system_with_neighbor_lists(system_of(atoms), model.requested_neighbor_lists())
    system.positions.requires_grad_(True)
    published = model([system], {"energy": ModelOutput(sample_kind="system")})["energy"].block().values.sum()
    nodes, distances, spherical, sr, _ = model.sr([system])
    lr, _ = model.lr([system], nodes, distances, spherical)
    atomic = sr + lr
    if not torch.allclose(atomic.sum(), published, rtol=1e-5, atol=1e-5):
        raise SystemExit(f"scaler/composition moved the energy: raw {atomic.sum().item()} published {published.item()}")
    atomic.sum().backward()
    forces = (-system.positions.grad).detach()
    nl = system.get_neighbor_list(model.requested_neighbor_lists()[0]).samples.values
    shifts = nl[:, 2:].to(system.positions.dtype)
    vectors = system.positions[nl[:, 1]] - system.positions[nl[:, 0]] + shifts @ system.cell
    return {
        "energy": float(published.detach()),
        "node_energy": atomic.detach().cpu().tolist(),
        "sr": sr.detach().cpu().tolist(),
        "lr": lr.detach().cpu().tolist(),
        "forces": forces.cpu().tolist(),
        "edges": int(nl.shape[0]),
        "center": nl[:, 0].cpu().tolist(),
        "neighbor": nl[:, 1].cpu().tolist(),
        "vector": vectors.detach().cpu().tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="public/models/lorem-demo")
    ap.add_argument("--ref", default="tests/reference")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    torch.set_default_dtype(torch.float32)

    hypers = dict(get_default_hypers("experimental.lorem")["model"])
    hypers.update(cutoff=5.0, max_degree=2, max_degree_lr=1, num_features=16, num_spherical_features=4,
                  num_radial=6, num_species=8, num_message_passing=1, equivariant_message_passing=True,
                  initialize_node_features=True)
    info = DatasetInfo(
        length_unit="Angstrom", atomic_types=ELEMENTS,
        targets={"energy": get_energy_target_info("energy", {"quantity": "energy", "unit": "eV"})},
    )
    model = LOREM(hypers, info).eval()
    tensors = export_tensors(model)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "kind": "lorem",
        "name": "LOREM demo",
        "atomic_numbers": ELEMENTS,
        "embedding_rows": int(model.sr.chemical_embedding.num_embeddings),
        "cutoff": hypers["cutoff"],
        "max_degree": hypers["max_degree"],
        "max_degree_lr": hypers["max_degree_lr"],
        "num_features": hypers["num_features"],
        "num_spherical_features": hypers["num_spherical_features"],
        "num_radial": hypers["num_radial"],
        "num_species": hypers["num_species"],
        "num_message_passing": hypers["num_message_passing"],
        "equivariant_message_passing": hypers["equivariant_message_passing"],
        "initialize_node_features": hypers["initialize_node_features"],
        "smearing": hypers["cutoff"] / 4,
        "lr_wavelength": hypers["cutoff"] / 8,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    save_safetensors(tensors, out.with_suffix(".safetensors"))
    n = sum(v.size for v in tensors.values())
    print(f"wrote {out.with_suffix('.json').name}: {len(tensors)} tensors, {n} parameters")

    ref = Path(args.ref)
    ref.mkdir(parents=True, exist_ok=True)
    for name, atoms in cases().items():
        got = evaluate(model, atoms)
        payload = {
            "case": name,
            "atomic_numbers": atoms.numbers.tolist(),
            "positions": np.asarray(atoms.positions, dtype=np.float64).tolist(),
            "cell": np.asarray(atoms.cell.array, dtype=np.float64).tolist(),
            "pbc": atoms.pbc.tolist(),
            **got,
        }
        (ref / f"lorem-demo_{name}.json").write_text(json.dumps(payload) + "\n")
        print(f"  {name:8} E={got['energy']:+.6f}  |F|max={np.abs(got['forces']).max():.4f}  edges={got['edges']}")


if __name__ == "__main__":
    main()
