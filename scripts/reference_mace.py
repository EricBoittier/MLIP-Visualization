# /// script
# requires-python = ">=3.10"
# dependencies = ["mace-torch", "numpy", "huggingface_hub"]
# ///
"""Reference outputs of mace-torch (float64) for the MLIP-Visualization port of MACE.

    uv run scripts/reference_mace.py --model mace-mp-0b3-medium --out tests/reference

Writes <out>/<name>_<case>.json: energy, forces and per-atom energies; for the molecules also
the intermediates, in the app's layout (one [atoms * (2l + 1), channels] block per l):
the spherical harmonics and radial features of every edge, each interaction's output, each
product's output, the readouts and the ZBL pair energies.
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from ase.build import bulk, molecule

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)


def blocks_to_ls(x, irreps):
    """[N, irreps.dim] (e3nn layout: per block [mul, 2l+1]) -> {l: [N (2l+1), mul]}."""
    out, off = {}, 0
    for mul, ir in irreps:
        d = ir.dim
        out[ir.l] = x[:, off:off + mul * d].reshape(len(x), mul, d).transpose(0, 2, 1).reshape(-1, mul)
        off += mul * d
    return out


def r7(a):
    """8 significant digits (some intermediates, like ZBL energies at bond lengths, are ~1e-7 eV)."""
    a = np.asarray(a, dtype=float)
    return np.vectorize(lambda x: float(f"{x:.8g}"))(a).tolist() if a.ndim else float(f"{a:.8g}")


def cases():
    rng = np.random.default_rng(11)
    out = {}
    for name, atoms in dict(water=molecule("H2O"), ethanol=molecule("CH3CH2OH"), nacl=bulk("NaCl", "rocksalt", a=5.6),
                            silicon=bulk("Si", "diamond", a=5.43), sulfur=molecule("CH3SCH3")).items():
        atoms.positions += rng.normal(0, 0.06, atoms.positions.shape)
        out[name] = atoms
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mace-mp-0b3-medium")
    ap.add_argument("--out", default="tests/reference")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    from mace import data
    from mace.tools import torch_geometric, utils

    path = Path(args.model)
    if not path.exists():
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download("mace-foundations/mace-mp-0", f"{args.model}.model"))
    m = torch.load(path, map_location="cpu", weights_only=False).double().eval()
    name = args.name or path.stem
    z_table = utils.AtomicNumberTable(m.atomic_numbers.tolist())
    seen = {}
    hook = lambda key: (lambda mod, inp, out: seen.__setitem__(key, out))
    m.spherical_harmonics.register_forward_hook(hook("sh"))
    m.radial_embedding.register_forward_hook(hook("radial"))
    if hasattr(m, "pair_repulsion_fn"):
        m.pair_repulsion_fn.register_forward_hook(hook("pair"))
    for i, (it, pr, ro) in enumerate(zip(m.interactions, m.products, m.readouts)):
        it.register_forward_hook(hook(f"interaction{i}"))
        pr.register_forward_hook(hook(f"product{i}"))
        ro.register_forward_hook(hook(f"readout{i}"))
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    for case, atoms in cases().items():
        cfg = data.config_from_atoms(atoms)
        ad = data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(m.r_max), heads=["default"])
        batch = next(iter(torch_geometric.dataloader.DataLoader([ad], batch_size=1)))
        res = m(batch.to_dict(), compute_force=True)
        ref = dict(model=name, case=case, atomic_numbers=atoms.numbers.tolist(), positions=r7(atoms.positions),
                   energy=float(res["energy"]), forces=r7(res["forces"].detach()), node_energy=r7(res["node_energy"].detach()))
        if atoms.pbc.any():
            ref.update(cell=r7(atoms.cell.array), pbc=atoms.pbc.tolist())
        s, r = batch.edge_index.numpy()
        ref["edges"] = len(s)
        if case in ("water", "ethanol"):
            pos = batch.positions.detach().numpy()
            vec = pos[r] - pos[s] + batch.shifts.detach().numpy()
            ref["edge_list"] = dict(receiver=r.tolist(), sender=s.tolist(), vector=r7(vec))
            ref["sh"] = r7(seen["sh"].detach())
            ref["radial"] = r7(seen["radial"][0].detach())
            if "pair" in seen:
                ref["pair"] = r7(seen["pair"].detach())
            ref["layers"] = []
            for i, (it, pr) in enumerate(zip(m.interactions, m.products)):
                msg, sc = seen[f"interaction{i}"]  # [N, C, (lmax + 1)^2]
                msg = msg.detach().numpy()
                lofs = np.cumsum([0] + [ir.dim for _, ir in it.irreps_out])
                layer = dict(message={l: r7(msg[:, :, lofs[k]:lofs[k + 1]].transpose(0, 2, 1).reshape(-1, msg.shape[1]))
                                      for k, (_, ir) in enumerate(it.irreps_out) for l in [ir.l]},
                             product={l: r7(v) for l, v in blocks_to_ls(seen[f"product{i}"].detach().numpy(), pr.linear.irreps_out).items()},
                             readout=r7(seen[f"readout{i}"].detach()[:, 0]))
                if sc is not None:
                    layer["sc"] = {l: r7(v) for l, v in blocks_to_ls(sc.detach().numpy(), pr.linear.irreps_out).items()}
                ref["layers"].append(layer)
        (outdir / f"{name}_{case}.json").write_text(json.dumps(ref))
        print(f"{name} {case:8s} E = {ref['energy']:.6f} eV, {ref['edges']} edges")


if __name__ == "__main__":
    main()
