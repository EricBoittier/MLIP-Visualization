# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "safetensors"]
# ///
"""Export a physnetjax PhysNet (mmml) for MLIP-Visualization.

    uv run scripts/export_physnet.py --params params.json --out public/models/my-physnet [--elements 1,6,8]

`params.json` is physnetjax's portable JSON ({"params": ..., "config": ...}). Only the
invariant model (max_degree = 0) is supported: with max_degree >= 1 the l = 1 channels
feed the energy, and dropping them would silently change it. Train with
`max_degree: 0` (e.g. `mmml physnet-train`) for a model this viewer can show.

Writes <out>.json (config, elements) and <out>.safetensors, with flax kernels transposed to
[out, in] and parameter paths joined by '.'.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

SUPPORTED = {"features", "max_degree", "num_iterations", "num_basis_functions", "cutoff", "max_atomic_number",
             "charges", "n_refinement_blocks", "n_res", "zbl", "zbl_cuton", "zbl_cutoff", "trainable_zbl",
             "use_energy_bias", "include_electrostatics", "electrostatics_damping_sigma", "switch_start",
             "switch_end", "electrostatics_off_start", "electrostatics_off_end", "total_charge", "natoms",
             "max_padded_atoms", "debug", "efa", "use_pbc"}


def flatten(tree, prefix=""):
    for k, v in tree.items():
        name = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            yield from flatten(v, name)
        else:
            yield name, np.asarray(v, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--elements", default="1,6,7,8,9,16,17", help="atomic numbers the model was trained on")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    data = json.loads(Path(args.params).read_text())
    config, params = data["config"], data["params"]
    if config.get("max_degree", 0) != 0:
        raise SystemExit(f"max_degree = {config['max_degree']}: only invariant PhysNet (max_degree = 0) is supported; "
                         "its l >= 1 channels would otherwise be dropped and the energy would change.")
    if config.get("efa"):
        raise SystemExit("Euclidean fast attention (efa) is not supported.")
    if config.get("n_refinement_blocks", config.get("n_res", 1)) < 0:
        raise SystemExit("negative n_refinement_blocks (attention refinement) is not supported.")
    if config.get("use_pbc"):
        raise SystemExit("periodic PhysNet is not supported yet.")
    unknown = set(config) - SUPPORTED
    if unknown:
        print(f"note: ignoring config keys {sorted(unknown)}")
    tensors = {}
    for name, a in flatten(params):
        if name.endswith("embedding"):  # (Z+1, 1, 1, F)
            a = a.reshape(a.shape[0], a.shape[-1])
        elif name.endswith("tensor.kernel"):  # (1, 1, 1, 1, 1, 1, F)
            a = a.reshape(1, -1)
        elif name.endswith("kernel") and a.ndim == 2:
            a = a.T.copy()  # flax [in, out] -> [out, in]
        elif a.ndim == 0:
            a = a.reshape(1)
        tensors[name] = np.ascontiguousarray(a)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"kind": "physnet", "name": args.name or out.name, "source": str(args.params), "config": config,
            "elements": [int(z) for z in args.elements.split(",")]}
    (out.parent / (out.name + ".json")).write_text(json.dumps(meta, indent=1))
    save_file(tensors, str(out.parent / (out.name + ".safetensors")))
    print(f"wrote {out}.json/.safetensors: {len(tensors)} tensors")


if __name__ == "__main__":
    main()
