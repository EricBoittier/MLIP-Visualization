# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "torchani", "safetensors", "numpy"]
# ///
"""Convert TorchANI's ANI-2x (MIT licence) for MLIP-Visualization.

    uv run scripts/convert_ani.py --out public/models/ani-2x

Writes <out>.json (elements, AEV constants, self energies, layer sizes) and
<out>.safetensors (weights, named member{m}.{element}.{layer}.weight/bias).
"""
import argparse
import json
from pathlib import Path

import torch
import torchani
from safetensors.torch import save_file
from torchani.units import HARTREE_TO_EV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = torchani.models.ANI2x()
    sd = model.state_dict()
    aev = "potentials.nnp.aev_computer."
    nn = "potentials.nnp.neural_networks.members."
    symbols = list(model.symbols)
    numbers = sd["atomic_numbers"].tolist()
    members = len({k.split(".")[4] for k in sd if k.startswith(nn)})

    tensors, layers = {}, {}
    for m in range(members):
        for s in symbols:
            pre = f"{nn}{m}.atomics.{s}."
            names = sorted({k[len(pre):].rsplit(".", 1)[0] for k in sd if k.startswith(pre)},
                           key=lambda n: (n == "final_layer", n))
            dims = []
            for k, name in enumerate(names):
                w, b = sd[f"{pre}{name}.weight"], sd[f"{pre}{name}.bias"]
                tensors[f"member{m}.{s}.{k}.weight"] = w.float().contiguous()
                tensors[f"member{m}.{s}.{k}.bias"] = b.float().contiguous()
                dims.append(list(w.shape))
            layers[s] = [d[1] for d in dims] + [dims[-1][0]]

    meta = {
        "kind": "ani",
        "name": "ANI-2x",
        "source": f"torchani {torchani.__version__}, models.ANI2x()",
        "elements": numbers,
        "symbols": symbols,
        "members": members,
        "layers": layers,
        "activation": {"name": "celu", "alpha": 0.1},
        "radial": {"cutoff": model.potentials["nnp"].aev_computer.radial.cutoff,
                   "eta": sd[aev + "radial.eta"].item(), "shifts": sd[aev + "radial.shifts"].tolist()},
        "angular": {"cutoff": model.potentials["nnp"].aev_computer.angular.cutoff,
                    "eta": sd[aev + "angular.eta"].item(), "zeta": sd[aev + "angular.zeta"].item(),
                    "shifts": sd[aev + "angular.shifts"].tolist(), "sections": sd[aev + "angular.sections"].tolist()},
        "self_energies": sd["energy_shifter.self_energies"].tolist(),  # Hartree
        "hartree_to_ev": HARTREE_TO_EV,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / (out.name + ".json")).write_text(json.dumps(meta, indent=1))
    save_file(tensors, str(out.parent / (out.name + ".safetensors")))
    print(f"wrote {out}.json/.safetensors: {len(tensors)} tensors, layers {layers}")


if __name__ == "__main__":
    main()
