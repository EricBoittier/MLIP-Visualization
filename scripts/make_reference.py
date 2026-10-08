# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "metatrain", "upet", "metatomic-torch", "ase", "huggingface_hub"]
# ///
"""Reference energies/forces/stresses for the PETweb engine tests.

Evaluates a upet model through metatrain -> metatomic, exactly like pet-kokkos'
tools/make_golden.py, on a few deterministic structures:

    uv run scripts/make_reference.py --model pet-mad-xs --out tests/reference
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np


def structures():
    import ase
    from ase.build import molecule

    a = 3.567
    scaled = [[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5],
              [.25, .25, .25], [.75, .75, .25], [.75, .25, .75], [.25, .75, .75]]
    crystal = ase.Atoms("C8", scaled_positions=scaled, cell=[a, a, a], pbc=True)
    rattled = crystal.copy()
    i = np.arange(8)[:, None]
    rattled.positions += 0.25 * np.sin(np.array([1.0, 2.0, 3.0]) * i + np.array([0.0, 1.0, 2.0]))
    water = ase.Atoms("OHH", positions=[[0, 0, .11926], [0, .76323, -.47704], [0, -.76323, -.47704]])
    ethanol = molecule("CH3CH2OH")
    ethanol.positions += 0.05 * np.sin(np.arange(27).reshape(9, 3))
    tilted = ase.Atoms("Si2", scaled_positions=[[0, 0, 0], [.25, .27, .24]],
                       cell=[[0, 2.7, 2.75], [2.72, 0, 2.7], [2.7, 2.71, 0]], pbc=True)
    return dict(water=water, ethanol=ethanol, diamond=crystal, rattled=rattled, silicon=tilted)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--version", default="latest")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default="tests/reference")
    args = ap.parse_args()

    import metatomic.torch  # noqa: F401
    from metatomic.torch.ase_calculator import MetatomicCalculator
    from metatrain.utils.io import load_model
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from convert_pet import download_upet_checkpoint

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model = load_model(download_upet_checkpoint(args.model, args.version)).export()
        calc = MetatomicCalculator(model, device="cpu", non_conservative=False)

    label = args.label or args.model
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, atoms in structures().items():
        r = calc.compute_energy(atoms, compute_forces_and_stresses=True, per_atom=True)
        ref = dict(
            model=label, case=name,
            atomic_numbers=atoms.numbers.tolist(),
            positions=atoms.positions.tolist(),
            cell=np.asarray(atoms.cell).tolist(),
            pbc=atoms.pbc.tolist(),
            energy=float(np.asarray(r["energy"]).reshape(-1)[0]),
            energies=np.asarray(r["energies"], float).tolist(),
            forces=np.asarray(r["forces"], float).tolist(),
            stress=np.asarray(r["stress"], float).tolist() if atoms.pbc.any() else None,
        )
        (out / f"{label}_{name}.json").write_text(json.dumps(ref, indent=1))
        print(f"{label:>16} {name:>8}: E = {ref['energy']:.6f} eV")


if __name__ == "__main__":
    main()
