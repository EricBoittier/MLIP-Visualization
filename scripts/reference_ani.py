# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "torchani", "ase", "numpy"]
# ///
"""Reference energies and forces from TorchANI's ANI-2x (through its ASE calculator, in eV).

    uv run scripts/reference_ani.py --out tests/reference
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torchani
from ase import Atoms
from ase.build import molecule


def structures():
    ethanol = molecule("CH3CH2OH")
    ethanol.positions += 0.05 * np.sin(np.arange(27).reshape(9, 3))
    water = Atoms("OHH", positions=[[0, 0, .11926], [0, .76323, -.47704], [0, -.76323, -.47704]])
    mol = molecule("CH3SH") + molecule("CH3Cl")
    mol.positions[5:] += [3.0, 0.4, 0.2]
    hf = Atoms("FH", positions=[[0, 0, 0], [0.95, 0.1, 0]])
    box = molecule("H2O")
    box.cell = [3.2, 3.0, 3.1]
    box.pbc = True
    box.positions += [1.0, 1.1, 0.9]
    return dict(water=water, ethanol=ethanol, sulfur_chlorine=mol, hf=hf, water_pbc=box)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/reference")
    args = ap.parse_args()
    calc = torchani.models.ANI2x().ase()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, atoms in structures().items():
        atoms.calc = calc
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        ref = dict(model="ani-2x", case=name, atomic_numbers=atoms.numbers.tolist(),
                   positions=atoms.positions.tolist(), cell=np.asarray(atoms.cell).tolist(),
                   pbc=atoms.pbc.tolist(), energy=float(e), forces=f.tolist())
        (out / f"ani-2x_{name}.json").write_text(json.dumps(ref, indent=1))
        print(f"ani-2x {name:>16}: E = {e:.6f} eV, max|F| = {np.abs(f).max():.4f}")


if __name__ == "__main__":
    main()
