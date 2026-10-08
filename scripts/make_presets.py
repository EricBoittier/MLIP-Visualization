# /// script
# dependencies = ["ase", "numpy"]
# ///
"""Write the built-in structures to src/app/presets.ts: uv run scripts/make_presets.py"""
import json

import numpy as np
from ase import Atoms
from ase.build import bulk, molecule

glycine = Atoms("NCCOOHHHHH", positions=[
    [-1.336, -0.454, 0.0], [-0.065, 0.240, 0.0], [1.114, -0.712, 0.0], [0.979, -1.917, 0.0], [2.311, -0.087, 0.0],
    [-1.383, -1.069, 0.808], [-0.067, 0.894, 0.877], [-0.067, 0.894, -0.877], [3.016, -0.756, 0.0],
    [-1.383, -1.069, -0.808]])
structures = {
    "water": ("Water", molecule("H2O")),
    "ethanol": ("Ethanol", molecule("CH3CH2OH")),
    "benzene": ("Benzene", molecule("C6H6")),
    "glycine": ("Glycine", glycine),
    "diamond": ("Diamond (8-atom cell)", bulk("C", "diamond", a=3.567, cubic=True)),
    "silicon": ("Silicon (primitive cell)", bulk("Si", "diamond", a=5.43)),
    "nacl": ("Rock salt NaCl", bulk("NaCl", "rocksalt", a=5.64, cubic=True)),
}
out = {}
for name, (label, a) in structures.items():
    out[name] = dict(label=label, numbers=a.numbers.tolist(), positions=np.round(a.positions, 5).tolist())
    if a.pbc.any():
        out[name].update(cell=np.round(np.asarray(a.cell), 5).tolist(), pbc=a.pbc.tolist())
with open("src/app/presets.ts", "w") as f:
    f.write("// Built-in structures, written by scripts/make_presets.py (ASE).\n")
    f.write("import type { System } from '../pet/structure';\n\n")
    f.write(f"export const PRESETS: Record<string, System & {{ label: string }}> = {json.dumps(out)};\n")
