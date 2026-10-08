"""Reference outputs of mmml's physnetjax PhysNet (max_degree = 0, i.e. invariant) for the
MLIP-Visualization port, from a randomly initialised model (the weights do not matter for
checking the implementation).

Run with the Python environment that has mmml installed:

    ~/mmml-asv/.venv/bin/python scripts/reference_physnet.py --out tests/reference

Writes <out>/physnet-test.params.json (physnetjax's portable JSON: {"params", "config"})
and <out>/physnet-test_<case>.json (energy, forces, charges).
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from ase.build import molecule

from mmml.models.physnetjax.physnetjax.models.model import PhysNet


def all_pairs(n):
    i, j = np.where(~np.eye(n, dtype=bool))
    return jnp.asarray(i, dtype=jnp.int32), jnp.asarray(j, dtype=jnp.int32)


def to_lists(tree):
    if isinstance(tree, dict):
        return {k: to_lists(v) for k, v in tree.items()}
    return np.asarray(tree).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/reference")
    args = ap.parse_args()
    config = dict(features=16, max_degree=0, num_iterations=3, num_basis_functions=12, cutoff=4.0,
                  max_atomic_number=17, charges=True, n_refinement_blocks=2, zbl=True,
                  use_energy_bias=True, include_electrostatics=True)
    model = PhysNet(**config)
    ethanol = molecule("CH3CH2OH")
    ethanol.positions += 0.05 * np.sin(np.arange(27).reshape(9, 3))
    water = molecule("H2O")
    squeezed = molecule("CH4")
    squeezed.positions[1] *= 0.45  # a C-H pair inside the ZBL range (< 0.6 A)
    dimer = molecule("H2O") + molecule("H2O")
    dimer.positions[3:] += [2.9, 0.1, 0.0]
    cases = dict(ethanol=ethanol, water=water, squeezed=squeezed, dimer=dimer)

    Z0 = jnp.asarray(ethanol.numbers, dtype=jnp.int32)
    dst, src = all_pairs(len(ethanol))
    params = model.init(jax.random.PRNGKey(7), Z0, jnp.asarray(ethanol.positions, jnp.float32), dst, src)
    # give the zero-initialised heads and biases some weight, so the test sees them
    rng = np.random.default_rng(3)
    p = jax.tree_util.tree_map(lambda x: np.asarray(x), params["params"])
    for name in list(p):
        if name in ("energy_bias", "charge_bias"):
            p[name] = rng.normal(0, 0.3, p[name].shape).astype(np.float32)
    last = sorted((k for k in p if k.startswith("Dense_")), key=lambda k: int(k.split("_")[1]))
    for k in last[-4:]:
        for leaf in p[k]:
            arr = p[k][leaf] if not isinstance(p[k][leaf], dict) else None
            if arr is not None:
                p[k][leaf] = rng.normal(0, 0.5, arr.shape).astype(np.float32)
            else:
                for kk in p[k][leaf]:
                    p[k][leaf][kk] = rng.normal(0, 0.5, np.shape(p[k][leaf][kk])).astype(np.float32)
    params = {"params": p}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "physnet-test.params.json").write_text(json.dumps({"params": to_lists(p), "config": config}))
    for name, atoms in cases.items():
        Z = jnp.asarray(atoms.numbers, dtype=jnp.int32)
        R = jnp.asarray(atoms.positions, dtype=jnp.float32)
        dst, src = all_pairs(len(atoms))
        res = model.apply(params, Z, R, dst, src)
        ref = dict(model="physnet-test", case=name, atomic_numbers=atoms.numbers.tolist(),
                   positions=atoms.positions.tolist(),
                   energy=float(np.asarray(res["energy"]).reshape(-1)[0]),
                   forces=np.asarray(res["forces"]).reshape(-1, 3).tolist(),
                   charges=np.asarray(res["charges"]).reshape(-1).tolist(),
                   repulsion=np.asarray(res["repulsion"]).reshape(-1).tolist())
        (out / f"physnet-test_{name}.json").write_text(json.dumps(ref, indent=1))
        print(f"physnet-test {name:>9}: E = {ref['energy']:.6f}, max|F| = {np.abs(ref['forces']).max():.4f}, "
              f"sum q = {sum(ref['charges']):.4f}, repulsion = {sum(ref['repulsion']):.4f}")


if __name__ == "__main__":
    main()
