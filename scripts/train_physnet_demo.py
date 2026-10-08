"""Train a small invariant PhysNet (mmml physnetjax, max_degree = 0) on energies and forces of
the acetone-dimer MP2 set in mmml's examples, for the MLIP-Visualization demo, and save it in
physnetjax's portable JSON ({"params", "config"}) for scripts/export_physnet.py.

    ~/mmml-asv/.venv/bin/python scripts/train_physnet_demo.py --data ~/mmml-asv/examples/fixed-acetone-only_MP2_21000.npz --out physnet-acetone.params.json
"""
import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from mmml.models.physnetjax.physnetjax.models.model import PhysNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-valid", type=int, default=400)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--forces-weight", type=float, default=20.0)
    args = ap.parse_args()

    d = np.load(args.data)
    Z, R, E, F = d["Z"], d["R"], d["E"].reshape(-1), d["F"]
    n = Z.shape[1]
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(E))
    tr, va = idx[: args.n_train], idx[args.n_train: args.n_train + args.n_valid]
    # per-element reference energies by least squares, so the network learns the rest
    elements = sorted(set(Z.reshape(-1).tolist()) - {0})
    counts = np.stack([(Z[tr] == z).sum(1) for z in elements], 1)
    ref, *_ = np.linalg.lstsq(counts, E[tr], rcond=None)
    config = dict(features=32, max_degree=0, num_iterations=3, num_basis_functions=24, cutoff=5.0,
                  max_atomic_number=max(elements), charges=True, n_refinement_blocks=2, zbl=True,
                  use_energy_bias=True, include_electrostatics=True)
    model = PhysNet(**config)
    i, j = np.where(~np.eye(n, dtype=bool))
    dst, src = jnp.asarray(i, jnp.int32), jnp.asarray(j, jnp.int32)
    params = model.init(jax.random.PRNGKey(0), jnp.asarray(Z[0]), jnp.asarray(R[0], jnp.float32), dst, src)
    # start the per-element energy bias at the least-squares references
    bias = np.zeros(config["max_atomic_number"] + 1, np.float32)
    for z, r in zip(elements, ref):
        bias[z] = r
    params = jax.tree_util.tree_map(lambda x: x, params)
    params["params"]["energy_bias"] = jnp.asarray(bias)

    def predict(p, z, r):
        out = model.apply(p, z, r, dst, src)
        return out["energy"].reshape(()), out["forces"]

    def loss(p, z, r, e, f):
        pe, pf = jax.vmap(lambda zz, rr: predict(p, zz, rr))(z, r)
        return jnp.mean((pe - e) ** 2) + args.forces_weight * jnp.mean((pf - f) ** 2), (pe, pf)

    sched = optax.cosine_decay_schedule(1e-3, args.steps, alpha=0.02)
    opt = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(sched))
    state = opt.init(params)

    @jax.jit
    def step(p, s, z, r, e, f):
        (l, _), g = jax.value_and_grad(loss, has_aux=True)(p, z, r, e, f)
        u, s = opt.update(g, s, p)
        return optax.apply_updates(p, u), s, l

    @jax.jit
    def evaluate(p, z, r, e, f):
        _, (pe, pf) = loss(p, z, r, e, f)
        return jnp.sqrt(jnp.mean((pe - e) ** 2)), jnp.sqrt(jnp.mean((pf - f) ** 2))

    t0 = time.time()
    for k in range(args.steps):
        b = rng.choice(tr, args.batch, replace=False)
        params, state, l = step(params, state, jnp.asarray(Z[b]), jnp.asarray(R[b], jnp.float32), jnp.asarray(E[b], jnp.float32), jnp.asarray(F[b], jnp.float32))
        if k % 500 == 0 or k == args.steps - 1:
            v = va[:128]
            ve, vf = evaluate(params, jnp.asarray(Z[v]), jnp.asarray(R[v], jnp.float32), jnp.asarray(E[v], jnp.float32), jnp.asarray(F[v], jnp.float32))
            print(f"step {k:5d}  loss {float(l):.4f}  valid RMSE E {float(ve) * 1000:.1f} meV  F {float(vf) * 1000:.1f} meV/A  {time.time() - t0:.0f} s", flush=True)
    tolist = lambda t: {k: tolist(v) for k, v in t.items()} if isinstance(t, dict) else np.asarray(t).tolist()
    with open(args.out, "w") as fh:
        json.dump({"params": tolist(params["params"]), "config": config, "elements": elements,
                   "data": str(args.data)}, fh)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
