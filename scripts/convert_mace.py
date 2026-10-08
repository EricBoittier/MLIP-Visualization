# /// script
# requires-python = ">=3.10"
# dependencies = ["mace-torch", "safetensors", "numpy", "huggingface_hub"]
# ///
"""Convert a MACE model (mace-torch ScaleShiftMACE, e.g. MACE-MP-0, MIT licence) for MLIP-Visualization.

    uv run scripts/convert_mace.py --model mace-mp-0b3-medium --out public/models/mace-mp-0b3-medium

--model is a .model file or a file name in the Hugging Face repo mace-foundations/mace-mp-0.

e3nn's normalisation conventions are not re-derived: every linear map and tensor-product path
is measured by feeding unit inputs through the module itself. The converter then checks
  1. each measured block against its module on random inputs, and
  2. the whole export, evaluated in numpy exactly the way the app evaluates it, against mace-torch.

Writes <out>.json (hyperparameters, element table, E0s, ...) and <out>.safetensors (float32):
learned weights under their module names, plus fixed tables under cg.* (tensor-product
coefficients) and U.* (symmetrised generalised Clebsch-Gordan tables of the product basis).
"""
import argparse
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

INTERACTIONS = {  # class: (normalisation of the summed messages, residual skip)
    "RealAgnosticInteractionBlock": ("avg", False),
    "RealAgnosticResidualInteractionBlock": ("avg", True),
    "RealAgnosticDensityInteractionBlock": ("density", False),
    "RealAgnosticDensityResidualInteractionBlock": ("density", True),
}
ZBL_EXPONENTS = [3.2, 0.9423, 0.4028, 0.2016]  # mace.modules.radial.ZBLBasis, with 0.529 Å and 14.3996 eV Å
TOL = 1e-10


def load(model):
    path = Path(model)
    if not path.exists():
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download("mace-foundations/mace-mp-0", model if model.endswith(".model") else f"{model}.model"))
    return torch.load(path, map_location="cpu", weights_only=False).double().eval()


def blocks(irreps):
    """[(mul, l, parity, offset)] of an e3nn Irreps, whose data is laid out [mul, 2l+1] per block."""
    out, off = [], 0
    for mul, ir in irreps:
        out.append((mul, ir.l, ir.p, off))
        off += mul * ir.dim
    return out


def check(name, got, want):
    err = np.abs(got - want).max()
    assert err <= TOL * max(1.0, np.abs(want).max()), f"{name}: reconstruction off by {err:.3g}"


@torch.no_grad()
def linear_maps(f, irreps_in, irreps_out, name):
    """The [mul_in, mul_out] matrix from every input block to every output block of the same irrep."""
    bi, bo = blocks(irreps_in), blocks(irreps_out)
    din = irreps_in.dim
    assert f(torch.zeros(1, din)).abs().max() == 0, f"{name} has a bias"
    maps = {}
    for i, (mul, l, p, off) in enumerate(bi):
        X = torch.zeros(mul, din)
        X[torch.arange(mul), off + torch.arange(mul) * (2 * l + 1)] = 1  # channel u, component m = 0
        Y = f(X)
        for j, (mo, lo, po, oo) in enumerate(bo):
            if (lo, po) == (l, p):
                maps[i, j] = Y[:, oo + torch.arange(mo) * (2 * lo + 1)].numpy().copy()
    X = torch.randn(6, din)
    Y = f(X).numpy()
    R = np.zeros_like(Y)
    for (i, j), M in maps.items():
        mul, l, _, off = bi[i]
        mo, _, _, oo = bo[j]
        d = 2 * l + 1
        R[:, oo:oo + mo * d] += np.einsum("bum,uw->bwm", X[:, off:off + mul * d].numpy().reshape(6, mul, d), M).reshape(6, mo * d)
    check(name, R, Y)
    return maps


@torch.no_grad()
def tp_paths(tp, name):
    """C[m1, m2, m3] of each 'uvu' path of a weighted tensor product (path weight and normalisation included)."""
    b1, b2, bo = blocks(tp.irreps_in1), blocks(tp.irreps_in2), blocks(tp.irreps_out)
    paths, w_off = [], 0
    for ins in tp.instructions:
        assert ins.connection_mode == "uvu" and ins.has_weight, f"{name}: unsupported path {ins}"
        mul1, l1, _, o1 = b1[ins.i_in1]
        mul2, l2, _, o2 = b2[ins.i_in2]
        mo, l3, _, oo = bo[ins.i_out]
        assert mul2 == 1 and mo == mul1
        d1, d2, d3 = 2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1
        n = d1 * d2
        X1, X2, W = torch.zeros(n, tp.irreps_in1.dim), torch.zeros(n, tp.irreps_in2.dim), torch.zeros(n, tp.weight_numel)
        for a, (m1, m2) in enumerate(itertools.product(range(d1), range(d2))):
            X1[a, o1 + m1] = 1
            X2[a, o2 + m2] = 1
        W[:, w_off] = 1
        Y = tp(X1, X2, W)
        C = Y[:, oo:oo + d3].numpy().reshape(d1, d2, d3).copy()
        Y[:, oo:oo + d3] = 0
        assert Y.abs().max() < 1e-14, f"{name}: a path lights more than its output"
        paths.append(dict(i1=ins.i_in1, i2=ins.i_in2, mid=ins.i_out, l1=l1, l2=l2, l3=l3, w=w_off, C=C))
        w_off += int(np.prod(ins.path_shape))
    assert w_off == tp.weight_numel
    B = 5
    X1, X2, W = torch.randn(B, tp.irreps_in1.dim), torch.randn(B, tp.irreps_in2.dim), torch.randn(B, tp.weight_numel)
    Y = tp(X1, X2, W).numpy()
    R = np.zeros_like(Y)
    for p in paths:
        mul1, l1, _, o1 = b1[p["i1"]]
        _, l2, _, o2 = b2[p["i2"]]
        mo, l3, _, oo = bo[p["mid"]]
        d1, d2, d3 = 2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1
        x1 = X1[:, o1:o1 + mul1 * d1].numpy().reshape(B, mul1, d1)
        R[:, oo:oo + mo * d3] += np.einsum("bu,bui,bj,ijk->buk", W[:, p["w"]:p["w"] + mul1].numpy(), x1,
                                           X2[:, o2:o2 + d2].numpy(), p["C"]).reshape(B, mo * d3)
    check(name, R, Y)
    return paths


@torch.no_grad()
def mlp(net, name, act="silu"):
    """An e3nn FullyConnectedNet as ([out, in] matrices with e3nn's 1/sqrt(fan_in) folded in, activation scale)."""
    layers, scale = [], None
    for k in range(len(net.hs) - 1):
        L = getattr(net, f"layer{k}")
        if L.act is not None:
            W = L.weight / (L.h_in * L.var_in) ** 0.5
            assert L.act.f is getattr(torch.nn.functional, act), f"{name}: activation {L.act.f}"
            s = (1.0 if L.act._is_id else L.act.cst) * L.var_out ** 0.5
            assert scale in (None, s)
            scale = s
        else:
            W = L.weight / (L.h_in * L.var_in / L.var_out) ** 0.5
            assert k == len(net.hs) - 2, f"{name}: a hidden layer without activation"
        layers.append(W.T.numpy().copy())
    X = torch.randn(6, net.hs[0])
    h = X.numpy()
    for k, W in enumerate(layers):
        h = h @ W.T
        if k < len(layers) - 1:
            h = scale * h / (1 + np.exp(-h))
    check(name, h, net(X).numpy())
    return layers, scale


def contraction_tables(c, n_in):
    """Per correlation order nu: the U table summed over orderings of each sorted index tuple
    (x_j1 ... x_jnu is symmetric, so only the symmetric part of U contributes), [D*K, monomials]."""
    nu_max = c.correlation
    out = []
    for nu in range(1, nu_max + 1):
        U = getattr(c, f"U_matrix_{nu}").numpy()
        if U.ndim == nu + 1:
            U = U[None]  # a scalar output has no m axis
        D, K = U.shape[0], U.shape[-1]
        monos = list(itertools.combinations_with_replacement(range(n_in), nu))
        S = np.zeros((D, K, len(monos)))
        for a, t in enumerate(monos):
            for perm in set(itertools.permutations(t)):
                S[:, :, a] += U[(slice(None),) + perm]
        W = (c.weights_max if nu == nu_max else c.weights[nu_max - 1 - nu]).detach().numpy()  # [Z, K, C]
        out.append(dict(nu=nu, D=D, K=K, S=S.reshape(D * K, len(monos)), W=W, monos=monos))
    return out


def monomials(x, nu):
    """x [..., n] -> products over the sorted index tuples of length nu."""
    return np.stack([np.prod([x[..., j] for j in t], axis=0) for t in itertools.combinations_with_replacement(range(x.shape[-1]), nu)], -1)


@torch.no_grad()
def check_contraction(c, tabs, num_elements, name):
    B, C, n_in = 6, tabs[0]["W"].shape[2], tabs[0]["S"].shape[1]
    x = torch.randn(B, C, n_in)
    el = torch.randint(0, num_elements, (B,))
    y = torch.nn.functional.one_hot(el, num_elements).double()
    want = c(x, y).numpy().reshape(B, C, -1)
    got = 0
    for t in tabs:
        Bk = (monomials(x.numpy(), t["nu"]) @ t["S"].T).reshape(B, C, t["D"], t["K"])
        got = got + np.einsum("bcdk,bkc->bcd", Bk, t["W"][el.numpy()])
    check(name, got, want)


def convert(m):
    from e3nn import o3

    assert type(m).__name__ == "ScaleShiftMACE", f"{type(m).__name__} is not supported"
    assert len(getattr(m, "heads", ["default"])) == 1, "multi-head models are not supported"
    Zs = m.atomic_numbers.tolist()
    nZ = len(Zs)
    re = m.radial_embedding
    assert type(re.bessel_fn).__name__ == "BesselBasis", f"radial basis {type(re.bessel_fn).__name__}"
    assert getattr(re, "apply_cutoff", True), "apply_cutoff = False is not supported"
    sh = m.spherical_harmonics
    assert sh.normalize and sh.normalization == "component"
    lmax = sh._lmax
    assert sh._ls_list == list(range(lmax + 1)) and lmax <= 3, "spherical harmonics up to l = 3"
    T, meta = {}, {}
    meta.update(kind="mace", architecture="mace", atomic_numbers=Zs, r_max=float(m.r_max), sh_lmax=lmax,
                bessel=dict(weights=re.bessel_fn.bessel_weights.tolist(), prefactor=float(re.bessel_fn.prefactor)),
                cutoff_p=int(re.cutoff_fn.p), e0=m.atomic_energies_fn.atomic_energies.reshape(-1).tolist(),
                scale=float(m.scale_shift.scale), shift=float(m.scale_shift.shift))
    assert m.scale_shift.scale.numel() == 1 and len(meta["e0"]) == nZ
    radii = None
    if hasattr(re, "distance_transform"):
        dt = re.distance_transform
        assert type(dt).__name__ == "AgnesiTransform", f"distance transform {type(dt).__name__}"
        meta["agnesi"] = dict(a=float(dt.a), q=float(dt.q), p=float(dt.p))
        radii = dt.covalent_radii.tolist()
    if hasattr(m, "pair_repulsion"):
        z = m.pair_repulsion_fn
        meta["zbl"] = dict(c=z.c.tolist(), exponents=ZBL_EXPONENTS, p=int(z.p), a_exp=float(z.a_exp),
                           a_prefactor=float(z.a_prefactor), bohr=0.529, coulomb=14.3996)
        radii = radii or z.covalent_radii.tolist()
        assert radii == z.covalent_radii.tolist()
    meta["covalent_radii"] = radii

    attrs = o3.Irreps(f"{nZ}x0e")
    emb = linear_maps(m.node_embedding.linear, attrs, m.node_embedding.linear.irreps_out, "node_embedding")
    T["node_embedding"] = emb[0, 0]  # [Z, C]: one-hot elements times the embedding
    C = T["node_embedding"].shape[1]
    meta["channels"] = C
    sh_irreps = o3.Irreps.spherical_harmonics(lmax)

    def ls_of(irreps, what):
        bl = blocks(o3.Irreps(irreps))
        assert all(mul == C and p == (-1) ** l for mul, l, p, _ in bl), f"{what}: {irreps}"
        assert [l for _, l, _, _ in bl] == sorted({l for _, l, _, _ in bl}), f"{what}: {irreps}"
        return [l for _, l, _, _ in bl]

    meta["interactions"], meta["products"], meta["readouts"] = [], [], []
    for i, (b, prod) in enumerate(zip(m.interactions, m.products)):
        kind, residual = INTERACTIONS.get(type(b).__name__, (None, None))
        assert kind, f"interaction {type(b).__name__} is not supported"
        assert b.conv_tp.irreps_in2 == sh_irreps
        ls_in = ls_of(b.node_feats_irreps, "node features")
        ls_target = ls_of(b.irreps_out, "interaction output")
        up = linear_maps(b.linear_up, b.linear_up.irreps_in, b.linear_up.irreps_out, f"interactions.{i}.linear_up")
        assert sorted(up) == [(k, k) for k in range(len(ls_in))]
        for k, l in enumerate(ls_in):
            T[f"interactions.{i}.linear_up.{l}"] = up[k, k].T
        paths = tp_paths(b.conv_tp, f"interactions.{i}.conv_tp")
        mids = blocks(b.conv_tp.irreps_out)
        assert sorted(p["mid"] for p in paths) == list(range(len(mids)))
        for k, p in enumerate(paths):
            T[f"cg.{i}.{k}"] = p["C"].transpose(0, 2, 1).reshape(-1, 2 * p["l2"] + 1)  # [(m1, m3), m2]
        radial, act = mlp(b.conv_tp_weights, f"interactions.{i}.conv_tp_weights")
        for k, W in enumerate(radial):
            T[f"interactions.{i}.radial.{k}"] = W
        lin = linear_maps(b.linear, b.linear.irreps_in, b.linear.irreps_out, f"interactions.{i}.linear")
        for (k, j), M in lin.items():
            assert ls_target[j] == mids[k][1]
        mid_to_out = {k: j for k, j in lin}
        assert len(mid_to_out) == len(lin) == len(mids), "each tensor-product output feeds one output block"
        for k, j in lin:
            T[f"interactions.{i}.linear.{k}"] = lin[k, j].T
        entry = dict(kind=kind, residual=residual, ls_in=ls_in, ls_target=ls_target, radial_act=act,
                     radial=[list(W.shape) for W in radial],
                     paths=[dict(i1=p["i1"], l1=p["l1"], l2=p["l2"], l3=p["l3"], w=p["w"], out=mid_to_out[p["mid"]]) for p in paths])
        # the paths are listed by output block, in order; linear.{k} belongs to path k
        assert all(p["mid"] == k for k, p in enumerate(paths))
        if kind == "density":
            (W,), _ = mlp(b.density_fn, f"interactions.{i}.density_fn")
            T[f"interactions.{i}.density"] = W
        else:
            entry["avg_num_neighbors"] = float(b.avg_num_neighbors)
        skip_in = b.skip_tp.irreps_in1
        assert b.skip_tp.irreps_in2 == attrs
        skip_ls_out = ls_of(b.skip_tp.irreps_out, "skip output")
        skip_ls_in = [l for _, l, _, _ in blocks(skip_in)]
        per_el = []
        for e in range(nZ):
            onehot = torch.zeros(1, nZ)
            onehot[0, e] = 1
            per_el.append(linear_maps(lambda X, oh=onehot: b.skip_tp(X, oh.expand(X.shape[0], -1)), skip_in, b.skip_tp.irreps_out,
                                      f"interactions.{i}.skip_tp[{Zs[e]}]"))
        for (k, j) in per_el[0]:
            assert skip_ls_in[k] == skip_ls_out[j]
            T[f"interactions.{i}.skip.{skip_ls_out[j]}"] = np.stack([pe[k, j].T for pe in per_el])  # [Z, out, in]
        entry["skip_ls"] = skip_ls_out
        meta["interactions"].append(entry)

        # ---- product basis
        sc = prod.symmetric_contractions
        ls_out = ls_of(sc.irreps_out, "product output")
        n_in = sum(2 * l + 1 for l in ls_target)
        pe = dict(ls_out=ls_out, use_sc=bool(prod.use_sc), contractions=[])
        for c, L in zip(sc.contractions, ls_out):
            tabs = contraction_tables(c, n_in)
            check_contraction(c, tabs, nZ, f"products.{i}.contraction[{L}]")
            for t in tabs:
                T[f"U.{i}.{L}.{t['nu']}"] = t["S"]
                T[f"products.{i}.{L}.weights.{t['nu']}"] = t["W"].transpose(0, 2, 1).reshape(nZ * C, t["K"])  # rows (element, channel)
            pe["contractions"].append(dict(L=L, nu=[t["nu"] for t in tabs], K=[t["K"] for t in tabs]))
        pl = linear_maps(prod.linear, prod.linear.irreps_in, prod.linear.irreps_out, f"products.{i}.linear")
        for (k, j), M in pl.items():
            assert k == j
            T[f"products.{i}.linear.{ls_out[j]}"] = M.T
        meta["products"].append(pe)

    # ---- readouts (one per interaction)
    assert len(m.readouts) == len(m.interactions)
    for i, (ro, prod) in enumerate(zip(m.readouts, m.products)):
        irr = prod.linear.irreps_out
        if type(ro).__name__ == "LinearReadoutBlock":
            mp = linear_maps(ro.linear, irr, ro.linear.irreps_out, f"readouts.{i}")
            T[f"readouts.{i}.linear"] = mp[0, 0].T
            meta["readouts"].append(dict(kind="linear"))
        else:
            assert type(ro).__name__ == "NonLinearReadoutBlock", type(ro).__name__
            m1 = linear_maps(ro.linear_1, irr, ro.linear_1.irreps_out, f"readouts.{i}.linear_1")
            m2 = linear_maps(ro.linear_2, ro.linear_2.irreps_in, ro.linear_2.irreps_out, f"readouts.{i}.linear_2")
            (a,) = ro.non_linearity.acts
            assert a.f is torch.nn.functional.silu
            T[f"readouts.{i}.linear_1"], T[f"readouts.{i}.linear_2"] = m1[0, 0].T, m2[0, 0].T
            meta["readouts"].append(dict(kind="nonlinear", hidden=int(m1[0, 0].shape[1]), act=1.0 if a._is_id else a.cst))
    meta["monomials"] = {nu: [list(t) for t in itertools.combinations_with_replacement(range((lmax + 1) ** 2), nu)]
                         for nu in range(2, 1 + max(max(c["nu"]) for p in meta["products"] for c in p["contractions"]))}
    return meta, T


# ------------------------------------------------------------------ the export, evaluated as the app does

def envelope(x, r_max, p):
    t = x / r_max
    return (1 - (p + 1) * (p + 2) / 2 * t ** p + p * (p + 2) * t ** (p + 1) - p * (p + 1) / 2 * t ** (p + 2)) * (x < r_max)


def forward_np(meta, T, Z, sender, receiver, vectors, sh_fn):
    zi = np.array([meta["atomic_numbers"].index(z) for z in Z])
    N, C = len(Z), meta["channels"]
    r = np.linalg.norm(vectors, axis=1)
    Y = sh_fn(vectors / r[:, None])
    cut = envelope(r, meta["r_max"], meta["cutoff_p"])
    x = r
    if "agnesi" in meta:
        a, q, p = (meta["agnesi"][k] for k in "aqp")
        R = np.array(meta["covalent_radii"])
        s = r / (0.5 * (R[np.array(Z)[sender]] + R[np.array(Z)[receiver]]))
        x = 1 / (1 + a * s ** q / (1 + s ** (q - p)))
    w = np.array(meta["bessel"]["weights"])
    ef = meta["bessel"]["prefactor"] * np.sin(w * x[:, None]) / x[:, None] * cut[:, None]
    silu = lambda v: v / (1 + np.exp(-v))
    feats = {0: T["node_embedding"][zi][:, None, :]}  # l -> [N, 2l+1, C]
    node_es = np.zeros(N)
    if "zbl" in meta:
        zb = meta["zbl"]
        Zu, Zv = np.array(Z)[sender].astype(float), np.array(Z)[receiver].astype(float)
        aa = zb["a_prefactor"] * zb["bohr"] / (Zu ** zb["a_exp"] + Zv ** zb["a_exp"])
        phi = sum(c * np.exp(-k * r / aa) for c, k in zip(zb["c"], zb["exponents"]))
        R = np.array(meta["covalent_radii"])
        v = 0.5 * zb["coulomb"] * Zu * Zv / r * phi * envelope(r, R[np.array(Z)[sender]] + R[np.array(Z)[receiver]], zb["p"])
        np.add.at(node_es, receiver, v)
    lofs = np.cumsum([0] + [2 * l + 1 for l in range(meta["sh_lmax"] + 1)])
    for i, (it, pr) in enumerate(zip(meta["interactions"], meta["products"])):
        sc = None
        el_lin = lambda W, X: np.einsum("nmi,noi->nmo", X, W[zi])  # per-element [out, in]
        if it["residual"]:
            sc = {l: el_lin(T[f"interactions.{i}.skip.{l}"], feats[l]) for l in it["skip_ls"]}
        up = {l: feats[l] @ T[f"interactions.{i}.linear_up.{l}"].T for l in it["ls_in"]}
        h = ef
        for k in range(len(it["radial"])):
            h = h @ T[f"interactions.{i}.radial.{k}"].T
            if k < len(it["radial"]) - 1:
                h = it["radial_act"] * silu(h)
        msg = {l: np.zeros((N, 2 * l + 1, C)) for l in it["ls_target"]}
        for k, p in enumerate(it["paths"]):
            l1, l2, l3 = p["l1"], p["l2"], p["l3"]
            Tk = Y[:, lofs[l2]:lofs[l2 + 1]] @ T[f"cg.{i}.{k}"].T  # [E, (m1, m3)]
            Tk = Tk.reshape(-1, 2 * l1 + 1, 2 * l3 + 1)
            mji = np.einsum("eik,eic->ekc", Tk, up[l1][sender]) * h[:, None, p["w"]:p["w"] + C]
            mid = np.zeros((N, 2 * l3 + 1, C))
            np.add.at(mid, receiver, mji)
            msg[it["ls_target"][p["out"]]] += mid @ T[f"interactions.{i}.linear.{k}"].T
        if it["kind"] == "density":
            dens = np.zeros(N)
            np.add.at(dens, receiver, np.tanh((ef @ T[f"interactions.{i}.density"].T)[:, 0] ** 2))
            msg = {l: v / (dens[:, None, None] + 1) for l, v in msg.items()}
        else:
            msg = {l: v / it["avg_num_neighbors"] for l, v in msg.items()}
        if not it["residual"]:
            msg = {l: el_lin(T[f"interactions.{i}.skip.{l}"], msg[l]) for l in it["skip_ls"]}
        x = np.concatenate([msg[l].transpose(0, 2, 1) for l in it["ls_target"]], -1)  # [N, C, n_in]
        new = {}
        for c in pr["contractions"]:
            L = c["L"]
            out = 0
            for nu in c["nu"]:
                U = T[f"U.{i}.{L}.{nu}"]
                K = U.shape[0] // (2 * L + 1)
                Bk = (monomials(x, nu) @ U.T).reshape(N, C, 2 * L + 1, K)
                Wt = T[f"products.{i}.{L}.weights.{nu}"].reshape(-1, C, K)[zi]  # [N, C, K]
                out = out + np.einsum("ncdk,nck->ncd", Bk, Wt)
            new[L] = out.transpose(0, 2, 1) @ T[f"products.{i}.linear.{L}"].T
            if pr["use_sc"] and sc is not None:
                new[L] = new[L] + sc[L]
        feats = new
        ro = meta["readouts"][i]
        if ro["kind"] == "linear":
            node_es += (feats[0][:, 0] @ T[f"readouts.{i}.linear"].T)[:, 0]
        else:
            hh = ro["act"] * silu(feats[0][:, 0] @ T[f"readouts.{i}.linear_1"].T)
            node_es += (hh @ T[f"readouts.{i}.linear_2"].T)[:, 0]
    node_es = meta["scale"] * node_es + meta["shift"]
    return np.array(meta["e0"])[zi].sum() + node_es.sum()


def check_export(m, meta, T):
    from ase.build import bulk, molecule
    from e3nn import o3
    from mace import data
    from mace.tools import torch_geometric, utils

    z_table = utils.AtomicNumberTable(meta["atomic_numbers"])
    sh_fn = lambda u: o3.spherical_harmonics(list(range(meta["sh_lmax"] + 1)), torch.tensor(u), True, "component").numpy()
    rng = np.random.default_rng(0)
    for atoms in [molecule("CH3CH2OH"), bulk("NaCl", "rocksalt", a=5.64), molecule("H2O")]:
        atoms.positions += rng.normal(0, 0.05, atoms.positions.shape)
        cfg = data.config_from_atoms(atoms)
        ad = data.AtomicData.from_config(cfg, z_table=z_table, cutoff=meta["r_max"], heads=["default"])
        batch = next(iter(torch_geometric.dataloader.DataLoader([ad], batch_size=1)))
        want = float(m(batch.to_dict(), compute_force=False)["energy"])
        s, r = batch.edge_index.numpy()
        pos = batch.positions.detach().numpy()
        vec = pos[r] - pos[s] + batch.shifts.detach().numpy()
        got = forward_np(meta, T, atoms.numbers.tolist(), s, r, vec, sh_fn)
        print(f"  {atoms.get_chemical_formula():10s} mace {want:.10f}  export {got:.10f}  diff {got - want:.2e}")
        assert abs(got - want) < 1e-8 * max(1, abs(want)), "the export does not reproduce mace"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mace-mp-0b3-medium")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    m = load(args.model)
    meta, T = convert(m)
    print("checking the export against mace-torch:")
    check_export(m, meta, T)
    meta["name"] = args.label or Path(args.model).stem
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(meta))
    save_file({k: np.ascontiguousarray(v, dtype=np.float32) for k, v in T.items()}, str(out.with_suffix(".safetensors")))
    n = sum(v.size for k, v in T.items() if not k.startswith(("cg.", "U.")))
    print(f"wrote {out}.json and {out}.safetensors: {n / 1e6:.2f} M weights, {len(T)} tensors")


if __name__ == "__main__":
    main()
