// MACE (Batatia et al., NeurIPS 2022) as mace-torch's ScaleShiftMACE computes it, e.g. MACE-MP-0,
// from files written by scripts/convert_mace.py (which measures e3nn's normalisations from the
// modules themselves and checks the export against mace-torch).
//
// Equivariant features are kept one block per l: [atoms (2l + 1), channels], rows (atom, m), so
// that every channel mixing is a plain linear layer. Per interaction:
//   up      = linear(h)                                    per l
//   message = sum_j  R(r_ij) * (up_j (x) Y(r_ij))          tensor product, one path per (l1, l2 -> l3)
//   A       = linear(message) / (density + 1)              (or / the average neighbour count)
//   B       = sum_nu  W_nu[element] . U_nu x^nu            the symmetric product basis, up to x^3
//   h'      = linear(B) (+ element-dependent skip of h)
// and every layer reads out an energy. ZBL repulsion and E0s complete the energy.
import type { Backend } from '../../engine/backend';
import { type Graph, type Index, Tensor } from '../../engine/tensor';
import { eV, times, Å } from '../../engine/units';
import type { Tensors } from '../../common/safetensors';
import { SYMBOLS } from '../../common/elements';
import { neighborList, type System } from '../../common/structure';
import { atomRows, type Forward, type Model, range, type RowSpace } from '../types';
import { e3nnBasis } from './e3nn';

interface Path { i1: number; l1: number; l2: number; l3: number; w: number; out: number }
interface InteractionMeta {
  kind: 'density' | 'avg';
  residual: boolean;
  ls_in: number[];
  ls_target: number[];
  radial_act: number;
  radial: number[][];
  paths: Path[];
  skip_ls: number[];
  avg_num_neighbors?: number;
}
interface ProductMeta { ls_out: number[]; use_sc: boolean; contractions: { L: number; nu: number[]; K: number[] }[] }
type Readout = { kind: 'linear' } | { kind: 'nonlinear'; hidden: number; act: number };

export interface MaceMeta {
  kind: 'mace';
  name?: string;
  atomic_numbers: number[];
  r_max: number;
  sh_lmax: number;
  bessel: { weights: number[]; prefactor: number };
  cutoff_p: number;
  e0: number[];
  scale: number;
  shift: number;
  agnesi?: { a: number; q: number; p: number };
  zbl?: { c: number[]; exponents: number[]; p: number; a_exp: number; a_prefactor: number; bohr: number; coulomb: number };
  covalent_radii: number[] | null;
  channels: number;
  interactions: InteractionMeta[];
  products: ProductMeta[];
  readouts: Readout[];
  monomials: Record<string, number[][]>;
}

/** Internal tensors kept for the tests, in the layout of scripts/reference_mace.py. */
export interface MaceInternals {
  sh: Tensor;
  radial: Tensor;
  pair?: Tensor;
  layers: { message: Map<number, Tensor>; product: Map<number, Tensor>; readout: Tensor }[];
}

const dim = (l: number) => 2 * l + 1;
/** mace's PolynomialCutoff, 1 - (p+1)(p+2)/2 t^p + p(p+2) t^(p+1) - p(p+1)/2 t^(p+2) with t = r / r_max, in the
 *  identical form (1 - t)^3 sum_{k<p} C(k+2, 2) t^k: expanded, it cancels to nothing in float32 near t = 1. */
function envelope(g: Graph, t: Tensor, p: number, one: Tensor) {
  const c = (k: number) => g.constant(new Float32Array([((k + 1) * (k + 2)) / 2]), [1], `C(${k + 2},2)`);
  let q: Tensor = c(0);
  if (p > 1) {
    q = g.add(g.scale(t, ((p * (p + 1)) / 2)), c(p - 2));
    for (let k = p - 3; k >= 0; k--) q = g.add(g.mul(q, t), c(k));
  }
  const s = g.add(g.unary('neg', t), one, '1-t');
  return g.mul(g.mul(g.unary('square', s), s), q, 'envelope');
}

export class MACE implements Model {
  readonly kind = 'mace' as const;
  readonly params = new Map<string, Tensor>();
  /** Fixed tables: tensor-product coefficients (cg.*), product-basis tables (U.*), selections. */
  private consts = new Map<string, Tensor>();
  get elements() { return this.meta.atomic_numbers; }

  constructor(readonly be: Backend, readonly meta: MaceMeta, tensors: Tensors) {
    const up = (name: string, shape: number[], data: Float32Array) => new Tensor(shape, be.upload(data), false, name);
    for (const [name, t] of tensors) {
      if (/^(cg|U)\./.test(name)) this.consts.set(name, up(name, t.shape, t.data));
      else if (/^interactions\.\d+\.skip\.\d+$/.test(name)) {
        // [element, out, in]: one matrix per element, named by its symbol
        const [Z, o, i] = t.shape;
        for (let e = 0; e < Z; e++) {
          const n = `${name}.${SYMBOLS[meta.atomic_numbers[e]]}`;
          this.params.set(n, up(n, [o, i], t.data.subarray(e * o * i, (e + 1) * o * i)));
        }
      } else this.params.set(name, up(name, t.shape, t.data));
    }
    this.consts.set('sh_basis', up('e3nn_basis', [(meta.sh_lmax + 1) ** 2, (meta.sh_lmax + 1) ** 2], e3nnBasis(meta.sh_lmax)));
    // one-hot selections that build the monomials x_a x_b and x_a x_b x_c of the product basis
    const n = (meta.sh_lmax + 1) ** 2, pairs = meta.monomials['2'] ?? [], triples = meta.monomials['3'] ?? [];
    const pairIdx = new Map(pairs.map(([a, b], k) => [`${a},${b}`, k]));
    const select = (name: string, rows: number[], cols: number) => {
      const s = new Float32Array(rows.length * cols);
      rows.forEach((c, r) => { s[r * cols + c] = 1; });
      this.consts.set(name, up(name, [rows.length, cols], s));
    };
    select('pick_a', pairs.map((p) => p[0]), n);
    select('pick_b', pairs.map((p) => p[1]), n);
    select('pick_ab', triples.map(([a, b]) => pairIdx.get(`${a},${b}`)!), pairs.length);
    select('pick_c', triples.map((t) => t[2]), n);
  }

  private p(name: string) {
    const t = this.params.get(name);
    if (!t) throw new Error(`MACE: missing parameter ${name}`);
    return t;
  }
  private c(name: string) {
    const t = this.consts.get(name);
    if (!t) throw new Error(`MACE: missing table ${name}`);
    return t;
  }

  forward(g: Graph, sys: System, opts: { forces?: boolean } = {}): Forward & { internals: MaceInternals } {
    const m = this.meta, N = sys.numbers.length, C = m.channels, L = m.sh_lmax;
    const zi = sys.numbers.map((z) => {
      const i = m.atomic_numbers.indexOf(z);
      if (i < 0) throw new Error(`${m.name ?? 'MACE'} has no parameters for element ${SYMBOLS[z] ?? z}`);
      return i;
    });
    const nl = neighborList(sys, m.r_max);
    const E = nl.center.length;
    const recv = nl.center, send = nl.neighbor; // messages flow from the neighbour to the centre
    const ix = (a: ArrayLike<number>, nSrc: number, out?: string, src?: string): Index => g.index(Int32Array.from(a), nSrc, { out, src });
    const k = (data: ArrayLike<number>, shape: number[], name: string, kind?: string) => g.constant(Float32Array.from(data), shape, name, kind);
    const one = k([1], [1], 'one');
    const blockRows = (n: number, d: number, f: (r: number, mm: number) => number) => Array.from({ length: n * d }, (_, r) => f((r / d) | 0, r % d));
    const kindA = (l: number) => (l ? `atom.l${l}` : 'atom'), kindE = (l: number) => (l ? `edge.l${l}` : 'edge');

    const positions = k(sys.positions.flat(), [N, 3], 'positions', 'atom').withUnit(Å);
    positions.requiresGrad = !!opts.forces;
    const { v, vec, r } = g.scope('geometry', () => {
      const v = g.add(g.sub(g.gather(positions, ix(send, N, 'edge', 'atom'), 'r_j'), g.gather(positions, ix(recv, N, 'edge', 'atom'), 'r_i')),
                      k(nl.shiftVec, [E, 3], 'shift', 'edge'));
      const vec = g.unary('neg', v); // mace's edge vector points from the neighbour to the centre
      return { v, vec, r: g.rowNorm(vec) };
    });
    const Y = g.scope('spherical_harmonics', () =>
      g.linear(g.sph(g.div(vec, r), L), this.c('sh_basis'), undefined, 'e3nn_harmonics'));

    // ---- radial features: Bessel functions of the (Agnesi-transformed) distance, times the envelope
    const R = m.covalent_radii, Zs = sys.numbers;
    const ef = g.scope('radial', () => {
      const env = envelope(g, g.scale(r, 1 / m.r_max), m.cutoff_p, one);
      let x = r;
      if (m.agnesi) {
        const { a, q, p } = m.agnesi;
        const s = g.div(r, k(Array.from(recv, (i, e) => 0.5 * (R![Zs[i]] + R![Zs[send[e]]])), [E], 'r0', 'edge').withUnit(Å), 'r_over_r0');
        x = g.unary('pow', g.add(g.div(g.scale(g.unary('pow', s, q), a), g.add(g.unary('pow', s, q - p), one)), one), -1, 0);
      }
      const nb = m.bessel.weights.length;
      const arg = g.mul(g.repeatCols(x, nb, 'x'), k(m.bessel.weights, [1, nb], 'bessel_frequencies'));
      const bessel = g.scale(g.div(g.unary('sin', arg), x), m.bessel.prefactor, 'bessel');
      return g.mul(bessel, env, 'edge_features');
    });

    // ---- ZBL repulsion between pairs closer than the sum of their covalent radii
    let pair: Tensor | undefined;
    const rows: Record<string, RowSpace> = { atom: atomRows(N), edge: { label: 'edge', owner: recv, edge: range(E) } };
    const zb = m.zbl ? range(E).filter((e) => nl.dist[e] < R![Zs[recv[e]]] + R![Zs[send[e]]]) : new Int32Array(0);
    if (m.zbl && zb.length) {
      const z = m.zbl;
      rows.zbl_pair = { label: 'repulsive pair', owner: Int32Array.from(zb, (e) => recv[e]), edge: zb };
      pair = g.scope('repulsion', () => {
        const rz = g.gather(r, ix(zb, E, 'zbl_pair', 'edge'), 'pairs');
        const zu = Array.from(zb, (e) => Zs[send[e]]), zv = Array.from(zb, (e) => Zs[recv[e]]);
        const a = zu.map((u, n) => (z.a_prefactor * z.bohr) / (u ** z.a_exp + zv[n] ** z.a_exp));
        const xr = g.div(rz, k(a, [zb.length], 'screening_length', 'zbl_pair').withUnit(Å));
        const phi = g.linear(g.unary('exp', g.mul(g.repeatCols(xr, 4, 'x'), k(z.exponents.map((q) => -q), [1, 4], 'exponents'))),
                             k(z.c, [1, 4], 'coefficients'), undefined, 'screening');
        const coulomb = g.div(k(zu.map((u, n) => 0.5 * z.coulomb * u * zv[n]), [zb.length], 'ZuZv', 'zbl_pair').withUnit(times(eV, Å)), rz);
        const rmax = k(Array.from(zb, (e) => R![Zs[recv[e]]] + R![Zs[send[e]]]), [zb.length], 'covalent_sum', 'zbl_pair').withUnit(Å);
        const e = g.mul(g.mul(coulomb, g.reshape(phi, [zb.length], 'zbl_pair')), envelope(g, g.div(rz, rmax), z.p, one), 'pair_energy');
        return g.segmentSum(e, ix(Array.from(zb, (q) => recv[q]), N, 'zbl_pair', 'atom'), 'per_atom').withUnit(eV);
      });
    }

    // ---- per-l helpers
    for (let l = 1; l <= L; l++) {
      rows[kindA(l)] = { label: `atom, m component (l = ${l})`, owner: Int32Array.from(blockRows(N, dim(l), (n) => n)), atom: Int32Array.from(blockRows(N, dim(l), (n) => n)), all: true };
      rows[kindE(l)] = { label: `edge, m component (l = ${l})`, owner: Int32Array.from(blockRows(E, dim(l), (e) => recv[e])), edge: Int32Array.from(blockRows(E, dim(l), (e) => e)) };
    }
    rows.channel = { label: 'channel of an atom', owner: Int32Array.from(blockRows(N, C, (n) => n)), atom: Int32Array.from(blockRows(N, C, (n) => n)), all: true };
    const groups = new Map<number, number[]>();
    zi.forEach((e, n) => groups.set(e, [...(groups.get(e) ?? []), n]));
    /** Element-dependent linear map of each l block (mace's skip tensor product with the one-hot elements). */
    const perElement = (name: string, x: Map<number, Tensor>, ls: number[]) => new Map(ls.map((l) => {
      let out: Tensor | null = null;
      for (const [e, atoms] of groups) {
        const sym = SYMBOLS[m.atomic_numbers[e]];
        const rws = atoms.flatMap((n) => Array.from({ length: dim(l) }, (_, mm) => n * dim(l) + mm));
        const kind = `${sym}.l${l}`;
        rows[kind] ??= { label: `${sym} atom${l ? `, m component (l = ${l})` : ''}`, owner: Int32Array.from(rws, (q) => (q / dim(l)) | 0), atom: Int32Array.from(rws, (q) => (q / dim(l)) | 0), all: true };
        const sel = g.gather(x.get(l)!, ix(rws, N * dim(l), kind, kindA(l)), `${sym}_rows`);
        const y = g.linear(sel, this.p(`${name}.${l}.${sym}`), undefined, `skip_${sym}`);
        const placed = g.segmentSum(y, ix(rws, N * dim(l), kind, kindA(l)), `${sym}_back`);
        out = out ? g.add(out, placed, 'add') : placed;
      }
      return [l, out!] as const;
    }));
    /** l blocks [atoms (2l+1), C] -> [atoms C, (lmax+1)^2]: mace's reshape_irreps, rows (atom, channel). */
    const toChannels = (x: Map<number, Tensor>, ls: number[]) => {
      let flat: Tensor | null = null, base = 0;
      const bases = new Map<number, number>();
      for (const l of ls) {
        const t = g.reshape(x.get(l)!, [N * dim(l) * C, 1], undefined, `flatten_l${l}`);
        bases.set(l, base);
        base += N * dim(l) * C;
        flat = flat ? g.concatRows(flat, t) : t;
      }
      const n = ls.reduce((s, l) => s + dim(l), 0), offs = ls.map((_, i) => ls.slice(0, i).reduce((s, l) => s + dim(l), 0));
      const idx = new Int32Array(N * C * n);
      for (let a = 0; a < N; a++)
        for (let c = 0; c < C; c++)
          ls.forEach((l, li) => { for (let mm = 0; mm < dim(l); mm++) idx[(a * C + c) * n + offs[li] + mm] = bases.get(l)! + (a * dim(l) + mm) * C + c; });
      return g.reshape(g.gather(flat!, ix(idx, base), 'transpose'), [N * C, n], 'channel', 'per_channel');
    };
    /** [atoms C, 2L+1] -> [atoms (2L+1), C] */
    const toBlock = (x: Tensor, l: number) => {
      const d = dim(l), idx = new Int32Array(N * d * C);
      for (let a = 0; a < N; a++) for (let mm = 0; mm < d; mm++) for (let c = 0; c < C; c++) idx[(a * d + mm) * C + c] = (a * C + c) * d + mm;
      return g.reshape(g.gather(g.reshape(x, [N * C * d, 1], undefined, 'flatten'), ix(idx, N * C * d), 'transpose'), [N * d, C], kindA(l), `B_l${l}`);
    };

    // ---- embedding
    let feats = new Map<number, Tensor>([[0, g.scope('embedding', () => g.gather(this.p('node_embedding'), ix(zi, m.atomic_numbers.length, 'atom'), 'node_embedding'))]]);
    const internals: MaceInternals = { sh: Y, radial: ef, pair, layers: [] };
    const readouts: Tensor[] = [];
    m.interactions.forEach((it, i) => {
      const pr = m.products[i];
      const message = g.scope(`interaction.${i}`, () => {
        const sc = it.residual ? g.scope('skip', () => perElement(`interactions.${i}.skip`, feats, it.skip_ls)) : null;
        const up = new Map(it.ls_in.map((l) => [l, g.linear(feats.get(l)!, this.p(`interactions.${i}.linear_up.${l}`), undefined, `linear_up_l${l}`)]));
        const w = g.scope('radial_mlp', () => {
          let h = ef;
          it.radial.forEach((_, q) => {
            h = g.linear(h, this.p(`interactions.${i}.radial.${q}`), undefined, `layer${q}`);
            if (q < it.radial.length - 1) h = g.scale(g.silu(h), it.radial_act, 'act');
          });
          return h;
        });
        const msg = g.scope('message', () => {
          const xj = new Map(it.ls_in.map((l) => [l, g.gather(up.get(l)!, ix(blockRows(E, dim(l), (e, mm) => send[e] * dim(l) + mm), N * dim(l), kindE(l), kindA(l)), `x_j_l${l}`)]));
          const out = new Map<number, Tensor>();
          it.paths.forEach((p, q) => {
            const d1 = dim(p.l1), d3 = dim(p.l3), tag = `${p.l1}x${p.l2}→${p.l3}`;
            const T = g.linear(g.sliceCols(Y, p.l2 * p.l2, dim(p.l2), `Y_l${p.l2}`), this.c(`cg.${i}.${q}`), undefined, `coupling_${tag}`);
            const tp = g.rowMix(xj.get(p.l1)!, T, d1, d3, `tensor_product_${tag}`);
            const wq = g.gather(g.sliceCols(w, p.w, C, `R_${tag}`), ix(blockRows(E, d3, (e) => e), E, kindE(p.l3), 'edge'), `R_${tag}`);
            const mji = g.mul(tp, wq, `m_ji_${tag}`);
            const summed = g.segmentSum(mji, ix(blockRows(E, d3, (e, mm) => recv[e] * d3 + mm), N * d3, kindE(p.l3), kindA(p.l3)), `sum_${tag}`);
            const lt = it.ls_target[p.out], y = g.linear(summed, this.p(`interactions.${i}.linear.${q}`), undefined, `linear_${tag}`);
            out.set(lt, out.has(lt) ? g.add(out.get(lt)!, y, 'add') : y);
          });
          return out;
        });
        let norm: Map<number, Tensor>;
        if (it.kind === 'density') {
          norm = g.scope('density', () => {
            const de = g.unary('tanh', g.unary('square', g.linear(ef, this.p(`interactions.${i}.density`), undefined, 'density_fn')));
            const dens = g.add(g.segmentSum(g.reshape(de, [E], 'edge'), ix(recv, N, 'edge', 'atom'), 'density'), one, 'density_plus_1');
            return new Map(it.ls_target.map((l) => [l, g.div(msg.get(l)!, g.gather(dens, ix(blockRows(N, dim(l), (n) => n), N, kindA(l), 'atom'), `density_l${l}`), `A_l${l}`)]));
          });
        } else norm = new Map(it.ls_target.map((l) => [l, g.scale(msg.get(l)!, 1 / it.avg_num_neighbors!, `A_l${l}`)]));
        const A = it.residual ? norm : g.scope('skip', () => perElement(`interactions.${i}.skip`, norm, it.skip_ls));
        return { A, sc };
      });
      const next = g.scope(`product.${i}`, () => {
        const x = toChannels(message.A, it.ls_target);
        const maxNu = Math.max(...pr.contractions.flatMap((c) => c.nu));
        const phi = new Map<number, Tensor>([[1, x]]);
        if (maxNu >= 2) phi.set(2, g.mul(g.linear(x, this.c('pick_a'), undefined, 'x_a'), g.linear(x, this.c('pick_b'), undefined, 'x_b'), 'x_a x_b'));
        if (maxNu >= 3) phi.set(3, g.mul(g.linear(phi.get(2)!, this.c('pick_ab'), undefined, 'x_a x_b'), g.linear(x, this.c('pick_c'), undefined, 'x_c'), 'x_a x_b x_c'));
        const out = new Map<number, Tensor>();
        for (const c of pr.contractions) {
          const d = dim(c.L);
          let acc: Tensor | null = null;
          c.nu.forEach((nu, q) => {
            const K = c.K[q];
            const ck = c.L ? `channel.l${c.L}` : 'channel';
            rows[ck] ??= { label: `channel of an atom${c.L ? `, m component (l = ${c.L})` : ''}`, owner: Int32Array.from(blockRows(N * C, d, (q) => (q / C) | 0)), atom: Int32Array.from(blockRows(N * C, d, (q) => (q / C) | 0)), all: true };
            const Bk = g.reshape(g.linear(phi.get(nu)!, this.c(`U.${i}.${c.L}.${nu}`), undefined, `basis_L${c.L}_nu${nu}`), [N * C * d, K], ck, `basis_L${c.L}_nu${nu}`);
            const Wg = g.gather(this.p(`products.${i}.${c.L}.weights.${nu}`), ix(blockRows(N * C, d, (q) => zi[(q / C) | 0] * C + (q % C)), m.atomic_numbers.length * C, ck), `weights_nu${nu}`);
            const term = g.sumRows(g.mul(Bk, Wg), `B_L${c.L}_nu${nu}`);
            acc = acc ? g.add(acc, term, 'add') : term;
          });
          let h = g.linear(toBlock(acc!, c.L), this.p(`products.${i}.linear.${c.L}`), undefined, `linear_l${c.L}`);
          if (pr.use_sc && message.sc) h = g.add(h, message.sc.get(c.L)!, `residual_l${c.L}`);
          out.set(c.L, h);
        }
        return out;
      });
      feats = next;
      const ro = m.readouts[i];
      const e = g.scope(`readout.${i}`, () => {
        const h0 = feats.get(0)!;
        const y = ro.kind === 'linear'
          ? g.linear(h0, this.p(`readouts.${i}.linear`), undefined, 'linear')
          : g.linear(g.scale(g.silu(g.linear(h0, this.p(`readouts.${i}.linear_1`), undefined, 'linear_1')), ro.act, 'act'),
                     this.p(`readouts.${i}.linear_2`), undefined, 'linear_2');
        return g.reshape(y, [N], 'atom', 'atomic_energy');
      });
      readouts.push(e);
      internals.layers.push({ message: message.A, product: next, readout: e });
    });

    const perAtom = g.scope('energy', () => {
      let s = pair ?? null;
      for (const e of readouts) s = s ? g.add(s, e, 'add') : e;
      let inter = g.scale(s!, m.scale, 'scaled').withUnit(eV);
      if (m.shift) inter = g.add(inter, k([m.shift], [1], 'shift'), 'shifted');
      return g.add(inter, k(zi.map((e) => m.e0[e]), [N], 'E0', 'atom'), 'atomic_energy');
    });
    const energy = g.scope('energy', () => g.sumAll(perAtom, 'total_energy'));
    const graph = { center: recv, neighbor: send, shift: Float32Array.from(nl.shiftVec), label: 'message-passing' };
    return { energy, perAtom, positions, virialVectors: v, rows, graph, internals };
  }
}
