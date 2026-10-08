// PhysNet as implemented in mmml's physnetjax (models/model.py), invariant form
// (max_degree = 0): an element embedding, num_iterations rounds of e3x message
// passing (a filter from the radial basis times the neighbour's features times a
// learned per-feature weight, summed over neighbours) each followed by residual
// refinement blocks, then per-atom energy and charge heads with per-element
// biases, switched and erf-damped electrostatics between the predicted charges,
// and ZBL repulsion at very short range.
import type { Backend } from '../../engine/backend';
import { type Graph, Tensor } from '../../engine/tensor';
import type { Tensors } from '../../common/safetensors';
import { isPeriodic, neighborList, type System } from '../../common/structure';
import { atomRows, type Forward, type Model, range, type RowSpace } from '../types';

export interface PhysNetConfig {
  features: number;
  max_degree: number;
  num_iterations: number;
  num_basis_functions: number;
  cutoff: number;
  max_atomic_number: number;
  charges?: boolean;
  n_refinement_blocks?: number;
  n_res?: number;
  zbl?: boolean;
  zbl_cuton?: number | null;
  zbl_cutoff?: number;
  use_energy_bias?: boolean;
  include_electrostatics?: boolean;
  electrostatics_damping_sigma?: number;
  switch_start?: number;
  switch_end?: number;
  electrostatics_off_start?: number;
  electrostatics_off_end?: number;
}

export interface PhysNetMeta { kind: 'physnet'; name?: string; source?: string; config: PhysNetConfig; elements: number[] }

const COULOMB_PAIR = 7.199822675975274; // eV A, half of e^2 / (4 pi eps0): every pair is counted twice
const COULOMB = 14.3996454784255;
const BOHR = 0.529177249;

export class PhysNet implements Model {
  readonly kind = 'physnet' as const;
  readonly params = new Map<string, Tensor>();
  readonly c: Required<Omit<PhysNetConfig, 'n_res' | 'zbl_cuton'>> & { zbl_cuton: number };
  get elements() { return this.meta.elements; }

  constructor(readonly be: Backend, readonly meta: PhysNetMeta, tensors: Tensors) {
    const c = meta.config;
    if (c.max_degree !== 0) throw new Error(`PhysNet with max_degree = ${c.max_degree} is not supported (only the invariant max_degree = 0)`);
    this.c = {
      ...c, charges: c.charges ?? false, n_refinement_blocks: c.n_refinement_blocks ?? c.n_res ?? 3, zbl: c.zbl ?? true,
      zbl_cuton: c.zbl_cuton ?? 0.1, zbl_cutoff: c.zbl_cutoff ?? 0.6, use_energy_bias: c.use_energy_bias ?? false,
      include_electrostatics: c.include_electrostatics ?? true, electrostatics_damping_sigma: c.electrostatics_damping_sigma ?? 4,
      switch_start: c.switch_start ?? 1, switch_end: c.switch_end ?? 10, electrostatics_off_start: c.electrostatics_off_start ?? 8,
      electrostatics_off_end: c.electrostatics_off_end ?? 10,
    };
    if (this.c.n_refinement_blocks < 1) throw new Error('PhysNet needs at least one refinement block');
    for (const [name, t] of tensors) this.params.set(name, new Tensor(t.shape, be.upload(t.data), false, name));
    // ZBL constants: trained values if the checkpoint has them (as physnetjax takes absolute values)
    const r = (n: string, d: number[]) => Array.from(tensors.get(`repulsion.${n}`)?.data ?? d, Math.abs);
    this.zbl = { aCoef: r('a_coefficient', [BOHR * 0.8854])[0], aExp: r('a_exponent', [0.23])[0],
                 phiC: r('phi_coefficients', [0.18175, 0.50986, 0.28022, 0.02817]), phiE: r('phi_exponents', [3.1998, 0.94229, 0.4029, 0.20162]) };
  }
  private zbl: { aCoef: number; aExp: number; phiC: number[]; phiE: number[] };

  private p(name: string) {
    const t = this.params.get(name);
    if (!t) throw new Error(`PhysNet: missing parameter ${name}`);
    return t;
  }

  forward(g: Graph, sys: System, opts: { forces?: boolean } = {}): Forward {
    if (isPeriodic(sys)) throw new Error('periodic structures are not supported by this PhysNet');
    const c = this.c, N = sys.numbers.length, F = c.features, K = c.num_basis_functions;
    const elec = c.charges && c.include_electrostatics;
    const rmax = Math.max(c.cutoff, elec ? c.electrostatics_off_end : 0, c.zbl ? c.zbl_cutoff : 0);
    const nl = neighborList(sys, rmax);
    const P = nl.center.length;
    const pick = (pred: (p: number) => boolean) => range(P).filter(pred);
    const mp = pick((p) => nl.dist[p] < c.cutoff);
    const es = elec ? pick((p) => nl.dist[p] < c.electrostatics_off_end) : new Int32Array(0);
    const zb = c.zbl ? pick((p) => nl.dist[p] < c.zbl_cutoff) : new Int32Array(0);
    const ix = (a: ArrayLike<number>, nSrc: number, out?: string, src?: string) => g.index(Int32Array.from(a), nSrc, { out, src });
    const k = (data: ArrayLike<number>, shape: number[], name: string, kind?: string) => g.constant(Float32Array.from(data), shape, name, kind);
    const one = k([1], [1], 'one');
    const Zidx = ix(sys.numbers, c.max_atomic_number + 1, 'atom');

    const positions = k(sys.positions.flat(), [N, 3], 'positions', 'atom');
    positions.requiresGrad = !!opts.forces;
    const v = g.scope('geometry', () => g.add(
      g.sub(g.gather(positions, ix(nl.neighbor, N, 'pair', 'atom'), 'r_j'), g.gather(positions, ix(nl.center, N, 'pair', 'atom'), 'r_i')),
      k(nl.shiftVec, [P, 3], 'shift', 'pair')));

    // ---- radial basis: Chebyshev polynomials of exp(-r), times a smooth cutoff
    const basis = g.scope('basis', () => {
      const vm = g.gather(v, ix(mp, P, 'mp_pair', 'pair'), 'pairs_in_cutoff');
      const d = g.rowNorm(vm);
      const theta = g.unary('acos', g.add(g.scale(g.unary('exp', g.unary('neg', d)), 2), k([-1], [1], 'minus_one')));
      const cheb = g.unary('cos', g.mul(g.repeatCols(theta, K, 'angles'), k(range(K), [1, K], 'orders')), 0, 0);
      const x2 = g.scale(g.unary('square', d), 1 / (c.cutoff * c.cutoff));
      const cut = g.unary('exp', g.add(g.unary('neg', g.unary('pow', g.add(g.unary('neg', x2), one), -1)), one), 0, 0);
      return g.mul(cheb, cut, 'basis');
    });
    const mpDst = ix(Array.from(mp, (p) => nl.center[p]), N, 'mp_pair', 'atom');
    const mpSrc = ix(Array.from(mp, (p) => nl.neighbor[p]), N, 'mp_pair', 'atom');

    let x = g.scope('embedding', () => g.gather(this.p('Embed_0.embedding'), Zidx, 'embedding'));
    let dense = 0;
    const lin = (t: Tensor, name: string, bias = true) => g.linear(t, this.p(`${name}.kernel`), bias && this.params.has(`${name}.bias`) ? this.p(`${name}.bias`) : undefined, name.split('.')[0]);
    for (let it = 0; it < c.num_iterations; it++) {
      x = g.scope(`interaction.${it}`, () => {
        const m = g.scope('message', () => {
          const filters = lin(basis, `MessagePass_${it}.filter.0+`, false);
          const xj = g.gather(x, mpSrc, 'x_j');
          const prod = g.mul(g.mul(filters, xj, 'filter_x_j'), this.p(`MessagePass_${it}.tensor.kernel`), 'tensor_product');
          return g.segmentSum(prod, mpDst, 'sum_messages');
        });
        return g.scope('refinement', () => {
          let h = m, y: Tensor = m;
          for (let b = 0; b < c.n_refinement_blocks; b++) {
            y = lin(g.add(h, g.silu(h)), `Dense_${dense++}.0+`);
            h = g.add(h, y, 'residual');
          }
          return g.silu(lin(y, `Dense_${dense++}.0+`));
        });
      });
    }

    const head = (name: string, biasName: string, use: boolean) => {
      const a = lin(lin(x, `Dense_${dense++}.0+`, false), `Dense_${dense++}`, false);
      const out = g.sumRows(a, 'per_atom');
      return use ? g.add(out, g.gather(this.p(biasName), Zidx, `${biasName}`)) : out;
    };
    let perAtom = g.scope('energy_head', () => head('energy', 'energy_bias', c.use_energy_bias));
    const rows: Record<string, RowSpace> = {
      atom: atomRows(N),
      pair: { label: 'pair', owner: nl.center },
      mp_pair: { label: 'pair', owner: Int32Array.from(mp, (p) => nl.center[p]), edge: range(mp.length) },
    };

    if (c.charges) {
      const q = g.scope('charges', () => head('charges', 'charge_bias', true));
      if (elec) {
        rows.es_pair = { label: 'electrostatic pair', owner: Int32Array.from(es, (p) => nl.center[p]) };
        const E = g.scope('electrostatics', () => {
          const ve = g.gather(v, ix(es, P, 'es_pair', 'pair'), 'pairs');
          const r2 = g.sumRows(g.unary('square', ve), 'r2');
          const dist = g.unary('sqrt', g.unary('clamp', r2, 1e-4, Infinity));
          const s = g.unary('switch', dist, c.switch_start, c.switch_end);
          const off = g.add(g.unary('neg', g.unary('switch', dist, c.electrostatics_off_start, c.electrostatics_off_end)), one, 'off');
          const r1 = g.div(s, g.unary('sqrt', g.add(r2, one)), 'short_range');
          const rl = g.div(g.add(g.unary('neg', s), one), g.add(dist, k([1e-6], [1], 'eps')), 'long_range');
          let rr = g.add(r1, rl, '1/r');
          if (c.electrostatics_damping_sigma > 0) rr = g.mul(rr, g.unary('erf', g.scale(dist, 1 / c.electrostatics_damping_sigma)), 'damped');
          const eshift = g.add(g.scale(dist, 1 / c.switch_end ** 2), k([1e-6 / c.switch_end ** 2 - 2 / c.switch_end], [1], 'shift'), 'eshift');
          const qi = g.unary('clamp', g.gather(q, ix(Array.from(es, (p) => nl.center[p]), N, 'es_pair', 'atom'), 'q_i'), -10, 10);
          const qj = g.unary('clamp', g.gather(q, ix(Array.from(es, (p) => nl.neighbor[p]), N, 'es_pair', 'atom'), 'q_j'), -10, 10);
          const pair = g.mul(g.scale(g.mul(qi, qj), COULOMB_PAIR), g.mul(g.add(rr, eshift), off), 'pair_energy');
          return g.segmentSum(pair, ix(Array.from(es, (p) => nl.center[p]), N, 'es_pair', 'atom'), 'per_atom');
        });
        perAtom = g.add(perAtom, E, 'with_electrostatics');
      }
    }

    if (zb.length) {
      rows.zbl_pair = { label: 'short-range pair', owner: Int32Array.from(zb, (p) => nl.center[p]) };
      const E = g.scope('repulsion', () => {
        const { aCoef, aExp, phiC, phiE } = this.zbl;
        const norm = Math.hypot(...phiC);
        const Zi = Array.from(zb, (p) => sys.numbers[nl.center[p]]), Zj = Array.from(zb, (p) => sys.numbers[nl.neighbor[p]]);
        const a = Zi.map((z, n) => aCoef / (z ** aExp + Zj[n] ** aExp));
        const vz = g.gather(v, ix(zb, P, 'zbl_pair', 'pair'), 'pairs');
        const r = g.unary('clamp', g.rowNorm(vz), 1e-8, Infinity);
        const xr = g.div(r, k(a, [zb.length], 'screening_length', 'zbl_pair'));
        const phi = g.linear(g.unary('exp', g.mul(g.repeatCols(xr, 4, 'x'), k(phiE.map((e) => -e), [1, 4], 'exponents'))),
                             k(phiC.map((q) => q / norm), [1, 4], 'coefficients'), undefined, 'screening');
        const t = g.unary('clamp', g.scale(g.add(g.unary('neg', r), k([c.zbl_cutoff], [1], 'cutoff')), 1 / (c.zbl_cutoff - c.zbl_cuton)), 0, 1);
        const sw = g.mul(g.mul(g.unary('square', t), t), g.add(g.mul(g.add(g.scale(t, 6), k([-15], [1], 'c')), t), k([10], [1], 'c')), 'switch');
        const coul = g.div(k(Zi.map((z, n) => 0.5 * COULOMB * z * Zj[n]), [zb.length], 'ZiZj'), r, 'coulomb');
        const pair = g.mul(g.mul(coul, g.sumRows(phi)), sw, 'pair_energy');
        return g.segmentSum(pair, ix(Array.from(zb, (p) => nl.center[p]), N, 'zbl_pair', 'atom'), 'per_atom');
      });
      perAtom = g.add(perAtom, E, 'with_repulsion');
    }

    const energy = g.scope('energy', () => g.sumAll(perAtom, 'total_energy'));
    const graph = { center: Int32Array.from(mp, (p) => nl.center[p]), neighbor: Int32Array.from(mp, (p) => nl.neighbor[p]),
                    shift: new Float32Array(3 * mp.length), label: 'message-passing' };
    return { energy, perAtom, positions, rows, graph };
  }
}
