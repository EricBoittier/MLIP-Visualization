// ANI-2x (Smith, Devereux et al.), a Behler–Parrinello network: atomic
// environment vectors (radial and angular symmetry functions, per neighbour
// element and element pair) feed one network per element; an ensemble of 8
// such models is averaged and per-element self energies are added. Follows
// TorchANI's pyaev implementation.
import type { Backend } from '../../engine/backend';
import { type Graph, Tensor } from '../../engine/tensor';
import type { Tensors } from '../../common/safetensors';
import { neighborList, type System } from '../../common/structure';
import { atomRows, edgeRows, type Forward, type Model, range, type RowSpace } from '../types';

export interface AniMeta {
  kind: 'ani';
  name: string;
  elements: number[]; // atomic numbers, in the model's species order
  symbols: string[];
  members: number;
  layers: Record<string, number[]>; // layer widths per element, input first
  activation: { name: 'celu'; alpha: number };
  radial: { cutoff: number; eta: number; shifts: number[] };
  angular: { cutoff: number; eta: number; zeta: number; shifts: number[]; sections: number[] };
  self_energies: number[]; // Hartree, per species
  hartree_to_ev: number;
  source?: string;
}

/** Index of an unordered species pair in the upper triangle (TorchANI's triu_index). */
export function pairIndex(S: number) {
  const t: number[][] = Array.from({ length: S }, () => new Array(S).fill(0));
  let k = 0;
  for (let a = 0; a < S; a++) for (let b = a; b < S; b++) { t[a][b] = t[b][a] = k++; }
  return t;
}

export class ANI implements Model {
  readonly kind = 'ani' as const;
  readonly params = new Map<string, Tensor>();
  get elements() { return this.meta.elements; }

  constructor(readonly be: Backend, readonly meta: AniMeta, tensors: Tensors) {
    for (const [name, t] of tensors) this.params.set(name, new Tensor(t.shape, be.upload(t.data), false, name));
  }

  forward(g: Graph, sys: System, opts: { forces?: boolean } = {}): Forward {
    const m = this.meta, S = m.elements.length, NP = (S * (S + 1)) / 2;
    const R = m.radial, A = m.angular, nR = R.shifts.length, nA = A.shifts.length, nZ = A.sections.length;
    const N = sys.numbers.length;
    const spec = Int32Array.from(sys.numbers, (z) => m.elements.indexOf(z));
    const nl = neighborList(sys, R.cutoff);
    const P = nl.center.length;
    const ix = (a: Int32Array | number[], nSrc: number, out?: string, src?: string) => g.index(Int32Array.from(a), nSrc, { out, src });
    const c = (data: number[] | Float32Array, shape: number[], name: string, kind?: string) =>
      g.constant(Float32Array.from(data), shape, name, kind);

    const positions = g.constant(Float32Array.from(sys.positions.flat()), [N, 3], 'positions', 'atom');
    positions.requiresGrad = !!opts.forces;

    // ---- pairs within the radial cutoff
    const { v, d } = g.scope('geometry', () => {
      const v = g.add(g.sub(g.gather(positions, ix(nl.neighbor, N, 'pair', 'atom'), 'r_j'),
                            g.gather(positions, ix(nl.center, N, 'pair', 'atom'), 'r_i')),
                      c(Float32Array.from(nl.shiftVec), [P, 3], 'shift', 'pair'));
      return { v, d: g.rowNorm(v) };
    });

    // ---- radial AEV: 0.25 exp(-eta (r - mu)^2) fc(r), summed per neighbour element
    const radial = g.scope('radial', () => {
      const fc = g.cutoff('cosine', d, c(new Array(P).fill(R.cutoff), [P], 'cutoff', 'pair'), R.cutoff);
      const t = g.sub(g.repeatCols(d, nR, 'distances'), c(R.shifts, [1, nR], 'shifts'));
      const terms = g.mul(g.scale(g.unary('exp', g.scale(g.unary('square', t), -R.eta)), 0.25), fc);
      const slots = ix(Array.from(nl.center, (i, p) => i * S + spec[nl.neighbor[p]]), N * S, 'pair', 'atom_species');
      return g.reshape(g.segmentSum(terms, slots, 'sum_per_element'), [N, S * nR], 'atom', 'radial_aev');
    });

    // ---- angular AEV over pairs of neighbours (j, k) of each atom within the angular cutoff
    const tri: [number, number, number][] = []; // centre, pair j, pair k
    const byCenter: number[][] = Array.from({ length: N }, () => []);
    nl.dist.forEach((r, p) => { if (r < A.cutoff) byCenter[nl.center[p]].push(p); });
    byCenter.forEach((ps, i) => { for (let a = 0; a < ps.length; a++) for (let b = a + 1; b < ps.length; b++) tri.push([i, ps[a], ps[b]]); });
    const T = tri.length, pidx = pairIndex(S);
    const angular = g.scope('angular', () => {
      if (!T) return g.constant(new Float32Array(N * NP * nA * nZ), [N, NP * nA * nZ], 'angular_aev', 'atom');
      const tj = ix(tri.map((t) => t[1]), P, 'triplet', 'pair'), tk = ix(tri.map((t) => t[2]), P, 'triplet', 'pair');
      const vj = g.gather(v, tj, 'r_ij'), vk = g.gather(v, tk, 'r_ik');
      const dj = g.gather(d, tj, '|r_ij|'), dk = g.gather(d, tk, '|r_ik|');
      const cos = g.div(g.sumRows(g.mul(vj, vk), 'dot'), g.unary('clamp', g.mul(dj, dk), 1e-10, Infinity));
      const theta = g.unary('acos', g.scale(cos, 0.95));
      // angle part: 2 ((1 + cos(theta - theta_s)) / 2)^zeta
      const dth = g.unary('cos', g.sub(g.repeatCols(theta, nZ, 'angles'), c(A.sections, [1, nZ], 'sections')));
      const ang = g.scale(g.unary('pow', g.unary('clamp', g.scale(g.add(dth, c([1], [1], 'one')), 0.5), 0, Infinity), A.zeta), 2);
      // radial part: exp(-eta ((r_ij + r_ik) / 2 - mu)^2)
      const mean = g.scale(g.add(dj, dk), 0.5);
      const rad = g.unary('exp', g.scale(g.unary('square', g.sub(g.repeatCols(mean, nA, 'mean_distances'), c(A.shifts, [1, nA], 'shifts'))), -A.eta));
      // outer product: feature a * nZ + s
      const eR = new Float32Array(nA * nZ * nA), eZ = new Float32Array(nA * nZ * nZ);
      for (let a = 0; a < nA; a++) for (let s = 0; s < nZ; s++) { eR[(a * nZ + s) * nA + a] = 1; eZ[(a * nZ + s) * nZ + s] = 1; }
      const outer = g.mul(g.linear(rad, c(eR, [nA * nZ, nA], 'expand'), undefined, 'expand_radial'),
                          g.linear(ang, c(eZ, [nA * nZ, nZ], 'expand'), undefined, 'expand_angle'));
      const rc = c(new Array(T).fill(A.cutoff), [T], 'cutoff', 'triplet');
      const terms = g.mul(outer, g.mul(g.cutoff('cosine', dj, rc, A.cutoff), g.cutoff('cosine', dk, rc, A.cutoff)));
      const slots = ix(tri.map(([i, p, q]) => i * NP + pidx[spec[nl.neighbor[p]]][spec[nl.neighbor[q]]]), N * NP, 'triplet', 'atom_pairs');
      return g.reshape(g.segmentSum(terms, slots, 'sum_per_element_pair'), [N, NP * nA * nZ], 'atom', 'angular_aev');
    });

    const aev = g.scope('aev', () => g.concatCols([radial, angular], 'aev'));

    // ---- one network per element, averaged over the ensemble
    const rows: Record<string, RowSpace> = {
      atom: atomRows(N), pair: { ...edgeRows(nl.center), label: 'pair' },
      triplet: { label: 'angular triplet', owner: Int32Array.from(tri, (t) => t[0]) },
      atom_species: { label: 'neighbour element', owner: Int32Array.from({ length: N * S }, (_, r) => (r / S) | 0) },
      atom_pairs: { label: 'element pair', owner: Int32Array.from({ length: N * NP }, (_, r) => (r / NP) | 0) },
    };
    let atomic: Tensor | null = null;
    g.scope('networks', () => {
      m.symbols.forEach((sym, s) => {
        const idx = range(N).filter((i) => spec[i] === s);
        if (!idx.length) return;
        const kind = `atom_${sym}`;
        rows[kind] = { label: `${sym} atom`, owner: idx, atom: idx, all: true };
        g.scope(sym, () => {
          const x = g.gather(aev, ix(idx, N, kind, 'atom'), `${sym}_atoms`);
          let sum: Tensor | null = null;
          for (let k = 0; k < m.members; k++) {
            const out = g.scope(`member${k}`, () => {
              const L = m.layers[sym].length - 1;
              let h = x;
              for (let l = 0; l < L; l++) {
                h = g.linear(h, this.params.get(`member${k}.${sym}.${l}.weight`)!, this.params.get(`member${k}.${sym}.${l}.bias`)!, l < L - 1 ? `layer${l}` : 'output');
                if (l < L - 1) h = g.unary('celu', h, m.activation.alpha);
              }
              return h;
            });
            sum = sum ? g.add(sum, out) : out;
          }
          const mean = g.scale(sum!, 1 / m.members, 'ensemble_mean');
          const placed = g.segmentSum(mean, ix(idx, N, kind, 'atom'), 'to_atoms');
          atomic = atomic ? g.add(atomic, placed) : placed;
        });
      });
    });

    const perAtom = g.scope('energy', () => g.scale(
      g.add(g.sumRows(atomic!), c(Array.from(spec, (s) => m.self_energies[s]), [N], 'self_energies', 'atom')),
      m.hartree_to_ev, 'to_eV'));
    const energy = g.scope('energy', () => g.sumAll(perAtom, 'total_energy'));
    return { energy, perAtom, positions, virialVectors: v, rows,
             graph: { center: nl.center, neighbor: nl.neighbor, shift: Float32Array.from(nl.shiftVec), label: 'symmetry-function pair' } };
  }
}
