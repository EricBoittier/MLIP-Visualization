// LOREM (Bigi, Chong, Grasselli, Klawohn, Loche, Ceriotti, arXiv:2507.19382) as metatrain's
// experimental port computes it. A short-range equivariant density — Bernstein polynomials,
// Racah real spherical harmonics, one Clebsch–Gordan self-product, then message passing — and a
// long-range head that reads charges out of those features and evaluates their Coulomb potential.
// Periodic cells use an Ewald sum (torch-pme's formula, smearing = cutoff/4, wavelength = cutoff/8);
// molecules use plain 1/r over every pair. Each stage adds its own atomic-energy residual.
//
// scripts/convert_lorem.py writes the weights. A degree-wise matrix is stored [ℓ, out, in]
// (only ℓ = 0 would carry a bias). Each `*.kernel` is that module's learned tensor weight
// already multiplied by its Clebsch–Gordan coefficients.
import type { Backend } from '../../engine/backend';
import { type Graph, type Index, type Tensor } from '../../engine/tensor';
import { eV, Å } from '../../engine/units';
import type { Tensors } from '../../common/safetensors';
import { SYMBOLS } from '../../common/elements';
import { isPeriodic, neighborList, volume, type System } from '../../common/structure';
import { atomRows, type Forward, type Model, range, type RowSpace } from '../types';

export interface LoremMeta {
  kind: 'lorem';
  name?: string;
  /** Elements this checkpoint was built for. The embedding itself is indexed by atomic number. */
  atomic_numbers: number[];
  embedding_rows: number;
  cutoff: number;
  max_degree: number;
  max_degree_lr: number;
  num_features: number;
  num_spherical_features: number;
  num_radial: number;
  num_species: number;
  num_message_passing: number;
  equivariant_message_passing: boolean;
  initialize_node_features: boolean;
  /** Ewald Gaussian width (cutoff / 4) and reciprocal-space wavelength (cutoff / 8). */
  smearing: number;
  lr_wavelength: number;
}

export interface LoremInternals { sr: Tensor; lr: Tensor; charges: Tensor }

const nlmOf = (lmax: number) => (lmax + 1) ** 2;

function binomialRow(num: number) {
  const c = [1];
  for (let k = 1; k < num; k++) c.push(c[k - 1] * (num - k) / k);
  return c;
}

/** torch.fft.fftfreq(n) * n: 0, 1, …, then the negative frequencies. */
function fftInt(n: number) {
  const out: number[] = [];
  for (let i = 0; i < n; i++) out.push(i <= (n - 1 >> 1) ? i : i - n);
  return out;
}

function inv3(m: number[][]) {
  const det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const cof = (i: number, j: number) => {
    const r = [0, 1, 2].filter((x) => x !== i), s = [0, 1, 2].filter((x) => x !== j);
    return (m[r[0]][s[0]] * m[r[1]][s[1]] - m[r[0]][s[1]] * m[r[1]][s[0]]) * ((i + j) % 2 ? -1 : 1);
  };
  return [0, 1, 2].map((i) => [0, 1, 2].map((j) => cof(j, i) / det));
}

/** Reciprocal vectors of torch-pme's Ewald sum, in its fftfreq order. k = 0 is among them. */
function ewaldK(cell: number[][], wavelength: number) {
  const kCut = 2 * Math.PI / wavelength;
  const ns = [0, 1, 2].map((i) => Math.ceil(kCut * Math.hypot(...cell[i]) / (2 * Math.PI)));
  const inv = inv3(cell);
  const recip = [0, 1, 2].map((r) => [0, 1, 2].map((c) => 2 * Math.PI * inv[c][r]));
  const ks: number[][] = [];
  for (const mx of fftInt(ns[0])) for (const my of fftInt(ns[1])) for (const mz of fftInt(ns[2]))
    ks.push([0, 1, 2].map((c) => mx * recip[0][c] + my * recip[1][c] + mz * recip[2][c]));
  return ks;
}

export class LOREM implements Model {
  readonly kind = 'lorem' as const;
  readonly params = new Map<string, Tensor>();
  get elements() { return this.meta.atomic_numbers; }

  constructor(readonly be: Backend, readonly meta: LoremMeta, tensors: Tensors) {
    for (const [name, t] of tensors) this.params.set(name, new Tensor(t.shape, be.upload(t.data), false, name));
  }

  private p(name: string) {
    const t = this.params.get(name);
    if (!t) throw new Error(`LOREM: missing parameter ${name}`);
    return t;
  }
  private b(name: string) { return this.params.get(`${name}.bias`); }

  forward(g: Graph, sys: System, opts: { forces?: boolean } = {}): Forward & { internals: LoremInternals } {
    const m = this.meta, N = sys.numbers.length;
    const L = m.max_degree, Lr = m.max_degree_lr, d = m.num_features, s = m.num_spherical_features, R = m.num_radial;
    const M = nlmOf(L), Mr = nlmOf(Lr);
    for (const z of sys.numbers) {
      if (!m.atomic_numbers.includes(z)) throw new Error(`${m.name ?? 'LOREM'} has no parameters for element ${SYMBOLS[z] ?? z}`);
      if (z < 0 || z >= m.embedding_rows) throw new Error(`atomic number ${z} is outside the embedding table`);
    }
    const nl = neighborList(sys, m.cutoff);
    const E = nl.center.length, center = nl.center, neigh = nl.neighbor;
    const ix = (a: ArrayLike<number>, nSrc: number, out?: string, src?: string): Index => g.index(Int32Array.from(a), nSrc, { out, src });
    const k = (data: ArrayLike<number>, shape: number[], name: string, kind?: string) => g.constant(Float32Array.from(data), shape, name, kind);
    const lin = (x: Tensor, name: string) => g.linear(x, this.p(`${name}.weight`), this.b(name), name.split('.').pop()!);
    const mlp = (x: Tensor, name: string) => lin(g.silu(lin(x, `${name}.0`)), `${name}.2`);
    const energyMlp = (x: Tensor, name: string) => g.reshape(lin(g.silu(lin(g.silu(lin(x, `${name}.0`)), `${name}.2`)), `${name}.4`), [N], 'atom', 'atomic_energy');
    const update = (x: Tensor, y: Tensor, name: string) => {
      let h = g.add(x, mlp(y, `${name}.mlp0`));
      h = g.norm('layer', h, this.p(`${name}.norm0.weight`), this.p(`${name}.norm0.bias`), 1e-6);
      h = g.add(h, mlp(h, `${name}.mlp1`));
      return g.norm('layer', h, this.p(`${name}.norm1.weight`), this.p(`${name}.norm1.bias`), 1e-6);
    };
    // Rows (item, m) of one degree, m running −ℓ…+ℓ inside the ℓ² block.
    const degreeRows = (nItems: number, nComp: number, ell: number) => {
      const mdim = 2 * ell + 1, m0 = ell * ell, rows: number[] = [];
      for (let n = 0; n < nItems; n++) for (let mm = 0; mm < mdim; mm++) rows.push(n * nComp + m0 + mm);
      return rows;
    };
    // weight [ℓ, out, in]; a bias, if the layer has one, is added to the ℓ = 0 row only
    const degreeLinear = (x: Tensor, name: string, nItems: number, lmax: number, nComp: number, fout: number) => {
      const W = this.p(`${name}.weight`), bias = this.b(name);
      let acc: Tensor | null = null;
      for (let ell = 0; ell <= lmax; ell++) {
        const rows = degreeRows(nItems, nComp, ell);
        const block = g.gather(x, ix(rows, nItems * nComp), `l${ell}`);
        const y = g.linear(block, g.gather(W, ix(Array.from({ length: fout }, (_, o) => ell * fout + o), W.rows), `W${ell}`),
                           ell === 0 ? bias : undefined, `dense_l${ell}`);
        const placed = g.segmentSum(y, ix(rows, nItems * nComp), `place_l${ell}`);
        acc = acc ? g.add(acc, placed) : placed;
      }
      return acc!;
    };
    // per-degree norm, each scaled by (2ℓ+1)^(1/4): [nItems, (lmax+1) F]
    const degreeNorms = (x: Tensor, nItems: number, lmax: number, nComp: number, F: number) => {
      const eps = k([1e-12], [1], 'eps');
      const cols: Tensor[] = [];
      for (let ell = 0; ell <= lmax; ell++) {
        const mdim = 2 * ell + 1;
        const block = g.gather(x, ix(degreeRows(nItems, nComp, ell), nItems * nComp), `block_l${ell}`);
        const order = new Int32Array(nItems * F * mdim);
        for (let n = 0; n < nItems; n++) for (let f = 0; f < F; f++) for (let mm = 0; mm < mdim; mm++)
          order[(n * F + f) * mdim + mm] = (n * mdim + mm) * F + f;
        const gathered = g.gather(g.reshape(block, [block.size, 1]), ix(order, block.size), 'components');
        const nrm = g.unary('sqrt', g.add(g.sumRows(g.unary('square', g.reshape(gathered, [nItems * F, mdim]))), eps));
        cols.push(g.scale(g.reshape(nrm, [nItems, F], 'atom'), (2 * ell + 1) ** 0.25, `norm_l${ell}`));
      }
      return g.concatCols(cols, 'degree_norms');
    };
    // (n, ℓ, f) -> (n, m, f), the ℓ row repeated 2ℓ+1 times
    const repeatDegree = (x: Tensor, nItems: number, lmax: number) => {
      const src: number[] = [];
      for (let n = 0; n < nItems; n++) for (let ell = 0; ell <= lmax; ell++) for (let mm = 0; mm < 2 * ell + 1; mm++) src.push(n * (lmax + 1) + ell);
      return g.gather(x, ix(src, nItems * (lmax + 1)), 'repeat_degree');
    };
    const scatter = (x: Tensor, who: ArrayLike<number>, nComp: number) => {
      const seg: number[] = [];
      for (let e = 0; e < who.length; e++) for (let mm = 0; mm < nComp; mm++) seg.push(who[e] * nComp + mm);
      return g.segmentSum(x, ix(seg, N * nComp), 'onto_atoms');
    };
    const gatherSph = (x: Tensor, who: ArrayLike<number>, nComp: number, name: string) => {
      const src: number[] = [];
      for (let e = 0; e < who.length; e++) for (let mm = 0; mm < nComp; mm++) src.push(who[e] * nComp + mm);
      return g.gather(x, ix(src, N * nComp), name);
    };
    const bernstein = (r: Tensor) => {
      const x = g.scale(r, 1 / m.cutoff, 'r/cutoff');
      const one = k(Array.from({ length: E }, () => 1), [E], 'one', 'edge');
      const omx = g.sub(one, x, '1-x');
      return g.concatCols(binomialRow(R).map((c, p) => {
        const xp = p === 0 ? one : g.unary('pow', x, p);
        const yp = p === R - 1 ? one : g.unary('pow', omx, R - 1 - p);
        return g.reshape(g.scale(g.mul(xp, yp), c), [E, 1], 'edge', `B${p}`);
      }), 'bernstein');
    };
    // coefficients from the pair MLP, contracted with the radial basis: [E, d]
    const mixRadial = (nodes: Tensor, radial: Tensor, name: string) => {
      const pair = g.concatCols([
        g.gather(nodes, ix(center, N, 'edge', 'atom'), 'center'),
        g.gather(nodes, ix(neigh, N, 'edge', 'atom'), 'neighbor'),
      ], 'pair');
      const coeff = g.reshape(lin(g.silu(lin(pair, `${name}.0`)), `${name}.2`), [E * R, d], 'edge', 'coefficients');
      const scaled = g.mul(coeff, g.reshape(radial, [E * R, 1], 'edge'));
      return g.segmentSum(scaled, ix(Array.from({ length: E * R }, (_, t) => (t / R) | 0), E, 'edge'), 'edge_scalar');
    };
    // edge scalars -> per-degree coefficients, times Y_lm, summed onto the centre atom
    const sphericalEdges = (edges: Tensor, harmonics: Tensor, dense: string) => {
      const coeff = repeatDegree(g.reshape(lin(edges, dense), [E * (L + 1), s], 'edge'), E, L);
      return g.mul(coeff, g.reshape(harmonics, [E * M, 1], 'edge.sph'), 'Y_edges');
    };

    const positions = k(sys.positions.flat(), [N, 3], 'positions', 'atom').withUnit(Å);
    positions.requiresGrad = !!opts.forces;
    const { v, r } = g.scope('geometry', () => {
      const v = g.add(g.sub(g.gather(positions, ix(neigh, N, 'edge', 'atom'), 'r_j'),
                            g.gather(positions, ix(center, N, 'edge', 'atom'), 'r_i')),
                      k(nl.shiftVec, [E, 3], 'shift', 'edge'), 'edges');
      return { v, r: g.rowNorm(v, 0).withUnit(Å) };
    });
    const sh = g.scope('spherical_harmonics', () => {
      const Y = g.sph(g.div(v, r), L, 'Y_lm');
      const scale: number[] = [];
      for (let ell = 0; ell <= L; ell++) for (let mm = 0; mm < 2 * ell + 1; mm++) scale.push(Math.sqrt(4 * Math.PI / (2 * ell + 1)));
      return g.mul(Y, k(scale, [1, M], 'racah_scale'), 'racah');
    });
    const radial = g.scope('radial', () => {
      const cutoff = g.scale(g.add(g.unary('cos', g.scale(r, Math.PI / m.cutoff)), k([1], [1], 'one')), 0.5, 'cutoff');
      return g.mul(bernstein(r), cutoff, 'radial_basis');
    });
    const embed = g.scope('embedding', () =>
      g.gather(this.p('sr.chemical_embedding.weight'), ix(sys.numbers, m.embedding_rows, 'atom'), 'species'));

    let edgeScalar!: Tensor;
    let nodes = g.scope('density', () => {
      edgeScalar = mixRadial(embed, radial, 'sr.radial_coefficients');
      const h0 = m.initialize_node_features ? lin(embed, 'sr.dense0.0') : k(new Float32Array(N * d), [N, d], 'zeros', 'atom');
      const summed = g.segmentSum(lin(edgeScalar, 'sr.dense1'), ix(center, N, 'atom', 'edge'), 'messages');
      return update(h0, summed, 'sr.update0');
    });
    let spherical = g.scope('spherical', () => {
      const summed = scatter(sphericalEdges(edgeScalar, sh, 'sr.dense2'), center, M);
      const projected = degreeLinear(summed, 'sr.tensor_dense.dense', N, L, M, 2 * s);
      const self = g.couple(g.sliceCols(projected, 0, s), g.sliceCols(projected, s, s), this.p('sr.tensor_dense.kernel'), M, M, M, 'tensor_dense');
      self.kind = 'sph';
      nodes = update(nodes, degreeNorms(self, N, L, M, s), 'sr.update1');
      return self;
    });
    let sr = g.scope('readout', () => energyMlp(nodes, 'sr.energy_mlp'));

    for (let step = 0; step < m.num_message_passing; step++) {
      const pre = `sr.message_passing.${step}`;
      const out = g.scope('message', () => {
        const edges = mixRadial(nodes, radial, `${pre}.radial_coefficients`);
        let scalar = update(nodes, g.segmentSum(lin(edges, `${pre}.edge_dense`), ix(center, N, 'atom', 'edge'), 'messages'), `${pre}.update_edges`);
        let sph = spherical;
        if (m.equivariant_message_passing) {
          const basis = sphericalEdges(edges, sh, `${pre}.coeff_dense`);
          const filtered = degreeLinear(basis, `${pre}.message_pass.filter`, E, L, M, s);
          const sent = gatherSph(spherical, neigh, M, 'neighbor');
          const messages = scatter(g.couple(filtered, sent, this.p(`${pre}.message_pass.message_tensor.kernel`), M, M, M, 'cg_messages'), center, M);
          messages.kind = 'sph';
          const dx = degreeLinear(spherical, `${pre}.message_pass.combine_dense_x`, N, L, M, s);
          const dm = degreeLinear(messages, `${pre}.message_pass.combine_dense_m`, N, L, M, s);
          sph = g.couple(dx, dm, this.p(`${pre}.message_pass.combine_tensor.kernel`), M, M, M, 'combine');
          sph.kind = 'sph';
          const updated = update(scalar, degreeNorms(sph, N, L, M, s), `${pre}.update_norms`);
          return { scalar: updated, sph, energy: energyMlp(updated, `${pre}.energy_mlp`) };
        }
        return { scalar, sph, energy: energyMlp(scalar, `${pre}.energy_mlp`) };
      });
      nodes = out.scalar;
      spherical = out.sph;
      sr = g.add(sr, out.energy, 'add_sr');
    }

    const charges = g.scope('charges', () => {
      const scalar = lin(g.silu(lin(nodes, 'lr.scalar_charge_mlp.0')), 'lr.scalar_charge_mlp.2');
      const projected = degreeLinear(spherical, 'lr.spherical_charge_dense.dense', N, L, M, 2);
      const coupled = g.couple(g.sliceCols(projected, 0, 1), g.sliceCols(projected, 1, 1), this.p('lr.spherical_charge_dense.kernel'), M, M, Mr, 'spherical_charges');
      return g.concatCols([scalar, g.reshape(coupled, [N, Mr], 'atom', 'charge')], 'charges');
    });

    const potential = g.scope('potential', () => {
      const periodic = isPeriodic(sys);
      const pairs: number[] = [], js: number[] = [];
      if (!periodic) for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) { pairs.push(i); js.push(j); }
      // Half of q_j v(r_ij) on i, and, for a half neighbour list, the same with i and j swapped.
      const real = (whoI: ArrayLike<number>, whoJ: ArrayLike<number>, bare: Tensor, halfList: boolean) => {
        if (!whoI.length) return g.scale(charges, 0, 'no_pairs');
        const onI = g.scale(g.segmentSum(g.mul(g.gather(charges, ix(whoJ, N, 'pair', 'atom'), 'q_j'), bare), ix(whoI, N, 'atom', 'pair'), 'on_i'), 0.5);
        if (!halfList) return onI;
        const onJ = g.scale(g.segmentSum(g.mul(g.gather(charges, ix(whoI, N, 'pair', 'atom'), 'q_i'), bare), ix(whoJ, N, 'atom', 'pair'), 'on_j'), 0.5);
        return g.add(onI, onJ);
      };
      let realSpace: Tensor;
      if (periodic) {
        const arg = g.scale(r, 1 / (m.smearing * Math.SQRT2), 'r/σ√2');
        const erfc = g.add(g.unary('neg', g.unary('erf', arg)), k([1], [1], 'one'), 'erfc');
        realSpace = real(center, neigh, g.div(erfc, r, 'erfc/r'), false);
      } else if (!pairs.length) realSpace = g.scale(charges, 0, 'no_pairs');
      else {
        const ij = ix(pairs, N, 'pair', 'atom'), ji = ix(js, N, 'pair', 'atom');
        const dist = g.rowNorm(g.sub(g.gather(positions, ji, 'r_j'), g.gather(positions, ij, 'r_i'), 'pair'), 0);
        realSpace = real(pairs, js, g.div(k(Array.from({ length: pairs.length }, () => 1), [pairs.length], 'one'), dist, '1/r'), true);
      }
      if (!periodic) return realSpace;
      const ks = ewaldK(sys.cell!, m.lr_wavelength), K = ks.length, vol = volume(sys);
      const kv = new Float32Array(3 * K), gV = new Float32Array(K);
      ks.forEach((q, t) => {
        for (let c = 0; c < 3; c++) kv[c * K + t] = q[c];
        const k2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2];
        gV[t] = k2 === 0 ? 0 : 4 * Math.PI * Math.exp(-0.5 * m.smearing * m.smearing * k2) / k2 / vol;
      });
      const phase = g.mm(positions, k(kv, [3, K], 'k'), false, false, 'k·r');
      const cos = g.unary('cos', phase), sin = g.unary('sin', phase);
      const Gw = k(gV, [K], 'G/V');
      const structure = g.add(g.mm(cos, g.mul(g.mm(cos, charges, true, false, 'S_cos'), Gw), false, false, 'cos'),
                              g.mm(sin, g.mul(g.mm(sin, charges, true, false, 'S_sin'), Gw), false, false, 'sin'), 'structure_factor');
      const self = Math.sqrt(2 / Math.PI) / m.smearing;
      const Q = g.mm(k(Array.from({ length: N }, () => 1), [1, N], 'ones'), charges, false, false, 'total_charge');
      const neutral = g.sub(g.sub(structure, g.scale(charges, self, 'self')), g.scale(Q, 2 * Math.PI * m.smearing * m.smearing / vol, 'background'));
      return g.add(realSpace, g.scale(neutral, 0.5, 'ewald'), 'potential');
    });

    const lr = g.scope('long_range', () => {
      const scalar = g.sliceCols(potential, 0, 1, 'scalar_potential');
      const sphPot = degreeLinear(g.reshape(g.sliceCols(potential, 1, Mr), [N * Mr, 1], 'sph'), 'lr.potential_to_features', N, Lr, Mr, s);
      const mixed = g.couple(sphPot, spherical, this.p('lr.potential_product.kernel'), Mr, M, M, 'potential_product');
      mixed.kind = 'sph';
      const updated = update(nodes, g.concatCols([scalar, degreeNorms(mixed, N, L, M, s)], 'lr_features'), 'lr.update2');
      return energyMlp(updated, 'lr.energy_mlp');
    });

    const perAtom = g.scope('energy', () => g.add(sr, lr, 'atomic_energy').withUnit(eV));
    const energy = g.scope('energy', () => g.sumAll(perAtom, 'total_energy').withUnit(eV));
    const rows: Record<string, RowSpace> = {
      atom: atomRows(N),
      edge: { label: 'edge', owner: center, edge: range(E) },
      sph: { label: 'atom, m component', owner: Int32Array.from({ length: N * M }, (_, i) => (i / M) | 0), atom: Int32Array.from({ length: N * M }, (_, i) => (i / M) | 0), all: true },
      'edge.sph': { label: 'edge, m component', owner: Int32Array.from({ length: E * M }, (_, i) => center[(i / M) | 0] ?? 0), edge: Int32Array.from({ length: E * M }, (_, i) => (i / M) | 0) },
    };
    const charge = g.reshape(g.sliceCols(charges, 0, 1), [N], 'atom', 'charge');
    const graph = { center, neighbor: neigh, shift: Float32Array.from(nl.shiftVec), label: 'neighbour' };
    return { energy, perAtom, positions, rows, graph, extras: { charge }, internals: { sr, lr, charges } };
  }
}
