// Kernel ridge regression on SOAP power spectra, fitted in the browser.
//
// Training set: rattled copies of a structure, labelled by another model (the
// "teacher", e.g. PET-MAD). Atomic energies are a kernel expansion over M sparse
// environments chosen by farthest-point sampling:
//   eps_i = b_Z + sum_t alpha_t k(p_i, x_t),  k(p, x) = (p . x)^zeta  (same element only),
// and alpha solves the regularised least squares problem of sparse GPR / SOR:
//   (K_NM^T K_NM + lambda K_MM) alpha = K_NM^T (y - b),
// where K_NM sums the atomic kernels of each training structure.
import type { Backend } from '../../engine/backend';
import { Graph, type Tensor } from '../../engine/tensor';
import { eV, Å } from '../../engine/units';
import { neighborList, type System } from '../../common/structure';
import { atomRows, edgeRows, type Forward, type Model, range, type RowSpace } from '../types';
import { defaultSoap, radialTables, type SoapHypers } from './soap';

export interface KrrMeta {
  kind: 'krr';
  soap?: Partial<SoapHypers>;
  n_train?: number;
  n_test?: number;
  sparse_per_element?: number;
  rattle?: [number, number]; // range of displacement amplitudes, A
  regularization?: number; // relative to the scale of K_NM^T K_NM; default: picked on the held-out set
  teacher?: string; // id of the labelling model
  system?: System; // the structure to rattle
}

export interface FitReport {
  teacher: string;
  nTrain: number;
  nTest: number;
  nSparse: number;
  rmseTrain: number; // eV / atom
  rmseTest: number;
  lambda: number;
  lambdaRelative: number;
  ms: number;
}

type Log = (text: string, fraction?: number) => void;

export class KRR implements Model {
  readonly kind = 'krr' as const;
  readonly params = new Map<string, Tensor>();
  readonly h: SoapHypers;
  elements: number[] = [];
  report: FitReport | null = null;
  private tables: ReturnType<typeof radialTables>;
  private V!: Tensor;
  private D!: Tensor;
  // the fit
  private sparse = new Float32Array(0); // [M, F]
  private sparseZ = new Int32Array(0);
  private trainX = new Float32Array(0); // [atoms of all training structures, F]
  private trainZ = new Int32Array(0);
  private trainOf = new Int32Array(0); // structure of each training atom
  private trainY = new Float32Array(0); // energies minus baselines
  private alpha = new Float32Array(0);
  private lambda = 0;
  private baseline = new Map<number, number>();
  private F = 0;

  constructor(readonly be: Backend, readonly meta: KrrMeta) {
    this.h = { ...defaultSoap, ...meta.soap };
    this.tables = radialTables(this.h);
  }

  /** Label rattled copies of `sys` with `teacher`, describe them, pick sparse points and solve. */
  async fit(sys: System, teacher: Model, teacherName: string, log: Log) {
    const t0 = performance.now(), m = this.meta;
    const nTrain = m.n_train ?? 96, nTest = m.n_test ?? 24, [lo, hi] = m.rattle ?? [0.01, 0.08];
    this.elements = [...new Set(sys.numbers)].sort((a, b) => a - b);
    const bad = this.elements.filter((z) => !teacher.elements.includes(z));
    if (bad.length) throw new Error(`the teacher (${teacherName}) has no parameters for Z = ${bad.join(', ')}`);
    let seed = 12345;
    const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
    const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
    const structures = Array.from({ length: nTrain + nTest }, (_, s) => {
      const amp = s === 0 ? 0 : lo + (hi - lo) * rnd();
      return { ...sys, positions: sys.positions.map((p) => p.map((x) => x + amp * gauss())) };
    });
    // labels
    const energies: number[] = [];
    for (let s = 0; s < structures.length; s++) {
      if (s % 4 === 0) log(`KRR: labelling structure ${s + 1}/${structures.length} with ${teacherName}…`, 0.75 * (s / structures.length));
      const g = new Graph(this.be);
      const out = teacher.forward(g, structures[s], { forces: false });
      energies.push((await this.be.read(out.energy.buf))[0]);
      g.release();
    }
    // descriptors
    log('KRR: SOAP power spectra of the training set…', 0.75);
    const desc: Float32Array[] = [];
    for (const st of structures) {
      const g = new Graph(this.be);
      const p = this.describe(g, st, false).p;
      desc.push(await this.be.read(p.buf));
      this.F = p.cols;
      g.release();
    }
    const N = sys.numbers.length, F = this.F, Ztr = Int32Array.from(sys.numbers);
    // baseline: mean energy per atom (one composition, so per element baselines are not identifiable)
    const mean = energies.slice(0, nTrain).reduce((a, b) => a + b, 0) / nTrain / N;
    this.baseline = new Map(this.elements.map((z) => [z, mean]));
    // sparse points: farthest-point sampling per element over the training environments
    const per = m.sparse_per_element ?? 24;
    const sparse: number[] = [], sparseZ: number[] = [];
    for (const z of this.elements) {
      const pool: Float32Array[] = [];
      for (let s = 0; s < nTrain; s++) for (let i = 0; i < N; i++) if (Ztr[i] === z) pool.push(desc[s].subarray(i * F, (i + 1) * F));
      const dist = new Float64Array(pool.length).fill(Infinity);
      let pick = 0;
      for (let k = 0; k < Math.min(per, pool.length); k++) {
        sparse.push(...pool[pick]); sparseZ.push(z);
        let best = -1;
        for (let q = 0; q < pool.length; q++) {
          let d = 0;
          for (let f = 0; f < F; f++) d += (pool[q][f] - pool[pick][f]) ** 2;
          dist[q] = Math.min(dist[q], d);
          if (best < 0 || dist[q] > dist[best]) best = q;
        }
        pick = best;
      }
    }
    this.sparse = Float32Array.from(sparse);
    this.sparseZ = Int32Array.from(sparseZ);
    const M = sparseZ.length;
    log(`KRR: kernels and the solve for ${M} sparse environments…`, 0.88);
    // kernels (float64)
    const kern = (a: Float32Array, b: Float32Array) => { let d = 0; for (let f = 0; f < F; f++) d += a[f] * b[f]; return d ** this.h.zeta; };
    const sp = (t: number) => this.sparse.subarray(t * F, (t + 1) * F);
    const Knm = (s: number) => Float64Array.from({ length: M }, (_, t) => {
      let v = 0;
      for (let i = 0; i < N; i++) if (Ztr[i] === sparseZ[t]) v += kern(desc[s].subarray(i * F, (i + 1) * F), sp(t));
      return v;
    });
    const KNM = Array.from({ length: nTrain + nTest }, (_, s) => Knm(s));
    const KMM = Array.from({ length: M }, (_, a) => Float64Array.from({ length: M }, (_, b) => (sparseZ[a] === sparseZ[b] ? kern(sp(a), sp(b)) : 0)));
    const y = energies.map((e) => e - N * mean);
    const A = Array.from({ length: M }, (_, a) => Float64Array.from({ length: M }, (_, b) => {
      let v = 0;
      for (let s = 0; s < nTrain; s++) v += KNM[s][a] * KNM[s][b];
      return v;
    }));
    // lambda by hold-out: the last n_test structures pick it from a grid
    const trA = A.reduce((acc, r, i) => acc + r[i], 0) / M, trK = KMM.reduce((acc, r, i) => acc + r[i], 0) / M;
    const rhs = Float64Array.from({ length: M }, (_, a) => { let v = 0; for (let s = 0; s < nTrain; s++) v += KNM[s][a] * y[s]; return v; });
    const solveFor = (lam: number) => choleskySolve(A.map((r, a) => r.map((v, b) => v + lam * KMM[a][b] + (a === b ? 1e-10 * trA : 0))), rhs);
    const rmseOf = (alpha: Float64Array, from: number, to: number) =>
      Math.sqrt(range(to - from, from).reduce((acc, s) => acc + ((KNM[s].reduce((q, k, t) => q + k * alpha[t], 0) - y[s]) / N) ** 2, 0) / (to - from));
    const grid = m.regularization != null ? [m.regularization] : [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1];
    let best: { rel: number; alpha: Float64Array; err: number } = { rel: grid[0], alpha: new Float64Array(M), err: Infinity };
    for (const rel of grid) {
      try {
        const alpha = solveFor(rel * (trA / trK));
        const err = rmseOf(alpha, nTrain, nTrain + nTest);
        if (err < best.err) best = { rel, alpha, err };
      } catch { /* not positive definite at this lambda */ }
    }
    this.lambda = best.rel * (trA / trK);
    const alpha = best.alpha;
    this.alpha = Float32Array.from(alpha);
    const rmse = (from: number, to: number) => rmseOf(alpha, from, to);
    // what the pass replays: the training atoms, and their structures
    this.trainX = new Float32Array(nTrain * N * F);
    for (let s = 0; s < nTrain; s++) this.trainX.set(desc[s], s * N * F);
    this.trainZ = Int32Array.from({ length: nTrain * N }, (_, k) => Ztr[k % N]);
    this.trainOf = Int32Array.from({ length: nTrain * N }, (_, k) => (k / N) | 0);
    this.trainY = Float32Array.from(y.slice(0, nTrain));
    this.report = { teacher: teacherName, nTrain, nTest, nSparse: M, rmseTrain: rmse(0, nTrain), rmseTest: rmse(nTrain, nTrain + nTest),
                    lambda: this.lambda, lambdaRelative: best.rel, ms: performance.now() - t0 };
    this.params.clear();
    this.params.set('sparse_points', this.constParam('sparse_points', this.sparse, [M, F]));
    this.params.set('alpha', this.constParam('alpha', this.alpha, [1, M]));
    log(`KRR: fitted ${M} sparse environments on ${nTrain} structures; test RMSE ${(1000 * this.report.rmseTest).toFixed(2)} meV/atom`, 1);
  }

  private constParam(name: string, data: Float32Array, shape: number[]) {
    const g = new Graph(this.be); // only to make a tensor; the buffer is kept
    const t = g.tensor(shape, this.be.upload(data), name);
    t.name = name;
    return t;
  }

  /** SOAP power spectra of one structure (normalised), and its neighbour graph. */
  describe(g: Graph, sys: System, forces: boolean) {
    const h = this.h, N = sys.numbers.length, L = h.l_max, Mlm = (L + 1) ** 2, nmax = h.n_max;
    const Zs = this.elements.length ? this.elements : [...new Set(sys.numbers)].sort((a, b) => a - b), S = Zs.length;
    const spec = Int32Array.from(sys.numbers, (z) => Zs.indexOf(z));
    const nl = neighborList(sys, h.cutoff);
    const P = nl.center.length;
    const ix = (a: ArrayLike<number>, nSrc: number, out?: string, src?: string) => g.index(Int32Array.from(a), nSrc, { out, src });
    const c = (data: ArrayLike<number>, shape: number[], name: string, kind?: string) => g.constant(Float32Array.from(data), shape, name, kind);
    this.V ??= new Graph(this.be).tensor([this.tables.C, this.tables.K], this.be.upload(this.tables.V), 'radial_integrals');
    this.D ??= new Graph(this.be).tensor([this.tables.C, this.tables.K], this.be.upload(this.tables.D), 'radial_slopes');

    const positions = c(sys.positions.flat(), [N, 3], 'positions', 'atom').withUnit(Å);
    positions.requiresGrad = forces;
    const { v, d } = g.scope('geometry', () => {
      const v = g.add(g.sub(g.gather(positions, ix(nl.neighbor, N, 'pair', 'atom'), 'r_j'), g.gather(positions, ix(nl.center, N, 'pair', 'atom'), 'r_i')),
                      c(nl.shiftVec, [P, 3], 'shift', 'pair'));
      return { v, d: g.rowNorm(v) };
    });
    const radial = g.scope('radial', () => {
      const fc = g.cutoff('cosine', d, c(new Array(P).fill(h.cutoff), [P], 'cutoff', 'pair'), h.cutoff_width);
      return g.mul(g.spline(d, this.V, this.D, this.tables.h, 'radial_integrals'), fc);
    });
    const Y = g.scope('angular', () => g.sph(g.div(v, d), L));
    const density = g.scope('density', () => {
      // c[(i, Z_j, n), lm] = sum_j fc I_nl(r_ij) Y_lm(r_ij / r_ij)
      const eR = new Float32Array(nmax * Mlm * nmax * (L + 1)), eY = new Float32Array(nmax * Mlm * Mlm);
      for (let n = 0; n < nmax; n++) for (let l = 0; l <= L; l++) for (let q = l * l; q < (l + 1) ** 2; q++) {
        eR[(n * Mlm + q) * nmax * (L + 1) + n * (L + 1) + l] = 1;
        eY[(n * Mlm + q) * Mlm + q] = 1;
      }
      const terms = g.mul(g.linear(radial, c(eR, [nmax * Mlm, nmax * (L + 1)], 'expand'), undefined, 'per_lm'),
                          g.linear(Y, c(eY, [nmax * Mlm, Mlm], 'expand'), undefined, 'per_n'));
      const slots = ix(Array.from(nl.center, (i, p) => i * S + spec[nl.neighbor[p]]), N * S, 'pair', 'atom_element');
      return g.reshape(g.segmentSum(terms, slots, 'sum_per_element'), [N * S * nmax, Mlm], 'atom_channel', 'coefficients');
    });
    const p = g.scope('power_spectrum', () => {
      const raw = g.power(density, N, S * nmax, L);
      raw.kind = 'atom';
      return g.div(raw, g.rowNorm(raw), 'normalise');
    });
    const rows: Record<string, RowSpace> = {
      atom: atomRows(N), pair: { ...edgeRows(nl.center), label: 'pair' },
      atom_element: { label: 'neighbour element', owner: Int32Array.from({ length: N * S }, (_, r) => (r / S) | 0) },
      atom_channel: { label: 'element / radial channel', owner: Int32Array.from({ length: N * S * nmax }, (_, r) => (r / (S * nmax)) | 0) },
    };
    return { positions, v, p, rows, spec, graph: { center: nl.center, neighbor: nl.neighbor, shift: Float32Array.from(nl.shiftVec), label: 'SOAP neighbour' } };
  }

  private pow(g: Graph, x: Tensor) {
    const z = this.h.zeta;
    return z === 1 ? x : z === 2 ? g.unary('square', x) : z === 4 ? g.unary('square', g.unary('square', x)) : g.unary('pow', x, z);
  }

  forward(g: Graph, sys: System, opts: { forces?: boolean } = {}): Forward {
    if (!this.report) throw new Error('the KRR model has not been fitted');
    const N = sys.numbers.length, F = this.F, M = this.sparseZ.length;
    const c = (data: ArrayLike<number>, shape: number[], name: string, kind?: string) => g.constant(Float32Array.from(data), shape, name, kind);
    const Xs = this.params.get('sparse_points')!;
    const mask = (Z: ArrayLike<number>) => c(Array.from({ length: Z.length * M }, (_, k) => (Z[(k / M) | 0] === this.sparseZ[k % M] ? 1 : 0)), [Z.length, M], 'same_element');

    // ---- the fit, replayed: K_MM, K_NM, the normal equations and their solution
    const alpha = g.scope('fit', () => {
      const Kmm = g.mul(this.pow(g, g.linear(Xs, Xs, undefined, 'sparse_overlap')), mask(this.sparseZ), 'K_MM');
      const Xtr = c(this.trainX, [this.trainZ.length, F], 'training_environments');
      const katoms = g.mul(this.pow(g, g.linear(Xtr, Xs, undefined, 'overlap')), mask(this.trainZ), 'atomic_kernels');
      const Knm = g.segmentSum(katoms, g.index(this.trainOf, this.trainY.length), 'K_NM');
      const KnmT = g.transpose(Knm);
      const normal = g.add(g.linear(KnmT, KnmT, undefined, 'K_NM^T K_NM'), g.scale(Kmm, this.lambda, 'lambda_K_MM'), 'normal_matrix');
      const rhs = g.linear(KnmT, c(this.trainY, [1, this.trainY.length], 'energies'), undefined, 'K_NM^T y');
      return g.opaque('cholesky_solve', [normal, rhs], this.alpha, [1, M]);
    });

    const d = g.scope('soap', () => this.describe(g, sys, !!opts.forces));
    const k = g.scope('kernel', () => g.mul(this.pow(g, g.linear(d.p, Xs, undefined, 'overlap')), mask(sys.numbers), 'kernel'));
    const perAtom = g.scope('regression', () => {
      const eps = g.sumRows(g.linear(k, alpha, undefined, 'weighted_sum')).withUnit(eV);
      return g.add(eps, c(sys.numbers.map((z) => this.baseline.get(z) ?? 0), [N], 'baseline', 'atom'), 'atomic_energy');
    });
    const energy = g.scope('energy', () => g.sumAll(perAtom, 'total_energy'));
    return { energy, perAtom, positions: d.positions, virialVectors: d.v, rows: d.rows, graph: d.graph };
  }
}

/** Solve A x = b for symmetric positive definite A (float64). */
export function choleskySolve(A: Float64Array[], b: Float64Array): Float64Array {
  const n = b.length, L = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let v = A[i][j];
    for (let k = 0; k < j; k++) v -= L[i][k] * L[j][k];
    if (i === j) {
      if (v <= 0) throw new Error('KRR: the normal matrix is not positive definite; raise the regularisation');
      L[i][i] = Math.sqrt(v);
    } else L[i][j] = v / L[j][j];
  }
  const z = new Float64Array(n);
  for (let i = 0; i < n; i++) { let v = b[i]; for (let k = 0; k < i; k++) v -= L[i][k] * z[k]; z[i] = v / L[i][i]; }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) { let v = z[i]; for (let k = i + 1; k < n; k++) v -= L[k][i] * x[k]; x[i] = v / L[i][i]; }
  return x;
}
