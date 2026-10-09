// Diffusion Monte Carlo for the vibrational ground state of a potential energy surface: unguided
// (Anderson) DMC with discrete branching, in atomic units inside and eV / Angstrom outside. Walkers
// diffuse with variance dtau / m per coordinate, then each is copied or killed with weight
// exp(-dtau (V - E_ref)); E_ref is steered to keep the population near its target and averages to
// the ground-state energy. Only energies are needed, so the walkers are evaluated in batches.
import { MASSES } from '../common/elements';

export const HARTREE = 27.211386245988; // eV
export const BOHR = 0.529177210903; // Angstrom
export const AMU = 1822.888486209; // electron masses
export const CM = 8065.543937; // cm^-1 per eV
export const AU_FS = 0.02418884326585747; // fs per atomic unit of time

/** Energies (eV) of many configurations (each [3N], Angstrom) at once. */
export type EnergyFn = (walkers: Float64Array[]) => Promise<Float64Array>;

export interface DMCStep { eref: number; vmean: number; n: number }

const gauss = (rand: () => number) => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());

export class DMC {
  readonly n: number; // atoms
  readonly sigma: Float64Array; // diffusion step per coordinate, Angstrom
  x: Float64Array[];
  v: Float64Array;
  eref: number;
  step = 0;
  holes = 0; // walkers removed for falling far below the minimum
  /** Walkers evaluated so far. */
  samples = 0;

  /** `vmin`: the potential at `x0`, the relaxed structure every walker starts from (eV). */
  constructor(readonly numbers: number[], readonly x0: Float64Array, readonly vmin: number, readonly energy: EnergyFn,
              readonly target: number, public dtau = 10, readonly holeBelow = 0.5, readonly rand = Math.random) {
    this.n = numbers.length;
    this.sigma = Float64Array.from({ length: 3 * this.n }, (_, i) => BOHR * Math.sqrt(1 / ((MASSES[numbers[(i / 3) | 0]] ?? 1) * AMU)));
    this.x = Array.from({ length: target }, () => Float64Array.from(x0));
    this.v = new Float64Array(target).fill(vmin);
    this.eref = vmin;
  }

  get tau() { return this.step * this.dtau; }

  /** One step: diffuse, evaluate, branch, steer E_ref. */
  async advance(): Promise<DMCStep> {
    const s = Math.sqrt(this.dtau), r = this.rand;
    for (const w of this.x) for (let i = 0; i < w.length; i++) w[i] += s * this.sigma[i] * gauss(r);
    const v = await this.energy(this.x);
    this.samples += this.x.length;
    const nx: Float64Array[] = [], nv: number[] = [];
    for (let k = 0; k < this.x.length; k++) {
      if (!(v[k] > this.vmin - this.holeBelow)) { this.holes++; continue; } // a hole in the surface (or NaN)
      const copies = Math.min(Math.floor(Math.exp((-this.dtau * (v[k] - this.eref)) / HARTREE) + r()), 3);
      for (let c = 0; c < copies; c++) { nx.push(c ? Float64Array.from(this.x[k]) : this.x[k]); nv.push(v[k]); }
    }
    if (!nx.length) throw new Error('every walker died: try a smaller time step');
    this.x = nx;
    this.v = Float64Array.from(nv);
    const vmean = nv.reduce((a, b) => a + b, 0) / nv.length;
    // E_ref = <V> - alpha (N - N0) / N0, alpha = 1 / (2 dtau) hartree
    this.eref = vmean - ((HARTREE / (2 * this.dtau)) * (nv.length - this.target)) / this.target;
    this.step++;
    return { eref: this.eref, vmean, n: nv.length };
  }
}

/** Mean of the second half of a series, with a blocking error estimate (10 blocks). */
export function estimate(series: ArrayLike<number>): { mean: number; err: number } | null {
  const h = Math.floor(series.length / 2), m = series.length - h;
  if (m < 50) return null;
  const B = 10, size = Math.floor(m / B), blocks: number[] = [];
  for (let b = 0; b < B; b++) {
    let s = 0;
    for (let i = 0; i < size; i++) s += series[h + b * size + i];
    blocks.push(s / size);
  }
  let mean = 0;
  for (let i = h; i < series.length; i++) mean += series[i];
  mean /= m;
  const bm = blocks.reduce((a, b) => a + b, 0) / B;
  const err = Math.sqrt(blocks.reduce((a, b) => a + (b - bm) ** 2, 0) / (B * (B - 1)));
  return { mean, err };
}

/** x (n x 3, about its centroid) rotated onto ref (centred), by Horn's quaternion method. */
export function align(x: ArrayLike<number>, ref: ArrayLike<number>): Float64Array {
  const n = x.length / 3, cx = [0, 0, 0], cr = [0, 0, 0];
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) { cx[c] += x[3 * i + c] / n; cr[c] += ref[3 * i + c] / n; }
  const S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < n; i++)
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) S[a][b] += (x[3 * i + a] - cx[a]) * (ref[3 * i + b] - cr[b]);
  const [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = S;
  const N = [
    [xx + yy + zz, yz - zy, zx - xz, xy - yx],
    [yz - zy, xx - yy - zz, xy + yx, zx + xz],
    [zx - xz, xy + yx, -xx + yy - zz, yz + zy],
    [xy - yx, zx + xz, yz + zy, -xx - yy + zz],
  ];
  // the largest eigenvector, by power iteration on N + c I (all eigenvalues made positive)
  const c = Math.sqrt(N.flat().reduce((a, b) => a + b * b, 0)) + 1e-12;
  let q = [1, 0, 0, 0];
  for (let it = 0; it < 200; it++) {
    const nq = N.map((row, i) => row.reduce((a, v, j) => a + v * q[j], 0) + c * q[i]);
    const len = Math.hypot(...nq);
    q = nq.map((v) => v / len);
  }
  const [w, a, b, d] = q;
  const R = [
    [w * w + a * a - b * b - d * d, 2 * (a * b - w * d), 2 * (a * d + w * b)],
    [2 * (a * b + w * d), w * w - a * a + b * b - d * d, 2 * (b * d - w * a)],
    [2 * (a * d - w * b), 2 * (b * d + w * a), w * w - a * a - b * b + d * d],
  ];
  const out = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) {
    const p = [0, 1, 2].map((k) => x[3 * i + k] - cx[k]);
    for (let r = 0; r < 3; r++) out[3 * i + r] = cr[r] + R[r][0] * p[0] + R[r][1] * p[1] + R[r][2] * p[2];
  }
  return out;
}
