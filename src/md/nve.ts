// Velocity-Verlet molecular dynamics at constant N, V, E, in eV, Angstrom, amu and fs.
import { MASSES } from '../common/elements';

export const KB = 8.617333262e-5; // eV / K
/** 1 eV / (A amu) in A / fs^2. */
export const ACC = 9.64853321e-3;

export type ForceFn = (x: Float64Array) => Promise<{ energy: number; forces: Float64Array }>;

/** Standard normal deviates (Box-Muller). */
const gauss = (rand: () => number) => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());

export class NVE {
  readonly n: number;
  readonly mass: Float64Array;
  readonly v: Float64Array; // A / fs
  forces: Float64Array = new Float64Array(0); // eV / A
  energy = 0; // potential, eV
  step = 0;
  time = 0; // fs (dt may change during a run)

  private constructor(readonly numbers: number[], readonly x: Float64Array, readonly force: ForceFn, public dt: number) {
    this.n = numbers.length;
    this.mass = Float64Array.from(numbers, (z) => MASSES[z] ?? 1);
    this.v = new Float64Array(3 * this.n);
  }

  static async create(numbers: number[], x: Float64Array, force: ForceFn, dt = 0.5) {
    const md = new NVE(numbers, Float64Array.from(x), force, dt);
    await md.evaluate();
    return md;
  }

  /** Degrees of freedom: the centre of mass does not move. */
  get dof() { return this.n > 1 ? 3 * this.n - 3 : 3; }
  get kinetic() {
    let k = 0;
    for (let i = 0; i < 3 * this.n; i++) k += this.mass[(i / 3) | 0] * this.v[i] ** 2;
    return (0.5 * k) / ACC;
  }
  get temperature() { return (2 * this.kinetic) / (this.dof * KB); }
  get total() { return this.energy + this.kinetic; }

  /** Maxwell-Boltzmann velocities at T, with no net momentum, scaled to exactly T. */
  thermalize(T: number, rand = Math.random) {
    const { n, v, mass } = this;
    for (let i = 0; i < 3 * n; i++) v[i] = gauss(rand) * Math.sqrt((KB * T * ACC) / mass[(i / 3) | 0]);
    const M = mass.reduce((a, b) => a + b, 0);
    for (let c = 0; c < 3; c++) {
      let p = 0;
      for (let i = 0; i < n; i++) p += mass[i] * v[3 * i + c];
      for (let i = 0; i < n; i++) v[3 * i + c] -= p / M;
    }
    const s = T > 0 && this.temperature > 0 ? Math.sqrt(T / this.temperature) : 0;
    for (let i = 0; i < 3 * n; i++) v[i] *= s;
  }

  private async evaluate() {
    const { energy, forces } = await this.force(this.x);
    this.energy = energy;
    this.forces = forces;
  }

  private kick() {
    for (let i = 0; i < 3 * this.n; i++) this.v[i] += (0.5 * this.dt * ACC * this.forces[i]) / this.mass[(i / 3) | 0];
  }

  /** `steps` velocity-Verlet steps. */
  async advance(steps = 1) {
    for (let s = 0; s < steps; s++) {
      this.kick();
      for (let i = 0; i < 3 * this.n; i++) this.x[i] += this.dt * this.v[i];
      await this.evaluate();
      this.kick();
      this.step++;
      this.time += this.dt;
    }
  }
}
