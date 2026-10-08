// DMC and FIRE on analytic surfaces: no weights needed.
import { describe, expect, it } from 'vitest';
import { align, BOHR, DMC, estimate, HARTREE } from '../src/md/dmc';
import { relax } from '../src/md/relax';
import { MASSES } from '../src/common/elements';
import { PRESETS } from '../src/app/presets';

let seed = 3;
const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);

describe('DMC', () => {
  it('finds the zero-point energy of a 3D harmonic oscillator', async () => {
    const k = 30; // eV / A^2, on one hydrogen atom: ZPE = 3/2 hbar omega
    const omega = Math.sqrt((k * BOHR ** 2) / HARTREE / (MASSES[1] * 1822.888486209)); // hartree
    const exact = 1.5 * omega * HARTREE;
    const energy = async (ws: Float64Array[]) => Float64Array.from(ws, (w) => 0.5 * k * (w[0] ** 2 + w[1] ** 2 + w[2] ** 2));
    const dmc = new DMC([1], new Float64Array(3), 0, energy, 400, 5, 0.5, rand);
    const eref: number[] = [];
    for (let s = 0; s < 3000; s++) eref.push((await dmc.advance()).eref);
    const { mean, err } = estimate(eref)!;
    expect(Math.abs(mean - exact) / exact).toBeLessThan(0.03);
    expect(err).toBeLessThan(0.02 * exact);
  });

  it('aligns a rotated, shifted copy back onto the reference', () => {
    const ref = Float64Array.from(PRESETS.ethanol.positions.flat());
    const t = 0.7, c = Math.cos(t), s = Math.sin(t);
    const x = Float64Array.from(ref);
    for (let i = 0; i < x.length; i += 3) {
      const [a, b] = [ref[i] * c - ref[i + 2] * s, ref[i] * s + ref[i + 2] * c];
      [x[i], x[i + 1], x[i + 2]] = [a + 3, ref[i + 1] - 1, b + 2];
    }
    const y = align(x, ref);
    for (let i = 0; i < y.length; i++) expect(y[i]).toBeCloseTo(ref[i], 5);
  });
});

describe('FIRE', () => {
  it('relaxes a Morse dimer to its bond length', async () => {
    const r0 = 1.1, D = 5, a = 2;
    const force = async (x: Float64Array) => {
      const d = [x[3] - x[0], x[4] - x[1], x[5] - x[2]], r = Math.hypot(...d), q = Math.exp(-a * (r - r0));
      const dEdr = 2 * D * a * q * (1 - q), f = new Float64Array(6);
      for (let c = 0; c < 3; c++) { f[c] = (dEdr * d[c]) / r; f[3 + c] = (-dEdr * d[c]) / r; }
      return { energy: D * (1 - q) ** 2, forces: f };
    };
    const { x, fmax } = await relax(Float64Array.from([0, 0, 0, 1.6, 0.2, 0]), force, { fmax: 1e-4 });
    expect(fmax).toBeLessThan(1e-4);
    expect(Math.hypot(x[3] - x[0], x[4] - x[1], x[5] - x[2])).toBeCloseTo(r0, 4);
  });
});
