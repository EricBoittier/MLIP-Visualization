// NVE on an analytic potential: no weights needed.
import { describe, expect, it } from 'vitest';
import { NVE, type ForceFn } from '../src/md/nve';
import { PRESETS } from '../src/app/presets';

/** Pairwise Morse, all pairs: E = sum D (1 - exp(-a (r - r0)))^2. */
const morse = (D = 4, a = 2, r0 = 1.2): ForceFn => async (x) => {
  const n = x.length / 3, f = new Float64Array(x.length);
  let e = 0;
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      const d = [0, 1, 2].map((c) => x[3 * j + c] - x[3 * i + c]), r = Math.hypot(...d), q = Math.exp(-a * (r - r0));
      e += D * (1 - q) ** 2;
      const dEdr = 2 * D * a * q * (1 - q);
      for (let c = 0; c < 3; c++) { f[3 * i + c] += (dEdr * d[c]) / r; f[3 * j + c] -= (dEdr * d[c]) / r; }
    }
  return { energy: e, forces: f };
};

let seed = 1;
const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);

describe('NVE', () => {
  it('starts at the requested temperature with no net momentum', async () => {
    const s = PRESETS.ethanol;
    const md = await NVE.create(s.numbers, Float64Array.from(s.positions.flat()), morse());
    md.thermalize(300, rand);
    expect(md.temperature).toBeCloseTo(300, 6);
    for (let c = 0; c < 3; c++) {
      let p = 0;
      for (let i = 0; i < md.n; i++) p += md.mass[i] * md.v[3 * i + c];
      expect(Math.abs(p)).toBeLessThan(1e-12);
    }
  });

  it('conserves energy, with an error that falls as dt^2', async () => {
    const s = PRESETS.water;
    const drift = async (dt: number) => {
      seed = 7;
      const md = await NVE.create(s.numbers, Float64Array.from(s.positions.flat()), morse(), dt);
      md.thermalize(500, rand);
      const e0 = md.total;
      let worst = 0;
      for (let k = 0; k < Math.round(200 / dt); k++) { await md.advance(); worst = Math.max(worst, Math.abs(md.total - e0)); }
      return worst;
    };
    // water starts ~3.7 eV up this stiff potential, so the run is hot
    const [coarse, fine] = [await drift(0.2), await drift(0.1)];
    expect(coarse).toBeLessThan(0.05);
    expect(coarse / fine).toBeGreaterThan(3); // ~4 for a second-order integrator
  });
});
