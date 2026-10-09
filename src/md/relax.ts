// FIRE geometry optimisation (Bitzek et al. 2006, with ASE's defaults and unit masses).
import type { ForceFn } from './nve';

export interface Relaxed { x: Float64Array; energy: number; fmax: number; steps: number }

export async function relax(x0: Float64Array, force: ForceFn, { fmax = 0.01, maxSteps = 500, maxStep = 0.2,
  onStep = (_step: number, _fmax: number, _energy: number) => {} } = {}): Promise<Relaxed> {
  const x = Float64Array.from(x0), v = new Float64Array(x.length);
  let dt = 0.1, a = 0.1, nPos = 0, { energy, forces } = await force(x);
  const maxF = (f: Float64Array) => {
    let m = 0;
    for (let i = 0; i < f.length; i += 3) m = Math.max(m, Math.hypot(f[i], f[i + 1], f[i + 2]));
    return m;
  };
  for (let step = 0; step < maxSteps; step++) {
    const fm = maxF(forces);
    onStep(step, fm, energy);
    if (fm < fmax) return { x, energy, fmax: fm, steps: step };
    let vf = 0, vv = 0, ff = 0;
    for (let i = 0; i < x.length; i++) { vf += v[i] * forces[i]; vv += v[i] ** 2; ff += forces[i] ** 2; }
    if (vf > 0) {
      const s = Math.sqrt(vv / ff);
      for (let i = 0; i < x.length; i++) v[i] = (1 - a) * v[i] + a * s * forces[i];
      if (++nPos > 5) { dt = Math.min(dt * 1.1, 1); a *= 0.99; }
    } else {
      v.fill(0);
      dt *= 0.5; a = 0.1; nPos = 0;
    }
    for (let i = 0; i < x.length; i++) v[i] += dt * forces[i];
    let dmax = 0;
    for (let i = 0; i < x.length; i += 3) dmax = Math.max(dmax, dt * Math.hypot(v[i], v[i + 1], v[i + 2]));
    const k = dmax > maxStep ? maxStep / dmax : 1;
    for (let i = 0; i < x.length; i++) x[i] += k * dt * v[i];
    ({ energy, forces } = await force(x));
  }
  return { x, energy, fmax: maxF(forces), steps: maxSteps };
}
