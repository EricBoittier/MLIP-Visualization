// KRR/SOAP fitted to ANI-2x on rattled ethanol: fit quality, invariance, and forces
// against finite differences of the KRR energy.
import { describe, expect, it } from 'vitest';
import { Graph } from '../src/engine/tensor';
import { KRR } from '../src/models/krr/model';
import { PRESETS } from '../src/app/presets';
import { loadANI } from './util';

describe('KRR / SOAP', async () => {
  const teacher = loadANI();
  const krr = new KRR(teacher.be, { kind: 'krr' });
  const sys = PRESETS.ethanol;
  const logs: string[] = [];
  await krr.fit(sys, teacher, 'ANI-2x', (t) => logs.push(t));
  const energy = async (positions: number[][], forces = false) => {
    const g = new Graph(krr.be);
    const out = krr.forward(g, { ...sys, positions }, { forces });
    if (forces) g.backward(out.energy);
    const E = (await krr.be.read(out.energy.buf))[0];
    const F = forces ? Array.from(await krr.be.read(out.positions.grad!), (x) => -x) : [];
    g.release();
    return { E, F };
  };

  it("fits the teacher to about 10 meV per atom", () => {
    const r = krr.report!;
    process.stderr.write(`\nKRR: ${r.nSparse} sparse points, lambda_rel ${r.lambdaRelative}, train ${(1000 * r.rmseTrain).toFixed(2)} / held-out ${(1000 * r.rmseTest).toFixed(2)} meV/atom, ${r.ms.toFixed(0)} ms\n`);
    expect(r.rmseTest).toBeLessThan(0.02);
  });

  it('is invariant to rotations', async () => {
    const c = Math.cos(1.1), s = Math.sin(1.1);
    const rot = sys.positions.map(([x, y, z]) => [c * x - s * z, y, s * x + c * z]);
    const [a, b] = [await energy(sys.positions), await energy(rot)];
    // a few float32 ulps of the total energy
    expect(Math.abs(a.E - b.E)).toBeLessThan(4e-7 * Math.abs(a.E));
  });

  it('gives forces equal to -dE/dr', async () => {
    // difference the energy without the constant per-atom baseline, or float32 drowns it
    (krr as any).baseline = new Map((krr as any).baseline.keys().map((z: number) => [z, 0]));
    const base = sys.positions.map((p, i) => p.map((x, k) => x + 0.03 * Math.sin(3 * i + k)));
    const { F } = await energy(base, true);
    const h = 2e-3;
    for (const k of [0, 4, 7, 13, 20, 26]) {
      const p = base.map((v) => v.slice()), m = base.map((v) => v.slice());
      p[(k / 3) | 0][k % 3] += h; m[(k / 3) | 0][k % 3] -= h;
      const fd = -((await energy(p)).E - (await energy(m)).E) / (2 * h);
      expect(Math.abs(fd - F[k])).toBeLessThan(0.02 + 0.02 * Math.abs(fd));
    }
  });
});
