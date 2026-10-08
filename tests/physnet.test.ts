// PhysNet (mmml physnetjax, max_degree = 0) against the JAX implementation.
import { readdirSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { Graph } from '../src/engine/tensor';
import { loadPhysNet } from './util';

describe('PhysNet vs physnetjax', () => {
  const model = loadPhysNet();
  for (const file of readdirSync('tests/reference').filter((f: string) => f.startsWith('physnet-test_'))) {
    const ref = JSON.parse(readFileSync(`tests/reference/${file}`, 'utf8'));
    it(ref.case, async () => {
      const g = new Graph(model.be);
      const out = model.forward(g, { numbers: ref.atomic_numbers, positions: ref.positions }, { forces: true });
      g.backward(out.energy);
      const E = (await model.be.read(out.energy.buf))[0];
      const F = Array.from(await model.be.read(out.positions.grad!), (x) => -x);
      g.release();
      const dE = Math.abs(E - ref.energy), dF = Math.max(...F.map((f, i) => Math.abs(f - ref.forces.flat()[i])));
      const Fmax = Math.max(...ref.forces.flat().map(Math.abs));
      process.stderr.write(`\nphysnet ${ref.case.padEnd(9)} E=${E.toFixed(5)} ref=${ref.energy.toFixed(5)} dE=${dE.toExponential(2)} dF=${dF.toExponential(2)} (max |F| ${Fmax.toFixed(1)})`);
      expect(dE).toBeLessThan(1e-4 * Math.max(1, Math.abs(ref.energy)));
      expect(dF).toBeLessThan(1e-4 * Math.max(1, Fmax));
    });
  }
});
