import { readdirSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { Graph } from '../src/engine/tensor';
import { loadANI } from './util';

describe('ANI-2x vs TorchANI', () => {
  const model = loadANI();
  for (const file of readdirSync('tests/reference').filter((f: string) => f.startsWith('ani-2x_'))) {
    const ref = JSON.parse(readFileSync(`tests/reference/${file}`, 'utf8'));
    it(ref.case, async () => {
      const g = new Graph(model.be);
      const out = model.forward(g, { numbers: ref.atomic_numbers, positions: ref.positions, cell: ref.cell, pbc: ref.pbc }, { forces: true });
      g.backward(out.energy);
      const E = (await model.be.read(out.energy.buf))[0];
      const F = Array.from(await model.be.read(out.positions.grad!), (x) => -x);
      const dE = Math.abs(E - ref.energy), dF = Math.max(...F.map((f, i) => Math.abs(f - ref.forces.flat()[i])));
      console.log(`ani-2x ${ref.case.padEnd(16)} E=${E.toFixed(5)} dE=${dE.toExponential(2)} dF=${dF.toExponential(2)}`);
      g.release();
      expect(dE).toBeLessThan(2e-6 * Math.abs(ref.energy));
      expect(dF).toBeLessThan(2e-3);
    });
  }
});
