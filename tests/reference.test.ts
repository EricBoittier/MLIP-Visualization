import { readFileSync, readdirSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { evaluate, type PET } from '../src/models/pet/model';
import { loadModel } from './util';

const refs = readdirSync("tests/reference").filter((f: string) => f.endsWith('.json'));

const maxAbs = (a: ArrayLike<number>, b: ArrayLike<number>) =>
  Array.from(a).reduce((m, x, i) => Math.max(m, Math.abs(x - b[i])), 0);

describe('CPU engine vs metatrain', () => {
  const cache = new Map<string, PET>();
  for (const file of refs) {
    const ref = JSON.parse(readFileSync(`tests/reference/${file}`, 'utf8'));
    it(`${ref.model} ${ref.case}`, async () => {
      if (!cache.has(ref.model)) cache.set(ref.model, loadModel(ref.model));
      const res = await evaluate(cache.get(ref.model)!, {
        numbers: ref.atomic_numbers, positions: ref.positions, cell: ref.cell, pbc: ref.pbc,
      }, { forces: true });
      const dE = Math.abs(res.energy - ref.energy), dF = maxAbs(res.forces!, ref.forces.flat());
      const dS = ref.stress ? maxAbs(res.stress!.flat(), ref.stress.flat()) : 0;
      console.log(`${ref.model.padEnd(16)} ${ref.case.padEnd(8)} dE=${dE.toExponential(2)} dF=${dF.toExponential(2)} dS=${dS.toExponential(2)}`);
      expect(dE).toBeLessThan(2e-6 * Math.max(1, Math.abs(ref.energy)));
      expect(maxAbs(res.energies, ref.energies)).toBeLessThan(1e-4 * Math.max(1, Math.abs(ref.energy) / ref.energies.length));
      expect(dF).toBeLessThan(2e-4);
      expect(dS).toBeLessThan(2e-5);
    });
  }
});
