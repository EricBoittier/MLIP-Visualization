// Batched energies (many walkers in one pass) against one pass per configuration.
import { describe, expect, it } from 'vitest';
import { CpuBackend } from '../src/engine/cpu';
import { Graph } from '../src/engine/tensor';
import type { Model } from '../src/models/types';
import { PRESETS } from '../src/app/presets';
import { batchEnergies } from '../src/md/batch';
import { loadANI, loadMACE, loadModel } from './util';

const be = new CpuBackend();
const single = async (m: Model, numbers: number[], x: Float64Array) => {
  const g = new Graph(be);
  try {
    const positions = Array.from({ length: x.length / 3 }, (_, i) => [x[3 * i], x[3 * i + 1], x[3 * i + 2]]);
    return (await be.read(m.forward(g, { numbers, positions }, { forces: false }).energy.buf))[0];
  } finally { g.release(); }
};

let seed = 5;
const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);

describe.each([
  ['PET-MAD XS', () => loadModel('pet-mad-xs', be)],
  ['ANI-2x', () => loadANI(be)],
  ['MACE-MP-0b2 small', () => loadMACE('mace-mp-0b2-small', be)],
])('%s', (_name, load) => {
  it('gives every walker the energy of its own pass', async () => {
    const m = load(), s = PRESETS.ethanol, x0 = s.positions.flat();
    const walkers = Array.from({ length: 5 }, () => Float64Array.from(x0, (v) => v + 0.08 * (rand() - 0.5)));
    const batched = await batchEnergies(m, be, s, walkers, 30); // 3 walkers per pass: two passes
    for (let k = 0; k < walkers.length; k++) expect(batched[k]).toBeCloseTo(await single(m, s.numbers, walkers[k]), 3);
  });
});
