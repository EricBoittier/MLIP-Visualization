// MACE against mace-torch (float64, scripts/reference_mace.py): the spherical harmonics and
// radial features of every edge, each layer's messages, product-basis features and readouts,
// the ZBL energies, and the energy, atomic energies and forces.
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { Graph, type Tensor } from '../src/engine/tensor';
import { loadMACE } from './util';

const models = ['mace-mp-0b3-medium', 'mace-mp-0b2-small'].filter((m) => existsSync(`public/models/${m}.json`));
const relErr = (got: ArrayLike<number>, want: number[]) => {
  let m = 0, s = 1e-3;
  for (let i = 0; i < want.length; i++) { m = Math.max(m, Math.abs(got[i] - want[i])); s = Math.max(s, Math.abs(want[i])); }
  return m / s;
};

for (const name of models) {
  describe(`${name} vs mace-torch`, () => {
    const model = loadMACE(name);
    for (const file of readdirSync('tests/reference').filter((f) => f.startsWith(`${name}_`))) {
      const ref = JSON.parse(readFileSync(`tests/reference/${file}`, 'utf8'));
      it(ref.case, async () => {
        const be = model.be, g = new Graph(be);
        const sys = { numbers: ref.atomic_numbers, positions: ref.positions, cell: ref.cell, pbc: ref.pbc };
        const out = model.forward(g, sys, { forces: true });
        const read = async (t: Tensor) => Array.from(await be.read(t.buf));
        const checks: [string, number, number][] = []; // what, relative error, tolerance
        if (ref.sh) {
          // match our edges (centre = receiver, neighbour = sender) to mace's by atoms and vector
          const Y = await read(out.internals.sh), R = await read(out.internals.radial);
          const nb = R.length / out.graph.center.length, vec = ref.edge_list.vector as number[][];
          const pos = ref.positions as number[][];
          const ours = Array.from(out.graph.center, (c, e) => {
            const n = out.graph.neighbor[e], s = out.graph.shift;
            return [0, 1, 2].map((k) => pos[c][k] - pos[n][k] - s[3 * e + k]);
          });
          const match = vec.map((v, q) => ours.findIndex((u, e) => out.graph.center[e] === ref.edge_list.receiver[q]
            && out.graph.neighbor[e] === ref.edge_list.sender[q] && Math.hypot(u[0] - v[0], u[1] - v[1], u[2] - v[2]) < 1e-5));
          expect(match.every((e) => e >= 0) && out.graph.center.length === vec.length, 'same edges as mace').toBe(true);
          checks.push(['spherical harmonics', relErr(match.flatMap((e) => Y.slice(16 * e, 16 * e + 16)), ref.sh.flat()), 2e-6]);
          checks.push(['radial features', relErr(match.flatMap((e) => R.slice(nb * e, nb * e + nb)), ref.radial.flat()), 2e-6]);
          if (ref.pair) checks.push(['ZBL', relErr(out.internals.pair ? await read(out.internals.pair) : ref.pair.map(() => 0), ref.pair), 2e-5]);
          for (const [i, layer] of ref.layers.entries()) {
            const ours = out.internals.layers[i];
            for (const [l, want] of Object.entries(layer.message as Record<string, number[][]>))
              checks.push([`layer ${i} message l=${l}`, relErr(await read(ours.message.get(+l)!), want.flat()), 2e-5]);
            for (const [l, want] of Object.entries(layer.product as Record<string, number[][]>))
              checks.push([`layer ${i} product l=${l}`, relErr(await read(ours.product.get(+l)!), want.flat()), 5e-5]);
            checks.push([`layer ${i} readout`, relErr(await read(ours.readout), layer.readout), 5e-5]);
          }
        }
        g.backward(out.energy);
        const E = (await read(out.energy))[0], F = (await be.read(out.positions.grad!)).map((x) => -x);
        const Fref = (ref.forces as number[][]).flat(), Fmax = Math.max(...Fref.map(Math.abs));
        checks.push(['atomic energies', relErr(await read(out.perAtom), ref.node_energy), 2e-6]);
        g.release();
        const dE = Math.abs(E - ref.energy), dF = Math.max(...Fref.map((f, i) => Math.abs(F[i] - f)));
        process.stderr.write(`\n${name} ${ref.case.padEnd(8)} E=${E.toFixed(5)} ref=${ref.energy.toFixed(5)} dE=${dE.toExponential(2)} dF=${dF.toExponential(2)} (max |F| ${Fmax.toFixed(2)}, ${ref.edges} edges)`);
        for (const [what, err, tol] of checks) {
          if (err > tol / 10) process.stderr.write(`\n   ${what}: ${err.toExponential(2)}`);
          expect(err, what).toBeLessThan(tol);
        }
        expect(dE).toBeLessThan(1e-5 * Math.max(1, Math.abs(ref.energy)));
        expect(dF).toBeLessThan(1e-4 * Math.max(1, Fmax));
      });
    }
  });
}

describe('MACE symmetry', () => {
  it.each(models)('%s: energy invariant and forces equivariant under rotation and translation', async (name) => {
    const model = loadMACE(name), ref = JSON.parse(readFileSync(`tests/reference/${name}_sulfur.json`, 'utf8'));
    const [a, b, c] = [0.7, -1.1, 2.3];
    const Rz = [[Math.cos(a), -Math.sin(a), 0], [Math.sin(a), Math.cos(a), 0], [0, 0, 1]];
    const Rx = [[1, 0, 0], [0, Math.cos(b), -Math.sin(b)], [0, Math.sin(b), Math.cos(b)]];
    const Ry = [[Math.cos(c), 0, Math.sin(c)], [0, 1, 0], [-Math.sin(c), 0, Math.cos(c)]];
    const mm = (A: number[][], B: number[][]) => A.map((r) => B[0].map((_, j) => r.reduce((s, x, k) => s + x * B[k][j], 0)));
    const Rot = mm(mm(Rz, Rx), Ry), rot = (v: number[]) => Rot.map((r) => r[0] * v[0] + r[1] * v[1] + r[2] * v[2]);
    const run = async (pos: number[][]) => {
      const g = new Graph(model.be);
      const out = model.forward(g, { numbers: ref.atomic_numbers, positions: pos }, { forces: true });
      g.backward(out.energy);
      const E = (await model.be.read(out.energy.buf))[0], F = Array.from(await model.be.read(out.positions.grad!), (x) => -x);
      g.release();
      return { E, F };
    };
    const p0 = ref.positions as number[][], base = await run(p0);
    const moved = await run(p0.map((p) => rot(p).map((x, k) => x + [1.5, -2, 0.25][k])));
    const Fr = Array.from({ length: p0.length }, (_, i) => rot(base.F.slice(3 * i, 3 * i + 3))).flat();
    expect(Math.abs(moved.E - base.E)).toBeLessThan(1e-5 * Math.abs(base.E));
    expect(Math.max(...Fr.map((f, i) => Math.abs(f - moved.F[i])))).toBeLessThan(5e-5);
  });
});
