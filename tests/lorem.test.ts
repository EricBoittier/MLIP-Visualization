// LOREM against metatrain's experimental port (float32, scripts/convert_lorem.py):
// the energy, the atomic energies and the forces, for molecules and two periodic cells.
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { Graph, type Tensor } from '../src/engine/tensor';
import { loadLOREM } from './util';

const relErr = (got: ArrayLike<number>, want: number[]) => {
  let m = 0, s = 1e-3;
  for (let i = 0; i < want.length; i++) { m = Math.max(m, Math.abs(got[i] - want[i])); s = Math.max(s, Math.abs(want[i])); }
  return m / s;
};

describe('LOREM vs metatrain', () => {
  if (!existsSync('public/models/lorem-demo.json')) return;
  const model = loadLOREM();
  for (const file of readdirSync('tests/reference').filter((f) => f.startsWith('lorem-demo_'))) {
    const ref = JSON.parse(readFileSync(`tests/reference/${file}`, 'utf8'));
    it(ref.case, async () => {
      const be = model.be, g = new Graph(be);
      const sys = { numbers: ref.atomic_numbers, positions: ref.positions, cell: ref.cell, pbc: ref.pbc };
      const out = model.forward(g, sys, { forces: true });
      const read = async (t: Tensor) => Array.from(await be.read(t.buf));
      const pos = ref.positions as number[][];
      const ours = Array.from(out.graph.center, (c, e) => {
        const n = out.graph.neighbor[e], s = out.graph.shift;
        return [c, n, ...[0, 1, 2].map((k) => pos[n][k] - pos[c][k] + s[3 * e + k])] as number[];
      });
      const want = (ref.vector as number[][]).map((v, q) => [ref.center[q], ref.neighbor[q], ...v]);
      const match = want.map((w) => ours.findIndex((u) => u[0] === w[0] && u[1] === w[1]
        && Math.hypot(u[2] - w[2], u[3] - w[3], u[4] - w[4]) < 1e-4));
      expect(match.every((e) => e >= 0) && ours.length === want.length, `${ours.length} edges, metatrain has ${want.length}`).toBe(true);
      g.backward(out.energy);
      const E = (await read(out.energy))[0];
      const F = (await be.read(out.positions.grad!)).map((x) => -x);
      const atomic = await read(out.perAtom);
      const sr = await read(out.internals.sr), lr = await read(out.internals.lr);
      g.release();
      const Fref = (ref.forces as number[][]).flat();
      const dE = Math.abs(E - ref.energy), dF = Math.max(...Fref.map((f, i) => Math.abs(F[i] - f)));
      const Fmax = Math.max(...Fref.map(Math.abs), 1e-6);
      process.stderr.write(`\nlorem ${ref.case.padEnd(8)} E=${E.toFixed(5)} ref=${ref.energy.toFixed(5)} dE=${dE.toExponential(2)} dF=${dF.toExponential(2)} sr=${relErr(sr, ref.sr).toExponential(2)} lr=${relErr(lr, ref.lr).toExponential(2)}`);
      expect(relErr(atomic, ref.node_energy), 'atomic energies').toBeLessThan(1e-5);
      expect(dE).toBeLessThan(1e-5 * Math.max(1, Math.abs(ref.energy)));
      expect(dF).toBeLessThan(1e-4 * Math.max(1, Fmax));
    });
  }
});

describe('LOREM symmetry', () => {
  if (!existsSync('tests/reference/lorem-demo_water.json')) return;
  it('energy invariant and forces equivariant under rotation', async () => {
    const model = loadLOREM(), ref = JSON.parse(readFileSync('tests/reference/lorem-demo_water.json', 'utf8'));
    const a = 0.7, b = -1.1;
    const Rz = [[Math.cos(a), -Math.sin(a), 0], [Math.sin(a), Math.cos(a), 0], [0, 0, 1]];
    const Rx = [[1, 0, 0], [0, Math.cos(b), -Math.sin(b)], [0, Math.sin(b), Math.cos(b)]];
    const mm = (A: number[][], B: number[][]) => A.map((r) => B[0].map((_, j) => r.reduce((s, x, k) => s + x * B[k][j], 0)));
    const Rot = mm(Rz, Rx), rot = (v: number[]) => Rot.map((r) => r[0] * v[0] + r[1] * v[1] + r[2] * v[2]);
    const run = async (pos: number[][]) => {
      const g = new Graph(model.be);
      const out = model.forward(g, { numbers: ref.atomic_numbers, positions: pos }, { forces: true });
      g.backward(out.energy);
      const E = (await model.be.read(out.energy.buf))[0];
      const F = Array.from(await model.be.read(out.positions.grad!), (x) => -x);
      g.release();
      return { E, F };
    };
    const moved = (ref.positions as number[][]).map((p) => rot(p).map((x, i) => x + [0.4, -0.2, 0.7][i]));
    const [u, v] = [await run(ref.positions), await run(moved)];
    expect(Math.abs(u.E - v.E)).toBeLessThan(1e-5 * Math.max(1, Math.abs(u.E)));
    // F' = R F for the rotated, translated structure
    const turned = Array.from({ length: u.F.length / 3 }, (_, i) => rot([u.F[3 * i], u.F[3 * i + 1], u.F[3 * i + 2]]));
    const dF = Math.max(...v.F.map((f, i) => Math.abs(f - turned[(i / 3) | 0][i % 3])));
    expect(dF).toBeLessThan(1e-4 * Math.max(1, ...u.F.map(Math.abs)));
  });
});
