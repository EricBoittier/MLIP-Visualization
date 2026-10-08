// Many configurations of one molecule in a single forward pass: copies laid out on a grid far
// enough apart that no model's cutoff joins two of them, so each copy's energy is the sum of its
// atoms' energies. Only for local models of isolated molecules (no long-range terms, no cell).
import type { Backend } from '../engine/backend';
import { Graph } from '../engine/tensor';
import type { System } from '../common/structure';
import type { Model } from '../models/types';

/** Clear space between copies, Angstrom: more than any supported model's cutoff (PET 7.5, MACE 6, ANI 5.2). */
export const GAP = 12;

/** Energies (eV) of `walkers` (each [3N], Angstrom), in passes of at most `maxAtoms` atoms. */
export async function batchEnergies(m: Model, be: Backend, sys: System, walkers: Float64Array[], maxAtoms = 1024,
                                    read: (b: any) => Promise<Float32Array> = (b) => be.read(b)): Promise<Float64Array> {
  const n = sys.numbers.length, per = Math.max(1, Math.floor(maxAtoms / n)), out = new Float64Array(walkers.length);
  for (let start = 0; start < walkers.length; start += per) {
    const chunk = walkers.slice(start, start + per), B = chunk.length, side = Math.ceil(Math.cbrt(B));
    // each copy about its own centroid; the spacing covers the largest copy
    let R = 0;
    const centred = chunk.map((w) => {
      const c = [0, 1, 2].map((k) => { let s = 0; for (let i = 0; i < n; i++) s += w[3 * i + k]; return s / n; });
      const p = Float64Array.from(w, (v, i) => v - c[i % 3]);
      for (let i = 0; i < n; i++) R = Math.max(R, Math.hypot(p[3 * i], p[3 * i + 1], p[3 * i + 2]));
      return p;
    });
    const L = 2 * R + GAP, o = ((side - 1) * L) / 2, positions: number[][] = [];
    centred.forEach((p, b) => {
      const g = [b % side, Math.floor(b / side) % side, Math.floor(b / side ** 2)].map((k) => k * L - o);
      for (let i = 0; i < n; i++) positions.push([p[3 * i] + g[0], p[3 * i + 1] + g[1], p[3 * i + 2] + g[2]]);
    });
    const g = new Graph(be);
    try {
      const big: System = { numbers: Array.from({ length: B * n }, (_, i) => sys.numbers[i % n]), positions };
      const e = await read(m.forward(g, big, { forces: false }).perAtom.buf);
      for (let b = 0; b < B; b++) { let s = 0; for (let i = 0; i < n; i++) s += e[b * n + i]; out[start + b] = s; }
    } finally {
      g.release();
    }
  }
  return out;
}
