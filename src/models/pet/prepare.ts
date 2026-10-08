// PET's host-side preprocessing: adaptive cutoffs and the index maps the network
// needs. Data-dependent but not differentiated; the differentiable geometry is
// rebuilt on the graph in model.ts.
import { cutoffFn } from '../../engine/cpu';
import { type NeighborList, neighborList, type System } from '../../common/structure';
import type { Hypers } from './checkpoint';

/** metatrain's adaptive cutoff ("solver"): per-atom radius r where the smoothed
 *  neighbour count plus a cubic baseline reaches the target, and the slope there. */
export function adaptiveCutoffSolver(nl: NeighborList, N: number, target: number, rmax: number, width: number) {
  const inv = 1 / rmax;
  const count = (r: Float64Array) => {
    const n = new Float64Array(N), dn = new Float64Array(N);
    for (let e = 0; e < nl.center.length; e++) {
      const i = nl.center[e], s = (nl.dist[e] - (r[i] - width)) / width;
      if (s <= 0) n[i] += 1;
      else if (s < 1) {
        const [f, dfds] = cutoffFn('bump', nl.dist[e], r[i], width);
        n[i] += f;
        dn[i] -= dfds / width;
      }
    }
    for (let i = 0; i < N; i++) {
      const x = r[i] * inv;
      n[i] += target * x ** 3;
      dn[i] += 3 * target * x * x * inv;
    }
    return [n, dn];
  };
  let lo = new Float64Array(N), hi = new Float64Array(N).fill(rmax), r = new Float64Array(N).fill(0.5 * rmax);
  for (let it = 0; it < 10; it++) {
    const [n, dn] = count(r);
    for (let i = 0; i < N; i++) {
      const f = n[i] - target;
      if (f <= 0) lo[i] = r[i]; else hi[i] = r[i];
      const rn = r[i] - f / Math.max(dn[i], 1e-6);
      r[i] = rn >= lo[i] && rn <= hi[i] ? rn : 0.5 * (lo[i] + hi[i]);
    }
  }
  const [n, dn] = count(r);
  const atomic = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const v = r[i] - (n[i] - target) / Math.max(dn[i], 1e-6);
    atomic[i] = Math.min(Math.max(v, rmax / 16), rmax);
  }
  return { r, dn: dn.map((x) => Math.max(x, 1e-6)), atomic };
}

/** Everything the network reads that does not depend on the weights. */
export interface Prepared {
  N: number;
  species: Int32Array; // species index per atom
  raw: NeighborList; // neighbour list up to the model's max cutoff
  adaptive: { r: Float64Array; dn: Float64Array; atomic: Float64Array } | null;
  keep: Int32Array; // raw edge index of every kept edge, grouped by centre
  center: Int32Array; // per kept edge
  neighbor: Int32Array;
  reverse: Int32Array; // kept index of the (j, i, -S) edge, -1 if none
  offsets: Int32Array; // [N+1] kept edges of atom i are [offsets[i], offsets[i+1])
  pairCutoff: Float64Array; // per kept edge
}

export function prepare(sys: System, h: Hypers, speciesToIndex: number[]): Prepared {
  const N = sys.numbers.length;
  const species = Int32Array.from(sys.numbers, (z) => {
    const s = speciesToIndex[z] ?? -1;
    if (s < 0) throw new Error(`element Z=${z} is not supported by this model`);
    return s;
  });
  const raw = neighborList(sys, h.cutoff);
  let adaptive: Prepared['adaptive'] = null;
  const keep: number[] = [], pair: number[] = [];
  if (h.num_neighbors_adaptive != null) {
    adaptive = adaptiveCutoffSolver(raw, N, h.num_neighbors_adaptive, h.cutoff, h.cutoff_width_adaptive);
    for (let e = 0; e < raw.center.length; e++) {
      const pc = 0.5 * (adaptive.atomic[raw.center[e]] + adaptive.atomic[raw.neighbor[e]]);
      if (raw.dist[e] <= pc) { keep.push(e); pair.push(pc); }
    }
  } else {
    for (let e = 0; e < raw.center.length; e++) { keep.push(e); pair.push(h.cutoff); }
  }
  const E = keep.length;
  const center = Int32Array.from(keep, (e) => raw.center[e]);
  const neighbor = Int32Array.from(keep, (e) => raw.neighbor[e]);
  const offsets = new Int32Array(N + 1);
  for (const c of center) offsets[c + 1]++;
  for (let i = 0; i < N; i++) offsets[i + 1] += offsets[i];
  // reverse edges: key (i, j, shift) -> kept index
  const key = (i: number, j: number, s0: number, s1: number, s2: number) => `${i},${j},${s0},${s1},${s2}`;
  const lookup = new Map<string, number>();
  keep.forEach((e, k) => lookup.set(key(raw.center[e], raw.neighbor[e], raw.shift[3 * e], raw.shift[3 * e + 1], raw.shift[3 * e + 2]), k));
  const reverse = Int32Array.from(keep, (e) =>
    lookup.get(key(raw.neighbor[e], raw.center[e], -raw.shift[3 * e], -raw.shift[3 * e + 1], -raw.shift[3 * e + 2])) ?? -1);
  return { N, species, raw, adaptive, keep: Int32Array.from(keep), center, neighbor, reverse, offsets,
           pairCutoff: Float64Array.from(pair), };
}
