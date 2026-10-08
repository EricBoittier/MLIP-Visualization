// The module tree of a pass, from the op scopes: what the 3D frames, the 2D
// diagram and the narration are organised by. Each model names its modules;
// colours are by kind of module, shared across models.
import type { OpInfo } from '../worker/protocol';

/** Kinds of module, shared by all models so colours mean the same thing everywhere. */
export type ModType = string;

export const MOD_COLOR: Record<string, string> = {
  root: '#94a3b8', other: '#94a3b8',
  // geometry and descriptors
  geometry: '#2dd4bf', adaptive: '#22d3ee', cutoff: '#38bdf8', descriptor: '#2dd4bf', radial: '#22d3ee',
  angular: '#84cc16', density: '#22d3ee', spectrum: '#a3e635', rbf: '#22d3ee',
  // learned representations
  embedding: '#f472b6', gnn: '#94a3b8', edge_tokens: '#fb923c', transformer: '#818cf8', tokens: '#c084fc',
  attention: '#facc15', mlp: '#4ade80', node_update: '#a78bfa', message: '#60a5fa', interaction: '#60a5fa',
  residual: '#4ade80', network: '#818cf8', kernel: '#facc15', regression: '#fb923c', product: '#f59e0b',
  // outputs
  readout: '#f87171', head: '#fca5a5', energy: '#fb7185', ncforce: '#a3e635', charges: '#f0abfc', electrostatics: '#e879f9',
};

export interface ModDesc { type: ModType; title: string; short?: string }
/** How a model names the module of one scope segment (null: a generic module named after the segment). */
export type Describe = (seg: string, path: string) => ModDesc | null;

export interface Mod {
  id: string; // unique: scope path, plus #k if the scope is re-entered later
  path: string;
  type: ModType;
  title: string;
  short: string; // for the 2D diagram
  depth: number;
  parent: Mod | null;
  items: Item[];
  first: number; // first and last op index inside
  last: number;
}
export type Item = { mod: Mod } | { op: number };

export function buildTree(ops: OpInfo[], describe: Describe, rootTitle: string): Mod {
  const root: Mod = { id: '', path: '', type: 'root', title: rootTitle, short: rootTitle, depth: 0, parent: null, items: [], first: 0, last: ops.length - 1 };
  const desc = (seg: string, path: string) => {
    const d = describe(seg, path) ?? { type: 'other', title: seg || path };
    return { ...d, short: d.short ?? d.title };
  };
  const seen = new Map<string, number>();
  let chain: Mod[] = [root];
  ops.forEach((op, i) => {
    const segs = op.scope ? op.scope.split('/') : [];
    let k = 0;
    while (k < segs.length && k + 1 < chain.length && chain[k + 1].path === segs.slice(0, k + 1).join('/')) k++;
    chain = chain.slice(0, k + 1);
    for (; k < segs.length; k++) {
      const path = segs.slice(0, k + 1).join('/'), parent = chain[chain.length - 1];
      const n = seen.get(path) ?? 0;
      seen.set(path, n + 1);
      const mod: Mod = { id: n ? `${path}#${n}` : path, path, ...desc(segs[k], path), depth: k + 1, parent, items: [], first: i, last: i };
      parent.items.push({ mod });
      chain.push(mod);
    }
    chain[chain.length - 1].items.push({ op: i });
    for (const m of chain) m.last = i;
  });
  return root;
}

export function* walk(m: Mod): Generator<Mod> {
  yield m;
  for (const it of m.items) if ('mod' in it) yield* walk(it.mod);
}

/** The innermost module of every op. */
export function leafOf(root: Mod, nOps: number): Mod[] {
  const out: Mod[] = new Array(nOps);
  for (const m of walk(root)) for (const it of m.items) if ('op' in it) out[it.op] = m;
  return out;
}

export const ancestors = (m: Mod | null) => { const a: Mod[] = []; for (; m; m = m.parent) a.unshift(m); return a; };
