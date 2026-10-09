// The module tree of a pass, from the op scopes: what the 3D frames, the 2D
// diagram and the narration are organised by. Each model names its modules;
// colours are by kind of module, shared across models.
import type { OpInfo } from '../worker/protocol';
import { mode } from './theme';

/** Kinds of module, shared by all models so colours mean the same thing everywhere. */
export type ModType = string;

/** By kind of module, from the Metatensor and metatomic logos: cyan and blue for geometry and descriptors,
 *  indigo, violet, green and gold for learned representations, red and salmon for outputs. Tuned for the
 *  dark theme; modColor() deepens them for the light one. */
export const MOD_COLOR: Record<string, string> = {
  root: '#9aa3ad', other: '#9aa3ad',
  // geometry and descriptors
  geometry: '#62c0d6', adaptive: '#62c0d6', cutoff: '#6c92e0', descriptor: '#62c0d6', radial: '#6c92e0',
  angular: '#7cd18a', density: '#6c92e0', spectrum: '#7cd18a', rbf: '#6c92e0',
  // learned representations
  embedding: '#e0776c', gnn: '#9aa3ad', edge_tokens: '#e8955f', transformer: '#8c95e6', tokens: '#b08cec',
  attention: '#f2c94c', mlp: '#7cd18a', node_update: '#b08cec', message: '#6c92e0', interaction: '#6c92e0',
  residual: '#7cd18a', network: '#8c95e6', kernel: '#f2c94c', regression: '#e8955f', product: '#f2c94c',
  // outputs
  readout: '#e0776c', head: '#eaa199', energy: '#d75142', ncforce: '#7cd18a', charges: '#cf8be0', electrostatics: '#b97ad8',
};

/** The colour of a kind of module in the current theme (deeper on light backgrounds, to stay legible). */
export function modColor(type: string) {
  const c = MOD_COLOR[type] ?? MOD_COLOR.other;
  if (mode() === 'dark') return c;
  return '#' + [1, 3, 5].map((k) => Math.round(parseInt(c.slice(k, k + 2), 16) * 0.68).toString(16).padStart(2, '0')).join('');
}

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
