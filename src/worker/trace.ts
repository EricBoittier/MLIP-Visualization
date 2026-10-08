// Capture what the visualiser shows of one forward + backward pass: for every
// op, the selected atom's rows of its output and of its gradient (exact values,
// columns averaged down to at most MAX_COLS), the weights, and the attention.
import type { Backend, Buf, Thumb } from '../engine/backend';
import type { Graph, Tensor } from '../engine/tensor';
import type { Forward, Model, RowSpace } from '../models/types';
import type { AttentionMap, OpInfo, Trace } from './protocol';

const MAX_COLS = 128, MAX_ROWS = 64, MAX_W = 64;

type ThumbReq = { x: Buf; R: number; C: number; rows: number; cols: number };

async function thumbs(be: Backend, reqs: ThumbReq[]): Promise<Thumb[]> {
  const many = (be as any).thumbs as ((r: ThumbReq[]) => Promise<Thumb[]>) | undefined;
  if (many) return many.call(be, reqs);
  return Promise.all(reqs.map((r) => be.thumb(r.x, r.R, r.C, r.rows, r.cols)));
}

const dims = (shape: number[]) => {
  const C = shape.length > 1 ? shape[shape.length - 1] : 1;
  return [shape.reduce((a, b) => a * b, 1) / Math.max(C, 1), C];
};

export function topology(g: Graph, model: Model): OpInfo[] {
  const producer = new Map<number, number>();
  g.tape.forEach((n, i) => producer.set(n.out.id, i));
  const paramName = new Map<number, string>();
  for (const [name, t] of model.params) paramName.set(t.id, name);
  return g.tape.map((n) => ({
    op: n.op,
    scope: n.scope,
    shape: n.out.shape,
    kind: n.out.kind,
    inputs: n.inputs.filter((t) => !paramName.has(t.id)).map((t) => producer.get(t.id) ?? -1),
    params: n.inputs.filter((t) => paramName.has(t.id)).map((t) => paramName.get(t.id)!),
    grad: !!n.out.grad, // received a gradient in the backward pass
  }));
}

export const topologyKey = (ops: OpInfo[]) => ops.map((o) => `${o.scope}|${o.op}|${o.kind}|${o.grad ? 1 : 0}`).join(';');

/** The rows of a tensor that belong to atom a (null: show all rows, block-averaged). */
export function selectRows(space: RowSpace | undefined, R: number, a: number): Int32Array | null {
  if (!space || space.owner.length !== R) return null;
  if (space.all) {
    if (R <= MAX_ROWS) return Int32Array.from({ length: R }, (_, i) => i);
    const s = Math.max(0, Math.min(a - MAX_ROWS / 2, R - MAX_ROWS));
    return Int32Array.from(new Set([a, ...Array.from({ length: MAX_ROWS - 1 }, (_, i) => s + i)]));
  }
  const rows: number[] = [];
  space.owner.forEach((o, r) => { if (o === a && rows.length < MAX_ROWS) rows.push(r); });
  return Int32Array.from(rows);
}

export async function capture(g: Graph, model: Model, out: Forward, opts: { selected: number }) {
  const be = model.be;
  const a = Math.min(Math.max(opts.selected, 0), out.positions.shape[0] - 1);
  const temps: Buf[] = [];
  const view = (x: Buf, shape: number[], rows: Int32Array | null): ThumbReq => {
    const [R, C] = dims(shape);
    const cols = Math.min(C, MAX_COLS);
    if (!rows) return { x, R, C, rows: Math.max(1, Math.min(R, MAX_ROWS)), cols };
    const n = rows.length;
    const tmp = be.zeros(Math.max(n * C, 1)), idx = be.uploadI32(rows);
    if (n) be.gather(x, idx, tmp, n, C, false);
    temps.push(tmp, idx);
    return { x: tmp, R: n, C, rows: Math.max(n, 1), cols };
  };
  const rowSel = g.tape.map((n) => selectRows(n.out.kind ? out.rows[n.out.kind] : undefined, dims(n.out.shape)[0], a));
  const valueReqs = g.tape.map((n, i) => view(n.out.buf, n.out.shape, rowSel[i]));
  const withGrad = g.tape.map((n, i) => (n.out.grad ? i : -1)).filter((i) => i >= 0);
  const gradReqs = withGrad.map((i) => view(g.tape[i].out.grad!, g.tape[i].out.shape, rowSel[i]));
  const params = [...model.params.values()];
  const paramReqs = params.map((t: Tensor) => {
    const [R, C] = dims(t.shape);
    return { x: t.buf, R, C, rows: Math.min(R, MAX_W), cols: Math.min(C, MAX_W) };
  });
  // RMS of every row (all atoms / edges / tokens), to colour the structure view
  const normBufs: Buf[] = [], normOf: [number, 'v' | 'g'][] = [];
  const rowRms = (x: Buf, shape: number[]) => {
    const [R, C] = dims(shape);
    const sq = be.zeros(R * C), out = be.zeros(R);
    be.unary('square', x, sq, R * C, 0, 0);
    be.rowSum(sq, out, R, C, false);
    temps.push(sq, out);
    normBufs.push(out);
  };
  g.tape.forEach((n, i) => {
    if (!n.out.kind || !out.rows[n.out.kind] || out.rows[n.out.kind].owner.length !== dims(n.out.shape)[0]) return;
    rowRms(n.out.buf, n.out.shape); normOf.push([i, 'v']);
    if (n.out.grad) { rowRms(n.out.grad, n.out.shape); normOf.push([i, 'g']); }
  });
  const all = await thumbs(be, [...valueReqs, ...gradReqs, ...paramReqs]);
  const normData: Float32Array[] = normBufs.length
    ? ((be as any).readMany ? await (be as any).readMany(normBufs) : await Promise.all(normBufs.map((b) => be.read(b))))
    : [];
  temps.forEach((b) => be.free(b));
  const norms: (Float32Array | null)[] = g.tape.map(() => null), gradNorms: (Float32Array | null)[] = g.tape.map(() => null);
  normOf.forEach(([i, w], k) => {
    const C = dims(g.tape[i].out.shape)[1];
    (w === 'v' ? norms : gradNorms)[i] = normData[k].map((x) => Math.sqrt(x / C));
  });

  const values = all.slice(0, valueReqs.length);
  const grads: (Thumb | null)[] = g.tape.map(() => null);
  withGrad.forEach((i, k) => (grads[i] = all[valueReqs.length + k]));
  const paramThumbs: NonNullable<Trace['params']> = {};
  params.forEach((t, k) => (paramThumbs[t.name] = { value: all[valueReqs.length + gradReqs.length + k], grad: null }));

  // the selected atom's attention heads
  const specs = out.attention ?? [];
  const reads = specs.map((x) => x.node.aux!.probs.buf);
  const probsAll: Float32Array[] = !reads.length ? []
    : (be as any).readMany ? await (be as any).readMany(reads) : await Promise.all(reads.map((b) => be.read(b)));
  const attention: AttentionMap[] = specs.map((x, i) => {
    const { offset, atoms } = x.tokens(a), n = atoms.length;
    return { heads: x.heads, n, atoms, op: g.tape.indexOf(x.node), probs: probsAll[i].slice(offset, offset + x.heads * n * n) };
  });
  const paramShapes = Object.fromEntries([...model.params].map(([k, t]) => [k, t.shape]));
  return { values, grads, params: paramThumbs, paramShapes, attention, rowSel, selected: a, norms, gradNorms };
}
