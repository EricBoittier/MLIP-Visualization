// Space-filling layout. Every module is a frame whose children (ops and
// sub-modules) flow left to right and wrap like text, sized towards a target
// aspect ratio; a collapsed module is just its title. Units are tensor cells,
// x to the right and y downwards.
import type { OpInfo, Trace } from '../worker/protocol';
import { type Mod, walk } from './modules';

export interface Rect { x: number; y: number; w: number; h: number }
export interface Card {
  op: number;
  rect: Rect;
  block: Rect;
  weights: { name: string; rect: Rect }[];
  heads: { head: number; rect: Rect }[];
  visible: boolean;
}
export interface Frame { mod: Mod; rect: Rect; collapsed: boolean; visible: boolean; title: number }
export interface Layout {
  cards: Card[];
  frames: Map<string, Frame>;
  bounds: Rect;
  /** Where an op is drawn: its block, or the collapsed module hiding it. */
  anchor: (op: number) => Rect;
}

export const LABEL = 2.2; // height of an op's label row
/** Smallest title per depth; titles otherwise grow with their frame (see below). */
export const titleSize = (depth: number) => [8, 6, 3.2, 2.2, 1.7][Math.min(depth, 4)];
const PAD = 2.5;
const gap = (depth: number) => (depth === 0 ? 12 : 3);

export function layout(root: Mod, ops: OpInfo[], trace: Trace, collapsed: Set<string>, aspect: number): Layout {
  const cards: Card[] = new Array(ops.length);
  const frames = new Map<string, Frame>();
  const attn = new Map(trace.attention.map((a) => [a.op, a]));

  // ---- sizes, bottom-up
  const card = (i: number): Card => {
    const t = trace.values[i]!;
    const bw = Math.max(t.cols, 1), bh = Math.max(t.rows, 1);
    const H = Math.min(Math.max(bh, 8), 32);
    let x = 0;
    const weights = ops[i].params.flatMap((name) => {
      const p = trace.params?.[name]?.value;
      if (!p) return [];
      const s = Math.min(1, H / p.rows), w = Math.min(p.cols * s, 64), r = { x, y: LABEL, w, h: p.rows * s };
      x += w + 1.5;
      return [{ name, rect: r }];
    });
    const block = { x, y: LABEL, w: bw, h: bh };
    x += bw;
    const a = attn.get(i);
    const heads = a ? Array.from({ length: a.heads }, (_, hd) => ({ head: hd, rect: { x: x + 3 + hd * (a.n + 1.5), y: LABEL, w: a.n, h: a.n } })) : [];
    if (a) x += 3 + a.heads * (a.n + 1.5);
    const h = LABEL + Math.max(bh, ...weights.map((w) => w.rect.h), ...heads.map((q) => q.rect.h));
    return { op: i, rect: { x: 0, y: 0, w: Math.max(x, 8), h }, block, weights, heads, visible: true };
  };

  type Sized = { w: number; h: number; place: (x: number, y: number, visible: boolean) => void };
  const size = (m: Mod, visible: boolean): Sized => {
    const isCollapsed = collapsed.has(m.id) && m.depth > 0;
    const frame: Frame = { mod: m, rect: { x: 0, y: 0, w: 0, h: 0 }, collapsed: isCollapsed, visible, title: titleSize(m.depth) };
    if (m.depth > 0) frames.set(m.id, frame);
    const kids: Sized[] = m.items.map((it) => {
      if ('mod' in it) return size(it.mod, visible && !isCollapsed);
      const c = (cards[it.op] = card(it.op));
      return { w: c.rect.w, h: c.rect.h, place: (x, y, v) => { c.rect.x = x; c.rect.y = y; c.visible = v; } };
    });
    const pad = m.depth ? PAD : 0, g = gap(m.depth);
    let top = m.depth ? frame.title * 1.8 : 0;
    if (isCollapsed) {
      frame.title = titleSize(m.depth) * 1.5;
      top = frame.title * 1.8;
      const w = Math.max(24, m.title.length * frame.title * 0.62 + 2 * PAD), h = top + 5;
      return {
        w, h, place: (x, y, v) => {
          frame.rect = { x, y, w, h };
          frame.visible = v;
          kids.forEach((k) => k.place(x + PAD, y + top, false));
        },
      };
    }
    // flow children into rows no wider than W
    const area = kids.reduce((s, k) => s + (k.w + g) * (k.h + g), 0);
    const W = Math.max(...kids.map((k) => k.w), Math.sqrt(area * (m.depth ? 2.4 : aspect)));
    const rows: { items: Sized[]; h: number; w: number }[] = [];
    for (const k of kids) {
      const r = rows[rows.length - 1];
      if (!r || r.w + g + k.w > W) rows.push({ items: [k], h: k.h, w: k.w });
      else { r.items.push(k); r.w += g + k.w; r.h = Math.max(r.h, k.h); }
    }
    // the title grows with the frame, so it reads whenever the frame fills the view
    const rowsW = Math.max(...rows.map((r) => r.w));
    frame.title = Math.min(Math.max(rowsW * 0.03, titleSize(m.depth)), 40);
    top = m.depth ? frame.title * 1.8 : 0;
    const inner = { w: Math.max(rowsW, m.title.length * frame.title * 0.62),
                    h: rows.reduce((s, r) => s + r.h, 0) + g * Math.max(rows.length - 1, 0) };
    const w = inner.w + 2 * pad, h = inner.h + top + 2 * pad;
    return {
      w, h, place: (x, y, v) => {
        frame.rect = { x, y, w, h };
        frame.visible = v;
        let yy = y + top + pad;
        for (const r of rows) {
          let xx = x + pad;
          for (const k of r.items) { k.place(xx, yy, v); xx += k.w + g; }
          yy += r.h + g;
        }
      },
    };
  };
  const s = size(root, true);
  s.place(0, 0, true);

  // ops hidden in a collapsed module are drawn at that module's box
  const hiddenBy = new Map<number, Rect>();
  for (const m of walk(root)) {
    const f = frames.get(m.id);
    if (!f || !f.collapsed || !f.visible) continue;
    for (let i = m.first; i <= m.last; i++) hiddenBy.set(i, f.rect);
  }
  const anchor = (i: number) => {
    const h = hiddenBy.get(i);
    if (h) return h;
    const c = cards[i];
    return { x: c.rect.x + c.block.x, y: c.rect.y + c.block.y, w: c.block.w, h: c.block.h };
  };
  return { cards, frames, bounds: { x: 0, y: 0, w: s.w, h: s.h }, anchor };
}
