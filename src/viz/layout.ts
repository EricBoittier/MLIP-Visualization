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

/** `compact`: zen mode, no text rows and the blocks packed close together.
 *  `weightsOnly`: just the parameters, at their own size, grouped by module. */
export function layout(root: Mod, ops: OpInfo[], trace: Trace, collapsed: Set<string>, aspect: number, compact = false,
                       weightsOnly = false): Layout {
  const LBL = compact ? 0 : LABEL, PADc = compact ? 0.8 : PAD, SEP = compact ? 0.7 : 1.5;
  const gapOf = (depth: number) => (compact ? (depth === 0 ? 3 : 1) : gap(depth));
  const band = (title: number) => (compact ? 0 : title * 1.8);
  const cards: Card[] = new Array(ops.length);
  const frames = new Map<string, Frame>();
  const attn = new Map(trace.attention.map((a) => [a.op, a]));

  // ---- sizes, bottom-up
  const card = (i: number): Card => {
    if (weightsOnly) {
      let x = 0;
      const weights = ops[i].params.flatMap((name) => {
        const p = trace.params?.[name]?.value;
        if (!p) return [];
        const r = { x, y: LBL, w: p.cols, h: p.rows };
        x += p.cols + 3 * SEP;
        return [{ name, rect: r }];
      });
      const h = weights.length ? LBL + Math.max(...weights.map((w) => w.rect.h)) : 0;
      return { op: i, rect: { x: 0, y: 0, w: weights.length ? Math.max(x - 3 * SEP, 8) : 0, h }, block: { x: 0, y: LBL, w: 0, h: 0 },
               weights, heads: [], visible: weights.length > 0 };
    }
    const t = trace.values[i]!;
    const bw = Math.max(t.cols, 1), bh = Math.max(t.rows, 1);
    const H = compact ? bh : Math.min(Math.max(bh, 8), 32);
    let x = 0;
    const weights = ops[i].params.flatMap((name) => {
      const p = trace.params?.[name]?.value;
      if (!p) return [];
      const s = Math.min(1, H / p.rows), w = Math.min(p.cols * s, 64), r = { x, y: LBL, w, h: p.rows * s };
      x += w + SEP;
      return [{ name, rect: r }];
    });
    const block = { x, y: LBL, w: bw, h: bh };
    x += bw;
    const a = attn.get(i);
    const heads = a ? Array.from({ length: a.heads }, (_, hd) => ({ head: hd, rect: { x: x + 2 * SEP + hd * (a.n + SEP), y: LBL, w: a.n, h: a.n } })) : [];
    if (a) x += 2 * SEP + a.heads * (a.n + SEP);
    const h = LBL + Math.max(bh, ...weights.map((w) => w.rect.h), ...heads.map((q) => q.rect.h));
    return { op: i, rect: { x: 0, y: 0, w: Math.max(x, compact ? 1 : 8), h }, block, weights, heads, visible: true };
  };

  type Sized = { w: number; h: number; place: (x: number, y: number, visible: boolean) => void };
  const size = (m: Mod, visible: boolean): Sized => {
    const isCollapsed = collapsed.has(m.id) && m.depth > 0;
    const frame: Frame = { mod: m, rect: { x: 0, y: 0, w: 0, h: 0 }, collapsed: isCollapsed, visible, title: titleSize(m.depth) };
    if (m.depth > 0) frames.set(m.id, frame);
    const all: Sized[] = m.items.map((it) => {
      if ('mod' in it) return size(it.mod, visible && !isCollapsed);
      const c = (cards[it.op] = card(it.op));
      const empty = weightsOnly && !c.weights.length;
      return { w: c.rect.w, h: c.rect.h, place: (x, y, v) => { c.rect.x = x; c.rect.y = y; c.visible = v && !empty; } };
    });
    // with only the weights, ops (and modules) without any take no room
    const kids = weightsOnly ? all.filter((k) => k.w > 0) : all;
    if (weightsOnly && !kids.length && m.depth > 0) {
      return { w: 0, h: 0, place: (x, y) => { frame.rect = { x, y, w: 0, h: 0 }; frame.visible = false; all.forEach((k) => k.place(x, y, false)); } };
    }
    const pad = m.depth ? PADc : 0, g = gapOf(m.depth);
    let top = m.depth ? band(frame.title) : 0;
    if (isCollapsed) {
      frame.title = titleSize(m.depth) * 1.5;
      top = band(frame.title);
      const w = Math.max(24, m.title.length * frame.title * 0.62 + 2 * PAD), h = top + 5;
      return {
        w, h, place: (x, y, v) => {
          frame.rect = { x, y, w, h };
          frame.visible = v;
          kids.forEach((k) => k.place(x + PAD, y + top, false));
        },
      };
    }
    // flow children into rows: try a range of wrap widths, keep the tightest
    const flow = (W: number) => {
      const rows: { items: Sized[]; h: number; w: number }[] = [];
      for (const k of kids) {
        const r = rows[rows.length - 1];
        if (!r || r.w + g + k.w > W) rows.push({ items: [k], h: k.h, w: k.w });
        else { r.items.push(k); r.w += g + k.w; r.h = Math.max(r.h, k.h); }
      }
      return rows;
    };
    const target = m.depth ? 2.2 : aspect;
    const wMin = Math.max(...kids.map((k) => k.w)), wMax = kids.reduce((s, k) => s + k.w + g, 0);
    let rows = flow(wMin), best = Infinity;
    for (let q = 0; q <= 24; q++) {
      const W = wMin * (wMax / wMin) ** (q / 24);
      const rr = flow(W);
      const w = Math.max(...rr.map((r) => r.w)), h = rr.reduce((s, r) => s + r.h + g, 0);
      const cost = w * h * (1 + 0.6 * Math.abs(Math.log(w / h / target)));
      if (cost < best) { best = cost; rows = rr; }
    }
    // the title grows with the frame, so it reads whenever the frame fills the view
    const rowsW = Math.max(...rows.map((r) => r.w));
    frame.title = Math.min(Math.max(rowsW * 0.03, titleSize(m.depth)), 40);
    top = m.depth ? band(frame.title) : 0;
    const inner = { w: compact ? rowsW : Math.max(rowsW, m.title.length * frame.title * 0.62),
                    h: rows.reduce((s, r) => s + r.h, 0) + g * Math.max(rows.length - 1, 0) };
    const w = inner.w + 2 * pad, h = inner.h + top + 2 * pad;
    return {
      w, h, place: (x, y, v) => {
        frame.rect = { x, y, w, h };
        frame.visible = v;
        if (weightsOnly) all.filter((k) => k.w === 0).forEach((k) => k.place(x, y, false));
        let yy = y + top + pad;
        for (const r of rows) {
          let xx = x + pad;
          for (const k of r.items) { k.place(xx, yy, v); xx += k.w + g; }
          yy += r.h + g;
        }
      },
    };
  };
  const flat: Mod = { ...root, items: ops.map((_, i) => ({ op: i })) };
  const s = size(compact ? flat : root, true);
  s.place(0, 0, true);
  if (compact) for (const m of walk(root)) { const f = frames.get(m.id); if (f) f.visible = false; else if (m.depth) frames.set(m.id, { mod: m, rect: { x: 0, y: 0, w: 0, h: 0 }, collapsed: false, visible: false, title: 0 }); }

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
