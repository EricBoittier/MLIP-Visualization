// A TikZ-style 2D schematic of the architecture, generated from the module tree:
// boxes coloured by module type, arrows for the data flow, a dashed backward arrow
// from the energy to the positions, progress bars, and the active module lit up.
// Clicking a box jumps there; its ± collapses the module in the 3D view.
import { type Mod, MOD_COLOR } from './modules';

const NS = 'http://www.w3.org/2000/svg';
const LEAF = 30, TITLE = 19, PAD = 6, VGAP = 13, HGAP = 10;

interface BoxRef { mod: Mod; rect: SVGRectElement; fwd: SVGRectElement; bwd: SVGRectElement; w: number }

export class Diagram {
  private svg: SVGSVGElement;
  private boxes: BoxRef[] = [];
  onNavigate: ((m: Mod) => void) | null = null;
  onToggle: ((m: Mod) => void) | null = null;

  constructor(readonly el: HTMLElement) {
    this.svg = document.createElementNS(NS, 'svg');
    el.appendChild(this.svg);
  }

  build(root: Mod, subtitle: (m: Mod) => string, collapsed: Set<string>, inputLabel = 'positions r, elements Z') {
    const W = Math.max(this.el.clientWidth, 260), right = 26, w0 = W - right - 4;
    const parts: string[] = [];
    this.boxes = [];
    const refs: { mod: Mod; w: number }[] = [];
    const subs = (m: Mod) => m.items.flatMap((it) => ('mod' in it ? [it.mod] : []));
    const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');
    const arrowV = (x: number, y0: number, y1: number) =>
      parts.push(`<path d="M${x},${y0} L${x},${y1 - 1}" class="arrow" marker-end="url(#ah)"/>`);
    const arrowH = (x0: number, x1: number, y: number) =>
      parts.push(`<path d="M${x0},${y} L${x1 - 1},${y}" class="arrow" marker-end="url(#ah)"/>`);

    const leaf = (m: Mod, x: number, y: number, w: number, h = LEAF) => {
      const c = MOD_COLOR[m.type], k = refs.length;
      refs.push({ mod: m, w });
      const sub = subtitle(m);
      const toggle = subs(m).length ? `<text x="${x + w - 9}" y="${y + 12}" class="tg" data-t="${k}">${collapsed.has(m.id) ? '+' : '−'}</text>` : '';
      parts.push(`<g class="box" data-k="${k}">
        <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="5" fill="${c}" fill-opacity="0.16" stroke="${c}" class="b"/>
        <rect x="${x + 1}" y="${y + h - 3.5}" width="0" height="2.5" fill="#f99e29" class="pf"/>
        <rect x="${x + 1}" y="${y + h - 3.5}" width="0" height="2.5" fill="#a855f7" class="pb"/>
        <text x="${x + 7}" y="${y + 13}" class="t" fill="${c}">${esc(w < 90 ? m.short : m.title)}</text>
        ${sub && w >= 70 ? `<text x="${x + 7}" y="${y + 24}" class="s">${esc(sub)}</text>` : ''}${toggle}</g>`);
    };

    /** Draw a module at (x, y) with width w; returns its height. */
    const draw = (m: Mod, x: number, y: number, w: number): number => {
      const kids = subs(m);
      if (!kids.length || collapsed.has(m.id)) { leaf(m, x, y, w); return LEAF; }
      // container: title row, then children in a row (if they fit) or a column
      const c = MOD_COLOR[m.type], k = refs.length;
      refs.push({ mod: m, w });
      const inner = w - 2 * PAD;
      const horizontal = kids.every((q) => !subs(q).length) && kids.length > 1 && inner / kids.length >= 52;
      let h = TITLE;
      const body: (() => void)[] = [];
      if (horizontal) {
        const cw = (inner - HGAP * (kids.length - 1)) / kids.length;
        kids.forEach((q, i) => {
          const cx = x + PAD + i * (cw + HGAP);
          body.push(() => {
            leaf(q, cx, y + h, cw);
            if (i) arrowH(cx - HGAP, cx, y + h + LEAF / 2);
          });
        });
        h += LEAF + PAD;
      } else {
        let yy = y + TITLE;
        kids.forEach((q, i) => {
          const y0 = yy;
          body.push(() => { if (i) arrowV(x + w / 2, y0 - VGAP, y0); });
          const save = parts.length;
          const hh = draw(q, x + PAD, yy, inner);
          // keep the child's markup after the frame's background
          const child = parts.splice(save);
          body.push(() => parts.push(...child));
          yy += hh + VGAP;
        });
        h = yy - VGAP + PAD - y;
      }
      parts.push(`<g class="box frame" data-k="${k}">
        <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="7" fill="${c}" fill-opacity="0.06" stroke="${c}" stroke-opacity="0.7" class="b"/>
        <rect x="${x + 1}" y="${y + h - 3.5}" width="0" height="2.5" fill="#f99e29" class="pf"/>
        <rect x="${x + 1}" y="${y + h - 3.5}" width="0" height="2.5" fill="#a855f7" class="pb"/>
        <text x="${x + 7}" y="${y + 13}" class="t" fill="${c}">${esc(m.title)}</text>
        <text x="${x + w - 9}" y="${y + 12}" class="tg" data-t="${k}">−</text></g>`);
      body.forEach((f) => f());
      return h;
    };

    let y = 4;
    parts.push(`<g><rect x="4" y="${y}" width="${w0}" height="24" rx="12" class="io"/>
      <text x="${4 + w0 / 2}" y="${y + 16}" class="io-t">${esc(inputLabel)}</text></g>`);
    const yIn = y + 12;
    y += 24 + VGAP;
    const top = subs(root);
    top.forEach((m, i) => {
      arrowV(4 + w0 / 2, y - VGAP, y);
      y += draw(m, 4, y, w0) + VGAP;
      void i;
    });
    const yE = y - VGAP - LEAF / 2;
    // backward: energy -> positions
    const xb = 4 + w0 + 12;
    parts.push(`<path d="M${4 + w0},${yE} L${xb},${yE} L${xb},${yIn} L${4 + w0 + 3},${yIn}" class="back" marker-end="url(#ahb)"/>
      <text x="${xb + 4}" y="${(yE + yIn) / 2}" class="back-t" transform="rotate(-90 ${xb + 4} ${(yE + yIn) / 2})">backward: F = −∂E/∂r</text>`);
    this.svg.setAttribute('viewBox', `0 0 ${W} ${y}`);
    this.svg.setAttribute('width', `${W}`);
    this.svg.setAttribute('height', `${y}`);
    this.svg.innerHTML = `<defs>
      <marker id="ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#7b8496"/></marker>
      <marker id="ahb" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#a855f7"/></marker>
    </defs>${parts.join('')}`;
    this.boxes = refs.map((r, k) => {
      const g = this.svg.querySelector(`g.box[data-k="${k}"]`)!;
      return { ...r, rect: g.querySelector('rect.b')!, fwd: g.querySelector('rect.pf')!, bwd: g.querySelector('rect.pb')! };
    });
    this.svg.onclick = (e) => {
      const t = (e.target as Element).closest('[data-t]');
      if (t) return this.onToggle?.(refs[+t.getAttribute('data-t')!].mod);
      const g = (e.target as Element).closest('g.box');
      if (g) this.onNavigate?.(refs[+g.getAttribute('data-k')!].mod);
    };
  }

  /** Progress of every box, and which one holds the active op. */
  update(fwd: Float32Array, bwd: Float32Array, active: number) {
    for (const b of this.boxes) {
      const { first, last } = b.mod;
      let sf = 0, sb = 0;
      for (let i = first; i <= last; i++) { sf += fwd[i] ?? 0; sb += bwd[i] ?? 0; }
      const n = last - first + 1;
      b.fwd.setAttribute('width', `${Math.max(0, ((b.w - 2) * sf) / n)}`);
      b.bwd.setAttribute('width', `${Math.max(0, ((b.w - 2) * sb) / n)}`);
      const on = active >= first && active <= last;
      b.rect.classList.toggle('on', on);
    }
  }
}
