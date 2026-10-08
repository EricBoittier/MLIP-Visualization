// The walkthrough article: one section per module, one paragraph per step of the
// pass, written from the live trace (shapes, the selected atom, its neighbours,
// attention weights, energies). The paragraph of the current step lights up and
// scrolls into view; clicking one jumps there.
import { SYMBOLS } from '../common/elements';
import type { ModelUI } from '../models/ui';
import type { OpInfo, Pass } from '../worker/protocol';
import { ancestors, type Mod, MOD_COLOR } from './modules';

export interface Step {
  mod: Mod;
  dir: 'fwd' | 'bwd';
  ops: number[]; // in playback order
}

/** Steps: runs of ops in the same module with the same description; ops whose
 *  description is "continue" (small elementwise ops) join the step before them. */
export function buildSteps(ops: OpInfo[], leaf: Mod[], ctx: Ctx): Step[] {
  const steps: Step[] = [];
  let lastKey = '';
  ops.forEach((_, i) => {
    const key = describe(i, leaf[i], ctx).key;
    const last = steps[steps.length - 1];
    if (last && last.mod === leaf[i] && (key === 'cont' || key === lastKey)) last.ops.push(i);
    else { steps.push({ mod: leaf[i], dir: 'fwd', ops: [i] }); lastKey = key; }
    if (key !== 'cont') lastKey = key;
  });
  // backward: one step per module, in reverse
  const back = ops.map((o, i) => (o.grad ? i : -1)).filter((i) => i >= 0).reverse();
  for (const i of back) {
    const last = steps[steps.length - 1];
    if (last.dir === 'bwd' && last.mod === leaf[i]) last.ops.push(i);
    else steps.push({ mod: leaf[i], dir: 'bwd', ops: [i] });
  }
  return steps;
}

export interface Ctx { ops: OpInfo[]; pass: Pass; meta: any; ui: ModelUI }

export const fmt = (x: number, d = 3) => (Math.abs(x) >= 1000 || (Math.abs(x) < 1e-3 && x !== 0) ? x.toExponential(2) : x.toFixed(d));
export const dims = (s: number[]) => `<span class="dim">[${s.join(' × ')}]</span>`;
export const atomTag = (pass: Pass, k: number) => `<b class="atom">${SYMBOLS[pass.numbers[k]]}${k}</b>`;

/** Atom a's edges in the model's graph: neighbour, edge index and length. */
export function neighbours(pass: Pass, a: number) {
  const g = pass.trace.graph, out: { j: number; e: number; r: number }[] = [];
  for (let e = 0; e < g.center.length; e++) {
    if (g.center[e] !== a) continue;
    const j = g.neighbor[e];
    out.push({ j, e, r: Math.hypot(...[0, 1, 2].map((q) => pass.positions[j][q] + g.shift[3 * e + q] - pass.positions[a][q])) });
  }
  return out;
}

/** What op i does, as a paragraph, keyed by the kind of step ('cont': part of the step before). */
export function describe(i: number, mod: Mod, c: Ctx, lastOp = i): { key: string; html: string } {
  if (c.ui.step) return c.ui.step(i, mod, c, lastOp);
  const op = c.ops[i];
  return { key: op.op, html: `${op.op} ${dims(c.pass.trace.shapes[lastOp])}` };
}

function stepHTML(step: Step, c: Ctx): string {
  if (step.dir === 'bwd') {
    const m = step.mod;
    return `<span class="tag" style="color:${MOD_COLOR[m.type]}">${m.title}</span> ` + (c.ui.back?.(m, c, step.ops) ?? `Gradient through ${m.title.toLowerCase()}.`);
  }
  const anchor = step.ops.find((i) => describe(i, step.mod, c).key !== 'cont') ?? step.ops[0];
  return describe(anchor, step.mod, c, step.ops[step.ops.length - 1]).html;
}

export class Article {
  private stepEls: HTMLElement[] = [];
  private active = -1;
  private userScroll = 0;
  onSeek: ((step: number) => void) | null = null;

  constructor(readonly el: HTMLElement) {
    el.addEventListener('wheel', () => (this.userScroll = performance.now()), { passive: true });
    el.addEventListener('click', (e) => {
      const p = (e.target as HTMLElement).closest('[data-step]') as HTMLElement | null;
      if (p) this.onSeek?.(+p.dataset.step!);
    });
  }

  render(steps: Step[], ctx: Ctx, epilogue = '') {
    const out: string[] = [];
    let section: Mod | null = null, dir = 'fwd';
    steps.forEach((s, k) => {
      if (s.dir !== dir) {
        dir = s.dir;
        out.push(`<h2 class="sec bwd">${ctx.ui.terms?.backwardTitle ?? 'Backward pass: forces'}</h2><div class="lead">${ctx.ui.narration('backward', ctx.meta)}</div>`);
        section = null;
      }
      const head = s.dir === 'fwd' ? sectionOf(s.mod) : null;
      if (s.dir === 'fwd' && head !== section) {
        section = head;
        const crumbs = ancestors(s.mod).slice(1);
        const color = MOD_COLOR[s.mod.type];
        out.push(`<h2 class="sec" style="--c:${color}">${crumbs.map((m, i) => i < crumbs.length - 1 ? `<span>${m.title}</span> › ` : m.title).join('')}</h2>`);
        const lead = ctx.ui.narration(s.mod, ctx.meta);
        if (lead) out.push(`<div class="lead">${lead}</div>`);
      }
      const color = MOD_COLOR[s.mod.type];
      let html = '';
      try { html = stepHTML(s, ctx); } catch (e) { html = (e as Error).message; }
      out.push(`<p class="step ${s.dir}" data-step="${k}" style="--c:${color}">${html}</p>`);
    });
    if (epilogue) out.push(`<h2 class="sec" style="--c:#a3e635">Forces</h2><div class="lead">${epilogue}</div>`);
    this.el.innerHTML = out.join('');
    this.stepEls = [...this.el.querySelectorAll<HTMLElement>('[data-step]')];
    const a = this.active;
    this.active = -1;
    if (a >= 0) this.setActive(a, false);
  }

  setActive(k: number, scroll = true) {
    if (k === this.active) return;
    this.stepEls[this.active]?.classList.remove('on');
    this.active = k;
    const el = this.stepEls[k];
    if (!el) return;
    el.classList.add('on');
    if (scroll && performance.now() - this.userScroll > 2500) el.scrollIntoView({ block: 'center', behavior: 'smooth' });
  }
}

/** Sections are the innermost modules, except heads, which share their readout section. */
const sectionOf = (m: Mod) => (m.type === 'head' || m.type === 'mlp' && m.parent?.type === 'network' ? m.parent ?? m : m);

/** The closing paragraph: what the forces were and how they were obtained. */
export function epilogue(pass: Pass, hasNC: boolean): string {
  if (!pass.forces && !pass.ncForces) return '';
  const a = pass.trace.selected, at = `<b class="atom">${SYMBOLS[pass.numbers[a]]}${a}</b>`;
  const n3 = (F: Float32Array, i: number) => Math.hypot(F[3 * i], F[3 * i + 1], F[3 * i + 2]);
  const ms = pass.trace.ms;
  if (pass.mode === 'conservative') return `<p>These forces are exact derivatives of <i>E</i>, so they are conservative by construction.
    ${hasNC ? 'This model can also predict forces directly, skipping the backward pass: choose <b>Forces → direct</b> or <b>both</b> above.'
            : ''}</p>`;
  const F = pass.ncForces!;
  if (pass.mode === 'direct') return `<p>No backward pass this time: the forces (lime arrows) came straight out of the direct head, at the
    cost of the forward pass alone (${ms.forward.toFixed(1)} ms here). ${at} gets |<b>F</b>| = <b>${n3(F, a).toFixed(3)} eV/Å</b>.
    Choose <b>both</b> to compare them with −∂<i>E</i>/∂<b>r</b>.</p>`;
  const C = pass.forces!;
  let s2 = 0;
  for (let k = 0; k < C.length; k++) s2 += (C[k] - F[k]) ** 2;
  const dot = [0, 1, 2].reduce((s, k) => s + C[3 * a + k] * F[3 * a + k], 0);
  const ang = (Math.acos(Math.max(-1, Math.min(1, dot / (n3(C, a) * n3(F, a) || 1)))) * 180) / Math.PI;
  return `<p>Both: conservative forces (violet) from the backward pass, and direct forces (lime) from the head. For this structure
    they differ by <b>${Math.sqrt(s2 / C.length).toFixed(3)} eV/Å</b> RMS. At ${at}, |<b>F</b>| = ${n3(C, a).toFixed(3)} against
    |<b>F</b><sub>direct</sub>| = ${n3(F, a).toFixed(3)} eV/Å, ${ang.toFixed(0)}° apart. The backward pass took ${ms.backward.toFixed(1)} ms,
    compared with ${ms.forward.toFixed(1)} ms for the forward pass.</p>`;
}
