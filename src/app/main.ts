import './style.css';
import { SYMBOLS } from '../common/elements';
import type { System } from '../common/structure';
import type { ModelKind } from '../models/types';
import { type ModelUI, NN_TERMS } from '../models/ui';
import { UIS } from '../models/uis';
import { MoleculeView } from '../viz/molecule';
import { Article, buildSteps, epilogue, fmt, type Step } from '../viz/article';
import type { Thumb } from '../engine/backend';
import { eV, fmtUnit, times, type Unit } from '../engine/units';
import { Diagram } from '../viz/diagram';
import { ancestors, MOD_COLOR, walk } from '../viz/modules';
import { type Hit, NetworkView } from '../viz/network';
import { Timeline } from '../viz/timeline';
import type { ForceMode, FromWorker, OpInfo, Pass, ToWorker } from '../worker/protocol';
import { PRESETS } from './presets';
import { parseXYZ } from './xyz';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

// ------------------------------------------------------------------ engine
const worker = new Worker(new URL('../worker/engine.worker.ts', import.meta.url), { type: 'module' });
const send = (m: ToWorker, t: Transferable[] = []) => worker.postMessage(m, t);

let meta: any = null;
let ui: ModelUI | null = null;
const terms = () => ui?.terms ?? NN_TERMS;
let ops: OpInfo[] = [];
let pass: Pass | null = null;
let system: System = PRESETS.ethanol;
let selected = 0;
let busy = false, queued = false;
let backend = '';
let autoplay = true; // start playing when the next pass arrives

function evaluate() {
  if (!meta) return;
  if (busy) { queued = true; return; }
  busy = true;
  send({ type: 'evaluate', system, selected, mode: $<HTMLSelectElement>('force-mode').value as ForceMode });
}

// ------------------------------------------------------------------ views
const net = new NetworkView($('net'));
const graph = new MoleculeView($('molecule'));
const timeline = new Timeline();
graph.onSelect = (atom) => { selected = atom; evaluate(); };

const diagram = new Diagram($('diagram'));
let hover: Hit | null = null;

/** Article steps, and for every timeline position the step it belongs to. */
const article = new Article($('article'));
let steps: Step[] = [], stepAt = new Int32Array(0), stepStart: number[] = [], stepEnd: number[] = [];
function buildChapters() {
  steps = buildSteps(ops, net.leaf, ctx());
  stepAt = new Int32Array(timeline.steps.length + 1);
  stepStart = []; stepEnd = [];
  steps.forEach((st, k) => {
    const idx = st.ops.map((op) => timeline.indexOf(op, st.dir));
    stepStart[k] = Math.min(...idx); stepEnd[k] = Math.max(...idx) + 1;
    for (const i of idx) stepAt[i] = k;
  });
  stepAt[timeline.steps.length] = steps.length - 1;
  drawDiagram();
}
let hasNC = false;
const ctx = () => ({ ops, pass: pass!, meta, ui: ui! });
const renderArticle = () => pass && steps.length && article.render(steps, ctx(), epilogue(pass, hasNC));
article.onSeek = (k) => { timeline.seek(stepStart[k]); timeline.stopAt = stepEnd[k]; timeline.playing = true; follow(); refresh(true); };
const follow = () => { net.follow = true; $('follow').classList.add('on'); };
/** Play to the end of the current step (or the next one, at a boundary). */
function continueStep() {
  if (!steps.length) return;
  if (timeline.done) timeline.seek(0);
  const k = stepAt[Math.min(timeline.state().k, timeline.steps.length)];
  timeline.stopAt = stepEnd[k] > timeline.t + 1e-6 ? stepEnd[k] : stepEnd[Math.min(k + 1, steps.length - 1)];
  timeline.playing = true;
  follow();
  refresh();
}
const drawDiagram = () => net.root && ui && diagram.build(net.root, (m) => ui!.subtitle(m, meta), net.collapsed);
diagram.onNavigate = (m) => {
  timeline.playing = false;
  timeline.seek(m.first);
  net.focusModule(m.id);
  $('follow').classList.add('on');
  refresh(true);
};
diagram.onToggle = (m) => net.toggle(m.id);
net.onCollapse = () => { drawDiagram(); refresh(true); };
new ResizeObserver(() => drawDiagram()).observe($('diagram'));

let lastActive = -2, lastStep = -1;
function refresh(force = false) {
  if (!ops.length || !pass) return;
  const s = timeline.state();
  net.setProgress(s.fwd, s.bwd, s.active, s.dir);
  diagram.update(s.fwd, s.bwd, s.active);
  const k = stepAt[Math.min(s.k, timeline.steps.length)] ?? 0;
  if (k !== lastStep || force) {
    lastStep = k;
    article.setActive(k);
    const st = steps[k];
    const crumbs = st ? ancestors(st.mod).slice(1) : [];
    $('chapter-title').innerHTML = crumbs.map((a) => `<span style="color:${MOD_COLOR[a.type]}">${a.short}</span>`).join('<i> › </i>');
    $('chapter-phase').textContent = st?.dir === 'bwd' ? terms().backward : terms().forward;
    $('chapter-phase').className = st?.dir === 'bwd' ? 'bwd' : '';
  }
  if (s.active !== lastActive || force) {
    lastActive = s.active;
    if (s.active >= 0 && net.follow && !zen) net.focus(s.active);
    drawGraph(s.active, s.dir);
  }
  drawTimeline();
  if (zen) {
    const n = timeline.steps.length || 1, fe = timeline.forwardEnd, t = timeline.t;
    ($('zen-progress').querySelector('.f') as HTMLElement).style.width = `${(100 * Math.min(t, fe)) / n}%`;
    const b = $('zen-progress').querySelector('.b') as HTMLElement;
    b.style.left = `${(100 * fe) / n}%`;
    b.style.width = `${(100 * Math.max(0, t - fe)) / n}%`;
  }
  $('play').textContent = timeline.playing && timeline.stopAt === null ? '⏸' : '▶';
}

function drawGraph(active: number, dir: 'fwd' | 'bwd') {
  if (!pass) return;
  const op = active >= 0 ? ops[active] : null;
  const attention = op?.op === 'attention' ? pass.trace.attention.find((a) => a.op === active) ?? null : null;
  const done = timeline.done;
  const h = hover && hover.kind === 'op' ? rowTarget(hover.index, hover.row) : null;
  graph.render({ op: done ? -1 : active, dir, forces: done || (dir === 'bwd' && timeline.t >= timeline.steps.length - 1), attention, hover: h });
  const space = op?.kind ? pass.trace.rows[op.kind] : undefined;
  const what = !op || done ? `<b>Forces</b> on every atom (arrows), atoms by element. Click an atom to follow it through the network.`
    : attention ? `<b>Attention</b> of atom ${label(pass.trace.selected)} to its neighbours (mean over heads): the selected atom's edges are coloured by weight.`
    : space ? `<b>${opTitle(active)}</b>: ${dir === 'bwd' ? 'gradient' : 'value'} magnitude (RMS over features) of every ${space.label}${space.atom || space.edge ? '' : ', averaged onto its atom'}.`
    : `<b>${opTitle(active)}</b> has no per-atom or per-edge rows.`;
  $('graph-caption').innerHTML = what;
}

const opTitle = (i: number) => `${ops[i].op}${ops[i].params[0] ? ' · ' + ops[i].params[0].replace(/\.weight$/, '') : ''}`;
const label = (i: number) => `${SYMBOLS[pass!.numbers[i]]}${i}`;

/** What a row of an op's block stands for in the graph. */
function rowTarget(op: number, row: number): { atom?: number; edge?: number; owner?: number } | null {
  const sel = pass!.trace.rowSel[op], kind = ops[op].kind;
  const space = kind ? pass!.trace.rows[kind] : undefined;
  if (!sel || !space) return null;
  const r = sel[row];
  if (r === undefined) return null;
  const atom = space.atom?.[r] ?? -1, edge = space.edge?.[r] ?? -1;
  if (atom >= 0) return { atom };
  if (edge >= 0) return { edge };
  return { owner: space.owner[r] };
}

function rowLabel(op: number, row: number) {
  const t = rowTarget(op, row), g = pass!.trace.graph, kind = ops[op].kind!;
  if (!t) return pass!.trace.rowSel[op] ? '' : `rows averaged in blocks`;
  const what = pass!.trace.rows[kind].label;
  if (t.atom !== undefined) return `${what === 'atom' ? 'atom' : `${what} of atom`} ${label(t.atom)}`;
  if (t.edge !== undefined) return `${what} ${label(g.center[t.edge])} → ${label(g.neighbor[t.edge])}`;
  return `${what} ${row + 1} of atom ${label(t.owner!)}`;
}

/** ∂E/∂x is in eV per unit of x. */
const gradUnit = (u?: Unit | null) => (u ? fmtUnit(times(eV, u, -1)) : '');
const withUnit = (x: number, u: string) => `<span class="v">${x.toPrecision(4)}</span>${u ? ` <span class="k">${u}</span>` : ''}`;
/** Zen's hover card: what the matrix is, and its statistics (over the full tensor, not the thumbnail). */
function summary(h: Hit): string {
  const row = (name: string, t: Pick<Thumb, 'mean' | 'std' | 'min' | 'max'>, u: string, cls = 'v') =>
    `<tr><td class="k">${name}</td>${[t.mean, t.std, t.min, t.max].map((x) => `<td class="${cls}">${fmt(x, 3)}</td>`).join('')}<td class="k">${u}</td></tr>`;
  const table = (rows: string[]) => `<table class="stats"><tr class="k"><td></td><td>mean</td><td>std</td><td>min</td><td>max</td><td></td></tr>${rows.join('')}</table>`;
  if (h.kind === 'op') {
    const op = ops[h.index], v = pass!.trace.values[h.index]!, g = pass!.trace.grads[h.index], u = fmtUnit(op.unit);
    return `<b>${opTitle(h.index)}</b> <span class="k">[${pass!.trace.shapes[h.index].join('×')}]${u ? ` · in ${u}` : ''}</span><br>
      <span class="k">${op.scope || 'input'}</span>${table([row('value', v, u), ...(g ? [row('∂E/∂', g, gradUnit(op.unit), 'gv')] : [])])}`;
  }
  if (h.kind === 'weight') {
    const name = ops[h.index].params[h.sub], p = pass!.trace.params![name];
    return `<b>${name}</b> <span class="k">[${(pass!.trace.paramShapes?.[name] ?? [p.value.rows, p.value.cols]).join('×')}] · weights</span>
      ${table([row('value', p.value, ''), ...(p.grad ? [row('∂E/∂', p.grad, '', 'gv')] : [])])}`;
  }
  const a = pass!.trace.attention.find((x) => x.op === h.index)!, w = a.probs.subarray(h.sub * a.n * a.n, (h.sub + 1) * a.n * a.n);
  const mean = w.reduce((s, x) => s + x, 0) / w.length, std = Math.sqrt(w.reduce((s, x) => s + (x - mean) ** 2, 0) / w.length);
  return `<b>attention, head ${h.sub + 1}</b> <span class="k">[${a.n}×${a.n}] · around ${label(a.atoms[0])}</span>
    ${table([row('weight', { mean, std, min: Math.min(...w), max: Math.max(...w) }, '')])}`;
}

net.onHover = (h) => {
  hover = h;
  const tip = $('tooltip');
  if (!h || !pass) { tip.hidden = true; net.highlightRow(-1, -1); drawGraph(lastActive, timeline.state().dir); return; }
  let html = '';
  if (zen) html = summary(h);
  else if (h.kind === 'op') {
    const op = ops[h.index], shape = pass.trace.shapes[h.index], v = pass.trace.values[h.index]!;
    const C = shape.length > 1 ? shape[shape.length - 1] : 1;
    const c0 = Math.floor((h.col * C) / v.cols), c1 = Math.max(c0 + 1, Math.floor(((h.col + 1) * C) / v.cols));
    html = `<b>${opTitle(h.index)}</b> <span class="k">[${shape.join('×')}]</span><br>
      <span class="k">${op.scope || 'input'}</span><br>
      ${rowLabel(h.index, h.row)} · ${c1 - c0 > 1 ? `features ${c0}–${c1 - 1} (mean)` : `feature ${c0}`}<br>
      value ${withUnit(h.value, fmtUnit(op.unit))}${h.grad !== undefined ? ` · ∂E/∂ <span class="gv">${h.grad.toPrecision(4)}</span>${gradUnit(op.unit) ? ` <span class="k">${gradUnit(op.unit)}</span>` : ''}` : ''}`;
    net.highlightRow(h.index, h.row);
  } else if (h.kind === 'weight') {
    const name = ops[h.index].params[h.sub], p = pass.trace.params![name].value;
    html = `<b>${name}</b><br><span class="k">weights, shown ${p.rows}×${p.cols}${p.rows * p.cols < 64 * 64 ? '' : ' (block means)'}</span><br>value <span class="v">${h.value.toPrecision(4)}</span>`;
  } else {
    const a = pass.trace.attention.find((x) => x.op === h.index)!;
    html = `<b>attention, head ${h.sub + 1}</b><br>query ${label(a.atoms[h.row])}${h.row ? '' : ' (self)'} → key ${label(a.atoms[h.col])}${h.col ? '' : ' (self)'}<br>
      weight <span class="v">${h.value.toFixed(3)}</span>`;
  }
  tip.innerHTML = html;
  tip.hidden = false;
  drawGraph(lastActive, timeline.state().dir);
};
$('net').addEventListener('pointermove', (e) => {
  const r = $('net').getBoundingClientRect(), tip = $('tooltip');
  tip.style.left = `${Math.max(4, Math.min(e.clientX - r.left + 14, r.width - tip.offsetWidth - 8))}px`;
  const y = e.clientY - r.top; // below the cursor, or above it when there is no room
  tip.style.top = `${y + 14 + tip.offsetHeight > r.height ? Math.max(4, y - 10 - tip.offsetHeight) : y + 14}px`;
});
net.onClick = (h) => {
  if (!h || h.kind !== 'op') return;
  const t = rowTarget(h.index, h.row);
  const a = t?.atom ?? t?.owner;
  if (a !== undefined && a >= 0 && a !== selected) { selected = a; evaluate(); }
};
net.onUserCamera = () => $('follow').classList.remove('on');

net.onFrame = (dt) => {
  if (zen && scrubTo !== null) {
    // ease towards the scrolled-to position, then hand back to the loop after a pause
    timeline.seek(timeline.t + (scrubTo - timeline.t) * (1 - Math.exp(-dt * 10)));
    if (Math.abs(scrubTo - timeline.t) < 1e-3) timeline.seek(scrubTo);
    refresh();
    if (performance.now() - lastScrub > 5000) { scrubTo = null; timeline.playing = true; }
    return;
  }
  if (zen && timeline.done) {
    // hold the finished pass for a moment, then go again
    zenPause += dt;
    if (zenPause > 1.5) { zenPause = 0; timeline.seek(0); timeline.playing = true; }
  }
  if (timeline.advance(dt)) refresh();
};

// ------------------------------------------------------------------ transport
const speedOf = (x: number) => Math.round(Math.exp(Math.log(1) + x * (Math.log(120) - Math.log(1))) * 10) / 10;
const setSpeed = () => {
  timeline.speed = speedOf(+$<HTMLInputElement>('speed').value);
  $('speed-val').textContent = `${timeline.speed} op/s`;
};
$('speed').addEventListener('input', setSpeed);
setSpeed();
const play = () => {
  if (timeline.done) timeline.seek(0);
  const free = timeline.playing && timeline.stopAt === null;
  timeline.stopAt = null;
  timeline.playing = !free;
  if (!free) follow();
  refresh();
};
$('continue').onclick = continueStep;
$('play').onclick = play;
$('restart').onclick = () => { timeline.seek(0); refresh(true); };
$('end').onclick = () => { timeline.playing = false; timeline.seek(timeline.steps.length); refresh(true); };
$('prev').onclick = () => { timeline.playing = false; timeline.seek(Math.ceil(timeline.t) - 1); refresh(); };
$('next').onclick = () => { timeline.playing = false; timeline.seek(Math.floor(timeline.t) + 1); refresh(); };
$('follow').onclick = () => { net.follow = !net.follow; $('follow').classList.toggle('on', net.follow); if (net.follow && lastActive >= 0) net.focus(lastActive); };
$('overview').onclick = () => { net.overview(); $('follow').classList.remove('on'); };

// zen: full screen, no text, the pass on a loop
let zen = false, zenPause = 0, speedBeforeZen = 6;
function setZen(on: boolean) {
  zen = on;
  scrubTo = null;
  document.body.classList.toggle('zen', on);
  $('zen').classList.toggle('on', on);
  net.follow = true;
  $('follow').classList.add('on');
  if (on) { speedBeforeZen = timeline.speed; timeline.speed = Math.max(timeline.speed, 24); if (timeline.done) timeline.seek(0); timeline.stopAt = null; timeline.playing = true; }
  else { timeline.speed = speedBeforeZen; }
  net.setZen(on);
}
$('zen').onclick = () => setZen(!zen);
function setWeightsOnly(on: boolean) {
  $('weights-only').classList.toggle('on', on);
  document.body.classList.toggle('weights-only', on);
  if (on) { timeline.playing = false; net.follow = false; $('follow').classList.remove('on'); }
  net.setWeightsOnly(on);
  if (!on) { follow(); if (lastActive >= 0) net.focus(lastActive); }
}
$('weights-only').onclick = () => setWeightsOnly(!net.weightsOnly);

// in zen the wheel scrubs through the pass (ctrl + wheel, i.e. a pinch, still zooms)
let scrubTo: number | null = null, lastScrub = 0;
$('net').addEventListener('wheel', (e) => {
  if (!zen || e.ctrlKey) return;
  e.preventDefault();
  e.stopPropagation(); // keep it from the camera controls
  const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1;
  scrubTo = Math.max(0, Math.min((scrubTo ?? timeline.t) + (e.deltaY * unit) / 40, timeline.steps.length));
  lastScrub = performance.now();
  timeline.playing = false;
  zenPause = 0;
}, { capture: true, passive: false });
window.addEventListener('keydown', (e: KeyboardEvent) => {
  if ((e.target as HTMLElement).tagName === 'INPUT' || (e.target as HTMLElement).tagName === 'SELECT') return;
  if (e.key === 'Escape' && zen) setZen(false);
  if (zen && e.key === ' ') { e.preventDefault(); scrubTo = null; timeline.playing = !timeline.playing; return; }
  if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); if (timeline.playing) { timeline.playing = false; refresh(); } else continueStep(); }
  if (e.key === 'ArrowRight') $('next').click();
  if (e.key === 'ArrowLeft') $('prev').click();
  if (e.key === 'Home') $('restart').click();
  if (e.key === 'End') $('end').click();
});

const tl = $<HTMLCanvasElement>('timeline');
function drawTimeline() {
  const dpr = devicePixelRatio, w = tl.clientWidth, h = tl.clientHeight;
  if (tl.width !== w * dpr) { tl.width = w * dpr; tl.height = h * dpr; }
  const g = tl.getContext('2d')!;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const n = timeline.steps.length;
  if (!n) return;
  const x = (k: number) => (k / n) * w;
  const fe = timeline.forwardEnd;
  g.fillStyle = '#1a1e28'; g.fillRect(0, 10, w, 14);
  g.fillStyle = '#7a4a12'; g.fillRect(0, 10, x(Math.min(timeline.t, fe)), 14);
  if (timeline.t > fe) { g.fillStyle = '#4c2a78'; g.fillRect(x(fe), 10, x(timeline.t) - x(fe), 14); }
  g.fillStyle = '#3a4256';
  for (const k of stepStart) g.fillRect(x(k), 8, 1, 18);
  g.fillStyle = '#fff'; g.fillRect(x(timeline.t) - 1, 4, 2, 26);
  g.fillStyle = '#8a93a6'; g.font = '10px system-ui';
  g.fillText(terms().forward, 4, 8); g.fillText(terms().backward, x(fe) + 4, 8);
  const s = timeline.state();
  $('scrub-label').textContent = s.active >= 0 ? `${s.dir === 'fwd' ? '→' : '←'} op ${s.active + 1}/${ops.length} · ${ops[s.active].scope || 'input'} · ${ops[s.active].op}` : 'pass complete';
}
let dragging = false;
const scrub = (e: PointerEvent) => {
  const r = tl.getBoundingClientRect();
  timeline.seek(((e.clientX - r.left) / r.width) * timeline.steps.length);
  refresh();
};
tl.addEventListener('pointerdown', (e) => {
  dragging = true;
  timeline.playing = false;
  tl.setPointerCapture(e.pointerId);
  follow(); // scrubbing the timeline brings the camera back to the op it lands on
  lastActive = -2;
  scrub(e);
});
tl.addEventListener('pointermove', (e) => dragging && scrub(e));
tl.addEventListener('pointerup', () => (dragging = false));

// ------------------------------------------------------------------ loading
// While a model downloads, builds or (KRR) fits, the network gets an overlay and stops rendering
// until the new model's first pass is in.
const loader = {
  el: $('loading'),
  waitingForPass: false,
  show(title: string) {
    const grid = this.el.querySelector('.loader-grid')!;
    if (!grid.childElementCount)
      grid.innerHTML = Array.from({ length: 6 * 16 }, (_, k) => `<i style="--d:${(((k % 16) + Math.floor(k / 16)) * 0.06).toFixed(2)}s"></i>`).join('');
    (this.el.querySelector('.loader-title') as HTMLElement).textContent = title;
    this.stage('');
    this.el.hidden = false;
    timeline.playing = false;
    net.paused = true;
  },
  /** The current stage; without a fraction the bar is indeterminate. */
  stage(text: string, fraction?: number) {
    (this.el.querySelector('.loader-stage') as HTMLElement).textContent = text;
    const bar = this.el.querySelector('.loader-bar') as HTMLElement;
    bar.classList.toggle('indeterminate', fraction == null);
    (bar.firstElementChild as HTMLElement).style.width = fraction == null ? '' : `${Math.round(100 * Math.min(Math.max(fraction, 0), 1))}%`;
  },
  hide() {
    this.waitingForPass = false;
    this.el.hidden = true;
    net.paused = false;
  },
};

/** Fetch a file, reporting the fraction downloaded when the server says how big it is. */
async function fetchBytes(url: string, progress: (fraction: number, mb: string) => void): Promise<ArrayBuffer> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const total = +(r.headers.get('content-length') ?? 0);
  if (!r.body || !total) return r.arrayBuffer();
  const reader = r.body.getReader(), chunks: Uint8Array[] = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    progress(got / total, `${(got / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`);
  }
  const out = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out.buffer;
}

// ------------------------------------------------------------------ inputs
const structSel = $<HTMLSelectElement>('structure');
structSel.innerHTML = Object.entries(PRESETS).map(([k, s]) => `<option value="${k}">${s.label}</option>`).join('');
structSel.value = 'ethanol';
const setSystem = (s: System, restart = true) => {
  system = s;
  selected = Math.min(selected, s.numbers.length - 1);
  if (restart) { timeline.seek(0); autoplay = true; }
  // KRR is fitted to the structure: a new structure means a new fit (a rattle does not)
  if (restart && active?.kind === 'krr') loadModel(active);
  else evaluate();
};
$<HTMLSelectElement>('force-mode').onchange = () => { autoplay = true; timeline.seek(0); evaluate(); };
structSel.onchange = () => { selected = 0; setSystem(PRESETS[structSel.value]); };
$<HTMLInputElement>('xyz-file').onchange = async (e) => {
  const f = (e.target as HTMLInputElement).files?.[0];
  if (!f) return;
  try { selected = 0; setSystem(parseXYZ(await f.text())); structSel.value = ''; }
  catch (err) { status(`<b>Could not read ${f.name}:</b> ${(err as Error).message}`); }
};
$('rattle').onclick = () => {
  setSystem({ ...system, positions: system.positions.map((p) => p.map((x) => x + 0.06 * (Math.random() * 2 - 1))) }, false);
};

// models/index.json: [{ name, kind, label, meta, weights, params }]. KRR has no files: it is fitted
// here, on the current structure, against a teacher model.
interface ModelEntry { name: string; kind: ModelKind; label?: string; meta?: string; weights?: string; params?: any }
const kindSel = $<HTMLSelectElement>('kind'), modelSel = $<HTMLSelectElement>('model');
const KIND_ORDER: ModelKind[] = ['pet', 'mace', 'lorem', 'ani', 'physnet', 'krr'];
let fillVariants = () => {};
let entries: ModelEntry[] = [];
const loaded = new Set<string>(); // model ids the worker holds
const waiting = new Map<string, () => void>(); // model id -> resolve, for loads we await
let active: ModelEntry | null = null;

function loadModel(e: ModelEntry, meta0?: any, weights0?: ArrayBuffer, activate = true): Promise<void> {
  return (async () => {
    const name = e.label ?? e.name;
    try {
      if (e.kind === 'krr') {
        // KRR's 'weights' are a fit: the second menu says what to fit it to
        const t = entries.find((x) => x.name === modelSel.value && x.kind !== 'krr') ?? entries.find((x) => x.kind !== 'krr')!;
        if (activate) loader.show(`Fitting ${name} to ${t.label ?? t.name}`);
        if (!loaded.has(t.name)) await loadModel(t, undefined, undefined, false);
        status(`fitting ${name} to ${t.label ?? t.name} on this structure…`);
        loader.stage('Labelling rattled copies of this structure with the teacher…');
        meta0 = { kind: 'krr', ...e.params, teacher: t.name, teacherLabel: t.label ?? t.name, system };
      } else {
        status(`loading ${name}…`);
        if (activate) loader.show(`Loading ${name}`);
      }
      const m = meta0 ?? (e.meta ? await getJSON(stem(e.meta)) : {});
      const w = weights0 ?? (e.weights ? await fetchBytes(stem(e.weights), (f, mb) => loader.stage(`Downloading the ${activate ? '' : 'teacher\u2019s '}weights of ${name}: ${mb}`, f)) : null);
      if (w) loader.stage(activate ? `Building ${name}…` : `Building ${name} (teacher)…`);
      const done = new Promise<void>((r) => waiting.set(e.name, r));
      if (activate) active = e;
      send({ type: 'loadModel', id: e.name, kind: e.kind, meta: m, weights: w, label: name, activate }, w ? [w] : []);
      await done;
    } catch (err) {
      loader.hide();
      status(`<b>Could not load ${name}:</b> ${(err as Error).message}`);
      if (!activate) throw err; // a teacher that failed: the fit cannot go on
    }
  })();
}
// Model files come from public/models/ when it has them (local development), otherwise from the
// Hugging Face repository that scripts/publish_weights.py fills; ?models=<base url> picks another.
const HF_MODELS = 'https://huggingface.co/EricBoi/mlip-visualization-models/resolve/main/';
let base = 'models/';
const getJSON = async (u: string) => { const r = await fetch(u); if (!r.ok) throw new Error(`${u}: HTTP ${r.status}`); return r.json(); };
async function listModels() {
  const params = new URLSearchParams(location.search), custom = params.get('models');
  for (const b of custom ? [custom.replace(/\/?$/, '/')] : ['models/', HF_MODELS]) {
    try {
      const list: ModelEntry[] = await getJSON(`${b}index.json`);
      const probe = list.find((x) => x.meta);
      if (probe) await getJSON(b + probe.meta); // index.json is in git; the weights may not be
      [entries, base] = [list, b];
      break;
    } catch { entries = []; }
  }
  if (!entries.length) status('<b>No models found</b> locally or on Hugging Face. Convert some (see <code>public/models/README.md</code>) or open a pair of model files.');
  const q = params.get('model');
  if (q) {
    // ?model=<url stem>: a PET model at <stem>.json and <stem>.safetensors (e.g. a HuggingFace repo)
    entries.push({ name: q, kind: 'pet', label: q.split('/').pop(), meta: `${q}.json`, weights: `${q}.safetensors` });
  }
  // first menu: the kind of model; second: its weights (for KRR: what it is fitted to)
  const kinds = KIND_ORDER.filter((k) => entries.some((m) => m.kind === k));
  kindSel.innerHTML = kinds.map((k) => `<option value="${k}">${UIS[k]?.typeName ?? UIS[k]?.name ?? k} · ${UIS[k]?.family ?? ''}</option>`).join('');
  fillVariants = () => {
    const k = kindSel.value as ModelKind;
    const opts = k === 'krr'
      ? entries.filter((m) => m.kind !== 'krr').map((m) => [m.name, `fitted to ${m.label ?? m.name}`])
      : entries.filter((m) => m.kind === k).map((m) => [m.name, m.label ?? m.name]);
    modelSel.innerHTML = opts.map(([v, l]) => `<option value="${v}">${l}</option>`).join('');
    $('variant-label').textContent = k === 'krr' ? 'Fitted to' : 'Weights';
  };
  const current = () => {
    const k = kindSel.value as ModelKind;
    return k === 'krr' ? entries.find((m) => m.kind === 'krr')! : entries.find((m) => m.name === modelSel.value)!;
  };
  kindSel.onchange = () => { fillVariants(); loadModel(current()); };
  modelSel.onchange = () => loadModel(current());
  if (q) { kindSel.value = 'pet'; fillVariants(); modelSel.value = q; } else fillVariants();
  if (entries.length) loadModel(current());
}
const stem = (name: string) => (/^https?:/.test(name) ? name : base + name);
$<HTMLInputElement>('model-files').onchange = async (e) => {
  const files = [...((e.target as HTMLInputElement).files ?? [])];
  const j = files.find((f) => f.name.endsWith('.json')), w = files.find((f) => f.name.endsWith('.safetensors'));
  if (!j || !w) { status('<b>Pick both files</b> of a converted model: <code>name.json</code> and <code>name.safetensors</code>'); return; }
  const meta1 = JSON.parse(await j.text());
  const kind: ModelKind = meta1.architecture === 'pet' ? 'pet' : meta1.kind ?? 'pet';
  const entry: ModelEntry = { name: j.name.replace(/\.json$/, ''), kind, label: j.name.replace(/\.json$/, '') };
  if (!entries.some((x) => x.name === entry.name)) entries.push(entry);
  kindSel.value = kind;
  fillVariants();
  modelSel.value = entry.name;
  await loadModel(entry, meta1, await w.arrayBuffer());
};

function status(html: string) { $('status').innerHTML = html; }

// ------------------------------------------------------------------ messages
worker.onmessage = (e: MessageEvent<FromWorker>) => {
  const m = e.data;
  if (m.type === 'ready') {
    backend = m.backend === 'webgpu' ? `WebGPU · ${m.adapter}` : m.adapter;
    status(backend);
    listModels();
  } else if (m.type === 'model') {
    loaded.add(m.id);
    waiting.get(m.id)?.();
    waiting.delete(m.id);
    if (!m.activate) return;
    if (!loader.el.hidden) { loader.waitingForPass = true; loader.stage(`Running the first ${UIS[m.kind]?.terms?.forward ?? 'pass'}…`); }
    meta = m.meta;
    hasNC = m.hasNC;
    const u = (ui = UIS[m.kind]!);
    net.describe = u.describe;
    net.rootTitle = u.name;
    net.collapsed.clear();
    $('zen-params').textContent = m.nParams.toLocaleString('en-US');
    const fm = $<HTMLSelectElement>('force-mode');
    for (const o of fm.options) if (o.value !== 'conservative') o.disabled = !m.hasNC;
    if (!m.hasNC) fm.value = 'conservative';
    fm.title = m.hasNC ? '' : 'This model has no direct-force head';
    $('intro-text').innerHTML = u.intro?.(m.meta) ?? `<h1>${u.name}, step by step</h1>`;
    $('model-card').innerHTML = u.card(m.meta, m.nParams, m.label);
    const t = terms();
    $('legend-weights').textContent = t.weights;
    $('legend-attn').hidden = !t.attention;
    $('weights-only').textContent = t.weights[0].toUpperCase() + t.weights.slice(1);
    $('weights-only').title = `Show only the ${t.weights} of the model`;
    // a new model starts in follow mode, whatever view the last one was in
    if (net.weightsOnly) setWeightsOnly(false);
    follow();
    timeline.seek(0);
    autoplay = true;
    // a model that cannot handle this structure's elements gets the first preset it can
    const fits = (sys: System) => sys.numbers.every((z) => m.elements.includes(z));
    const pref = u.preferredStructure;
    if (pref && PRESETS[pref] && system !== PRESETS[pref]) {
      system = PRESETS[pref]; selected = 0; structSel.value = pref;
    } else if (m.kind !== 'krr' && !fits(system)) {
      const k = Object.keys(PRESETS).find((key) => fits(PRESETS[key]));
      if (k) { system = PRESETS[k]; selected = 0; structSel.value = k; status(`switched to ${PRESETS[k].label}: ${u.name} has no parameters for this structure's elements`); }
    }
    evaluate();
  } else if (m.type === 'pass') {
    busy = false;
    const p = m.pass;
    const fresh = !!p.trace.topology;
    if (fresh) {
      ops = p.trace.topology!;
      timeline.reset(ops);
    }
    pass = p;
    selected = p.trace.selected;
    net.show(ops, p.trace);
    if (fresh && ui?.collapse && net.root) net.setCollapsed([...walk(net.root)].filter(ui.collapse).map((x) => x.id));
    if (fresh) buildChapters();
    renderArticle();
    graph.setPass(p, ops);
    const ms = p.trace.ms;
    status(`<b>E = ${p.energy.toFixed(4)} eV</b> · ${p.numbers.length} atoms, ${p.trace.graph.center.length} ${p.trace.graph.label} edges · ` +
      `${terms().forward} ${ms.forward.toFixed(1)} ms${p.forces ? ` · ${terms().backward} ${ms.backward.toFixed(1)} ms` : ` · no ${terms().backward}`}<br>${backend}`);
    const norm3 = (F: Float32Array, i: number) => Math.hypot(F[3 * i], F[3 * i + 1], F[3 * i + 2]);
    const a = selected, parts = [`selected ${label(a)}: ε = <b>${p.energies[a].toFixed(4)}</b> eV`];
    if (p.forces) parts.push(`<span class="fc">|F| = <b>${norm3(p.forces, a).toFixed(3)}</b></span>`);
    if (p.ncForces) parts.push(`<span class="fnc">|F<sub>direct</sub>| = <b>${norm3(p.ncForces, a).toFixed(3)}</b></span>`);
    if (p.forces && p.ncForces) {
      let s2 = 0;
      for (let k = 0; k < p.forces.length; k++) s2 += (p.forces[k] - p.ncForces[k]) ** 2;
      parts.push(`RMS(F − F<sub>direct</sub>) = <b>${Math.sqrt(s2 / p.forces.length).toFixed(3)}</b> eV/Å`);
    }
    $('energies').innerHTML = parts.join(' · ');
    lastActive = -2;
    if (loader.waitingForPass) loader.hide();
    if (autoplay) { autoplay = false; timeline.seek(0); continueStep(); }
    refresh(true);
    if (queued) { queued = false; evaluate(); }
  } else if (m.type === 'progress') {
    status(m.text);
    if (!loader.el.hidden) loader.stage(m.text.replace(/^KRR: /, ''), m.fraction);
  } else if (m.type === 'error') {
    busy = false;
    loader.hide();
    status(`<b>Error:</b> ${m.text}`);
    console.error(m.text);
  }
};
// handy from the console
(window as any).mlipviz = { net, graph, timeline, get pass() { return pass; }, get ops() { return ops; }, get steps() { return steps; } };

send({ type: 'init', backend: new URLSearchParams(location.search).get('backend') as any ?? 'auto' });
