import './style.css';
import { SYMBOLS } from '../common/elements';
import type { ModelKind } from '../models/types';
import type { ModelUI } from '../models/ui';
import { UIS } from '../models/uis';
import type { System } from '../common/structure';
import { GraphView } from '../viz/graph';
import { Diagram } from '../viz/diagram';
import { ancestors, type Mod, MOD_COLOR, walk } from '../viz/modules';
import { type Hit, NetworkView } from '../viz/network';
import { Timeline } from '../viz/timeline';
import type { FromWorker, OpInfo, Pass, ToWorker } from '../worker/protocol';
import { PRESETS } from './presets';
import { parseXYZ } from './xyz';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

// ------------------------------------------------------------------ engine
const worker = new Worker(new URL('../worker/engine.worker.ts', import.meta.url), { type: 'module' });
const send = (m: ToWorker, t: Transferable[] = []) => worker.postMessage(m, t);

let meta: any = null;
let ui: ModelUI | null = null;
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
  send({ type: 'evaluate', system, selected });
}

// ------------------------------------------------------------------ views
const net = new NetworkView($('net'));
const graph = new GraphView($('chemiscope'));
const timeline = new Timeline();
graph.onSelect = (atom) => { selected = atom; evaluate(); };

const diagram = new Diagram($('diagram'));
let hover: Hit | null = null;

/** Chapters: runs of ops in the same innermost module, then the backward pass. */
let chapters: { mod: Mod | null; first: number }[] = [];
function buildChapters() {
  chapters = [];
  net.leaf.forEach((m, i) => { if (!chapters.length || chapters[chapters.length - 1].mod !== m) chapters.push({ mod: m, first: i }); });
  chapters.push({ mod: null, first: timeline.forwardEnd });
  drawDiagram();
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

/** The chapter of the current timeline position. */
function currentChapter(k: number) {
  if (k >= timeline.forwardEnd) return chapters.length - 1;
  let c = 0;
  chapters.forEach((ch, i) => { if (ch.mod && ch.first <= k) c = i; });
  return c;
}

let lastActive = -2, lastChapter = -1;
function refresh(force = false) {
  if (!ops.length || !pass) return;
  const s = timeline.state();
  net.setProgress(s.fwd, s.bwd, s.active, s.dir);
  diagram.update(s.fwd, s.bwd, s.active);
  const k = Math.min(s.k, timeline.steps.length);
  const ci = currentChapter(k);
  if (ci !== lastChapter || force) {
    lastChapter = ci;
    const m = chapters[ci].mod;
    const crumbs = m ? ancestors(m).slice(1) : [];
    $('chapter-title').innerHTML = m
      ? crumbs.map((a, i) => `<span style="color:${MOD_COLOR[a.type]}">${i < crumbs.length - 1 ? a.short : a.title}</span>`).join('<i> › </i>')
      : 'Backward pass → forces';
    $('chapter-phase').textContent = m ? 'forward pass' : 'backward pass';
    $('chapter-phase').className = m ? '' : 'bwd';
    $('chapter-text').innerHTML = ui!.narration(m ?? 'backward', meta) || (m ? ui!.narration(m.parent ?? m, meta) : '');
  }
  if (s.active !== lastActive || force) {
    lastActive = s.active;
    if (s.active >= 0 && net.follow) net.focus(s.active);
    drawGraph(s.active, s.dir);
  }
  drawTimeline();
  $('play').textContent = timeline.playing ? '⏸' : '▶';
}

function drawGraph(active: number, dir: 'fwd' | 'bwd') {
  if (!pass) return;
  const op = active >= 0 ? ops[active] : null;
  const attention = op?.op === 'attention' ? pass.trace.attention.find((a) => a.op === active) ?? null : null;
  const done = timeline.done;
  const h = hover && hover.kind === 'op' ? rowTarget(hover.index, hover.row) : null;
  graph.render({ op: done ? -1 : active, dir, forces: done || (dir === 'bwd' && timeline.t >= timeline.steps.length - 1), attention, hover: h });
  const space = op?.kind ? pass.trace.rows[op.kind] : undefined;
  const what = !op || done ? `<b>Forces</b> −∂E/∂r on every atom (arrows), atoms by element. Click an atom to follow it through the network.`
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

net.onHover = (h) => {
  hover = h;
  const tip = $('tooltip');
  if (!h || !pass) { tip.hidden = true; net.highlightRow(-1, -1); drawGraph(lastActive, timeline.state().dir); return; }
  let html = '';
  if (h.kind === 'op') {
    const op = ops[h.index], shape = pass.trace.shapes[h.index], v = pass.trace.values[h.index]!;
    const C = shape.length > 1 ? shape[shape.length - 1] : 1;
    const c0 = Math.floor((h.col * C) / v.cols), c1 = Math.max(c0 + 1, Math.floor(((h.col + 1) * C) / v.cols));
    html = `<b>${opTitle(h.index)}</b> <span class="k">[${shape.join('×')}]</span><br>
      <span class="k">${op.scope || 'input'}</span><br>
      ${rowLabel(h.index, h.row)} · ${c1 - c0 > 1 ? `features ${c0}–${c1 - 1} (mean)` : `feature ${c0}`}<br>
      value <span class="v">${h.value.toPrecision(4)}</span>${h.grad !== undefined ? ` · ∂E/∂ <span class="gv">${h.grad.toPrecision(4)}</span>` : ''}`;
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
  tip.style.left = `${Math.min(e.clientX - r.left + 14, r.width - 340)}px`;
  tip.style.top = `${e.clientY - r.top + 14}px`;
});
net.onClick = (h) => {
  if (!h || h.kind !== 'op') return;
  const t = rowTarget(h.index, h.row);
  const a = t?.atom ?? t?.owner;
  if (a !== undefined && a >= 0 && a !== selected) { selected = a; evaluate(); }
};
net.onUserCamera = () => $('follow').classList.remove('on');

net.onFrame = (dt) => { if (timeline.advance(dt)) refresh(); };

// ------------------------------------------------------------------ transport
const speedOf = (x: number) => Math.round(Math.exp(Math.log(1) + x * (Math.log(120) - Math.log(1))) * 10) / 10;
const setSpeed = () => {
  timeline.speed = speedOf(+$<HTMLInputElement>('speed').value);
  $('speed-val').textContent = `${timeline.speed} op/s`;
};
$('speed').addEventListener('input', setSpeed);
setSpeed();
const play = () => { if (timeline.done) timeline.seek(0); timeline.playing = !timeline.playing; refresh(); };
$('play').onclick = play;
$('restart').onclick = () => { timeline.seek(0); refresh(true); };
$('end').onclick = () => { timeline.playing = false; timeline.seek(timeline.steps.length); refresh(true); };
$('prev').onclick = () => { timeline.playing = false; timeline.seek(Math.ceil(timeline.t) - 1); refresh(); };
$('next').onclick = () => { timeline.playing = false; timeline.seek(Math.floor(timeline.t) + 1); refresh(); };
$('follow').onclick = () => { net.follow = !net.follow; $('follow').classList.toggle('on', net.follow); if (net.follow && lastActive >= 0) net.focus(lastActive); };
$('overview').onclick = () => { net.overview(); $('follow').classList.remove('on'); };
window.addEventListener('keydown', (e: KeyboardEvent) => {
  if ((e.target as HTMLElement).tagName === 'INPUT' || (e.target as HTMLElement).tagName === 'SELECT') return;
  if (e.key === ' ') { e.preventDefault(); play(); }
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
  for (const c of chapters) { const cx = c.mod ? x(c.first) : x(fe); g.fillRect(cx, 6, 1, 22); }
  g.fillStyle = '#fff'; g.fillRect(x(timeline.t) - 1, 4, 2, 26);
  g.fillStyle = '#8a93a6'; g.font = '10px system-ui';
  g.fillText('forward', 4, 8); g.fillText('backward', x(fe) + 4, 8);
  const s = timeline.state();
  $('scrub-label').textContent = s.active >= 0 ? `${s.dir === 'fwd' ? '→' : '←'} op ${s.active + 1}/${ops.length} · ${ops[s.active].scope || 'input'} · ${ops[s.active].op}` : 'pass complete';
}
let dragging = false;
const scrub = (e: PointerEvent) => {
  const r = tl.getBoundingClientRect();
  timeline.seek(((e.clientX - r.left) / r.width) * timeline.steps.length);
  refresh();
};
tl.addEventListener('pointerdown', (e) => { dragging = true; timeline.playing = false; tl.setPointerCapture(e.pointerId); scrub(e); });
tl.addEventListener('pointermove', (e) => dragging && scrub(e));
tl.addEventListener('pointerup', () => (dragging = false));

// ------------------------------------------------------------------ inputs
const structSel = $<HTMLSelectElement>('structure');
structSel.innerHTML = Object.entries(PRESETS).map(([k, s]) => `<option value="${k}">${s.label}</option>`).join('');
structSel.value = 'ethanol';
const setSystem = (s: System, restart = true) => {
  system = s;
  selected = Math.min(selected, s.numbers.length - 1);
  if (restart) { timeline.seek(0); autoplay = true; }
  evaluate();
};
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

// models/index.json: [{ name, kind, label, files: [json, weights] }]; weights may be absent (e.g. KRR)
interface ModelEntry { name: string; kind: ModelKind; label?: string; meta?: string; weights?: string }
const modelSel = $<HTMLSelectElement>('model');
let entries: ModelEntry[] = [];
async function loadModel(e: ModelEntry, meta0?: any, weights0?: ArrayBuffer) {
  status(`loading ${e.label ?? e.name}…`);
  const m = meta0 ?? (e.meta ? await (await fetch(stem(e.meta))).json() : {});
  const w = weights0 ?? (e.weights ? await (await fetch(stem(e.weights))).arrayBuffer() : null);
  send({ type: 'loadModel', id: e.name, kind: e.kind, meta: m, weights: w, label: e.label ?? e.name }, w ? [w] : []);
}
async function listModels() {
  try { entries = await (await fetch('models/index.json')).json(); } catch { entries = []; }
  if (!entries.length) entries = [{ name: 'pet-mad-xs', kind: 'pet', meta: 'pet-mad-xs.json', weights: 'pet-mad-xs.safetensors' }];
  const q = new URLSearchParams(location.search).get('model');
  if (q) {
    // ?model=<url stem>: a PET model at <stem>.json and <stem>.safetensors (e.g. a HuggingFace repo)
    entries.push({ name: q, kind: 'pet', label: q.split('/').pop(), meta: `${q}.json`, weights: `${q}.safetensors` });
  }
  modelSel.innerHTML = entries.map((m) => `<option value="${m.name}">${m.label ?? m.name} · ${UIS[m.kind]?.family ?? m.kind}</option>`).join('');
  if (q) modelSel.value = q;
  modelSel.onchange = () => loadModel(entries.find((m) => m.name === modelSel.value)!);
  modelSel.onchange(new Event('change'));
}
const stem = (name: string) => (/^https?:/.test(name) ? name : `models/${name}`);
$<HTMLInputElement>('model-files').onchange = async (e) => {
  const files = [...((e.target as HTMLInputElement).files ?? [])];
  const j = files.find((f) => f.name.endsWith('.json')), w = files.find((f) => f.name.endsWith('.safetensors'));
  if (!j || !w) { status('<b>Pick both files</b> of a converted model: <code>name.json</code> and <code>name.safetensors</code>'); return; }
  const meta1 = JSON.parse(await j.text());
  const kind: ModelKind = meta1.architecture === 'pet' ? 'pet' : meta1.kind ?? 'pet';
  await loadModel({ name: j.name.replace(/\.json$/, ''), kind }, meta1, await w.arrayBuffer());
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
    meta = m.meta;
    const u = (ui = UIS[m.kind]!);
    net.describe = u.describe;
    net.rootTitle = u.name;
    net.collapsed.clear();
    $('model-card').innerHTML = u.card(m.meta, m.nParams, m.label);
    timeline.seek(0);
    autoplay = true;
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
    graph.setPass(p, ops);
    const ms = p.trace.ms;
    status(`<b>E = ${p.energy.toFixed(4)} eV</b> · ${p.numbers.length} atoms, ${p.trace.graph.center.length} ${p.trace.graph.label} edges · ` +
      `forward ${ms.forward.toFixed(1)} ms · backward ${ms.backward.toFixed(1)} ms<br>${backend}`);
    const Fmax = Math.max(...Array.from({ length: p.numbers.length }, (_, i) => Math.hypot(p.forces[3 * i], p.forces[3 * i + 1], p.forces[3 * i + 2])));
    $('energies').innerHTML = `selected ${label(selected)}: ε = <b>${p.energies[selected].toFixed(4)}</b> eV · ` +
      `|F| = <b>${Math.hypot(p.forces[3 * selected], p.forces[3 * selected + 1], p.forces[3 * selected + 2]).toFixed(3)}</b> eV/Å · max |F| ${Fmax.toFixed(3)} eV/Å`;
    lastActive = -2;
    if (autoplay) { autoplay = false; timeline.seek(0); timeline.playing = true; }
    refresh(true);
    if (queued) { queued = false; evaluate(); }
  } else if (m.type === 'progress') {
    status(m.text);
  } else if (m.type === 'error') {
    busy = false;
    status(`<b>Error:</b> ${m.text}`);
    console.error(m.text);
  }
};
// handy from the console
(window as any).mlipviz = { net, graph, timeline, get pass() { return pass; }, get ops() { return ops; }, get chapters() { return chapters; } };

send({ type: 'init', backend: new URLSearchParams(location.search).get('backend') as any ?? 'auto' });
