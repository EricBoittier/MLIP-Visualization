import '../app/style.css';
import './md.css';
import type { System } from '../common/structure';
import { fetchBytes, findModels, getJSON, KIND_ORDER, type ModelEntry, resolve } from '../app/catalog';
import { PRESETS } from '../app/presets';
import { parseXYZ } from '../app/xyz';
import { UIS } from '../models/uis';
import { LineChart } from './chart';
import type { Frame, FromMD, MDForces, ToMD } from './md.worker';
import { TrajectoryView } from './view';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const worker = new Worker(new URL('./md.worker.ts', import.meta.url), { type: 'module' });
const send = (m: ToMD, t: Transferable[] = []) => worker.postMessage(m, t);
const status = (html: string) => ($('status').innerHTML = html);

// categorical slots 1-3 of the chart palette, validated on the dark panel
const BLUE = '#3987e5', ORANGE = '#d95926', AQUA = '#199e70';
const view = new TrajectoryView($('traj'));
const energy = new LineChart($('energy-chart'), [{ name: 'Total', color: AQUA }, { name: 'Potential', color: BLUE }, { name: 'Kinetic', color: ORANGE }], 'eV');
const temp = new LineChart($('temp-chart'), [{ name: 'Temperature', color: BLUE }], 'K');

// ------------------------------------------------------------------ state
let system: System = PRESETS.ethanol;
let backend = '', modelLabel = '', elements: number[] = [], hasNC = false;
let running = true, inflight = false, run = 0;
let e0 = 0, tSum = 0, nT = 0, worstDrift = 0, steps = 4;
const num = (id: string, lo: number, hi: number, d: number) => {
  const v = +$<HTMLInputElement>(id).value;
  return Number.isFinite(v) ? Math.min(Math.max(v, lo), hi) : d;
};
const T0 = () => num('temperature', 0, 5000, 300), dt = () => num('dt', 0.05, 5, 0.5);
const forceMode = () => $<HTMLSelectElement>('force-mode').value as MDForces;

/** New velocities at T0 from the starting structure. */
function restart() {
  if (!elements.length) return;
  const bad = system.numbers.filter((z) => !elements.includes(z));
  if (bad.length) { status(`<b>${modelLabel} cannot handle this structure</b> (no parameters for Z = ${[...new Set(bad)].join(', ')})`); return; }
  run++;
  inflight = true;
  [e0, tSum, nT, worstDrift] = [NaN, 0, 0, 0];
  energy.clear(); temp.clear();
  temp.refs = [{ y: T0(), label: 'T₀' }, { y: T0() / 2, label: 'T₀/2' }];
  view.setSystem(system);
  send({ type: 'start', run, system, temperature: T0(), dt: dt(), forces: forceMode() });
}
function advance() {
  if (!running || inflight || !elements.length) return;
  inflight = true;
  send({ type: 'advance', run, steps, dt: dt() });
}
function setRunning(on: boolean) {
  running = on;
  $('run').textContent = on ? 'Pause' : 'Run';
  $('run').classList.toggle('on', on);
  advance();
}

function show(f: Frame) {
  const N = system.numbers.length, total = f.potential + f.kinetic;
  if (Number.isNaN(e0)) e0 = total;
  view.update(f.positions);
  energy.push(f.time, [total - e0, f.potential - e0, f.kinetic]);
  temp.push(f.time, [f.temperature]);
  energy.draw(); temp.draw();
  if (f.step > 0) { tSum += f.temperature; nT++; }
  const drift = ((total - e0) / N) * 1000;
  worstDrift = Math.max(worstDrift, Math.abs(drift));
  $('t-now').textContent = `${f.temperature.toFixed(0)} K`;
  $('t-mean').textContent = nT ? `mean ${(tSum / nT).toFixed(0)} K` : `T₀ ${T0()} K`;
  $('drift').textContent = `${drift >= 0 ? '+' : '−'}${Math.abs(drift).toFixed(2)}`;
  $('drift-max').textContent = `meV/atom · worst ${worstDrift.toFixed(2)}`;
  if (f.step > 0) {
    $('speed').textContent = `${f.msPerStep.toFixed(1)} ms`;
    $('ms').textContent = `per step · ${((1000 / f.msPerStep) * dt()).toFixed(0)} fs/s`;
    // aim for a frame about every 60 ms, whatever the model costs
    steps = Math.min(Math.max(Math.round(60 / f.msPerStep), 1), 50);
  }
  $('clock').textContent = `t = ${f.time >= 1000 ? `${(f.time / 1000).toFixed(2)} ps` : `${f.time.toFixed(1)} fs`} · step ${f.step}`;
  status(`<b>${modelLabel}</b> · ${N} atoms · ${forceMode() === 'direct' && hasNC ? 'direct' : 'conservative'} forces<br>${backend}`);
}

// ------------------------------------------------------------------ models
const kindSel = $<HTMLSelectElement>('kind'), modelSel = $<HTMLSelectElement>('model');
const loaded = new Set<string>();
let entries: ModelEntry[] = [], base = 'models/';
const loading = {
  el: $('md-loading'),
  show(title: string) { this.el.hidden = false; this.el.querySelector('.loader-title')!.textContent = title; this.stage('', 0); },
  stage(text: string, f?: number) {
    this.el.querySelector('.loader-stage')!.textContent = text;
    (this.el.querySelector('.loader-bar i') as HTMLElement).style.width = `${100 * (f ?? 0)}%`;
  },
  hide() { this.el.hidden = true; },
};

async function loadModel(e: ModelEntry) {
  const name = e.label ?? e.name;
  elements = [];
  run++; // frames still on their way belong to the old model
  inflight = false;
  try {
    status(`loading ${name}…`);
    if (loaded.has(e.name)) { send({ type: 'loadModel', id: e.name, kind: e.kind, meta: null, weights: null, label: name }); return; }
    loading.show(`Loading ${name}`);
    const meta = e.meta ? await getJSON(resolve(base, e.meta)) : {};
    const w = e.weights ? await fetchBytes(resolve(base, e.weights), (f, mb) => loading.stage(`Downloading the weights: ${mb}`, f)) : null;
    loading.stage(`Building ${name}…`, 1);
    send({ type: 'loadModel', id: e.name, kind: e.kind, meta, weights: w, label: name }, w ? [w] : []);
  } catch (err) {
    loading.hide();
    status(`<b>Could not load ${name}:</b> ${(err as Error).message}`);
  }
}

async function listModels() {
  ({ entries, base } = await findModels());
  entries = entries.filter((m) => m.kind !== 'krr'); // KRR is fitted per structure: it lives on the main page
  if (!entries.length) { status('<b>No models found</b> locally or on Hugging Face.'); return; }
  const kinds = KIND_ORDER.filter((k) => entries.some((m) => m.kind === k));
  kindSel.innerHTML = kinds.map((k) => `<option value="${k}">${UIS[k]?.typeName ?? UIS[k]?.name ?? k} · ${UIS[k]?.family ?? ''}</option>`).join('');
  const fill = () => {
    modelSel.innerHTML = entries.filter((m) => m.kind === kindSel.value).map((m) => `<option value="${m.name}">${m.label ?? m.name}</option>`).join('');
  };
  const current = () => entries.find((m) => m.name === modelSel.value)!;
  const q = new URLSearchParams(location.search);
  const want = entries.find((m) => m.name === q.get('model')) ?? entries.find((m) => m.kind === q.get('kind'));
  if (want) kindSel.value = want.kind;
  fill();
  if (want) modelSel.value = want.name;
  kindSel.onchange = () => { fill(); loadModel(current()); };
  modelSel.onchange = () => loadModel(current());
  loadModel(current());
}

// ------------------------------------------------------------------ inputs
const structSel = $<HTMLSelectElement>('structure');
structSel.innerHTML = Object.entries(PRESETS).map(([k, s]) => `<option value="${k}">${s.label}</option>`).join('');
const q0 = new URLSearchParams(location.search).get('structure');
if (q0 && PRESETS[q0]) system = PRESETS[q0];
structSel.value = Object.keys(PRESETS).find((k) => PRESETS[k] === system)!;
structSel.onchange = () => { system = PRESETS[structSel.value]; restart(); };
$<HTMLInputElement>('xyz-file').onchange = async (e) => {
  const f = (e.target as HTMLInputElement).files?.[0];
  if (!f) return;
  try {
    system = parseXYZ(await f.text());
    const opt = document.createElement('option');
    opt.textContent = f.name;
    structSel.prepend(opt);
    structSel.selectedIndex = 0;
    restart();
  } catch (err) { status(`<b>Could not read ${f.name}:</b> ${(err as Error).message}`); }
};
$('force-mode').onchange = restart;
$('temperature').onchange = restart;
$('restart').onclick = restart;
$('run').onclick = () => setRunning(!running);
document.addEventListener('keydown', (e) => {
  if ((e.target as HTMLElement).closest('input, select')) return;
  if (e.key === ' ') { e.preventDefault(); setRunning(!running); }
  if (e.key === 'r' || e.key === 'R') restart();
});

// ------------------------------------------------------------------ messages
worker.onmessage = (e: MessageEvent<FromMD>) => {
  const m = e.data;
  if (m.type === 'ready') {
    backend = m.backend;
    status(backend);
    listModels();
  } else if (m.type === 'model') {
    loaded.add(m.id);
    loading.hide();
    [modelLabel, elements, hasNC] = [m.label, m.elements, m.hasNC];
    $('force-mode').querySelector<HTMLOptionElement>('[value=direct]')!.disabled = !hasNC;
    if (!hasNC) $<HTMLSelectElement>('force-mode').value = 'conservative';
    // a structure this model cannot handle gives way to the first preset it can
    const fits = (s: System) => s.numbers.every((z) => elements.includes(z));
    if (!fits(system)) {
      const k = Object.keys(PRESETS).find((key) => fits(PRESETS[key]));
      if (k) { system = PRESETS[k]; structSel.value = k; }
    }
    restart();
  } else if (m.type === 'frame') {
    if (m.frame.run !== run) return;
    inflight = false;
    show(m.frame);
    advance();
  } else if (m.type === 'progress') {
    loading.stage(m.text, m.fraction);
  } else if (m.type === 'error') {
    inflight = false;
    loading.hide();
    status(`<b>Error:</b> ${m.text}`);
    console.error(m.text);
  }
};

send({ type: 'init', backend: new URLSearchParams(location.search).get('backend') as any ?? 'auto' });
