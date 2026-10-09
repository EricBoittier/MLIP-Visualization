import '../app/style.css';
import './md.css';
import { isPeriodic, type System } from '../common/structure';
import { canSample, fetchBytes, findModels, getJSON, KIND_ORDER, type ModelEntry, resolve } from '../app/catalog';
import { PRESETS } from '../app/presets';
import { parseXYZ } from '../app/xyz';
import type { ModelKind } from '../models/types';
import { UIS } from '../models/uis';
import { LineChart } from './chart';
import { AU_FS, CM, estimate } from './dmc';
import type { DMCFrame, Flops, Frame, FromMD, MDForces, ToMD } from './md.worker';
import { TrajectoryView } from './view';
import { onTheme, themeButton } from '../viz/theme';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const worker = new Worker(new URL('./md.worker.ts', import.meta.url), { type: 'module' });
const send = (m: ToMD, t: Transferable[] = []) => worker.postMessage(m, t);
const status = (html: string) => ($('status').innerHTML = html);
const params = new URLSearchParams(location.search);
/** 1.2e9 -> '1.20 G' */
const si = (x: number) => {
  const k = Math.min(Math.max(Math.floor(Math.log10(Math.max(x, 1)) / 3), 0), 4);
  return `${(x / 1000 ** k).toPrecision(3)} ${['', 'k', 'M', 'G', 'T'][k]}`;
};

// series colours are theme tokens: Metatensor blue, red and aqua, validated per theme (app/style.css)
const view = new TrajectoryView($('traj'));
const energy = new LineChart($('energy-chart'), [{ name: 'Total', color: 'series-3' }, { name: 'Potential', color: 'series-1' }, { name: 'Kinetic', color: 'series-2' }], 'eV');
const temp = new LineChart($('temp-chart'), [{ name: 'Temperature', color: 'series-1' }], 'K');
const erefChart = new LineChart($('eref-chart'), [{ name: 'E_ref', color: 'series-1' }], 'cm⁻¹', 'a.u.');
const popChart = new LineChart($('pop-chart'), [{ name: 'Walkers', color: 'series-1' }], '', 'a.u.');
themeButton($<HTMLButtonElement>('theme'));
onTheme(() => [energy, temp, erefChart, popChart].forEach((c) => c.draw()));

// ------------------------------------------------------------------ state
type Mode = 'md' | 'dmc';
let mode: Mode = params.get('mode') === 'dmc' ? 'dmc' : 'md';
let system: System = PRESETS[mode === 'dmc' ? 'water' : 'ethanol'];
let backend = '', modelLabel = '', elements: number[] = [], hasNC = false;
let running = true, inflight = false, run = 0, steps = 4;
let e0 = 0, tSum = 0, nT = 0, worstDrift = 0; // NVE
let eref: number[] = []; // DMC: E_ref - V_min per step, cm^-1
const num = (id: string, lo: number, hi: number, d: number) => {
  const v = +$<HTMLInputElement>(id).value;
  return Number.isFinite(v) ? Math.min(Math.max(v, lo), hi) : d;
};
const T0 = () => num('temperature', 0, 5000, 300), dt = () => num('dt', 0.05, 5, 0.5);
const walkers = () => Math.round(num('walkers', 20, 20000, 500)), dtau = () => num('dtau', 0.5, 50, 10);
const forceMode = () => $<HTMLSelectElement>('force-mode').value as MDForces;
/** Atoms per batched DMC pass: as many as each model handles well (MACE's per-edge tensors are large). */
const PASS: Partial<Record<ModelKind, number>> = { pet: 4096, ani: 4096, mace: 512 };
let kind: ModelKind = 'pet';
const maxAtoms = () => Math.max(64, +(params.get('maxAtoms') ?? 0) || PASS[kind] || 1024);

function setMode(m: Mode) {
  mode = m;
  document.body.classList.toggle('mode-md', m === 'md');
  document.body.classList.toggle('mode-dmc', m === 'dmc');
  document.querySelectorAll<HTMLButtonElement>('#modes button').forEach((b) => b.classList.toggle('on', b.dataset.mode === m));
  const u = new URL(location.href);
  if (m === 'dmc') u.searchParams.set('mode', 'dmc'); else u.searchParams.delete('mode');
  history.replaceState(null, '', u); // a shared link opens in this mode
  restart();
}

/** Start again from the starting structure: new velocities (NVE) or a fresh population (DMC). */
function restart() {
  if (!elements.length) return;
  const bad = system.numbers.filter((z) => !elements.includes(z));
  if (bad.length) { status(`<b>${modelLabel} cannot handle this structure</b> (no parameters for Z = ${[...new Set(bad)].join(', ')})`); return; }
  if (mode === 'dmc' && isPeriodic(system)) { status('<b>DMC here is for isolated molecules:</b> pick one without a unit cell'); run++; return; }
  run++;
  inflight = true;
  steps = 1; // until the first frame says how long a step takes
  view.atomScale = mode === 'dmc' ? 0.45 : 1;
  view.setSystem(system);
  if (mode === 'md') {
    [e0, tSum, nT, worstDrift] = [NaN, 0, 0, 0];
    energy.clear(); temp.clear();
    temp.refs = [{ y: T0(), label: 'T₀' }, { y: T0() / 2, label: 'T₀/2' }];
    send({ type: 'start', run, system, temperature: T0(), dt: dt(), forces: forceMode() });
  } else {
    eref = [];
    erefChart.clear(); popChart.clear();
    erefChart.refs = [];
    popChart.refs = [{ y: walkers(), label: 'target' }];
    for (const id of ['zpe', 'walkers-now', 'samples', 'dmc-flops']) $(id).textContent = '–';
    for (const id of ['zpe-detail', 'walkers-detail', 'samples-detail', 'dmc-flops-detail']) $(id).textContent = '';
    loading.show('Relaxing the structure');
    loading.stage('DMC measures energies from the minimum of the model’s surface', 0);
    send({ type: 'startDMC', run, system, walkers: walkers(), dtau: dtau(), maxAtoms: maxAtoms() });
  }
}
function advance() {
  if (!running || inflight || !elements.length) return;
  inflight = true;
  send(mode === 'md' ? { type: 'advance', run, steps, dt: dt() } : { type: 'advanceDMC', run, steps, dtau: dtau() });
}
function setRunning(on: boolean) {
  running = on;
  $('run').textContent = on ? 'Pause' : 'Run';
  $('run').classList.toggle('on', on);
  advance();
}

const flopsTile = (b: string, detail: string, f: Flops, ms: number, per = '') => {
  const total = f.forward + f.backward;
  $(b).textContent = `${si((total / ms) * 1000)}FLOP/s`;
  $(detail).textContent = `${si(total)}FLOP per step` + (f.backward ? ` · forward ${si(f.forward)}, backward ${si(f.backward)}` : '') + per +
    ` · ${((100 * f.matmul) / (total || 1)).toFixed(0)}% matmul`;
};

function showMD(f: Frame) {
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
    // ns per day: dt fs per step, 1000 / ms steps per second, 86400 s per day, 1e6 fs per ns
    $('speed').textContent = `${((dt() * 86.4) / f.msPerStep).toPrecision(3)} ns/day`;
    $('ms').textContent = `${f.msPerStep.toFixed(1)} ms per step · ${((1000 / f.msPerStep) * dt()).toFixed(0)} fs/s`;
    flopsTile('flops', 'flops-detail', f.flops, f.msPerStep);
    // aim for a frame about every 60 ms, whatever the model costs
    steps = Math.min(Math.max(Math.round(60 / f.msPerStep), 1), 50);
  }
  $('clock').textContent = `t = ${f.time >= 1000 ? `${(f.time / 1000).toFixed(2)} ps` : `${f.time.toFixed(1)} fs`} · step ${f.step}`;
  status(`<b>${modelLabel}</b> · ${N} atoms · ${forceMode() === 'direct' && hasNC ? 'direct' : 'conservative'} forces<br>${backend}`);
}

function showDMC(f: DMCFrame) {
  if (!f.steps.length) { // relaxed: the population starts here
    loading.hide();
    view.update(f.ref);
  }
  view.setCloud(f.cloud);
  f.steps.forEach((s, i) => {
    const tau = f.tau - (f.steps.length - 1 - i) * dtau();
    eref.push((s.eref - f.vmin) * CM);
    erefChart.push(tau, [eref[eref.length - 1]]);
    popChart.push(tau, [s.n]);
  });
  const est = estimate(eref);
  erefChart.refs = est ? [{ y: est.mean, label: 'ZPE' }] : [];
  erefChart.draw(); popChart.draw();
  $('zpe').textContent = est ? `${est.mean.toFixed(0)} ± ${est.err.toFixed(0)} cm⁻¹` : 'equilibrating…';
  $('zpe-detail').textContent = est ? `${((est.mean / CM) * 1000).toFixed(1)} meV · mean of the last ${eref.length - Math.floor(eref.length / 2)} steps` : `${eref.length} steps so far`;
  $('walkers-now').textContent = `${f.walkers}`;
  $('walkers-detail').textContent = `target ${walkers()} · ${f.holes} lost to holes`;
  if (f.steps.length) {
    const perDay = (f.samplesPerStep / f.msPerStep) * 1000 * 86400;
    $('samples').textContent = `${(perDay / 1e9).toPrecision(3)} billion`;
    const passes = Math.ceil(f.samplesPerStep / Math.max(1, Math.floor(f.passAtoms / system.numbers.length)));
    $('samples-detail').textContent = `samples/day · ${f.msPerStep.toFixed(0)} ms per step of ${Math.round(f.samplesPerStep)} walkers in ${passes} pass${passes > 1 ? 'es' : ''}`;
    flopsTile('dmc-flops', 'dmc-flops-detail', f.flops, f.msPerStep, ` · ${si(f.flops.forward / (f.samplesPerStep || 1))}FLOP per walker`);
    // aim for a frame about every 250 ms
    steps = Math.min(Math.max(Math.round(250 / f.msPerStep), 1), 20);
  }
  $('clock').textContent = `τ = ${f.tau.toFixed(0)} a.u. (${(f.tau * AU_FS).toFixed(1)} fs) · step ${f.step}`;
  status(`<b>${modelLabel}</b> · ${system.numbers.length} atoms · V<sub>min</sub> = ${f.vmin.toFixed(4)} eV<br>${backend}`);
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
  entries = entries.filter(canSample); // production weights only (catalog.ts)
  if (!entries.length) { status('<b>No models found</b> locally or on Hugging Face.'); return; }
  const kinds = KIND_ORDER.filter((k) => entries.some((m) => m.kind === k));
  kindSel.innerHTML = kinds.map((k) => `<option value="${k}">${UIS[k]?.typeName ?? UIS[k]?.name ?? k} · ${UIS[k]?.family ?? ''}</option>`).join('');
  const fill = () => {
    modelSel.innerHTML = entries.filter((m) => m.kind === kindSel.value).map((m) => `<option value="${m.name}">${m.label ?? m.name}</option>`).join('');
  };
  const current = () => entries.find((m) => m.name === modelSel.value)!;
  const want = entries.find((m) => m.name === params.get('model')) ?? entries.find((m) => m.kind === params.get('kind'));
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
const q0 = params.get('structure');
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
for (const id of ['force-mode', 'temperature', 'walkers']) $(id).onchange = restart;
$('restart').onclick = restart;
$('run').onclick = () => setRunning(!running);
document.querySelectorAll<HTMLButtonElement>('#modes button').forEach((b) => (b.onclick = () => b.dataset.mode !== mode && setMode(b.dataset.mode as Mode)));
document.addEventListener('keydown', (e) => {
  if ((e.target as HTMLElement).closest('input, select')) return;
  if (e.key === ' ') { e.preventDefault(); setRunning(!running); }
  if (e.key === 'r' || e.key === 'R') restart();
});
document.body.classList.toggle('mode-md', mode === 'md');
document.body.classList.toggle('mode-dmc', mode === 'dmc');
document.querySelectorAll<HTMLButtonElement>('#modes button').forEach((b) => b.classList.toggle('on', b.dataset.mode === mode));

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
    [modelLabel, elements, hasNC, kind] = [m.label, m.elements, m.hasNC, m.kind];
    $('force-mode').querySelector<HTMLOptionElement>('[value=direct]')!.disabled = !hasNC;
    if (!hasNC) $<HTMLSelectElement>('force-mode').value = 'conservative';
    // a structure this model cannot handle gives way to the first preset it can
    const fits = (s: System) => s.numbers.every((z) => elements.includes(z));
    if (!fits(system)) {
      const k = Object.keys(PRESETS).find((key) => fits(PRESETS[key]));
      if (k) { system = PRESETS[k]; structSel.value = k; }
    }
    restart();
  } else if (m.type === 'frame' || m.type === 'dmc') {
    if (m.frame.run !== run) return;
    inflight = false;
    if (m.type === 'frame') showMD(m.frame); else showDMC(m.frame);
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

send({ type: 'init', backend: params.get('backend') as any ?? 'auto' });
