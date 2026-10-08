// The engine runs off the main thread. It holds every loaded model and answers
// each structure with one forward + backward pass of the active one: energy,
// forces and the trace the visualiser plays.
import type { Backend, Buf } from '../engine/backend';
import { CpuBackend } from '../engine/cpu';
import { Graph } from '../engine/tensor';
import { WebGPUBackend } from '../engine/webgpu';
import { type System, volume } from '../common/structure';
import { createModel } from '../models/registry';
import type { Model } from '../models/types';
import type { ForceMode, FromWorker, Pass, ToWorker } from './protocol';
import { capture, topology, topologyKey } from './trace';

const post = (m: FromWorker, transfer: Transferable[] = []) => (self as any).postMessage(m, transfer);

let be: Backend = new CpuBackend();
const models = new Map<string, Model>();
let active: Model | null = null;
let lastTopo = '';
let passId = 0;

async function evaluate(m: Model, sys: System, selected: number, mode: ForceMode): Promise<Pass> {
  const bad = sys.numbers.filter((z) => !m.elements.includes(z));
  if (bad.length) throw new Error(`this model has no parameters for element Z = ${[...new Set(bad)].join(', ')}`);
  if (mode !== 'conservative' && !m.hasNC) mode = 'conservative';
  const back = mode !== 'direct';
  const g = new Graph(be);
  try {
    const t0 = performance.now();
    const out = m.forward(g, sys, { forces: back, nc: mode !== 'conservative' });
    await be.sync();
    const t1 = performance.now();
    if (back) g.backward(out.energy);
    await be.sync();
    const t2 = performance.now();
    const bufs: Buf[] = [out.energy.buf, out.perAtom.buf];
    const virial = back && !!out.virialVectors?.grad;
    if (back) bufs.push(out.positions.grad!);
    if (virial) bufs.push(out.virialVectors!.buf, out.virialVectors!.grad!);
    if (out.ncForces) bufs.push(out.ncForces.buf);
    const extras = Object.entries(out.extras ?? {});
    bufs.push(...extras.map(([, t]) => t.buf));
    const read: Float32Array[] = (be as any).readMany ? await (be as any).readMany(bufs) : await Promise.all(bufs.map((b) => be.read(b)));
    const ops = topology(g, m);
    const key = m.kind + topologyKey(ops);
    const c = await capture(g, m, out, { selected });
    const pass: Pass = {
      id: ++passId, positions: sys.positions, cell: sys.cell, numbers: sys.numbers, mode,
      energy: read[0][0], energies: read[1],
      trace: { ...c, shapes: g.tape.map((n) => n.out.shape), topology: key !== lastTopo ? ops : undefined,
               rows: out.rows, graph: out.graph,
               ms: { forward: t1 - t0, backward: back ? t2 - t1 : 0, capture: performance.now() - t2 } },
    };
    lastTopo = key;
    if (back) pass.forces = read[2].map((x) => -x);
    const V = volume(sys);
    if (virial && Number.isFinite(V)) {
      const [v, gv] = read.slice(3, 5);
      const s = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      for (let k = 0; k < v.length / 3; k++) for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) s[a][b] += (v[3 * k + a] * gv[3 * k + b]) / V;
      pass.stress = s;
    }
    if (out.ncForces) pass.ncForces = read[read.length - 1 - extras.length];
    if (extras.length) pass.atomProps = Object.fromEntries(extras.map(([k], n) => [k, read[read.length - extras.length + n]]));
    return pass;
  } finally {
    g.release();
  }
}

async function handle(msg: ToWorker) {
  switch (msg.type) {
    case 'init': {
      if (msg.backend !== 'cpu') {
        try {
          be = await WebGPUBackend.create();
        } catch (e) {
          if (msg.backend === 'webgpu') throw e;
        }
      }
      post({ type: 'ready', backend: be.name, adapter: be instanceof WebGPUBackend ? be.adapterName : 'JavaScript (no WebGPU)' });
      break;
    }
    case 'loadModel': {
      const m = await createModel(be, msg.kind, msg.meta, msg.weights, { models, progress: (text) => post({ type: 'progress', text }) });
      models.set(msg.id, m);
      if (msg.activate) { active = m; lastTopo = ''; }
      const nParams = [...m.params.values()].reduce((s, t) => s + t.size, 0);
      // a fitted model reports how the fit went (and the structure stays here)
      const meta = (m as any).report ? { ...msg.meta, system: undefined, report: (m as any).report, elements: m.elements } : msg.meta;
      post({ type: 'model', id: msg.id, kind: m.kind, label: msg.label, meta, nParams, hasNC: !!m.hasNC, activate: msg.activate, elements: m.elements });
      break;
    }
    case 'use':
      active = models.get(msg.id) ?? active;
      lastTopo = '';
      break;
    case 'evaluate': {
      if (!active) throw new Error('load a model first');
      const pass = await evaluate(active, msg.system, msg.selected, msg.mode);
      const t: Transferable[] = [];
      for (const x of [...pass.trace.values, ...pass.trace.grads]) if (x) t.push(x.data.buffer);
      post({ type: 'pass', pass }, t);
      break;
    }
  }
}

self.onmessage = (e: MessageEvent<ToWorker>) => {
  handle(e.data).catch((err) => post({ type: 'error', text: err?.message ?? String(err) }));
};
