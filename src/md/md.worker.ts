// Molecular dynamics off the main thread: the loaded model gives energy and forces, NVE moves the
// atoms. The page asks for a few steps at a time and draws each frame it gets back.
import type { Backend, Buf } from '../engine/backend';
import { CpuBackend } from '../engine/cpu';
import { Graph } from '../engine/tensor';
import { WebGPUBackend } from '../engine/webgpu';
import type { System } from '../common/structure';
import { createModel } from '../models/registry';
import type { Model, ModelKind } from '../models/types';
import { type ForceFn, NVE } from './nve';

/** conservative: -dE/dr by backprop; direct: the model's force head (not a gradient, so E drifts). */
export type MDForces = 'conservative' | 'direct';

export type ToMD =
  | { type: 'init'; backend: 'webgpu' | 'cpu' | 'auto' }
  | { type: 'loadModel'; id: string; kind: ModelKind; meta: any; weights: ArrayBuffer | null; label: string }
  | { type: 'start'; run: number; system: System; temperature: number; dt: number; forces: MDForces }
  | { type: 'advance'; run: number; steps: number; dt: number };

export interface Frame {
  run: number; // which start this frame belongs to
  step: number;
  time: number; // fs
  positions: Float32Array; // [N*3]
  potential: number; // eV
  kinetic: number; // eV
  temperature: number; // K
  msPerStep: number;
}

export type FromMD =
  | { type: 'ready'; backend: string }
  | { type: 'model'; id: string; label: string; kind: ModelKind; elements: number[]; hasNC: boolean; nParams: number }
  | { type: 'frame'; frame: Frame }
  | { type: 'progress'; text: string; fraction?: number }
  | { type: 'error'; text: string };

const post = (m: FromMD, transfer: Transferable[] = []) => (self as any).postMessage(m, transfer);

let be: Backend = new CpuBackend();
const models = new Map<string, Model>();
let model: Model | null = null;
let md: NVE | null = null, run = 0;

function forceFn(m: Model, sys: System, mode: MDForces): ForceFn {
  const bad = sys.numbers.filter((z) => !m.elements.includes(z));
  if (bad.length) throw new Error(`this model has no parameters for element Z = ${[...new Set(bad)].join(', ')}`);
  const direct = mode === 'direct' && !!m.hasNC;
  return async (x) => {
    const g = new Graph(be);
    try {
      const positions = Array.from({ length: x.length / 3 }, (_, i) => [x[3 * i], x[3 * i + 1], x[3 * i + 2]]);
      const out = m.forward(g, { ...sys, positions }, { forces: !direct, nc: direct });
      if (!direct) g.backward(out.energy);
      const bufs: Buf[] = [out.energy.buf, direct ? out.ncForces!.buf : out.positions.grad!];
      const [e, f] = await Promise.all(bufs.map((b) => be.read(b)));
      return { energy: e[0], forces: Float64Array.from(f, (v) => (direct ? v : -v)) };
    } finally {
      g.release();
    }
  };
}

const frame = (ms: number): Frame => ({
  run, step: md!.step, time: md!.time, positions: Float32Array.from(md!.x),
  potential: md!.energy, kinetic: md!.kinetic, temperature: md!.temperature, msPerStep: ms,
});

async function handle(msg: ToMD) {
  switch (msg.type) {
    case 'init':
      if (msg.backend !== 'cpu') {
        try { be = await WebGPUBackend.create(); } catch (e) { if (msg.backend === 'webgpu') throw e; }
      }
      post({ type: 'ready', backend: be instanceof WebGPUBackend ? `WebGPU · ${be.adapterName}` : 'JavaScript (no WebGPU)' });
      break;
    case 'loadModel': {
      const m = models.get(msg.id) ?? await createModel(be, msg.kind, msg.meta, msg.weights, { models, progress: (text, fraction) => post({ type: 'progress', text, fraction }) });
      models.set(msg.id, m);
      model = m;
      md = null;
      const nParams = [...m.params.values()].reduce((s, t) => s + t.size, 0);
      post({ type: 'model', id: msg.id, label: msg.label, kind: m.kind, elements: m.elements, hasNC: !!m.hasNC, nParams });
      break;
    }
    case 'start': {
      if (!model) throw new Error('load a model first');
      const t0 = performance.now();
      run = msg.run;
      md = await NVE.create(msg.system.numbers, Float64Array.from(msg.system.positions.flat()), forceFn(model, msg.system, msg.forces), msg.dt);
      md.thermalize(msg.temperature);
      const f = frame(performance.now() - t0);
      post({ type: 'frame', frame: f }, [f.positions.buffer]);
      break;
    }
    case 'advance': {
      if (!md || msg.run !== run) break; // a request for a run that has been replaced
      md.dt = msg.dt;
      const t0 = performance.now();
      await md.advance(msg.steps);
      const f = frame((performance.now() - t0) / msg.steps);
      post({ type: 'frame', frame: f }, [f.positions.buffer]);
      break;
    }
  }
}

// one message at a time: a restart must not land in the middle of a step
let queue = Promise.resolve();
self.onmessage = (e: MessageEvent<ToMD>) => {
  queue = queue.then(() => handle(e.data)).catch((err) => post({ type: 'error', text: err?.message ?? String(err) }));
};
