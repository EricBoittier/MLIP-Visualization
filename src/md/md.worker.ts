// Dynamics off the main thread. NVE: the loaded model gives energy and forces and velocity Verlet
// moves the atoms. DMC: relax, then diffuse a population of walkers whose energies are evaluated in
// batches. The page asks for a few steps at a time and draws each frame it gets back.
import type { Backend, Buf } from '../engine/backend';
import { CpuBackend } from '../engine/cpu';
import { Graph } from '../engine/tensor';
import { WebGPUBackend } from '../engine/webgpu';
import { countFlops, type FlopCounter } from '../engine/flops';
import type { System } from '../common/structure';
import { createModel } from '../models/registry';
import type { Model, ModelKind } from '../models/types';
import { type ForceFn, NVE } from './nve';
import { batchEnergies } from './batch';
import { align, DMC, type DMCStep } from './dmc';
import { relax } from './relax';

/** conservative: -dE/dr by backprop; direct: the model's force head (not a gradient, so E drifts). */
export type MDForces = 'conservative' | 'direct';

export type ToMD =
  | { type: 'init'; backend: 'webgpu' | 'cpu' | 'auto' }
  | { type: 'loadModel'; id: string; kind: ModelKind; meta: any; weights: ArrayBuffer | null; label: string }
  | { type: 'start'; run: number; system: System; temperature: number; dt: number; forces: MDForces }
  | { type: 'advance'; run: number; steps: number; dt: number }
  | { type: 'startDMC'; run: number; system: System; walkers: number; dtau: number; maxAtoms: number }
  | { type: 'advanceDMC'; run: number; steps: number; dtau: number };

export interface Frame {
  run: number; // which start this frame belongs to
  step: number;
  time: number; // fs
  positions: Float32Array; // [N*3]
  potential: number; // eV
  kinetic: number; // eV
  temperature: number; // K
  msPerStep: number;
  flops: Flops; // of the last force evaluation
}

export interface DMCFrame {
  run: number;
  step: number;
  tau: number; // atomic units
  steps: DMCStep[]; // since the last frame (eV)
  vmin: number; // the relaxed structure's energy, eV
  ref: Float32Array; // the relaxed structure, [N*3]
  cloud: Float32Array; // some walkers, aligned onto ref, [K*N*3]
  walkers: number;
  holes: number;
  msPerStep: number;
  samplesPerStep: number; // walker energies per step
  passAtoms: number; // atoms per batched pass
  flops: Flops; // per step
}

/** FLOPs of one force evaluation, estimated from the kernels it ran (engine/flops.ts). */
export interface Flops { forward: number; backward: number; matmul: number }

export type FromMD =
  | { type: 'ready'; backend: string }
  | { type: 'model'; id: string; label: string; kind: ModelKind; elements: number[]; hasNC: boolean; nParams: number }
  | { type: 'frame'; frame: Frame }
  | { type: 'dmc'; frame: DMCFrame }
  | { type: 'progress'; text: string; fraction?: number }
  | { type: 'error'; text: string };

const post = (m: FromMD, transfer: Transferable[] = []) => (self as any).postMessage(m, transfer);

let raw: Backend = new CpuBackend();
let counter: FlopCounter = countFlops(raw);
const be = () => counter.be;
let flops: Flops = { forward: 0, backward: 0, matmul: 0 };
const models = new Map<string, Model>();
let model: Model | null = null;
let md: NVE | null = null, dmc: DMC | null = null, run = 0, passAtoms = 1024;

function forceFn(m: Model, sys: System, mode: MDForces): ForceFn {
  const bad = sys.numbers.filter((z) => !m.elements.includes(z));
  if (bad.length) throw new Error(`this model has no parameters for element Z = ${[...new Set(bad)].join(', ')}`);
  const direct = mode === 'direct' && !!m.hasNC;
  return async (x) => {
    const g = new Graph(be());
    try {
      const positions = Array.from({ length: x.length / 3 }, (_, i) => [x[3 * i], x[3 * i + 1], x[3 * i + 2]]);
      counter.reset();
      const out = m.forward(g, { ...sys, positions }, { forces: !direct, nc: direct });
      const forward = counter.total;
      if (!direct) g.backward(out.energy);
      flops = { forward, backward: counter.total - forward, matmul: counter.byKernel.get('matmul') ?? 0 };
      const bufs: Buf[] = [out.energy.buf, direct ? out.ncForces!.buf : out.positions.grad!];
      const [e, f] = await Promise.all(bufs.map((b) => raw.read(b)));
      return { energy: e[0], forces: Float64Array.from(f, (v) => (direct ? v : -v)) };
    } finally {
      g.release();
    }
  };
}

const frame = (ms: number): Frame => ({
  run, step: md!.step, time: md!.time, positions: Float32Array.from(md!.x),
  potential: md!.energy, kinetic: md!.kinetic, temperature: md!.temperature, msPerStep: ms, flops,
});

/** Up to this many walkers are drawn as the cloud. */
const CLOUD = 120;
function dmcFrame(steps: DMCStep[], ms: number, samplesPerStep: number, fl: Flops): DMCFrame {
  const d = dmc!, ref = d.x0, W = d.x.length, K = Math.min(CLOUD, W), cloud = new Float32Array(K * ref.length);
  for (let k = 0; k < K; k++) cloud.set(align(d.x[Math.floor((k * W) / K)], ref), k * ref.length);
  return { run, step: d.step, tau: d.tau, steps, vmin: d.vmin, ref: Float32Array.from(ref), cloud, walkers: W, holes: d.holes,
           msPerStep: ms, samplesPerStep, passAtoms, flops: fl };
}

async function handle(msg: ToMD) {
  switch (msg.type) {
    case 'init':
      if (msg.backend !== 'cpu') {
        try { raw = await WebGPUBackend.create(); } catch (e) { if (msg.backend === 'webgpu') throw e; }
      }
      counter = countFlops(raw); // models are built on the counted backend, so every kernel they run is counted
      post({ type: 'ready', backend: raw instanceof WebGPUBackend ? `WebGPU · ${raw.adapterName}` : 'JavaScript (no WebGPU)' });
      break;
    case 'loadModel': {
      const m = models.get(msg.id) ?? await createModel(be(), msg.kind, msg.meta, msg.weights, { models, progress: (text, fraction) => post({ type: 'progress', text, fraction }) });
      models.set(msg.id, m);
      model = m;
      md = null;
      dmc = null;
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
    case 'startDMC': {
      if (!model) throw new Error('load a model first');
      run = msg.run;
      dmc = null;
      const sys = msg.system, m = model, mine = run;
      const relaxed = await relax(Float64Array.from(sys.positions.flat()), forceFn(m, sys, 'conservative'), {
        onStep: (k, fm) => k % 10 === 0 && post({ type: 'progress', text: `Relaxing the structure: step ${k}, max force ${fm.toFixed(3)} eV/Å`, fraction: Math.min(1, Math.log(1 / fm) / Math.log(100)) }),
      });
      if (mine !== run) break;
      const read = (b: Buf) => raw.read(b);
      passAtoms = msg.maxAtoms;
      // a pass too big for this GPU's buffers is retried in halves
      const energy = async (ws: Float64Array[]): Promise<Float64Array> => {
        try {
          return await batchEnergies(m, be(), sys, ws, passAtoms, read);
        } catch (e) {
          if (passAtoms <= 4 * sys.numbers.length) throw e;
          passAtoms = Math.floor(passAtoms / 2);
          return energy(ws);
        }
      };
      dmc = new DMC(sys.numbers, relaxed.x, relaxed.energy, energy, msg.walkers, msg.dtau);
      post({ type: 'dmc', frame: dmcFrame([], 0, 0, flops) });
      break;
    }
    case 'advanceDMC': {
      if (!dmc || msg.run !== run) break;
      dmc.dtau = msg.dtau;
      const t0 = performance.now(), s0 = dmc.samples, steps: DMCStep[] = [];
      let fl = 0, mm = 0;
      for (let k = 0; k < msg.steps; k++) {
        counter.reset();
        steps.push(await dmc.advance());
        fl += counter.total; mm += counter.byKernel.get('matmul') ?? 0;
      }
      const n = msg.steps;
      post({ type: 'dmc', frame: dmcFrame(steps, (performance.now() - t0) / n, (dmc.samples - s0) / n, { forward: fl / n, backward: 0, matmul: mm / n }) });
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
