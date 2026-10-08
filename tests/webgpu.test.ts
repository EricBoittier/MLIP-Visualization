// The WebGPU backend against the CPU reference: every kernel, then whole models.
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { create, globals } from 'webgpu';
import { beforeAll, describe, expect, it } from 'vitest';
import type { Backend } from '../src/engine/backend';
import { CpuBackend } from '../src/engine/cpu';
import { Graph, type Tensor } from '../src/engine/tensor';
import { WebGPUBackend } from '../src/engine/webgpu';
import { evaluate } from '../src/models/pet/model';
import { loadANI, loadMACE, loadModel } from './util';

Object.assign(globalThis, globals);
let gpu: WebGPUBackend;
const gpuErrors: string[] = [];
beforeAll(async () => {
  gpu = await WebGPUBackend.create(create([]));
  gpu.device.addEventListener('uncapturederror', (e: any) => { gpuErrors.push(e.error.message); });
});

const rand = (n: number, seed = 1) => {
  let s = seed;
  return Float32Array.from({ length: n }, () => ((s = (s * 16807) % 2147483647) / 2147483647) * 2 - 1);
};
const maxRel = (a: Float32Array, b: Float32Array) => {
  let m = 0, scale = 1e-6;
  for (let i = 0; i < a.length; i++) scale = Math.max(scale, Math.abs(b[i]));
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m / scale;
};

/** Run the same little graph on both backends, compare outputs and input gradients. */
async function compare(build: (g: Graph, be: Backend) => { out: Tensor; inputs: Tensor[] }) {
  const res: Float32Array[][] = [];
  for (const be of [new CpuBackend(), gpu] as Backend[]) {
    const g = new Graph(be);
    const { out, inputs } = build(g, be);
    g.backward(out, rand(out.size, 7));
    res.push([await be.read(out.buf), ...(await Promise.all(inputs.map((t) => be.read(t.grad!))))]);
    g.release();
  }
  expect(gpuErrors.splice(0)).toEqual([]);
  res[0].forEach((c, i) => expect(maxRel(res[1][i], c), `tensor ${i}`).toBeLessThan(2e-5));
}

describe('WebGPU kernels', () => {
  const T = (g: Graph, shape: number[], seed: number) => {
    const t = g.constant(rand(shape.reduce((a, b) => a * b), seed), shape);
    t.requiresGrad = true;
    return t;
  };
  it('linear', () => compare((g) => {
    const x = T(g, [37, 70], 1), W = T(g, [53, 70], 2), b = T(g, [53], 3);
    return { out: g.linear(x, W, b), inputs: [x, W, b] };
  }));
  it('unary ops', () => compare((g) => {
    const x = T(g, [11, 13], 4);
    const y = g.add(g.add(g.silu(x), g.sigmoid(x)), g.add(g.unary('tanh', x), g.unary('clamp', x, -0.3, 0.4)));
    return { out: g.add(y, g.unary('logclamp', g.unary('square', x), 1e-3)), inputs: [x] };
  }));
  it('acos / cos / pow / celu / erf / switch', () => compare((g) => {
    const x = T(g, [9, 7], 21);
    const y = g.add(g.unary('acos', g.scale(x, 0.9)), g.unary('cos', g.scale(x, 3)));
    const z = g.add(g.unary('pow', g.add(g.unary('square', x), g.constant(new Float32Array([0.1]), [1])), 3.7), g.unary('celu', g.scale(x, 2), 0.1));
    const w = g.add(g.unary('erf', g.scale(x, 1.5)), g.unary('switch', x, -0.6, 0.7));
    return { out: g.add(g.add(y, z), w), inputs: [x] };
    return { out: g.add(y, z), inputs: [x] };
  }));
  it('spline / spherical harmonics / power spectrum / transpose', () => compare((g) => {
    const K = 30, C = 5, h = 0.2;
    const V = g.constant(Float32Array.from({ length: C * K }, (_, i) => Math.sin(i * 0.37)), [C, K]);
    const D = g.constant(Float32Array.from({ length: C * K }, (_, i) => Math.cos(i * 0.21)), [C, K]);
    const r = g.constant(Float32Array.from({ length: 7 }, (_, i) => 0.3 + i * 0.71), [7]); r.requiresGrad = true;
    const v = T(g, [7, 3], 31);
    const u = g.div(v, g.rowNorm(v));
    const Y = g.sph(u, 4);
    const S = g.spline(r, V, D, h);
    // three "atoms" of A = 2 density rows, built from both
    const rows = g.linear(g.concatCols([Y, S]), g.constant(rand(25 * 30, 33), [25, 30]));
    const P = g.power(g.reshape(g.sliceCols(rows, 0, 25), [7, 25]), 1, 7, 4);
    return { out: g.add(g.transpose(g.transpose(P)), g.scale(P, 0.5)), inputs: [r, v] };
  }));
  it('sin / row mix', () => compare((g) => {
    const B = 5, d1 = 3, d3 = 5, C = 7;
    const x = T(g, [B * d1, C], 41), A = T(g, [B, d1 * d3], 42), t = T(g, [B * d3, C], 43);
    return { out: g.mul(g.rowMix(x, A, d1, d3), g.unary('sin', g.scale(t, 2))), inputs: [x, A, t] };
  }));
  it('column broadcast', () => compare((g) => {
    const a = T(g, [11, 6], 22), b = T(g, [1, 6], 23);
    return { out: g.mul(g.sub(a, b), g.add(a, b)), inputs: [a, b] };
  }));
  it('binary broadcast', () => compare((g) => {
    const a = T(g, [9, 5], 5), b = T(g, [9], 6), c = T(g, [1], 7), d = T(g, [9, 5], 8);
    return { out: g.div(g.mul(g.sub(a, b), c), g.add(g.unary('square', d), g.constant(new Float32Array([1]), [1]))), inputs: [a, b, c, d] };
  }));
  it('gather / segment / cols / rows', () => compare((g) => {
    const x = T(g, [6, 4], 9), y = T(g, [3, 4], 10);
    const ix = g.index(Int32Array.from([5, 0, -1, 3, 3, 1, 2]), 6);
    const seg = g.index(Int32Array.from([0, 2, 2, 1, 0, 1, 2]), 3);
    const s = g.segmentSum(g.gather(x, ix), seg);
    const cat = g.concatCols([s, y, g.sliceCols(x, 1, 2)].slice(0, 2));
    return { out: g.add(g.concatRows(cat, g.concatCols([y, y])), g.constant(rand(48, 3), [6, 8])), inputs: [x, y] };
  }));
  it('norms', () => compare((g) => {
    const x = T(g, [7, 33], 11), w = T(g, [33], 12), b = T(g, [33], 13);
    return { out: g.add(g.norm('layer', x, w, b, 1e-5), g.norm('rms', x, w, null, 1e-6)), inputs: [x, w, b] };
  }));
  it('attention', () => compare((g, be) => {
    const off = Int32Array.from([0, 3, 4, 9]), H = 2;
    const poff = new Int32Array(4), tok2seq = new Int32Array(9);
    for (let a = 0; a < 3; a++) { const L = off[a + 1] - off[a]; poff[a + 1] = poff[a] + H * L * L; for (let t = off[a]; t < off[a + 1]; t++) tok2seq[t] = a; }
    const L = { nSeq: 3, nTok: 9, nProb: poff[3], off: g.own(be.uploadI32(off)), poff: g.own(be.uploadI32(poff)), tok2seq: g.own(be.uploadI32(tok2seq)) };
    const qkv = T(g, [9, 3 * 8], 14), bias = T(g, [9], 15);
    return { out: g.attention(qkv, bias, L, H, 1.3), inputs: [qkv, bias] };
  }));
  it('cutoffs', () => compare((g) => {
    const d = g.constant(Float32Array.from({ length: 40 }, (_, i) => 3 + i * 0.05), [40]); d.requiresGrad = true;
    const rc = g.constant(new Float32Array(40).fill(4.5), [40]); rc.requiresGrad = true;
    return { out: g.add(g.cutoff('bump', d, rc, 1.0), g.cutoff('cosine', d, rc, 0.7)), inputs: [d, rc] };
  }));
});

describe('WebGPU PET vs CPU', () => {
  const refs = readdirSync('tests/reference').filter((f: string) => f.startsWith('pet-'));
  for (const file of refs) {
    const ref = JSON.parse(readFileSync(`tests/reference/${file}`, 'utf8'));
    it(`${ref.model} ${ref.case}`, async () => {
      const model = loadModel(ref.model, gpu);
      const sys = { numbers: ref.atomic_numbers, positions: ref.positions, cell: ref.cell, pbc: ref.pbc };
      const t0 = performance.now();
      const res = await evaluate(model, sys, { forces: true });
      const ms = performance.now() - t0;
      const dE = Math.abs(res.energy - ref.energy);
      const dF = Math.max(...Array.from(res.forces!, (x, i) => Math.abs(x - ref.forces.flat()[i])));
      console.log(`gpu ${ref.model.padEnd(16)} ${ref.case.padEnd(8)} dE=${dE.toExponential(2)} dF=${dF.toExponential(2)} ${ms.toFixed(0)} ms`);
      expect(dE).toBeLessThan(2e-6 * Math.max(1, Math.abs(ref.energy)));
      expect(dF).toBeLessThan(5e-4);
    });
  }
});

describe('WebGPU ANI-2x vs CPU', () => {
  it('ethanol and the periodic water box', async () => {
    const cpu = loadANI(), gpuModel = loadANI(gpu);
    for (const file of ['ani-2x_ethanol.json', 'ani-2x_water_pbc.json', 'ani-2x_sulfur_chlorine.json']) {
      const ref = JSON.parse(readFileSync(`tests/reference/${file}`, 'utf8'));
      const sys = { numbers: ref.atomic_numbers, positions: ref.positions, cell: ref.cell, pbc: ref.pbc };
      const res = [];
      for (const m of [cpu, gpuModel]) {
        const g = new Graph(m.be);
        const out = m.forward(g, sys, { forces: true });
        g.backward(out.energy);
        res.push([(await m.be.read(out.energy.buf))[0], await m.be.read(out.positions.grad!)] as const);
        g.release();
      }
      expect(Math.abs(res[0][0] - res[1][0])).toBeLessThan(2e-6 * Math.abs(ref.energy));
      expect(maxRel(res[1][1], res[0][1])).toBeLessThan(1e-4);
    }
  });
});

describe('WebGPU MACE vs mace-torch', () => {
  for (const name of ['mace-mp-0b3-medium', 'mace-mp-0b2-small'].filter((m) => existsSync(`public/models/${m}.json`))) {
    it(name, async () => {
      const model = loadMACE(name, gpu);
      for (const c of ['ethanol', 'silicon', 'sulfur']) {
        const ref = JSON.parse(readFileSync(`tests/reference/${name}_${c}.json`, 'utf8'));
        const g = new Graph(gpu), t0 = performance.now();
        const out = model.forward(g, { numbers: ref.atomic_numbers, positions: ref.positions, cell: ref.cell, pbc: ref.pbc }, { forces: true });
        g.backward(out.energy);
        const E = (await gpu.read(out.energy.buf))[0], F = await gpu.read(out.positions.grad!);
        const ms = performance.now() - t0;
        g.release();
        expect(gpuErrors.splice(0)).toEqual([]);
        const dE = Math.abs(E - ref.energy), dF = Math.max(...Array.from(F, (x, i) => Math.abs(-x - ref.forces.flat()[i])));
        console.log(`gpu ${name.padEnd(19)} ${c.padEnd(8)} dE=${dE.toExponential(2)} dF=${dF.toExponential(2)} ${ms.toFixed(0)} ms`);
        expect(dE).toBeLessThan(1e-5 * Math.abs(ref.energy));
        expect(dF).toBeLessThan(1e-4 * Math.max(1, ...ref.forces.flat().map(Math.abs)));
      }
    });
  }
});
