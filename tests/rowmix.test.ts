// rowMix (each block of rows mixed by its own matrix) and sin against finite differences on the CPU.
import { expect, it } from 'vitest';
import { CpuBackend } from '../src/engine/cpu';
import { Graph } from '../src/engine/tensor';

it('rowMix and sin: values and gradients', async () => {
  const be = new CpuBackend(), B = 4, d1 = 3, d3 = 5, C = 6;
  const xs = Float32Array.from({ length: B * d1 * C }, (_, i) => Math.sin(1.3 * i + 0.2));
  const As = Float32Array.from({ length: B * d1 * d3 }, (_, i) => Math.cos(0.7 * i));
  const run = async (x: Float32Array, A: Float32Array, grad = false) => {
    const g = new Graph(be);
    const tx = g.constant(x, [B * d1, C]), tA = g.constant(A, [B, d1 * d3]);
    tx.requiresGrad = tA.requiresGrad = grad;
    const y = g.rowMix(tx, tA, d1, d3);
    const E = g.sumAll(g.unary('sin', y));
    if (grad) g.backward(E);
    const out = { y: await be.read(y.buf), E: (await be.read(E.buf))[0], dx: grad ? await be.read(tx.grad!) : null, dA: grad ? await be.read(tA.grad!) : null };
    g.release();
    return out;
  };
  const r = await run(xs, As, true);
  for (let b = 0; b < B; b++)
    for (let j = 0; j < d3; j++)
      for (let c = 0; c < C; c++) {
        let s = 0;
        for (let i = 0; i < d1; i++) s += As[b * d1 * d3 + i * d3 + j] * xs[(b * d1 + i) * C + c];
        expect(r.y[(b * d3 + j) * C + c]).toBeCloseTo(s, 5);
      }
  const h = 1e-2;
  for (const [which, base, grad] of [['x', xs, r.dx!], ['A', As, r.dA!]] as const) {
    for (const k of [0, 5, base.length - 1, (base.length / 2) | 0]) {
      const p = Float32Array.from(base), m = Float32Array.from(base);
      p[k] += h; m[k] -= h;
      const fd = which === 'x' ? ((await run(p, As)).E - (await run(m, As)).E) / (2 * h) : ((await run(xs, p)).E - (await run(xs, m)).E) / (2 * h);
      expect(Math.abs(fd - grad[k]), `d${which}[${k}]`).toBeLessThan(2e-3 * Math.max(1, Math.abs(fd)));
    }
  }
});
