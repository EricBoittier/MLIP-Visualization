// The spherical harmonics and SOAP ops on the CPU: orthonormality, rotation invariance of the
// power spectrum, and gradients against finite differences.
import { describe, expect, it } from 'vitest';
import { CpuBackend, realSph } from '../src/engine/cpu';
import { Graph } from '../src/engine/tensor';

describe('real spherical harmonics', () => {
  it('are orthonormal on the sphere', () => {
    const L = 4, M = (L + 1) ** 2, G = new Float64Array(M * M);
    const nt = 240, np = 480;
    for (let a = 0; a < nt; a++) {
      // Gauss–Chebyshev-ish: midpoint rule in cos(theta) and phi
      const z = -1 + (2 * (a + 0.5)) / nt, st = Math.sqrt(1 - z * z);
      for (let b = 0; b < np; b++) {
        const ph = (2 * Math.PI * (b + 0.5)) / np;
        const y = realSph(st * Math.cos(ph), st * Math.sin(ph), z, L, false).y;
        const w = (2 / nt) * ((2 * Math.PI) / np);
        for (let i = 0; i < M; i++) for (let j = 0; j < M; j++) G[i * M + j] += w * y[i] * y[j];
      }
    }
    for (let i = 0; i < M; i++) for (let j = 0; j < M; j++) expect(Math.abs(G[i * M + j] - (i === j ? 1 : 0))).toBeLessThan(1e-3);
  });

  it('give a rotation-invariant power spectrum with correct gradients', async () => {
    const be = new CpuBackend();
    const vecs = [[0.3, -1.1, 0.7], [1.2, 0.4, -0.2], [-0.5, 0.9, 1.3], [0.1, 0.2, -1.4]];
    const rot = (v: number[]) => { const c = Math.cos(0.7), s = Math.sin(0.7); return [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]]; };
    const ps = async (vs: number[][], grad = false) => {
      const g = new Graph(be);
      const v = g.constant(Float32Array.from(vs.flat()), [vs.length, 3]);
      v.requiresGrad = grad;
      const Y = g.sph(g.div(v, g.rowNorm(v)), 3);
      const w = g.constant(Float32Array.from({ length: vs.length * 2 }, (_, i) => 0.3 + 0.1 * i), [2, vs.length]);
      const c = g.linear(g.transpose(Y), w); // [16, 2]: two "radial channels"
      const P = g.power(g.transpose(c), 1, 2, 3);
      const E = g.sumAll(g.unary('square', P));
      if (grad) g.backward(E);
      const out = { P: await be.read(P.buf), E: (await be.read(E.buf))[0], dv: grad ? await be.read(v.grad!) : null };
      g.release();
      return out;
    };
    const a = await ps(vecs), b = await ps(vecs.map(rot));
    a.P.forEach((x, i) => expect(Math.abs(x - b.P[i])).toBeLessThan(1e-5 * Math.max(1, Math.abs(x))));
    const { dv } = await ps(vecs, true);
    const h = 1e-3;
    for (let k = 0; k < 12; k++) {
      const p = vecs.map((v) => v.slice()), m = vecs.map((v) => v.slice());
      p[(k / 3) | 0][k % 3] += h; m[(k / 3) | 0][k % 3] -= h;
      const fd = ((await ps(p)).E - (await ps(m)).E) / (2 * h);
      // float32 finite differences: compare on the scale of the whole gradient
      expect(Math.abs(fd - dv![k])).toBeLessThan(1e-3 * Math.max(...Array.from(dv!, Math.abs)));
    }
  });
});
