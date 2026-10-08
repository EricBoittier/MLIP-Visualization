// Floating-point operations, counted from the shapes of the kernels a backend is asked to run.
// A multiply-add is 2 FLOPs and every other arithmetic op or transcendental is 1, the usual
// convention, so the totals are estimates of the arithmetic, not of what a GPU executes.
// Copies, gathers and fills move memory and count 0 unless they accumulate.
import type { Backend, Buf, SeqLayout } from './backend';

type Count = (...a: any[]) => number;
const some = (...bs: (Buf | null | undefined)[]) => bs.filter(Boolean).length;
const acc = (on: boolean, n: number) => (on ? n : 0);

/** FLOPs of each kernel, from its arguments (see backend.ts for what they mean). */
export const FLOPS: Partial<Record<keyof Backend, Count>> = {
  scale: (_x, _y, _a, n, ac) => n + acc(ac, n),
  matmul: (_A, _B, _C, M, N, K, _tA, _tB, ac, bias) => 2 * M * N * K + acc(ac, M * N) + (bias ? M * N : 0),
  unary: (_op, _x, _y, n) => n,
  unaryGrad: (_op, _x, _y, _dy, _dx, n) => 3 * n, // f'(x), times dy, accumulate
  binary: (_op, _a, _b, _y, n) => n,
  binaryGrad: (op, _a, _b, _dy, da, db, n) => (op === 'div' ? 3 : 2) * n * some(da, db),
  gather: (_x, _idx, _y, nOut, d, ac) => acc(ac, nOut * d),
  segment: (_x, _off, src: Buf, _y, _nOut, d) => src.n * d,
  copyCols: (_x, _xc, _xo, _y, _yc, _yo, rows, w, ac) => acc(ac, rows * w),
  colSum: (_x, _y, rows, cols) => rows * cols,
  rowSum: (_x, _y, rows, cols) => rows * cols,
  norm: (_k, _x, _w, b, _y, _s, rows, d) => (b ? 8 : 7) * rows * d, // mean, variance, normalise, scale (+ shift)
  normGrad: (_k, _x, _w, _s, _dy, dx, dw, db, rows, d) => rows * d * ((dx ? 8 : 0) + (dw ? 2 : 0) + (db ? 1 : 0)),
  // scores Q.K and the weighted sum P.V are dh multiply-adds per probability; softmax ~5 per probability
  attention: (_qkv, _bias, L: SeqLayout, _o, _p, _H, dh) => L.nProb * (4 * dh + 5),
  attentionGrad: (_qkv, _p, _dO, L: SeqLayout, _dqkv, _db, _H, dh) => L.nProb * (8 * dh + 5),
  cutoff: (_k, _d, _rc, _y, n) => 10 * n,
  cutoffGrad: (_k, _d, _rc, _dy, dd, drc, n) => 10 * n * some(dd, drc),
  adam: (_p, _g, _m, _v, n) => 12 * n,
  spline: (_x, _V, _D, _y, n, C) => 10 * n * C, // cubic Hermite: basis once, 4 multiply-adds per channel
  splineGrad: (_x, _V, _D, _dy, _dx, n, C) => 12 * n * C,
  sph: (_u, _y, n, lmax) => 6 * n * (lmax + 1) ** 2,
  sphGrad: (_u, _dy, _du, n, lmax) => 18 * n * (lmax + 1) ** 2,
  power: (_x, _p, B, A, lmax) => 2 * B * ((A * (A + 1)) / 2) * (lmax + 1) ** 2,
  powerGrad: (_x, _dp, _dx, B, A, lmax) => 4 * B * ((A * (A + 1)) / 2) * (lmax + 1) ** 2,
  rowMix: (_x, _A, _y, B, d1, d3, C) => 2 * B * d1 * d3 * C,
  rowMixGrad: (_x, _A, _dy, dx, dA, B, d1, d3, C) => 2 * B * d1 * d3 * C * some(dx, dA),
  couple: (_l, _r, _K, _y, N, n1, n2, n3, F) => 3 * N * n1 * n2 * n3 * F, // two multiplies and an add per term
  coupleGrad: (_l, _r, _K, _dy, dL, dR, dK, N, n1, n2, n3, F) => 3 * N * n1 * n2 * n3 * F * some(dL, dR, dK),
};

export interface FlopCounter {
  readonly be: Backend;
  /** FLOPs since the last reset, in total and per kernel. */
  total: number;
  byKernel: Map<string, number>;
  reset(): void;
}

/** `be`, with every kernel call counted. Calls a kernel makes internally are not counted twice. */
export function countFlops(be: Backend): FlopCounter {
  const c: FlopCounter = {
    total: 0,
    byKernel: new Map(),
    reset() { c.total = 0; c.byKernel.clear(); },
    be: new Proxy(be, {
      get(t, k, r) {
        const v = Reflect.get(t, k, r), f = FLOPS[k as keyof Backend];
        if (typeof v !== 'function') return v;
        if (!f) return v.bind(t);
        return (...a: any[]) => {
          const n = f(...a);
          c.total += n;
          c.byKernel.set(k as string, (c.byKernel.get(k as string) ?? 0) + n);
          return v.apply(t, a);
        };
      },
    }),
  };
  return c;
}
