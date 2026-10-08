// Reference backend: plain typed arrays. Every other backend is tested against it.
import type { Backend, Binary, BMode, Buf, CutoffKind, NormKind, SeqLayout, Thumb, Unary } from './backend';

type F = Buf & { a: Float32Array };
type I = Buf & { a: Int32Array };
const f = (b: Buf) => (b as F).a;
const ix = (b: Buf) => (b as I).a;

export type MatmulKernel = (A: Float32Array, B: Float32Array, C: Float32Array,
                            M: number, N: number, K: number, acc: boolean) => boolean;

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

function unaryF(op: Unary, x: number, a: number, b: number): number {
  switch (op) {
    case 'silu': return x * sigmoid(x);
    case 'sigmoid': return sigmoid(x);
    case 'exp': return Math.exp(x);
    case 'square': return x * x;
    case 'sqrt': return Math.sqrt(x);
    case 'neg': return -x;
    case 'tanh': return Math.tanh(x);
    case 'logclamp': return Math.log(Math.max(x, a));
    case 'clamp': return Math.min(Math.max(x, a), b);
    case 'acos': return Math.acos(x);
    case 'cos': return Math.cos(x);
    case 'pow': return Math.pow(x, a);
    case 'celu': return x > 0 ? x : a * Math.expm1(x / a);
    case 'erf': return erf(x);
    case 'switch': return smoothSwitch(x, a, b)[0];
    case 'sin': return Math.sin(x);
  }
}

function unaryD(op: Unary, x: number, y: number, a: number, b: number): number {
  switch (op) {
    case 'silu': { const s = sigmoid(x); return s * (1 + x * (1 - s)); }
    case 'sigmoid': return y * (1 - y);
    case 'exp': return y;
    case 'square': return 2 * x;
    case 'sqrt': return 0.5 / y;
    case 'neg': return -1;
    case 'tanh': return 1 - y * y;
    case 'logclamp': return x >= a ? 1 / x : 0;
    case 'clamp': return x >= a && x <= b ? 1 : 0;
    case 'acos': return -1 / Math.sqrt(1 - x * x);
    case 'cos': return -Math.sin(x);
    case 'pow': return a * Math.pow(x, a - 1);
    case 'celu': return x > 0 ? 1 : Math.exp(x / a);
    case 'erf': return (2 / Math.sqrt(Math.PI)) * Math.exp(-x * x);
    case 'switch': return smoothSwitch(x, a, b)[1];
    case 'sin': return Math.cos(x);
  }
}

const bIndex = (mode: BMode, i: number, inner: number) =>
  mode === 'full' ? i : mode === 'scalar' ? 0 : mode === 'row' ? (i / inner) | 0 : i % inner;

/** erf to ~1e-7 (Abramowitz & Stegun 7.1.26). */
export function erf(x: number): number {
  const s = Math.sign(x), t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}

/** e3x.nn.smooth_switch: 1 / (1 + exp(sqrt(3)/2 (1/s - 1/(1-s)))), s = (x - x0) / (x1 - x0), and its derivative. */
export function smoothSwitch(x: number, x0: number, x1: number): [number, number] {
  const w = x1 - x0, s = (x - x0) / w, eps = 5.960464477539063e-8;
  if (s < eps) return [0, 0];
  if (s > 1 - eps) return [1, 0];
  const c = Math.sqrt(3) / 2, g = Math.exp(c * (1 / s - 1 / (1 - s)));
  const f = 1 / (1 + g);
  return [f, (g * c * (1 / (s * s) + 1 / ((1 - s) * (1 - s)))) * f * f / w];
}

const binF = (op: Binary, a: number, b: number) =>
  op === 'add' ? a + b : op === 'sub' ? a - b : op === 'mul' ? a * b : a / b;

/** Smooth cutoff and its derivative with respect to the scaled distance s. */
export function cutoffFn(kind: CutoffKind, d: number, rc: number, w: number): [number, number] {
  const s = (d - (rc - w)) / w;
  if (kind === 'cosine') {
    const c = Math.min(Math.max(s, 0), 1);
    return [0.5 * (1 + Math.cos(Math.PI * c)), s >= 0 && s <= 1 ? -0.5 * Math.PI * Math.sin(Math.PI * c) : 0];
  }
  const lo = 1e-6, hi = 1 - 1e-6;
  const c = Math.min(Math.max(s, lo), hi);
  const sn = Math.sin(Math.PI * c), cot = Math.cos(Math.PI * c) / sn, t = Math.tanh(cot);
  return [0.5 * (1 + t), s >= lo && s <= hi ? (-0.5 * Math.PI * (1 - t * t)) / (sn * sn) : 0];
}

export class CpuBackend implements Backend {
  readonly name = 'cpu' as const;
  /** Optional fast path for C = A B^T (the linear-layer forward), e.g. WASM SIMD. */
  matmulNT: MatmulKernel | null = null;

  zeros(n: number): Buf { return { n, a: new Float32Array(Math.max(n, 1)) } as F; }
  upload(data: Float32Array): Buf { return { n: data.length, a: data.slice() } as F; }
  uploadI32(data: Int32Array): Buf { return { n: data.length, i32: true, a: data.slice() } as I; }
  async read(b: Buf) { return f(b).slice(0, b.n); }
  readSync(b: Buf) { return f(b).subarray(0, b.n); }
  free(_b: Buf) {}
  async sync() {}

  fill(y: Buf, v: number, n: number) { f(y).fill(v, 0, n); }

  scale(x: Buf, y: Buf, alpha: number, n: number, acc: boolean) {
    const X = f(x), Y = f(y);
    if (acc) for (let i = 0; i < n; i++) Y[i] += alpha * X[i];
    else for (let i = 0; i < n; i++) Y[i] = alpha * X[i];
  }

  matmul(Ab: Buf, Bb: Buf, Cb: Buf, M: number, N: number, K: number,
         tA: boolean, tB: boolean, acc: boolean, bias?: Buf) {
    const A = f(Ab), B = f(Bb), C = f(Cb);
    if (!acc) C.fill(0, 0, M * N);
    if (!tA && tB && this.matmulNT?.(A, B, C, M, N, K, true)) {
      // fast path done
    } else if (!tA && tB) {
      for (let i = 0; i < M; i++) {
        const ai = i * K;
        for (let j = 0; j < N; j++) {
          const bj = j * K;
          let s = 0;
          for (let k = 0; k < K; k++) s += A[ai + k] * B[bj + k];
          C[i * N + j] += s;
        }
      }
    } else if (!tA && !tB) {
      for (let i = 0; i < M; i++) {
        const ci = i * N;
        for (let k = 0; k < K; k++) {
          const a = A[i * K + k];
          if (a === 0) continue;
          const bk = k * N;
          for (let j = 0; j < N; j++) C[ci + j] += a * B[bk + j];
        }
      }
    } else if (tA && !tB) {
      // A stored [K, M]
      for (let k = 0; k < K; k++) {
        const ak = k * M, bk = k * N;
        for (let i = 0; i < M; i++) {
          const a = A[ak + i];
          if (a === 0) continue;
          const ci = i * N;
          for (let j = 0; j < N; j++) C[ci + j] += a * B[bk + j];
        }
      }
    } else {
      for (let i = 0; i < M; i++)
        for (let j = 0; j < N; j++) {
          let s = 0;
          for (let k = 0; k < K; k++) s += A[k * M + i] * B[j * K + k];
          C[i * N + j] += s;
        }
    }
    if (bias) {
      const b = f(bias);
      for (let i = 0; i < M; i++) for (let j = 0; j < N; j++) C[i * N + j] += b[j];
    }
  }

  unary(op: Unary, x: Buf, y: Buf, n: number, a: number, b: number) {
    const X = f(x), Y = f(y);
    for (let i = 0; i < n; i++) Y[i] = unaryF(op, X[i], a, b);
  }

  unaryGrad(op: Unary, x: Buf, y: Buf, dy: Buf, dx: Buf, n: number, a: number, b: number) {
    const X = f(x), Y = f(y), DY = f(dy), DX = f(dx);
    for (let i = 0; i < n; i++) DX[i] += unaryD(op, X[i], Y[i], a, b) * DY[i];
  }

  binary(op: Binary, a: Buf, b: Buf, y: Buf, n: number, mode: BMode, inner: number) {
    const A = f(a), B = f(b), Y = f(y);
    for (let i = 0; i < n; i++) {
      const bv = B[bIndex(mode, i, inner)];
      Y[i] = binF(op, A[i], bv);
    }
  }

  binaryGrad(op: Binary, a: Buf, b: Buf, dy: Buf, da: Buf | null, db: Buf | null, n: number,
             mode: BMode, inner: number) {
    const A = f(a), B = f(b), DY = f(dy), DA = da && f(da), DB = db && f(db);
    for (let i = 0; i < n; i++) {
      const j = bIndex(mode, i, inner);
      const av = A[i], bv = B[j], g = DY[i];
      if (DA) DA[i] += op === 'mul' ? g * bv : op === 'div' ? g / bv : g;
      if (DB) DB[j] += op === 'mul' ? g * av : op === 'div' ? (-g * av) / (bv * bv) : op === 'sub' ? -g : g;
    }
  }

  gather(x: Buf, idx: Buf, y: Buf, nOut: number, d: number, acc: boolean) {
    const X = f(x), Y = f(y), J = ix(idx);
    for (let i = 0; i < nOut; i++) {
      const j = J[i], yo = i * d;
      if (j < 0) { if (!acc) Y.fill(0, yo, yo + d); continue; }
      const xo = j * d;
      if (acc) for (let c = 0; c < d; c++) Y[yo + c] += X[xo + c];
      else for (let c = 0; c < d; c++) Y[yo + c] = X[xo + c];
    }
  }

  segment(x: Buf, off: Buf, src: Buf, y: Buf, nOut: number, d: number, acc: boolean) {
    const X = f(x), Y = f(y), O = ix(off), S = ix(src);
    for (let t = 0; t < nOut; t++) {
      const yo = t * d;
      if (!acc) Y.fill(0, yo, yo + d);
      for (let k = O[t]; k < O[t + 1]; k++) {
        const xo = S[k] * d;
        for (let c = 0; c < d; c++) Y[yo + c] += X[xo + c];
      }
    }
  }

  copyCols(x: Buf, xCols: number, xOff: number, y: Buf, yCols: number, yOff: number,
           rows: number, w: number, acc: boolean) {
    const X = f(x), Y = f(y);
    for (let r = 0; r < rows; r++) {
      const xo = r * xCols + xOff, yo = r * yCols + yOff;
      if (acc) for (let c = 0; c < w; c++) Y[yo + c] += X[xo + c];
      else for (let c = 0; c < w; c++) Y[yo + c] = X[xo + c];
    }
  }

  colSum(x: Buf, y: Buf, rows: number, cols: number, acc: boolean) {
    const X = f(x), Y = f(y);
    if (!acc) Y.fill(0, 0, cols);
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) Y[c] += X[r * cols + c];
  }

  rowSum(x: Buf, y: Buf, rows: number, cols: number, acc: boolean) {
    const X = f(x), Y = f(y);
    for (let r = 0; r < rows; r++) {
      let s = 0;
      for (let c = 0; c < cols; c++) s += X[r * cols + c];
      Y[r] = acc ? Y[r] + s : s;
    }
  }

  norm(kind: NormKind, x: Buf, w: Buf, b: Buf | null, y: Buf, saved: Buf, rows: number, d: number, eps: number) {
    const X = f(x), W = f(w), B = b && f(b), Y = f(y), S = f(saved);
    for (let r = 0; r < rows; r++) {
      const o = r * d;
      let mean = 0;
      if (kind === 'layer') { for (let c = 0; c < d; c++) mean += X[o + c]; mean /= d; }
      let v = 0;
      for (let c = 0; c < d; c++) { const t = X[o + c] - mean; v += t * t; }
      const rstd = 1 / Math.sqrt(v / d + eps);
      S[2 * r] = mean; S[2 * r + 1] = rstd;
      for (let c = 0; c < d; c++) Y[o + c] = (X[o + c] - mean) * rstd * W[c] + (B ? B[c] : 0);
    }
  }

  normGrad(kind: NormKind, x: Buf, w: Buf, saved: Buf, dy: Buf, dx: Buf | null, dw: Buf | null,
           db: Buf | null, rows: number, d: number) {
    const X = f(x), W = f(w), S = f(saved), DY = f(dy);
    const DX = dx && f(dx), DW = dw && f(dw), DB = db && f(db);
    for (let r = 0; r < rows; r++) {
      const o = r * d, mean = S[2 * r], rstd = S[2 * r + 1];
      let sg = 0, sgx = 0;
      for (let c = 0; c < d; c++) {
        const xh = (X[o + c] - mean) * rstd, g = DY[o + c] * W[c];
        sg += g; sgx += g * xh;
        if (DW) DW[c] += DY[o + c] * xh;
        if (DB) DB[c] += DY[o + c];
      }
      if (!DX) continue;
      if (kind === 'rms') sg = 0;
      for (let c = 0; c < d; c++) {
        const xh = (X[o + c] - mean) * rstd;
        DX[o + c] += rstd * (DY[o + c] * W[c] - sg / d - (xh * sgx) / d);
      }
    }
  }

  attention(qkv: Buf, bias: Buf, L: SeqLayout, out: Buf, probs: Buf, H: number, dh: number, scale: number) {
    const Q = f(qkv), Bs = f(bias), O = f(out), P = f(probs), off = ix(L.off), poff = ix(L.poff);
    const D = H * dh, D3 = 3 * D;
    const s = new Float64Array(256);
    for (let a = 0; a < L.nSeq; a++) {
      const t0 = off[a], n = off[a + 1] - t0;
      const sc = s.length >= n ? s : new Float64Array(n);
      for (let h = 0; h < H; h++) {
        for (let i = 0; i < n; i++) {
          const qo = (t0 + i) * D3 + h * dh;
          let mx = -Infinity;
          for (let j = 0; j < n; j++) {
            const ko = (t0 + j) * D3 + D + h * dh;
            let v = 0;
            for (let c = 0; c < dh; c++) v += Q[qo + c] * Q[ko + c];
            v = v * scale + Bs[t0 + j];
            sc[j] = v; if (v > mx) mx = v;
          }
          let z = 0;
          for (let j = 0; j < n; j++) { sc[j] = Math.exp(sc[j] - mx); z += sc[j]; }
          const po = poff[a] + (h * n + i) * n, oo = (t0 + i) * D + h * dh;
          for (let c = 0; c < dh; c++) O[oo + c] = 0;
          for (let j = 0; j < n; j++) {
            const p = sc[j] / z;
            P[po + j] = p;
            const vo = (t0 + j) * D3 + 2 * D + h * dh;
            for (let c = 0; c < dh; c++) O[oo + c] += p * Q[vo + c];
          }
        }
      }
    }
  }

  attentionGrad(qkv: Buf, probs: Buf, dOut: Buf, L: SeqLayout, dqkv: Buf, dbias: Buf | null,
                H: number, dh: number, scale: number) {
    const Q = f(qkv), P = f(probs), DO = f(dOut), DQ = f(dqkv), DB = dbias && f(dbias);
    const off = ix(L.off), poff = ix(L.poff);
    const D = H * dh, D3 = 3 * D;
    for (let a = 0; a < L.nSeq; a++) {
      const t0 = off[a], n = off[a + 1] - t0;
      const ds = new Float64Array(n);
      for (let h = 0; h < H; h++) {
        for (let i = 0; i < n; i++) {
          const po = poff[a] + (h * n + i) * n, oo = (t0 + i) * D + h * dh;
          let dot = 0;
          for (let j = 0; j < n; j++) {
            const vo = (t0 + j) * D3 + 2 * D + h * dh;
            let dp = 0;
            for (let c = 0; c < dh; c++) dp += DO[oo + c] * Q[vo + c];
            ds[j] = dp; dot += dp * P[po + j];
            // dV_j += p_ij dO_i
            const p = P[po + j];
            for (let c = 0; c < dh; c++) DQ[vo + c] += p * DO[oo + c];
          }
          const qo = (t0 + i) * D3 + h * dh;
          for (let j = 0; j < n; j++) {
            const g = P[po + j] * (ds[j] - dot);
            if (DB) DB[t0 + j] += g;
            const ko = (t0 + j) * D3 + D + h * dh;
            for (let c = 0; c < dh; c++) {
              DQ[qo + c] += g * scale * Q[ko + c];
              DQ[ko + c] += g * scale * Q[qo + c];
            }
          }
        }
      }
    }
  }

  cutoff(kind: CutoffKind, d: Buf, rc: Buf, y: Buf, n: number, width: number) {
    const Dd = f(d), R = f(rc), Y = f(y);
    for (let i = 0; i < n; i++) Y[i] = cutoffFn(kind, Dd[i], R[i], width)[0];
  }

  cutoffGrad(kind: CutoffKind, d: Buf, rc: Buf, dy: Buf, dd: Buf | null, drc: Buf | null, n: number, width: number) {
    const Dd = f(d), R = f(rc), DY = f(dy), DD = dd && f(dd), DR = drc && f(drc);
    for (let i = 0; i < n; i++) {
      const g = (cutoffFn(kind, Dd[i], R[i], width)[1] * DY[i]) / width;
      if (DD) DD[i] += g;
      if (DR) DR[i] -= g;
    }
  }

  spline(x: Buf, V: Buf, D: Buf, y: Buf, n: number, C: number, K: number, h: number) {
    const X = f(x), Vv = f(V), Dd = f(D), Y = f(y);
    for (let e = 0; e < n; e++) for (let c = 0; c < C; c++) Y[e * C + c] = hermite(Vv, Dd, c, K, h, X[e])[0];
  }

  splineGrad(x: Buf, V: Buf, D: Buf, dy: Buf, dx: Buf, n: number, C: number, K: number, h: number) {
    const X = f(x), Vv = f(V), Dd = f(D), DY = f(dy), DX = f(dx);
    for (let e = 0; e < n; e++) {
      let g = 0;
      for (let c = 0; c < C; c++) g += DY[e * C + c] * hermite(Vv, Dd, c, K, h, X[e])[1];
      DX[e] += g;
    }
  }

  rowMix(x: Buf, A: Buf, y: Buf, B: number, d1: number, d3: number, C: number) {
    const X = f(x), Am = f(A), Y = f(y);
    for (let b = 0; b < B; b++)
      for (let j = 0; j < d3; j++)
        for (let c = 0; c < C; c++) {
          let s = 0;
          for (let i = 0; i < d1; i++) s += Am[b * d1 * d3 + i * d3 + j] * X[(b * d1 + i) * C + c];
          Y[(b * d3 + j) * C + c] = s;
        }
  }

  couple(left: Buf, right: Buf, K: Buf, y: Buf, N: number, n1: number, n2: number, n3: number, F: number) {
    const L = f(left), R = f(right), Ker = f(K), Y = f(y);
    for (let n = 0; n < N; n++) for (let k = 0; k < n3; k++) for (let c = 0; c < F; c++) {
      let s = 0;
      for (let i = 0; i < n1; i++) {
        const lv = L[(n * n1 + i) * F + c];
        for (let j = 0; j < n2; j++) s += lv * R[(n * n2 + j) * F + c] * Ker[(((i * n2 + j) * n3 + k) * F) + c];
      }
      Y[(n * n3 + k) * F + c] = s;
    }
  }

  coupleGrad(left: Buf, right: Buf, K: Buf, dy: Buf, dLeft: Buf | null, dRight: Buf | null, dK: Buf | null,
             N: number, n1: number, n2: number, n3: number, F: number) {
    const L = f(left), R = f(right), Ker = f(K), DY = f(dy);
    const DL = dLeft && f(dLeft), DR = dRight && f(dRight), DK = dK && f(dK);
    const at = (i: number, j: number, k: number, c: number) => (((i * n2 + j) * n3 + k) * F) + c;
    for (let n = 0; n < N; n++) for (let c = 0; c < F; c++) {
      for (let i = 0; i < n1; i++) if (DL) {
        let s = 0;
        for (let j = 0; j < n2; j++) for (let k = 0; k < n3; k++)
          s += DY[(n * n3 + k) * F + c] * R[(n * n2 + j) * F + c] * Ker[at(i, j, k, c)];
        DL[(n * n1 + i) * F + c] += s;
      }
      for (let j = 0; j < n2; j++) if (DR) {
        let s = 0;
        for (let i = 0; i < n1; i++) for (let k = 0; k < n3; k++)
          s += DY[(n * n3 + k) * F + c] * L[(n * n1 + i) * F + c] * Ker[at(i, j, k, c)];
        DR[(n * n2 + j) * F + c] += s;
      }
    }
    if (DK) for (let i = 0; i < n1; i++) for (let j = 0; j < n2; j++) for (let k = 0; k < n3; k++) for (let c = 0; c < F; c++) {
      let s = 0;
      for (let n = 0; n < N; n++) s += DY[(n * n3 + k) * F + c] * L[(n * n1 + i) * F + c] * R[(n * n2 + j) * F + c];
      DK[at(i, j, k, c)] += s;
    }
  }

  rowMixGrad(x: Buf, A: Buf, dy: Buf, dx: Buf | null, dA: Buf | null, B: number, d1: number, d3: number, C: number) {
    const X = f(x), Am = f(A), DY = f(dy), DX = dx && f(dx), DA = dA && f(dA);
    for (let b = 0; b < B; b++)
      for (let i = 0; i < d1; i++)
        for (let j = 0; j < d3; j++) {
          const a = Am[b * d1 * d3 + i * d3 + j], xo = (b * d1 + i) * C, yo = (b * d3 + j) * C;
          let s = 0;
          for (let c = 0; c < C; c++) {
            if (DX) DX[xo + c] += a * DY[yo + c];
            s += X[xo + c] * DY[yo + c];
          }
          if (DA) DA[b * d1 * d3 + i * d3 + j] += s;
        }
  }

  sph(u: Buf, y: Buf, n: number, lmax: number) {
    const U = f(u), Y = f(y), M = (lmax + 1) ** 2;
    for (let e = 0; e < n; e++) {
      const r = realSph(U[3 * e], U[3 * e + 1], U[3 * e + 2], lmax, false);
      for (let k = 0; k < M; k++) Y[e * M + k] = r.y[k];
    }
  }

  sphGrad(u: Buf, dy: Buf, du: Buf, n: number, lmax: number) {
    const U = f(u), DY = f(dy), DU = f(du), M = (lmax + 1) ** 2;
    for (let e = 0; e < n; e++) {
      const r = realSph(U[3 * e], U[3 * e + 1], U[3 * e + 2], lmax, true);
      for (let k = 0; k < M; k++) {
        const g = DY[e * M + k];
        DU[3 * e] += g * r.dx[k]; DU[3 * e + 1] += g * r.dy[k]; DU[3 * e + 2] += g * r.dz[k];
      }
    }
  }

  power(x: Buf, p: Buf, B: number, A: number, lmax: number) {
    const X = f(x), P = f(p), M = (lmax + 1) ** 2, L = lmax + 1, NP = (A * (A + 1)) / 2;
    for (let b = 0; b < B; b++) {
      let k = 0;
      for (let a = 0; a < A; a++) for (let a2 = a; a2 < A; a2++, k++) {
        const w = a === a2 ? 1 : Math.SQRT2, ro = (b * A + a) * M, r2 = (b * A + a2) * M;
        for (let l = 0; l < L; l++) {
          let s = 0;
          for (let q = l * l; q < (l + 1) * (l + 1); q++) s += X[ro + q] * X[r2 + q];
          P[(b * NP + k) * L + l] = w * Math.PI * Math.sqrt(8 / (2 * l + 1)) * s;
        }
      }
    }
  }

  powerGrad(x: Buf, dp: Buf, dx: Buf, B: number, A: number, lmax: number) {
    const X = f(x), DP = f(dp), DX = f(dx), M = (lmax + 1) ** 2, L = lmax + 1, NP = (A * (A + 1)) / 2;
    for (let b = 0; b < B; b++) {
      let k = 0;
      for (let a = 0; a < A; a++) for (let a2 = a; a2 < A; a2++, k++) {
        const w = a === a2 ? 2 : Math.SQRT2, ro = (b * A + a) * M, r2 = (b * A + a2) * M;
        for (let l = 0; l < L; l++) {
          const g = DP[(b * NP + k) * L + l] * Math.PI * Math.sqrt(8 / (2 * l + 1));
          for (let q = l * l; q < (l + 1) * (l + 1); q++) {
            if (a === a2) DX[ro + q] += w * g * X[ro + q];
            else { DX[ro + q] += w * g * X[r2 + q]; DX[r2 + q] += w * g * X[ro + q]; }
          }
        }
      }
    }
  }

  adam(p: Buf, g: Buf, m: Buf, v: Buf, n: number, lr: number, b1: number, b2: number, eps: number, t: number) {
    const Pp = f(p), G = f(g), Mm = f(m), V = f(v);
    const c1 = 1 - b1 ** t, c2 = 1 - b2 ** t;
    for (let i = 0; i < n; i++) {
      Mm[i] = b1 * Mm[i] + (1 - b1) * G[i];
      V[i] = b2 * V[i] + (1 - b2) * G[i] * G[i];
      Pp[i] -= (lr * (Mm[i] / c1)) / (Math.sqrt(V[i] / c2) + eps);
    }
  }

  async thumb(x: Buf, R: number, C: number, rows: number, cols: number): Promise<Thumb> {
    return thumbOf(f(x), R, C, rows, cols);
  }
}

/** Cubic Hermite interpolation of curve c at x, and its slope. */
export function hermite(V: Float32Array, D: Float32Array, c: number, K: number, h: number, x: number): [number, number] {
  const t = x / h, k = Math.floor(t);
  if (k < 0 || k >= K - 1) return [0, 0];
  const s = t - k, o = c * K + k;
  const s2 = s * s, s3 = s2 * s;
  const h00 = 2 * s3 - 3 * s2 + 1, h10 = s3 - 2 * s2 + s, h01 = -2 * s3 + 3 * s2, h11 = s3 - s2;
  const v = h00 * V[o] + h10 * h * D[o] + h01 * V[o + 1] + h11 * h * D[o + 1];
  const dv = ((6 * s2 - 6 * s) * V[o] + (3 * s2 - 4 * s + 1) * h * D[o] + (-6 * s2 + 6 * s) * V[o + 1] + (3 * s2 - 2 * s) * h * D[o + 1]) / h;
  return [v, dv];
}

/** Orthonormal real spherical harmonics (no Condon-Shortley phase) of (x, y, z), assumed a unit
 *  vector, and their derivatives with respect to x, y, z of the polynomial extension. */
export function realSph(x: number, y: number, z: number, lmax: number, grad: boolean) {
  const M = (lmax + 1) ** 2;
  const out = { y: new Float64Array(M), dx: new Float64Array(M), dy: new Float64Array(M), dz: new Float64Array(M) };
  // C_m + i S_m = (x + i y)^m, with derivatives
  const C = [1], S = [0], Cx = [0], Cy = [0], Sx = [0], Sy = [0];
  for (let m = 0; m < lmax; m++) {
    C.push(x * C[m] - y * S[m]); S.push(x * S[m] + y * C[m]);
    Cx.push(C[m] + x * Cx[m] - y * Sx[m]); Cy.push(x * Cy[m] - S[m] - y * Sy[m]);
    Sx.push(S[m] + x * Sx[m] + y * Cx[m]); Sy.push(x * Sy[m] + C[m] + y * Cy[m]);
  }
  for (let m = 0; m <= lmax; m++) {
    // Q_l^m(z) = P_l^m(z) / sin^m(theta), from l = m upwards
    let qPrev = 0, dPrev = 0;
    let q = 1;
    for (let k = 1; k <= 2 * m - 1; k += 2) q *= k;
    let d = 0;
    for (let l = m; l <= lmax; l++) {
      if (l > m) {
        const qn = l === m + 1 ? (2 * m + 1) * z * q : ((2 * l - 1) * z * q - (l + m - 1) * qPrev) / (l - m);
        const dn = l === m + 1 ? (2 * m + 1) * q : ((2 * l - 1) * (q + z * d) - (l + m - 1) * dPrev) / (l - m);
        qPrev = q; dPrev = d; q = qn; d = dn;
      }
      let fact = 1;
      for (let k = l - m + 1; k <= l + m; k++) fact *= k; // (l+m)!/(l-m)!
      const N = Math.sqrt((2 * l + 1) / (4 * Math.PI) / fact) * (m ? Math.SQRT2 : 1);
      const ip = l * l + l + m, im = l * l + l - m;
      out.y[ip] = N * q * C[m];
      if (m) out.y[im] = N * q * S[m];
      if (grad) {
        out.dx[ip] = N * q * Cx[m]; out.dy[ip] = N * q * Cy[m]; out.dz[ip] = N * d * C[m];
        if (m) { out.dx[im] = N * q * Sx[m]; out.dy[im] = N * q * Sy[m]; out.dz[im] = N * d * S[m]; }
      }
    }
  }
  return out;
}

export function thumbOf(X: Float32Array, R: number, C: number, rows: number, cols: number): Thumb {
  const data = new Float32Array(rows * cols);
  let s = 0, s2 = 0, mn = Infinity, mx = -Infinity;
  for (let r = 0; r < rows; r++) {
    const r0 = Math.floor((r * R) / rows), r1 = Math.max(r0 + 1, Math.floor(((r + 1) * R) / rows));
    for (let c = 0; c < cols; c++) {
      const c0 = Math.floor((c * C) / cols), c1 = Math.max(c0 + 1, Math.floor(((c + 1) * C) / cols));
      let acc = 0;
      for (let i = r0; i < r1; i++) for (let j = c0; j < c1; j++) acc += X[i * C + j];
      data[r * cols + c] = acc / ((r1 - r0) * (c1 - c0));
    }
  }
  const n = R * C;
  for (let i = 0; i < n; i++) { const v = X[i]; s += v; s2 += v * v; if (v < mn) mn = v; if (v > mx) mx = v; }
  const mean = n ? s / n : 0;
  return { rows, cols, data, mean, std: n ? Math.sqrt(Math.max(s2 / n - mean * mean, 0)) : 0,
           min: n ? mn : 0, max: n ? mx : 0, absmax: n ? Math.max(Math.abs(mn), Math.abs(mx)) : 0 };
}
