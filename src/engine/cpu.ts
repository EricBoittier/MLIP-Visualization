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
  }
}

const bIndex = (mode: BMode, i: number, inner: number) =>
  mode === 'full' ? i : mode === 'scalar' ? 0 : mode === 'row' ? (i / inner) | 0 : i % inner;

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
