// The kernels every backend (CPU, WebGPU) provides. Autograd (tensor.ts) is
// written once against this interface, so the forward and the backward pass run
// unchanged on either. Buffers are opaque f32 (or i32 for indices) handles;
// sizes are in elements. `acc` = accumulate into the output instead of writing.

export type Buf = { readonly n: number; readonly i32?: boolean };

export type Unary =
  | 'silu' | 'sigmoid' | 'exp' | 'square' | 'sqrt' | 'neg' | 'tanh'
  | 'logclamp' // log(max(x, a))
  | 'clamp'; // min(max(x, a), b)

export type Binary = 'add' | 'sub' | 'mul' | 'div';
export type NormKind = 'layer' | 'rms';
export type CutoffKind = 'cosine' | 'bump';

/** Per-sequence layout of the attention tokens: atom i owns tokens [off[i], off[i+1]),
 *  probabilities [poff[i], poff[i] + H L_i^2), and tok2seq maps a token to its atom. */
export interface SeqLayout {
  nSeq: number;
  nTok: number;
  nProb: number;
  off: Buf;
  poff: Buf;
  tok2seq: Buf;
}

export interface Thumb {
  rows: number;
  cols: number;
  data: Float32Array; // block-averaged values, rows*cols
  mean: number;
  std: number;
  min: number;
  max: number;
  absmax: number;
}

export interface Backend {
  readonly name: 'cpu' | 'webgpu';
  zeros(n: number): Buf;
  upload(data: Float32Array): Buf;
  uploadI32(data: Int32Array): Buf;
  read(b: Buf): Promise<Float32Array>;
  free(b: Buf): void;
  /** Wait for queued work (no-op on CPU). */
  sync(): Promise<void>;

  fill(y: Buf, value: number, n: number): void;
  /** y (+)= alpha * x */
  scale(x: Buf, y: Buf, alpha: number, n: number, acc: boolean): void;
  /** C[M,N] (+)= op(A) op(B) (+ bias[N]); op(A) is [M,K], op(B) is [K,N]. */
  matmul(A: Buf, B: Buf, C: Buf, M: number, N: number, K: number,
         tA: boolean, tB: boolean, acc: boolean, bias?: Buf): void;
  unary(op: Unary, x: Buf, y: Buf, n: number, a: number, b: number): void;
  /** dx (+)= f'(x) dy, with y = f(x) the saved forward output. */
  unaryGrad(op: Unary, x: Buf, y: Buf, dy: Buf, dx: Buf, n: number, a: number, b: number): void;
  /** y = a op b; b of size n or 1 (broadcast scalar), or of size n/inner broadcast along rows
   *  when `bRows` (b[r] for row r of length inner). */
  binary(op: Binary, a: Buf, b: Buf, y: Buf, n: number, bMode: 'full' | 'scalar' | 'row', inner: number): void;
  /** Gradients of binary: da (+)= ..., db (+)= ... (reduced when broadcast). Either may be null. */
  binaryGrad(op: Binary, a: Buf, b: Buf, dy: Buf, da: Buf | null, db: Buf | null, n: number,
             bMode: 'full' | 'scalar' | 'row', inner: number): void;
  /** y[i, :] (+)= x[idx[i], :]; idx < 0 gives zeros. */
  gather(x: Buf, idx: Buf, y: Buf, nOut: number, d: number, acc: boolean): void;
  /** y[t, :] (+)= sum_{k in [off[t], off[t+1])} x[src[k], :] */
  segment(x: Buf, off: Buf, src: Buf, y: Buf, nOut: number, d: number, acc: boolean): void;
  /** y[r, yOff:yOff+w] (+)= x[r, xOff:xOff+w] */
  copyCols(x: Buf, xCols: number, xOff: number, y: Buf, yCols: number, yOff: number,
           rows: number, w: number, acc: boolean): void;
  /** y[c] (+)= sum_r x[r, c] */
  colSum(x: Buf, y: Buf, rows: number, cols: number, acc: boolean): void;
  /** y[r] (+)= sum_c x[r, c] */
  rowSum(x: Buf, y: Buf, rows: number, cols: number, acc: boolean): void;

  /** Row normalisation; saves per-row [mean, rstd] in `saved` (2*rows). b may be null. */
  norm(kind: NormKind, x: Buf, w: Buf, b: Buf | null, y: Buf, saved: Buf, rows: number, d: number, eps: number): void;
  normGrad(kind: NormKind, x: Buf, w: Buf, saved: Buf, dy: Buf, dx: Buf | null, dw: Buf | null,
           db: Buf | null, rows: number, d: number): void;

  /** Multi-head attention over ragged sequences; qkv [T, 3D], bias [T] (added to every
   *  score against key t). Writes out [T, D] and probs (for the backward and the viz). */
  attention(qkv: Buf, bias: Buf, L: SeqLayout, out: Buf, probs: Buf, H: number, dh: number, scale: number): void;
  attentionGrad(qkv: Buf, probs: Buf, dOut: Buf, L: SeqLayout, dqkv: Buf, dbias: Buf | null,
                H: number, dh: number, scale: number): void;

  /** y = fc(d, rc) for a smooth cutoff of the given width. */
  cutoff(kind: CutoffKind, d: Buf, rc: Buf, y: Buf, n: number, width: number): void;
  cutoffGrad(kind: CutoffKind, d: Buf, rc: Buf, dy: Buf, dd: Buf | null, drc: Buf | null, n: number, width: number): void;

  /** Adam step on p with gradient g and moments m, v. */
  adam(p: Buf, g: Buf, m: Buf, v: Buf, n: number, lr: number, b1: number, b2: number, eps: number, t: number): void;

  /** Block-averaged rows x cols thumbnail of a [R, C] tensor plus summary statistics. */
  thumb(x: Buf, R: number, C: number, rows: number, cols: number): Promise<Thumb>;
}

export const thumbShape = (R: number, C: number, maxR = 48, maxC = 64) =>
  [Math.max(1, Math.min(R, maxR)), Math.max(1, Math.min(C, maxC))] as const;
