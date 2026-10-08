// Tape-based reverse-mode autograd over a Backend. Every op records the module
// scope it ran in, so the visualiser can show each forward and backward step.
import type { Backend, Binary, Buf, CutoffKind, NormKind, SeqLayout, Unary } from './backend';

/** What the rows of a tensor index: lets the visualiser pick one atom's rows. */
export type RowKind = string;

export class Tensor {
  static nextId = 0;
  readonly id = Tensor.nextId++;
  grad: Buf | null = null;
  node: Node | null = null;
  kind?: RowKind;
  constructor(
    readonly shape: number[],
    public buf: Buf,
    public requiresGrad = false,
    public name = '',
  ) {}
  get size() { return this.shape.reduce((a, b) => a * b, 1); }
  get rows() { return this.shape.length > 1 ? this.shape.slice(0, -1).reduce((a, b) => a * b, 1) : this.shape[0] ?? 1; }
  get cols() { return this.shape.length > 1 ? this.shape[this.shape.length - 1] : 1; }
}

export interface Node {
  id: number;
  op: string;
  scope: string;
  inputs: Tensor[];
  out: Tensor;
  /** Extra saved tensors worth showing (e.g. attention probabilities). */
  aux?: Record<string, { buf: Buf; shape: number[] }>;
  backward?: () => void;
}

export interface Tracer {
  forward?(node: Node): void;
  backward?(node: Node): void;
}

/** Row indices known on the host, with the inverse map the backward needs. */
export class Index {
  idx: Buf;
  inv: { off: Buf; src: Buf };
  constructor(be: Backend, readonly host: Int32Array, readonly nSrc: number,
              readonly kind: { out?: RowKind; src?: RowKind } = {}) {
    this.idx = be.uploadI32(host);
    // inverse CSR: for each source row, the outputs that read it
    const cnt = new Int32Array(nSrc + 1);
    for (const j of host) if (j >= 0) cnt[j + 1]++;
    for (let i = 0; i < nSrc; i++) cnt[i + 1] += cnt[i];
    const src = new Int32Array(cnt[nSrc]), fillp = cnt.slice(0, nSrc);
    host.forEach((j, i) => { if (j >= 0) src[fillp[j]++] = i; });
    this.inv = { off: be.uploadI32(cnt), src: be.uploadI32(src) };
  }
  get n() { return this.host.length; }
}

export class Graph {
  tape: Node[] = [];
  private scopes: string[] = [];
  private owned: Buf[] = [];
  tracer: Tracer | null = null;
  recording = true;

  constructor(readonly be: Backend) {}

  scope<T>(name: string, fn: () => T): T {
    this.scopes.push(name);
    try { return fn(); } finally { this.scopes.pop(); }
  }
  get scopeName() { return this.scopes.join('/'); }

  /** A tensor whose buffer this graph frees in release(). */
  tensor(shape: number[], buf?: Buf, name = ''): Tensor {
    const n = shape.reduce((a, b) => a * b, 1);
    const b = buf ?? this.be.zeros(n);
    this.owned.push(b);
    return new Tensor(shape, b, false, name);
  }
  constant(data: Float32Array, shape: number[], name = '', kind?: RowKind) {
    const t = this.tensor(shape, this.be.upload(data), name);
    t.kind = kind;
    return t;
  }
  index(host: Int32Array, nSrc: number, kind: Index['kind'] = {}) {
    const ix = new Index(this.be, host, nSrc, kind);
    this.owned.push(ix.idx, ix.inv.off, ix.inv.src);
    return ix;
  }
  own(b: Buf) { this.owned.push(b); return b; }

  private record(op: string, inputs: Tensor[], out: Tensor, backward: () => void,
                 aux?: Node['aux']): Tensor {
    const node: Node = { id: out.id, op, scope: this.scopeName, inputs, out, aux };
    out.kind ??= inputs.find((t) => t.kind && t.rows === out.rows)?.kind;
    out.requiresGrad = this.recording && inputs.some((t) => t.requiresGrad);
    if (out.requiresGrad) { node.backward = backward; out.node = node; }
    this.tape.push(node);
    this.tracer?.forward?.(node);
    return out;
  }

  /** dX buffer for an input that needs it (allocated zeroed on first use). */
  private g(t: Tensor): Buf | null {
    if (!t.requiresGrad) return null;
    if (!t.grad) t.grad = this.own(this.be.zeros(t.size));
    return t.grad;
  }

  // ---------------------------------------------------------------- ops

  linear(x: Tensor, W: Tensor, b?: Tensor, name = 'linear'): Tensor {
    const [M, K] = [x.rows, x.cols], N = W.shape[0];
    const y = this.tensor([...x.shape.slice(0, -1), N]);
    this.be.matmul(x.buf, W.buf, y.buf, M, N, K, false, true, false, b?.buf);
    return this.record(name, b ? [x, W, b] : [x, W], y, () => {
      const dy = y.grad!, dx = this.g(x), dW = this.g(W), db = b && this.g(b);
      if (dx) this.be.matmul(dy, W.buf, dx, M, K, N, false, false, true);
      if (dW) this.be.matmul(dy, x.buf, dW, N, K, M, true, false, true);
      if (db) this.be.colSum(dy, db, M, N, true);
    });
  }

  unary(op: Unary, x: Tensor, a = 0, b = 0): Tensor {
    const y = this.tensor(x.shape);
    this.be.unary(op, x.buf, y.buf, x.size, a, b);
    return this.record(op, [x], y, () => {
      this.be.unaryGrad(op, x.buf, y.buf, y.grad!, this.g(x)!, x.size, a, b);
    });
  }
  silu(x: Tensor) { return this.unary('silu', x); }
  sigmoid(x: Tensor) { return this.unary('sigmoid', x); }

  binary(op: Binary, a: Tensor, b: Tensor, name: string = op): Tensor {
    // b: same size, a scalar, one per row, or (shape [1, C]) one per column
    const colwise = b.shape.length === 2 && b.shape[0] === 1 && b.size === a.cols && a.rows > 1;
    const mode = b.size === a.size ? 'full' : b.size === 1 ? 'scalar' : colwise ? 'col' : 'row';
    if (mode === 'row' && b.size !== a.rows) throw new Error(`${op}: cannot broadcast ${b.shape} onto ${a.shape}`);
    const y = this.tensor(a.shape);
    this.be.binary(op, a.buf, b.buf, y.buf, a.size, mode, a.cols);
    return this.record(name, [a, b], y, () => {
      this.be.binaryGrad(op, a.buf, b.buf, y.grad!, this.g(a), this.g(b), a.size, mode, a.cols);
    });
  }
  add(a: Tensor, b: Tensor, name = 'add') { return this.binary('add', a, b, name); }
  sub(a: Tensor, b: Tensor, name = 'sub') { return this.binary('sub', a, b, name); }
  mul(a: Tensor, b: Tensor, name = 'mul') { return this.binary('mul', a, b, name); }
  div(a: Tensor, b: Tensor, name = 'div') { return this.binary('div', a, b, name); }

  scale(x: Tensor, alpha: number, name = 'scale'): Tensor {
    const y = this.tensor(x.shape);
    this.be.scale(x.buf, y.buf, alpha, x.size, false);
    return this.record(name, [x], y, () => this.be.scale(y.grad!, this.g(x)!, alpha, x.size, true));
  }

  /** Rows of x picked by idx (rows with idx < 0 are zero). */
  gather(x: Tensor, ix: Index, name = 'gather'): Tensor {
    const d = x.cols, y = this.tensor(x.shape.length > 1 ? [ix.n, d] : [ix.n]);
    y.kind = ix.kind.out;
    this.be.gather(x.buf, ix.idx, y.buf, ix.n, d, false);
    return this.record(name, [x], y, () => {
      this.be.segment(y.grad!, ix.inv.off, ix.inv.src, this.g(x)!, ix.nSrc, d, true);
    });
  }

  /** Sum the rows of x into segments: y[s] = sum_{i: seg[i] = s} x[i]. */
  segmentSum(x: Tensor, seg: Index, name = 'segment_sum'): Tensor {
    const d = x.cols, y = this.tensor(x.shape.length > 1 ? [seg.nSrc, d] : [seg.nSrc]);
    y.kind = seg.kind.src;
    this.be.segment(x.buf, seg.inv.off, seg.inv.src, y.buf, seg.nSrc, d, false);
    return this.record(name, [x], y, () => {
      this.be.gather(y.grad!, seg.idx, this.g(x)!, seg.n, d, true);
    });
  }

  concatCols(xs: Tensor[], name = 'concat'): Tensor {
    const rows = xs[0].rows, w = xs.reduce((s, t) => s + t.cols, 0);
    const y = this.tensor([rows, w]);
    let o = 0;
    for (const t of xs) { this.be.copyCols(t.buf, t.cols, 0, y.buf, w, o, rows, t.cols, false); o += t.cols; }
    return this.record(name, xs, y, () => {
      let o2 = 0;
      for (const t of xs) {
        const g = this.g(t);
        if (g) this.be.copyCols(y.grad!, w, o2, g, t.cols, 0, rows, t.cols, true);
        o2 += t.cols;
      }
    });
  }

  sliceCols(x: Tensor, start: number, width: number, name = 'slice'): Tensor {
    const rows = x.rows, y = this.tensor([rows, width]);
    this.be.copyCols(x.buf, x.cols, start, y.buf, width, 0, rows, width, false);
    return this.record(name, [x], y, () => {
      this.be.copyCols(y.grad!, width, 0, this.g(x)!, x.cols, start, rows, width, true);
    });
  }

  /** Stack rows: [a; b]. */
  concatRows(a: Tensor, b: Tensor, name = 'concat_rows'): Tensor {
    const d = a.cols, y = this.tensor([a.rows + b.rows, d]);
    y.kind = 'stack';
    this.be.scale(a.buf, y.buf, 1, a.size, false);
    this.be.copyCols(b.buf, b.size, 0, y.buf, y.size, a.size, 1, b.size, false);
    return this.record(name, [a, b], y, () => {
      const ga = this.g(a), gb = this.g(b);
      if (ga) this.be.scale(y.grad!, ga, 1, a.size, true);
      if (gb) this.be.copyCols(y.grad!, y.size, a.size, gb, b.size, 0, 1, b.size, true);
    });
  }

  /** Same data, new shape (a copy, so it shows up as its own block). */
  reshape(x: Tensor, shape: number[], kind?: string, name = 'reshape'): Tensor {
    const y = this.tensor(shape);
    y.kind = kind;
    this.be.scale(x.buf, y.buf, 1, x.size, false);
    return this.record(name, [x], y, () => this.be.scale(y.grad!, this.g(x)!, 1, x.size, true));
  }

  /** Repeat a column vector [R] (or [R, 1]) across k columns: [R, k]. */
  repeatCols(x: Tensor, k: number, name = 'repeat'): Tensor {
    return this.linear(x.shape.length > 1 ? x : this.reshape(x, [x.size, 1], x.kind, 'column'),
                       this.constant(new Float32Array(k).fill(1), [k, 1], 'ones'), undefined, name);
  }

  sumRows(x: Tensor, name = 'sum'): Tensor {
    const y = this.tensor([x.rows]);
    this.be.rowSum(x.buf, y.buf, x.rows, x.cols, false);
    return this.record(name, [x], y, () => {
      // dx[r, c] += dy[r]: a gather of dy rows broadcast over columns
      const ones = this.own(this.be.upload(new Float32Array(x.cols).fill(1)));
      this.be.matmul(y.grad!, ones, this.g(x)!, x.rows, x.cols, 1, false, false, true);
    });
  }

  sumAll(x: Tensor, name = 'sum'): Tensor {
    const y = this.tensor([1]);
    this.be.rowSum(x.buf, y.buf, 1, x.size, false);
    return this.record(name, [x], y, () => {
      const ones = this.own(this.be.upload(new Float32Array(x.size).fill(1)));
      this.be.matmul(ones, y.grad!, this.g(x)!, x.size, 1, 1, false, false, true);
    });
  }

  norm(kind: NormKind, x: Tensor, w: Tensor, b: Tensor | null, eps: number): Tensor {
    const rows = x.rows, d = x.cols, y = this.tensor(x.shape);
    const saved = this.own(this.be.zeros(2 * rows));
    this.be.norm(kind, x.buf, w.buf, b?.buf ?? null, y.buf, saved, rows, d, eps);
    return this.record(kind === 'layer' ? 'layer_norm' : 'rms_norm', b ? [x, w, b] : [x, w], y, () => {
      this.be.normGrad(kind, x.buf, w.buf, saved, y.grad!, this.g(x), this.g(w), b ? this.g(b) : null, rows, d);
    });
  }

  attention(qkv: Tensor, bias: Tensor, L: SeqLayout, H: number, temperature: number): Tensor {
    const D = qkv.cols / 3, dh = D / H, scale = 1 / (Math.sqrt(dh) * temperature);
    const out = this.tensor([L.nTok, D]);
    const probs = this.own(this.be.zeros(L.nProb));
    this.be.attention(qkv.buf, bias.buf, L, out.buf, probs, H, dh, scale);
    return this.record('attention', [qkv, bias], out, () => {
      this.be.attentionGrad(qkv.buf, probs, out.grad!, L, this.g(qkv)!, this.g(bias), H, dh, scale);
    }, { probs: { buf: probs, shape: [L.nProb] } });
  }

  cutoff(kind: CutoffKind, d: Tensor, rc: Tensor, width: number): Tensor {
    const y = this.tensor(d.shape);
    this.be.cutoff(kind, d.buf, rc.buf, y.buf, d.size, width);
    return this.record(`cutoff_${kind}`, [d, rc], y, () => {
      this.be.cutoffGrad(kind, d.buf, rc.buf, y.grad!, this.g(d), this.g(rc), d.size, width);
    });
  }

  /** Tabulated functions of x [E]: y[e, c] = spline through V[c, :] and slopes D[c, :] at x = k h. */
  spline(x: Tensor, V: Tensor, D: Tensor, h: number, name = 'spline'): Tensor {
    const [C, K] = V.shape, n = x.size, y = this.tensor([n, C]);
    this.be.spline(x.buf, V.buf, D.buf, y.buf, n, C, K, h);
    return this.record(name, [x, V, D], y, () => this.be.splineGrad(x.buf, V.buf, D.buf, y.grad!, this.g(x)!, n, C, K, h));
  }

  /** Real spherical harmonics of unit vectors u [E, 3]: [E, (lmax+1)^2]. */
  sph(u: Tensor, lmax: number, name = 'spherical_harmonics'): Tensor {
    const n = u.rows, y = this.tensor([n, (lmax + 1) ** 2]);
    this.be.sph(u.buf, y.buf, n, lmax);
    return this.record(name, [u], y, () => this.be.sphGrad(u.buf, y.grad!, this.g(u)!, n, lmax));
  }

  /** SOAP power spectrum of B blocks of A density rows each (see Backend.power). */
  power(c: Tensor, B: number, A: number, lmax: number, name = 'power_spectrum'): Tensor {
    const y = this.tensor([B, ((A * (A + 1)) / 2) * (lmax + 1)]);
    this.be.power(c.buf, y.buf, B, A, lmax);
    return this.record(name, [c], y, () => this.be.powerGrad(c.buf, y.grad!, this.g(c)!, B, A, lmax));
  }

  /** x^T for a [R, C] tensor. */
  transpose(x: Tensor, name = 'transpose'): Tensor {
    const [R, C] = [x.rows, x.cols], y = this.tensor([C, R]);
    const I = this.own(this.be.upload(Float32Array.from({ length: R * R }, (_, k) => (k % (R + 1) === 0 ? 1 : 0))));
    this.be.matmul(x.buf, I, y.buf, C, R, R, true, false, false);
    return this.record(name, [x], y, () => this.be.matmul(I, y.grad!, this.g(x)!, R, C, R, false, true, true));
  }

  /** A result computed elsewhere (e.g. a linear solve on the host), recorded so it can be shown;
   *  no gradient flows through it. */
  opaque(op: string, inputs: Tensor[], data: Float32Array, shape: number[]): Tensor {
    const y = this.tensor(shape, this.be.upload(data));
    const wasRecording = this.recording;
    this.recording = false;
    try { return this.record(op, inputs, y, () => {}); } finally { this.recording = wasRecording; }
  }

  /** |v| per row of a [E, 3] tensor, as sqrt(sum v^2 + eps). */
  rowNorm(v: Tensor, eps = 1e-15): Tensor {
    const sq = this.unary('square', v);
    const s = this.sumRows(sq, 'sum_sq');
    const shifted = this.binary('add', s, this.constant(new Float32Array([eps]), [1]));
    return this.unary('sqrt', shifted);
  }

  // ------------------------------------------------------------ backward

  /** Reverse pass from a scalar (or with an explicit seed gradient). */
  backward(root: Tensor, seed?: Float32Array) {
    if (!root.requiresGrad) return;
    root.grad = this.own(this.be.upload(seed ?? new Float32Array(root.size).fill(1)));
    for (let i = this.tape.length - 1; i >= 0; i--) {
      const node = this.tape[i];
      if (!node.backward || !node.out.grad) continue;
      this.tracer?.backward?.(node);
      node.backward();
    }
  }

  /** Free every buffer this graph allocated (parameters live elsewhere). */
  release() {
    for (const b of this.owned) this.be.free(b);
    this.owned = [];
    this.tape = [];
  }
}
