// WebGPU backend: every kernel of backend.ts as a WGSL compute shader. Commands
// are recorded into one encoder and submitted when a result is read back, so a
// whole forward + backward pass is one submission.
import type { Backend, Binary, BMode, Buf, CutoffKind, NormKind, SeqLayout, Thumb, Unary } from './backend';

type GBuf = Buf & { g: GPUBuffer; cap: number };
const gb = (b: Buf) => (b as GBuf).g;

const UNARY: Record<Unary, number> = { silu: 0, sigmoid: 1, exp: 2, square: 3, sqrt: 4, neg: 5, tanh: 6, logclamp: 7, clamp: 8, acos: 9, cos: 10, pow: 11, celu: 12 };
const BINARY: Record<Binary, number> = { add: 0, sub: 1, mul: 2, div: 3 };
const BMODE = { full: 0, scalar: 1, row: 2, col: 3 } as const;
const WG = 256;

// --------------------------------------------------------------------------- WGSL
const idx1 = /* wgsl */ `
fn gidx(g: vec3u, n: vec3u) -> u32 { return g.x + g.y * n.x * ${WG}u; }`;

const sh = {
  fill: `
struct P { n: u32, v: f32 }
@group(0) @binding(0) var<storage, read_write> y: array<f32>;
@group(0) @binding(1) var<uniform> p: P;
${idx1}
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let i = gidx(g, nw); if (i < p.n) { y[i] = p.v; }
}`,
  scale: `
struct P { n: u32, a: f32, acc: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
${idx1}
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let i = gidx(g, nw); if (i >= p.n) { return; }
  var v = p.a * x[i]; if (p.acc != 0u) { v += y[i]; } y[i] = v;
}`,
  matmul: `
struct P { M: u32, N: u32, K: u32, tA: u32, tB: u32, acc: u32, bias: u32 }
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
const T = 16u;
const R = 4u; // outputs per thread along each axis: 64x64 tile per workgroup
var<workgroup> As: array<array<f32, 64>, 16>; // [k][m]
var<workgroup> Bs: array<array<f32, 64>, 16>; // [k][n]
fn a(m: u32, k: u32) -> f32 {
  if (m >= p.M || k >= p.K) { return 0.0; }
  if (p.tA != 0u) { return A[k * p.M + m]; } return A[m * p.K + k];
}
fn b(k: u32, n: u32) -> f32 {
  if (k >= p.K || n >= p.N) { return 0.0; }
  if (p.tB != 0u) { return B[n * p.K + k]; } return B[k * p.N + n];
}
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let m0 = wg.y * 64u; let n0 = wg.x * 64u;
  var s: array<array<f32, 4>, 4>;
  let lid = l.y * 16u + l.x;
  for (var k0 = 0u; k0 < p.K; k0 += T) {
    // 16 x 64 elements of each tile, 4 per thread
    for (var r = 0u; r < 4u; r++) {
      let e = lid + r * 256u; let kk = e / 64u; let mm = e % 64u;
      As[kk][mm] = a(m0 + mm, k0 + kk);
      Bs[kk][mm] = b(k0 + kk, n0 + mm);
    }
    workgroupBarrier();
    for (var k = 0u; k < T; k++) {
      var av: array<f32, 4>; var bv: array<f32, 4>;
      for (var i = 0u; i < R; i++) { av[i] = As[k][l.y + 16u * i]; bv[i] = Bs[k][l.x + 16u * i]; }
      for (var i = 0u; i < R; i++) { for (var j = 0u; j < R; j++) { s[i][j] += av[i] * bv[j]; } }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < R; i++) {
    let m = m0 + l.y + 16u * i; if (m >= p.M) { continue; }
    for (var j = 0u; j < R; j++) {
      let n = n0 + l.x + 16u * j; if (n >= p.N) { continue; }
      var v = s[i][j];
      if (p.bias != 0u) { v += bias[n]; }
      if (p.acc != 0u) { v += C[m * p.N + n]; }
      C[m * p.N + n] = v;
    }
  }
}`,
  unary: `
struct P { n: u32, op: u32, a: f32, b: f32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
${idx1}
fn sig(v: f32) -> f32 { return 1.0 / (1.0 + exp(-v)); }
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let i = gidx(g, nw); if (i >= p.n) { return; }
  let v = x[i]; var r = 0.0;
  switch p.op {
    case 0u: { r = v * sig(v); }
    case 1u: { r = sig(v); }
    case 2u: { r = exp(v); }
    case 3u: { r = v * v; }
    case 4u: { r = sqrt(v); }
    case 5u: { r = -v; }
    case 6u: { r = tanh(clamp(v, -20.0, 20.0)); }
    case 7u: { r = log(max(v, p.a)); }
    case 8u: { r = min(max(v, p.a), p.b); }
    case 9u: { r = acos(v); }
    case 10u: { r = cos(v); }
    case 11u: { r = pow(v, p.a); }
    default: { r = select(p.a * (exp(v / p.a) - 1.0), v, v > 0.0); }
  }
  y[i] = r;
}`,
  unaryGrad: `
struct P { n: u32, op: u32, a: f32, b: f32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read> dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
${idx1}
fn sig(v: f32) -> f32 { return 1.0 / (1.0 + exp(-v)); }
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let i = gidx(g, nw); if (i >= p.n) { return; }
  let v = x[i]; let o = y[i]; var d = 0.0;
  switch p.op {
    case 0u: { let s = sig(v); d = s * (1.0 + v * (1.0 - s)); }
    case 1u: { d = o * (1.0 - o); }
    case 2u: { d = o; }
    case 3u: { d = 2.0 * v; }
    case 4u: { d = 0.5 / o; }
    case 5u: { d = -1.0; }
    case 6u: { d = 1.0 - o * o; }
    case 7u: { d = select(0.0, 1.0 / v, v >= p.a); }
    case 8u: { d = select(0.0, 1.0, v >= p.a && v <= p.b); }
    case 9u: { d = -1.0 / sqrt(1.0 - v * v); }
    case 10u: { d = -sin(v); }
    case 11u: { d = p.a * pow(v, p.a - 1.0); }
    default: { d = select(exp(v / p.a), 1.0, v > 0.0); }
  }
  dx[i] += d * dy[i];
}`,
  binary: `
struct P { n: u32, op: u32, mode: u32, inner: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
${idx1}
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let i = gidx(g, nw); if (i >= p.n) { return; }
  var j = i; if (p.mode == 1u) { j = 0u; } else if (p.mode == 2u) { j = i / p.inner; } else if (p.mode == 3u) { j = i % p.inner; }
  let u = a[i]; let v = b[j]; var r = 0.0;
  switch p.op { case 0u: { r = u + v; } case 1u: { r = u - v; } case 2u: { r = u * v; } default: { r = u / v; } }
  y[i] = r;
}`,
  // da for every mode, and db when b is full-size
  binaryGradA: `
struct P { n: u32, op: u32, mode: u32, inner: u32, wantA: u32, wantB: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> da: array<f32>;
@group(0) @binding(4) var<storage, read_write> db: array<f32>;
@group(0) @binding(5) var<uniform> p: P;
${idx1}
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let i = gidx(g, nw); if (i >= p.n) { return; }
  var j = i; if (p.mode == 1u) { j = 0u; } else if (p.mode == 2u) { j = i / p.inner; } else if (p.mode == 3u) { j = i % p.inner; }
  let u = a[i]; let v = b[j]; let gr = dy[i];
  if (p.wantA != 0u) {
    var d = gr; if (p.op == 2u) { d = gr * v; } else if (p.op == 3u) { d = gr / v; }
    da[i] += d;
  }
  if (p.wantB != 0u && p.mode == 0u) {
    var d = gr; if (p.op == 1u) { d = -gr; } else if (p.op == 2u) { d = gr * u; } else if (p.op == 3u) { d = -gr * u / (v * v); }
    db[i] += d;
  }
}`,
  // db for broadcast b: one thread per element of b
  binaryGradB: `
struct P { nb: u32, op: u32, count: u32, stride: u32, step: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> db: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
${idx1}
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let j = gidx(g, nw); if (j >= p.nb) { return; }
  let v = b[j]; var s = 0.0;
  for (var k = 0u; k < p.count; k++) {
    let i = j * p.stride + k * p.step; let gr = dy[i];
    var d = gr; if (p.op == 1u) { d = -gr; } else if (p.op == 2u) { d = gr * a[i]; } else if (p.op == 3u) { d = -gr * a[i] / (v * v); }
    s += d;
  }
  db[j] += s;
}`,
  gather: `
struct P { nOut: u32, d: u32, acc: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> idx: array<i32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
${idx1}
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let e = gidx(g, nw); if (e >= p.nOut * p.d) { return; }
  let i = e / p.d; let c = e % p.d; let j = idx[i];
  var v = 0.0; if (j >= 0) { v = x[u32(j) * p.d + c]; }
  if (p.acc != 0u) { v += y[e]; } y[e] = v;
}`,
  segment: `
struct P { nOut: u32, d: u32, acc: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> off: array<i32>;
@group(0) @binding(2) var<storage, read> src: array<i32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
${idx1}
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let e = gidx(g, nw); if (e >= p.nOut * p.d) { return; }
  let t = e / p.d; let c = e % p.d;
  var v = 0.0;
  for (var k = off[t]; k < off[t + 1u]; k++) { v += x[u32(src[k]) * p.d + c]; }
  if (p.acc != 0u) { v += y[e]; } y[e] = v;
}`,
  copyCols: `
struct P { rows: u32, w: u32, xCols: u32, xOff: u32, yCols: u32, yOff: u32, acc: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
${idx1}
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let e = gidx(g, nw); if (e >= p.rows * p.w) { return; }
  let r = e / p.w; let c = e % p.w;
  let yi = r * p.yCols + p.yOff + c;
  var v = x[r * p.xCols + p.xOff + c]; if (p.acc != 0u) { v += y[yi]; } y[yi] = v;
}`,
  // y[i] (+)= sum over the strided axis: colSum (stride = cols) and rowSum (stride = 1)
  reduce: `
struct P { nOut: u32, len: u32, outStride: u32, inStride: u32, acc: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
${idx1}
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let i = gidx(g, nw); if (i >= p.nOut) { return; }
  var s = 0.0;
  for (var k = 0u; k < p.len; k++) { s += x[i * p.outStride + k * p.inStride]; }
  if (p.acc != 0u) { s += y[i]; } y[i] = s;
}`,
  norm: `
struct P { rows: u32, d: u32, rms: u32, hasB: u32, eps: f32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read_write> saved: array<f32>;
@group(0) @binding(5) var<uniform> p: P;
${idx1}
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let r = gidx(g, nw); if (r >= p.rows) { return; }
  let o = r * p.d;
  var mean = 0.0;
  if (p.rms == 0u) { for (var c = 0u; c < p.d; c++) { mean += x[o + c]; } mean /= f32(p.d); }
  var v = 0.0;
  for (var c = 0u; c < p.d; c++) { let t = x[o + c] - mean; v += t * t; }
  let rstd = 1.0 / sqrt(v / f32(p.d) + p.eps);
  saved[2u * r] = mean; saved[2u * r + 1u] = rstd;
  for (var c = 0u; c < p.d; c++) {
    var bb = 0.0; if (p.hasB != 0u) { bb = b[c]; }
    y[o + c] = (x[o + c] - mean) * rstd * w[c] + bb;
  }
}`,
  normGradX: `
struct P { rows: u32, d: u32, rms: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> saved: array<f32>;
@group(0) @binding(3) var<storage, read> dy: array<f32>;
@group(0) @binding(4) var<storage, read_write> dx: array<f32>;
@group(0) @binding(5) var<uniform> p: P;
${idx1}
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let r = gidx(g, nw); if (r >= p.rows) { return; }
  let o = r * p.d; let mean = saved[2u * r]; let rstd = saved[2u * r + 1u];
  var sg = 0.0; var sgx = 0.0;
  for (var c = 0u; c < p.d; c++) {
    let xh = (x[o + c] - mean) * rstd; let gg = dy[o + c] * w[c];
    sg += gg; sgx += gg * xh;
  }
  if (p.rms != 0u) { sg = 0.0; }
  let d = f32(p.d);
  for (var c = 0u; c < p.d; c++) {
    let xh = (x[o + c] - mean) * rstd;
    dx[o + c] += rstd * (dy[o + c] * w[c] - sg / d - xh * sgx / d);
  }
}`,
  normGradW: `
struct P { rows: u32, d: u32, wantW: u32, wantB: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> saved: array<f32>;
@group(0) @binding(2) var<storage, read> dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> dw: array<f32>;
@group(0) @binding(4) var<storage, read_write> db: array<f32>;
@group(0) @binding(5) var<uniform> p: P;
${idx1}
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let c = gidx(g, nw); if (c >= p.d) { return; }
  var sw = 0.0; var sb = 0.0;
  for (var r = 0u; r < p.rows; r++) {
    let i = r * p.d + c;
    sw += dy[i] * (x[i] - saved[2u * r]) * saved[2u * r + 1u]; sb += dy[i];
  }
  if (p.wantW != 0u) { dw[c] += sw; }
  if (p.wantB != 0u) { db[c] += sb; }
}`,
  // one thread per (query token, head)
  attention: `
struct P { nTok: u32, H: u32, dh: u32, scale: f32 }
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read> off: array<i32>;
@group(0) @binding(3) var<storage, read> poff: array<i32>;
@group(0) @binding(4) var<storage, read> tok2seq: array<i32>;
@group(0) @binding(5) var<storage, read_write> outp: array<f32>;
@group(0) @binding(6) var<storage, read_write> probs: array<f32>;
@group(0) @binding(7) var<uniform> p: P;
${idx1}
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let e = gidx(g, nw); if (e >= p.nTok * p.H) { return; }
  let t = e / p.H; let h = e % p.H;
  let a = u32(tok2seq[t]); let t0 = u32(off[a]); let n = u32(off[a + 1u]) - t0; let i = t - t0;
  let D = p.H * p.dh; let D3 = 3u * D;
  let po = u32(poff[a]) + (h * n + i) * n;
  let qo = t * D3 + h * p.dh;
  var mx = -3.4e38;
  for (var j = 0u; j < n; j++) {
    let ko = (t0 + j) * D3 + D + h * p.dh;
    var s = 0.0;
    for (var c = 0u; c < p.dh; c++) { s += qkv[qo + c] * qkv[ko + c]; }
    s = s * p.scale + bias[t0 + j];
    probs[po + j] = s; mx = max(mx, s);
  }
  var z = 0.0;
  for (var j = 0u; j < n; j++) { let v = exp(probs[po + j] - mx); probs[po + j] = v; z += v; }
  let oo = t * D + h * p.dh;
  for (var c = 0u; c < p.dh; c++) { outp[oo + c] = 0.0; }
  for (var j = 0u; j < n; j++) {
    let pr = probs[po + j] / z; probs[po + j] = pr;
    let vo = (t0 + j) * D3 + 2u * D + h * p.dh;
    for (var c = 0u; c < p.dh; c++) { outp[oo + c] += pr * qkv[vo + c]; }
  }
}`,
  // pass 1, per (query, head): rowdot D_i = sum_j P_ij dP_ij and dq
  attnGradQ: `
struct P { nTok: u32, H: u32, dh: u32, scale: f32 }
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> probs: array<f32>;
@group(0) @binding(2) var<storage, read> dout: array<f32>;
@group(0) @binding(3) var<storage, read> off: array<i32>;
@group(0) @binding(4) var<storage, read> poff: array<i32>;
@group(0) @binding(5) var<storage, read> tok2seq: array<i32>;
@group(0) @binding(6) var<storage, read_write> dqkv: array<f32>;
@group(0) @binding(7) var<storage, read_write> rowdot: array<f32>;
@group(0) @binding(8) var<uniform> p: P;
${idx1}
fn dP(t: u32, j: u32, h: u32) -> f32 {
  let D = p.H * p.dh; let oo = t * D + h * p.dh; let vo = j * 3u * D + 2u * D + h * p.dh;
  var s = 0.0; for (var c = 0u; c < p.dh; c++) { s += dout[oo + c] * qkv[vo + c]; } return s;
}
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let e = gidx(g, nw); if (e >= p.nTok * p.H) { return; }
  let t = e / p.H; let h = e % p.H;
  let a = u32(tok2seq[t]); let t0 = u32(off[a]); let n = u32(off[a + 1u]) - t0; let i = t - t0;
  let D = p.H * p.dh; let D3 = 3u * D;
  let po = u32(poff[a]) + (h * n + i) * n;
  var dot = 0.0;
  for (var j = 0u; j < n; j++) { dot += probs[po + j] * dP(t, t0 + j, h); }
  rowdot[e] = dot;
  let qo = t * D3 + h * p.dh;
  for (var j = 0u; j < n; j++) {
    let gs = probs[po + j] * (dP(t, t0 + j, h) - dot) * p.scale;
    let ko = (t0 + j) * D3 + D + h * p.dh;
    for (var c = 0u; c < p.dh; c++) { dqkv[qo + c] += gs * qkv[ko + c]; }
  }
}`,
  // pass 2, per (key, head): dk, dv and the bias gradient per head
  attnGradKV: `
struct P { nTok: u32, H: u32, dh: u32, scale: f32 }
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> probs: array<f32>;
@group(0) @binding(2) var<storage, read> dout: array<f32>;
@group(0) @binding(3) var<storage, read> off: array<i32>;
@group(0) @binding(4) var<storage, read> poff: array<i32>;
@group(0) @binding(5) var<storage, read> tok2seq: array<i32>;
@group(0) @binding(6) var<storage, read_write> dqkv: array<f32>;
@group(0) @binding(7) var<storage, read_write> rowdot: array<f32>;
@group(0) @binding(8) var<uniform> p: P;
${idx1}
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let e = gidx(g, nw); if (e >= p.nTok * p.H) { return; }
  let t = e / p.H; let h = e % p.H;
  let a = u32(tok2seq[t]); let t0 = u32(off[a]); let n = u32(off[a + 1u]) - t0; let j = t - t0;
  let D = p.H * p.dh; let D3 = 3u * D;
  let ko = t * D3 + D + h * p.dh; let vo = t * D3 + 2u * D + h * p.dh;
  for (var i = 0u; i < n; i++) {
    let q = t0 + i;
    let pr = probs[u32(poff[a]) + (h * n + i) * n + j];
    let oo = q * D + h * p.dh;
    var dp = 0.0; for (var c = 0u; c < p.dh; c++) { dp += dout[oo + c] * qkv[vo + c]; }
    let gs = pr * (dp - rowdot[q * p.H + h]);
    let qo = q * D3 + h * p.dh;
    for (var c = 0u; c < p.dh; c++) {
      dqkv[ko + c] += gs * p.scale * qkv[qo + c];
      dqkv[vo + c] += pr * dout[oo + c];
    }
  }
}`,
  // pass 3: per (key, head) bias gradients (recomputed) summed over heads
  attnGradBias: `
struct P { nTok: u32, H: u32, dh: u32, scale: f32 }
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> probs: array<f32>;
@group(0) @binding(2) var<storage, read> dout: array<f32>;
@group(0) @binding(3) var<storage, read> off: array<i32>;
@group(0) @binding(4) var<storage, read> poff: array<i32>;
@group(0) @binding(5) var<storage, read> tok2seq: array<i32>;
@group(0) @binding(6) var<storage, read_write> dbias: array<f32>;
@group(0) @binding(7) var<storage, read_write> rowdot: array<f32>;
@group(0) @binding(8) var<uniform> p: P;
${idx1}
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let t = gidx(g, nw); if (t >= p.nTok) { return; }
  let a = u32(tok2seq[t]); let t0 = u32(off[a]); let n = u32(off[a + 1u]) - t0; let j = t - t0;
  let D = p.H * p.dh; let D3 = 3u * D;
  var db = 0.0;
  for (var h = 0u; h < p.H; h++) {
    let vo = t * D3 + 2u * D + h * p.dh;
    for (var i = 0u; i < n; i++) {
      let q = t0 + i;
      let pr = probs[u32(poff[a]) + (h * n + i) * n + j];
      let oo = q * D + h * p.dh;
      var dp = 0.0; for (var c = 0u; c < p.dh; c++) { dp += dout[oo + c] * qkv[vo + c]; }
      db += pr * (dp - rowdot[q * p.H + h]);
    }
  }
  dbias[t] += db;
}`,
  cutoff: `
struct P { n: u32, bump: u32, w: f32, grad: u32, wantD: u32, wantR: u32 }
@group(0) @binding(0) var<storage, read> d: array<f32>;
@group(0) @binding(1) var<storage, read> rc: array<f32>;
@group(0) @binding(2) var<storage, read> dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;   // value, or dd
@group(0) @binding(4) var<storage, read_write> drc: array<f32>;
@group(0) @binding(5) var<uniform> p: P;
${idx1}
const PI = 3.14159265358979;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let i = gidx(g, nw); if (i >= p.n) { return; }
  let s = (d[i] - (rc[i] - p.w)) / p.w;
  var f = 0.0; var dfds = 0.0;
  if (p.bump == 0u) {
    let c = clamp(s, 0.0, 1.0);
    f = 0.5 * (1.0 + cos(PI * c));
    dfds = select(0.0, -0.5 * PI * sin(PI * c), s >= 0.0 && s <= 1.0);
  } else {
    let lo = 1e-6; let hi = 1.0 - 1e-6;
    let c = clamp(s, lo, hi);
    let sn = sin(PI * c); let ct = cos(PI * c) / sn; let th = tanh(clamp(ct, -20.0, 20.0));
    f = 0.5 * (1.0 + th);
    dfds = select(0.0, -0.5 * PI * (1.0 - th * th) / (sn * sn), s >= lo && s <= hi);
  }
  if (p.grad == 0u) { y[i] = f; return; }
  let gg = dfds * dy[i] / p.w;
  if (p.wantD != 0u) { y[i] += gg; }
  if (p.wantR != 0u) { drc[i] -= gg; }
}`,
  adam: `
struct P { n: u32, lr: f32, b1: f32, b2: f32, eps: f32, c1: f32, c2: f32 }
@group(0) @binding(0) var<storage, read_write> w: array<f32>;
@group(0) @binding(1) var<storage, read> gr: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
${idx1}
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let i = gidx(g, nw); if (i >= p.n) { return; }
  m[i] = p.b1 * m[i] + (1.0 - p.b1) * gr[i];
  v[i] = p.b2 * v[i] + (1.0 - p.b2) * gr[i] * gr[i];
  w[i] -= p.lr * (m[i] / p.c1) / (sqrt(v[i] / p.c2) + p.eps);
}`,
  spline: `
struct P { n: u32, C: u32, K: u32, h: f32, grad: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> V: array<f32>;
@group(0) @binding(2) var<storage, read> D: array<f32>;
@group(0) @binding(3) var<storage, read> dy: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>; // values, or dx
@group(0) @binding(5) var<uniform> p: P;
${idx1}
fn herm(c: u32, xv: f32) -> vec2f {
  let t = xv / p.h; let kf = floor(t);
  if (kf < 0.0 || kf >= f32(p.K - 1u)) { return vec2f(0.0, 0.0); }
  let k = u32(kf); let s = t - kf; let o = c * p.K + k;
  let s2 = s * s; let s3 = s2 * s;
  let v = (2.0 * s3 - 3.0 * s2 + 1.0) * V[o] + (s3 - 2.0 * s2 + s) * p.h * D[o] + (-2.0 * s3 + 3.0 * s2) * V[o + 1u] + (s3 - s2) * p.h * D[o + 1u];
  let dv = ((6.0 * s2 - 6.0 * s) * V[o] + (3.0 * s2 - 4.0 * s + 1.0) * p.h * D[o] + (-6.0 * s2 + 6.0 * s) * V[o + 1u] + (3.0 * s2 - 2.0 * s) * p.h * D[o + 1u]) / p.h;
  return vec2f(v, dv);
}
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  if (p.grad == 0u) {
    let i = gidx(g, nw); if (i >= p.n * p.C) { return; }
    y[i] = herm(i % p.C, x[i / p.C]).x;
  } else {
    let e = gidx(g, nw); if (e >= p.n) { return; }
    var s = 0.0;
    for (var c = 0u; c < p.C; c++) { s += dy[e * p.C + c] * herm(c, x[e]).y; }
    y[e] += s;
  }
}`,
  sph: `
struct P { n: u32, L: u32, grad: u32 }
@group(0) @binding(0) var<storage, read> u: array<f32>;
@group(0) @binding(1) var<storage, read> dy: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>; // Y, or du
@group(0) @binding(3) var<uniform> p: P;
${idx1}
const PI = 3.14159265358979;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let e = gidx(g, nw); if (e >= p.n) { return; }
  let x = u[3u * e]; let yy = u[3u * e + 1u]; let z = u[3u * e + 2u];
  let M = (p.L + 1u) * (p.L + 1u);
  var C: array<f32, 7>; var S: array<f32, 7>; var Cx: array<f32, 7>; var Cy: array<f32, 7>; var Sx: array<f32, 7>; var Sy: array<f32, 7>;
  C[0] = 1.0; S[0] = 0.0; Cx[0] = 0.0; Cy[0] = 0.0; Sx[0] = 0.0; Sy[0] = 0.0;
  for (var m = 0u; m < p.L; m++) {
    C[m + 1u] = x * C[m] - yy * S[m]; S[m + 1u] = x * S[m] + yy * C[m];
    Cx[m + 1u] = C[m] + x * Cx[m] - yy * Sx[m]; Cy[m + 1u] = x * Cy[m] - S[m] - yy * Sy[m];
    Sx[m + 1u] = S[m] + x * Sx[m] + yy * Cx[m]; Sy[m + 1u] = x * Sy[m] + C[m] + yy * Cy[m];
  }
  var gx = 0.0; var gy = 0.0; var gz = 0.0;
  for (var m = 0u; m <= p.L; m++) {
    var q = 1.0; var d = 0.0; var qPrev = 0.0; var dPrev = 0.0;
    for (var k = 1u; k + 1u <= 2u * m; k += 2u) { q *= f32(k); }
    for (var l = m; l <= p.L; l++) {
      if (l > m) {
        var qn = 0.0; var dn = 0.0;
        if (l == m + 1u) { qn = f32(2u * m + 1u) * z * q; dn = f32(2u * m + 1u) * q; }
        else {
          qn = (f32(2u * l - 1u) * z * q - f32(l + m - 1u) * qPrev) / f32(l - m);
          dn = (f32(2u * l - 1u) * (q + z * d) - f32(l + m - 1u) * dPrev) / f32(l - m);
        }
        qPrev = q; dPrev = d; q = qn; d = dn;
      }
      var fact = 1.0;
      for (var k = l - m + 1u; k <= l + m; k++) { fact *= f32(k); }
      var N = sqrt(f32(2u * l + 1u) / (4.0 * PI) / fact);
      if (m > 0u) { N *= 1.41421356237; }
      let ip = l * l + l + m; let im = l * l + l - m;
      if (p.grad == 0u) {
        y[e * M + ip] = N * q * C[m];
        if (m > 0u) { y[e * M + im] = N * q * S[m]; }
      } else {
        let gp = dy[e * M + ip];
        gx += gp * N * q * Cx[m]; gy += gp * N * q * Cy[m]; gz += gp * N * d * C[m];
        if (m > 0u) {
          let gm = dy[e * M + im];
          gx += gm * N * q * Sx[m]; gy += gm * N * q * Sy[m]; gz += gm * N * d * S[m];
        }
      }
    }
  }
  if (p.grad != 0u) { y[3u * e] += gx; y[3u * e + 1u] += gy; y[3u * e + 2u] += gz; }
}`,
  // forward: one thread per (block, pair, l)
  power: `
struct P { B: u32, A: u32, L: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> pw: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
${idx1}
const PI = 3.14159265358979;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let NP = p.A * (p.A + 1u) / 2u; let L1 = p.L + 1u; let M = L1 * L1;
  let i = gidx(g, nw); if (i >= p.B * NP * L1) { return; }
  let l = i % L1; let k = (i / L1) % NP; let b = i / (L1 * NP);
  // pair index k -> (a, a2) with a <= a2
  var a = 0u; var start = 0u;
  loop { let rowLen = p.A - a; if (k < start + rowLen) { break; } start += rowLen; a++; }
  let a2 = a + (k - start);
  let ro = (b * p.A + a) * M; let r2 = (b * p.A + a2) * M;
  var s = 0.0;
  for (var q = l * l; q < (l + 1u) * (l + 1u); q++) { s += x[ro + q] * x[r2 + q]; }
  var w = 1.0; if (a != a2) { w = 1.41421356237; }
  pw[i] = w * PI * sqrt(8.0 / f32(2u * l + 1u)) * s;
}`,
  // backward: one thread per element of x
  powerGrad: `
struct P { B: u32, A: u32, L: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> dp: array<f32>;
@group(0) @binding(2) var<storage, read_write> dx: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
${idx1}
const PI = 3.14159265358979;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let L1 = p.L + 1u; let M = L1 * L1; let NP = p.A * (p.A + 1u) / 2u;
  let i = gidx(g, nw); if (i >= p.B * p.A * M) { return; }
  let q = i % M; let row = i / M; let a = row % p.A; let b = row / p.A;
  var l = 0u; loop { if ((l + 1u) * (l + 1u) > q) { break; } l++; }
  let f = PI * sqrt(8.0 / f32(2u * l + 1u));
  var s = 0.0;
  for (var a2 = 0u; a2 < p.A; a2++) {
    let lo = min(a, a2); let hi = max(a, a2);
    let k = lo * p.A - (lo * (lo - 1u)) / 2u + (hi - lo);
    var w = 1.41421356237; if (a == a2) { w = 2.0; }
    s += w * f * dp[(b * NP + k) * L1 + l] * x[(b * p.A + a2) * M + q];
  }
  dx[i] += s;
}`,
  // thumbnail cell = block mean, written at an offset into a shared output
  thumb: `
struct P { R: u32, C: u32, rows: u32, cols: u32, o: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
${idx1}
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let e = gidx(g, nw); if (e >= p.rows * p.cols) { return; }
  let r = e / p.cols; let c = e % p.cols;
  let r0 = (r * p.R) / p.rows; let r1 = max(r0 + 1u, ((r + 1u) * p.R) / p.rows);
  let c0 = (c * p.C) / p.cols; let c1 = max(c0 + 1u, ((c + 1u) * p.C) / p.cols);
  var s = 0.0;
  for (var i = r0; i < r1; i++) { for (var j = c0; j < c1; j++) { s += x[i * p.C + j]; } }
  y[p.o + e] = s / f32((r1 - r0) * (c1 - c0));
}`,
  // sum, sum of squares, min, max with one workgroup
  stats: `
struct P { n: u32, o: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
var<workgroup> s1: array<f32, 256>; var<workgroup> s2: array<f32, 256>;
var<workgroup> mn: array<f32, 256>; var<workgroup> mx: array<f32, 256>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_id) l: vec3u) {
  var a = 0.0; var b = 0.0; var lo = 3.4e38; var hi = -3.4e38;
  for (var i = l.x; i < p.n; i += 256u) { let v = x[i]; a += v; b += v * v; lo = min(lo, v); hi = max(hi, v); }
  s1[l.x] = a; s2[l.x] = b; mn[l.x] = lo; mx[l.x] = hi;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (l.x < k) {
      s1[l.x] += s1[l.x + k]; s2[l.x] += s2[l.x + k];
      mn[l.x] = min(mn[l.x], mn[l.x + k]); mx[l.x] = max(mx[l.x], mx[l.x + k]);
    }
    workgroupBarrier();
  }
  if (l.x == 0u) { y[p.o] = s1[0]; y[p.o + 1u] = s2[0]; y[p.o + 2u] = mn[0]; y[p.o + 3u] = mx[0]; }
}`,
};

type Kernel = keyof typeof sh;
type Param = number | { f: number }; // integers are u32, {f} are f32
const F = (f: number) => ({ f });

export class WebGPUBackend implements Backend {
  readonly name = 'webgpu' as const;
  private pipes = new Map<Kernel, GPUComputePipeline>();
  private pool = new Map<number, GPUBuffer[]>();
  private enc: GPUCommandEncoder | null = null;
  private uni: GPUBuffer;
  private uniData = new ArrayBuffer(1 << 20);
  private uniUsed = 0;
  private readonly uniAlign: number;
  private dummy: GPUBuffer;
  dispatches = 0;
  owners: unknown[] = [];

  static async create(gpu?: GPU): Promise<WebGPUBackend> {
    const g = gpu ?? (globalThis.navigator as any)?.gpu;
    if (!g) throw new Error('WebGPU is not available in this browser');
    const adapter = await g.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('no WebGPU adapter');
    const lim = adapter.limits;
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBuffersPerShaderStage: Math.min(10, lim.maxStorageBuffersPerShaderStage),
        maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
        maxBufferSize: lim.maxBufferSize,
      },
    });
    const be = new WebGPUBackend(device, adapter.info?.description || adapter.info?.vendor || 'GPU');
    // keep the instance and adapter alive as long as the device (Dawn's Node
    // bindings free the instance when it is collected)
    be.owners = [g, adapter];
    return be;
  }

  constructor(readonly device: GPUDevice, readonly adapterName = 'GPU') {
    this.uniAlign = device.limits.minUniformBufferOffsetAlignment;
    this.uni = device.createBuffer({ size: this.uniData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.dummy = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
  }

  // ---------------------------------------------------------------- buffers
  private alloc(n: number): GPUBuffer {
    const cap = 1 << Math.max(4, Math.ceil(Math.log2(Math.max(n, 1) * 4)));
    const free = this.pool.get(cap);
    if (free?.length) return free.pop()!;
    return this.device.createBuffer({ size: cap, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  }
  private wrap(g: GPUBuffer, n: number, i32 = false): GBuf { return { n, i32, g, cap: g.size }; }

  zeros(n: number): Buf {
    const g = this.alloc(n);
    this.encoder().clearBuffer(g);
    return this.wrap(g, n);
  }
  private uploadRaw(data: Float32Array | Int32Array, i32: boolean): Buf {
    const n = data.length;
    const g = this.device.createBuffer({
      size: 1 << Math.max(4, Math.ceil(Math.log2(Math.max(n, 1) * 4))),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, mappedAtCreation: true,
    });
    (i32 ? new Int32Array(g.getMappedRange()) : new Float32Array(g.getMappedRange())).set(data as any);
    g.unmap();
    return this.wrap(g, n, i32);
  }
  upload(data: Float32Array) { return this.uploadRaw(data, false); }
  uploadI32(data: Int32Array) { return this.uploadRaw(data, true); }
  free(b: Buf) {
    const g = gb(b);
    const list = this.pool.get(g.size) ?? [];
    list.push(g);
    this.pool.set(g.size, list);
  }

  private encoder() { return (this.enc ??= this.device.createCommandEncoder()); }

  private flush() {
    if (!this.enc) return;
    if (this.uniUsed) this.device.queue.writeBuffer(this.uni, 0, this.uniData, 0, this.uniUsed);
    this.device.queue.submit([this.enc.finish()]);
    this.enc = null;
    this.uniUsed = 0;
  }
  async sync() { this.flush(); await this.device.queue.onSubmittedWorkDone(); }

  async read(b: Buf): Promise<Float32Array> {
    return (await this.readMany([b]))[0];
  }
  async readMany(bs: Buf[]): Promise<Float32Array[]> {
    const total = bs.reduce((s, b) => s + Math.max(b.n, 1) * 4, 0);
    const stage = this.device.createBuffer({ size: Math.max(total, 4), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.encoder();
    let o = 0;
    for (const b of bs) { if (b.n) enc.copyBufferToBuffer(gb(b), 0, stage, o, b.n * 4); o += Math.max(b.n, 1) * 4; }
    this.flush();
    await stage.mapAsync(GPUMapMode.READ);
    const all = stage.getMappedRange();
    o = 0;
    const out = bs.map((b) => {
      const arr = b.i32 ? Float32Array.from(new Int32Array(all, o, b.n)) : new Float32Array(all.slice(o, o + b.n * 4));
      o += Math.max(b.n, 1) * 4;
      return arr;
    });
    stage.unmap();
    stage.destroy();
    return out;
  }

  // --------------------------------------------------------------- dispatch
  private pipe(k: Kernel) {
    let p = this.pipes.get(k);
    if (!p) {
      const module = this.device.createShaderModule({ code: sh[k], label: k });
      p = this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' }, label: k });
      this.pipes.set(k, p);
    }
    return p;
  }

  private run(k: Kernel, bufs: (Buf | GPUBuffer | null)[], params: Param[], groups: [number, number?, number?]) {
    if (this.uniUsed + this.uniAlign > this.uniData.byteLength) this.flush();
    const off = this.uniUsed;
    const dv = new DataView(this.uniData, off, 64);
    params.forEach((v, i) => (typeof v === 'number' ? dv.setUint32(4 * i, v, true) : dv.setFloat32(4 * i, v.f, true)));
    this.uniUsed += Math.max(this.uniAlign, 64);
    const pipe = this.pipe(k);
    const entries: GPUBindGroupEntry[] = bufs.map((b, i) => ({
      binding: i,
      resource: { buffer: b === null ? this.dummy : 'g' in b ? gb(b as Buf) : (b as GPUBuffer) },
    }));
    entries.push({ binding: bufs.length, resource: { buffer: this.uni, offset: off, size: Math.ceil(params.length * 4 / 16) * 16 || 16 } });
    const bg = this.device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    const pass = this.encoder().beginComputePass();
    pass.setPipeline(pipe);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(groups[0], groups[1] ?? 1, groups[2] ?? 1);
    pass.end();
    this.dispatches++;
  }

  private grid(n: number): [number, number] {
    const g = Math.max(1, Math.ceil(n / WG));
    return g <= 65535 ? [g, 1] : [65535, Math.ceil(g / 65535)];
  }

  // ---------------------------------------------------------------- kernels
  fill(y: Buf, v: number, n: number) { this.run('fill', [y], [n, F(v)], this.grid(n)); }
  scale(x: Buf, y: Buf, a: number, n: number, acc: boolean) { this.run('scale', [x, y], [n, F(a), +acc], this.grid(n)); }

  matmul(A: Buf, B: Buf, C: Buf, M: number, N: number, K: number, tA: boolean, tB: boolean, acc: boolean, bias?: Buf) {
    if (!M || !N) return;
    if (!K) throw new Error('matmul with K = 0');
    this.run('matmul', [A, B, C, bias ?? null], [M, N, K, +tA, +tB, +acc, +!!bias], [Math.ceil(N / 64), Math.ceil(M / 64)]);
  }
  unary(op: Unary, x: Buf, y: Buf, n: number, a: number, b: number) {
    if (n) this.run('unary', [x, y], [n, UNARY[op], F(a), F(b)], this.grid(n));
  }
  unaryGrad(op: Unary, x: Buf, y: Buf, dy: Buf, dx: Buf, n: number, a: number, b: number) {
    if (n) this.run('unaryGrad', [x, y, dy, dx], [n, UNARY[op], F(a), F(b)], this.grid(n));
  }
  binary(op: Binary, a: Buf, b: Buf, y: Buf, n: number, mode: BMode, inner: number) {
    if (n) this.run('binary', [a, b, y], [n, BINARY[op], BMODE[mode], inner], this.grid(n));
  }
  binaryGrad(op: Binary, a: Buf, b: Buf, dy: Buf, da: Buf | null, db: Buf | null, n: number,
             mode: BMode, inner: number) {
    if (!n) return;
    if (da || (db && mode === 'full'))
      this.run('binaryGradA', [a, b, dy, da, mode === 'full' ? db : null], [n, BINARY[op], BMODE[mode], inner, +!!da, +(!!db && mode === 'full')], this.grid(n));
    if (db && mode !== 'full') {
      // b[j] collects the elements it was broadcast to: count of them, j * stride + k * step
      const [nb, count, stride, step] = mode === 'scalar' ? [1, n, 0, 1] : mode === 'row' ? [n / inner, inner, inner, 1] : [inner, n / inner, 1, inner];
      this.run('binaryGradB', [a, b, dy, db], [nb, BINARY[op], count, stride, step], this.grid(nb));
    }
  }
  gather(x: Buf, idx: Buf, y: Buf, nOut: number, d: number, acc: boolean) {
    if (nOut * d) this.run('gather', [x, idx, y], [nOut, d, +acc], this.grid(nOut * d));
  }
  segment(x: Buf, off: Buf, src: Buf, y: Buf, nOut: number, d: number, acc: boolean) {
    if (nOut * d) this.run('segment', [x, off, src, y], [nOut, d, +acc], this.grid(nOut * d));
  }
  copyCols(x: Buf, xCols: number, xOff: number, y: Buf, yCols: number, yOff: number, rows: number, w: number, acc: boolean) {
    if (rows * w) this.run('copyCols', [x, y], [rows, w, xCols, xOff, yCols, yOff, +acc], this.grid(rows * w));
  }
  colSum(x: Buf, y: Buf, rows: number, cols: number, acc: boolean) {
    if (cols) this.run('reduce', [x, y], [cols, rows, 1, cols, +acc], this.grid(cols));
  }
  rowSum(x: Buf, y: Buf, rows: number, cols: number, acc: boolean) {
    if (rows) this.run('reduce', [x, y], [rows, cols, cols, 1, +acc], this.grid(rows));
  }
  norm(kind: NormKind, x: Buf, w: Buf, b: Buf | null, y: Buf, saved: Buf, rows: number, d: number, eps: number) {
    if (rows) this.run('norm', [x, w, b, y, saved], [rows, d, +(kind === 'rms'), +!!b, F(eps)], this.grid(rows));
  }
  normGrad(kind: NormKind, x: Buf, w: Buf, saved: Buf, dy: Buf, dx: Buf | null, dw: Buf | null, db: Buf | null, rows: number, d: number) {
    if (!rows) return;
    if (dx) this.run('normGradX', [x, w, saved, dy, dx], [rows, d, +(kind === 'rms')], this.grid(rows));
    if (dw || db) this.run('normGradW', [x, saved, dy, dw, db], [rows, d, +!!dw, +!!db], this.grid(d));
  }
  attention(qkv: Buf, bias: Buf, L: SeqLayout, out: Buf, probs: Buf, H: number, dh: number, scale: number) {
    if (L.nTok) this.run('attention', [qkv, bias, L.off, L.poff, L.tok2seq, out, probs], [L.nTok, H, dh, F(scale)], this.grid(L.nTok * H));
  }
  attentionGrad(qkv: Buf, probs: Buf, dOut: Buf, L: SeqLayout, dqkv: Buf, dbias: Buf | null, H: number, dh: number, scale: number) {
    if (!L.nTok) return;
    const rowdot = this.zeros(L.nTok * H);
    const common = [qkv, probs, dOut, L.off, L.poff, L.tok2seq];
    const P = [L.nTok, H, dh, F(scale)];
    this.run('attnGradQ', [...common, dqkv, rowdot], P, this.grid(L.nTok * H));
    this.run('attnGradKV', [...common, dqkv, rowdot], P, this.grid(L.nTok * H));
    if (dbias) this.run('attnGradBias', [...common, dbias, rowdot], P, this.grid(L.nTok));
    this.free(rowdot);
  }
  cutoff(kind: CutoffKind, d: Buf, rc: Buf, y: Buf, n: number, width: number) {
    if (n) this.run('cutoff', [d, rc, d, y, null], [n, +(kind === 'bump'), F(width), 0, 0, 0], this.grid(n));
  }
  cutoffGrad(kind: CutoffKind, d: Buf, rc: Buf, dy: Buf, dd: Buf | null, drc: Buf | null, n: number, width: number) {
    if (n && (dd || drc)) this.run('cutoff', [d, rc, dy, dd, drc], [n, +(kind === 'bump'), F(width), 1, +!!dd, +!!drc], this.grid(n));
  }
  spline(x: Buf, V: Buf, D: Buf, y: Buf, n: number, C: number, K: number, h: number) {
    if (n * C) this.run('spline', [x, V, D, x, y], [n, C, K, F(h), 0], this.grid(n * C));
  }
  splineGrad(x: Buf, V: Buf, D: Buf, dy: Buf, dx: Buf, n: number, C: number, K: number, h: number) {
    if (n) this.run('spline', [x, V, D, dy, dx], [n, C, K, F(h), 1], this.grid(n));
  }
  sph(u: Buf, y: Buf, n: number, lmax: number) {
    if (lmax > 6) throw new Error('sph: lmax > 6 is not supported on WebGPU');
    if (n) this.run('sph', [u, u, y], [n, lmax, 0], this.grid(n));
  }
  sphGrad(u: Buf, dy: Buf, du: Buf, n: number, lmax: number) {
    if (n) this.run('sph', [u, dy, du], [n, lmax, 1], this.grid(n));
  }
  power(x: Buf, p: Buf, B: number, A: number, lmax: number) {
    const n = B * ((A * (A + 1)) / 2) * (lmax + 1);
    if (n) this.run('power', [x, p], [B, A, lmax], this.grid(n));
  }
  powerGrad(x: Buf, dp: Buf, dx: Buf, B: number, A: number, lmax: number) {
    const n = B * A * (lmax + 1) ** 2;
    if (n) this.run('powerGrad', [x, dp, dx], [B, A, lmax], this.grid(n));
  }

  adam(p: Buf, g: Buf, m: Buf, v: Buf, n: number, lr: number, b1: number, b2: number, eps: number, t: number) {
    this.run('adam', [p, g, m, v], [n, F(lr), F(b1), F(b2), F(eps), F(1 - b1 ** t), F(1 - b2 ** t)], this.grid(n));
  }

  async thumb(x: Buf, R: number, C: number, rows: number, cols: number): Promise<Thumb> {
    return (await this.thumbs([{ x, R, C, rows, cols }]))[0];
  }

  /** Many thumbnails with a single readback. */
  async thumbs(reqs: { x: Buf; R: number; C: number; rows: number; cols: number }[]): Promise<Thumb[]> {
    let o = 0;
    const offs = reqs.map((r) => { const s = o; o += r.rows * r.cols + 4; return s; });
    const out = this.zeros(o);
    reqs.forEach((r, i) => {
      if (!(r.R * r.C)) return;
      this.run('thumb', [r.x, out], [r.R, r.C, r.rows, r.cols, offs[i]], this.grid(r.rows * r.cols));
      this.run('stats', [r.x, out], [r.R * r.C, offs[i] + r.rows * r.cols], [1]);
    });
    const all = await this.read(out);
    this.free(out);
    return reqs.map((r, i) => {
      const n = r.R * r.C, s = offs[i] + r.rows * r.cols;
      const [s1, s2, mn, mx] = all.subarray(s, s + 4);
      const mean = n ? s1 / n : 0;
      return { rows: r.rows, cols: r.cols, data: all.slice(offs[i], s), mean,
               std: n ? Math.sqrt(Math.max(s2 / n - mean * mean, 0)) : 0, min: n ? mn : 0, max: n ? mx : 0,
               absmax: n ? Math.max(Math.abs(mn), Math.abs(mx)) : 0 };
    });
  }
}
