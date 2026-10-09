// A small streaming line chart on a canvas: one y axis, a legend, end labels and a hover crosshair.
// Colours are theme tokens (app/style.css), read at every draw.
import { css } from '../viz/theme';

/** `color`: a colour token, e.g. 'series-1' for --series-1. */
export interface Series { name: string; color: string; dash?: number[] }
const MAX_POINTS = 4000;

/** 'Nice' tick values covering [lo, hi]. */
function ticks(lo: number, hi: number, n = 5) {
  const span = hi - lo || Math.abs(hi) || 1, raw = span / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw)!;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return { values: out, step };
}
const fmt = (v: number, step: number) => v.toFixed(Math.max(0, -Math.floor(Math.log10(step) + 1e-9)));

export class LineChart {
  private canvas = document.createElement('canvas');
  private tip = document.createElement('div');
  private t: number[] = [];
  private y: number[][];
  private hover: number | null = null;
  /** Horizontal reference lines, e.g. a target temperature. */
  refs: { y: number; label: string }[] = [];

  constructor(readonly el: HTMLElement, readonly series: Series[], readonly unit: string, readonly xUnit = 'fs') {
    this.y = series.map(() => []);
    el.classList.add('chart');
    if (series.length > 1) {
      const legend = document.createElement('div');
      legend.className = 'chart-legend';
      legend.innerHTML = series.map((s) => `<span><i style="background:var(--${s.color})"></i>${s.name}</span>`).join('');
      el.appendChild(legend);
    }
    const plot = document.createElement('div');
    plot.className = 'chart-plot';
    this.tip.className = 'chart-tip';
    this.tip.hidden = true;
    plot.append(this.canvas, this.tip);
    el.appendChild(plot);
    new ResizeObserver(() => this.draw()).observe(plot);
    this.canvas.addEventListener('pointermove', (e) => { this.hover = e.offsetX; this.draw(); });
    this.canvas.addEventListener('pointerleave', () => { this.hover = null; this.draw(); });
  }

  clear() { this.t = []; this.y = this.series.map(() => []); this.draw(); }

  push(t: number, values: number[]) {
    this.t.push(t);
    values.forEach((v, k) => this.y[k].push(v));
    if (this.t.length > MAX_POINTS) { // keep every other point: long runs stay cheap to draw
      const keep = (a: number[]) => a.filter((_, i) => i % 2 === 0 || i === a.length - 1);
      this.t = keep(this.t);
      this.y = this.y.map(keep);
    }
  }

  draw() {
    const c = this.canvas, box = c.parentElement!, W = box.clientWidth, H = box.clientHeight, dpr = Math.min(devicePixelRatio, 2);
    if (!W || !H) return;
    if (c.width !== W * dpr || c.height !== H * dpr) { c.width = W * dpr; c.height = H * dpr; c.style.width = `${W}px`; c.style.height = `${H}px`; }
    const g = c.getContext('2d')!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    g.font = '11px system-ui, -apple-system, sans-serif';
    const TEXT = css('text'), MUTED = css('muted'), GRID = css('line'), SURFACE = css('panel'), color = (s: Series) => css(s.color);

    const n = this.t.length;
    let lo = Infinity, hi = -Infinity;
    for (const a of this.y) for (const v of a) { if (v < lo) lo = v; if (v > hi) hi = v; }
    for (const r of this.refs) { lo = Math.min(lo, r.y); hi = Math.max(hi, r.y); }
    if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
    const pad = (hi - lo) * 0.08 || Math.abs(hi) * 0.05 || 1;
    const yt = ticks(lo - pad, hi + pad), xt = ticks(0, Math.max(this.t[n - 1] ?? 0, 1e-6), 5);
    const y0 = yt.values[0] - (yt.values[0] > lo - pad ? yt.step : 0), y1 = yt.values[yt.values.length - 1] + (yt.values[yt.values.length - 1] < hi + pad ? yt.step : 0);
    const x1 = xt.values[xt.values.length - 1] < (this.t[n - 1] ?? 0) ? xt.values[xt.values.length - 1] + xt.step : xt.values[xt.values.length - 1] || 1;
    const labelW = Math.max(...yt.values.map((v) => g.measureText(fmt(v, yt.step)).width)) + 10;
    const L = labelW, R = this.series.length > 1 ? 52 : 12, T = 8, B = 22;
    const X = (t: number) => L + ((W - L - R) * t) / x1, Y = (v: number) => T + ((H - T - B) * (y1 - v)) / (y1 - y0);

    // recessive grid and axes
    g.strokeStyle = GRID; g.lineWidth = 1; g.fillStyle = MUTED;
    g.textAlign = 'right'; g.textBaseline = 'middle';
    for (const v of yt.values) {
      if (v < y0 || v > y1) continue;
      g.beginPath(); g.moveTo(L, Math.round(Y(v)) + 0.5); g.lineTo(W - R, Math.round(Y(v)) + 0.5); g.stroke();
      g.fillText(fmt(v, yt.step), L - 6, Y(v));
    }
    g.textAlign = 'center'; g.textBaseline = 'top';
    const unitW = g.measureText(this.xUnit).width + 14; // keep the last tick clear of the unit
    for (const t of xt.values) if (t <= x1 && X(t) + g.measureText(fmt(t, xt.step)).width / 2 < W - R - unitW) g.fillText(fmt(t, xt.step), X(t), H - B + 6);
    g.textAlign = 'right';
    g.fillText(this.xUnit, W - R, H - B + 6);

    for (const r of this.refs) {
      g.setLineDash([4, 4]); g.strokeStyle = MUTED;
      g.beginPath(); g.moveTo(L, Y(r.y)); g.lineTo(W - R, Y(r.y)); g.stroke();
      g.setLineDash([]);
      g.textAlign = 'left'; g.textBaseline = 'bottom'; g.fillStyle = MUTED;
      g.fillText(r.label, L + 4, Y(r.y) - 2);
    }

    // lines
    g.lineWidth = 2; g.lineJoin = 'round';
    this.series.forEach((s, k) => {
      if (!n) return;
      g.strokeStyle = color(s); g.setLineDash(s.dash ?? []);
      g.beginPath();
      this.t.forEach((t, i) => (i ? g.lineTo(X(t), Y(this.y[k][i])) : g.moveTo(X(t), Y(this.y[k][i]))));
      g.stroke();
    });
    g.setLineDash([]);
    // direct labels at the line ends, nudged apart
    if (n && this.series.length > 1) {
      const ends = this.series.map((s, k) => ({ s, y: Y(this.y[k][n - 1]) })).sort((a, b) => a.y - b.y);
      for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 13);
      g.textAlign = 'left'; g.textBaseline = 'middle'; g.fillStyle = TEXT;
      for (const e of ends) g.fillText(e.s.name.split(' ')[0], X(this.t[n - 1]) + 6, e.y);
    }

    // hover: crosshair at the nearest sample, ringed markers, a tooltip with every value
    if (this.hover === null || !n || this.hover < L || this.hover > W - R) { this.tip.hidden = true; return; }
    const tq = ((this.hover - L) / (W - L - R)) * x1;
    let i = 0;
    for (let a = 0, b = n - 1; a <= b;) { const m = (a + b) >> 1; if (this.t[m] < tq) a = m + 1; else b = m - 1; i = Math.min(a, n - 1); }
    if (i > 0 && Math.abs(this.t[i - 1] - tq) < Math.abs(this.t[i] - tq)) i--;
    const x = X(this.t[i]);
    g.strokeStyle = MUTED; g.lineWidth = 1;
    g.beginPath(); g.moveTo(Math.round(x) + 0.5, T); g.lineTo(Math.round(x) + 0.5, H - B); g.stroke();
    this.series.forEach((s, k) => {
      g.beginPath(); g.arc(x, Y(this.y[k][i]), 4, 0, 2 * Math.PI);
      g.fillStyle = color(s); g.fill(); g.lineWidth = 2; g.strokeStyle = SURFACE; g.stroke();
    });
    this.tip.hidden = false;
    this.tip.innerHTML = `<div class="tt-head">t = ${this.t[i].toFixed(1)} ${this.xUnit}</div>` + this.series.map((s, k) =>
      `<div><i style="background:var(--${s.color})"></i>${s.name}<b>${fmt(this.y[k][i], yt.step / 10)} ${this.unit}</b></div>`).join('');
    const tw = this.tip.offsetWidth;
    this.tip.style.left = `${x + 12 + tw > W ? x - 12 - tw : x + 12}px`;
    this.tip.style.top = `${T}px`;
  }
}
