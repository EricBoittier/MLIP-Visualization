// Playback of one pass: every op forwards in execution order, then every op on
// the gradient path backwards in reverse order.
import type { OpInfo } from '../worker/protocol';

export interface Step { op: number; dir: 'fwd' | 'bwd' }

export class Timeline {
  steps: Step[] = [];
  t = 0; // position in steps (fractional)
  playing = false;
  speed = 6; // ops per second
  stopAt: number | null = null; // pause when reaching this position

  constructor(ops: OpInfo[] = []) { this.reset(ops); }

  reset(ops: OpInfo[]) {
    const fwd = ops.map((_, i) => ({ op: i, dir: 'fwd' as const }));
    const bwd = ops.map((o, i) => ({ op: i, dir: 'bwd' as const, g: o.grad })).filter((s) => s.g).reverse()
      .map(({ op, dir }) => ({ op, dir }));
    this.steps = [...fwd, ...bwd];
    this.nOps = ops.length;
    this.t = Math.min(this.t, this.steps.length);
  }
  private nOps = 0;

  get forwardEnd() { return this.nOps; }
  get done() { return this.t >= this.steps.length; }

  advance(dt: number) {
    if (!this.playing) return false;
    this.t = Math.min(this.t + dt * this.speed, this.steps.length);
    if (this.stopAt !== null && this.t >= this.stopAt) { this.t = this.stopAt; this.stopAt = null; this.playing = false; }
    if (this.done) this.playing = false;
    return true;
  }

  seek(k: number) { this.t = Math.max(0, Math.min(k, this.steps.length)); this.stopAt = null; }

  /** Position of an op's forward or backward step. */
  indexOf(op: number, dir: 'fwd' | 'bwd') { return this.steps.findIndex((s) => s.op === op && s.dir === dir); }

  /** Progress of every op's forward and backward reveal, plus the active step. */
  state() {
    const fwd = new Float32Array(this.nOps), bwd = new Float32Array(this.nOps);
    // paused exactly at a boundary: the step just finished is the current one
    const k = Number.isInteger(this.t) && this.t > 0 && !this.playing ? this.t - 1 : Math.floor(this.t), frac = this.t - k;
    this.steps.forEach((s, i) => {
      const v = i < k ? 1 : i === k ? frac : 0;
      (s.dir === 'fwd' ? fwd : bwd)[s.op] = v;
    });
    const active = this.steps[Math.min(k, this.steps.length - 1)];
    return { fwd, bwd, active: this.done && this.playing ? -1 : active?.op ?? -1, dir: active?.dir ?? 'fwd', k };
  }
}
