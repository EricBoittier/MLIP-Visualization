// One palette for the 3D blocks (GLSL) and the structure view, read from the theme's CSS custom properties
// (app/style.css) and refreshed when the theme changes.
import { mode, onTheme, rgb, type RGB } from './theme';

export const PALETTE = {
  value: { neg: [0, 0, 0] as RGB, pos: [0, 0, 0] as RGB }, // blue / red
  grad: { neg: [0, 0, 0] as RGB, pos: [0, 0, 0] as RGB }, // violet / green
  weight: { neg: [0, 0, 0] as RGB, pos: [0, 0, 0] as RGB }, // indigo / cyan
  attention: [0, 0, 0] as RGB,
  mid: [0, 0, 0] as RGB, // zero
  empty: [0, 0, 0] as RGB, // not computed yet
  hi: [0, 0, 0] as RGB, // highlighted row
  glow: [0, 0, 0] as RGB, // the active op
  bg: [0, 0, 0] as RGB,
  edge: [0, 0, 0] as RGB,
  edgeMine: [0, 0, 0] as RGB,
  force: [0, 0, 0] as RGB,
  forceNC: [0, 0, 0] as RGB,
  dark: true,
};

function refresh() {
  Object.assign(PALETTE, {
    value: { neg: rgb('value-neg'), pos: rgb('value') },
    grad: { neg: rgb('grad'), pos: rgb('grad-pos') },
    weight: { neg: rgb('weight-neg'), pos: rgb('weight') },
    attention: rgb('attn'), mid: rgb('mid'), empty: rgb('empty'), hi: rgb('hi'), glow: rgb('accent-fill'), bg: rgb('bg'),
    edge: rgb('edge'), edgeMine: rgb('edge-mine'), force: rgb('force'), forceNC: rgb('force-nc'), dark: mode() === 'dark',
  });
}
refresh();
onTheme(refresh);

/** 0..1 RGB as #rrggbb */
export const hex = (c: number[]) => '#' + c.map((x) => Math.round(Math.min(Math.max(x, 0), 1) * 255).toString(16).padStart(2, '0')).join('');
export const mix = (a: number[], b: number[], t: number) => a.map((x, i) => x + (b[i] - x) * t);
