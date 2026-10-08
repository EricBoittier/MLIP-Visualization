// One palette for the 3D blocks (GLSL) and the structure view.

export const PALETTE = {
  value: { neg: [0.23, 0.51, 0.96], pos: [0.98, 0.62, 0.16] }, // blue / amber
  grad: { neg: [0.66, 0.33, 0.97], pos: [0.13, 0.77, 0.37] }, // violet / green
  weight: { neg: [0.35, 0.42, 0.85], pos: [0.55, 0.82, 0.98] },
  attention: [1.0, 0.86, 0.32],
  mid: [0.17, 0.19, 0.24],
};

export const glslVec = (c: number[]) => `vec3(${c.map((x) => x.toFixed(3)).join(', ')})`;

const hex = (c: number[]) => '#' + c.map((x) => Math.round(Math.min(Math.max(x, 0), 1) * 255).toString(16).padStart(2, '0')).join('');
const mix = (a: number[], b: number[], t: number) => a.map((x, i) => x + (b[i] - x) * t);
