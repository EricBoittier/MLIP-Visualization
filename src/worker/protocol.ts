// Messages between the UI and the engine worker.
import type { Thumb } from '../engine/backend';
import type { System } from '../common/structure';
import type { GraphSpec, ModelKind, RowSpace } from '../models/types';

/** One recorded op: what the 3D view draws as a block. Keyed by tape index. */
export interface OpInfo {
  op: string;
  scope: string;
  shape: number[];
  kind?: string; // the row family (see Trace.rows)
  inputs: number[]; // tape indices of producing ops (-1: constant)
  params: string[]; // weight names read by this op
  grad: boolean; // takes part in the backward pass
}

export interface AttentionMap {
  heads: number;
  n: number; // tokens of the selected atom: itself, then its neighbours
  probs: Float32Array; // [heads, n, n]
  atoms: Int32Array; // [n] atom index of each token
  op: number; // tape index of the attention op
}

/** One forward + backward pass, as the visualiser sees it. */
export interface Trace {
  topology?: OpInfo[]; // sent when the graph changes
  shapes: number[][]; // actual output shape of every op
  values: (Thumb | null)[];
  grads: (Thumb | null)[];
  params?: Record<string, { value: Thumb; grad: Thumb | null }>;
  paramShapes?: Record<string, number[]>; // full shapes (the thumbnails are block averages)
  attention: AttentionMap[];
  selected: number; // the atom whose rows are shown
  rowSel: (Int32Array | null)[]; // per op: which rows of the output are shown (null: all, averaged)
  norms: (Float32Array | null)[]; // per op with atom/edge/token rows: RMS of every row
  gradNorms: (Float32Array | null)[]; // the same for the gradient
  rows: Record<string, RowSpace>;
  graph: GraphSpec;
  ms: { forward: number; backward: number; capture: number };
}

export interface Pass {
  id: number;
  positions: number[][];
  cell?: number[][];
  numbers: number[];
  energy: number;
  energies: Float32Array;
  mode: ForceMode;
  forces?: Float32Array; // conservative, -dE/dr
  ncForces?: Float32Array; // direct head
  atomProps?: Record<string, Float32Array>; // other per-atom outputs (e.g. charge)
  stress?: number[][];
  trace: Trace;
}

export type ToWorker =
  | { type: 'init'; backend: 'webgpu' | 'cpu' | 'auto' }
  | { type: 'loadModel'; id: string; kind: ModelKind; meta: any; weights: ArrayBuffer | null; label: string; activate: boolean }
  | { type: 'use'; id: string }
  | { type: 'evaluate'; system: System; selected: number; mode: ForceMode };

/** conservative: -dE/dr by backprop; direct: the model's direct force head, no backward pass; both. */
export type ForceMode = 'conservative' | 'direct' | 'both';

export type FromWorker =
  | { type: 'ready'; backend: string; adapter: string }
  | { type: 'model'; id: string; kind: ModelKind; label: string; meta: any; nParams: number; hasNC: boolean; activate: boolean; elements: number[] }
  | { type: 'pass'; pass: Pass }
  | { type: 'progress'; text: string }
  | { type: 'error'; text: string };
