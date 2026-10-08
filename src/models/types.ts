// What every model gives the engine worker and the visualiser.
import type { Backend } from '../engine/backend';
import type { Graph, Node, Tensor } from '../engine/tensor';
import type { System } from '../common/structure';

/** What the rows of a family of tensors stand for. Tensors carry the family name in `kind`. */
export interface RowSpace {
  label: string; // 'atom', 'edge', 'angular triplet', ...
  owner: Int32Array; // the atom each row belongs to (-1: none); the 3D view shows the selected atom's rows
  all?: boolean; // show every row (e.g. atoms), not only the selected atom's
  atom?: Int32Array; // the graph node a row stands for (-1: none)
  edge?: Int32Array; // the graph edge a row stands for (-1: none)
}

/** The graph drawn over the structure: edge e goes from center[e] to neighbor[e] + shift. */
export interface GraphSpec {
  center: Int32Array;
  neighbor: Int32Array;
  shift: Float32Array; // [E*3]
  label: string; // 'neighbour', 'symmetry-function pair', ...
}

/** Attention probabilities kept by an op, for the selected atom's heads. */
export interface AttentionSpec {
  node: Node;
  heads: number;
  /** Where atom a's [heads, n, n] block starts in the probabilities, and its tokens' atoms. */
  tokens: (a: number) => { offset: number; atoms: Int32Array };
}

export interface Forward {
  energy: Tensor; // [1] eV
  perAtom: Tensor; // [N] eV
  ncForces?: Tensor; // [N, 3] eV/A, direct (non-conservative) forces, for models that predict them
  positions: Tensor; // [N, 3], requires grad when forces are wanted
  virialVectors?: Tensor; // [K, 3] vectors v with W = -sum v (x) dE/dv (periodic stress)
  graph: GraphSpec;
  rows: Record<string, RowSpace>;
  attention?: AttentionSpec[];
  /** Other per-atom outputs worth colouring the structure by, e.g. { charge: [N] }. */
  extras?: Record<string, Tensor>;
}

export type ModelKind = 'pet' | 'ani' | 'krr' | 'physnet' | 'mace' | 'lorem';

export interface Model {
  readonly kind: ModelKind;
  readonly be: Backend;
  readonly params: Map<string, Tensor>;
  /** Elements the model can handle (atomic numbers). */
  readonly elements: number[];
  /** Whether the model also predicts forces directly (without a backward pass). */
  readonly hasNC?: boolean;
  /** `forces`: positions take part in the backward pass; `nc`: run the direct force head too. */
  forward(g: Graph, sys: System, opts: { forces?: boolean; nc?: boolean }): Forward;
}

export const range = (n: number, s = 0) => Int32Array.from({ length: n }, (_, i) => s + i);

/** Rows that are atoms. */
export const atomRows = (N: number): RowSpace => ({ label: 'atom', owner: range(N), atom: range(N), all: true });
/** Rows that are graph edges, owned by their centre atom. */
export const edgeRows = (center: Int32Array): RowSpace => ({ label: 'edge', owner: center, edge: range(center.length) });
