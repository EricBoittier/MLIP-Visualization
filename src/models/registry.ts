// Build a model of any kind from its metadata and weights.
import type { Backend } from '../engine/backend';
import { parseSafetensors } from '../common/safetensors';
import { checkSupported } from './pet/checkpoint';
import { ANI } from './ani/model';
import { PET } from './pet/model';
import type { Model, ModelKind } from './types';

export interface ModelContext {
  /** Models already loaded, by id (KRR labels its training set with one of them). */
  models: Map<string, Model>;
  progress: (text: string) => void;
}

export async function createModel(be: Backend, kind: ModelKind, meta: any, weights: ArrayBuffer | null, ctx: ModelContext): Promise<Model> {
  void ctx;
  switch (kind) {
    case 'pet':
      checkSupported(meta);
      return new PET(be, meta, parseSafetensors(weights!));
    case 'ani':
      return new ANI(be, meta, parseSafetensors(weights!));
    default:
      throw new Error(`model kind "${kind}" is not available yet`);
  }
}
