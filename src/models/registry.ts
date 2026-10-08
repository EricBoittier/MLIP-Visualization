// Build a model of any kind from its metadata and weights.
import type { Backend } from '../engine/backend';
import { parseSafetensors } from '../common/safetensors';
import { checkSupported } from './pet/checkpoint';
import { ANI } from './ani/model';
import { KRR } from './krr/model';
import { PhysNet } from './physnet/model';
import { PET } from './pet/model';
import type { Model, ModelKind } from './types';

export interface ModelContext {
  /** Models already loaded, by id (KRR labels its training set with one of them). */
  models: Map<string, Model>;
  progress: (text: string, fraction?: number) => void;
}

export async function createModel(be: Backend, kind: ModelKind, meta: any, weights: ArrayBuffer | null, ctx: ModelContext): Promise<Model> {
  switch (kind) {
    case 'pet':
      checkSupported(meta);
      return new PET(be, meta, parseSafetensors(weights!));
    case 'ani':
      return new ANI(be, meta, parseSafetensors(weights!));
    case 'physnet':
      return new PhysNet(be, meta, parseSafetensors(weights!));
    case 'krr': {
      const teacher = ctx.models.get(meta.teacher);
      if (!teacher || teacher.kind === 'krr') throw new Error(`KRR needs a loaded teacher model (got "${meta.teacher}")`);
      if (!meta.system) throw new Error('KRR needs a structure to fit on');
      const m = new KRR(be, meta);
      await m.fit(meta.system, teacher, meta.teacherLabel ?? meta.teacher, ctx.progress);
      return m;
    }
    default:
      throw new Error(`model kind "${kind}" is not available yet`);
  }
}
