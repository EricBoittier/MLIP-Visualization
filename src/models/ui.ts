// How each kind of model presents itself in the visualiser.
import type { Ctx } from '../viz/article';
import type { Mod, ModDesc } from '../viz/modules';
import type { ModelKind } from './types';

/** What the parts of a pass are called for this kind of model. */
export interface Terms {
  forward: string; // 'forward pass', or 'prediction' for a kernel model
  backward: string; // 'backward pass', or 'force evaluation'
  backwardTitle: string; // heading of the backward section of the article
  weights: string; // what the parameters are, e.g. 'weights' or 'sparse points & α'
  attention?: boolean; // whether the model has attention maps
}

export const NN_TERMS: Terms = { forward: 'forward pass', backward: 'backward pass', backwardTitle: 'Backward pass: forces', weights: 'weights' };

export interface ModelUI {
  terms?: Terms;
  kind: ModelKind;
  name: string; // short name, e.g. 'PET'
  typeName?: string; // the kind of model, for the model menu (default: name), e.g. 'BPNN' for ANI-2x
  family: string; // e.g. 'Behler–Parrinello network'
  describe: (seg: string, path: string) => ModDesc | null;
  subtitle: (m: Mod, meta: any) => string; // one line under a box of the 2D diagram
  narration: (m: Mod | 'backward', meta: any) => string; // HTML for the walkthrough panel
  card: (meta: any, nParams: number, label: string) => string; // HTML model card
  collapse?: (m: Mod) => boolean; // modules that start collapsed
  intro?: (meta: any) => string; // HTML at the top of the walkthrough
  preferredStructure?: string; // preset to switch to when the model is loaded (its training domain)
  /** The walkthrough paragraph for op i (key 'cont': part of the step before). */
  step?: (i: number, mod: Mod, c: Ctx, lastOp: number) => { key: string; html: string };
  /** The walkthrough paragraph for a module in the backward pass. */
  back?: (mod: Mod, c: Ctx, ops: number[]) => string;
}
