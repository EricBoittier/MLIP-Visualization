// How each kind of model presents itself in the visualiser.
import type { Ctx } from '../viz/article';
import type { Mod, ModDesc } from '../viz/modules';
import type { ModelKind } from './types';

export interface ModelUI {
  kind: ModelKind;
  name: string; // short name, e.g. 'PET'
  family: string; // e.g. 'Behler–Parrinello network'
  describe: (seg: string, path: string) => ModDesc | null;
  subtitle: (m: Mod, meta: any) => string; // one line under a box of the 2D diagram
  narration: (m: Mod | 'backward', meta: any) => string; // HTML for the walkthrough panel
  card: (meta: any, nParams: number, label: string) => string; // HTML model card
  collapse?: (m: Mod) => boolean; // modules that start collapsed
  intro?: (meta: any) => string; // HTML at the top of the walkthrough
  /** The walkthrough paragraph for op i (key 'cont': part of the step before). */
  step?: (i: number, mod: Mod, c: Ctx, lastOp: number) => { key: string; html: string };
  /** The walkthrough paragraph for a module in the backward pass. */
  back?: (mod: Mod, c: Ctx, ops: number[]) => string;
}
