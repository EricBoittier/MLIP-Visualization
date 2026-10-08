import type { ModelKind } from './types';
import type { ModelUI } from './ui';
import { aniUI } from './ani/ui';
import { krrUI } from './krr/ui';
import { physnetUI } from './physnet/ui';
import { petUI } from './pet/ui';
import { maceUI } from './mace/ui';
import { loremUI } from './lorem/ui';

export const UIS: Partial<Record<ModelKind, ModelUI>> & Record<'pet', ModelUI> = { pet: petUI, ani: aniUI, krr: krrUI, physnet: physnetUI, mace: maceUI, lorem: loremUI };
