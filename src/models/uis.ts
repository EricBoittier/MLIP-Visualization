import type { ModelKind } from './types';
import type { ModelUI } from './ui';
import { aniUI } from './ani/ui';
import { krrUI } from './krr/ui';
import { petUI } from './pet/ui';

export const UIS: Partial<Record<ModelKind, ModelUI>> & Record<'pet', ModelUI> = { pet: petUI, ani: aniUI, krr: krrUI };
