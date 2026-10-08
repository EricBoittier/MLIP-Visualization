import { readFileSync } from 'node:fs';
import { CpuBackend } from '../src/engine/cpu';
import type { Backend } from '../src/engine/backend';
import { parseSafetensors } from '../src/common/safetensors';
import { checkSupported, type ModelMeta } from '../src/models/pet/checkpoint';
import { PET } from '../src/models/pet/model';

export function loadModel(name: string, be: Backend = new CpuBackend()) {
  const meta: ModelMeta = JSON.parse(readFileSync(`public/models/${name}.json`, 'utf8'));
  checkSupported(meta);
  const buf = readFileSync(`public/models/${name}.safetensors`);
  return new PET(be, meta, parseSafetensors(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)));
}
