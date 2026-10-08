import { readFileSync } from 'node:fs';
import { CpuBackend } from '../src/engine/cpu';
import type { Backend } from '../src/engine/backend';
import { parseSafetensors } from '../src/common/safetensors';
import { checkSupported, type ModelMeta } from '../src/models/pet/checkpoint';
import { PET } from '../src/models/pet/model';
import { ANI } from '../src/models/ani/model';
import { PhysNet } from '../src/models/physnet/model';

export function loadModel(name: string, be: Backend = new CpuBackend()) {
  const meta: ModelMeta = JSON.parse(readFileSync(`public/models/${name}.json`, 'utf8'));
  checkSupported(meta);
  const buf = readFileSync(`public/models/${name}.safetensors`);
  return new PET(be, meta, parseSafetensors(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)));
}

export function loadANI(be: Backend = new CpuBackend()) {
  const meta = JSON.parse(readFileSync('public/models/ani-2x.json', 'utf8'));
  const buf = readFileSync('public/models/ani-2x.safetensors');
  return new ANI(be, meta, parseSafetensors(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)));
}


export function loadPhysNet(stem = 'tests/fixtures/physnet-test', be: Backend = new CpuBackend()) {
  const meta = JSON.parse(readFileSync(`${stem}.json`, 'utf8'));
  const buf = readFileSync(`${stem}.safetensors`);
  return new PhysNet(be, meta, parseSafetensors(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)));
}
