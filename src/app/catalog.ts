// Where the converted models live, shared by the visualiser and the MD page.
// models/index.json: [{ name, kind, label, meta, weights, params }]. KRR has no files: it is fitted
// in the page, on the current structure, against a teacher model.
import type { ModelKind } from '../models/types';

export interface ModelEntry { name: string; kind: ModelKind; label?: string; meta?: string; weights?: string; params?: any }

export const KIND_ORDER: ModelKind[] = ['pet', 'mace', 'lorem', 'ani', 'physnet', 'krr'];

// Model files come from public/models/ when it has them (local development), otherwise from the
// Hugging Face repository that scripts/publish_weights.py fills; ?models=<base url> picks another.
export const HF_MODELS = 'https://huggingface.co/EricBoi/mlip-visualization-models/resolve/main/';

export const getJSON = async (u: string) => { const r = await fetch(u); if (!r.ok) throw new Error(`${u}: HTTP ${r.status}`); return r.json(); };

/** The first model index that answers, and the base URL its files are relative to. */
export async function findModels(custom = new URLSearchParams(location.search).get('models')): Promise<{ entries: ModelEntry[]; base: string }> {
  for (const b of custom ? [custom.replace(/\/?$/, '/')] : ['models/', HF_MODELS]) {
    try {
      const list: ModelEntry[] = await getJSON(`${b}index.json`);
      const probe = list.find((x) => x.meta);
      if (probe) await getJSON(b + probe.meta); // index.json is in git; the weights may not be
      return { entries: list, base: b };
    } catch { /* try the next one */ }
  }
  return { entries: [], base: 'models/' };
}

export const resolve = (base: string, name: string) => (/^https?:/.test(name) ? name : base + name);

/** Download with progress (fraction, 'x / y MB'). */
export async function fetchBytes(url: string, progress: (fraction: number, mb: string) => void): Promise<ArrayBuffer> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const total = +(r.headers.get('content-length') ?? 0);
  if (!r.body || !total) return r.arrayBuffer();
  const reader = r.body.getReader(), chunks: Uint8Array[] = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    progress(got / total, `${(got / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`);
  }
  const out = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out.buffer;
}
