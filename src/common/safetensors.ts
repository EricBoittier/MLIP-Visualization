// safetensors: an 8-byte header length, a JSON header, then raw little-endian data.

export type Tensors = Map<string, { shape: number[]; data: Float32Array }>;

export function parseSafetensors(buffer: ArrayBuffer): Tensors {
  const view = new DataView(buffer);
  const headerLen = Number(view.getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, headerLen)));
  const base = 8 + headerLen;
  const out: Tensors = new Map();
  for (const [name, info] of Object.entries<any>(header)) {
    if (name === '__metadata__') continue;
    if (info.dtype !== 'F32') throw new Error(`safetensors: ${name} has dtype ${info.dtype}, expected F32`);
    const [b, e] = info.data_offsets;
    // copy: the source offset may not be 4-byte aligned
    const data = new Float32Array(buffer.slice(base + b, base + e));
    out.set(name, { shape: info.shape, data });
  }
  return out;
}

/** Tensors back to safetensors. */
export function writeSafetensors(tensors: Tensors): ArrayBuffer {
  const header: Record<string, unknown> = {};
  let off = 0;
  for (const [name, t] of tensors) {
    header[name] = { dtype: 'F32', shape: t.shape, data_offsets: [off, off + 4 * t.data.length] };
    off += 4 * t.data.length;
  }
  let hjson = JSON.stringify(header);
  hjson += ' '.repeat((8 - ((8 + hjson.length) % 8)) % 8);
  const hb = new TextEncoder().encode(hjson);
  const out = new ArrayBuffer(8 + hb.length + off);
  new DataView(out).setBigUint64(0, BigInt(hb.length), true);
  new Uint8Array(out, 8, hb.length).set(hb);
  let o = 8 + hb.length;
  for (const t of tensors.values()) { new Float32Array(out, o, t.data.length).set(t.data); o += 4 * t.data.length; }
  return out;
}
