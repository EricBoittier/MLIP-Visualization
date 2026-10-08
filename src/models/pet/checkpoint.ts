import { parseSafetensors, type Tensors } from '../../common/safetensors';

// pet-kokkos model format: <name>.json (hypers, species, composition energies)
// and <name>.safetensors (fp32 weights, PyTorch state_dict names).

export interface Hypers {
  d_pet: number;
  d_head: number;
  d_node: number;
  d_feedforward: number;
  num_heads: number;
  num_attention_layers: number;
  num_gnn_layers: number;
  num_readout_layers: number;
  normalization: 'LayerNorm' | 'RMSNorm';
  activation: 'SiLU' | 'SwiGLU';
  transformer_type: 'PostLN' | 'PreLN';
  featurizer_type: 'feedforward' | 'residual';
  attention_temperature: number;
  cutoff: number;
  cutoff_width: number;
  cutoff_function: 'Cosine' | 'Bump';
  num_neighbors_adaptive: number | null;
  adaptive_cutoff_method: 'solver' | 'grid';
  cutoff_width_adaptive: number;
  system_conditioning: boolean;
  max_charge: number;
  max_spin_multiplicity: number;
  zbl?: boolean;
  long_range_enabled?: boolean;
}

export interface ModelMeta {
  format_version: number;
  architecture: 'pet';
  source_model?: string;
  hypers: Hypers;
  energy_scale: number;
  atomic_types: number[];
  species_to_index: number[];
  composition_energies: number[];
  /** Direct (non-conservative) force heads, if the converter kept them. */
  non_conservative?: Record<string, { scale: number[] }>;
}

export interface Checkpoint {
  meta: ModelMeta;
  tensors: Tensors;
}

export function checkSupported(meta: ModelMeta) {
  const h = meta.hypers;
  if (meta.architecture !== 'pet') throw new Error(`not a PET model: ${meta.architecture}`);
  if (h.zbl) throw new Error('ZBL models are not supported yet');
  if (h.long_range_enabled) throw new Error('long-range models are not supported yet');
  if (h.num_neighbors_adaptive != null && h.adaptive_cutoff_method !== 'solver')
    throw new Error(`adaptive_cutoff_method=${h.adaptive_cutoff_method} is not supported yet (only "solver")`);
}

export async function loadCheckpoint(json: ModelMeta | string, weights: ArrayBuffer | string): Promise<Checkpoint> {
  const meta: ModelMeta = typeof json === 'string' ? await (await fetch(json)).json() : json;
  const buf = typeof weights === 'string' ? await (await fetch(weights)).arrayBuffer() : weights;
  checkSupported(meta);
  return { meta, tensors: parseSafetensors(buf) };
}
