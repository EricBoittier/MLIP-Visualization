// The PET network (metatrain's PETBackend / CartesianTransformer), written on the
// autograd Graph. Edges are packed (no NEF padding): atom i's attention sequence
// is its own token followed by its kept edges.
import type { Backend, SeqLayout } from '../../engine/backend';
import { Graph, Index, type RowKind, Tensor } from '../../engine/tensor';
import type { Checkpoint, Hypers, ModelMeta } from './checkpoint';
import { type System, volume } from '../../common/structure';
import { type AttentionSpec, atomRows, edgeRows, type Forward, type Model, range, type RowSpace } from '../types';
import { prepare, type Prepared } from './prepare';

export interface Output extends Forward {
  edgeVectors: Tensor; // [E_raw, 3], for the virial
  cutoffFactors: Tensor; // [E]
  attention: AttentionSpec[];
  prepared: Prepared;
  layout: SeqLayout;
}

/** PET's row families: atoms, kept edges, raw (pre-adaptive-cutoff) edges, tokens and [atoms; edges] stacks. */
function petRows(P: Prepared): Record<string, RowSpace> {
  const N = P.N, E = P.keep.length, Eraw = P.raw.center.length;
  const keptOf = new Int32Array(Eraw).fill(-1);
  P.keep.forEach((r, k) => (keptOf[r] = k));
  const tokOwner = new Int32Array(N + E), tokAtom = new Int32Array(N + E).fill(-1), tokEdge = new Int32Array(N + E).fill(-1);
  for (let i = 0, t = 0; i < N; i++) {
    tokOwner[t] = i; tokAtom[t++] = i;
    for (let e = P.offsets[i]; e < P.offsets[i + 1]; e++) { tokOwner[t] = i; tokEdge[t++] = e; }
  }
  const stackOwner = Int32Array.from([...range(N), ...P.center]);
  return {
    atom: atomRows(N),
    edge: edgeRows(P.center),
    raw_edge: { label: 'candidate edge', owner: P.raw.center, edge: keptOf },
    token: { label: 'token', owner: tokOwner, atom: tokAtom, edge: tokEdge },
    stack: { label: 'atom / edge', owner: stackOwner, atom: Int32Array.from({ length: N + E }, (_, r) => (r < N ? r : -1)),
             edge: Int32Array.from({ length: N + E }, (_, r) => (r < N ? -1 : r - N)) },
  };
}

export class PET implements Model {
  readonly kind = 'pet' as const;
  readonly params = new Map<string, Tensor>();
  readonly h: Hypers;
  get elements() { return this.meta.atomic_types; }

  constructor(readonly be: Backend, readonly meta: ModelMeta, tensors: Checkpoint['tensors']) {
    this.h = meta.hypers;
    for (const [name, t] of tensors) this.params.set(name, new Tensor(t.shape, be.upload(t.data), false, name));
  }

  p(name: string) {
    const t = this.params.get(name);
    if (!t) throw new Error(`missing weight ${name}`);
    return t;
  }
  has(name: string) { return this.params.has(name); }

  setTrainable(on: boolean) { for (const t of this.params.values()) t.requiresGrad = on; }

  /** Copy weights back to the host (e.g. to save a fine-tuned model). */
  async exportTensors(): Promise<Checkpoint['tensors']> {
    const out: Checkpoint['tensors'] = new Map();
    for (const [name, t] of this.params) out.set(name, { shape: t.shape, data: await this.be.read(t.buf) });
    return out;
  }

  prepare(sys: System) { return prepare(sys, this.h, this.meta.species_to_index); }

  get hasNC() { return !!this.meta.non_conservative?.non_conservative_force; }

  /** `forces`: positions take part in the backward pass; `nc`: also run the direct force head. */
  forward(g: Graph, sys: System, opts: { forces?: boolean; nc?: boolean; prepared?: Prepared } = {}): Output {
    const h = this.h, be = this.be;
    const P = opts.prepared ?? this.prepare(sys);
    const { N } = P, E = P.keep.length, Eraw = P.raw.center.length;
    const D = h.d_pet;
    // op labels: the weight's short name (compress.0, w_in, input_linear, ...)
    const short = (name: string) => {
      const p = name.split('.');
      return /^\d+$/.test(p[p.length - 1]) ? p.slice(-2).join('.') : p[p.length - 1] === 'energy___0' ? 'last_layer' : p[p.length - 1];
    };
    const lin = (x: Tensor, name: string) =>
      g.linear(x, this.p(`${name}.weight`), this.has(`${name}.bias`) ? this.p(`${name}.bias`) : undefined, short(name));
    const normK = h.normalization === 'RMSNorm' ? 'rms' : 'layer';
    const normf = (x: Tensor, name: string) =>
      g.norm(normK, x, this.p(`${name}.weight`), this.has(`${name}.bias`) ? this.p(`${name}.bias`) : null,
             normK === 'rms' ? 1.1920928955078125e-7 : 1e-5);
    const ff = (x: Tensor, name: string) => {
      const y = lin(x, `${name}.w_in`);
      if (h.activation === 'SwiGLU') {
        const w = y.cols / 2;
        return lin(g.mul(g.sliceCols(y, 0, w, 'value'), g.sigmoid(g.sliceCols(y, w, w, 'gate'))), `${name}.w_out`);
      }
      return lin(g.silu(y), `${name}.w_out`);
    };
    const mlp2 = (x: Tensor, name: string) => lin(g.silu(lin(x, `${name}.0`)), `${name}.2`);

    // ---- inputs and geometry
    const ix = (a: Int32Array, nSrc: number, out?: RowKind, src?: RowKind) => g.index(a, nSrc, { out, src });
    const atomsSpecies = ix(P.species, this.meta.atomic_types.length, 'atom');
    const nbSpecies = ix(Int32Array.from(P.neighbor, (j) => P.species[j]), this.meta.atomic_types.length, 'edge');
    const positions = g.constant(Float32Array.from(sys.positions.flat()), [N, 3], 'positions', 'atom');
    positions.requiresGrad = !!opts.forces;

    const { edgeVectors, kept, pair } = g.scope('geometry', () => {
      const vRaw = g.add(
        g.sub(g.gather(positions, ix(P.raw.neighbor, N, 'raw_edge', 'atom'), 'r_j'), g.gather(positions, ix(P.raw.center, N, 'raw_edge', 'atom'), 'r_i')),
        g.constant(Float32Array.from(P.raw.shiftVec), [Eraw, 3], 'shift', 'raw_edge'),
      );
      if (!P.adaptive) {
        vRaw.kind = 'edge';
        return { edgeVectors: vRaw, kept: vRaw, pair: g.constant(Float32Array.from(P.pairCutoff), [E], 'cutoff', 'edge') };
      }
      const A = P.adaptive;
      const atomic = g.scope('adaptive_cutoff', () => {
        const dRaw = g.rowNorm(vRaw);
        const rEdge = g.constant(Float32Array.from(P.raw.center, (i) => A.r[i]), [Eraw], 'r_probe', 'raw_edge');
        const counts = g.segmentSum(g.cutoff('bump', dRaw, rEdge, h.cutoff_width_adaptive), ix(P.raw.center, N, 'raw_edge', 'atom'), 'neighbour_count');
        const target = h.num_neighbors_adaptive!;
        const base = Float32Array.from(A.r, (r) => target * (r / h.cutoff) ** 3 - target);
        const residual = g.add(counts, g.constant(base, [N], 'baseline', 'atom'));
        const step = g.div(residual, g.constant(Float32Array.from(A.dn), [N], 'slope', 'atom'));
        const a = g.sub(g.constant(Float32Array.from(A.r), [N], 'root', 'atom'), step);
        return g.unary('clamp', a, h.cutoff / 16, h.cutoff);
      });
      const keep = ix(P.keep, Eraw, 'edge', 'raw_edge');
      const pairC = g.scale(g.add(g.gather(atomic, ix(P.center, N, 'edge', 'atom'), 'cutoff_i'), g.gather(atomic, ix(P.neighbor, N, 'edge', 'atom'), 'cutoff_j')), 0.5);
      return { edgeVectors: vRaw, kept: g.gather(vRaw, keep, 'keep'), pair: pairC };
    });

    const { dist, cf } = g.scope('cutoff', () => {
      const dist = g.rowNorm(kept);
      return { dist, cf: g.cutoff(h.cutoff_function === 'Bump' ? 'bump' : 'cosine', dist, pair, h.cutoff_width) };
    });

    // ---- token layout: atom i -> [node token, its edges]
    const tokIdx = new Int32Array(N + E), centerTok = new Int32Array(N), edgeTok = new Int32Array(E);
    const off = new Int32Array(N + 1), poff = new Int32Array(N + 1), tok2seq = new Int32Array(N + E);
    for (let i = 0, t = 0; i < N; i++) {
      off[i] = t; centerTok[i] = t; tok2seq[t] = i; tokIdx[t++] = i;
      for (let e = P.offsets[i]; e < P.offsets[i + 1]; e++) { edgeTok[e] = t; tok2seq[t] = i; tokIdx[t++] = N + e; }
      off[i + 1] = t;
      const L = off[i + 1] - off[i];
      poff[i + 1] = poff[i] + h.num_heads * L * L;
    }
    const layout: SeqLayout = { nSeq: N, nTok: N + E, nProb: poff[N], off: g.own(be.uploadI32(off)),
                                poff: g.own(be.uploadI32(poff)), tok2seq: g.own(be.uploadI32(tok2seq)) };
    const toTokens = ix(tokIdx, N + E, 'token', 'stack');
    const takeCenter = ix(centerTok, N + E, 'atom', 'token'), takeEdges = ix(edgeTok, N + E, 'edge', 'token');
    const reverse = ix(P.reverse, E, 'edge', 'edge');
    const bias = g.scope('cutoff', () => g.unary('logclamp',
      g.gather(g.concatRows(g.constant(new Float32Array(N).fill(1), [N], 'self', 'atom'), cf), toTokens, 'tokens'), 1e-15));

    const x4 = g.scope('cutoff', () => g.concatCols([kept, dist], 'x4'));
    const attention: Output['attention'] = [];
    const attnTokens = (a: number) => ({
      offset: poff[a],
      atoms: Int32Array.from({ length: off[a + 1] - off[a] }, (_, t) => (t === 0 ? a : P.neighbor[P.offsets[a] + t - 1])),
    });

    const transformer = (L: number, node: Tensor, edges: Tensor) => {
      const expanded = h.d_node !== D;
      for (let b = 0; b < h.num_attention_layers; b++) {
        const pre = `gnn_layers.${L}.trans.layers.${b}`;
        [node, edges] = g.scope(`trans.${b}`, () => {
          const tokens = g.scope('tokens', () => {
            const nodeIn = expanded ? lin(node, `${pre}.center_contraction`) : node;
            return g.gather(g.concatRows(nodeIn, edges), toTokens, 'tokens');
          });
          const attn = (x: Tensor) => {
            const qkv = lin(x, `${pre}.attention.input_linear`);
            const o = g.attention(qkv, bias, layout, h.num_heads, h.attention_temperature);
            attention.push({ node: g.tape[g.tape.length - 1], heads: h.num_heads, tokens: attnTokens });
            return lin(o, `${pre}.attention.output_linear`);
          };
          let nodeTok: Tensor, edgeOut: Tensor;
          if (h.transformer_type === 'PostLN') {
            const t1 = g.scope('attention', () => normf(g.add(tokens, attn(tokens)), `${pre}.norm_attention`));
            [nodeTok, edgeOut] = g.scope('mlp', () => {
              const t = normf(g.add(t1, ff(t1, `${pre}.mlp`)), `${pre}.norm_mlp`);
              return [g.gather(t, takeCenter, 'node_token'), g.gather(t, takeEdges, 'edge_tokens')];
            });
          } else {
            [nodeTok, edgeOut] = g.scope('attention', () => {
              const t = attn(normf(tokens, `${pre}.norm_attention`));
              return [g.gather(t, takeCenter, 'node_token'), g.add(edges, g.gather(t, takeEdges, 'edge_tokens'))];
            });
            edgeOut = g.scope('mlp', () => g.add(edgeOut, ff(normf(edgeOut, `${pre}.norm_mlp`), `${pre}.mlp`)));
          }
          if (!expanded) return [nodeTok, edgeOut];
          const n2 = g.scope('node_update', () => {
            const n = g.add(node, lin(nodeTok, `${pre}.center_expansion`));
            return g.add(n, ff(normf(n, `${pre}.norm_center_features`), `${pre}.center_mlp`));
          });
          return [n2, edgeOut];
        });
      }
      return [node, edges] as const;
    };

    const gnn = (L: number, node: Tensor, messages: Tensor) => g.scope(`gnn.${L}`, () => {
      const pre = `gnn_layers.${L}`;
      const edgeTokens = g.scope('edge_tokens', () => {
        const edgeEmb = lin(x4, `${pre}.edge_embedder`);
        const parts = L === 0 ? [edgeEmb, messages]
          : [edgeEmb, g.gather(this.p(`${pre}.neighbor_embedder.weight`), nbSpecies, 'neighbor_embedding'), messages];
        return mlp2(g.concatCols(parts), `${pre}.compress`);
      });
      return transformer(L, node, edgeTokens);
    });

    const embed = (name: string, idx: Index) => g.scope('embedding', () => g.gather(this.p(`${name}.weight`), idx, name));
    const cond = this.conditioning(g, sys, N);

    const nodeFeats: Tensor[] = [], edgeFeats: Tensor[] = [];
    let edgeIn = embed('edge_embedder', nbSpecies);
    if (h.featurizer_type === 'feedforward') {
      let nodeIn = embed('node_embedders.0', atomsSpecies);
      for (let L = 0; L < h.num_gnn_layers; L++) {
        let [nodeOut, edgeOut] = gnn(L, nodeIn, edgeIn);
        if (cond) nodeOut = g.add(nodeOut, cond);
        nodeIn = nodeOut;
        edgeIn = g.scope(`gnn.${L}/message_passing`, () => {
          const cat = g.concatCols([edgeOut, g.gather(edgeOut, reverse, 'reverse_messages')]);
          const mixed = mlp2(g.norm('layer', cat, this.p(`combination_norms.${L}.weight`),
                                    this.p(`combination_norms.${L}.bias`), 1e-5), `combination_mlps.${L}`);
          return g.add(g.add(edgeIn, edgeOut), mixed);
        });
      }
      nodeFeats.push(nodeIn); edgeFeats.push(edgeIn);
    } else {
      for (let L = 0; L < h.num_gnn_layers; L++) {
        let [nodeOut, edgeOut] = gnn(L, embed(`node_embedders.${L}`, atomsSpecies), edgeIn);
        if (cond) nodeOut = g.add(nodeOut, cond);
        nodeFeats.push(nodeOut); edgeFeats.push(edgeOut);
        edgeIn = g.scope(`gnn.${L}/message_passing`, () =>
          g.scale(g.add(edgeIn, g.gather(edgeOut, reverse, 'reverse_messages')), 0.5));
      }
    }

    // ---- readout: node and edge heads per readout layer, edge terms weighted by f_c
    const centers = ix(P.center, N, 'edge', 'atom');
    const readout = (target: string) => {
      let total: Tensor | null = null;
      nodeFeats.forEach((nf, i) => {
        const head = (x: Tensor, kind: 'node' | 'edge') => g.scope(`${kind}_head.${i}`, () => {
          const pre = `${kind}_heads.${target}.${i}`;
          const z = g.silu(lin(g.silu(lin(x, `${pre}.0`)), `${pre}.2`));
          return lin(z, `${kind}_last_layers.${target}.${i}.${target}___0`);
        });
        const nodePred = head(nf, 'node');
        const edgePred = g.segmentSum(g.mul(head(edgeFeats[i], 'edge'), cf), centers, 'sum_edges');
        const s = g.add(nodePred, edgePred);
        total = total ? g.add(total, s) : s;
      });
      return total!;
    };
    const perAtomNet = g.scope('readout', () => readout('energy'));
    let ncForces: Tensor | undefined;
    if (opts.nc && this.hasNC) {
      const scale = this.meta.non_conservative!.non_conservative_force.scale;
      ncForces = g.scope('nc_forces', () => {
        const F = g.mul(readout('non_conservative_force'), g.constant(Float32Array.from(P.species, (s) => scale[s]), [N], 'scale', 'atom'));
        // remove the net force (as metatomic does): F_i - mean_j F_j
        const all = g.index(new Int32Array(N), 1, { src: 'atom' });
        const mean = g.scale(g.segmentSum(F, all, 'net_force'), 1 / N);
        return g.sub(F, g.gather(mean, all, 'mean_force'));
      });
    }
    const comp = Float32Array.from(P.species, (s) => this.meta.composition_energies[s]);
    const perAtom = g.scope('energy', () =>
      g.add(g.scale(g.sumRows(perAtomNet), this.meta.energy_scale), g.constant(comp, [N], 'composition', 'atom')));
    const energy = g.scope('energy', () => g.sumAll(perAtom, 'total_energy'));
    const graph = { center: P.center, neighbor: P.neighbor, label: 'neighbour',
                    shift: Float32Array.from({ length: 3 * E }, (_, k) => P.raw.shiftVec[3 * P.keep[(k / 3) | 0] + (k % 3)]) };
    return { energy, ncForces, perAtom, positions, edgeVectors, virialVectors: edgeVectors, cutoffFactors: cf, attention,
             prepared: P, layout, graph, rows: petRows(P) };
  }

  private conditioning(g: Graph, sys: System, N: number): Tensor | null {
    if (!this.h.system_conditioning) return null;
    return g.scope('conditioning', () => {
      const c = (sys.charge ?? 0) + this.h.max_charge, s = (sys.spin ?? 1) - 1;
      const pick = (name: string, i: number) => g.gather(this.p(name), g.index(Int32Array.of(i), this.p(name).shape[0]));
      const e = g.concatCols([pick('system_conditioning.charge_embedding.weight', c),
                              pick('system_conditioning.spin_multiplicity_embedding.weight', s)]);
      const x = g.linear(e, this.p('system_conditioning.project.0.weight'), this.p('system_conditioning.project.0.bias'));
      const y = g.linear(g.silu(x), this.p('system_conditioning.project.2.weight'), this.p('system_conditioning.project.2.bias'));
      return g.gather(y, g.index(new Int32Array(N), 1), 'broadcast');
    });
  }
}

export interface Result {
  energy: number;
  energies: Float32Array;
  forces?: Float32Array; // [N*3]
  stress?: number[][]; // eV/A^3
  prepared: Prepared;
}

/** One evaluation: energy, and forces (and stress for periodic cells) by backprop. */
export async function evaluate(model: PET, sys: System, opts: { forces?: boolean; graph?: Graph } = {}): Promise<Result> {
  const g = opts.graph ?? new Graph(model.be);
  try {
    const out = model.forward(g, sys, { forces: opts.forces });
    if (opts.forces) g.backward(out.energy);
    const be = model.be;
    const [e, energies] = [await be.read(out.energy.buf), await be.read(out.perAtom.buf)];
    const res: Result = { energy: e[0], energies, prepared: out.prepared };
    if (opts.forces) {
      const gp = await be.read(out.positions.grad!);
      res.forces = gp.map((x) => -x);
      const V = volume(sys);
      if (Number.isFinite(V) && out.edgeVectors.grad) {
        const v = await be.read(out.edgeVectors.buf), gv = await be.read(out.edgeVectors.grad);
        const s = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
        for (let k = 0; k < v.length / 3; k++)
          for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) s[a][b] += (v[3 * k + a] * gv[3 * k + b]) / V;
        res.stress = s;
      }
    }
    return res;
  } finally {
    if (!opts.graph) g.release();
  }
}
