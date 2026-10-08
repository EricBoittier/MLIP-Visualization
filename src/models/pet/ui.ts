// PET in the visualiser: module names, diagram subtitles, narration and model card.
import { ancestors, type ModDesc, type Mod } from '../../viz/modules';
import type { ModelUI } from '../ui';
import { petBack, petStep } from './article';
import type { Hypers, ModelMeta } from './checkpoint';

function describe(seg: string, path: string): ModDesc | null {
  let m: RegExpMatchArray | null;
  if ((m = seg.match(/^gnn\.(\d+)$/))) return { type: 'gnn', title: `GNN layer ${+m[1] + 1}` };
  if ((m = seg.match(/^trans\.(\d+)$/))) return { type: 'transformer', title: `Transformer block ${+m[1] + 1}` };
  if ((m = seg.match(/^(node|edge)_head\.(\d+)$/))) return { type: 'head', title: `${m[1] === 'node' ? 'Node' : 'Edge'} head` };
  const named: Record<string, [string, string, string?]> = {
    geometry: ['geometry', 'Geometry'], adaptive_cutoff: ['adaptive', 'Adaptive cutoff'], cutoff: ['cutoff', 'Distances & cutoff'],
    embedding: ['embedding', 'Embeddings'], edge_tokens: ['edge_tokens', 'Edge tokens'], tokens: ['tokens', 'Token sequences', 'Tokens'],
    attention: ['attention', 'Attention'], mlp: ['mlp', 'Feed-forward', 'FFN'], node_update: ['node_update', 'Atom update'],
    message_passing: ['message', 'Message passing'], readout: ['readout', 'Readout'], energy: ['energy', 'Energy'],
    conditioning: ['embedding', 'Charge & spin'], nc_forces: ['ncforce', 'Direct forces'],
  };
  const d = named[seg];
  void path;
  return d ? { type: d[0], title: d[1], short: d[2] } : null;
}


const layerOf = (m: Mod) => {
  const g = ancestors(m).find((a) => a.type === 'gnn');
  return g ? +g.path.split('.')[1] + 1 : 0;
};

function subtitle(m: Mod, h: Hypers): string {
  switch (m.type) {
    case 'geometry': return 'r_ij = r_j − r_i (+ shift)';
    case 'adaptive': return `≈ ${h.num_neighbors_adaptive} neighbours / atom`;
    case 'cutoff': return `f_c(r), ${h.cutoff_function.toLowerCase()}, ${h.cutoff} Å`;
    case 'embedding': return 'element → vector';
    case 'gnn': return `${h.num_attention_layers} transformer block${h.num_attention_layers > 1 ? 's' : ''}`;
    case 'edge_tokens': return '(x, y, z, r) ⊕ message → MLP';
    case 'transformer': return `${h.transformer_type}, ${h.normalization}`;
    case 'tokens': return 'atom ⊕ its edges';
    case 'attention': return `${h.num_heads} heads, bias log f_c`;
    case 'mlp': return `${h.activation}, ${h.d_feedforward}`;
    case 'node_update': return `${h.d_node} ↔ ${h.d_pet}`;
    case 'message': return 'i→j  ⇄  j→i';
    case 'readout': return 'node + Σ f_c · edge';
    case 'head': return 'Linear–SiLU ×2 → 1';
    case 'energy': return 'Σ_i s·ε_i + E_Z';
    case 'ncforce': return 'F_i predicted directly, no backprop';
    default: return '';
  }
}

function narration(m: Mod | 'backward', h: Hypers): string {
  if (m === 'backward') return `
    <p>Forces are <i>F</i> = −∂<i>E</i>/∂<i>r</i>. Reverse-mode differentiation walks the same graph
    backwards: each op receives the gradient of the energy with respect to its output
    (<span class="g">violet / green</span> cells) and hands it on to its inputs.</p>
    <p>The gradient flows back through the readout, every GNN layer and attention head, then through
    the cutoff function${h.num_neighbors_adaptive != null ? ' and the adaptive-cutoff root' : ''}
    to the edge vectors and finally the atomic positions. Its negative is the force on each atom,
    drawn as arrows in the structure.</p>`;
  const L = layerOf(m);
  const expanded = h.d_node !== h.d_pet;
  switch (m.type) {
    case 'geometry': return `
      <p>PET never sees absolute positions. Each atom <i>i</i> gets the neighbours <i>j</i> within
      ${h.cutoff} Å, periodic images included, and the vectors <b>r</b><sub>ij</sub> = <b>r</b><sub>j</sub> − <b>r</b><sub>i</sub> (+ cell shift).</p>
      <p>Rows are the selected atom's edges, columns <i>x, y, z</i>. In the structure, these edges are the graph PET
      works on, which is not the same thing as the chemical bonds.</p>`;
    case 'adaptive': return `
      <p>An <b>adaptive cutoff</b> gives each atom about ${h.num_neighbors_adaptive} neighbours. It solves for
      the radius at which a smooth neighbour count, plus a small cubic baseline, reaches the target (Newton steps
      with bisection as a fallback). A pair's cutoff is the mean of the two atoms' radii.</p>
      <p>The root is differentiated with the implicit function theorem, so forces also include how moving an atom
      changes its neighbours' cutoffs.</p>`;
    case 'cutoff': return `
      <p>Each distance goes through a smooth ${h.cutoff_function.toLowerCase()} cutoff <i>f</i><sub>c</sub>, which falls
      to zero over the last ${h.cutoff_width} Å. log <i>f</i><sub>c</sub> is added to the attention score of every key,
      and each edge's energy contribution is multiplied by <i>f</i><sub>c</sub>, so the energy stays smooth as
      neighbours come and go. Each edge's geometric input is the 4-vector (<i>x, y, z, r</i>).</p>`;
    case 'embedding': return `
      <p>Lookup tables. Each atom's element picks a row of the node embedding (${h.d_node} wide), and each edge picks the
      row of its neighbour's element (${h.d_pet} wide), which is the first "message" along that edge.</p>`;
    case 'edge_tokens': return `
      <p><b>GNN layer ${L}: edge tokens.</b> (<i>x, y, z, r</i>) is mapped linearly to ${h.d_pet} features and concatenated
      with the incoming message${L > 1 ? " and an embedding of the neighbour's element" : ''}. A two-layer MLP compresses the
      result to one token per edge.</p>`;
    case 'tokens': return `
      <p><b>Token sequences.</b> Each atom's sequence is its own token followed by one token per neighbour, so the
      transformer runs separately inside every atomic environment.${expanded ? ` The atom's ${h.d_node}-wide state is first
      contracted to ${h.d_pet}.` : ''}</p>`;
    case 'attention': return `
      <p><b>Local attention.</b> ${h.num_heads} heads of width ${h.d_pet / h.num_heads} attend within each sequence,
      with log <i>f</i><sub>c</sub> added to the score of every key. ${h.transformer_type === 'PreLN' ? 'The tokens are normalised first (pre-LN).' : 'The residual is added and then normalised (post-LN).'}</p>
      <p>The yellow squares are each head's attention matrix for the selected atom (rows: queries, columns: keys; first
      row and column: the atom itself). In the structure, the atom's edges are coloured by how strongly its own token
      attends to each neighbour.</p>`;
    case 'mlp': return `
      <p><b>Feed-forward.</b> A ${h.activation} network of width ${h.d_feedforward} on every edge token, with a residual
      connection and ${h.normalization}.</p>`;
    case 'node_update': return `
      <p><b>Atom update.</b> The atom's token is expanded back to ${h.d_node} features, added to its state, and passed
      through its own ${h.activation} MLP.</p>`;
    case 'message': return h.featurizer_type === 'feedforward' ? `
      <p><b>Message passing.</b> The output token of edge <i>i→j</i>, reversed, becomes the input message of edge
      <i>j→i</i> in the next layer. Both directions are concatenated, normalised, mixed by an MLP and added to the edge
      stream, so information travels one hop further with every layer.</p>` : `
      <p><b>Message passing.</b> The output of edge <i>i→j</i> is averaged with that of the reversed edge <i>j→i</i> to
      form the next layer's message, so information travels one hop further with every layer.</p>`;
    case 'readout': case 'head': return `
      <p><b>Readout.</b> Node and edge heads (Linear–SiLU–Linear–SiLU) and a final linear layer give one number per
      atom and one per edge${h.featurizer_type === 'residual' ? ', for every GNN layer' : ''}. The edge numbers are
      weighted by <i>f</i><sub>c</sub> and summed onto their central atom.</p>`;
    case 'ncforce': return `
      <p><b>Direct (non-conservative) forces.</b> PET-MAD carries a second set of heads that read the same final features
      and output a 3-vector per atom (node head) and per edge (edge head, weighted by <i>f</i><sub>c</sub> and summed onto
      the centre atom). The result is scaled per element, and the mean is subtracted so the forces sum to zero.</p>
      <p>This replaces the whole backward pass with a few small matrix products. The catch is that these forces are not the
      derivative of any energy, so they are not conservative: molecular dynamics driven by them alone does not conserve
      energy. A common remedy is multiple time stepping, which uses direct forces for most steps and corrects them with
      conservative forces every few steps.</p>`;
    case 'energy': return `
      <p>Each per-atom number is scaled and shifted by a per-element reference energy, and the results are summed:
      <i>E</i> = Σ<sub>i</sub> (<i>s</i> ε<sub>i</sub> + <i>E</i><sub>Z<sub>i</sub></sub>). This ends the forward pass;
      the backward pass follows.</p>`;
    default: return '';
  }
}

export const petUI: ModelUI = {
  kind: 'pet',
  name: 'PET',
  family: 'Point Edge Transformer (graph transformer)',
  describe,
  subtitle: (m, meta: ModelMeta) => subtitle(m, meta.hypers),
  narration: (m, meta: ModelMeta) => narration(m, meta.hypers) || (m !== 'backward' && m.parent ? narration(m.parent, meta.hypers) : ''),
  step: petStep,
  back: petBack,
  intro: () => `<h1>PET, step by step</h1>
    <p>PET, the <i>Point Edge Transformer</i>, predicts the energy of a structure from its atoms. Each atom looks
    at its neighbours through a small transformer, a few message-passing layers share that information, and the
    energy is a sum over atoms. Forces are the gradient of that energy with respect to the positions (or, with the
    <i>direct</i> option of PET-MAD, predicted outright by extra heads).</p>
    <p>Each operation's output is a block of numbers: rows are the selected atom's tokens, columns the features.
    On the right, the structure is drawn as the graph PET actually uses. <b>Click an atom</b> to follow it.</p>`,
  card: (meta: ModelMeta, nParams, label) => {
    const h = meta.hypers;
    return `<b>${label}</b> · ${(nParams / 1e6).toFixed(2)} M parameters<br>
      <code>d_pet</code> ${h.d_pet} · <code>d_node</code> ${h.d_node} · ${h.num_heads} heads · ${h.num_gnn_layers} GNN layers ×
      ${h.num_attention_layers} transformer blocks<br>${h.normalization} · ${h.activation} · ${h.transformer_type} ·
      ${h.featurizer_type} featurizer<br>cutoff ${h.cutoff} Å (${h.cutoff_function.toLowerCase()})${h.num_neighbors_adaptive != null ? `, adaptive: ~${h.num_neighbors_adaptive} neighbours` : ''}`;
  },
};
