// PET's walkthrough paragraphs: one per step of the pass, from the live trace.
import { atomTag, type Ctx, dims, fmt, neighbours } from '../../viz/article';
import type { Mod } from '../../viz/modules';
import type { OpInfo } from '../../worker/protocol';
import type { Hypers } from './checkpoint';

/** What op i does, as a paragraph, with a key naming the kind of step ('cont': part of the previous one). */
export function petStep(i: number, mod: Mod, c: Ctx, lastOp = i): { key: string; html: string } {
  const { ops, pass } = c, h: Hypers = c.meta.hypers;
  const t = pass.trace, a = t.selected, E = t.graph.center.length;
  const at = (k: number) => atomTag(pass, k);
  const mine = neighbours(pass, a), deg = mine.length, dist = mine.map((x) => x.r);
  const op = ops[i], name = op.params[0]?.replace(/\.weight$/, '') ?? op.op;
  const out = dims(t.shapes[lastOp]);
  const S = (key: string, html: string) => ({ key, html });
  const cont = { key: 'cont', html: '' };
  const norm = isNorm(op);

  switch (mod.type) {
    case 'geometry':
      if (op.op === 'r_j') return S('vec', `For every pair (<i>i</i>, <i>j</i>) closer than ${h.cutoff} Å, including periodic images, gather the
        neighbour's and the centre's positions and subtract them, adding the cell shift: <b>r</b><sub>ij</sub> = <b>r</b><sub>j</sub> − <b>r</b><sub>i</sub> + <b>S</b>·cell.
        There are ${t.shapes[i][0]} such vectors; the block shows the ${t.rowSel[i]?.length ?? 0} that start at ${at(a)}.`);
      if (op.op === 'cutoff_i') return S('pair', `Each pair's cutoff is the mean of its two atoms' adaptive radii, (<i>r</i><sub>i</sub> + <i>r</i><sub>j</sub>) / 2.`);
      if (op.op === 'keep') return S('keep', `Keep the pairs inside their cutoff: ${E} of ${t.shapes[0][0]} candidates remain, ${deg} of them around ${at(a)}. ${out}`);
      return cont;
    case 'adaptive':
      if (op.op === 'square') return S('dist', `Distances |<b>r</b><sub>ij</sub>| of all candidate pairs.`);
      if (op.op.startsWith('cutoff')) return S('count', `Count each atom's neighbours smoothly at the radius <i>r</i><sub>i</sub> found by the root solver:
        every pair contributes a bump function of its distance, and the contributions are summed per atom.`);
      if (op.op === 'clamp') return S('ift', `The step <i>r</i><sub>i</sub> − (<i>n</i>(<i>r</i><sub>i</sub>) − ${h.num_neighbors_adaptive}) / <i>n</i>′(<i>r</i><sub>i</sub>) gives each atom's cutoff.
        Its value is the root itself; the formula is there to carry the gradient (implicit function theorem). It is clamped to [${(h.cutoff / 16).toFixed(2)}, ${h.cutoff}] Å.`);
      return cont;
    case 'cutoff':
      if (op.op === 'square') return S('len', `Edge lengths <i>r</i> = |<b>r</b><sub>ij</sub>|. ${at(a)}'s ${deg} neighbours sit between
        ${fmt(Math.min(...dist), 2)} and ${fmt(Math.max(...dist), 2)} Å.`);
      if (op.op.startsWith('cutoff')) return S('fc', `The smooth cutoff <i>f</i><sub>c</sub>(<i>r</i>) (${h.cutoff_function.toLowerCase()}, falling to zero over the last
        ${h.cutoff_width} Å of the pair's cutoff): 1 for close neighbours, fading to 0 for the farthest.`);
      if (op.op === 'concat_rows') return S('bias', `Arrange <i>f</i><sub>c</sub> in token order (1 for the atom's own token) and take its logarithm:
        log <i>f</i><sub>c</sub> will be added to every attention score, so neighbours at the cutoff fade out smoothly.`);
      if (op.op === 'x4') return S('x4', `Stack (<i>x, y, z, r</i>) into a 4-vector per edge ${out}: the only geometric input the network gets.`);
      return cont;
    case 'embedding':
      return name.startsWith('node') ? S('node', `Look up each atom's element in the node embedding table: one ${h.d_node}-dimensional row per element. ${out}`)
        : S('edge', `Look up the <i>neighbour's</i> element for every edge (${h.d_pet} features), the first message along that edge. ${out}`);
    case 'edge_tokens':
      if (name.endsWith('edge_embedder')) return S('emb', `Multiply each edge's (<i>x, y, z, r</i>) by a 4 × ${h.d_pet} matrix and add a bias: a linear embedding of the geometry. ${out}`);
      if (op.op === 'neighbor_embedding') return S('nb', `Look up the neighbour's element again, in this layer's own table.`);
      if (op.op === 'concat') return S('cat', `Concatenate the geometry embedding, ${op.inputs.length > 2 ? "the neighbour's element and " : ''}the incoming message into one vector per edge ${out}.`);
      if (/compress\.0$/.test(name)) return S('c1', `Compression MLP, first layer and SiLU. ${out}`);
      if (/compress\.2$/.test(name)) return S('c2', `Compression MLP, second layer: one ${h.d_pet}-wide token per edge. ${out}`);
      return cont;
    case 'tokens':
      if (name.endsWith('center_contraction')) return S('contract', `Contract the atom's ${h.d_node}-wide state to ${h.d_pet} so it can sit in the same sequence as its edges.`);
      if (op.op === 'concat_rows') return S('seq', `Build the sequences: ${at(a)} first, then its ${deg} edges, so ${deg + 1} tokens of ${h.d_pet} features ${out}.
        Every atom has its own sequence; the rows shown are ${at(a)}'s.`);
      return cont;
    case 'attention': {
      if (norm && h.transformer_type === 'PreLN') return S('norm', `${h.normalization} of every token before attention (pre-LN).`);
      if (name.endsWith('input_linear')) return S('qkv', `Project each token to a query, a key and a value for each of the ${h.num_heads} heads:
        3 × ${h.num_heads} × ${h.d_pet / h.num_heads} = ${3 * h.d_pet} numbers per token ${dims(t.shapes[i])}.`);
      if (op.op === 'attention') {
        const A = t.attention.find((x) => x.op === i);
        let extra = '';
        if (A && A.n > 1) {
          const n = A.n, w = Array.from({ length: n }, (_, k) => { let s = 0; for (let q = 0; q < A.heads; q++) s += A.probs[q * n * n + k]; return s / A.heads; });
          let best = 1;
          for (let k = 2; k < n; k++) if (w[k] > w[best]) best = k;
          extra = ` Averaged over heads, ${at(a)} keeps ${(100 * w[0]).toFixed(0)}% of its attention on itself and gives the most,
            ${(100 * w[best]).toFixed(0)}%, to ${at(A.atoms[best])} at ${fmt(dist[best - 1], 2)} Å.`;
        }
        return S('attn', `Each head scores every query against every key of the same sequence, softmax(<b>q</b>·<b>k</b>/√${h.d_pet / h.num_heads} + log <i>f</i><sub>c</sub>),
          and averages the values with those weights. The ${h.num_heads} yellow squares are ${at(a)}'s ${deg + 1} × ${deg + 1} attention matrices.${extra}`);
      }
      if (name.endsWith('output_linear')) return S('out', `Mix the ${h.num_heads} heads' outputs back into ${h.d_pet} features.`);
      if (op.op === 'node_token') return S('split', `Split the sequence again: the first token is the atom's, the others are its edges, which add the attention output to their residual stream.`);
      if (op.op === 'add' && h.transformer_type === 'PostLN') return S('res', `Residual connection, then ${h.normalization} (post-LN).`);
      return cont;
    }
    case 'mlp':
      if (norm && h.transformer_type === 'PreLN') return S('norm', `${h.normalization} of every edge token.`);
      if (name.endsWith('w_in')) return S('ffn1', h.activation === 'SwiGLU'
        ? `Feed-forward, first layer: ${h.d_pet} → 2 × ${h.d_feedforward}, split into a value and a gate; SwiGLU multiplies the value by sigmoid(gate).`
        : `Feed-forward, first layer: ${h.d_pet} → ${h.d_feedforward}, then SiLU.`);
      if (name.endsWith('w_out')) return S('ffn2', `Feed-forward, second layer: back to ${h.d_pet}, added to the residual stream.`);
      if (op.op === 'add' && h.transformer_type === 'PostLN') return S('res', `Residual connection, then ${h.normalization} (post-LN).`);
      if (op.op === 'node_token') return S('split', `Split the sequence into the atom's token and its edge tokens.`);
      return cont;
    case 'node_update':
      if (name.endsWith('center_expansion')) return S('expand', `Expand the atom's token back to ${h.d_node} features and add it to the atom's state.`);
      if (norm) return S('norm', `${h.normalization} of the atom's state.`);
      if (name.endsWith('w_in')) return S('cmlp', `The atom's own feed-forward network (${h.d_node} → ${2 * h.d_node}${h.activation === 'SwiGLU' ? ' × 2, SwiGLU' : ''} → ${h.d_node}), added to its state.`);
      return cont;
    case 'message':
      if (op.op === 'reverse_messages') return S('rev', `Reverse every edge: row <i>i→j</i> takes the output of <i>j→i</i>. For ${at(a)}, these are the messages its neighbours computed about it.`);
      if (op.op === 'concat') return S('cat', `Put both directions side by side ${out}.`);
      if (norm) return S('norm', `LayerNorm of the pair.`);
      if (/combination_mlps\.\d+\.0$/.test(name)) return S('m1', `Combination MLP, first layer and SiLU.`);
      if (/combination_mlps\.\d+\.2$/.test(name)) return S('m2', `Combination MLP, second layer; the result is added to the edge stream and becomes the input of the next layer.`);
      if (op.op === 'add' && h.featurizer_type !== 'feedforward') return S('avg', `Average forward and reversed messages: the input of the next layer.`);
      return cont;
    case 'head': {
      const node = mod.title.startsWith('Node'), direct = mod.parent?.type === 'ncforce';
      if (name.includes('last_layers')) return S('last', direct
        ? `Final linear layer: three numbers, a force vector, per ${node ? 'atom' : 'edge'}${node ? '' : ', weighted by <i>f</i><sub>c</sub> and summed onto its centre atom'}.`
        : `Final linear layer: one number per ${node ? 'atom' : 'edge'}.`);
      if (/\.0$/.test(name)) return S('h1', `${node ? 'Node' : 'Edge'} head, first layer and SiLU, on every ${node ? 'atom' : 'edge'} ${out}.`);
      if (/\.2$/.test(name)) return S('h2', `${node ? 'Node' : 'Edge'} head, second layer and SiLU.`);
      return cont;
    }
    case 'readout':
      if (op.op === 'mul') return S('sum', `Weight each edge's number by its <i>f</i><sub>c</sub>, sum the edges onto their centre atom, and add the atom's own number.`);
      return cont;
    case 'ncforce': {
      if (op.op === 'mul') {
        const F = pass.ncForces;
        return S('ncscale', `Add node and edge vectors and multiply by a per-element scale${F ? `. The final force on ${at(a)} is
          <b>F</b> = (${[0, 1, 2].map((k) => F[3 * a + k].toFixed(3)).join(', ')}) eV/Å` : ''}.`);
      }
      if (op.op === 'net_force') return S('net', `Subtract the mean over atoms. Predicted per-atom forces need not sum to zero, and a
        leftover net force would push the whole structure; removing it is how metatomic serves these forces too.`);
      return cont;
    }
    case 'energy':
      if (op.op === 'total_energy') return S('total', `Sum over atoms: <b>E = ${pass.energy.toFixed(4)} eV</b>.`);
      if (op.op === 'sum') return S('scale', `Scale by ${fmt(c.meta.energy_scale)} and add each element's reference energy: ${at(a)} contributes ε = <b>${pass.energies[a].toFixed(4)} eV</b>.`);
      return cont;
    default:
      return S(op.op, `${op.op} ${dims(t.shapes[i])}`);
  }
}

const isNorm = (op: OpInfo) => op.op === 'layer_norm' || op.op === 'rms_norm';

export function petBack(m: Mod, c: Ctx, ops: number[]): string {
  const h: Hypers = c.meta.hypers;
  const atom = atomTag(c.pass, c.pass.trace.selected);
  if (m.type === 'geometry' && !ops.some((i) => c.ops[i].op === 'r_j')) return 'Back through the selection of kept edges and the pair cutoffs.';
  return backBody(m, h, c, atom);
}

function backBody(m: Mod, h: Hypers, c: Ctx, atom: string): string {
  switch (m.type) {
    case 'energy': return `The gradient enters at the top: ∂E/∂ε<sub>i</sub> = ${fmt(c.meta.energy_scale)} for every atom (the sum and the reference energies pass it straight through).`;
    case 'readout': return `Each atom's number receives the energy gradient; each edge's number receives it times <i>f</i><sub>c</sub>, and <i>f</i><sub>c</sub> in turn receives the edge's number.`;
    case 'head': return `Back through the ${m.title.toLowerCase()}: each linear layer multiplies the incoming gradient by its transposed weights, each SiLU by its slope.`;
    case 'message': return `Messages flow back to the edge that sent them: the reversed gather becomes a scatter, so each edge <i>j→i</i> receives the gradient of <i>i→j</i>.`;
    case 'node_update': return `Back through the atom's MLP and expansion to its token.`;
    case 'mlp': return `Back through the feed-forward network and its normalisation.`;
    case 'attention': return `Through attention: the gradient reaches the values (weighted by the attention), the queries and keys (through the softmax),
      and the bias log <i>f</i><sub>c</sub>. Summing the score gradients over heads and queries gives ∂E/∂log <i>f</i><sub>c</sub> for every key,
      one of the ways the forces feel the cutoff.`;
    case 'tokens': return `Split the token gradients back into the atom's state and its edges.`;
    case 'edge_tokens': return `Back through the compression MLP to the incoming message and to the 4-vector (<i>x, y, z, r</i>) of every edge.`;
    case 'embedding': return `The embeddings are constants of the structure; the gradient stops here.`;
    case 'cutoff': return `Collect everything that depends on the geometry: the (<i>x, y, z, r</i>) inputs of every layer and <i>f</i><sub>c</sub> (attention bias and readout weights),
      through d<i>f</i><sub>c</sub>/d<i>r</i> and d<i>r</i>/d<b>r</b><sub>ij</sub> = <b>r</b><sub>ij</sub>/<i>r</i>.`;
    case 'adaptive': return `Through the adaptive cutoff (implicit function theorem): moving any atom shifts the radius where the neighbour count is ${h.num_neighbors_adaptive}.`;
    case 'geometry': return `Finally, ∂E/∂<b>r</b><sub>ij</sub> is scattered onto the two atoms of each pair (+ at <i>j</i>, − at <i>i</i>). The negative is the force:
      ${atom} feels |<b>F</b>| = <b>${Math.hypot(...[0, 1, 2].map((k) => c.pass.forces![3 * c.pass.trace.selected + k])).toFixed(3)} eV/Å</b>.`;
    default: return `Gradient through ${m.title}.`;
  }
}

