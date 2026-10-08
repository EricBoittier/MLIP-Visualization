// MACE (mace-torch ScaleShiftMACE, e.g. MACE-MP-0) in the visualiser.
import { atomTag, type Ctx, dims, neighbours } from '../../viz/article';
import type { Mod, ModDesc } from '../../viz/modules';
import { type ModelUI, NN_TERMS } from '../ui';
import type { MaceMeta } from './model';

const seg = (m: Mod) => m.path.split('/').pop()!.replace(/\.\d+$/, '');
const layer = (m: Mod) => +(m.path.match(/(?:interaction|product|readout)\.(\d+)/)?.[1] ?? 0) + 1;
const irreps = (C: number, ls: number[]) => ls.map((l) => `${C}×${l}${l % 2 ? 'o' : 'e'}`).join(' + ');
const corr = (meta: MaceMeta) => Math.max(...meta.products.flatMap((p) => p.contractions.flatMap((c) => c.nu)));

function describe(s: string): ModDesc | null {
  let m: RegExpMatchArray | null;
  if ((m = s.match(/^interaction\.(\d+)$/))) return { type: 'interaction', title: `Interaction ${+m[1] + 1}`, short: `Int. ${+m[1] + 1}` };
  if ((m = s.match(/^product\.(\d+)$/))) return { type: 'product', title: `Product basis ${+m[1] + 1}`, short: `Prod. ${+m[1] + 1}` };
  if ((m = s.match(/^readout\.(\d+)$/))) return { type: 'readout', title: `Readout ${+m[1] + 1}`, short: `ε ${+m[1] + 1}` };
  const named: Record<string, [string, string, string?]> = {
    geometry: ['geometry', 'Edges', 'r_ij'], spherical_harmonics: ['angular', 'Spherical harmonics', 'Y_lm'],
    radial: ['rbf', 'Radial features', 'R(r)'], repulsion: ['electrostatics', 'ZBL repulsion', 'ZBL'],
    embedding: ['embedding', 'Element embedding', 'Embed'], skip: ['residual', 'Element-wise skip', 'Skip'],
    radial_mlp: ['mlp', 'Radial MLP', 'MLP(R)'], message: ['message', 'Equivariant messages', 'Messages'],
    density: ['density', 'Density normalisation', '÷ (ρ + 1)'], energy: ['energy', 'Energy'],
  };
  const d = named[s];
  return d ? { type: d[0], title: d[1], short: d[2] } : null;
}

function subtitle(m: Mod, meta: MaceMeta): string {
  const C = meta.channels, i = layer(m) - 1;
  switch (seg(m)) {
    case 'geometry': return `pairs within ${meta.r_max} Å`;
    case 'spherical_harmonics': return `Y_lm(r̂_ij), l ≤ ${meta.sh_lmax}`;
    case 'radial': return `${meta.bessel.weights.length} Bessel functions × envelope${meta.agnesi ? ' (Agnesi distance)' : ''}`;
    case 'repulsion': return 'screened nuclear repulsion';
    case 'embedding': return `Z → ${C} channels`;
    case 'interaction': return `${irreps(C, meta.interactions[i].ls_in)} → ${irreps(C, meta.interactions[i].ls_target)}`;
    case 'skip': return 'a linear map per element';
    case 'radial_mlp': return `${meta.interactions[i].radial.map((s) => s[1]).join('→')}→${meta.interactions[i].radial.at(-1)![0]}`;
    case 'message': return 'Σ_j R(r_ij) h_j ⊗ Y(r̂_ij)';
    case 'density': return 'divide by 1 + Σ_j tanh(f(r_ij)²)';
    case 'product': return `Σ_ν W_ν(Z) · (A ⊗ … ⊗ A), ν ≤ ${corr(meta)} → ${irreps(C, meta.products[i].ls_out)}`;
    case 'readout': return meta.readouts[i].kind === 'linear' ? 'linear → ε_i' : 'MLP → ε_i';
    case 'energy': return `Σ_i E0(Z_i) + ${meta.scale.toFixed(3)} Σ_layers ε_i`;
    default: return '';
  }
}

function narration(m: Mod | 'backward', meta: MaceMeta): string {
  const C = meta.channels;
  if (m === 'backward') return `
    <p>Forces are −∂<i>E</i>/∂<b>r</b>. The gradient flows back through the readouts, through every product basis (a
    polynomial in the messages, so each term passes gradient to all its factors), through the tensor products to the
    spherical harmonics and radial features of every edge, and from those to the atoms at both ends.</p>`;
  switch (seg(m)) {
    case 'geometry': return `<p><b>MACE</b> (Batatia et al., 2022) is an equivariant message-passing network: its features are
      not just numbers but vectors and tensors that rotate with the structure, built from the directions to neighbours.
      Its many-body product basis then gives each layer the expressiveness of high body-order ACE, so two layers suffice.</p>
      <p>Every pair of atoms closer than ${meta.r_max} Å is an edge.</p>`;
    case 'spherical_harmonics': return `<p><b>Spherical harmonics.</b> The direction of every edge is written as the real spherical
      harmonics <i>Y</i><sub>lm</sub>(<b>r̂</b><sub>ij</sub>) up to <i>l</i> = ${meta.sh_lmax}: one number for <i>l</i> = 0, three
      (a vector) for <i>l</i> = 1, five for <i>l</i> = 2, seven for <i>l</i> = 3. Rotating the structure mixes the components of
      each <i>l</i> among themselves and nothing else: this is what makes the features equivariant.</p>`;
    case 'radial': return `<p><b>Radial features.</b> The distance goes into ${meta.bessel.weights.length} Bessel functions
      sin(<i>nπx</i>/<i>r</i><sub>max</sub>)/<i>x</i>${meta.agnesi ? `, of a transformed distance <i>x</i> (the Agnesi transform, scaled by the two atoms' covalent radii)` : ''},
      times a polynomial envelope that brings them smoothly to zero at ${meta.r_max} Å.</p>`;
    case 'repulsion': return `<p><b>ZBL repulsion.</b> Pairs closer than the sum of their covalent radii feel a screened nuclear
      repulsion, so the model never lets atoms collapse onto each other.</p>`;
    case 'embedding': return `<p><b>Embedding.</b> Each atom starts with ${C} scalar channels, a learned vector per element.</p>`;
    case 'interaction': return `<p><b>Interaction ${layer(m)}.</b> Every neighbour <i>j</i> sends its features, coupled to the edge's
      spherical harmonics by a tensor product: an <i>l</i><sub>1</sub> feature times an <i>l</i><sub>2</sub> harmonic gives
      <i>l</i><sub>3</sub> parts for each allowed |<i>l</i><sub>1</sub> − <i>l</i><sub>2</sub>| ≤ <i>l</i><sub>3</sub> ≤ <i>l</i><sub>1</sub> + <i>l</i><sub>2</sub>,
      weighted per channel by a function of the distance. The sum over neighbours is the atomic basis <i>A</i>.</p>`;
    case 'skip': return `<p><b>Element-wise skip.</b> A linear map chosen by the atom's element (mace's "selector" tensor product with the
      one-hot elements).</p>`;
    case 'radial_mlp': return `<p><b>Radial MLP.</b> The radial features of each edge go through a small network that gives one weight
      per channel and per tensor-product path.</p>`;
    case 'message': return `<p><b>Messages.</b> For each path (<i>l</i><sub>1</sub> ⊗ <i>l</i><sub>2</sub> → <i>l</i><sub>3</sub>), the
      Clebsch–Gordan coefficients combine the neighbour's <i>l</i><sub>1</sub> components with the edge's <i>l</i><sub>2</sub> harmonics
      into <i>l</i><sub>3</sub> components; the result is weighted by the radial MLP, summed onto the atom, and mixed across channels.</p>`;
    case 'density': return `<p><b>Density normalisation.</b> Instead of dividing by a fixed average number of neighbours, each atom
      divides by 1 + a smooth count of its neighbours, so dense and sparse environments are on the same scale.</p>`;
    case 'product': return `<p><b>Product basis ${layer(m)}.</b> The heart of MACE: the atomic basis <i>A</i> is multiplied with itself,
      up to ${corr(meta)} times, and the products are projected back to features of each <i>l</i> by fixed generalised
      Clebsch–Gordan tables <i>U</i>, with learned weights per element and channel. A product of ν messages, each from a
      neighbour, makes (ν + 1)-body features in a single layer.</p>`;
    case 'readout': return `<p><b>Readout ${layer(m)}.</b> Every layer contributes an atomic energy from its scalar (<i>l</i> = 0) channels.</p>`;
    case 'energy': return `<p>The energy: the readouts (and repulsion), scaled by ${meta.scale.toFixed(4)}, plus a fixed reference energy
      per element (E0). The backward pass follows.</p>`;
    default: return '';
  }
}

function step(i: number, mod: Mod, c: Ctx, lastOp: number): { key: string; html: string } {
  const { ops, pass } = c, meta = c.meta as MaceMeta;
  const t = pass.trace, a = t.selected, op = ops[i], out = dims(t.shapes[lastOp]), sh = dims(t.shapes[i]);
  const at = (k: number) => atomTag(pass, k);
  const S = (key: string, html: string) => ({ key, html });
  const cont = { key: 'cont', html: '' };
  const L = layer(mod), name = op.op;
  let m: RegExpMatchArray | null;
  switch (seg(mod)) {
    case 'geometry': return name === 'r_j' ? S('vec', `${t.shapes[i][0]} edges, ${neighbours(pass, a).length} of them around ${at(a)}.`) : cont;
    case 'spherical_harmonics':
      return name === 'e3nn_harmonics' ? S('Y', `The harmonics of every edge direction, ${(meta.sh_lmax + 1) ** 2} numbers ${sh}.`) : cont;
    case 'radial':
      return name === 'edge_features' ? S('R', `Bessel functions times the envelope: the radial features ${sh}.`) : cont;
    case 'repulsion': return name === 'per_atom' ? S('zbl', `Repulsion of the pairs inside their covalent radii, summed onto atoms.`) : cont;
    case 'embedding': return S('emb', `Look up each atom's element ${out}.`);
    case 'skip': return (m = name.match(/^skip_(\w+)$/)) ? S(`skip${m[1]}`, `The ${m[1]} atoms' own linear map.`) : cont;
    case 'radial_mlp': return name === 'layer0' ? S('mlp', `Interaction ${L}: the radial MLP gives ${t.shapes[lastOp][1]} weights per edge.`) : cont;
    case 'message':
      if ((m = name.match(/^x_j_l(\d)$/))) return S(`xj${m[1]}`, `The neighbours' <i>l</i> = ${m[1]} features, one block of ${2 * +m[1] + 1} rows per edge ${sh}.`);
      if ((m = name.match(/^tensor_product_(\d)x(\d)→(\d)$/))) return S(`tp${m.slice(1).join('')}`, `Path ${m[1]} ⊗ ${m[2]} → ${m[3]}: Clebsch–Gordan coupling with the edge's <i>l</i> = ${m[2]} harmonics ${sh}.`);
      if ((m = name.match(/^sum_(\d)x(\d)→(\d)$/))) return S(`sum${m.slice(1).join('')}`, `Weighted by the radial MLP and summed onto each atom ${sh}.`);
      return cont;
    case 'density': return name === 'density' ? S('rho', `Each atom's smooth neighbour count ρ; the messages are divided by ρ + 1.`) : cont;
    case 'product':
      if (name === 'per_channel') return S('A', `Product ${L}: the atomic basis rearranged per channel, ${t.shapes[i][1]} components each ${sh}.`);
      if (name === 'x_a x_b') return S('A2', `All products of two components (${t.shapes[i][1]} distinct ones).`);
      if (name === 'x_a x_b x_c') return S('A3', `…and of three (${t.shapes[i][1]}).`);
      if ((m = name.match(/^B_L(\d)_nu(\d)$/))) return S(`B${m[1]}${m[2]}`, `Order ${m[2]}, projected to <i>L</i> = ${m[1]} by <i>U</i> and weighted per element and channel.`);
      if ((m = name.match(/^linear_l(\d)$/))) return S(`lin${m[1]}`, `Mixed across channels: the new <i>l</i> = ${m[1]} features ${sh}.`);
      return cont;
    case 'readout': return name === 'atomic_energy' ? S('eps', `Layer ${L}'s atomic energies ${out}.`) : cont;
    case 'energy': return name === 'total_energy' ? S('total', `<b>E = ${pass.energy.toFixed(4)} eV</b>.`) : cont;
    default: return cont;
  }
}

export const maceUI: ModelUI = {
  kind: 'mace',
  name: 'MACE',
  family: 'equivariant message passing (ACE)',
  terms: NN_TERMS,
  describe,
  subtitle,
  narration,
  step,
  back: (m) => ({ energy: 'The gradient enters at the energy and splits over the layers\' readouts.',
    readout: 'Back through the readout to the scalar channels.',
    product: 'Through the polynomial: every factor of every product receives gradient.',
    density: 'Through the normalisation, also to the neighbour count.',
    message: 'Each message passes its gradient to the neighbour that sent it, to the harmonics and to the radial weights.',
    radial_mlp: 'Through the radial MLP to the distances.', skip: 'Through the element-wise maps.',
    interaction: 'Back through the interaction.', radial: 'Through the Bessel functions to the distances.',
    spherical_harmonics: 'Through the harmonics to the edge directions.', geometry: 'Finally onto the atoms.' } as Record<string, string>)[seg(m)] ?? '',
  collapse: (m) => (/(interaction|product)\.[1-9]/.test(m.path) && m.depth === 1) || seg(m) === 'skip',
  intro: (meta: MaceMeta) => `<h1>MACE, step by step</h1>
    <p>MACE builds equivariant features from the directions to neighbouring atoms and multiplies them together into
    many-body features in every layer. This is ${meta.name ? `<b>${meta.name}</b>, ` : ''}mace-torch's model with
    ${meta.interactions.length} layers of ${irreps(meta.channels, meta.products[0].ls_out)} features, run here op by op.</p>
    <p>Rows of the equivariant blocks are (atom, <i>m</i>): three rows per atom for vectors, five for <i>l</i> = 2.
    <b>Click an atom</b> to follow it.</p>`,
  card: (meta: MaceMeta, nParams, label) => `<b>${label}</b> · ${(nParams / 1e6).toFixed(2)} M parameters<br>
    ${meta.interactions.length} layers · ${irreps(meta.channels, meta.products[0].ls_out)} · harmonics to l = ${meta.sh_lmax} ·
    correlation ${corr(meta)} · cutoff ${meta.r_max} Å<br>
    ${meta.atomic_numbers.length} elements${meta.zbl ? ' · ZBL' : ''}${meta.agnesi ? ' · Agnesi distance' : ''}<br>
    <span style="opacity:.7">mace-torch ScaleShiftMACE</span>`,
};
