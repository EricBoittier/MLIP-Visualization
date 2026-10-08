// LOREM in the visualiser.
import { atomTag, type Ctx, dims, neighbours } from '../../viz/article';
import type { Mod, ModDesc } from '../../viz/modules';
import { type ModelUI, NN_TERMS } from '../ui';
import type { LoremMeta } from './model';

const seg = (m: Mod) => m.path.split('/').pop()!;

function describe(s: string): ModDesc | null {
  const named: Record<string, [string, string, string?]> = {
    geometry: ['geometry', 'Edges', 'r_ij'],
    spherical_harmonics: ['angular', 'Spherical harmonics', 'Y_lm'],
    radial: ['rbf', 'Radial basis', 'Bernstein'],
    embedding: ['embedding', 'Species embedding', 'Embed'],
    density: ['message', 'Scalar density', 'Density'],
    spherical: ['product', 'Tensor dense', 'CG'],
    readout: ['readout', 'Short-range energy', 'ε_sr'],
    message: ['interaction', 'Message passing', 'MP'],
    charges: ['charges', 'Learned charges', 'q'],
    potential: ['electrostatics', 'Coulomb potential', 'V(q)'],
    long_range: ['electrostatics', 'Long-range energy', 'ε_lr'],
    energy: ['energy', 'Energy'],
  };
  const d = named[s];
  return d ? { type: d[0], title: d[1], short: d[2] } : null;
}

function subtitle(m: Mod, meta: LoremMeta): string {
  const nlm = (meta.max_degree + 1) ** 2;
  switch (seg(m)) {
    case 'geometry': return `pairs within ${meta.cutoff} Å`;
    case 'spherical_harmonics': return `Racah Y_lm, ℓ ≤ ${meta.max_degree} (${nlm} components)`;
    case 'radial': return `${meta.num_radial} Bernstein polynomials × cosine cutoff`;
    case 'embedding': return `Z → ${meta.num_species} → ${meta.num_features} channels`;
    case 'density': return 'pair features summed onto each atom, then two residual MLPs';
    case 'spherical': return `self-product of the ℓ ≤ ${meta.max_degree} features`;
    case 'readout': return 'an atomic energy from the scalar features';
    case 'message': return meta.equivariant_message_passing ? 'scalar messages, then a Clebsch–Gordan message' : 'scalar messages';
    case 'charges': return `one scalar and ${(meta.max_degree_lr + 1) ** 2} spherical channels`;
    case 'potential': return 'Ewald if the cell is periodic, otherwise 1/r over every pair';
    case 'long_range': return 'the potential mixed back into the spherical features';
    case 'energy': return 'short-range residual + long-range residual';
    default: return '';
  }
}

function narration(m: Mod | 'backward', meta: LoremMeta): string {
  if (m === 'backward') return `
    <p>Forces are −∂<i>E</i>/∂<b>r</b>. The gradient comes back through both energy heads: through the long-range
    potential (every pair, or the Ewald sum) into the charges, and through the short-range tensor products into the
    spherical harmonics and the distances.</p>`;
  switch (seg(m)) {
    case 'geometry': return `<p><b>LOREM</b> (Bigi et al., 2025) keeps a short-range equivariant message passing model and a
      long-range Coulomb head in the same network. The head predicts charges — a scalar and a few spherical components —
      and the potential of those charges is folded back into the features.</p>
      <p>Every pair closer than ${meta.cutoff} Å is an edge. This checkpoint is a small random initialization of the
      architecture in metatrain's experimental port, so the energy is not a physical prediction; the calculation is the real one.</p>`;
    case 'spherical_harmonics': return `<p><b>Spherical harmonics.</b> The direction of each edge becomes the real spherical
      harmonics up to ℓ = ${meta.max_degree}, rescaled to Racah normalisation (each ℓ block times √(4π/(2ℓ+1))), the
      convention the rest of the tensor products assume. The cutoff does not multiply them.</p>`;
    case 'radial': return `<p><b>Radial basis.</b> ${meta.num_radial} Bernstein polynomials of <i>r</i> / ${meta.cutoff} Å,
      times the cosine cutoff ½(cos(π<i>r</i>/<i>r</i><sub>c</sub>) + 1).</p>`;
    case 'embedding': return `<p><b>Species embedding.</b> Each atomic number picks a learned vector of width ${meta.num_species}.
      The first scalar features are a linear map of that vector.</p>`;
    case 'density': return `<p><b>Scalar density.</b> The two embeddings on an edge go through a small MLP, which mixes the
      Bernstein basis into ${meta.num_features} features per edge. Those are summed onto the centre atom and folded in by
      a residual MLP with layer normalisation — twice, which is the port's <i>Update</i>.</p>`;
    case 'spherical': return `<p><b>Tensor dense.</b> The same edge features become one coefficient per degree, repeated across
      that degree's 2ℓ+1 components and multiplied by the harmonics. Summed onto the atom, the spherical tensor is projected
      and coupled with itself through a weighted Clebsch–Gordan product. The norm of each degree goes back into the scalars.</p>`;
    case 'readout': return `<p><b>Short-range energy.</b> A three-layer MLP reads an atomic energy from the scalar features.
      Later stages add their own residual rather than replacing this one.</p>`;
    case 'message': return `<p><b>Message passing.</b> One more round. Scalar messages are built the same way, now from the
      node features rather than the species embedding. The equivariant half filters the edge's spherical basis, couples it
      with the neighbour's spherical features, and combines that with the atom's own features — two Clebsch–Gordan products.</p>`;
    case 'charges': return `<p><b>Charges.</b> A scalar charge from the node features, and a spherical charge from another
      tensor dense, kept up to ℓ = ${meta.max_degree_lr}. These are not physical partial charges; they are the channels the
      Coulomb head contracts.</p>`;
    case 'potential': return `<p><b>Potential.</b> For a molecule, every pair contributes q<sub>j</sub>/(2<i>r</i><sub>ij</sub>).
      For a periodic cell the same charges go through an Ewald sum: erfc(<i>r</i>) inside the cutoff, and a reciprocal-space
      sum with Gaussian smearing ${meta.smearing} Å. The ½ is the double-counting factor of the pair sum, and it is learned around.</p>`;
    case 'long_range': return `<p><b>Long-range energy.</b> The spherical part of the potential is mixed, degree by degree, up to
      the short-range feature width and coupled with the short-range spherical features. Degree norms of that product, plus the
      scalar potential, update the nodes, and a last MLP reads the long-range atomic energy.</p>`;
    case 'energy': return `<p>The energy is the sum of the short-range residuals and the long-range residual. No composition
      baseline has been fitted on this checkpoint, so nothing else is added.</p>`;
    default: return '';
  }
}

function step(i: number, mod: Mod, c: Ctx, lastOp: number): { key: string; html: string } {
  const { ops, pass } = c, meta = c.meta as LoremMeta;
  const t = pass.trace, a = t.selected, op = ops[i], sh = dims(t.shapes[i]);
  const at = (k: number) => atomTag(pass, k);
  const S = (key: string, html: string) => ({ key, html });
  const cont = { key: 'cont', html: '' };
  const name = op.op;
  switch (seg(mod)) {
    case 'geometry': return name === 'edges' ? S('edges', `${t.shapes[i][0]} edges, ${neighbours(pass, a).length} of them around ${at(a)}.`) : cont;
    case 'spherical_harmonics': return name === 'racah' ? S('Y', `Harmonics of every edge direction, ${(meta.max_degree + 1) ** 2} numbers ${sh}, in Racah normalisation.`) : cont;
    case 'radial': return name === 'radial_basis' ? S('R', `${meta.num_radial} Bernstein functions times the cosine cutoff ${sh}.`) : cont;
    case 'embedding': return name === 'species' ? S('emb', `Each atom's species vector ${sh}.`) : cont;
    case 'density': return name === 'edge_scalar' ? S('edge', `The radial basis mixed by the pair MLP: ${t.shapes[i][0]} edges × ${meta.num_features} features.`) : cont;
    case 'spherical': return name === 'tensor_dense' ? S('cg', `Clebsch–Gordan self-product of the spherical features ${sh}.`) : cont;
    case 'readout': return name === 'atomic_energy' ? S('sr', `This stage's short-range atomic energies.`) : cont;
    case 'message': return name === 'cg_messages' ? S('msg', `Each neighbour's spherical features coupled with the filtered edge ${sh}.`)
      : name === 'combine' ? S('combine', `The messages combined with the atom's own spherical features.`) : cont;
    case 'charges': return name === 'charges' ? S('q', `Scalar charge plus spherical charges up to ℓ = ${meta.max_degree_lr} ${sh}.`) : cont;
    case 'potential': return name === 'ewald' || name === '1/r' ? S('V', name === 'ewald' ? `Ewald potential of the learned charges ${sh}.` : `Direct 1/r potential of the learned charges ${sh}.`) : cont;
    case 'long_range': return name === 'potential_product' ? S('mix', `Potential × short-range features, one Clebsch–Gordan product ${sh}.`)
      : name === 'atomic_energy' ? S('lr', `The long-range atomic energies.`) : cont;
    case 'energy': return name === 'total_energy' ? S('total', `<b>E = ${pass.energy.toFixed(4)} eV</b>.`) : cont;
    default: return cont;
  }
}

export const loremUI: ModelUI = {
  kind: 'lorem',
  name: 'LOREM',
  family: 'equivariant, with a long-range Coulomb head',
  terms: NN_TERMS,
  describe,
  subtitle,
  narration,
  step,
  back: (m) => ({
    energy: 'The gradient enters at the total energy and splits into the short-range and long-range residuals.',
    long_range: 'Back through the long-range MLP into the potential product and the charges.',
    potential: 'Through the Ewald sum or the pair potential, into the charges and the positions.',
    charges: 'The charge MLPs pass gradient to the scalar and spherical features.',
    message: 'Each Clebsch–Gordan product passes gradient to both factors: the neighbour that sent the message, and the edge.',
    readout: 'Back through the short-range energy MLP.',
    spherical: 'Through the self-product into the summed harmonics.',
    density: 'Through the scalar messages to both atoms of each edge.',
    radial: 'Through the Bernstein basis to the distances.',
    spherical_harmonics: 'Through the harmonics to the edge directions.',
    geometry: 'And so onto the atoms.',
    embedding: 'The embedding is constant for a given element.',
  } as Record<string, string>)[seg(m)] ?? '',
  preferredStructure: 'water',
  intro: (meta: LoremMeta) => `<h1>LOREM, step by step</h1>
    <p>${meta.name ? `<b>${meta.name}</b> is a` : 'A'} small LOREM: ${meta.num_features} scalar channels,
    spherical features of width ${meta.num_spherical_features} up to ℓ = ${meta.max_degree}, one message-passing step,
    and a long-range head up to ℓ = ${meta.max_degree_lr}. It is the architecture of metatrain's experimental port,
    with random weights, run here op by op.</p>
    <p>Rows of a spherical block are (atom, <i>m</i>). <b>Click an atom</b> to follow it. The scalar charge is in the property menu.</p>`,
  card: (meta: LoremMeta, nParams, label) => `<b>${label}</b> · ${nParams >= 1e6 ? `${(nParams / 1e6).toFixed(2)} M` : `${(nParams / 1e3).toFixed(1)} k`} parameters<br>
    ${meta.num_features} channels · ℓ ≤ ${meta.max_degree} · long range ℓ ≤ ${meta.max_degree_lr} · cutoff ${meta.cutoff} Å<br>
    ${meta.atomic_numbers.length} elements · ${meta.num_message_passing} message-passing step${meta.num_message_passing === 1 ? '' : 's'}<br>
    <span style="opacity:.7">random initialization of metatrain's LOREM — not a trained potential</span>`,
};
