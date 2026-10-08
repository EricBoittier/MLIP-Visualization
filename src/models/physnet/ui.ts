// PhysNet (mmml physnetjax) in the visualiser.
import { atomTag, type Ctx, dims, neighbours } from '../../viz/article';
import type { Mod, ModDesc } from '../../viz/modules';
import { type ModelUI, NN_TERMS } from '../ui';
import type { PhysNetMeta } from './model';

const cfg = (m: PhysNetMeta) => ({ n_refinement_blocks: m.config.n_refinement_blocks ?? m.config.n_res ?? 3, switch_end: 10, ...m.config });
const seg = (m: Mod) => m.path.split('/').pop()!.replace(/\.\d+$/, '');

function describe(s: string): ModDesc | null {
  let m: RegExpMatchArray | null;
  if ((m = s.match(/^interaction\.(\d+)$/))) return { type: 'interaction', title: `Interaction ${+m[1] + 1}`, short: `Int. ${+m[1] + 1}` };
  const named: Record<string, [string, string, string?]> = {
    geometry: ['geometry', 'Atom pairs', 'Pairs'], basis: ['rbf', 'Radial basis', 'g(r)'], embedding: ['embedding', 'Element embedding', 'Embed'],
    message: ['message', 'Message passing', 'Messages'], refinement: ['residual', 'Refinement', 'Refine'],
    energy_head: ['readout', 'Atomic energies', 'ε_i'], charges: ['charges', 'Atomic charges', 'q_i'],
    electrostatics: ['electrostatics', 'Electrostatics', 'Coulomb'], repulsion: ['electrostatics', 'ZBL repulsion', 'ZBL'], energy: ['energy', 'Energy'],
  };
  const d = named[s];
  return d ? { type: d[0], title: d[1], short: d[2] } : null;
}

function subtitle(m: Mod, meta: PhysNetMeta): string {
  const c = cfg(meta);
  switch (seg(m)) {
    case 'geometry': return `r_ij within ${c.cutoff} Å (and ${c.electrostatics_off_end ?? 10} Å for charges)`;
    case 'basis': return `${c.num_basis_functions} Chebyshev(e^{-r}) · cutoff`;
    case 'embedding': return `Z → ${c.features} features`;
    case 'interaction': return `message + ${c.n_refinement_blocks} residual blocks`;
    case 'message': return 'Σ_j W g(r_ij) ⊙ x_j ⊙ w';
    case 'refinement': return 'x + Dense(x + SiLU(x))';
    case 'energy_head': return 'linear → ε_i + b_Z';
    case 'charges': return 'linear → q_i + b_Z';
    case 'electrostatics': return `q_i q_j / r, switched, erf-damped`;
    case 'repulsion': return 'nuclear repulsion < 0.6 Å';
    case 'energy': return 'Σ_i (ε_i + E_elec,i + E_rep,i)';
    default: return '';
  }
}

function narration(m: Mod | 'backward', meta: PhysNetMeta): string {
  const c = cfg(meta);
  if (m === 'backward') return `
    <p>Forces are −∂<i>E</i>/∂<b>r</b>. The gradient runs back through the energy and charge heads, through every
    interaction (each message depends on the pair's distance through the radial basis), and through the Coulomb terms,
    which depend on the positions both directly (1/<i>r</i>) and through the charges.</p>`;
  switch (seg(m)) {
    case 'geometry': return `<p><b>PhysNet</b> (Unke & Meuwly, 2019) is a message-passing network: atoms exchange information
      with their neighbours over several rounds, and the energy is a sum of atomic contributions plus explicit electrostatics.
      This is mmml's JAX implementation (<code>physnetjax</code>) in its invariant form.</p>
      <p>It starts from every pair of atoms: those within ${c.cutoff} Å pass messages; pairs up to
      ${c.electrostatics_off_end ?? 10} Å also feel each other's charges.</p>`;
    case 'basis': return `<p><b>Radial basis.</b> Each pair distance is expanded in ${c.num_basis_functions} functions,
      Chebyshev polynomials <i>T</i><sub>k</sub>(2e<sup>−r</sup> − 1), multiplied by a smooth cutoff
      exp(1 − 1/(1 − (<i>r</i>/${c.cutoff})²)) that goes to zero, with all its derivatives, at ${c.cutoff} Å.</p>`;
    case 'embedding': return `<p><b>Embedding.</b> Each atom starts as a learned vector of ${c.features} numbers for its element.</p>`;
    case 'interaction': case 'message': return `<p><b>Message passing.</b> Atom <i>i</i> receives, from each neighbour <i>j</i>,
      its features <b>x</b><sub>j</sub> multiplied elementwise by a filter computed from the pair's radial basis
      (a linear map of <i>g</i>(<i>r</i><sub>ij</sub>)) and by a learned weight per feature, and sums them. Because the filters vanish at the
      cutoff, the energy stays smooth when neighbours come and go.</p>`;
    case 'refinement': return `<p><b>Refinement.</b> ${c.n_refinement_blocks} residual blocks, each adding Dense(<b>x</b> + SiLU(<b>x</b>))
      to the features, then a final Dense layer and SiLU: the atom's new state after this interaction.</p>`;
    case 'energy_head': return `<p><b>Atomic energies.</b> Two linear maps of the final features give one number per atom${c.use_energy_bias ? ', plus a learned bias per element' : ''}.</p>`;
    case 'charges': return `<p><b>Atomic charges.</b> A second head predicts a partial charge on every atom from the same features
      (plus a bias per element). PhysNet learns them from reference dipole moments; here they also drive the electrostatic energy.</p>`;
    case 'electrostatics': return `<p><b>Electrostatics.</b> Each pair contributes <i>q</i><sub>i</sub><i>q</i><sub>j</sub> times a
      shielded 1/<i>r</i>: at short range the Coulomb law is smoothly replaced by 1/√(<i>r</i>² + 1), it is damped by
      erf(<i>r</i>/${c.electrostatics_damping_sigma ?? 4}), shifted so that it vanishes at ${c.switch_end} Å, and switched off between
      ${c.electrostatics_off_start ?? 8} and ${c.electrostatics_off_end ?? 10} Å.</p>`;
    case 'repulsion': return `<p><b>ZBL repulsion.</b> A screened nuclear repulsion (Ziegler–Biersack–Littmark) for atoms closer than
      ${c.zbl_cutoff ?? 0.6} Å, so that the model never lets atoms collapse onto each other. It only appears when a pair is that close.</p>`;
    case 'energy': return `<p>The energy is the sum over atoms of the atomic energies, their electrostatic energies and any
      repulsion. The backward pass follows.</p>`;
    default: return '';
  }
}

function step(i: number, mod: Mod, c: Ctx, lastOp: number): { key: string; html: string } {
  const { ops, pass } = c, meta = c.meta as PhysNetMeta, cf = cfg(meta);
  const t = pass.trace, a = t.selected, op = ops[i], out = dims(t.shapes[lastOp]);
  const at = (k: number) => atomTag(pass, k);
  const S = (key: string, html: string) => ({ key, html });
  const cont = { key: 'cont', html: '' };
  const it = +(mod.path.match(/interaction\.(\d+)/)?.[1] ?? 0) + 1;
  switch (seg(mod)) {
    case 'geometry': return op.op === 'r_j' ? S('vec', `${t.shapes[i][0]} pairs: <b>r</b><sub>ij</sub> = <b>r</b><sub>j</sub> − <b>r</b><sub>i</sub>.`) : cont;
    case 'basis':
      if (op.op === 'pairs_in_cutoff') return S('mp', `${t.shapes[i][0]} of them are within ${cf.cutoff} Å; ${neighbours(pass, a).length} around ${at(a)}.`);
      if (op.op === 'angles') return S('cheb', `Chebyshev polynomials of 2e<sup>−r</sup> − 1, written as cos(k arccos(·)) ${dims([t.shapes[i][0], cf.num_basis_functions])}.`);
      if (op.op === 'basis') return S('cut', `Times the smooth cutoff: the basis <i>g</i>(<i>r</i><sub>ij</sub>) of every pair ${out}.`);
      return cont;
    case 'embedding': return S('emb', `Look up each atom's element ${out}.`);
    case 'message':
      if (op.op.startsWith('MessagePass')) return S('filter', `Interaction ${it}: a filter per pair, a linear map of its basis ${dims(t.shapes[i])}.`);
      if (op.op === 'x_j') return S('xj', `The neighbours' features, one row per pair.`);
      if (op.op === 'sum_messages') return S('sum', `Multiply filter, neighbour features and the learned weights, and sum onto each atom ${out}.`);
      return cont;
    case 'refinement':
      if (op.op === 'silu' && ops[i - 1]?.op === 'sum_messages') return S('ref', `${cf.n_refinement_blocks} residual blocks refine the result.`);
      return op.op === 'silu' && i === lastOp ? S('out', `Final layer and SiLU: the atoms' features after interaction ${it}.`) : cont;
    case 'energy_head':
      return op.op === 'per_atom' || op.op === 'energy_bias' ? S('e', `Atomic energies ${out}.`) : cont;
    case 'charges': {
      const q = pass.atomProps?.charge;
      return op.op === 'per_atom' || op.op === 'charge_bias' ? S('q', `Partial charges${q ? `: ${at(a)} gets <b>${q[a].toFixed(3)} e</b>, the structure ${q.reduce((s, x) => s + x, 0).toFixed(3)} e in total` : ''}.`) : cont;
    }
    case 'electrostatics':
      if (op.op === 'pairs') return S('pairs', `${t.shapes[i][0]} pairs within ${cf.electrostatics_off_end ?? 10} Å.`);
      if (op.op === 'damped' || op.op === '1/r') return S('r', `The shielded, damped 1/<i>r</i> of every pair.`);
      if (op.op === 'pair_energy') return S('pe', `<i>q</i><sub>i</sub><i>q</i><sub>j</sub> times that, half per direction (every pair appears twice).`);
      if (op.op === 'per_atom') return S('sum', `Summed onto each atom.`);
      return cont;
    case 'repulsion': return op.op === 'pair_energy' ? S('zbl', `Screened nuclear repulsion of the pairs closer than ${cf.zbl_cutoff ?? 0.6} Å.`) : cont;
    case 'energy': return op.op === 'total_energy' ? S('total', `<b>E = ${pass.energy.toFixed(4)} eV</b>.`) : cont;
    default: return S(op.op, `${op.op} ${dims(t.shapes[i])}`);
  }
}

export const physnetUI: ModelUI = {
  kind: 'physnet',
  name: 'PhysNet',
  family: 'message-passing network',
  terms: NN_TERMS,
  preferredStructure: 'acetone_dimer',
  describe,
  subtitle,
  narration,
  step,
  back: (m) => ({ energy: 'The gradient enters at the energy.', electrostatics: 'Through q_i q_j / r: to the positions directly and to the charges.',
    charges: 'Back through the charge head to the features.', energy_head: 'Back through the energy head to the features.',
    refinement: 'Back through the residual blocks.', message: 'Each message passes its gradient to the neighbour that sent it and to the pair\'s basis.',
    interaction: 'Back through the interaction.', basis: 'Through the radial basis to the pair distances.', geometry: 'Finally onto the atoms.' } as Record<string, string>)[seg(m)] ?? '',
  collapse: (m) => /interaction\.[1-9]/.test(m.path) && m.depth === 1,
  intro: (meta: PhysNetMeta) => `<h1>PhysNet, step by step</h1>
    <p>PhysNet passes messages between neighbouring atoms for ${meta.config.num_iterations} rounds, then predicts an energy and a
    partial charge for every atom; the charges add an explicit Coulomb energy. This model is mmml's <code>physnetjax</code>,
    in its invariant form, ${meta.name ? `<b>${meta.name}</b>` : ''}.</p>
    <p>Pick <b>colour: charge</b> in the structure view to see the predicted charges. <b>Click an atom</b> to follow it.</p>`,
  card: (meta: PhysNetMeta, nParams, label) => {
    const c = cfg(meta);
    return `<b>${label}</b> · ${(nParams / 1e3).toFixed(1)} k parameters<br>
      ${c.features} features · ${c.num_iterations} interactions · ${c.num_basis_functions} radial functions · cutoff ${c.cutoff} Å<br>
      ${c.charges ? 'charges + electrostatics · ' : ''}${c.zbl ? 'ZBL · ' : ''}elements ${meta.elements.join(', ')}<br>
      <span style="opacity:.7">mmml physnetjax (max_degree = 0)</span>`;
  },
};
