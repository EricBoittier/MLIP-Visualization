// ANI-2x's walkthrough paragraphs: one per step of the pass, from the live trace.
import { SYMBOLS } from '../../common/elements';
import { atomTag, type Ctx, dims, fmt, neighbours } from '../../viz/article';
import type { Mod } from '../../viz/modules';
import type { AniMeta } from './model';

const num = (x: number) => +x.toPrecision(4);

export function aniStep(i: number, mod: Mod, c: Ctx, lastOp = i): { key: string; html: string } {
  const { ops, pass } = c, m: AniMeta = c.meta;
  const t = pass.trace, a = t.selected, op = ops[i], out = dims(t.shapes[lastOp]);
  const at = (k: number) => atomTag(pass, k);
  const S = (key: string, html: string) => ({ key, html });
  const cont = { key: 'cont', html: '' };
  const R = m.radial, A = m.angular;
  const near = neighbours(pass, a);
  const angular = near.filter((x) => x.r < A.cutoff);
  switch (mod.type) {
    case 'geometry':
      if (op.op === 'r_j') return S('vec', `For every pair (<i>i</i>, <i>j</i>) closer than ${R.cutoff} Å, including periodic images, subtract
        the positions: <b>r</b><sub>ij</sub> = <b>r</b><sub>j</sub> − <b>r</b><sub>i</sub> (+ cell shift). There are ${t.shapes[i][0]} such vectors;
        the block shows the ${near.length} that start at ${at(a)}.`);
      if (op.op === 'sqrt') return S('len', `Their lengths: ${at(a)}'s neighbours sit between ${fmt(Math.min(...near.map((x) => x.r)), 2)} and
        ${fmt(Math.max(...near.map((x) => x.r)), 2)} Å.`);
      return cont;
    case 'radial':
      if (op.op === 'cutoff_cosine') return S('fc', `The cosine cutoff <i>f</i><sub>c</sub>(<i>r</i>) = ½ cos(π<i>r</i>/${R.cutoff}) + ½ of every pair:
        1 at short range, 0 at ${R.cutoff} Å.`);
      if (op.op === 'distances') return S('gauss', `Compare each distance with ${R.shifts.length} shells: ¼ exp(−${num(R.eta)} (<i>r</i> − μ<sub>k</sub>)²) · <i>f</i><sub>c</sub>(<i>r</i>),
        one column per shell ${dims([t.shapes[i][0], R.shifts.length])}.`);
      if (op.op === 'sum_per_element') {
        const count = new Map<number, number>();
        for (const x of near) count.set(pass.numbers[x.j], (count.get(pass.numbers[x.j]) ?? 0) + 1);
        const list = [...count].map(([z, n]) => `${n} ${SYMBOLS[z]}`).join(', ');
        return S('sum', `Sum the shells over neighbours of the same element. ${at(a)} has ${list} within ${R.cutoff} Å, so its block has
          ${m.symbols.length} rows (${m.symbols.join(', ')}), the empty elements staying zero.`);
      }
      if (op.op === 'radial_aev') return S('flat', `Lay the ${m.symbols.length} × ${R.shifts.length} block out as one row per atom: the radial part of each atom's vector ${out}.`);
      return cont;
    case 'angular':
      if (op.op === 'r_ij') {
        const n = angular.length;
        return S('tri', `Pair up the neighbours closer than ${A.cutoff} Å. ${at(a)} has ${n} of them, so ${(n * (n - 1)) / 2} pairs (<i>j</i>, <i>k</i>);
          there are ${t.shapes[i][0]} pairs in the whole structure.`);
      }
      if (op.op === 'dot') return S('cos', `cos θ<sub>ijk</sub> = <b>r</b><sub>ij</sub> · <b>r</b><sub>ik</sub> / (<i>r</i><sub>ij</sub> <i>r</i><sub>ik</sub>).`);
      if (op.op === 'acos') return S('theta', `θ = acos(0.95 cos θ<sub>ijk</sub>). The factor 0.95 keeps the derivative of acos finite at 0° and 180°.`);
      if (op.op === 'angles') return S('ang', `Angular part for ${A.sections.length} angle centres θ<sub>s</sub>: 2 ((1 + cos(θ − θ<sub>s</sub>)) / 2)<sup>${num(A.zeta)}</sup>,
        sharply peaked around each centre.`);
      if (op.op === 'mean_distances') return S('rad', `Radial part for ${A.shifts.length} shells: exp(−${num(A.eta)} ((<i>r</i><sub>ij</sub> + <i>r</i><sub>ik</sub>)/2 − μ)²).`);
      if (op.op === 'expand_radial') return S('outer', `Multiply every radial shell by every angle centre: ${A.shifts.length} × ${A.sections.length} = ${A.shifts.length * A.sections.length}
        features per pair of neighbours, then by both cutoffs <i>f</i><sub>c</sub>(<i>r</i><sub>ij</sub>) <i>f</i><sub>c</sub>(<i>r</i><sub>ik</sub>).`);
      if (op.op === 'sum_per_element_pair') {
        const S2 = m.symbols.length, NP = (S2 * (S2 + 1)) / 2;
        return S('sum', `Sum over the pairs of neighbours by the elements of the pair (H–H, H–C, …): ${NP} element pairs × ${A.shifts.length * A.sections.length} features.
          The block is ${at(a)}'s; only the rows of element pairs that occur around it are non-zero.`);
      }
      if (op.op === 'angular_aev') return S('flat', `One row per atom: the angular part of each atom's vector ${out}.`);
      return cont;
    case 'descriptor':
      return S('aev', `Concatenate both parts into the atomic environment vector ${out}. This vector is all the network ever sees of ${at(a)}'s
        surroundings.`);
    case 'network':
      if (op.op.endsWith('_atoms')) {
        const sym = op.op.split('_')[0], n = pass.numbers.filter((z) => SYMBOLS[z] === sym).length;
        return S('pick', `Take the ${n} ${sym} atom${n > 1 ? 's' : ''}' vectors: every element has its own network (${m.layers[sym]?.join(' → ')}).`);
      }
      if (op.op === 'ensemble_mean') return S('mean', `Average the ${m.members} members' outputs: one atomic energy (in Hartree) per atom.`);
      return cont;
    case 'mlp': {
      const k = +(mod.path.match(/member(\d+)/)?.[1] ?? 0), sym = mod.parent?.title.split(' ')[0] ?? '';
      if (k > 0) return S('member', `Member ${k + 1}: the same layer widths with independently trained weights.`);
      if (op.op === 'layer0') return S('l0', `First layer: ${m.layers[sym]?.[0]} → ${m.layers[sym]?.[1]}, then CELU(α = ${m.activation.alpha}),
        which is linear for positive inputs and a flattened exponential below zero ${out}.`);
      if (op.op === 'layer1' || op.op === 'layer2') return S(op.op, `Hidden layer ${op.op === 'layer1' ? 2 : 3}, then CELU ${out}.`);
      if (op.op === 'output') return S('out', `Output layer: one number per atom, its energy contribution from member 1.`);
      return cont;
    }
    case 'energy':
      if (op.op === 'to_eV') return S('sae', `Add each element's self energy and convert from Hartree to eV (× ${num(m.hartree_to_ev)}):
        ${at(a)} contributes ε = <b>${pass.energies[a].toFixed(4)} eV</b>.`);
      if (op.op === 'total_energy') return S('total', `Sum over atoms: <b>E = ${pass.energy.toFixed(4)} eV</b>.`);
      return cont;
    default:
      return S(op.op, `${op.op} ${dims(t.shapes[i])}`);
  }
}

export function aniBack(m: Mod, c: Ctx): string {
  const a = c.pass.trace.selected, F = c.pass.forces;
  switch (m.type) {
    case 'energy': return `The gradient enters at the top: ∂E/∂ε<sub>i</sub> = 1 eV per eV for every atom.`;
    case 'network': return m.path === 'networks' ? `Each element's atoms receive the gradient of their atomic energy.`
      : `The ensemble mean hands each member 1/${c.meta.members} of the gradient.`;
    case 'mlp': return `Back through the member's layers: transposed weights times the incoming gradient, times the CELU slope, down to ∂E/∂(AEV).`;
    case 'descriptor': return `The gradient on the vector splits into its radial and angular parts.`;
    case 'angular': return `Through the angular terms to the pair distances and the angles, and through acos and the dot product to both edge vectors of every triplet.`;
    case 'radial': return `Through the Gaussian shells and the cosine cutoff to every pair distance.`;
    case 'geometry': return `Finally ∂E/∂<b>r</b><sub>ij</sub> is scattered onto the two atoms of each pair. The negative is the force: ${atomTag(c.pass, a)}
      feels |<b>F</b>| = <b>${F ? Math.hypot(F[3 * a], F[3 * a + 1], F[3 * a + 2]).toFixed(3) : '?'} eV/Å</b>.`;
    default: return `Gradient through ${m.title.toLowerCase()}.`;
  }
}
