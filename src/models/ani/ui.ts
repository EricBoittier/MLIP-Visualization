// ANI-2x in the visualiser.
import type { Mod, ModDesc } from '../../viz/modules';
import type { ModelUI } from '../ui';
import { aniBack, aniStep } from './article';
import type { AniMeta } from './model';


function describe(seg: string): ModDesc | null {
  let m: RegExpMatchArray | null;
  if ((m = seg.match(/^member(\d+)$/))) return { type: 'mlp', title: `Ensemble member ${+m[1] + 1}`, short: `#${+m[1] + 1}` };
  const named: Record<string, [string, string, string?]> = {
    geometry: ['geometry', 'Atom pairs', 'Pairs'], radial: ['radial', 'Radial symmetry functions', 'Radial AEV'],
    angular: ['angular', 'Angular symmetry functions', 'Angular AEV'], aev: ['descriptor', 'Atomic environment vector', 'AEV'],
    networks: ['network', 'Atomic networks', 'Networks'], energy: ['energy', 'Energy'],
  };
  if (named[seg]) return { type: named[seg][0], title: named[seg][1], short: named[seg][2] };
  if (/^[A-Z][a-z]?$/.test(seg)) return { type: 'network', title: `${seg} network`, short: seg };
  return null;
}

const list = (xs: number[], d = 2) => xs.map((x) => x.toFixed(d)).join(', ');
const num = (x: number) => +x.toPrecision(4);

export const aniUI: ModelUI = {
  kind: 'ani',
  name: 'ANI-2x',
  family: 'Behler–Parrinello network',
  describe,
  collapse: (m: Mod) => m.type === 'mlp' && /member[1-9]/.test(m.path),
  subtitle: (m: Mod, meta: AniMeta) => {
    const S = meta.elements.length, NP = (S * (S + 1)) / 2, R = meta.radial, A = meta.angular;
    switch (m.type) {
      case 'geometry': return `r_ij within ${R.cutoff} Å`;
      case 'radial': return `${S} elements × ${R.shifts.length} shells`;
      case 'angular': return `${NP} pairs × ${A.shifts.length}×${A.sections.length}`;
      case 'descriptor': return `${S * R.shifts.length + NP * A.shifts.length * A.sections.length} features / atom`;
      case 'network': return m.path === 'networks' ? `${meta.members} members, one net per element` : meta.layers[m.title.split(' ')[0]]?.join('→') ?? '';
      case 'mlp': return 'CELU(α=0.1)';
      case 'energy': return 'mean + E_self(Z)';
      default: return '';
    }
  },
  narration: (m, meta: AniMeta) => {
    const R = meta.radial, A = meta.angular, S = meta.elements.length, NP = (S * (S + 1)) / 2;
    if (m === 'backward') return `
      <p>Forces are <i>F</i> = −∂<i>E</i>/∂<i>r</i>. The gradient runs back through every element network, splits
      into the radial and angular symmetry functions each atom's vector was built from, and reaches the positions
      through the pair distances, the angles and the cutoff functions.</p>
      <p>Because each symmetry function sums over neighbours, an atom's force collects contributions from every atom
      within ${R.cutoff} Å of it and of its neighbours.</p>`;
    switch (m.type) {
      case 'geometry': return `
        <p>A <b>Behler–Parrinello network</b> describes each atom by a fixed vector of symmetry functions of its
        neighbourhood, then predicts the atom's energy from that vector with a network specific to its element.</p>
        <p>ANI-2x starts from every pair within ${R.cutoff} Å (rows: the selected atom's pairs, columns: <i>x, y, z</i>).
        The chemiscope graph shows these pairs.</p>`;
      case 'radial': return `
        <p><b>Radial symmetry functions.</b> For each pair, ${R.shifts.length} Gaussians
        ¼ exp(−η(<i>r</i> − μ<sub>k</sub>)²) with η = ${num(R.eta)} Å⁻² and centres μ from ${R.shifts[0].toFixed(2)} to
        ${R.shifts[R.shifts.length - 1].toFixed(2)} Å, multiplied by a cosine cutoff that reaches zero at ${R.cutoff} Å.</p>
        <p>They are summed separately for each element of neighbour, so the selected atom's block is
        ${S} elements (rows: ${meta.symbols.join(', ')}) × ${R.shifts.length} shells.</p>`;
      case 'angular': return `
        <p><b>Angular symmetry functions</b> (ANI's modified Behler G4). For every pair of neighbours <i>j, k</i>
        within ${A.cutoff} Å of atom <i>i</i>, with angle θ<sub>ijk</sub>:</p>
        <p>2 ((1 + cos(θ − θ<sub>s</sub>)) / 2)<sup>ζ</sup> · exp(−η((<i>r</i><sub>ij</sub> + <i>r</i><sub>ik</sub>)/2 − μ)²) · f<sub>c</sub>(<i>r</i><sub>ij</sub>) f<sub>c</sub>(<i>r</i><sub>ik</sub>)</p>
        <p>with ζ = ${num(A.zeta)}, η = ${num(A.eta)}, ${A.sections.length} angle centres θ<sub>s</sub> (${list(A.sections)}) and
        ${A.shifts.length} distance centres. (θ is taken as acos(0.95 cos θ), which keeps the derivative finite.) The terms
        are summed per unordered pair of neighbour elements: ${NP} pairs × ${A.shifts.length * A.sections.length} features.</p>`;
      case 'descriptor': return `
        <p>The <b>atomic environment vector</b> concatenates the radial and angular parts:
        ${S * R.shifts.length} + ${NP * A.shifts.length * A.sections.length} = ${S * R.shifts.length + NP * A.shifts.length * A.sections.length}
        numbers per atom. It is invariant to rotations, translations and swapping atoms of the same element.</p>`;
      case 'network': return m.path === 'networks' ? `
        <p><b>Atomic networks.</b> Each element has its own feed-forward network
        (e.g. H: ${meta.layers.H?.join(' → ')}) with CELU activations (α = ${meta.activation.alpha}). The network maps an atom's
        environment vector to its contribution to the energy. ANI-2x is an ensemble of ${meta.members} independently
        trained models whose atomic energies are averaged; members 2–${meta.members} start collapsed.</p>` : `
        <p><b>${m.title}.</b> The ${m.title.split(' ')[0]} atoms' environment vectors go through
        ${meta.members} copies of the network (layer widths ${meta.layers[m.title.split(' ')[0]]?.join(' → ')}); the outputs
        are averaged and placed back at those atoms.</p>`;
      case 'mlp': return `<p><b>${m.title}</b>: linear layers with CELU(α = ${meta.activation.alpha}) in between, ending in one number per atom.</p>`;
      case 'energy': return `
        <p>The averaged atomic energies plus a fixed self-energy per element (in Hartree) are converted to eV and
        summed: <i>E</i> = Σ<sub>i</sub> (ε<sub>i</sub> + <i>E</i><sub>self</sub>(<i>Z</i><sub>i</sub>)). The backward pass follows.</p>`;
      default: return '';
    }
  },
  step: aniStep,
  back: aniBack,
  intro: (meta: AniMeta) => `<h1>ANI-2x, step by step</h1>
    <p>ANI-2x is a <i>Behler–Parrinello</i> network: each atom's neighbourhood is summarised by a fixed vector of
    symmetry functions (radial Gaussians and angular terms, sorted by neighbour element), and a small network per element
    turns that vector into the atom's energy. The total energy is the sum. ${meta.members} independently trained copies are averaged.</p>
    <p>Each block is an operation's output for the selected atom; in the structure panel, the pairs within ${meta.radial.cutoff} Å that
    enter its radial symmetry functions. <b>Click an atom</b> to follow it.</p>`,
  card: (meta: AniMeta, nParams, label) => `<b>${label}</b> · ${(nParams / 1e6).toFixed(2)} M parameters (${meta.members}-member ensemble)<br>
    elements ${meta.symbols.join(' ')} · ωB97X/6-31G(d)<br>radial cutoff ${meta.radial.cutoff} Å · angular cutoff ${meta.angular.cutoff} Å<br>
    <span style="opacity:.7">${meta.source ?? 'TorchANI'} · MIT licence</span>`,
};
