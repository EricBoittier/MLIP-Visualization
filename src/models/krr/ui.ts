// KRR / SOAP in the visualiser.
import { SYMBOLS } from '../../common/elements';
import { atomTag, type Ctx, dims, fmt, neighbours } from '../../viz/article';
import type { Mod, ModDesc } from '../../viz/modules';
import type { ModelUI } from '../ui';
import type { FitReport, KrrMeta } from './model';
import { defaultSoap, type SoapHypers } from './soap';

type Meta = KrrMeta & { report?: FitReport; elements?: number[] };
const hyp = (m: Meta): SoapHypers => ({ ...defaultSoap, ...m.soap });
const meV = (x: number) => (1000 * x).toFixed(1);

function describe(seg: string): ModDesc | null {
  const named: Record<string, [string, string, string?]> = {
    fit: ['regression', 'The fit (replayed)', 'Fit'], soap: ['descriptor', 'SOAP power spectrum', 'SOAP'],
    geometry: ['geometry', 'Neighbours', 'Pairs'], radial: ['radial', 'Radial integrals', 'Radial'],
    angular: ['angular', 'Spherical harmonics', 'Y_lm'], density: ['density', 'Density coefficients', 'c_nlm'],
    power_spectrum: ['spectrum', 'Power spectrum', 'p_nn′l'], kernel: ['kernel', 'Kernel', 'Kernel'],
    regression: ['readout', 'Kernel regression', 'Σ α k'], energy: ['energy', 'Energy'],
  };
  const d = named[seg];
  return d ? { type: d[0], title: d[1], short: d[2] } : null;
}

function subtitle(m: Mod, meta: Meta): string {
  const h = hyp(meta), r = meta.report;
  switch (m.path.split('/').pop()) {
    case 'fit': return r ? `${r.nTrain} structures, ${r.nSparse} sparse points` : '';
    case 'soap': return `n ≤ ${h.n_max}, l ≤ ${h.l_max}, σ = ${h.sigma} Å`;
    case 'geometry': return `r_ij within ${h.cutoff} Å`;
    case 'radial': return `I_nl(r) · f_c(r)`;
    case 'angular': return `Y_lm(r̂), l ≤ ${h.l_max}`;
    case 'density': return 'Σ_j per element';
    case 'power_spectrum': return 'Σ_m c c, normalised';
    case 'kernel': return `(p · x_t)^${h.zeta}, same element`;
    case 'regression': return 'ε_i = b + Σ_t α_t k_it';
    case 'energy': return 'E = Σ_i ε_i';
    default: return '';
  }
}

function narration(m: Mod | 'backward', meta: Meta): string {
  const h = hyp(meta), r = meta.report;
  if (m === 'backward') return `
    <p>Forces are −∂<i>E</i>/∂<b>r</b>. The fit is finished, so the gradient does not go through it. It runs from the energy
    through the weights α to the kernel values, through the normalised power spectrum and the density coefficients to the
    radial integrals (by the slope of their spline) and the spherical harmonics, and from there to the pair vectors.</p>
    <p>Kernel models give smooth, exactly conservative forces this way, even though this one was fitted to energies only.</p>`;
  switch (m.path.split('/').pop()) {
    case 'fit': return `
      <p><b>The fit.</b> Kernel ridge regression has no pre-trained weights: here the training set was made in your browser.
      ${r ? `${r.nTrain + r.nTest} rattled copies of the structure were labelled by <b>${r.teacher}</b>; ${r.nSparse} representative
      environments (<i>sparse points</i>) were chosen by farthest-point sampling,` : ''} and the weights α solve the regularised
      least-squares problem (<b>K</b><sub>NM</sub><sup>T</sup><b>K</b><sub>NM</sub> + λ<b>K</b><sub>MM</sub>) α = <b>K</b><sub>NM</sub><sup>T</sup> <b>y</b>.</p>
      <p>This section replays that solve: the kernel between the sparse points, between the training structures and the sparse
      points, the normal matrix, and its solution.${r ? ` λ was chosen on ${r.nTest} held-out structures; the fit reaches
      ${meV(r.rmseTrain)} meV/atom on the training set and ${meV(r.rmseTest)} meV/atom on the held-out set.` : ''}</p>`;
    case 'soap': return `<p><b>SOAP</b> (smooth overlap of atomic positions) describes each atom's neighbourhood as a smoothed
      density of its neighbours, separately for each element, and keeps the rotation-invariant part of that density.</p>`;
    case 'geometry': return `<p>Every pair within ${h.cutoff} Å, including periodic images. Each neighbour will contribute a
      Gaussian of width σ = ${h.sigma} Å to the centre atom's density.</p>`;
    case 'radial': return `<p><b>Radial integrals.</b> The density is expanded in ${h.n_max} orthonormal radial functions
      <i>R</i><sub>n</sub>(<i>r</i>) times spherical harmonics. For a Gaussian at distance <i>r</i>, the radial part of the overlap is
      <i>I</i><sub>nl</sub>(<i>r</i>) = 4π∫<i>R</i><sub>n</sub>(<i>r</i>′) e<sup>−(<i>r</i>′² + <i>r</i>²)/2σ²</sup> <i>i</i><sub>l</sub>(<i>r r</i>′/σ²) <i>r</i>′² d<i>r</i>′,
      with <i>i</i><sub>l</sub> a modified spherical Bessel function. It is tabulated once and interpolated with a spline, then
      multiplied by a cosine cutoff that vanishes at ${h.cutoff} Å.</p>`;
    case 'angular': return `<p><b>Spherical harmonics</b> Y<sub>lm</sub> up to l = ${h.l_max} of each pair's direction:
      ${(h.l_max + 1) ** 2} numbers per pair.</p>`;
    case 'density': return `<p><b>Density coefficients</b> <i>c</i><sub>Znlm</sub> = Σ<sub>j∈Z</sub> f<sub>c</sub> <i>I</i><sub>nl</sub>(<i>r</i><sub>ij</sub>) Y<sub>lm</sub>(r̂<sub>ij</sub>):
      the neighbours' contributions summed per element. They change when the molecule rotates.</p>`;
    case 'power_spectrum': return `<p><b>Power spectrum.</b> Summing products over m,
      p<sub>(Zn)(Z′n′)l</sub> = π√(8/(2l+1)) Σ<sub>m</sub> <i>c</i><sub>Znlm</sub> <i>c</i><sub>Z′n′lm</sub>, removes the
      orientation: these numbers are invariant to rotations. They are normalised to unit length, so the kernel compares shapes of
      environments.</p>`;
    case 'kernel': return `<p><b>Kernel.</b> Each atom is compared with every sparse point of its own element:
      <i>k</i> = (<b>p</b><sub>i</sub> · <b>x</b><sub>t</sub>)<sup>${h.zeta}</sup>, 1 for identical environments.</p>`;
    case 'regression': return `<p><b>Regression.</b> The atom's energy is a weighted sum of its kernel values,
      ε<sub>i</sub> = <i>b</i> + Σ<sub>t</sub> α<sub>t</sub> <i>k</i>(<b>p</b><sub>i</sub>, <b>x</b><sub>t</sub>), with the weights α from the fit
      and a constant baseline <i>b</i> (the mean energy per atom of the training set).</p>`;
    case 'energy': return `<p>The total energy is the sum of the atomic energies. The backward pass follows.</p>`;
    default: return '';
  }
}

function step(i: number, mod: Mod, c: Ctx, lastOp: number): { key: string; html: string } {
  const { ops, pass } = c, meta = c.meta as Meta, h = hyp(meta), r = meta.report;
  const t = pass.trace, a = t.selected, op = ops[i], out = dims(t.shapes[lastOp]);
  const at = (k: number) => atomTag(pass, k);
  const S = (key: string, html: string) => ({ key, html });
  const cont = { key: 'cont', html: '' };
  const near = neighbours(pass, a);
  const seg = mod.path.split('/').pop();
  switch (seg) {
    case 'fit':
      if (op.op === 'sparse_overlap') return S('kmm', `Overlap of the ${t.shapes[i][0]} sparse environments with each other, raised to the power ${h.zeta} and
        masked to equal elements: <b>K</b><sub>MM</sub> ${dims(t.shapes[i])}.`);
      if (op.op === 'overlap') return S('knm', `The same kernel between every atom of the training set and every sparse point.`);
      if (op.op === 'K_NM') return S('knm2', `Summed over the atoms of each training structure (energies are sums over atoms):
        <b>K</b><sub>NM</sub> ${dims(t.shapes[i])}.`);
      if (op.op === 'normal_matrix') return S('normal', `The normal matrix <b>K</b><sub>NM</sub><sup>T</sup><b>K</b><sub>NM</sub> + λ<b>K</b><sub>MM</sub>${r ? `, λ = ${fmt(r.lambda)}` : ''}.`);
      if (op.op === 'K_NM^T y') return S('rhs', `The right-hand side <b>K</b><sub>NM</sub><sup>T</sup> <b>y</b>: the energies minus the baseline.`);
      if (op.op === 'cholesky_solve') return S('solve', `Solve for α (a Cholesky factorisation, in double precision): one weight per sparse
        environment ${dims(t.shapes[i])}. This finishes the fit; everything after this is prediction.`);
      return cont;
    case 'geometry':
      if (op.op === 'r_j') return S('vec', `Pairs within ${h.cutoff} Å: ${t.shapes[i][0]} in the structure, ${near.length} around ${at(a)}.`);
      return cont;
    case 'radial':
      if (op.op === 'radial_integrals') return S('ri', `Look up <i>I</i><sub>nl</sub>(<i>r</i>) for each of ${at(a)}'s pairs:
        ${h.n_max} × ${h.l_max + 1} values per pair ${dims(t.shapes[i])}, then times the cutoff.`);
      return cont;
    case 'angular':
      if (op.op === 'spherical_harmonics') return S('ylm', `Y<sub>lm</sub> of each pair's direction ${dims(t.shapes[i])}.`);
      return cont;
    case 'density':
      if (op.op === 'per_lm') return S('outer', `Multiply radial and angular parts: one coefficient per (n, l, m) and pair.`);
      if (op.op === 'sum_per_element') {
        const count = new Map<number, number>();
        for (const x of near) count.set(pass.numbers[x.j], (count.get(pass.numbers[x.j]) ?? 0) + 1);
        return S('sum', `Sum over neighbours of each element: ${at(a)} has ${[...count].map(([z, n]) => `${n} ${SYMBOLS[z]}`).join(', ')} within ${h.cutoff} Å.`);
      }
      if (op.op === 'coefficients') return S('coef', `Arrange as one row per (element, n) and one column per (l, m) ${out}.`);
      return cont;
    case 'power_spectrum':
      if (op.op === 'power_spectrum') return S('ps', `Contract over m for every pair of rows and every l: ${t.shapes[i][1]} invariant numbers per atom ${dims(t.shapes[i])}.`);
      if (op.op === 'normalise') return S('norm', `Normalise each atom's power spectrum to unit length.`);
      return cont;
    case 'kernel':
      if (op.op === 'overlap') return S('k', `Overlap of every atom with every sparse point, to the power ${h.zeta}, and zero between different elements ${out}.`);
      return cont;
    case 'regression':
      if (op.op === 'weighted_sum') return S('sum', `Weight the kernel values by α and sum: each atom's energy relative to the baseline.`);
      if (op.op === 'atomic_energy') return S('base', `Add the baseline: ${at(a)} contributes ε = <b>${pass.energies[a].toFixed(4)} eV</b>.`);
      return cont;
    case 'energy':
      return op.op === 'total_energy' ? S('total', `Sum over atoms: <b>E = ${pass.energy.toFixed(4)} eV</b>.`) : cont;
    default:
      return S(op.op, `${op.op} ${dims(t.shapes[i])}`);
  }
}

function back(m: Mod, c: Ctx): string {
  const a = c.pass.trace.selected, F = c.pass.forces;
  switch (m.path.split('/').pop()) {
    case 'energy': case 'regression': return `Each atom's energy hands its gradient to its kernel values, weighted by α.`;
    case 'kernel': return `Through the power ${hyp(c.meta).zeta} and the overlap, to the normalised power spectrum.`;
    case 'power_spectrum': return `Through the normalisation and the products over m, to the density coefficients.`;
    case 'density': return `Each coefficient's gradient goes back to the pairs it was summed from.`;
    case 'angular': return `Through the spherical harmonics to each pair's direction.`;
    case 'radial': return `Through the spline of the radial integrals and the cutoff to each pair's length.`;
    case 'geometry': return `Finally onto the two atoms of each pair. ${atomTag(c.pass, a)} feels |<b>F</b>| =
      <b>${F ? Math.hypot(F[3 * a], F[3 * a + 1], F[3 * a + 2]).toFixed(3) : '?'} eV/Å</b>.`;
    default: return '';
  }
}

export const krrUI: ModelUI = {
  kind: 'krr',
  name: 'KRR / SOAP',
  family: 'kernel ridge regression',
  describe,
  subtitle,
  narration,
  step,
  back,
  collapse: (m) => m.path === 'fit',
  intro: (meta: Meta) => `<h1>KRR with SOAP, step by step</h1>
    <p>A kernel model does not learn features: it describes every atom with a fixed, rotation-invariant fingerprint of its
    neighbourhood (the SOAP power spectrum) and predicts its energy as a weighted sum of similarities to environments it has seen.</p>
    <p>${meta.report ? `This one was fitted a moment ago, in your browser, to ${meta.report.teacher}'s energies of
    ${meta.report.nTrain} rattled copies of this structure.` : 'It is fitted in your browser to another model\'s energies.'}
    Changing the structure fits it again; <b>Rattle</b> asks it to predict a new geometry. <b>Click an atom</b> to follow it.</p>`,
  card: (meta: Meta, nParams, label) => {
    const h = hyp(meta), r = meta.report;
    return `<b>${label}</b> · ${(nParams / 1e3).toFixed(1)} k numbers (sparse points + weights)<br>
      SOAP: cutoff ${h.cutoff} Å, σ ${h.sigma} Å, n ≤ ${h.n_max}, l ≤ ${h.l_max}; kernel (p·x)<sup>${h.zeta}</sup><br>
      ${r ? `fitted to <b>${r.teacher}</b>: ${r.nTrain} + ${r.nTest} structures, ${r.nSparse} sparse points, ${(r.ms / 1000).toFixed(1)} s<br>
      RMSE ${meV(r.rmseTrain)} (train) / ${meV(r.rmseTest)} (held-out) meV/atom` : ''}<br>
      elements ${(meta.elements ?? []).map((z) => SYMBOLS[z]).join(' ')}`;
  },
};
