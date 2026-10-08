// SOAP (Bartók, Kondor & Csányi 2013): each atom's neighbours become Gaussians of
// width sigma, the density is expanded in radial functions R_n(r) times real
// spherical harmonics Y_lm, and the rotationally invariant power spectrum is
// p_{a a' l} = pi sqrt(8 / (2l+1)) sum_m c_{a l m} c_{a' l m}, a = (element, n).
//
// The expansion of one Gaussian at r_j is analytic in the angles:
//   c_nlm = fc(r_j) Y_lm(r_j / |r_j|) I_nl(|r_j|),
//   I_nl(r') = 4 pi int_0^rc R_n(r) r^2 exp(-(r^2 + r'^2) / (2 sigma^2)) i_l(r r' / sigma^2) dr,
// with i_l the modified spherical Bessel functions. I_nl is tabulated on the host
// (float64) and splined on the graph, as featomic and librascal do.

export interface SoapHypers {
  cutoff: number; // A
  cutoff_width: number; // cosine taper over the last cutoff_width A
  sigma: number; // Gaussian width, A
  n_max: number;
  l_max: number;
  zeta: number; // kernel exponent (1, 2 or 4)
}

export const defaultSoap: SoapHypers = { cutoff: 4.0, cutoff_width: 0.5, sigma: 0.3, n_max: 6, l_max: 4, zeta: 2 };

/** e^{-x} i_l(x) for l = 0..L (modified spherical Bessel functions of the first kind, scaled). */
export function scaledBesselI(x: number, L: number): Float64Array {
  const out = new Float64Array(L + 1);
  if (x < 1e-12) { out[0] = 1; return out; }
  if (x < 20) {
    // power series i_l(x) = x^l / (2l+1)!! sum_k (x^2/2)^k / (k! (2l+3)(2l+5)...(2l+2k+1))
    const ex = Math.exp(-x);
    let pre = 1; // x^l / (2l+1)!!
    for (let l = 0; l <= L; l++) {
      if (l > 0) pre *= x / (2 * l + 1);
      let term = 1, sum = 1;
      for (let k = 1; k < 200; k++) {
        term *= (x * x) / 2 / (k * (2 * l + 2 * k + 1));
        sum += term;
        if (term < 1e-17 * sum) break;
      }
      out[l] = ex * pre * sum;
    }
    return out;
  }
  // upward recurrence, stable for x > l
  const e2 = Math.exp(-2 * x);
  out[0] = (1 - e2) / (2 * x);
  if (L >= 1) out[1] = (1 + e2) / (2 * x) - (1 - e2) / (2 * x * x);
  for (let l = 1; l < L; l++) out[l + 1] = out[l - 1] - ((2 * l + 1) / x) * out[l];
  return out;
}

/** Orthonormal radial basis on [0, rc] (weight r^2): Gaussians on a grid, orthonormalised by
 *  Cholesky (Gram–Schmidt). Returns R(r) for all n at the quadrature points. */
function radialBasis(h: SoapHypers, r: Float64Array, w: Float64Array): Float64Array[] {
  const N = h.n_max, s = h.cutoff / N;
  const phi = Array.from({ length: N }, (_, n) => {
    const c = (h.cutoff * n) / (N - 1);
    return r.map((x) => Math.exp(-((x - c) ** 2) / (2 * s * s)));
  });
  const S = Array.from({ length: N }, (_, a) => Array.from({ length: N }, (_, b) =>
    phi[a].reduce((acc, v, q) => acc + v * phi[b][q] * r[q] * r[q] * w[q], 0)));
  // S = L L^T; R = L^{-1} phi
  const Lm = Array.from({ length: N }, () => new Float64Array(N));
  for (let i = 0; i < N; i++) for (let j = 0; j <= i; j++) {
    let v = S[i][j];
    for (let k = 0; k < j; k++) v -= Lm[i][k] * Lm[j][k];
    Lm[i][j] = i === j ? Math.sqrt(v) : v / Lm[j][j];
  }
  const R: Float64Array[] = [];
  for (let i = 0; i < N; i++) {
    const row = phi[i].slice();
    for (let k = 0; k < i; k++) for (let q = 0; q < row.length; q++) row[q] -= Lm[i][k] * R[k][q];
    R.push(row.map((v) => v / Lm[i][i]));
  }
  return R;
}

/** I_nl(r') and its slope at K knots r' = k h on [0, rc]: values and slopes [n_max (l_max+1), K],
 *  column c = n (l_max+1) + l. */
export function radialTables(h: SoapHypers, K = 160) {
  const Q = 600, rc = h.cutoff, dr = rc / Q;
  // Simpson's rule on [0, rc]
  const r = Float64Array.from({ length: Q + 1 }, (_, q) => q * dr);
  const w = Float64Array.from({ length: Q + 1 }, (_, q) => (dr / 3) * (q === 0 || q === Q ? 1 : q % 2 ? 4 : 2));
  const R = radialBasis(h, r, w);
  const L = h.l_max, C = h.n_max * (L + 1), s2 = h.sigma * h.sigma;
  const integral = (rp: number) => {
    const out = new Float64Array(C);
    for (let q = 0; q <= Q; q++) {
      const x = r[q], g = Math.exp(-((x - rp) ** 2) / (2 * s2)), bes = scaledBesselI((x * rp) / s2, L);
      const base = 4 * Math.PI * w[q] * x * x * g;
      for (let n = 0; n < h.n_max; n++) for (let l = 0; l <= L; l++) out[n * (L + 1) + l] += base * R[n][q] * bes[l];
    }
    return out;
  };
  const step = rc / (K - 1), eps = 1e-5;
  const V = new Float32Array(C * K), D = new Float32Array(C * K);
  for (let k = 0; k < K; k++) {
    const x = k * step, v = integral(x), vp = integral(x + eps), vm = integral(Math.max(x - eps, 0));
    for (let c = 0; c < C; c++) {
      V[c * K + k] = v[c];
      D[c * K + k] = (vp[c] - vm[c]) / (x + eps - Math.max(x - eps, 0));
    }
  }
  return { V, D, K, C, h: step };
}
