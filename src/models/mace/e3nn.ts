// e3nn's real spherical harmonics (o3.spherical_harmonics with normalize=True and
// normalization='component', as MACE uses them), reached from the engine's orthonormal
// harmonics (the sph op) by a fixed change of basis.
import { realSph } from '../../engine/cpu';

/** e3nn's harmonics of a unit vector, l <= 3: e3nn/o3/_spherical_harmonics.py times sqrt(2l + 1). */
export function e3nnSph(x: number, y: number, z: number): number[] {
  const s3 = Math.sqrt(3), y2 = y * y, x2z2 = x * x + z * z;
  const l2 = [s3 * x * z, s3 * x * y, y2 - 0.5 * x2z2, s3 * y * z, (s3 / 2) * (z * z - x * x)];
  const l3 = [Math.sqrt(5 / 6) * (l2[0] * z + l2[4] * x), Math.sqrt(5) * l2[0] * y, Math.sqrt(3 / 8) * (4 * y2 - x2z2) * x,
              0.5 * y * (2 * y2 - 3 * x2z2), Math.sqrt(3 / 8) * z * (4 * y2 - x2z2), Math.sqrt(5) * l2[4] * y,
              Math.sqrt(5 / 6) * (l2[4] * z - l2[0] * x)];
  return [1, ...[x, y, z].map((v) => s3 * v), ...l2.map((v) => Math.sqrt(5) * v), ...l3.map((v) => Math.sqrt(7) * v)];
}

/** Solve A X = B for small dense A [n, n] and B [n, m] (Gaussian elimination, partial pivoting). */
function solve(A: number[][], B: number[][]): number[][] {
  const n = A.length, M = A.map((r, i) => [...r, ...B[i]]);
  for (let c = 0; c < n; c++) {
    const p = M.reduce((best, r, i) => (i >= c && Math.abs(r[c]) > Math.abs(M[best][c]) ? i : best), c);
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c] / M[c][c];
      for (let k = c; k < M[r].length; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((r, i) => r.slice(n).map((v) => v / M[i][i]));
}

/** W [16, 16] (as a linear layer's [out, in]) with e3nn(u) = W sph(u), block-diagonal in l. It is
 *  fitted by least squares at 64 directions; both bases span the same harmonics, so the fit is exact. */
export function e3nnBasis(lmax = 3): Float32Array {
  if (lmax > 3) throw new Error('e3nn harmonics are implemented up to l = 3');
  const M = (lmax + 1) ** 2, S = 64, W = new Float64Array(M * M);
  const dirs = Array.from({ length: S }, (_, k) => {
    const z = 1 - (2 * (k + 0.5)) / S, r = Math.sqrt(1 - z * z), ph = k * Math.PI * (3 - Math.sqrt(5));
    return [r * Math.cos(ph), r * Math.sin(ph), z];
  });
  const ours = dirs.map(([x, y, z]) => realSph(x, y, z, lmax, false).y), theirs = dirs.map(([x, y, z]) => e3nnSph(x, y, z));
  for (let l = 0; l <= lmax; l++) {
    const o = l * l, d = 2 * l + 1, ix = Array.from({ length: d }, (_, i) => o + i);
    const AtA = ix.map((i) => ix.map((j) => ours.reduce((s, y) => s + y[i] * y[j], 0)));
    const AtB = ix.map((i) => ix.map((j) => ours.reduce((s, y, k) => s + y[i] * theirs[k][j], 0)));
    const X = solve(AtA, AtB); // e3nn_j = sum_i ours_i X[i][j]
    const err = Math.max(...ours.flatMap((y, k) => ix.map((j, jj) => Math.abs(ix.reduce((s, i, ii) => s + y[i] * X[ii][jj], 0) - theirs[k][j]))));
    if (err > 1e-9) throw new Error(`e3nn harmonics: l = ${l} is not a change of basis (residual ${err})`);
    ix.forEach((i, ii) => ix.forEach((j, jj) => { W[j * M + i] = X[ii][jj]; }));
  }
  return Float32Array.from(W);
}
