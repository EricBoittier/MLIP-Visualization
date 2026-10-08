// Structures and the periodic neighbour list every model starts from.
export interface System {
  numbers: number[];
  positions: number[][]; // [N][3] Angstrom
  cell?: number[][]; // rows are lattice vectors
  pbc?: boolean[];
  charge?: number;
  spin?: number;
}

export interface NeighborList {
  center: Int32Array;
  neighbor: Int32Array;
  shift: Int32Array; // [E*3] integer cell shifts
  shiftVec: Float64Array; // [E*3] shift . cell
  dist: Float64Array;
}

const det3 = (m: number[][]) =>
  m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
  m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
  m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

function inv3(m: number[][]): number[][] {
  const d = det3(m);
  const c = (i: number, j: number) => {
    const r = [0, 1, 2].filter((x) => x !== i), s = [0, 1, 2].filter((x) => x !== j);
    return (m[r[0]][s[0]] * m[r[1]][s[1]] - m[r[0]][s[1]] * m[r[1]][s[0]]) * ((i + j) % 2 ? -1 : 1);
  };
  return [0, 1, 2].map((i) => [0, 1, 2].map((j) => c(j, i) / d));
}

const cross = (a: number[], b: number[]) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: number[]) => Math.hypot(a[0], a[1], a[2]);

export const isPeriodic = (s: System) => !!s.cell && !!s.pbc?.some(Boolean) && Math.abs(det3(s.cell)) > 1e-10;
export const volume = (s: System) => (isPeriodic(s) ? Math.abs(det3(s.cell!)) : Infinity);

/** Full neighbour list (every i -> j, with periodic images) within `cutoff`. */
export function neighborList(sys: System, cutoff: number): NeighborList {
  const N = sys.numbers.length;
  const per = isPeriodic(sys);
  const cell = per ? sys.cell! : [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const pbc = per ? sys.pbc! : [false, false, false];
  // wrap into the cell along periodic directions; shifts are corrected below
  const wrap = sys.positions.map(() => [0, 0, 0]);
  const pos = sys.positions.map((p) => p.slice());
  let reps = [0, 0, 0];
  if (per) {
    const inv = inv3(cell);
    sys.positions.forEach((p, i) => {
      for (let k = 0; k < 3; k++) {
        if (!pbc[k]) continue;
        const fk = p[0] * inv[0][k] + p[1] * inv[1][k] + p[2] * inv[2][k];
        wrap[i][k] = Math.floor(fk);
      }
      for (let c = 0; c < 3; c++) pos[i][c] -= wrap[i][0] * cell[0][c] + wrap[i][1] * cell[1][c] + wrap[i][2] * cell[2][c];
    });
    const V = Math.abs(det3(cell));
    reps = [0, 1, 2].map((k) => (pbc[k] ? Math.ceil(cutoff / (V / norm(cross(cell[(k + 1) % 3], cell[(k + 2) % 3])))) : 0));
  }
  const C: number[] = [], J: number[] = [], S: number[] = [], SV: number[] = [], Dd: number[] = [];
  const c2 = cutoff * cutoff;
  for (let i = 0; i < N; i++) {
    for (let a = -reps[0]; a <= reps[0]; a++)
      for (let b = -reps[1]; b <= reps[1]; b++)
        for (let c = -reps[2]; c <= reps[2]; c++) {
          const sv = [0, 1, 2].map((k) => a * cell[0][k] + b * cell[1][k] + c * cell[2][k]);
          for (let j = 0; j < N; j++) {
            if (i === j && a === 0 && b === 0 && c === 0) continue;
            const dx = pos[j][0] + sv[0] - pos[i][0], dy = pos[j][1] + sv[1] - pos[i][1], dz = pos[j][2] + sv[2] - pos[i][2];
            const r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= c2) continue;
            // shift in terms of the unwrapped positions
            const s = [a - wrap[j][0] + wrap[i][0], b - wrap[j][1] + wrap[i][1], c - wrap[j][2] + wrap[i][2]];
            C.push(i); J.push(j); S.push(...s);
            SV.push(...[0, 1, 2].map((k) => s[0] * cell[0][k] + s[1] * cell[1][k] + s[2] * cell[2][k]));
            Dd.push(Math.sqrt(r2));
          }
        }
  }
  return { center: Int32Array.from(C), neighbor: Int32Array.from(J), shift: Int32Array.from(S),
           shiftVec: Float64Array.from(SV), dist: Float64Array.from(Dd) };
}

