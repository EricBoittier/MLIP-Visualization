// The first frame of an XYZ / extended-XYZ file (Lattice="..." and pbc="T T T" are honoured).
import { Z } from '../common/elements';
import type { System } from '../common/structure';

export function parseXYZ(text: string): System {
  const lines = text.split(/\r?\n/);
  const n = parseInt(lines[0]);
  if (!Number.isFinite(n) || n <= 0) throw new Error('not an XYZ file: the first line should be the number of atoms');
  const comment = lines[1] ?? '';
  const numbers: number[] = [], positions: number[][] = [];
  for (let i = 0; i < n; i++) {
    const f = (lines[2 + i] ?? '').trim().split(/\s+/);
    const sym = f[0].replace(/[^A-Za-z]/g, '');
    const z = /^\d+$/.test(f[0]) ? +f[0] : Z(sym[0].toUpperCase() + sym.slice(1).toLowerCase());
    if (z <= 0) throw new Error(`unknown element "${f[0]}" on line ${3 + i}`);
    numbers.push(z);
    positions.push([+f[1], +f[2], +f[3]]);
  }
  const lat = comment.match(/Lattice\s*=\s*"([^"]+)"/i);
  if (!lat) return { numbers, positions };
  const v = lat[1].trim().split(/\s+/).map(Number);
  const pbcStr = comment.match(/pbc\s*=\s*"([^"]+)"/i)?.[1].trim().split(/\s+/);
  const pbc = pbcStr ? pbcStr.map((s) => /^(t|true|1)$/i.test(s)) : [true, true, true];
  return { numbers, positions, cell: [v.slice(0, 3), v.slice(3, 6), v.slice(6, 9)], pbc };
}
