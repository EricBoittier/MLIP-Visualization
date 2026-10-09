import { expect, it } from 'vitest';
import { neighborList } from '../src/common/structure';

let seed = 11;
const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);

it('finds the same pairs, in the same order, as checking every pair', () => {
  // clusters of 9 atoms on a loose grid, as a DMC batch lays them out, plus a dense blob
  const pos: number[][] = [];
  for (let b = 0; b < 40; b++) for (let i = 0; i < 9; i++) pos.push([(b % 4) * 11 + 3 * rand(), (Math.floor(b / 4) % 4) * 11 + 3 * rand(), Math.floor(b / 16) * 11 - 3 * rand()]);
  for (let i = 0; i < 60; i++) pos.push([60 + 8 * rand(), 8 * rand(), 8 * rand()]);
  const cutoff = 5.2, nl = neighborList({ numbers: pos.map(() => 6), positions: pos }, cutoff);
  const C: number[] = [], J: number[] = [], D: number[] = [];
  for (let i = 0; i < pos.length; i++)
    for (let j = 0; j < pos.length; j++) {
      const r = Math.hypot(pos[j][0] - pos[i][0], pos[j][1] - pos[i][1], pos[j][2] - pos[i][2]);
      if (i !== j && r < cutoff) { C.push(i); J.push(j); D.push(r); }
    }
  expect([...nl.center]).toEqual(C);
  expect([...nl.neighbor]).toEqual(J);
  nl.dist.forEach((d, e) => expect(d).toBeCloseTo(D[e], 12));
  expect(nl.shift.every((s) => s === 0)).toBe(true);
});
