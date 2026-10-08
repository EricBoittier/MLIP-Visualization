import { expect, test } from 'vitest';
import { e, eV, fmtUnit, power, rules, times, Å } from '../src/engine/units';

test('unit algebra and formatting', () => {
  expect(fmtUnit(times(eV, Å, -1))).toBe('eV/Å');
  expect(fmtUnit(power(Å, 2))).toBe('Å²');
  expect(fmtUnit(power(Å, -1))).toBe('1/Å');
  expect(fmtUnit(times(times(e, e), power(Å, -1)))).toBe('e²/Å');
  expect(rules.keep([Å, undefined])).toEqual(Å);
  expect(rules.keep([Å, eV])).toBeNull();
  expect(rules.div([power(Å, 2), power(Å, 2)])).toEqual({});
  expect(rules.mul([Å, undefined])).toBeNull();
});
