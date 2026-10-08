// Physical units, as powers of base symbols: {Å: 2} is Å², {eV: 1, Å: -1} is eV/Å, {} a pure number.
// On a tensor, `undefined` is an untagged constant (it takes whatever unit an addition needs) and
// `null` is no physical unit at all (a learned feature, say).
export type Unit = Readonly<Record<string, number>>;
export type MaybeUnit = Unit | null | undefined;
export type Rule = (us: MaybeUnit[]) => MaybeUnit;

export const unit = (sym: string, e = 1): Unit => ({ [sym]: e });
export const Å = unit('Å'), eV = unit('eV'), e = unit('e'), Ha = unit('Ha');

const clean = (u: Record<string, number>): Unit => Object.fromEntries(Object.entries(u).filter(([, x]) => Math.abs(x) > 1e-9));
export const times = (a: Unit, b: Unit, k = 1): Unit => {
  const u: Record<string, number> = { ...a };
  for (const [s, x] of Object.entries(b)) u[s] = (u[s] ?? 0) + k * x;
  return clean(u);
};
export const power = (a: Unit, p: number): Unit => clean(Object.fromEntries(Object.entries(a).map(([s, x]) => [s, x * p])));
/** A pure number (radians count as one). */
export const pure = (u: MaybeUnit) => !!u && Object.keys(u).every((s) => s === 'rad');

const SUP: Record<string, string> = { '-': '⁻', '.': '·', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
const sup = (x: number) => (x === 1 ? '' : Number.isInteger(x) ? [...String(x)].map((c) => SUP[c]).join('') : `^${+x.toFixed(3)}`);
/** 'eV/Å', 'Å²', 'e²', '1/Å'; '' for a pure number or no unit. */
export function fmtUnit(u: MaybeUnit): string {
  if (!u) return '';
  const order = ['eV', 'Ha', 'e', 'Å', 'rad'], ks = Object.keys(u).sort((a, b) => order.indexOf(a) - order.indexOf(b));
  const num = ks.filter((s) => u[s] > 0).map((s) => s + sup(u[s])).join('·');
  const den = ks.filter((s) => u[s] < 0).map((s) => s + sup(-u[s])).join('·');
  return den ? `${num || '1'}/${den}` : num;
}

/** How an op's unit follows from its inputs'. */
export const rules = {
  /** addition, gathers, sums, reshapes: the inputs that carry a unit must agree */
  keep: ((us) => {
    const known = us.filter((u) => u !== undefined);
    if (!known.length) return undefined;
    if (known.some((u) => u === null)) return null;
    return known.every((u) => fmtUnit(u) === fmtUnit(known[0])) ? known[0] : null;
  }) as Rule,
  mul: ((us) => (us.every((u) => u) ? us.reduce((a, b) => times(a!, b!)) : null)) as Rule,
  div: (([a, b]) => (a && b ? times(a, b, -1) : null)) as Rule,
  pow: (p: number): Rule => ([a]) => (a ? power(a, p) : null),
  /** exp, cos, silu…: only of a pure number, and pure */
  fn: (([a]) => (pure(a) ? {} : null)) as Rule,
  /** a smooth step of a distance (cutoff, switch): pure, whatever the distance's unit */
  step: (([a]) => (a ? {} : null)) as Rule,
  acos: (([a]) => (pure(a) ? unit('rad') : null)) as Rule,
  none: (() => null) as Rule,
};

