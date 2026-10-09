// Light and dark. The choice (system, light or dark) is saved and applied as <html data-theme>; the colours
// themselves are CSS custom properties (app/style.css), which the canvases and 3D views read from here and
// repaint with when the theme changes.
export type Mode = 'light' | 'dark';
export type Choice = Mode | 'system';
export type RGB = [number, number, number];

/** Also read by the inline script in each page's <head>, so the first paint is already in the right theme. */
const KEY = 'mlipviz-theme';
const root = document.documentElement;
const light = matchMedia('(prefers-color-scheme: light)');
const listeners = new Set<(m: Mode) => void>();

export function choice(): Choice {
  try {
    const c = localStorage.getItem(KEY);
    return c === 'light' || c === 'dark' ? c : 'system';
  } catch { return 'system'; }
}
export const mode = (): Mode => (root.dataset.theme === 'light' ? 'light' : 'dark');

function apply() {
  const c = choice(), m: Mode = c === 'system' ? (light.matches ? 'light' : 'dark') : c;
  if (root.dataset.theme === m) return;
  root.dataset.theme = m;
  listeners.forEach((f) => f(m));
}
light.addEventListener('change', apply);
apply();

/** Call `f` whenever the theme flips. */
export function onTheme(f: (m: Mode) => void) { listeners.add(f); }

export function setChoice(c: Choice) {
  try { if (c === 'system') localStorage.removeItem(KEY); else localStorage.setItem(KEY, c); } catch { /* not saved */ }
  apply();
}

/** A colour token, e.g. css('value') for --value. */
export const css = (name: string) => getComputedStyle(root).getPropertyValue(`--${name}`).trim();
/** A colour token as 0..1 RGB. The browser parses it: the built stylesheet is minified (#ffffff becomes #fff,
 *  and other forms may appear), so the token's text is not a fixed format. */
const probe = document.createElement('canvas').getContext('2d')!;
export function rgb(name: string): RGB {
  probe.fillStyle = '#000';
  probe.fillStyle = css(name) || '#000';
  const c = String(probe.fillStyle); // '#rrggbb', or 'rgba(r, g, b, a)' when translucent
  if (c.startsWith('#')) return [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16) / 255) as RGB;
  return (c.match(/[\d.]+/g) ?? ['0', '0', '0']).slice(0, 3).map((v) => +v / 255) as RGB;
}

/** A button that cycles system -> light -> dark. */
export function themeButton(b: HTMLButtonElement) {
  const icon = { system: '◐', light: '☀', dark: '☾' }, order: Choice[] = ['system', 'light', 'dark'];
  const show = () => {
    const c = choice();
    b.textContent = icon[c];
    b.title = `Theme: ${c === 'system' ? `as the system (${mode()})` : c}. Click for ${order[(order.indexOf(c) + 1) % 3]}`;
  };
  b.onclick = () => { setChoice(order[(order.indexOf(choice()) + 1) % 3]); show(); };
  onTheme(show);
  show();
}
