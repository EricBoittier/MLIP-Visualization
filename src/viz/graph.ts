// chemiscope as both the structure viewer and the GNN's graph: atoms are nodes,
// PET's neighbour list is drawn as edges (each directed edge i->j as the half
// from i towards j, so both directions stay visible), and both are coloured by
// whatever op the walkthrough is on.
import bootstrapCss from 'bootstrap/dist/css/bootstrap.min.css?inline';
import { MoleculeViewer } from 'chemiscope';
import chemiscopeCss from 'chemiscope/src/static/chemiscope.css?inline';
import { SYMBOLS } from '../common/elements';
import type { AttentionMap, OpInfo, Pass } from '../worker/protocol';
import { magnitudeColor } from './colors';

export interface GraphState {
  op: number; // active op, -1 for none
  dir: 'fwd' | 'bwd';
  forces: boolean;
  attention: AttentionMap | null;
  hover: { atom?: number; edge?: number } | null;
}

export class GraphView {
  readonly viewer: MoleculeViewer;
  private pass: Pass | null = null;
  private ops: OpInfo[] = [];
  private lastKey = '';
  private timer = 0;
  private pending: GraphState | null = null;
  onSelect: ((atom: number) => void) | null = null;

  constructor(el: HTMLElement) {
    // chemiscope's viewer and its settings dialog take their styles (Bootstrap and
    // chemiscope's own) from the shadow root they are created in
    const sheet = (css: string) => { const s = new CSSStyleSheet(); s.replaceSync(css); return s; };
    const root = el.attachShadow({ mode: 'open' });
    root.adoptedStyleSheets = [sheet(bootstrapCss), sheet(chemiscopeCss)];
    const inner = document.createElement('div');
    inner.style.cssText = 'position:absolute;inset:0';
    root.appendChild(inner);
    this.viewer = new MoleculeViewer(inner, ['node', 'energy', 'force']);
    this.viewer.onselect = (atom) => this.onSelect?.(atom);
  }

  setPass(pass: Pass, ops: OpInfo[]) {
    this.pass = pass;
    this.ops = ops;
    this.lastKey = '';
  }

  /** Redraw (throttled: chemiscope reloads the whole model). */
  render(s: GraphState) {
    this.pending = s;
    if (this.timer) return;
    this.timer = window.setTimeout(() => {
      this.timer = 0;
      if (this.pending) this.draw(this.pending);
    }, 60);
  }

  private draw(s: GraphState) {
    const p = this.pass;
    if (!p) return;
    const key = JSON.stringify([p.id, s.op, s.dir, s.forces, s.hover, s.attention?.op]);
    if (key === this.lastKey) return;
    this.lastKey = key;
    const { rows, selected, graph: gr } = p.trace;
    const N = p.numbers.length, pos = p.positions, E = gr.center.length;
    const op = s.op >= 0 ? this.ops[s.op] : null;
    const norms = op ? (s.dir === 'bwd' ? p.trace.gradNorms[s.op] ?? p.trace.norms[s.op] : p.trace.norms[s.op]) : null;
    const which = s.dir === 'bwd' && p.trace.gradNorms[s.op] ? 'grad' : 'value';

    // node and edge magnitudes for this op, through its rows' meaning
    let node: number[] | null = null, edge: (number | null)[] | null = null;
    const space = op?.kind ? rows[op.kind] : undefined;
    if (norms && space) {
      if (space.atom || space.edge) {
        const nd: number[] = new Array(N).fill(0), ed: (number | null)[] = new Array(E).fill(null);
        let anyNode = false, anyEdge = false;
        space.atom?.forEach((i, r) => { if (i >= 0) { nd[i] = Math.max(nd[i], norms[r]); anyNode = true; } });
        space.edge?.forEach((e, r) => { if (e >= 0) { ed[e] = norms[r]; anyEdge = true; } });
        if (anyNode) node = nd;
        if (anyEdge) edge = ed;
      } else {
        // rows with no node or edge of their own (e.g. angular triplets): mean over each owner atom
        const sum = new Array(N).fill(0), cnt = new Array(N).fill(0);
        space.owner.forEach((i, r) => { if (i >= 0) { sum[i] += norms[r]; cnt[i]++; } });
        node = sum.map((x, i) => (cnt[i] ? x / cnt[i] : 0));
      }
    }
    // attention: how much the selected atom's own token attends to each neighbour (mean over heads)
    let attn: Map<number, number> | null = null;
    if (s.attention && op?.op === 'attention') {
      const a = s.attention, n = a.n;
      const mine = Array.from(gr.center.keys()).filter((e) => gr.center[e] === selected);
      attn = new Map();
      for (let t = 1; t < n; t++) {
        let w = 0;
        for (let h = 0; h < a.heads; h++) w += a.probs[h * n * n + t];
        if (mine[t - 1] !== undefined) attn.set(mine[t - 1], w / a.heads);
      }
    }

    const emax = edge ? Math.max(...edge.map((x) => x ?? 0), 1e-12) : 1;
    const amax = attn ? Math.max(...attn.values(), 1e-12) : 1;
    const bases: [number, number, number][] = [], vectors: [number, number, number][] = [];
    const radii: number[] = [], colors: string[] = [];
    for (let e = 0; e < E; e++) {
      const i = gr.center[e], j = gr.neighbor[e];
      const v = [0, 1, 2].map((k) => 0.5 * (pos[j][k] + gr.shift[3 * e + k] - pos[i][k])) as [number, number, number];
      const mine = i === selected;
      const hot = s.hover?.edge === e;
      bases.push([pos[i][0], pos[i][1], pos[i][2]]);
      vectors.push(v);
      let c = mine ? '#9aa3b2' : '#c9ccd3';
      if (attn) c = attn.has(e) ? magnitudeColor(attn.get(e)! / amax, 'attention') : '#d9dbe0';
      else if (edge && edge[e] != null) c = magnitudeColor(edge[e]! / emax, which);
      if (hot) c = '#111827';
      colors.push(c);
      radii.push(hot ? 0.09 : mine ? 0.06 : 0.025);
    }
    const shapes: Record<string, any> = {
      graph: { kind: 'cylinders', parameters: { global: { vectors, bases, radii, colors } } },
    };
    let shapeList = 'graph';
    if (s.forces) {
      const F = p.forces, fmax = Math.max(...Array.from({ length: N }, (_, i) => Math.hypot(F[3 * i], F[3 * i + 1], F[3 * i + 2])), 1e-9);
      const scale = 1.2 / fmax;
      shapes.forces = {
        kind: 'arrow',
        parameters: {
          global: { baseRadius: 0.05, headRadius: 0.11, headLength: 0.2, color: '#7c3aed' },
          atom: Array.from({ length: N }, (_, i) => ({ vector: [F[3 * i] * scale, F[3 * i + 1] * scale, F[3 * i + 2] * scale] })),
        },
      };
      shapeList += ',forces';
    }
    const structure = {
      size: N,
      names: p.numbers.map((z) => SYMBOLS[z]),
      x: pos.map((r) => r[0]), y: pos.map((r) => r[1]), z: pos.map((r) => r[2]),
      cell: p.cell && p.cell.flat().some((x) => x !== 0) ? p.cell.flat() : undefined,
      shapes,
    };
    const nodeProp = node ?? Array.from({ length: N }, () => 0);
    const props = {
      node: nodeProp.map((x, i) => (s.hover?.atom === i ? Math.max(...nodeProp) * 1.2 : x)),
      energy: Array.from(p.energies),
      force: Array.from({ length: N }, (_, i) => Math.hypot(p.forces[3 * i], p.forces[3 * i + 1], p.forces[3 * i + 2])),
    };
    this.viewer.load(structure as any, props, { keepOrientation: true });
    this.viewer.applySettings({
      bonds: false,
      shape: shapeList,
      unitCell: !!structure.cell,
      color: node ? { property: 'node', palette: which === 'grad' ? 'magma' : 'inferno', min: 0, max: Math.max(...nodeProp, 1e-9) }
                  : { property: 'element' },
    } as any);
  }
}
