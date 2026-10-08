// The structure, drawn with three.js: atoms, the model's graph (each directed edge
// i -> j as the half from i towards j), force arrows and the unit cell. Atoms are
// coloured by a property chosen from a menu: the op the pass is on, element,
// atomic energy, |F|, charge (when the model predicts charges) or attention.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { SYMBOLS } from '../common/elements';
import type { AttentionMap, OpInfo, Pass } from '../worker/protocol';
import { PALETTE } from './colors';

export interface MolState {
  op: number; // active op, -1 for none
  dir: 'fwd' | 'bwd';
  forces: boolean;
  attention: AttentionMap | null;
  hover: { atom?: number; edge?: number } | null;
}

type ColorBy = 'pass' | 'element' | 'energy' | 'force' | 'charge' | 'attention';

// Jmol colours and covalent radii (A) for the first rows; others fall back to grey / 1.5 A
export const JMOL: Record<number, number> = {
  1: 0xffffff, 3: 0xcc80ff, 5: 0xffb5b5, 6: 0x909090, 7: 0x3050f8, 8: 0xff0d0d, 9: 0x90e050, 11: 0xab5cf2, 12: 0x8aff00,
  13: 0xbfa6a6, 14: 0xf0c8a0, 15: 0xff8000, 16: 0xffff30, 17: 0x1ff01f, 19: 0x8f40d4, 20: 0x3dff00, 26: 0xe06633,
  29: 0xc88033, 30: 0x7d80b0, 35: 0xa62929, 53: 0x940094,
};
export const RCOV: Record<number, number> = { 1: 0.31, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 11: 1.66, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02, 35: 1.2, 53: 1.39 };

/** Sequential (dark to bright) and diverging (blue / grey / red) colour maps. */
const SEQ = [[0.12, 0.13, 0.18], [0.42, 0.22, 0.55], [0.85, 0.35, 0.33], [0.99, 0.74, 0.28], [1, 0.98, 0.75]];
const DIV = [[0.2, 0.42, 0.9], [0.55, 0.68, 0.95], [0.82, 0.83, 0.86], [0.96, 0.62, 0.5], [0.85, 0.2, 0.2]];
function ramp(stops: number[][], t: number) {
  t = Math.min(Math.max(t, 0), 1) * (stops.length - 1);
  const k = Math.min(Math.floor(t), stops.length - 2), f = t - k;
  return stops[k].map((x, i) => x + (stops[k + 1][i] - x) * f);
}

const UP = new THREE.Vector3(0, 1, 0);
const SPHERE = new THREE.SphereGeometry(1, 28, 18);
const CYL = new THREE.CylinderGeometry(1, 1, 1, 10, 1).translate(0, 0.5, 0); // base at the origin, along +y
const CONE = new THREE.ConeGeometry(1, 1, 14, 1).translate(0, 0.5, 0);

export class MoleculeView {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.PerspectiveCamera(30, 1, 0.05, 1000);
  readonly controls: OrbitControls;
  private group = new THREE.Group();
  private atoms: THREE.InstancedMesh | null = null;
  private edges: THREE.InstancedMesh | null = null;
  private arrows: THREE.InstancedMesh[] = [];
  private cell: THREE.LineSegments | null = null;
  private halo: THREE.Mesh;
  private pass: Pass | null = null;
  private ops: OpInfo[] = [];
  private structureKey = '';
  private state: MolState = { op: -1, dir: 'fwd', forces: false, attention: null, hover: null };
  private ray = new THREE.Raycaster();
  private dirty = true;
  colorBy: ColorBy = 'pass';
  showEdges = true;
  showForces = true;
  onSelect: ((atom: number) => void) | null = null;
  private bar: HTMLCanvasElement;
  private barText: HTMLElement;
  private select: HTMLSelectElement;

  constructor(readonly el: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setClearColor(0x000000, 0); // see-through: the network shows behind
    this.renderer.domElement.className = 'mol-canvas';
    el.appendChild(this.renderer.domElement);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x404858, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(3, 5, 6);
    this.camera.add(sun);
    this.scene.add(this.camera, this.group);
    this.halo = new THREE.Mesh(SPHERE, new THREE.MeshBasicMaterial({ color: 0xf99e29, transparent: true, opacity: 0.28, depthWrite: false }));
    this.group.add(this.halo);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.addEventListener('change', () => (this.dirty = true));

    // overlay: colour menu, toggles, colour bar
    const ui = document.createElement('div');
    ui.className = 'mol-ui';
    ui.innerHTML = `<select title="Colour atoms by"></select>
      <label title="The model's graph"><input type="checkbox" checked data-t="edges">graph</label>
      <label title="Force arrows"><input type="checkbox" checked data-t="forces">forces</label>`;
    el.appendChild(ui);
    this.select = ui.querySelector('select')!;
    this.select.onchange = () => { this.colorBy = this.select.value as ColorBy; this.update(); };
    ui.querySelectorAll<HTMLInputElement>('input[type=checkbox]').forEach((c) => (c.onchange = () => {
      if (c.dataset.t === 'edges') this.showEdges = c.checked; else this.showForces = c.checked;
      this.update();
    }));
    const legend = document.createElement('div');
    legend.className = 'mol-legend';
    this.bar = document.createElement('canvas');
    this.bar.width = 160; this.bar.height = 8;
    this.barText = document.createElement('div');
    legend.append(this.bar, this.barText);
    el.appendChild(legend);

    // picking: a click (not a drag) on an atom selects it
    let down = [0, 0];
    const cv = this.renderer.domElement;
    cv.addEventListener('pointerdown', (e) => (down = [e.clientX, e.clientY]));
    cv.addEventListener('pointerup', (e) => {
      if (Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4 || !this.atoms) return;
      const r = cv.getBoundingClientRect();
      this.ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), this.camera);
      const hit = this.ray.intersectObject(this.atoms, false)[0];
      if (hit?.instanceId !== undefined) this.onSelect?.(hit.instanceId);
    });

    new ResizeObserver(() => this.resize()).observe(el);
    this.resize();
    const loop = () => {
      this.controls.update();
      if (this.dirty) { this.renderer.render(this.scene, this.camera); this.dirty = false; }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  private resize() {
    const { clientWidth: w, clientHeight: h } = this.el;
    if (!w || !h) return;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.dirty = true;
  }

  setPass(pass: Pass, ops: OpInfo[]) {
    this.pass = pass;
    this.ops = ops;
    const key = pass.numbers.join() + (pass.cell?.flat().join() ?? '');
    const fresh = key !== this.structureKey;
    this.structureKey = key;
    // the menu: what this pass can colour atoms by
    const opts: [ColorBy, string][] = [['pass', 'colour: the current op'], ['element', 'colour: element'], ['energy', 'colour: atomic energy'],
      ['force', 'colour: |force|']];
    if (pass.atomProps?.charge) opts.push(['charge', 'colour: charge']);
    if (pass.trace.attention.length) opts.push(['attention', 'colour: attention']);
    if (!opts.some(([k]) => k === this.colorBy)) this.colorBy = 'pass';
    this.select.innerHTML = opts.map(([k, l]) => `<option value="${k}"${k === this.colorBy ? ' selected' : ''}>${l}</option>`).join('');
    this.build(fresh);
  }

  render(s: MolState) {
    this.state = s;
    this.update();
  }

  // ---------------------------------------------------------------- geometry
  private build(fit: boolean) {
    const p = this.pass!, N = p.numbers.length, g = p.trace.graph, E = g.center.length;
    for (const o of [this.atoms, this.edges, ...this.arrows, this.cell]) if (o) { this.group.remove(o); if (o !== this.cell) (o as THREE.InstancedMesh).dispose(); }
    this.atoms = new THREE.InstancedMesh(SPHERE, new THREE.MeshStandardMaterial({ roughness: 0.45, metalness: 0.05 }), N);
    const m = new THREE.Matrix4(), q = new THREE.Quaternion();
    p.positions.forEach((r, i) => {
      const rad = 0.18 + 0.32 * (RCOV[p.numbers[i]] ?? 1.5);
      this.atoms!.setMatrixAt(i, m.compose(new THREE.Vector3(...r), q, new THREE.Vector3(rad, rad, rad)));
      this.atoms!.setColorAt(i, new THREE.Color(0xffffff));
    });
    this.group.add(this.atoms);
    this.edges = new THREE.InstancedMesh(CYL, new THREE.MeshStandardMaterial({ roughness: 0.6 }), Math.max(E, 1));
    this.edges.count = E;
    this.group.add(this.edges);
    this.arrows = [0, 1].map(() => {
      const shaft = new THREE.InstancedMesh(CYL, new THREE.MeshStandardMaterial({ roughness: 0.5 }), N);
      const head = new THREE.InstancedMesh(CONE, shaft.material, N);
      shaft.add(head);
      this.group.add(shaft);
      return shaft;
    });
    if (p.cell && p.cell.flat().some((x) => x !== 0)) {
      const [a, b, c] = p.cell.map((v) => new THREE.Vector3(...v)), o = new THREE.Vector3();
      const corners = [o, a, b, c, a.clone().add(b), a.clone().add(c), b.clone().add(c), a.clone().add(b).add(c)];
      const e = [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7]];
      this.cell = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(e.flatMap(([i, j]) => [corners[i], corners[j]])),
                                         new THREE.LineBasicMaterial({ color: 0x8a93a6 }));
      this.group.add(this.cell);
    } else this.cell = null;
    if (fit) this.fit();
    this.update();
  }

  private fit() {
    const pos = this.pass!.positions, ctr = new THREE.Vector3();
    pos.forEach((r) => ctr.add(new THREE.Vector3(...r)));
    ctr.divideScalar(pos.length);
    let R = 1.5;
    pos.forEach((r) => (R = Math.max(R, ctr.distanceTo(new THREE.Vector3(...r)) + 1)));
    const dist = R / Math.sin(THREE.MathUtils.degToRad(this.camera.fov / 2));
    this.controls.target.copy(ctr);
    this.camera.position.copy(ctr).add(new THREE.Vector3(0, 0, dist));
    this.camera.near = dist / 50; this.camera.far = dist * 10;
    this.camera.updateProjectionMatrix();
  }

  /** Colours, edges, arrows and the halo for the current state. */
  private update() {
    const p = this.pass;
    if (!p || !this.atoms || !this.edges) return;
    const s = this.state, N = p.numbers.length, g = p.trace.graph, E = g.center.length, a = p.trace.selected;
    const op = s.op >= 0 ? this.ops[s.op] : null;

    // per-atom and per-edge values for the 'pass' colouring
    let node: Float64Array | null = null, edgeKind: 'value' | 'grad' | 'attention' = 'value';
    const edgeVals = new Float64Array(E).fill(NaN);
    let hasEdge = false;
    const norms = op ? (s.dir === 'bwd' ? p.trace.gradNorms[s.op] ?? p.trace.norms[s.op] : p.trace.norms[s.op]) : null;
    if (s.dir === 'bwd' && op && p.trace.gradNorms[s.op]) edgeKind = 'grad';
    const space = op?.kind ? p.trace.rows[op.kind] : undefined;
    if (norms && space) {
      if (space.atom || space.edge) {
        const nd = new Float64Array(N);
        let hasNode = false;
        space.atom?.forEach((i, r) => { if (i >= 0) { nd[i] = Math.max(nd[i], norms[r]); hasNode = true; } });
        space.edge?.forEach((e, r) => { if (e >= 0) { edgeVals[e] = norms[r]; hasEdge = true; } });
        if (hasNode) node = nd;
      } else {
        const sum = new Float64Array(N), cnt = new Float64Array(N);
        space.owner.forEach((i, r) => { if (i >= 0) { sum[i] += norms[r]; cnt[i]++; } });
        node = sum.map((x, i) => (cnt[i] ? x / cnt[i] : 0));
      }
    }
    // attention of the selected atom's own token to its neighbours (mean over heads)
    let attnEdge: Map<number, number> | null = null;
    const A = s.attention ?? (this.colorBy === 'attention' ? p.trace.attention[p.trace.attention.length - 1] ?? null : null);
    if (A && (s.attention || this.colorBy === 'attention')) {
      const mine = Array.from(g.center.keys()).filter((e) => g.center[e] === a), n = A.n;
      attnEdge = new Map();
      for (let t = 1; t < n; t++) {
        let w = 0;
        for (let h = 0; h < A.heads; h++) w += A.probs[h * n * n + t];
        if (mine[t - 1] !== undefined) attnEdge.set(mine[t - 1], w / A.heads);
      }
      edgeKind = 'attention';
    }

    // ---- atoms
    const F = p.forces ?? p.ncForces;
    const props: Record<ColorBy, (Float32Array | Float64Array | null)> = {
      pass: node, element: null, energy: p.energies, force: F ? Float64Array.from({ length: N }, (_, i) => Math.hypot(F[3 * i], F[3 * i + 1], F[3 * i + 2])) : null,
      charge: p.atomProps?.charge ?? null,
      attention: attnEdge ? Float64Array.from({ length: N }, (_, i) => {
        let w = 0;
        attnEdge!.forEach((v, e) => { if (g.neighbor[e] === i) w = Math.max(w, v); });
        return i === a ? 1 : w;
      }) : null,
    };
    const raw = props[this.colorBy], vals = raw ? Float64Array.from(raw) : null;
    const signed = this.colorBy === 'charge' || this.colorBy === 'energy';
    let lo = Infinity, hi = -Infinity;
    if (vals) for (const v of vals) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    if (signed && vals) {
      if (this.colorBy === 'energy') { const mean = vals.reduce((x, y) => x + y, 0) / N; lo -= mean; hi -= mean; }
      const m = Math.max(Math.abs(lo), Math.abs(hi), 1e-9); lo = -m; hi = m;
    }
    const meanE = this.colorBy === 'energy' && vals ? vals.reduce((x, y) => x + y, 0) / N : 0;
    const col = new THREE.Color();
    for (let i = 0; i < N; i++) {
      if (!vals) col.setHex(JMOL[p.numbers[i]] ?? 0xb0b0b0);
      else if (signed) col.setRGB(...(ramp(DIV, ((vals[i] - meanE) - lo) / (hi - lo || 1)) as [number, number, number]));
      else col.setRGB(...(ramp(SEQ, (vals[i] - Math.min(lo, 0)) / ((hi - Math.min(lo, 0)) || 1)) as [number, number, number]));
      if (s.hover?.atom === i) col.lerp(new THREE.Color(1, 1, 1), 0.5);
      this.atoms.setColorAt(i, col);
    }
    this.atoms.instanceColor!.needsUpdate = true;
    this.legend(vals ? { lo: signed ? lo : Math.min(lo, 0), hi, signed, label: this.legendLabel(op, s.dir), offset: meanE } : null);

    // ---- edges: directed halves, coloured by the pass (or attention)
    this.edges.visible = this.showEdges;
    let emax = 1e-12;
    if (hasEdge) for (const v of edgeVals) if (v === v) emax = Math.max(emax, v);
    const amax = attnEdge ? Math.max(...attnEdge.values(), 1e-12) : 1;
    const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), from = new THREE.Vector3(), vec = new THREE.Vector3();
    for (let e = 0; e < E; e++) {
      const i = g.center[e], j = g.neighbor[e];
      from.set(...(p.positions[i] as [number, number, number]));
      vec.set(...([0, 1, 2].map((k) => 0.5 * (p.positions[j][k] + g.shift[3 * e + k] - p.positions[i][k])) as [number, number, number]));
      const len = vec.length(), mine = i === a, hot = s.hover?.edge === e;
      const rad = hot ? 0.09 : mine ? 0.06 : 0.03;
      q.setFromUnitVectors(UP, vec.clone().normalize());
      this.edges.setMatrixAt(e, m4.compose(from, q, new THREE.Vector3(rad, len, rad)));
      let c: number[] = mine ? [0.62, 0.66, 0.74] : [0.4, 0.43, 0.5];
      if (attnEdge) c = attnEdge.has(e) ? ramp([[0.3, 0.3, 0.3], PALETTE.attention], Math.sqrt(attnEdge.get(e)! / amax)) : [0.3, 0.32, 0.36];
      else if (hasEdge && edgeVals[e] === edgeVals[e]) c = ramp([[0.3, 0.32, 0.38], edgeKind === 'grad' ? PALETTE.grad.neg : PALETTE.value.pos], Math.sqrt(edgeVals[e] / emax));
      if (hot) c = [1, 1, 1];
      this.edges.setColorAt(e, new THREE.Color(...(c as [number, number, number])));
    }
    this.edges.instanceMatrix.needsUpdate = true;
    if (this.edges.instanceColor) this.edges.instanceColor.needsUpdate = true;

    // ---- forces: conservative (violet) and direct (lime), on one scale
    const sets: [Float32Array | undefined, number][] = [[p.forces, 0x8b5cf6], [p.ncForces, 0x84cc16]];
    const fmax = Math.max(1e-9, ...sets.flatMap(([Fs]) => (Fs ? Array.from({ length: N }, (_, i) => Math.hypot(Fs[3 * i], Fs[3 * i + 1], Fs[3 * i + 2])) : [])));
    sets.forEach(([Fs, color], k) => {
      const shaft = this.arrows[k], head = shaft.children[0] as THREE.InstancedMesh;
      shaft.visible = !!Fs && s.forces && this.showForces;
      (shaft.material as THREE.MeshStandardMaterial).color.setHex(color);
      if (!Fs) return;
      for (let i = 0; i < N; i++) {
        const f = new THREE.Vector3(Fs[3 * i], Fs[3 * i + 1], Fs[3 * i + 2]), L = (1.3 * f.length()) / fmax;
        const dir = f.lengthSq() > 0 ? f.normalize() : UP.clone();
        q.setFromUnitVectors(UP, dir);
        const base = new THREE.Vector3(...(p.positions[i] as [number, number, number]));
        const hl = Math.min(0.25, L * 0.4), sl = Math.max(L - hl, 0);
        shaft.setMatrixAt(i, m4.compose(base, q, new THREE.Vector3(0.05, sl, 0.05)));
        head.setMatrixAt(i, m4.compose(base.clone().addScaledVector(dir, sl), q, new THREE.Vector3(0.12, hl, 0.12)));
      }
      shaft.instanceMatrix.needsUpdate = true;
      head.instanceMatrix.needsUpdate = true;
    });

    // ---- the selected atom
    const ra = 0.18 + 0.32 * (RCOV[p.numbers[a]] ?? 1.5) + 0.22;
    this.halo.position.set(...(p.positions[a] as [number, number, number]));
    this.halo.scale.setScalar(ra);
    this.dirty = true;
  }

  private legendLabel(op: OpInfo | null, dir: 'fwd' | 'bwd') {
    switch (this.colorBy) {
      case 'pass': return op ? `${op.op}: ${dir === 'bwd' ? 'gradient' : 'value'} RMS` : '';
      case 'energy': return 'atomic energy − mean (eV)';
      case 'force': return '|F| (eV/Å)';
      case 'charge': return 'charge (e)';
      case 'attention': return 'attention from the selected atom';
      default: return '';
    }
  }

  private legend(l: { lo: number; hi: number; signed: boolean; label: string; offset: number } | null) {
    const parent = this.bar.parentElement!;
    parent.hidden = !l;
    if (!l) return;
    const g = this.bar.getContext('2d')!;
    for (let x = 0; x < this.bar.width; x++) {
      const c = ramp(l.signed ? DIV : SEQ, x / (this.bar.width - 1));
      g.fillStyle = `rgb(${c.map((v) => Math.round(255 * v)).join(',')})`;
      g.fillRect(x, 0, 1, this.bar.height);
    }
    const f = (x: number) => (Math.abs(x) >= 100 || (Math.abs(x) < 0.01 && x !== 0) ? x.toExponential(1) : x.toFixed(2));
    this.barText.innerHTML = `<span>${f(l.lo)}</span><b>${l.label}</b><span>${f(l.hi)}</span>`;
  }
}

export const elementName = (z: number) => SYMBOLS[z];
