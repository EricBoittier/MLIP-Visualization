// The 3D network. Every op's output is a block of cells (rows: the selected
// atom's atoms / edges / tokens, columns: features) with the weights it reads to
// its left and, for attention, one matrix per head to its right. Modules are
// tinted frames that can be collapsed. Cells fill in as the forward pass reaches
// them and turn to gradient colours on the way back. Text is world-space, so
// module titles read from afar and op labels appear as you zoom in.
import interFont from '@fontsource/inter/files/inter-latin-400-normal.woff?url';
import interBold from '@fontsource/inter/files/inter-latin-700-normal.woff?url';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Text } from 'troika-three-text';
import type { Thumb } from '../engine/backend';
import type { OpInfo, Trace } from '../worker/protocol';
import { glslVec, PALETTE } from './colors';
import { type Layout, layout, LABEL, type Rect } from './layout';
import { buildTree, type Describe, leafOf, type Mod, MOD_COLOR, walk } from './modules';

const VERT = /* glsl */ `
varying vec2 vUv;
varying vec3 vN;
void main() { vUv = uv; vN = normal; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;

const FRAG = /* glsl */ `
precision highp float;
uniform sampler2D uVal;
uniform sampler2D uGrad;
uniform vec2 uDims;
uniform float uVScale, uGScale, uReveal, uGReveal, uActive, uKind, uHasGrad, uHiRow, uCellPx;
varying vec2 vUv;
varying vec3 vN;
vec3 diverge(float t, vec3 neg, vec3 pos) {
  t = clamp(t, -1.0, 1.0);
  vec3 mid = ${glslVec(PALETTE.mid)};
  return t < 0.0 ? mix(mid, neg, sqrt(-t)) : mix(mid, pos, sqrt(t));
}
void main() {
  vec2 g = vUv * uDims;
  vec2 cell = min(floor(g), uDims - 1.0);
  vec2 f = fract(g);
  float row = uDims.y - 1.0 - cell.y;
  vec2 tc = vec2((cell.x + 0.5) / uDims.x, (row + 0.5) / uDims.y);
  float order = (row + (cell.x + 0.5) / uDims.x) / uDims.y;
  vec3 col = vec3(0.115, 0.125, 0.155);
  if (order < uReveal) {
    float v = texture2D(uVal, tc).r / uVScale;
    if (uKind > 1.5) col = mix(${glslVec(PALETTE.mid)}, ${glslVec(PALETTE.attention)}, sqrt(clamp(v, 0.0, 1.0)));
    else if (uKind > 0.5) col = diverge(v, ${glslVec(PALETTE.weight.neg)}, ${glslVec(PALETTE.weight.pos)});
    else col = diverge(v, ${glslVec(PALETTE.value.neg)}, ${glslVec(PALETTE.value.pos)});
  }
  if (uHasGrad > 0.5 && 1.0 - order < uGReveal) {
    float gv = texture2D(uGrad, tc).r / uGScale;
    col = diverge(gv, ${glslVec(PALETTE.grad.neg)}, ${glslVec(PALETTE.grad.pos)});
  }
  float e = min(min(f.x, 1.0 - f.x), min(f.y, 1.0 - f.y));
  col *= mix(mix(0.5, 1.0, smoothstep(0.03, 0.15, e)), 1.0, 1.0 - smoothstep(2.5, 6.0, uCellPx));
  if (abs(row - uHiRow) < 0.5) col = mix(col, vec3(1.0), 0.35);
  if (abs(vN.z) < 0.5) col *= 0.55;
  col += uActive * vec3(0.12, 0.13, 0.18);
  gl_FragColor = vec4(col, 1.0);
}`;

export interface Hit { kind: 'op' | 'weight' | 'head'; index: number; sub: number; row: number; col: number; value: number; grad?: number }

interface Block {
  kind: Hit['kind'];
  op: number;
  sub: number; // weight / head index within the op
  mesh: THREE.Mesh;
  mat: THREE.ShaderMaterial;
  val: THREE.DataTexture;
  grad: THREE.DataTexture;
  thumb: Thumb;
  gthumb: Thumb | null;
}

interface FrameObj {
  mod: Mod;
  group: THREE.Group;
  fill: THREE.Mesh;
  border: THREE.LineSegments;
  title: Text;
  bar: THREE.Mesh; // progress of a collapsed module
  barMat: THREE.MeshBasicMaterial;
}

interface Label { text: Text; size: number; maxPx: number }

/** On-screen size of one cell, shared by every block's material. */
const CELL_PX = { value: 10 };

const EMPTY = new THREE.DataTexture(new Float32Array(1), 1, 1, THREE.RedFormat, THREE.FloatType);
EMPTY.needsUpdate = true;

function texture(t: { data: Float32Array; rows: number; cols: number }) {
  const tex = new THREE.DataTexture(t.data, t.cols, t.rows, THREE.RedFormat, THREE.FloatType);
  tex.magFilter = tex.minFilter = THREE.NearestFilter;
  tex.needsUpdate = true;
  return tex;
}

function text(str: string, size: number, color: string, bold = false): Text {
  const t = new Text();
  t.text = str;
  t.font = bold ? interBold : interFont;
  t.fontSize = size;
  t.color = color;
  t.anchorX = 'left';
  t.anchorY = 'top';
  t.sync();
  return t;
}

const opLabel = (op: OpInfo) => op.op;
const OP_FONT = 1.1, LABEL_EM = 0.6 * OP_FONT; // Inter runs at most ~0.58 em per character

/** Unit box geometry scaled per block, so re-layout is just a transform. */
const BOX = new THREE.BoxGeometry(1, 1, 1);
const PLANE = new THREE.PlaneGeometry(1, 1);
const SQUARE = new THREE.EdgesGeometry(PLANE);

export class NetworkView {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;
  private world = new THREE.Group();
  private blocks: Block[] = [];
  private opBlock: Block[] = [];
  private frames = new Map<string, FrameObj>();
  private labels: Label[] = [];
  private opLabels: Text[] = [];
  private links: { line: THREE.LineSegments; colors: Float32Array; spans: { from: number; to: number; start: number }[] } | null = null;
  private key = '';
  private targets = new Map<THREE.Object3D, THREE.Vector3>();
  private scales = new Map<THREE.Object3D, THREE.Vector3>();
  private moving = 0;
  root: Mod | null = null;
  leaf: Mod[] = [];
  lay: Layout | null = null;
  ops: OpInfo[] = [];
  trace: Trace | null = null;
  collapsed = new Set<string>();
  /** How the current model names its modules. */
  describe: Describe = () => null;
  rootTitle = '';
  follow = true;
  zen = false;
  /** Show only the parameters. */
  weightsOnly = false;
  private zenTime = 0;
  private want = new THREE.Vector3();
  private wantDist = 120;
  private ray = new THREE.Raycaster();
  private state: { fwd: Float32Array; bwd: Float32Array; active: number; dir: 'fwd' | 'bwd' } = { fwd: new Float32Array(0), bwd: new Float32Array(0), active: -1, dir: 'fwd' };
  onFrame: ((dt: number) => void) | null = null;
  onHover: ((h: Hit | null) => void) | null = null;
  onClick: ((h: Hit | null) => void) | null = null;
  onCollapse: (() => void) | null = null;
  onUserCamera: (() => void) | null = null;

  constructor(readonly el: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setClearColor(0x0c0e13);
    el.appendChild(this.renderer.domElement);
    this.camera = new THREE.PerspectiveCamera(35, 1, 0.5, 200000);
    this.camera.position.set(0, 0, 400);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.screenSpacePanning = true;
    this.controls.zoomToCursor = true;
    this.controls.mouseButtons = { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.ROTATE };
    this.controls.touches = { ONE: THREE.TOUCH.PAN, TWO: THREE.TOUCH.DOLLY_ROTATE };
    this.controls.addEventListener('start', () => { this.follow = false; this.onUserCamera?.(); });
    this.scene.add(this.world);
    new ResizeObserver(() => this.resize()).observe(el);
    this.resize();

    const ndc = (e: PointerEvent) => {
      const r = this.renderer.domElement.getBoundingClientRect();
      return new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    };
    const pick = (e: PointerEvent): Hit | null => {
      this.ray.setFromCamera(ndc(e), this.camera);
      const hit = this.ray.intersectObjects(this.blocks.filter((b) => b.mesh.visible).map((b) => b.mesh), false)[0];
      if (!hit?.uv) return null;
      const b = hit.object.userData.block as Block;
      const col = Math.min(Math.floor(hit.uv.x * b.thumb.cols), b.thumb.cols - 1);
      const row = b.thumb.rows - 1 - Math.min(Math.floor(hit.uv.y * b.thumb.rows), b.thumb.rows - 1);
      const k = row * b.thumb.cols + col;
      return { kind: b.kind, index: b.op, sub: b.sub, row, col, value: b.thumb.data[k], grad: b.gthumb?.data[k] };
    };
    const pickFrame = (e: PointerEvent): Mod | null => {
      this.ray.setFromCamera(ndc(e), this.camera);
      const objs = [...this.frames.values()].filter((f) => f.group.visible).map((f) => f.fill);
      for (const hit of this.ray.intersectObjects(objs, false)) {
        const f = hit.object.userData.frame as FrameObj;
        const fr = this.lay!.frames.get(f.mod.id)!;
        // the title band (or anywhere on a collapsed box) toggles
        const yTop = -fr.rect.y, local = yTop - hit.point.y;
        if (fr.collapsed || local < fr.title * 1.8) return f.mod;
      }
      return null;
    };
    let down = [0, 0];
    const cv = this.renderer.domElement;
    cv.addEventListener('pointermove', (e) => {
      const h = pick(e);
      cv.style.cursor = h ? 'crosshair' : pickFrame(e) ? 'pointer' : '';
      this.onHover?.(h);
    });
    cv.addEventListener('pointerleave', () => this.onHover?.(null));
    cv.addEventListener('pointerdown', (e) => (down = [e.clientX, e.clientY]));
    cv.addEventListener('pointerup', (e) => {
      if (Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
      const h = pick(e);
      if (h) return this.onClick?.(h);
      const m = pickFrame(e);
      if (m) this.toggle(m.id);
    });
    cv.addEventListener('contextmenu', (e) => e.preventDefault());

    let last = performance.now();
    const loop = (now: number) => {
      const dt = Math.min((now - last) / 1000, 0.1);
      last = now;
      this.onFrame?.(dt);
      this.animate(dt);
      this.controls.update();
      CELL_PX.value = this.pxPerUnit(Math.max(this.camera.position.distanceTo(this.controls.target), 1));
      this.updateLabels();
      this.renderer.render(this.scene, this.camera);
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  private layoutAspect = 0;
  private resize() {
    const { clientWidth: w, clientHeight: h } = this.el;
    if (!w || !h) return;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    if (this.root && Math.abs(Math.log(this.camera.aspect / (this.layoutAspect || 1))) > 0.25) this.relayout(false);
  }

  private pxPerUnit(dist: number) {
    return this.el.clientHeight / (2 * dist * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)));
  }

  // ------------------------------------------------------------------ data
  /** Show a pass; rebuilds only when the graph's shape changed. */
  show(ops: OpInfo[], trace: Trace) {
    this.trace = trace;
    const key = this.rootTitle + ops.map((o, i) => `${o.scope}${o.op}${trace.values[i]?.rows}x${trace.values[i]?.cols}`).join() +
      trace.attention.map((a) => a.n).join();
    if (key !== this.key) {
      this.key = key;
      this.build(ops, trace);
      return;
    }
    for (const b of this.blocks) {
      if (b.kind === 'op') this.setData(b, trace.values[b.op]!, trace.grads[b.op]);
      else if (b.kind === 'head') this.setData(b, this.headThumb(trace, b.op, b.sub), null);
      else {
        const p = trace.params?.[ops[b.op].params[b.sub]]?.value;
        if (p) this.setData(b, p, null);
      }
    }
  }

  private setData(b: Block, t: Thumb, g: Thumb | null) {
    b.thumb = t;
    b.gthumb = g;
    (b.val.image as any).data = t.data;
    b.val.needsUpdate = true;
    if (g) {
      if (b.grad === EMPTY) { b.grad = texture(g); b.mat.uniforms.uGrad.value = b.grad; }
      else { (b.grad.image as any).data = g.data; b.grad.needsUpdate = true; }
    }
    b.mat.uniforms.uVScale.value = b.kind === 'head' ? 1 : t.absmax || 1;
    b.mat.uniforms.uGScale.value = g?.absmax || 1;
    b.mat.uniforms.uHasGrad.value = g ? 1 : 0;
  }

  private headThumb(trace: Trace, op: number, head: number): Thumb {
    const a = trace.attention.find((x) => x.op === op)!, n = a.n;
    return { rows: n, cols: n, data: a.probs.slice(head * n * n, (head + 1) * n * n), mean: 0, std: 0, min: 0, max: 1, absmax: 1 };
  }

  private disposeAll() {
    for (const b of this.blocks) { b.mat.dispose(); b.val.dispose(); if (b.grad !== EMPTY) b.grad.dispose(); }
    for (const l of this.labels) l.text.dispose();
    for (const f of this.frames.values()) { (f.fill.material as THREE.Material).dispose(); (f.border.material as THREE.Material).dispose(); f.barMat.dispose(); }
    this.world.clear();
    this.blocks = []; this.opBlock = []; this.labels = []; this.opLabels = []; this.frames.clear();
    this.targets.clear(); this.scales.clear();
  }

  private build(ops: OpInfo[], trace: Trace) {
    this.disposeAll();
    this.ops = ops;
    this.paramShapes = new Map(Object.entries(trace.paramShapes ?? {}));
    this.root = buildTree(ops, this.describe, this.rootTitle);
    this.leaf = leafOf(this.root, ops.length);
    const block = (kind: Hit['kind'], op: number, sub: number, t: Thumb, g: Thumb | null) => {
      const val = texture(t), grad = g ? texture(g) : EMPTY;
      const mat = new THREE.ShaderMaterial({
        vertexShader: VERT, fragmentShader: FRAG,
        uniforms: {
          uVal: { value: val }, uGrad: { value: grad }, uDims: { value: new THREE.Vector2(t.cols, t.rows) },
          uVScale: { value: kind === 'head' ? 1 : t.absmax || 1 }, uGScale: { value: g?.absmax || 1 },
          uReveal: { value: 0 }, uGReveal: { value: 0 }, uActive: { value: 0 },
          uKind: { value: kind === 'weight' ? 1 : kind === 'head' ? 2 : 0 }, uHasGrad: { value: g ? 1 : 0 }, uHiRow: { value: -1 },
          uCellPx: CELL_PX,
        },
      });
      const mesh = new THREE.Mesh(BOX, mat);
      const b: Block = { kind, op, sub, mesh, mat, val, grad, thumb: t, gthumb: g };
      mesh.userData.block = b;
      this.world.add(mesh);
      this.blocks.push(b);
      return b;
    };
    const label = (str: string, size: number, color: string, bold: boolean, maxPx = 1e9) => {
      const t = text(str, size, color, bold);
      this.world.add(t);
      this.labels.push({ text: t, size, maxPx });
      return t;
    };
    ops.forEach((op, i) => {
      this.opBlock.push(block('op', i, 0, trace.values[i]!, trace.grads[i]));
      op.params.forEach((name, k) => { const p = trace.params?.[name]?.value; if (p) block('weight', i, k, p, null); });
      const a = trace.attention.find((x) => x.op === i);
      if (a) for (let h = 0; h < a.heads; h++) block('head', i, h, this.headThumb(trace, i, h), null);
      const shape = trace.shapes[i];
      this.opText[i] = `${opLabel(op)}  [${shape.join('×')}]`;
      this.opLabels.push(label(this.opText[i], OP_FONT, '#aab2c2', false, 90));
    });
    for (const m of walk(this.root)) {
      if (m.depth === 0) continue;
      const color = MOD_COLOR[m.type];
      const group = new THREE.Group();
      const fill = new THREE.Mesh(PLANE, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.07 + 0.02 * Math.min(m.depth, 3), depthWrite: false }));
      fill.userData.frame = null;
      const border = new THREE.LineSegments(SQUARE, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.55 }));
      const barMat = new THREE.MeshBasicMaterial({ color: '#f99e29' });
      const bar = new THREE.Mesh(PLANE, barMat);
      group.add(fill, border, bar);
      this.world.add(group);
      const title = label(m.title, 1, color, true, 260);
      const f: FrameObj = { mod: m, group, fill, border, title, bar, barMat };
      fill.userData.frame = f;
      this.frames.set(m.id, f);
    }
    this.links = null;
    this.relayout(true);
  }

  // ---------------------------------------------------------------- layout
  toggle(id: string) {
    if (this.collapsed.has(id)) this.collapsed.delete(id); else this.collapsed.add(id);
    this.relayout(false);
    this.onCollapse?.();
  }
  setCollapsed(ids: Iterable<string>) {
    this.collapsed = new Set(ids);
    if (this.root) this.relayout(false);
  }

  private moveTo(o: THREE.Object3D, x: number, y: number, z: number, sx = 1, sy = 1, sz = 1, now = false) {
    const p = new THREE.Vector3(x, y, z), s = new THREE.Vector3(sx, sy, sz);
    if (now) { o.position.copy(p); o.scale.copy(s); this.targets.delete(o); this.scales.delete(o); return; }
    this.targets.set(o, p);
    this.scales.set(o, s);
    this.moving = 1;
  }

  private relayout(now: boolean) {
    if (!this.root || !this.trace) return;
    const aspect = Math.max(this.camera.aspect, 0.6);
    this.layoutAspect = this.camera.aspect;
    const labelWidth = (i: number, weights: string[]) => LABEL_EM * (this.weightsOnly ? this.paramLabel(weights) : this.opText[i]).length;
    const lay = (this.lay = layout(this.root, this.ops, this.trace, this.zen || this.weightsOnly ? new Set() : this.collapsed, aspect, this.zen,
                                   this.weightsOnly, labelWidth));
    const place = (o: THREE.Object3D, r: Rect, z: number, d = 1) => this.moveTo(o, r.x + r.w / 2, -(r.y + r.h / 2), z, r.w, r.h, d, now);
    for (const b of this.blocks) {
      const c = lay.cards[b.op];
      b.mesh.visible = c.visible && (!this.weightsOnly || b.kind === 'weight');
      const r = b.kind === 'op' ? c.block : b.kind === 'weight' ? c.weights.find((w) => w.name === this.ops[b.op].params[b.sub])?.rect : c.heads[b.sub]?.rect;
      if (b.kind === 'weight' && !r) b.mesh.visible = false; // shown at another op
      if (r) place(b.mesh, { x: c.rect.x + r.x, y: c.rect.y + r.y, w: r.w, h: r.h }, b.kind === 'weight' ? 0.3 : 0.6, b.kind === 'weight' ? 0.6 : 1.2);
    }
    this.opLabels.forEach((t, i) => {
      const c = lay.cards[i];
      t.visible = c.visible && !this.zen;
      const want = this.weightsOnly ? this.paramLabel(c.weights.map((w) => w.name)) : this.opText[i];
      if (t.text !== want) { t.text = want; t.sync(); }
      this.moveTo(t, c.rect.x, -c.rect.y - 0.4, 0.8, 1, 1, 1, now);
    });
    for (const f of this.frames.values()) {
      const fr = lay.frames.get(f.mod.id)!;
      f.group.visible = fr.visible;
      f.title.visible = fr.visible && !this.zen;
      f.border.visible = !this.zen;
      const r = fr.rect, z = -1 - 0.05 * f.mod.depth;
      this.moveTo(f.fill, r.x + r.w / 2, -(r.y + r.h / 2), z, r.w, r.h, 1, now);
      this.moveTo(f.border, r.x + r.w / 2, -(r.y + r.h / 2), z + 0.01, r.w, r.h, 1, now);
      (f.fill.material as THREE.MeshBasicMaterial).opacity = this.zen ? 0.035 : fr.collapsed ? 0.22 : 0.07 + 0.02 * Math.min(f.mod.depth, 3);
      f.bar.visible = fr.collapsed;
      this.moveTo(f.bar, r.x + 1.5, -(r.y + r.h - 2), z + 0.02, 0.001, 1, 1, now);
      f.title.fontSize = fr.title;
      this.labels.find((l) => l.text === f.title)!.size = fr.title;
      this.moveTo(f.title, r.x + 1.6, -r.y - fr.title * 0.4, 0.5, 1, 1, 1, now);
      f.title.text = fr.collapsed ? `${f.mod.title}  +` : f.mod.title;
      f.title.sync();
    }
    this.buildLinks();
    this.setProgress(this.state.fwd, this.state.bwd, this.state.active, this.state.dir);
  }

  private animate(dt: number) {
    if (this.zen && this.follow && this.lay) {
      // a slow sway around the overview, for depth
      this.zenTime += dt;
      const { w, h } = this.lay.bounds, t = this.zenTime;
      const yaw = 0.22 * Math.sin(t * 0.11), pitch = 0.12 * Math.sin(t * 0.07 + 1);
      const fit = Math.max(h * 1.25, (w * 1.2) / this.camera.aspect) / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)));
      const dist = (this.wantDist = THREE.MathUtils.lerp(this.wantDist, fit, 1 - Math.exp(-dt * 3)));
      this.want.set(w / 2, -h / 2 - 0.06 * h, 0); // leave room for the parameter count
      const target = this.controls.target.lerp(this.want, 1 - Math.exp(-dt * 2));
      this.camera.position.set(target.x + dist * Math.sin(yaw), target.y + dist * Math.sin(pitch), target.z + dist * Math.cos(yaw) * Math.cos(pitch));
      this.camera.lookAt(target);
    } else if (this.follow) {
      const k = 1 - Math.exp(-dt * 3.5);
      const t = this.controls.target, cam = this.camera.position;
      const dir = cam.clone().sub(t).normalize();
      const dist = THREE.MathUtils.lerp(cam.distanceTo(t), this.wantDist, k);
      t.lerp(this.want, k);
      cam.copy(t).addScaledVector(dir, dist);
    }
    if (!this.moving) return;
    const k = 1 - Math.exp(-dt * 9);
    let left = 0;
    for (const [o, p] of this.targets) {
      o.position.lerp(p, k);
      o.scale.lerp(this.scales.get(o)!, k);
      if (o.position.distanceToSquared(p) > 1e-4 || o.scale.distanceToSquared(this.scales.get(o)!) > 1e-4) left++;
      else { o.position.copy(p); o.scale.copy(this.scales.get(o)!); this.targets.delete(o); this.scales.delete(o); }
    }
    this.moving = left;
    this.buildLinks();
  }

  /** Links between ops, from where each block currently is. */
  private buildLinks() {
    if (!this.lay) return;
    const pos = (i: number) => {
      const b = this.opBlock[i].mesh;
      if (b.visible) return { x: b.position.x, y: b.position.y, w: b.scale.x, h: b.scale.y };
      const r = this.lay!.anchor(i);
      return { x: r.x + r.w / 2, y: -(r.y + r.h / 2), w: r.w, h: r.h };
    };
    const SEG = 10, pts: number[] = [], spans: { from: number; to: number; start: number }[] = [];
    this.ops.forEach((op, i) => {
      const b = pos(i);
      for (const j of new Set(op.inputs)) {
        if (j < 0) continue;
        const a = pos(j);
        if (a.x === b.x && a.y === b.y) continue; // inside the same collapsed module
        if (j === i - 1 && this.lay!.cards[i].visible) continue; // reading order already says this
        const p0 = new THREE.Vector3(a.x + a.w / 2, a.y, 0.2), p3 = new THREE.Vector3(b.x - b.w / 2, b.y, 0.2);
        const dx = Math.max(Math.abs(p3.x - p0.x) * 0.5, 4);
        const curve = new THREE.CubicBezierCurve3(p0, new THREE.Vector3(p0.x + dx, p0.y, 0.2), new THREE.Vector3(p3.x - dx, p3.y, 0.2), p3);
        const p = curve.getPoints(SEG);
        spans.push({ from: j, to: i, start: pts.length / 3 });
        for (let k = 0; k < SEG; k++) pts.push(p[k].x, p[k].y, p[k].z, p[k + 1].x, p[k + 1].y, p[k + 1].z);
      }
    });
    if (this.links) { this.world.remove(this.links.line); this.links.line.geometry.dispose(); }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    const colors = new Float32Array(pts.length);
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    const line = new THREE.LineSegments(geo, this.links?.line.material ?? new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.85 }));
    line.visible = !this.zen && !this.weightsOnly;
    this.world.add(line);
    this.links = { line, colors, spans };
    this.colorLinks();
  }

  private colorLinks() {
    if (!this.links) return;
    const { fwd, bwd, active, dir } = this.state;
    const { colors, spans, line } = this.links;
    const base = [0.14, 0.15, 0.19];
    const hot = dir === 'fwd' ? PALETTE.value.pos : PALETTE.grad.neg;
    for (const s of spans) {
      const on = dir === 'fwd' ? s.to === active : s.from === active || s.to === active;
      const c = on ? hot : base;
      for (let k = 0; k < 20; k++) colors.set(c, 3 * (s.start + k));
    }
    line.geometry.attributes.color.needsUpdate = true;
  }

  // -------------------------------------------------------------- playback
  /** Per-op forward and backward progress (0..1), the active op and direction. */
  setProgress(fwd: Float32Array, bwd: Float32Array, active: number, dir: 'fwd' | 'bwd') {
    this.state = { fwd, bwd, active, dir };
    if (!this.lay) return;
    for (const b of this.blocks) {
      const u = b.mat.uniforms;
      if (b.kind === 'weight') { u.uReveal.value = 1; u.uActive.value = b.op === active ? 1 : 0; continue; }
      u.uReveal.value = fwd[b.op] ?? 0;
      if (b.kind === 'op') { u.uGReveal.value = bwd[b.op] ?? 0; u.uActive.value = b.op === active ? 1 : 0; }
    }
    // collapsed modules show their progress as a bar
    for (const f of this.frames.values()) {
      const fr = this.lay.frames.get(f.mod.id)!;
      if (!fr.collapsed) continue;
      const n = f.mod.last - f.mod.first + 1;
      let sf = 0, sb = 0;
      for (let i = f.mod.first; i <= f.mod.last; i++) { sf += fwd[i] ?? 0; sb += bwd[i] ?? 0; }
      const back = sb > 0;
      const frac = (back ? sb : sf) / n;
      f.barMat.color.set(back ? '#a855f7' : '#f99e29');
      const w = Math.max((fr.rect.w - 3) * frac, 0.001);
      this.moveTo(f.bar, fr.rect.x + 1.5 + w / 2, -(fr.rect.y + fr.rect.h - 2), -0.9, w, 1, 1, true);
      (f.fill.material as THREE.MeshBasicMaterial).opacity = active >= f.mod.first && active <= f.mod.last ? 0.4 : 0.22;
    }
    this.colorLinks();
  }

  highlightRow(op: number, row: number) {
    for (const b of this.opBlock) b.mat.uniforms.uHiRow.value = b.op === op ? row : -1;
  }

  /** Fly the camera to an op (or the collapsed module that holds it). */
  focus(op: number, immediate = false) {
    if (!this.lay) return;
    const c = this.lay.cards[op];
    const r = c.visible ? c.rect : this.lay.anchor(op);
    this.want.set(r.x + r.w / 2, -(r.y + r.h / 2), 0);
    const fit = Math.max(r.h * 2.6, (r.w * 1.5) / this.camera.aspect, 22);
    this.wantDist = fit / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)));
    if (immediate) {
      this.controls.target.copy(this.want);
      this.camera.position.set(this.want.x, this.want.y, this.wantDist);
    }
  }

  private opText: string[] = [];
  /** An op's parameters, for the weights-only view: their names and full shapes. */
  private paramLabel(names: string[]) {
    return names.map((n) => `${n.replace(/^gnn_layers\.\d+\.(trans\.layers\.\d+\.)?/, '')} [${this.paramShapes.get(n)?.join('×') ?? ''}]`).join('   ');
  }
  /** Full shapes of the parameters (the thumbnails are averaged down). */
  paramShapes = new Map<string, number[]>();

  /** Only the weights, grouped by module, at their own size. */
  setWeightsOnly(on: boolean) {
    this.weightsOnly = on;
    this.relayout(false);
    this.overview();
  }

  /** Zen mode: no text, no links, blocks packed tight, the whole network in view, slowly drifting. */
  setZen(on: boolean) {
    this.zen = on;
    this.relayout(false);
    this.overview();
    if (on) this.zenTime = 0;
  }

  /** Frame a whole module. */
  focusModule(id: string) {
    const f = this.lay?.frames.get(id);
    if (!f) return;
    const r = f.rect;
    this.follow = true;
    this.want.set(r.x + r.w / 2, -(r.y + r.h / 2), 0);
    this.wantDist = Math.max(r.h * 1.15, (r.w * 1.1) / this.camera.aspect) / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)));
  }

  overview() {
    if (!this.lay) return;
    const { w, h } = this.lay.bounds;
    this.follow = true;
    this.want.set(w / 2, -h / 2, 0);
    this.wantDist = Math.max(h * 1.08, (w * 1.08) / this.camera.aspect) / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)));
  }

  /** Level of detail: text fades in when it is legible and out when it fills the screen. */
  private updateLabels() {
    const cam = this.camera.position;
    for (const l of this.labels) {
      const d = Math.max(cam.distanceTo(l.text.position), 1);
      const px = l.size * this.pxPerUnit(d);
      const a = THREE.MathUtils.smoothstep(px, 4, 8) * (1 - THREE.MathUtils.smoothstep(px, l.maxPx, l.maxPx * 1.6));
      (l.text as any).fillOpacity = a;
      (l.text as any).material.visible = a > 0.02;
    }
  }
}
