// The moving structure: atoms, bonds from covalent radii and the unit cell (atoms wrapped into it).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { type System, inv3, isPeriodic } from '../common/structure';
import { JMOL, RCOV } from '../viz/molecule';

const SPHERE = new THREE.SphereGeometry(1, 24, 16);
const CYL = new THREE.CylinderGeometry(1, 1, 1, 10, 1).translate(0, 0.5, 0);
const UP = new THREE.Vector3(0, 1, 0);
const rcov = (z: number) => RCOV[z] ?? 1.5;

export class TrajectoryView {
  private renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(30, 1, 0.05, 1000);
  private controls: OrbitControls;
  private group = new THREE.Group();
  private atoms: THREE.InstancedMesh | null = null;
  private bonds: THREE.InstancedMesh | null = null;
  private cell: THREE.LineSegments | null = null;
  private sys: System | null = null;
  private inv: number[][] | null = null; // cell inverse, for wrapping
  private dirty = true;

  constructor(readonly el: HTMLElement) {
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setClearColor(0x000000, 0);
    el.appendChild(this.renderer.domElement);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x404858, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(3, 5, 6);
    this.camera.add(sun);
    this.scene.add(this.camera, this.group);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.addEventListener('change', () => (this.dirty = true));
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

  /** A new structure: rebuild the meshes and frame it. */
  setSystem(sys: System) {
    this.sys = sys;
    const N = sys.numbers.length;
    for (const o of [this.atoms, this.bonds]) if (o) { this.group.remove(o); o.dispose(); }
    if (this.cell) this.group.remove(this.cell);
    this.atoms = new THREE.InstancedMesh(SPHERE, new THREE.MeshStandardMaterial({ roughness: 0.45, metalness: 0.05 }), N);
    sys.numbers.forEach((z, i) => this.atoms!.setColorAt(i, new THREE.Color(JMOL[z] ?? 0xb0b0b0)));
    this.bonds = new THREE.InstancedMesh(CYL, new THREE.MeshStandardMaterial({ roughness: 0.6, color: 0x6b7385 }), Math.max((N * (N - 1)) / 2, 1));
    this.group.add(this.atoms, this.bonds);
    this.cell = null;
    this.inv = null;
    if (isPeriodic(sys)) {
      const [a, b, c] = sys.cell!.map((v) => new THREE.Vector3(...v)), o = new THREE.Vector3();
      const k = [o, a, b, c, a.clone().add(b), a.clone().add(c), b.clone().add(c), a.clone().add(b).add(c)];
      const e = [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7]];
      this.cell = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(e.flatMap(([i, j]) => [k[i], k[j]])),
                                         new THREE.LineBasicMaterial({ color: 0x8a93a6 }));
      this.group.add(this.cell);
      this.inv = inv3(sys.cell!);
    }
    this.update(Float32Array.from(sys.positions.flat()));
    this.fit();
  }

  /** Wrapped into the cell along periodic directions. */
  private wrap(p: Float32Array) {
    const { inv, sys } = this;
    if (!inv || !sys) return p;
    const out = Float32Array.from(p), C = sys.cell!;
    for (let i = 0; i < p.length / 3; i++) {
      const r = [p[3 * i], p[3 * i + 1], p[3 * i + 2]];
      for (let k = 0; k < 3; k++) {
        if (!sys.pbc?.[k]) continue;
        const f = Math.floor(r[0] * inv[0][k] + r[1] * inv[1][k] + r[2] * inv[2][k]);
        for (let c = 0; c < 3; c++) out[3 * i + c] -= f * C[k][c];
      }
    }
    return out;
  }

  update(positions: Float32Array) {
    const { atoms, bonds, sys } = this;
    if (!atoms || !bonds || !sys) return;
    const p = this.wrap(positions), N = sys.numbers.length, m = new THREE.Matrix4(), q = new THREE.Quaternion();
    const v = (i: number) => new THREE.Vector3(p[3 * i], p[3 * i + 1], p[3 * i + 2]);
    for (let i = 0; i < N; i++) {
      const r = 0.18 + 0.32 * rcov(sys.numbers[i]);
      atoms.setMatrixAt(i, m.compose(v(i), q, new THREE.Vector3(r, r, r)));
    }
    atoms.instanceMatrix.needsUpdate = true;
    let k = 0;
    for (let i = 0; i < N; i++)
      for (let j = i + 1; j < N; j++) {
        const a = v(i), d = v(j).sub(a), len = d.length();
        if (len > 1.2 * (rcov(sys.numbers[i]) + rcov(sys.numbers[j]))) continue;
        q.setFromUnitVectors(UP, d.clone().normalize());
        bonds.setMatrixAt(k++, m.compose(a, q, new THREE.Vector3(0.09, len, 0.09)));
      }
    bonds.count = k;
    bonds.instanceMatrix.needsUpdate = true;
    this.dirty = true;
  }

  private fit() {
    const pos = this.sys!.positions, ctr = new THREE.Vector3();
    const pts = [...pos, ...(this.cell ? [this.sys!.cell!.reduce((s, r) => s.map((x, c) => x + r[c] / 2), [0, 0, 0])] : [])];
    pts.forEach((r) => ctr.add(new THREE.Vector3(...r)));
    ctr.divideScalar(pts.length);
    let R = 2;
    pos.forEach((r) => (R = Math.max(R, ctr.distanceTo(new THREE.Vector3(...r)) + 1.5)));
    if (this.cell) this.sys!.cell!.forEach((r) => (R = Math.max(R, new THREE.Vector3(...r).length() * 0.9)));
    const dist = R / Math.sin(THREE.MathUtils.degToRad(this.camera.fov / 2));
    this.controls.target.copy(ctr);
    this.camera.position.copy(ctr).add(new THREE.Vector3(0, 0, dist));
    this.camera.near = dist / 50; this.camera.far = dist * 10;
    this.camera.updateProjectionMatrix();
    this.dirty = true;
  }
}
