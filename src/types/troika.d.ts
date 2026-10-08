declare module 'troika-three-text' {
  import type { Mesh } from 'three';
  export class Text extends Mesh {
    text: string;
    font: string;
    fontSize: number;
    color: string | number;
    anchorX: number | 'left' | 'center' | 'right';
    anchorY: number | 'top' | 'middle' | 'bottom';
    fillOpacity: number;
    maxWidth: number;
    sync(cb?: () => void): void;
    dispose(): void;
  }
}
