import { describe, expect, it } from 'vitest';
import { CpuBackend } from '../src/engine/cpu';
import { countFlops } from '../src/engine/flops';

describe('FLOP counter', () => {
  it('counts kernels from their shapes and leaves results unchanged', async () => {
    const raw = new CpuBackend(), c = countFlops(raw), be = c.be;
    expect(be.name).toBe('cpu');
    const [M, N, K] = [2, 3, 4];
    const A = be.upload(Float32Array.from({ length: M * K }, (_, i) => i)), B = be.upload(Float32Array.from({ length: K * N }, (_, i) => 1 - i));
    const C = be.zeros(M * N), D = raw.zeros(M * N);
    be.matmul(A, B, C, M, N, K, false, false, false);
    raw.matmul(A, B, D, M, N, K, false, false, false);
    expect(await be.read(C)).toEqual(await raw.read(D));
    expect(c.total).toBe(2 * M * N * K);
    be.unary('exp', C, D, M * N, 0, 0);
    be.fill(D, 0, M * N); // memory only
    expect(c.total).toBe(2 * M * N * K + M * N);
    expect(c.byKernel.get('matmul')).toBe(2 * M * N * K);
    c.reset();
    expect(c.total).toBe(0);
  });
});
