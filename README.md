# MLIP-Visualization

Machine-learning interatomic potentials running in the browser, with every
forward and backward pass drawn the way Brendan Bycroft's
[LLM Visualization](https://bbycroft.net/llm) draws a transformer. Each op's
tensor is a block of cells holding its real values. Cells fill in as the forward
pass reaches them and change to gradient colours as the backward pass (which
gives the forces) flows back. [chemiscope](https://chemiscope.org) shows the
structure and the model's graph, coloured by whatever op is playing.

Models:

| model | family | status |
|---|---|---|
| PET (PET-MAD, PET-MOLS) | point edge transformer | runs, verified against metatrain |
| ANI-2x | Behler–Parrinello network | runs, verified against TorchANI |
| KRR with SOAP | kernel method, fitted in the browser to another model | runs; forces checked by finite differences |
| PhysNet | message-passing network | in progress |

MACE, SpookyNet and other equivariant models are left for later.

Everything runs locally: a small tape-based autograd engine in TypeScript with
a CPU backend and a WebGPU backend (WGSL kernels for every op, forward and
backward), in a Web Worker.

## Running

```bash
npm install
npm run dev
```

Model weights are not in the repository. Convert them into `public/models/`:

```bash
uv run scripts/convert_pet.py --model pet-mad-xs --out public/models/pet-mad-xs   # add --non-conservative for direct forces
uv run scripts/convert_ani.py --out public/models/ani-2x
```

KRR/SOAP needs no files: it is fitted in the browser, on rattled copies of the
current structure labelled by another loaded model (the "teacher").

`public/models/index.json` lists the models the app offers.

## Tests

```bash
npm test
```

They compare energies, forces and stresses with reference values from the
original implementations (`scripts/make_reference.py`), and the WebGPU kernels
with the CPU ones (through Node's `webgpu` package).

## Credits

ANI-2x is from [TorchANI](https://github.com/aiqm/torchani) (MIT).
`scripts/convert_pet.py` comes from
[pet-kokkos](https://github.com/EricBoittier/pet-kokkos). PET and PET-MAD are
from the [lab-cosmo/upet](https://huggingface.co/lab-cosmo/upet) models
(BSD-3-Clause).
