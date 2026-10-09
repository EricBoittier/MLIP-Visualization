# MLIP-Visualization

Machine-learning interatomic potentials running in the browser, with every
forward and backward pass drawn the way Brendan Bycroft's
[LLM Visualization](https://bbycroft.net/llm) draws a transformer. Each op's
tensor is a block of cells holding its real values. Cells fill in as the forward
pass reaches them and change to gradient colours as the backward pass (which
gives the forces) flows back. A structure view beside it shows the atoms, the
model's graph and the forces, coloured by whatever op is playing or by a property
picked from its menu (atomic energy, |F|, charge, attention).

Models:

| model | family | status |
|---|---|---|
| PET (PET-MAD, PET-MOLS) | point edge transformer | runs, verified against metatrain |
| ANI-2x | Behler–Parrinello network | runs, verified against TorchANI |
| KRR with SOAP | kernel method, fitted in the browser to another model | runs; forces checked by finite differences |
| PhysNet (mmml physnetjax, invariant) | message-passing network with charges and electrostatics | runs, verified against the JAX code; demo model trained on acetone dimers |
| MACE (MACE-MP-0) | equivariant message passing (ACE) | runs, verified against mace-torch |
| LOREM | equivariant message passing with a long-range Coulomb head | runs, verified against metatrain's experimental port; the shipped checkpoint is a small random initialization |

SpookyNet is left for later.

Everything runs locally: a small tape-based autograd engine in TypeScript with
a CPU backend and a WebGPU backend (WGSL kernels for every op, forward and
backward), in a Web Worker.

## Dynamics

`md.html` (the *MD & DMC* link in the header) simulates with the models that have production weights and
local interactions: PET, MACE and ANI-2x. PhysNet and LOREM ship demo weights, and KRR is fitted per
structure, so they stay on the visualiser. `?mode=dmc&model=<name>&structure=<preset>` picks the starting point.

- **NVE dynamics**: velocity Verlet from Maxwell–Boltzmann velocities with no net momentum. It plots kinetic,
  potential and total energy and the temperature. It can also use a model's direct force head, which is not
  the gradient of an energy, so you can watch the total energy drift.
- **Diffusion Monte Carlo**: the vibrational ground state of an isolated molecule. The structure is relaxed
  (FIRE) first, then an unguided DMC population is propagated (Anderson, discrete branching). The page
  reports the zero-point energy as the mean of E_ref − V_min over the second half of the run, with a blocking
  error, and draws the walker cloud aligned onto the minimum. It needs only energies, so every walker is
  evaluated in a few large batched passes. Copies of the molecule are laid out further apart than any
  model's cutoff, and each copy's energy is the sum of its atoms' energies. The pass size is 4096 atoms for PET and ANI-2x and 512 for
  MACE. It halves automatically if the GPU refuses a buffer, and `?maxAtoms=` overrides it. Walkers far below the minimum have found holes in the surface and are removed and counted.

Both report the FLOPs per step, estimated from the shapes of the kernels the model runs
(`src/engine/flops.ts`; a multiply-add counts as 2), the rate they run at, and ns/day (MD) or billion
samples/day (DMC).

## Running

```bash
npm install
npm run dev
```

Model weights are not in the repository. Convert them into `public/models/`:

```bash
uv run scripts/convert_pet.py --model pet-mad-xs --out public/models/pet-mad-xs   # add --non-conservative for direct forces
uv run scripts/convert_ani.py --out public/models/ani-2x
uv run scripts/convert_mace.py --model mace-mp-0b3-medium --out public/models/mace-mp-0b3-medium
# LOREM needs the metatrain checkout's lorem-tests env (the experimental port is not on PyPI):
#   <metatrain>/.tox/lorem-tests/bin/python scripts/convert_lorem.py --out public/models/lorem-demo
uv run scripts/convert_mace.py --model mace-mp-0b2-small --out public/models/mace-mp-0b2-small
```

PhysNet models come from mmml's `physnetjax` (invariant, `max_degree = 0`):

```bash
~/mmml-asv/.venv/bin/python scripts/train_physnet_demo.py --data ~/mmml-asv/examples/fixed-acetone-only_MP2_21000.npz --out acetone.params.json
uv run scripts/export_physnet.py --params acetone.params.json --out public/models/physnet-acetone --elements 1,6,8
```

KRR/SOAP needs no files: it is fitted in the browser, on rattled copies of the
current structure labelled by another loaded model (the "teacher"). Only models with production
weights (PET, MACE, ANI-2x) can be teachers, since PhysNet and LOREM ship demo weights.

`public/models/index.json` lists the models the app offers. When `public/models/` has no
weights (a deployed build), the app downloads them from the Hugging Face repository
[EricBoi/mlip-visualization-models](https://huggingface.co/EricBoi/mlip-visualization-models);
`?models=<base url>` points it at any other folder with an `index.json`. To publish the local
models there, with a model card giving each one's source and licence:

```bash
HF_TOKEN=hf_... uv run scripts/publish_weights.py   # --dry-run lists the files first
```

## Deployment

The app is a static site: `npm run build` writes it to `dist/`, and everything runs in the
visitor's browser. `.github/workflows/deploy.yml` builds it on every push to `main` and publishes it:

- **GitHub Pages**: https://ericboittier.github.io/MLIP-Visualization/ (Settings → Pages → Source: GitHub Actions).
- **Hugging Face Space** (static): https://huggingface.co/spaces/EricBoi/mlip-visualization, once the repository has an
  `HF_TOKEN` secret with write access (`gh secret set HF_TOKEN`). Without it the job skips itself.
  `uv run scripts/deploy_space.py` does the same from your machine after `npm run build`.

Neither deployment carries the weights. The deployed app downloads them from the public model repository
https://huggingface.co/EricBoi/mlip-visualization-models, which `scripts/publish_weights.py` creates and fills from
`public/models/` (log in first with `hf auth login`, then `uv run scripts/publish_weights.py --dry-run` to check).

## Tests

```bash
npm test
```

They compare energies, forces and stresses with reference values from the
original implementations (`scripts/make_reference.py`), and the WebGPU kernels
with the CPU ones (through Node's `webgpu` package).

## Credits

MACE-MP-0 is from [mace-foundations/mace-mp-0](https://huggingface.co/mace-foundations/mace-mp-0) (MIT).
LOREM follows [metatrain's experimental port](https://github.com/lab-cosmo/metatrain) of Bigi et al., arXiv:2507.19382;
the demo checkpoint is an untrained draw of that architecture.
ANI-2x is from [TorchANI](https://github.com/aiqm/torchani) (MIT).
`scripts/convert_pet.py` comes from
[pet-kokkos](https://github.com/EricBoittier/pet-kokkos). PET and PET-MAD are
from the [lab-cosmo/upet](https://huggingface.co/lab-cosmo/upet) models
(BSD-3-Clause).
