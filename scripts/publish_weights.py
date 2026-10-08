# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub>=0.25"]
# ///
"""Upload the converted model files to a Hugging Face model repository, which the app
loads from when it has no local public/models/ (e.g. when deployed).

    export HF_TOKEN=hf_...            # a token with write access, or: huggingface-cli login
    uv run scripts/publish_weights.py --dry-run     # check what would be uploaded
    uv run scripts/publish_weights.py               # create EricBoi/mlip-visualization-models and upload

Uploads everything listed in public/models/index.json (each model's .json and
.safetensors), index.json itself, and a model card (README.md) with the source and
licence of every model.
"""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

CARD = """---
license: other
license_name: mixed
license_link: LICENSE.md
library_name: safetensors
tags: [interatomic-potential, chemistry, materials, webgpu]
---

# Models for MLIP-Visualization

Converted weights for [MLIP-Visualization](https://github.com/EricBoittier/MLIP-Visualization),
which runs machine-learning interatomic potentials in the browser and draws every forward and
backward pass. The app downloads these files directly; they are plain safetensors plus a JSON
file of hyperparameters per model.

| file | model | source | licence |
|---|---|---|---|
{rows}

`index.json` lists what the app offers.

## Credits

- **PET-MAD / PET-MOLS**: [lab-cosmo/upet](https://huggingface.co/lab-cosmo/upet) (BSD-3-Clause), converted with
  [pet-kokkos](https://github.com/EricBoittier/pet-kokkos)'s `convert_pet.py`. Cite the PET and PET-MAD papers when you use them.
- **MACE-MP-0**: [mace-foundations/mace-mp-0](https://huggingface.co/mace-foundations/mace-mp-0) (MIT), Batatia et al.,
  *A foundation model for atomistic materials chemistry* (2023), converted with `scripts/convert_mace.py`.
- **ANI-2x**: [TorchANI](https://github.com/aiqm/torchani) (MIT), Devereux et al., *J. Chem. Theory Comput.* 16, 4192 (2020),
  converted with `scripts/convert_ani.py`.
- **PhysNet (acetone)**: a small invariant PhysNet ([mmml](https://github.com/EricBoittier/mmml) `physnetjax`, MIT,
  max_degree = 0) trained for this app on mmml's acetone-dimer MP2 example data, with `scripts/train_physnet_demo.py`.
"""

LICENCES = """# Licences

Each model keeps the licence of its source:

- `pet-*`: BSD-3-Clause, from lab-cosmo/upet.
- `ani-2x.*`: MIT, from TorchANI (Copyright 2018- Xiang Gao and other ANI developers).
- `mace-*`: MIT, from mace-foundations/mace-mp-0 (MACE-MP-0, Batatia et al.).
- `physnet-*`: {physnet_licence}, trained with mmml (MIT, Copyright 2025 Eric Boittier).
"""

SOURCES = {
    "pet": ("lab-cosmo/upet, converted with convert_pet.py", "BSD-3-Clause"),
    "ani": ("TorchANI models.ANI2x(), converted with convert_ani.py", "MIT"),
    "physnet": ("mmml physnetjax, trained with train_physnet_demo.py", None),
    "mace": ("mace-foundations/mace-mp-0, converted with convert_mace.py", "MIT"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="EricBoi/mlip-visualization-models")
    ap.add_argument("--dir", default="public/models")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--physnet-licence", default="MIT", help="licence of the PhysNet demo model (mmml is MIT)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.dir)
    index = json.loads((root / "index.json").read_text())
    files, rows = ["index.json"], []
    for e in index:
        for key in ("meta", "weights"):
            if e.get(key):
                if not (root / e[key]).is_file():
                    raise SystemExit(f"{root / e[key]} is listed in index.json but missing")
                files.append(e[key])
        if e["kind"] in SOURCES:
            src, lic = SOURCES[e["kind"]]
            rows.append(f"| `{e['name']}` | {e.get('label', e['name'])} | {src} | {lic or args.physnet_licence} |")
    card = CARD.format(rows="\n".join(rows))
    licences = LICENCES.format(physnet_licence=args.physnet_licence)

    size = sum((root / f).stat().st_size for f in files)
    print(f"{args.repo}: {len(files)} files, {size / 1e6:.1f} MB")
    for f in files:
        print(f"  {f}")
    if args.dry_run:
        print("\n--- README.md ---\n" + card)
        return

    api = HfApi()
    print(f"as {api.whoami()['name']}")
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=args.repo)
    api.upload_file(path_or_fileobj=licences.encode(), path_in_repo="LICENSE.md", repo_id=args.repo)
    api.upload_folder(folder_path=str(root), repo_id=args.repo, allow_patterns=files,
                      commit_message="Update converted models")
    print(f"done: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
