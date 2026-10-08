# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub>=0.25"]
# ///
"""Publish the built app (dist/) as a static Hugging Face Space.

    npm run build
    uv run scripts/deploy_space.py              # creates EricBoi/mlip-visualization if needed

Needs a token with write access (HF_TOKEN, or `hf auth login`). Without one it says so and
exits successfully, so the GitHub workflow can run before the secret is set. The weights are
not uploaded here: the app downloads them from the model repository (publish_weights.py).
"""
import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

README = """---
title: MLIP Visualization
emoji: ⚛️
colorFrom: yellow
colorTo: purple
sdk: static
app_file: index.html
pinned: false
license: bsd-3-clause
short_description: Interatomic potentials in the browser, every pass drawn
---

Machine-learning interatomic potentials running in your browser (WebGPU, with a CPU fallback),
with every forward and backward pass drawn. Source: https://github.com/EricBoittier/MLIP-Visualization
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="EricBoi/mlip-visualization")
    ap.add_argument("--dir", default="dist")
    args = ap.parse_args()

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    try:
        api.whoami()
    except Exception as e:  # no token, or a bad one: nothing to do yet
        print(f"No usable Hugging Face token ({type(e).__name__}); not deploying the Space.")
        return 0

    root = Path(args.dir)
    if not (root / "index.html").exists():
        sys.exit(f"{root}/index.html not found: run `npm run build` first")
    api.create_repo(args.space, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_file(path_or_fileobj=README.encode(), path_in_repo="README.md", repo_id=args.space, repo_type="space")
    # weights stay in the model repository; only the app goes to the Space
    api.upload_folder(folder_path=root, repo_id=args.space, repo_type="space",
                      ignore_patterns=["models/*"],
                      delete_patterns=["assets/*"], commit_message="Deploy the app")
    print(f"https://huggingface.co/spaces/{args.space}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
