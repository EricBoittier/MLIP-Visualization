Converted model files go here (they are not in git). `index.json` lists what the app offers.

    uv run scripts/convert_pet.py --model pet-mad-xs --out public/models/pet-mad-xs
    uv run scripts/convert_ani.py --out public/models/ani-2x
    uv run scripts/convert_mace.py --model mace-mp-0b3-medium --out public/models/mace-mp-0b3-medium

Without these files the app loads the published copies from
https://huggingface.co/EricBoi/mlip-visualization-models (see `scripts/publish_weights.py`).
