"""Upload the fine-tuned weights to a public Hugging Face model repo (log in once: `hf auth login`).

    python scripts/hf_weights.py           # models/finetuned/ -> <you>/ecoconsensus-weights
    python scripts/hf_weights.py --name    # only print the repo id (for WEIGHTS_REPO in the server's .env)

Only muril_lora/, indicbert_lora/, meta/ and baselines/ are uploaded. The weights never go to GitHub; the API server
downloads them from this repo at startup.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

WEIGHT_DIRS = ["muril_lora", "indicbert_lora", "meta", "baselines"]


def model_card(repo_id: str) -> str:
    from src.config import CFG
    tm = CFG["transformers"]["models"]
    return f"""---
library_name: peft
base_model:
- {tm['muril']['hf_name']}
- {tm['indicbert']['hf_name']}
tags: [stance-detection, hindi, bengali, lora, climate-change]
---

# EcoConsensus weights

Fine-tuned parts of the EcoConsensus stance-detection ensemble (claim: *"Climate change and global warming is a serious
concern."*; labels 0 favour, 1 neutral, 2 against):

- `muril_lora/`: LoRA adapter + trained classifier for `{tm['muril']['hf_name']}`
- `indicbert_lora/`: LoRA adapter + trained classifier for `{tm['indicbert']['hf_name']}`
- `meta/`: logistic-regression meta-models (scikit-learn joblib) over both transformers and two LLMs
- `baselines/`: majority and TF-IDF + logistic-regression baselines

Input: the (claim, comment) sentence pair, max_length 128. The joblib files are pickles: load them only if you trust
this repo. The EcoConsensus API downloads `{repo_id}` at startup.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="ecoconsensus-weights", help="model repo name")
    ap.add_argument("--owner", help="HF user or organisation (default: the logged-in user)")
    ap.add_argument("--name", action="store_true", help="print the repo id and stop")
    args = ap.parse_args()
    from huggingface_hub import HfApi

    from src.config import load_env, path
    load_env()
    api = HfApi()
    repo_id = f"{args.owner or api.whoami()['name']}/{args.repo}"
    if args.name:
        print(repo_id)
        return
    src = path("finetuned")
    missing = [d for d in WEIGHT_DIRS if not (src / d).is_dir()]
    if missing:
        sys.exit(f"missing in {src}: {missing}")
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=src,
                      allow_patterns=[f"{d}/*" for d in WEIGHT_DIRS], ignore_patterns=["**/.DS_Store", ".DS_Store"],
                      commit_message="Upload EcoConsensus weights")
    api.upload_file(repo_id=repo_id, repo_type="model", path_or_fileobj=model_card(repo_id).encode(),
                    path_in_repo="README.md", commit_message="Model card")
    print(f"weights: https://huggingface.co/{repo_id}\nput this line in the server's .env:  WEIGHTS_REPO={repo_id}")


if __name__ == "__main__":
    main()
