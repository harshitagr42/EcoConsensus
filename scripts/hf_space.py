"""Everything on Hugging Face, from the project root (log in once: `hf auth login`, or HF_TOKEN in .env).

    python scripts/hf_space.py weights     # models/finetuned/ -> public model repo <you>/ecoconsensus-weights
    python scripts/hf_space.py deploy      # api/ + src/ + configs/ + prompts/ + inference.py -> Space <you>/ecoconsensus-api
    python scripts/hf_space.py secrets     # LLM settings from .env -> Space variables, API keys -> Space secrets
    python scripts/hf_space.py status      # Space state and its URLs
    python scripts/hf_space.py pause       # stop the backend (keys unused, the website shows "offline")
    python scripts/hf_space.py restart     # start it again

Only the files listed in BUNDLE are uploaded to the Space; nothing else in the repository can reach it. The weights
never go to GitHub or to the Space repo: the Space downloads them from the model repo while its image is built.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Space repo layout: (source in this repo, destination in the Space). Directories are copied without caches.
BUNDLE = [("api/Dockerfile", "Dockerfile"), ("api/README.md", "README.md"), ("api/requirements.txt", "requirements.txt"),
          ("api", "api"), ("src", "src"), ("configs", "configs"), ("prompts", "prompts"),
          ("inference.py", "inference.py")]
SKIP_DIRS = {"__pycache__", "training", ".pytest_cache"}          # src/training/ is not needed to serve
WEIGHT_DIRS = ["muril_lora", "indicbert_lora", "meta", "baselines"]
VARIABLES = ["PROVIDER", "MODEL", "BASE_URL", "DAILY_LIMIT", "EXTRA_BODY"]
DEFAULT_ORIGIN = "https://ecoconsensus.harshitagr42.com"


def hf():
    from huggingface_hub import HfApi

    from src.config import load_env
    load_env()                                   # HF_TOKEN from .env, if set
    return HfApi()


def names(api, args) -> tuple[str, str]:
    owner = args.owner or api.whoami()["name"]
    return f"{owner}/{args.space}", f"{owner}/{args.weights}"


def space_urls(space_id: str) -> tuple[str, str]:
    sub = space_id.lower().replace("/", "-").replace("_", "-").replace(".", "-")
    return f"https://huggingface.co/spaces/{space_id}", f"https://{sub}.hf.space"


def model_card(weights_id: str) -> str:
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
this repo. Used by the EcoConsensus API Space, which downloads `{weights_id}` while its image is built.
"""


def cmd_weights(api, args) -> None:
    from src.config import path
    _, weights_id = names(api, args)
    src = path("finetuned")
    missing = [d for d in WEIGHT_DIRS if not (src / d).is_dir()]
    if missing:
        sys.exit(f"missing in {src}: {missing}")
    api.create_repo(weights_id, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(repo_id=weights_id, repo_type="model", folder_path=src,
                      allow_patterns=[f"{d}/*" for d in WEIGHT_DIRS], ignore_patterns=["**/.DS_Store", ".DS_Store"],
                      commit_message="Upload EcoConsensus weights")
    api.upload_file(repo_id=weights_id, repo_type="model", path_or_fileobj=model_card(weights_id).encode(),
                    path_in_repo="README.md", commit_message="Model card")
    print(f"weights uploaded: https://huggingface.co/{weights_id}")


def build_bundle(dest: Path, weights_id: str) -> list[str]:
    for src_rel, dst_rel in BUNDLE:
        s, d = ROOT / src_rel, dest / dst_rel
        if s.is_dir():
            shutil.copytree(s, d, dirs_exist_ok=True,
                            ignore=lambda _dir, names: [n for n in names if n in SKIP_DIRS or n.endswith(".pyc")
                                                        or n == ".DS_Store"])
        else:
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
    (dest / "api" / "deploy.json").write_text(json.dumps({"weights_repo": weights_id}, indent=2) + "\n")
    return sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file())


def cmd_deploy(api, args) -> None:
    space_id, weights_id = names(api, args)
    with tempfile.TemporaryDirectory() as tmp:
        files = build_bundle(Path(tmp), weights_id)
        print(f"{len(files)} files:\n  " + "\n  ".join(files))
        if args.dry_run:
            return
        api.create_repo(space_id, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
        api.upload_folder(repo_id=space_id, repo_type="space", folder_path=tmp, delete_patterns=["*"],
                          commit_message="Deploy EcoConsensus API")
    page, url = space_urls(space_id)
    print(f"deployed; the Space now builds (~5-10 min). Page: {page}\nAPI URL for web/config.js: {url}")


def cmd_secrets(api, args) -> None:
    from src.config import env
    from src.llm_api import api_keys
    space_id, _ = names(api, args)
    done = []
    for s in "AB":
        for k in VARIABLES:
            v = env(f"LLM_{s}_{k}")
            if v:
                api.add_space_variable(space_id, f"LLM_{s}_{k}", v)
                done.append(f"LLM_{s}_{k}")
        keys = api_keys(s.lower())
        if not keys:
            print(f"WARNING: no LLM_{s}_API_KEYS / LLM_{s}_API_KEY in .env; LLM-{s} will be unavailable")
            continue
        api.add_space_secret(space_id, f"LLM_{s}_API_KEYS", ",".join(keys))
        done.append(f"LLM_{s}_API_KEYS (secret, {len(keys)} key(s))")
    if env("LLM_BATCH_SIZE"):
        api.add_space_variable(space_id, "LLM_BATCH_SIZE", env("LLM_BATCH_SIZE"))
        done.append("LLM_BATCH_SIZE")
    api.add_space_variable(space_id, "ALLOWED_ORIGINS", args.origin)
    done.append(f"ALLOWED_ORIGINS={args.origin}")
    print("set on " + space_id + ":\n  " + "\n  ".join(done) + "\nThe Space restarts to pick them up.")


def cmd_status(api, args) -> None:
    space_id, weights_id = names(api, args)
    page, url = space_urls(space_id)
    try:
        rt = api.get_space_runtime(space_id)
        print(f"Space {space_id}: {rt.stage} (hardware {rt.hardware})")
    except Exception as e:  # noqa: BLE001
        print(f"Space {space_id}: not found ({type(e).__name__})")
    print(f"Page: {page}\nAPI : {url}  (health: {url}/health)\nWeights: https://huggingface.co/{weights_id}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["weights", "deploy", "secrets", "status", "pause", "restart"])
    ap.add_argument("--owner", help="HF user or organisation (default: the logged-in user)")
    ap.add_argument("--space", default="ecoconsensus-api", help="Space name")
    ap.add_argument("--weights", default="ecoconsensus-weights", help="model repo name for the weights")
    ap.add_argument("--origin", default=DEFAULT_ORIGIN, help="website allowed to call the API (secrets)")
    ap.add_argument("--dry-run", action="store_true", help="deploy: list the files, upload nothing")
    args = ap.parse_args()
    if args.command == "deploy" and args.dry_run:
        with tempfile.TemporaryDirectory() as tmp:
            files = build_bundle(Path(tmp), f"{args.owner or '<you>'}/{args.weights}")
            print(f"{len(files)} files would be uploaded:\n  " + "\n  ".join(files))
        return
    api = hf()
    if args.command in ("pause", "restart"):
        space_id, _ = names(api, args)
        (api.pause_space if args.command == "pause" else api.restart_space)(space_id)
        print(f"{args.command}d {space_id}")
        return
    {"weights": cmd_weights, "deploy": cmd_deploy, "secrets": cmd_secrets, "status": cmd_status}[args.command](api, args)


if __name__ == "__main__":
    main()
