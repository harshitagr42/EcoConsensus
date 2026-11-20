"""Model files for the Space: the fine-tuned weights (LoRA adapters, meta-models, baselines) come from a separate
Hugging Face model repo into models/finetuned/; the base models come from the Hub cache.

    python -m api.weights        # download both and load each transformer once (run while the Docker image is built)
"""
from __future__ import annotations

import logging
import sys

from api import settings

log = logging.getLogger("api.weights")


def missing_files() -> list[str]:
    from src.ensembling import LLMS, TRANSFORMERS
    from src.pipeline import check_model_files
    return check_model_files(TRANSFORMERS + LLMS, need_meta=True)


def ensure_weights() -> None:
    """Download the weights repo into models/finetuned/ when files are missing. Raises with a readable message."""
    if not missing_files():
        return
    if not settings.WEIGHTS_REPO:
        raise RuntimeError("Model files missing and WEIGHTS_REPO is not set (Space variable or api/deploy.json)")
    from huggingface_hub import snapshot_download

    from src.config import path
    log.info("downloading %s -> %s", settings.WEIGHTS_REPO, path("finetuned"))
    snapshot_download(repo_id=settings.WEIGHTS_REPO, repo_type="model", local_dir=path("finetuned"))
    still = missing_files()
    if still:
        raise RuntimeError(f"{settings.WEIGHTS_REPO} does not contain: {', '.join(still)}")


def warm_base_models() -> None:
    """Load each transformer once so the base model is in the Hub cache (no download during the first job)."""
    from src.ensembling import TRANSFORMERS
    from src.pipeline import TransformerComponent
    for name in TRANSFORMERS:
        log.info("loading %s once", name)
        TransformerComponent(name, settings.DEVICE).load().unload()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.path.insert(0, str(settings.ROOT))
    ensure_weights()
    warm_base_models()
    print("model files ready")
