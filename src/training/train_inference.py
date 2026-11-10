"""Join the transformer probabilities (from the two Colab notebooks) and the cached LLM answers into one feature
table per split. Runs on the Mac; sends no LLM requests and needs no GPU.

    python -m src.training.train_inference                                   # meta, val, semeval_test; all 4
    python -m src.training.train_inference --splits meta --components muril indicbert llm_a

Inputs  outputs/<muril|indicbert>/predictions/probs_<model>_<split>.csv   (unzip <model>_results.zip in the project root)
        outputs/llm/cache/<provider>__<model>__<prompt_version>.jsonl  (filled by run_llm_batch.py)
Output  outputs/ensemble/predictions/features_<split>.csv  (12 features + <component>_available + ids/labels for reports)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # started as a file (python src/training/train_inference.py): make the project root importable
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from src.config import CFG, ROOT, out_path, path
from src.data_loading import load_split
from src.ensembling import COMPONENTS, LLMS, TRANSFORMERS, add_llm, add_transformer, available
from src.llm_api import FatalProviderError, LLMComponent
from src.utils import file_sha256, get_logger

KEEP = ["id", "source_id", "group_id", "dataset", "language", "variant", "script_group", "label"]


def local_data_hash() -> str:
    """Same formula as the notebooks' DATA_HASH, so a model trained on another bundle is detected."""
    d = path("processed")
    return hashlib.sha1("".join(file_sha256(d / f"{s}.csv") for s in
                                ["train_base", "meta", "val", "semeval_test"]).encode()).hexdigest()[:12]


def transformer_probs(comp: str, split: str, base: pd.DataFrame, log) -> np.ndarray | None:
    p = out_path(comp, "predictions") / f"probs_{comp}_{split}.csv"
    if not p.exists():
        log.warning("%s: %s missing (download %s_results.zip from Drive and unzip it in the project root)",
                    comp, p.relative_to(ROOT), comp)
        return None
    df = pd.read_csv(p, dtype={"id": str})
    if set(df["id"]) != set(base["id"]):
        raise ValueError(f"{p.name}: ids do not match data/processed/{split}.csv "
                         f"({len(set(base['id']) - set(df['id']))} missing, {len(set(df['id']) - set(base['id']))} extra). "
                         f"The model was trained on a different bundle: retrain or rebuild the splits.")
    df = df.set_index("id").loc[base["id"]]
    if (df["label"].to_numpy() != base["label"].to_numpy()).any():
        raise ValueError(f"{p.name}: labels differ from data/processed/{split}.csv")
    probs = df[[f"{comp}_p{j}" for j in range(3)]].to_numpy(dtype=float)
    bad = ~np.isclose(probs.sum(1), 1.0, atol=1e-3)
    if bad.any():
        log.warning("%s/%s: %d rows whose probabilities do not sum to 1", comp, split, int(bad.sum()))
    return probs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=["meta", "val", "semeval_test"])
    ap.add_argument("--components", nargs="+", default=COMPONENTS, choices=COMPONENTS)
    args = ap.parse_args(argv)
    log = get_logger("train_inference", "ensemble")

    data_hash = local_data_hash()
    for comp in [c for c in args.components if c in TRANSFORMERS]:
        ri = ROOT / CFG["transformers"]["models"][comp]["adapter_dir"] / "run_info.json"
        if ri.exists():
            info = json.loads(ri.read_text(encoding="utf-8"))
            if info.get("data_hash") != data_hash:
                log.warning("%s was trained on data hash %s, local splits have %s. Check that both Colab accounts "
                            "used the current ecoconsensus_data.zip.", comp, info.get("data_hash"), data_hash)
            else:
                log.info("%s: run %s, best epoch %s, val Macro-F1 %.4f", comp, info.get("run_name"),
                         info.get("best_epoch"), info.get("metrics", {}).get("val", {}).get("macro_f1", float("nan")))
        else:
            log.warning("%s: %s not found (adapter not downloaded yet?)", comp, ri.relative_to(ROOT))

    llms = {}
    for comp in [c for c in args.components if c in LLMS]:
        try:
            llms[comp] = LLMComponent(comp[-1], lookup_only=True)
            log.info("%s: cache %s (%d records)", comp, llms[comp].cache.path.name, len(llms[comp].cache.records))
        except FatalProviderError as e:
            log.warning("%s left out: %s", comp, e)

    out_dir = out_path("ensemble", "predictions", create=True)
    for split in args.splits:
        base = load_split(split)
        table = base[KEEP].copy()
        for comp in TRANSFORMERS:
            probs = transformer_probs(comp, split, base, log) if comp in args.components else None
            add_transformer(table, comp, probs)
        for comp in LLMS:
            if comp in llms:
                res = llms[comp].lookup(base["comment"].tolist())
                add_llm(table, comp, [r["label"] if r["status"] == "ok" else None for r in res])
            else:
                add_llm(table, comp, None)
        p = out_dir / f"features_{split}.csv"
        table.to_csv(p, index=False)
        cov = "  ".join(f"{c} {available(table, c).mean() * 100:5.1f}%" for c in COMPONENTS)
        log.info("%-13s %5d rows -> %s | available: %s", split, len(table), p.relative_to(ROOT), cov)
        for c in LLMS:
            if c in llms and available(table, c).mean() < 1:
                log.info("  %s: %d rows without an LLM answer (run run_llm_batch.py --provider %s --split %s)",
                         c, int((~available(table, c)).sum()), c[-1], split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
