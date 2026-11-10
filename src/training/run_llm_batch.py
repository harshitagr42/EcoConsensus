"""Fill the LLM cache for one or more splits (runs on the Mac while Colab trains). Resumable: rerun the same
command and only uncached comments are sent. Run the two providers in two terminals to go twice as fast.

    python -m src.training.run_llm_batch --provider a --check                 # 3-comment smoke test
    python -m src.training.run_llm_batch --provider a --split val --estimate  # requests needed, nothing sent
    python -m src.training.run_llm_batch --provider a --pilot                 # batch-size pilot on 100 val comments
    python -m src.training.run_llm_batch --provider a --split val --score     # prompt check on val (zero/few-shot)
    python -m src.training.run_llm_batch --provider a --split meta val        # Day 3 cache fill
    python -m src.training.run_llm_batch --provider b --split semeval_test    # only after prompt_frozen: true

Cut-line: --variants english/original hindi/pure_hindi bengali/pure_bengali sends only those versions.
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import pandas as pd

from src.config import CFG, FIXED_CLAIM, LABELS, out_path
from src.data_loading import load_split
from src.llm_api import (STATUS_MISSING, STATUS_OK, FatalProviderError, LLMComponent, StopRun, cache_key,
                         default_batch_size)
from src.metrics import compute_metrics
from src.reporting import Report, md_table
from src.utils import get_logger, mlflow_start

LLM_CFG = CFG["llm"]
CHECK_COMMENTS = [
    "Global warming is a hoax invented to sell solar panels.",
    "जलवायु परिवर्तन एक गंभीर समस्या है, हमें अभी कदम उठाने होंगे।",
    "aaj match kaun jeetega?",
]


def select_rows(split: str, variants: list[str] | None) -> pd.DataFrame:
    df = load_split(split)
    if variants:
        lv = df["language"] + "/" + df["variant"]
        df = df[lv.isin(variants)]
    return df


def estimate(comp: LLMComponent, df: pd.DataFrame, batch_size: int) -> dict:
    keys = {cache_key(FIXED_CLAIM, c) for c in df["comment"]}
    todo = [k for k in keys if (r := comp.cache.get(k)) is None or r["status"] != STATUS_OK]
    req = math.ceil(len(todo) / batch_size * 1.1) if todo else 0
    return dict(rows=len(df), unique=len(keys), cached=len(keys) - len(todo), to_send=len(todo),
                est_requests=req, budget_left_today=comp.budget.remaining())


def coverage(comp: LLMComponent, df: pd.DataFrame) -> tuple[dict, list[dict]]:
    res = comp.lookup(df["comment"].tolist())
    st = pd.Series([r["status"] for r in res]).value_counts().to_dict()
    return dict(rows=len(df), ok=st.get(STATUS_OK, 0), unavailable=st.get("unavailable", 0),
                missing=st.get(STATUS_MISSING, 0)), res


def score(df: pd.DataFrame, res: list[dict]) -> dict | None:
    mask = np.array([r["status"] == STATUS_OK for r in res])
    if not mask.any():
        return None
    y = df["label"].to_numpy()[mask]
    p = np.array([r["label"] for r in res])[mask].astype(int)
    m = compute_metrics(y, p, groups={"script_group": df["script_group"].to_numpy()[mask]})
    m["coverage"] = float(mask.mean())
    return m


def _acc(preds, y, idx) -> float:
    idx = np.asarray(idx, dtype=int)
    return float((preds[idx] == y[idx]).mean()) if len(idx) else float("nan")


def run_pilot(comp: LLMComponent, log) -> list[dict]:
    """Same 100 val comments at several batch sizes (no cache): Macro-F1, missing ids, first-10 vs last-10 accuracy."""
    pcfg = LLM_CFG.get("pilot", {})
    val = load_split("val")
    n = int(pcfg.get("n_comments", 100))
    rng = np.random.default_rng(0)
    idx = []
    for _, g in val.groupby("label"):                   # stratified by label, mixed variants
        k = min(len(g), max(1, round(n * len(g) / len(val))))
        idx += list(rng.choice(g.index.to_numpy(), size=k, replace=False))
    sample = val.loc[rng.permutation(idx)].head(n)
    texts, y = sample["comment"].tolist(), sample["label"].to_numpy()
    rows = []
    for bs in pcfg.get("batch_sizes", [10, 25, 50, 100]):
        preds = np.full(len(texts), -1)
        missing = req = 0
        first, last = [], []
        try:
            for s in range(0, len(texts), bs):
                chunk = texts[s:s + bs]
                got = comp._ask(chunk)                 # one raw request, no repair: measures skipped ids
                req += 1
                missing += len(chunk) - len(got)
                for i, lab in got.items():
                    preds[s + i - 1] = lab
                idx = np.arange(s, s + len(chunk))
                first += list(idx[:10])
                last += list(idx[-10:])
        except StopRun as e:
            log.warning("pilot stopped at batch size %d: %s", bs, e)
            break
        ok = preds >= 0
        rows.append(dict(batch_size=bs, requests=req, missing_ids=missing,
                         macro_f1=compute_metrics(y[ok], preds[ok])["macro_f1"] if ok.any() else 0.0,
                         macro_f1_missing_as_wrong=compute_metrics(y, np.where(ok, preds, (y + 1) % 3))["macro_f1"],
                         acc_first10=_acc(preds, y, first), acc_last10=_acc(preds, y, last)))
        log.info("pilot bs=%-3d requests %d  missing ids %d  Macro-F1 %.3f  first10 %.2f  last10 %.2f",
                 bs, req, missing, rows[-1]["macro_f1"], rows[-1]["acc_first10"], rows[-1]["acc_last10"])
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", nargs="+", default=["a", "b"], choices=["a", "b"])
    ap.add_argument("--split", nargs="+", default=["meta"], choices=["val", "meta", "semeval_test", "train_base"])
    ap.add_argument("--batch-size", type=int, default=None, help="default: LLM_BATCH_SIZE / llm.yaml")
    ap.add_argument("--variants", nargs="+", default=None, help="language/variant subset (cut-line)")
    ap.add_argument("--max-requests", type=int, default=None, help="stop after this many requests in this run")
    ap.add_argument("--retry-unavailable", action="store_true", help="re-send comments cached as 'unavailable'")
    ap.add_argument("--estimate", action="store_true", help="print the request estimate and exit")
    ap.add_argument("--check", action="store_true", help="send 3 test comments and print the answers")
    ap.add_argument("--pilot", action="store_true", help="batch-size pilot on val (spends ~20 requests)")
    ap.add_argument("--score", action="store_true", help="only print metrics from the cache (no requests)")
    args = ap.parse_args(argv)
    log = get_logger("run_llm_batch", "llm")
    bs = args.batch_size or default_batch_size()

    if "semeval_test" in args.split and not (args.estimate or args.score) and not LLM_CFG.get("prompt_frozen"):
        log.error("Refusing to send semeval_test: set prompt_frozen: true in configs/llm.yaml first "
                  "(the prompt must be frozen before any test comment reaches an LLM).")
        return 2
    if "train_base" in args.split and not args.estimate:
        log.warning("train_base is not needed for stacking; only few-shot examples come from it.")

    rep = Report("LLM batch run")
    exit_code = 0
    for slot in args.provider:
        try:
            comp = LLMComponent(slot, batch_size=bs, max_requests=args.max_requests)
        except FatalProviderError as e:
            log.error("%s", e)
            exit_code = 4
            continue
        log.info("%s: %r  prompt %s  batch %d  cache %s", comp.name, comp.cfg, comp.prompt.version, comp.batch_size,
                 comp.cache.path.name)

        if args.check:
            try:
                res = comp.classify_batch([(f"check{i}", c) for i, c in enumerate(CHECK_COMMENTS)])
            except StopRun as e:
                log.error("%s: %s", comp.name, e)
                exit_code = 5
                continue
            for c, r in zip(CHECK_COMMENTS, res):
                log.info("  %-10s %s", LABELS[r["label"]] if r["label"] is not None else r["status"], c)
            continue

        if args.pilot:
            rows = run_pilot(comp, log)
            rep.add("Batch-size pilot", f"**{comp.name}** ({comp.cfg.provider} / {comp.cfg.model}), "
                                        f"prompt {comp.prompt.version}\n\n" + md_table(rows))
            rep.add("Batch-size pilot", "Pick the largest batch whose Macro-F1 is within noise of the smallest "
                                        "and that has no missing ids; set LLM_BATCH_SIZE in .env.")
            continue

        for split in args.split:
            df = select_rows(split, args.variants)
            est = estimate(comp, df, comp.batch_size)
            log.info("%s %-12s rows %d  unique %d  cached %d  to send %d  ~%d requests  budget left today %s",
                     comp.name, split, est["rows"], est["unique"], est["cached"], est["to_send"],
                     est["est_requests"], est["budget_left_today"])
            if not (args.estimate or args.score):
                comp.run(df["comment"].tolist(), retry_unavailable=args.retry_unavailable)
            cov, res = coverage(comp, df)
            log.info("%s %-12s coverage: ok %d  unavailable %d  missing %d", comp.name, split, cov["ok"],
                     cov["unavailable"], cov["missing"])
            row = dict(component=comp.name, split=split, **cov)
            m = score(df, res) if split != "semeval_test" else None      # test labels are never looked at here
            if m:
                row.update(macro_f1=m["macro_f1"], accuracy=m["accuracy"], f1_against=m["per_class"]["against"]["f1"])
                log.info("%s %-12s Macro-F1 %.4f on %d answered rows (acc %.3f, F1 against %.3f)", comp.name, split,
                         m["macro_f1"], m["n"], m["accuracy"], m["per_class"]["against"]["f1"])
            rep.add("LLM usage", md_table([row]))
            if split == "val" and m:                                # prompt comparison is tracked in MLflow
                run = mlflow_start("llm", "llm_prompts", f"{comp.name}__{comp.prompt.version}__val", log)
                if run:
                    import mlflow
                    with run:
                        mlflow.log_params(dict(provider=comp.cfg.provider, model=comp.cfg.model,
                                               prompt_version=comp.prompt.version, batch_size=comp.batch_size))
                        mlflow.log_metrics(dict(val_macro_f1=m["macro_f1"], val_accuracy=m["accuracy"],
                                                val_coverage=m["coverage"]))
            if comp.stopped:
                exit_code = exit_code or 1
                break
        rep.add("LLM usage", f"{comp.name} stats: {comp.stats}" + (f" — stopped: {comp.stopped}" if comp.stopped else ""))

    if rep.sections and not (args.estimate or args.check):
        rp = rep.save(out_path("llm", "reports", create=True) / ("llm_pilot.md" if args.pilot else "llm_batch.md"))
        log.info("report: %s", rp)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
