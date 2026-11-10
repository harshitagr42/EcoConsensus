"""Score every model that has outputs on the frozen splits and write one report. Sends no LLM requests, needs no GPU.

    python -m src.evaluate                                   # all splits, every system it can find
    python -m src.evaluate --splits val meta                 # leave the held-out test split out
    python -m src.evaluate --systems muril indicbert soft_vote

Systems (each is skipped with a warning when its outputs are missing):
  majority, tfidf_lr    models/finetuned/baselines/ (predicted here, so every split is covered)
  muril, indicbert      outputs/<model>/predictions/probs_<model>_<split>.csv (unzip <model>_results.zip in the root)
  llm_a, llm_b          the LLM cache (outputs/llm/cache/); unanswered rows lower the coverage, they are not guessed
  soft_vote             mean of the two transformers' probabilities
  majority_vote         hard vote of the available components (only when an LLM has answers)
  stacked               models/finetuned/meta/ with per-row fallback, rows with no variant fall back to tfidf_lr.
                        Not scored on `meta` (it was trained there).

Writes into outputs/evaluate/: logs/evaluate_<time>.txt, reports/evaluation_report_<time>.md,
metrics/evaluation_metrics_<time>.json + evaluation_summary_<time>.csv, predictions/evaluation_predictions_<split>_<time>.csv,
and an MLflow run in outputs/evaluate/metrics/mlflow.db.

`semeval_test` is the final held-out score: read it, never tune a prompt, a threshold or a model choice on it.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from src.baselines import Majority, load_tfidf, pair_text
from src.config import CFG, LABELS, ROOT, out_path, path
from src.data_loading import load_split
from src.ensembling import (COMPONENTS, LLMS, TRANSFORMERS, VARIANTS, add_llm, add_transformer, component_pred,
                            majority_vote, predict_with_fallback, soft_vote)
from src.llm_api import FatalProviderError, LLMComponent
from src.metrics import bootstrap_macro_f1, compute_metrics, confusion_md
from src.reporting import Report, md_table
from src.training.train_inference import local_data_hash, transformer_probs
from src.utils import env_info, get_logger, mlflow_start, safe_write, timestamp, to_jsonable

SPLITS = ["val", "meta", "semeval_test"]
SYSTEMS = ["majority", "tfidf_lr", *TRANSFORMERS, *LLMS, "soft_vote", "majority_vote", "stacked"]
SPLIT_NOTE = {
    "val": "Pol-NLI validation (early stopping of the transformers, prompt checks)",
    "meta": "held-out Pol-NLI slice the meta-model is trained on (unseen by the transformers)",
    "semeval_test": "SemEval test, final held-out score (never tune on it)",
}
KEEP = ["id", "source_id", "group_id", "language", "variant", "script_group", "label"]


# ---------------------------------------------------------------- loading every system's predictions
class Sources:
    """Loads each model once; `predict(split, df)` returns {system: pred ids, -1 where it has no answer}."""

    def __init__(self, systems: list[str], log, rep: Report):
        self.systems, self.log, self.rep = systems, log, rep
        self.found: dict[str, str] = {}
        self.baselines = self._load_baselines()
        self.llms = self._load_llms()
        self.meta_models = self._load_meta()

    def _skip(self, system: str, why: str) -> None:
        self.log.warning("%s skipped: %s", system, why)
        self.rep.warn(f"`{system}` not scored: {why}")

    def _load_baselines(self) -> dict:
        out = {}
        d = path("baselines")
        for name, f, loader in (("majority", "majority.json", Majority.load), ("tfidf_lr", "tfidf_lr.joblib", load_tfidf)):
            need = name in self.systems or (name == "tfidf_lr" and "stacked" in self.systems)
            if not need:
                continue
            if (d / f).exists():
                out[name] = loader(d / f)
                self.found[name] = str((d / f).relative_to(ROOT))
            elif name in self.systems:
                self._skip(name, f"{(d / f).relative_to(ROOT)} missing (run python -m src.baselines)")
        return out

    def _load_llms(self) -> dict:
        out = {}
        for c in LLMS:
            if c not in self.systems and not ({"majority_vote", "stacked"} & set(self.systems)):
                continue
            try:
                comp = LLMComponent(c[-1], lookup_only=True)
            except FatalProviderError as e:
                if c in self.systems:
                    self._skip(c, str(e))
                continue
            if not comp.cache.path.exists():
                if c in self.systems:
                    self._skip(c, f"no cache yet ({comp.cache.path.relative_to(ROOT)}); run run_llm_batch.py")
                continue
            out[c] = comp
            self.found[c] = f"{comp.cfg.provider}/{comp.cfg.model}, prompt {comp.prompt.version}"
        return out

    def _load_meta(self) -> dict:
        if "stacked" not in self.systems:
            return {}
        d = path("meta_models")
        fitted = {v: joblib.load(d / f"{v}.joblib") for v in VARIANTS if (d / f"{v}.joblib").exists()}
        if fitted:
            self.found["stacked"] = f"{d.relative_to(ROOT)} ({', '.join(fitted)})"
        else:
            self._skip("stacked", f"no meta-models in {d.relative_to(ROOT)} (run python -m src.training.train_meta)")
        return fitted

    def predict(self, split: str, df: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, str]]:
        preds, used = {}, {}
        for name, model in self.baselines.items():
            preds[name] = np.asarray(model.predict(pair_text(df)), dtype=int)

        table = df[KEEP].copy()                   # same feature table as train_inference.py
        for c in TRANSFORMERS:
            probs = None
            if c in self.systems or {"soft_vote", "majority_vote", "stacked"} & set(self.systems):
                try:
                    probs = transformer_probs(c, split, df, self.log)
                except ValueError as e:
                    self.log.warning("%s/%s ignored: %s", c, split, e)
                    self.rep.warn(f"`{c}` on `{split}` ignored: {e}")
            if probs is not None:
                self.found.setdefault(c, str(out_path(c, "predictions").relative_to(ROOT)))
            add_transformer(table, c, probs)
        for c in LLMS:
            if c in self.llms:
                res = self.llms[c].lookup(df["comment"].tolist())
                add_llm(table, c, [r["label"] if r["status"] == "ok" else None for r in res])
            else:
                add_llm(table, c, None)

        for c in COMPONENTS:
            p = component_pred(table, c)
            if (p >= 0).any():
                preds[c] = p
        if all(c in preds for c in TRANSFORMERS):
            preds["soft_vote"] = soft_vote(table)
        if any(c in preds for c in LLMS):
            preds["majority_vote"] = majority_vote(table)
        if self.meta_models and split != "meta":
            p, _, variant = predict_with_fallback(table, self.meta_models)
            if (p < 0).any() and "tfidf_lr" in preds:
                p = np.where(p >= 0, p, preds["tfidf_lr"])
                variant = np.where(variant == None, "tfidf_lr fallback", variant)  # noqa: E711
            preds["stacked"] = p
            used = pd.Series(variant).fillna("none").value_counts().to_dict()
        return {s: preds[s] for s in self.systems if s in preds}, used


# ---------------------------------------------------------------- scoring
def score(df: pd.DataFrame, pred: np.ndarray, bootstrap: bool) -> dict:
    ok = pred >= 0
    y, p = df["label"].to_numpy()[ok], pred[ok]
    m = compute_metrics(y, p, groups={"script_group": df["script_group"].to_numpy()[ok],
                                      "language": df["language"].to_numpy()[ok],
                                      "variant": (df["language"] + "/" + df["variant"]).to_numpy()[ok]})
    m["coverage"] = float(ok.mean())
    if bootstrap and ok.any():
        m["bootstrap"] = bootstrap_macro_f1(y, p, df["source_id"].to_numpy()[ok])
    return m


def summary_row(system: str, split: str, m: dict) -> dict:
    b = m.get("bootstrap")
    return dict(system=system, split=split, n=m["n"], coverage=m["coverage"], macro_f1=m["macro_f1"],
                ci95=f"{b['low']:.3f}-{b['high']:.3f}" if b else "", accuracy=m["accuracy"],
                weighted_f1=m["weighted_f1"], mcc=m["mcc"],
                **{f"f1_{c}": m["per_class"][c]["f1"] for c in LABELS})


def training_summary(log) -> tuple[list[dict], dict[str, list[dict]]]:
    """run_info.json of each transformer: settings, best epoch, the per-epoch val curve and the data-hash check."""
    rows, curves = [], {}
    data_hash = local_data_hash()
    for c in TRANSFORMERS:
        ri = ROOT / CFG["transformers"]["models"][c]["adapter_dir"] / "run_info.json"
        if not ri.exists():
            continue
        info = json.loads(ri.read_text(encoding="utf-8"))
        cfg = info.get("config", {})
        same = info.get("data_hash") == data_hash
        if not same:
            log.warning("%s was trained on data hash %s, local splits are %s", c, info.get("data_hash"), data_hash)
        rows.append(dict(model=c, base_model=info.get("base_model"), lora=f"r{cfg.get('r')} a{cfg.get('alpha')} "
                         f"{'+'.join(cfg.get('targets', []))}", lr=f"{cfg.get('lr', 0):g}", best_epoch=info.get("best_epoch"),
                         epochs_run=info.get("epochs_run"), train_min=round((info.get("train_time_sec") or 0) / 60, 1),
                         trainable_params=info.get("trainable_params"), data_hash_matches=same))
        curves[c] = [{**h, "lr": f"{h['lr']:.2e}"} if "lr" in h else h for h in info.get("history", [])]
    return rows, curves


def by_group_table(results: dict, split: str, key: str) -> str:
    """One row per system, one column per group value (Macro-F1)."""
    rows = []
    for (system, s), m in results.items():
        if s == split and f"by_{key}" in m:
            rows.append({"system": system, **{g: v["macro_f1"] for g, v in m[f"by_{key}"].items()}})
    return md_table(rows) if rows else "_(no rows)_"


def error_analysis(df: pd.DataFrame, preds: dict[str, np.ndarray]) -> str:
    """How often the components agree, and where every available system is wrong."""
    names = [s for s in preds if s in COMPONENTS or s == "tfidf_lr"]
    if len(names) < 2:
        return ""
    y = df["label"].to_numpy()
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = (preds[a] >= 0) & (preds[b] >= 0)
            if both.any():
                rows.append(dict(pair=f"{a} / {b}", rows=int(both.sum()),
                                 agree=float((preds[a][both] == preds[b][both]).mean()),
                                 both_wrong=float(((preds[a][both] != y[both]) & (preds[b][both] != y[both])).mean())))
    tr = [s for s in TRANSFORMERS if s in preds]
    text = md_table(rows)
    if tr:
        all_wrong = np.all([preds[s] != y for s in tr], axis=0)
        conf = pd.crosstab(pd.Series(y[all_wrong]).map(dict(enumerate(LABELS))).rename("true"),
                           pd.Series(preds[tr[0]][all_wrong]).map(dict(enumerate(LABELS))).rename(f"{tr[0]} said"))
        text += (f"\n\nRows every transformer gets wrong: **{int(all_wrong.sum())}** of {len(y)} "
                 f"({all_wrong.mean():.1%}).\n\n```\n{conf.to_string()}\n```")
    return text


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    ap.add_argument("--systems", nargs="+", default=SYSTEMS, choices=SYSTEMS)
    ap.add_argument("--no-bootstrap", action="store_true", help="skip the 95%% intervals (faster)")
    args = ap.parse_args(argv)
    log = get_logger("evaluate")
    ts = timestamp()

    rep = Report("Model evaluation")
    src = Sources(args.systems, log, rep)
    results: dict[tuple[str, str], dict] = {}
    summary, fallback, data_rows, details = [], {}, [], {}
    for split in args.splits:
        df = load_split(split)
        preds, used = src.predict(split, df)
        if used:
            fallback[split] = used
        dist = df["label"].value_counts().reindex(range(3), fill_value=0)
        data_rows.append(dict(split=split, rows=len(df), source_comments=df["source_id"].nunique(),
                              **{LABELS[i]: int(dist[i]) for i in range(3)}, role=SPLIT_NOTE[split]))
        out = df[KEEP].copy()
        for system, p in preds.items():
            m = score(df, p, not args.no_bootstrap)
            results[(system, split)] = m
            summary.append(summary_row(system, split, m))
            out[f"{system}_pred"] = p
            log.info("%-13s %-14s cov %5.1f%%  Macro-F1 %.4f  acc %.4f  F1 against %.4f", split, system,
                     m["coverage"] * 100, m["macro_f1"], m["accuracy"], m["per_class"]["against"]["f1"])
        details[split] = (df, preds)
        if preds:
            pp = safe_write(out_path("evaluate", "predictions", create=True) / f"evaluation_predictions_{split}_{ts}.csv",
                            out.to_csv(index=False), add_timestamp=False)
            log.info("predictions: %s", pp.relative_to(ROOT))

    if not results:
        log.error("Nothing to score: no model outputs were found.")
        return 3

    # ---- report
    rep.add("Header", "Every system with outputs on disk, scored on the frozen splits from `data/processed/`. "
                      "Macro-F1 is the primary metric; `ci95` is a bootstrap interval that resamples whole source "
                      "comments (all language versions together). `coverage` < 1 means some rows had no answer "
                      "(LLM not cached yet); those rows are left out, not counted as wrong.\n\n"
                      + "\n".join(f"- `{k}`: {v}" for k, v in src.found.items()))
    if "semeval_test" in args.splits:
        rep.add("Header", "`semeval_test` is the final held-out score. Do not change a prompt, a hyperparameter or "
                          "a model choice because of it.")
    rep.add("Data summary", md_table(data_rows))
    cols = ["system", "n", "coverage", "macro_f1", "ci95", "accuracy", "weighted_f1", "mcc",
            *[f"f1_{c}" for c in LABELS]]
    for split in args.splits:
        rows = sorted([r for r in summary if r["split"] == split], key=lambda r: -r["macro_f1"])
        if rows:
            full = [r for r in rows if r["coverage"] >= 0.999]
            best = (full or rows)[0]
            rep.add("System comparison", f"**{split}** ({SPLIT_NOTE[split]}). Best: **{best['system']}** "
                                         f"(Macro-F1 {best['macro_f1']:.4f})\n\n" + md_table(rows, cols))
    tr_rows, curves = training_summary(log)
    if tr_rows:
        rep.add("Training summary", md_table(tr_rows))
        for c, h in curves.items():
            rep.add("Training summary", f"**{c}** validation curve\n\n" + md_table(h))
        for r in tr_rows:
            if not r["data_hash_matches"]:
                rep.warn(f"{r['model']} was trained on a different data bundle than the local splits")
    for (system, split), m in results.items():
        rep.add("Per-class results", f"**{system} / {split}**\n\n" +
                md_table([{"class": k, **v} for k, v in m["per_class"].items()]))
        rep.add("Confusion matrices", f"**{system} / {split}**\n\n" + confusion_md(m["confusion"]))
    for split in args.splits:
        if any(s == split for _, s in results):
            rep.add("Results by script group", f"**{split}** (Macro-F1)\n\n" + by_group_table(results, split, "script_group"))
            rep.add("Results by language version", f"**{split}** (Macro-F1)\n\n" + by_group_table(results, split, "variant"))
    for split, (df, preds) in details.items():
        text = error_analysis(df, preds)
        if text:
            rep.add("Error analysis", f"**{split}**\n\n{text}")
    if fallback:
        rep.add("Fallback usage", "\n".join(f"- `{s}`: " + ", ".join(f"{k} {v}" for k, v in u.items())
                                            for s, u in fallback.items()))
    rep.add("Environment", md_table([env_info()]))
    rp = safe_write(out_path("evaluate", "reports", create=True) / f"evaluation_report_{ts}.md", rep.render(),
                    add_timestamp=False)
    log.info("report: %s", rp.relative_to(ROOT))

    mdir = out_path("evaluate", "metrics", create=True)
    mp = safe_write(mdir / f"evaluation_metrics_{ts}.json", json.dumps(to_jsonable(dict(
        created=datetime.now().isoformat(timespec="seconds"), splits=args.splits, sources=src.found,
        fallback=fallback, training=tr_rows, results={f"{s}/{sp}": m for (s, sp), m in results.items()})),
        indent=2, ensure_ascii=False), add_timestamp=False)
    cp = safe_write(mdir / f"evaluation_summary_{ts}.csv", pd.DataFrame(summary).to_csv(index=False), add_timestamp=False)
    log.info("metrics: %s, %s", mp.relative_to(ROOT), cp.relative_to(ROOT))

    run = mlflow_start("evaluate", "evaluation", f"evaluate__{ts}", log)
    if run:
        import mlflow
        with run:
            for (system, split), m in results.items():
                mlflow.log_metric(f"{system}_{split}_macro_f1", m["macro_f1"])
                mlflow.log_metric(f"{system}_{split}_coverage", m["coverage"])
            mlflow.log_artifact(str(rp))
            mlflow.log_artifact(str(cp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
