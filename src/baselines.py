"""Baselines: majority class and TF-IDF (char + word n-grams) + logistic regression.

    python -m src.baselines              # train on train_base, pick C on val, score val and semeval_test

Saves models/finetuned/baselines/{majority.json, tfidf_lr.joblib}, per-split predictions in outputs/baselines/predictions/,
a timestamped report in outputs/baselines/reports/ and an MLflow run (outputs/baselines/metrics/mlflow.db).
"""
from __future__ import annotations

import argparse
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

if __package__ in (None, ""):  # started as a file (python src/baselines.py): make the project root importable
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from src.config import LABELS, SEED, out_path, path
from src.data_loading import load_split
from src.metrics import bootstrap_macro_f1, compute_metrics, confusion_md
from src.reporting import Report, md_table
from src.utils import env_info, get_logger, mlflow_start, set_seed, write_json

SEP = " [SEP] "
C_GRID = [0.1, 0.3, 1.0, 3.0, 10.0]


def pair_text(df: pd.DataFrame) -> list[str]:
    """TF-IDF input: claim + separator + comment (lower-cased by the vectorizer)."""
    return (df["claim"].astype(str) + SEP + df["comment"].astype(str)).tolist()


def make_tfidf_lr(C: float, seed: int = SEED) -> Pipeline:
    feats = FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, lowercase=True)),
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True,
                                 token_pattern=r"(?u)\b\w+\b")),
    ])
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=seed)
    return Pipeline([("features", feats), ("clf", clf)])


class Majority:
    def __init__(self, label: int):
        self.label = int(label)

    def predict(self, texts) -> np.ndarray:
        return np.full(len(texts), self.label, dtype=int)

    def predict_proba(self, texts) -> np.ndarray:
        p = np.zeros((len(texts), 3))
        p[:, self.label] = 1.0
        return p

    @classmethod
    def load(cls, p) -> "Majority":
        return cls(json.loads(open(p, encoding="utf-8").read())["label"])


def load_tfidf(p=None) -> Pipeline:
    """Only load joblib files you created yourself (they are pickles)."""
    return joblib.load(p or path("baselines") / "tfidf_lr.joblib")


def _metrics(df, pred):
    return compute_metrics(df["label"], pred, groups={"script_group": df["script_group"],
                                                      "variant": df["language"] + "/" + df["variant"]})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-splits", nargs="+", default=["val", "semeval_test"])
    ap.add_argument("--no-bootstrap", action="store_true")
    args = ap.parse_args(argv)
    log = get_logger("baselines")
    set_seed(SEED)

    train, val = load_split("train_base"), load_split("val")
    evals = {s: load_split(s) for s in args.eval_splits}
    out_dir = path("baselines", create=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_path("baselines", "predictions", create=True)

    # majority
    maj_label = int(train["label"].value_counts().idxmax())
    majority = Majority(maj_label)
    write_json(out_dir / "majority.json", {"label": maj_label, "label_name": LABELS[maj_label],
                                           "trained_on": "train_base"})

    # TF-IDF + LR: C chosen on val Macro-F1
    search = []
    for C in C_GRID:
        m = make_tfidf_lr(C).fit(pair_text(train), train["label"])
        f1 = compute_metrics(val["label"], m.predict(pair_text(val)))["macro_f1"]
        search.append({"C": C, "val_macro_f1": f1})
        log.info("TF-IDF C=%-5g val Macro-F1 %.4f", C, f1)
    best_C = max(search, key=lambda r: r["val_macro_f1"])["C"]
    tfidf = make_tfidf_lr(best_C).fit(pair_text(train), train["label"])
    joblib.dump(tfidf, out_dir / "tfidf_lr.joblib")
    log.info("chosen C=%g; saved %s", best_C, out_dir / "tfidf_lr.joblib")

    results, rows = {}, []
    for split, df in evals.items():
        for name, model in (("majority", majority), ("tfidf_lr", tfidf)):
            texts = pair_text(df)
            pred = model.predict(texts)
            proba = model.predict_proba(texts)
            m = _metrics(df, pred)
            if not args.no_bootstrap:
                m["bootstrap"] = bootstrap_macro_f1(df["label"], pred, df["source_id"])
            results[(name, split)] = m
            rows.append({"model": name, "split": split, "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                         "f1_against": m["per_class"]["against"]["f1"],
                         "ci95": f"{m['bootstrap']['low']:.3f}-{m['bootstrap']['high']:.3f}" if "bootstrap" in m else ""})
            pf = df[["id", "source_id", "language", "variant", "script_group", "label"]].copy()
            pf["pred"] = pred
            for j in range(3):
                pf[f"{name}_p{j}"] = proba[:, j].round(5)
            pf.to_csv(pred_dir / f"baseline_{name}_{split}.csv", index=False)
            log.info("%-9s %-13s Macro-F1 %.4f  acc %.4f  F1(against) %.4f", name, split, m["macro_f1"],
                     m["accuracy"], m["per_class"]["against"]["f1"])

    rep = Report("Baselines")
    rep.add("Header", f"Trained on `train_base` ({len(train)} rows). Majority label: **{LABELS[maj_label]}**. "
                      f"TF-IDF + LR `C` chosen on `val`: **{best_C}**.")
    rep.add("System comparison", md_table(rows))
    rep.add("System comparison", "C search on val:\n\n" + md_table(search))
    for (name, split), m in results.items():
        rep.add("Per-class results", f"**{name} / {split}**\n\n" + md_table(
            [{"class": k, **v} for k, v in m["per_class"].items()]))
        rep.add("Confusion matrices", f"**{name} / {split}**\n\n" + confusion_md(m["confusion"]))
        rep.add("Results by script group", f"**{name} / {split}**\n\n" + md_table(
            [{"group": k, **v} for k, v in m["by_script_group"].items()]))
    rep.add("Environment", md_table([env_info()]))
    rp = rep.save(out_path("baselines", "reports", create=True) / "baselines_report.md")
    log.info("report: %s", rp)

    run = mlflow_start("baselines", "baselines", "baselines__tfidf_lr", log)
    if run:
        import mlflow
        with run:
            mlflow.log_params({"C": best_C, "majority_label": LABELS[maj_label], "train_rows": len(train)})
            for (name, split), m in results.items():
                mlflow.log_metric(f"{name}_{split}_macro_f1", m["macro_f1"])
                mlflow.log_metric(f"{name}_{split}_accuracy", m["accuracy"])
            mlflow.log_artifact(str(rp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
