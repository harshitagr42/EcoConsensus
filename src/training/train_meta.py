"""Fit the stacking meta-model and its fallback variants on the `meta` feature table (CPU, runs on the Mac).

    python -m src.training.train_meta

For each variant (meta_all4, meta_no_llm_a, meta_no_llm_b, meta_transformers):
  - rows of `meta` where all its components answered,
  - C chosen by 5-fold cross-validation grouped by source comment (all 7 versions in one fold), stratified by label,
  - final fit on all those rows -> models/finetuned/meta/<variant>.joblib + meta_info.json.
Then it compares, on `meta` (out-of-fold for the meta-models) and on `val`: every single component, best single
transformer, best single LLM, majority vote of the four, soft vote of the two transformers and the meta-models
(with per-row fallback). `semeval_test` is never touched here. Weights are printed and sanity-checked.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import LABELS, ROOT, SEED, path
from src.data_loading import load_split
from src.ensembling import (COMPONENTS, LLMS, TRANSFORMERS, VARIANTS, available, component_pred, majority_vote,
                            make_meta_model, predict_with_fallback, rows_for_variant, soft_vote, variant_cols,
                            weight_sanity, weights_table)
from src.metrics import bootstrap_macro_f1, compute_metrics, confusion_md, macro_f1
from src.reporting import Report, md_table
from src.utils import env_info, get_logger, mlflow_start, set_seed, write_json

C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
MIN_ROWS = 60


def load_features(split: str) -> pd.DataFrame:
    p = path("predictions") / f"features_{split}.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing. Run: python -m src.training.train_inference --splits {split}")
    return pd.read_csv(p, dtype={"id": str, "source_id": str, "group_id": str})


def leakage_check(meta: pd.DataFrame) -> None:
    """meta rows must be exactly meta-split rows; none of their comments may be in val or semeval_test."""
    meta_ids = set(load_split("meta")["id"])
    extra = set(meta["id"]) - meta_ids
    if extra:
        raise ValueError(f"{len(extra)} rows of features_meta.csv are not in the meta split, e.g. {sorted(extra)[:3]}")
    for other in ("val", "semeval_test", "train_base"):
        o = load_split(other)
        both = set(meta["source_id"]) & set(o["source_id"])
        if both:
            raise ValueError(f"{len(both)} meta source ids also in {other}: stacking would leak")


def cv_select_C(X, y, groups, n_splits: int, seed: int):
    """Grouped stratified CV. Returns (best C, table of mean Macro-F1 per C, out-of-fold predictions for best C)."""
    n_splits = min(n_splits, int(np.bincount(y, minlength=3)[np.bincount(y, minlength=3) > 0].min()),
                   len(np.unique(groups)))
    cv = StratifiedGroupKFold(n_splits=max(2, n_splits), shuffle=True, random_state=seed)
    folds = list(cv.split(X, y, groups))
    rows, oof_by_C = [], {}
    for C in C_GRID:
        oof = np.full(len(y), -1)
        scores = []
        for tr, te in folds:
            m = make_meta_model(C, seed).fit(X[tr], y[tr])
            oof[te] = m.predict(X[te])
            scores.append(macro_f1(y[te], oof[te]))
        rows.append(dict(C=C, cv_macro_f1_mean=float(np.mean(scores)), cv_macro_f1_std=float(np.std(scores)),
                         oof_macro_f1=macro_f1(y, oof)))
        oof_by_C[C] = oof
    best = max(rows, key=lambda r: (round(r["cv_macro_f1_mean"], 4), -r["C"]))   # ties -> stronger regularisation
    return best["C"], rows, oof_by_C[best["C"]], len(folds)


def score_row(name, y, pred, extra=None):
    ok = pred >= 0
    m = compute_metrics(y[ok], pred[ok]) if ok.any() else None
    r = dict(system=name, coverage=float(ok.mean()), macro_f1=m["macro_f1"] if m else None,
             accuracy=m["accuracy"] if m else None, f1_against=m["per_class"]["against"]["f1"] if m else None)
    r.update(extra or {})
    return r, m


def compare(table: pd.DataFrame, meta_preds: dict[str, np.ndarray]) -> list[dict]:
    """Single components, voting and meta predictions on one table. Coverage < 1 means some rows had no answer."""
    y = table["label"].to_numpy()
    rows = []
    singles = {c: component_pred(table, c) for c in COMPONENTS}
    for c, p in singles.items():
        rows.append(score_row(c, y, p)[0])
    scored = {r["system"]: r for r in rows}
    for group, label in ((TRANSFORMERS, "best single transformer"), (LLMS, "best single LLM")):
        cands = [scored[c] for c in group if scored[c]["macro_f1"] is not None]
        if cands:
            b = max(cands, key=lambda r: r["macro_f1"])
            rows.append({**b, "system": f"{label} ({b['system']})"})
    rows.append(score_row("soft vote (2 transformers)", y, soft_vote(table))[0])
    rows.append(score_row("majority vote (4 components)", y, majority_vote(table))[0])
    for name, p in meta_preds.items():
        rows.append(score_row(name, y, p)[0])
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    args = ap.parse_args(argv)
    log = get_logger("train_meta")
    set_seed(args.seed)

    meta = load_features("meta")
    leakage_check(meta)
    val = load_features("val")
    y_meta = meta["label"].to_numpy()
    for c in COMPONENTS:
        log.info("meta availability %-9s %5.1f%%   val %5.1f%%", c, available(meta, c).mean() * 100,
                 available(val, c).mean() * 100)

    out_dir = path("meta_models", create=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = Report("Meta-model (stacking) training")
    fitted, info, oof_full, cv_tables = {}, {}, {}, {}
    for v in args.variants:
        mask = rows_for_variant(meta, v)
        if mask.sum() < MIN_ROWS:
            log.warning("%s skipped: only %d meta rows have all of %s", v, int(mask.sum()), VARIANTS[v])
            rep.warn(f"{v} not trained: {int(mask.sum())} usable meta rows")
            continue
        cols = variant_cols(v)
        X = meta.loc[mask, cols].to_numpy(dtype=float)
        y = y_meta[mask]
        groups = meta.loc[mask, "group_id"].to_numpy()
        C, table, oof, n_folds = cv_select_C(X, y, groups, args.folds, args.seed)
        model = make_meta_model(C, args.seed).fit(X, y)
        joblib.dump(model, out_dir / f"{v}.joblib")
        fitted[v] = model
        full = np.full(len(meta), -1)
        full[mask] = oof
        oof_full[v] = full
        cv_tables[v] = table
        issues = weight_sanity(model, cols)
        info[v] = dict(components=VARIANTS[v], features=cols, C=C, n_rows=int(mask.sum()),
                       n_source_rows=int(meta.loc[mask, "source_id"].nunique()), folds=n_folds,
                       cv_macro_f1=max(r["cv_macro_f1_mean"] for r in table), oof_macro_f1=macro_f1(y, oof),
                       weight_issues=issues, file=f"{v}.joblib")
        log.info("%-18s rows %5d  C=%-5g  CV Macro-F1 %.4f  (OOF %.4f)", v, mask.sum(), C,
                 info[v]["cv_macro_f1"], info[v]["oof_macro_f1"])
        w = weights_table(model, cols)
        log.info("%s weights:\n%s", v, w.round(3).to_string())
        for i in issues:
            log.warning("%s: %s", v, i)

    if not fitted:
        log.error("No meta-model could be trained. Check features_meta.csv availability.")
        return 3

    write_json(out_dir / "meta_info.json", dict(
        created=datetime.now().isoformat(timespec="seconds"), labels={i: l for i, l in enumerate(LABELS)},
        variant_order=list(VARIANTS), variants=info, trained_on="meta", seed=args.seed,
        note="Per row, the first variant in variant_order whose components are all available is used; "
             "if none applies (both transformers missing) fall back to the TF-IDF baseline."))
    log.info("saved %d meta-models to %s", len(fitted), out_dir.relative_to(ROOT))

    # ---- comparisons: meta (out-of-fold) and val (fitted models, with per-row fallback)
    used_meta = np.full(len(meta), None, dtype=object)
    meta_fallback_oof = np.full(len(meta), -1)
    for v in VARIANTS:                     # same preference order as at test time
        if v in oof_full:
            m = (oof_full[v] >= 0) & (used_meta == None)  # noqa: E711
            meta_fallback_oof[m] = oof_full[v][m]
            used_meta[m] = v
    cmp_meta = compare(meta, {**{f"{v} (out-of-fold)": oof_full[v] for v in fitted},
                              "stacked + fallback (out-of-fold)": meta_fallback_oof})
    val_preds = {}
    for v, model in fitted.items():
        mask = rows_for_variant(val, v)
        p = np.full(len(val), -1)
        if mask.any():
            p[mask] = model.predict(val.loc[mask, variant_cols(v)].to_numpy(dtype=float))
        val_preds[v] = p
    stacked, _, used_val = predict_with_fallback(val, fitted)
    val_preds["stacked + fallback"] = stacked
    cmp_val = compare(val, val_preds)

    for name, rows in (("meta", cmp_meta), ("val", cmp_val)):
        log.info("---- comparison on %s ----", name)
        for r in rows:
            log.info("%-42s cov %5.1f%%  Macro-F1 %s  F1 against %s", r["system"], r["coverage"] * 100,
                     f"{r['macro_f1']:.4f}" if r["macro_f1"] is not None else "  -   ",
                     f"{r['f1_against']:.3f}" if r["f1_against"] is not None else "-")

    verdict = []
    for name, rows in (("val", cmp_val), ("meta (out-of-fold)", cmp_meta)):
        st = next((r for r in rows if r["system"].startswith("stacked")), None)
        single = max([r for r in rows if r["system"] in COMPONENTS and r["macro_f1"] is not None],
                     key=lambda r: r["macro_f1"], default=None)
        votes = [r for r in rows if "vote" in r["system"] and r["macro_f1"] is not None]
        if st and single:
            verdict.append(f"- **{name}**: stacked {st['macro_f1']:.4f} vs best single component "
                           f"{single['system']} {single['macro_f1']:.4f}"
                           + (f" vs best vote {max(r['macro_f1'] for r in votes):.4f}" if votes else "")
                           + (" → stacking does **not** beat the best single model." if st["macro_f1"] <= single["macro_f1"]
                              else " → stacking beats the best single model.")
                           + (" It does **not** beat simple voting." if votes and st["macro_f1"] <= max(
                               r["macro_f1"] for r in votes) else ""))

    # ---- report
    rep.add("Header", "Meta-models trained on the `meta` slice only (held-out Pol-NLI, never seen by the "
                      "transformers). `val` is used for a small check; `semeval_test` is not used here.")
    rep.add("Data summary", md_table([{"component": c, "meta_available": available(meta, c).mean(),
                                       "val_available": available(val, c).mean()} for c in COMPONENTS]))
    rep.add("System comparison", "**Verdict**\n\n" + "\n".join(verdict))
    rep.add("System comparison", "**On `val`** (meta-models fitted on all of `meta`)\n\n" + md_table(cmp_val))
    rep.add("System comparison", "**On `meta`** (meta-models out-of-fold, grouped CV)\n\n" + md_table(cmp_meta))
    rep.add("Fallback usage", "On `val`: " + ", ".join(f"{k}: {v}" for k, v in
                                                     pd.Series(used_val).value_counts(dropna=False).items()))
    vm = compute_metrics(val["label"].to_numpy()[stacked >= 0], stacked[stacked >= 0],
                         groups={"script_group": val["script_group"].to_numpy()[stacked >= 0]})
    rep.add("Per-class results", "Stacked + fallback on `val`\n\n" +
            md_table([{"class": k, **v} for k, v in vm["per_class"].items()]))
    rep.add("Confusion matrices", "Stacked + fallback on `val`\n\n" + confusion_md(vm["confusion"]))
    rep.add("Results by script group", md_table([{"group": k, **v} for k, v in vm["by_script_group"].items()]))
    bs = bootstrap_macro_f1(val["label"].to_numpy()[stacked >= 0], stacked[stacked >= 0],
                            val["source_id"].to_numpy()[stacked >= 0])
    rep.add("Per-class results", f"Bootstrap 95% interval for val Macro-F1 (grouped by source comment): "
                                 f"{bs['low']:.3f} – {bs['high']:.3f}")
    for v, model in fitted.items():
        rep.add("Explainability", f"**{v}** (C = {info[v]['C']}, {info[v]['n_rows']} rows, CV Macro-F1 "
                                  f"{info[v]['cv_macro_f1']:.4f})\n\n```\n{weights_table(model, variant_cols(v)).round(3).to_string()}\n```")
        rep.add("Configuration", f"**{v}** C search\n\n" + md_table(cv_tables[v]))
        for i in info[v]["weight_issues"]:
            rep.warn(f"{v}: {i}")
    rep.add("Environment", md_table([env_info()]))
    rp = rep.save(path("reports", create=True) / "meta_report.md")
    log.info("report: %s", rp)

    run = mlflow_start("stacking", f"meta__{len(fitted)}variants", log)
    if run:
        import mlflow
        with run:
            for v, i in info.items():
                mlflow.log_params({f"{v}_C": i["C"], f"{v}_rows": i["n_rows"]})
                mlflow.log_metric(f"{v}_cv_macro_f1", i["cv_macro_f1"])
            st = next(r for r in cmp_val if r["system"] == "stacked + fallback")
            if st["macro_f1"] is not None:
                mlflow.log_metric("val_stacked_macro_f1", st["macro_f1"])
            mlflow.log_artifact(str(rp))
            mlflow.log_artifact(str(out_dir / "meta_info.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
