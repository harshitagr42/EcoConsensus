"""Evaluator test runner: score every system on the file placed in test/ and write the full comparison.

    python test.py --inspect          # columns, 5 sample rows, distinct label values, planned config; runs no model
    python test.py                    # full run -> test/result.txt, test/predictions.csv
    python test.py --no-llm           # transformers + baselines only (no API calls)

Edit TEST_CONFIG below or pass flags (flags win over the block, the block wins over auto-detection).
Full guide: TESTING.md (next to this file).
"""

# ============================ TEST_CONFIG: edit the values you need ============================
TEST_CONFIG = {
    "data_file":    None,        # None = the only data file in test/; or e.g. "comments.csv"
    "id_col":       None,        # e.g. "ID"; None = auto-detect, else the row number is used
    "text_col":     None,        # e.g. "Tweet"; the comment text (required, auto-detected if None)
    "label_col":    None,        # e.g. "Stance"; None = auto-detect, no labels found = prediction-only
    "claim_col":    None,        # None = always use the fixed claim below
    "language_col": None,        # optional; if None the script group is detected from the text
    "label_map": {               # file value (any case) -> favour / none / against
        "FAVOR": "favour", "FAVOUR": "favour", "SUPPORT": "favour",
        "AGAINST": "against", "OPPOSE": "against",
        "NONE": "none", "NEUTRAL": "none", "UNRELATED": "none",
    },
    "delimiter":    None,        # None = auto-detect
    "encoding":     None,        # None = try utf-8-sig, utf-8, utf-16, cp1252, latin-1
    "fixed_claim":  "Climate change and global warming is a serious concern.",
    "use_llm":      True,
    "llm_budget":   None,        # max requests per provider this run; None = use the .env daily limit
}
# ===============================================================================================

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEST_DIR = ROOT / "test"
OUT_DIR: Path | None = None          # None = outputs/test/ (tests point it elsewhere)
DATA_EXT = {".csv", ".tsv", ".json", ".jsonl", ".xlsx", ".xls", ".txt"}
NOT_DATA = {"result.txt", "predictions.csv"}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="test.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-f", "--file", help="use this file instead of the one in test/")
    ap.add_argument("--inspect", action="store_true", help="show the file and the planned config, then stop")
    for k in ("id", "text", "label", "claim", "language"):
        ap.add_argument(f"--{k}-col", dest=f"{k}_col", metavar="NAME")
    ap.add_argument("--label-map", metavar='"A=favour,B=none,C=against"', help="adds to the TEST_CONFIG map")
    ap.add_argument("--allow-unknown-labels", action="store_true", help="unknown label values -> unlabeled rows")
    ap.add_argument("--delimiter", metavar="CHAR")
    ap.add_argument("--encoding", metavar="NAME")
    ap.add_argument("--claim", metavar="TEXT", help="another fixed claim (models were trained for the default)")
    ap.add_argument("--no-llm", action="store_true", help="call no LLM; the ensemble uses its transformers-only variant")
    ap.add_argument("--require-llm", action="store_true", help="exit 5 if an LLM is unavailable instead of falling back")
    ap.add_argument("--llm-budget", type=int, metavar="N", help="at most N requests per provider in this run")
    ap.add_argument("--batch-size", type=int, metavar="N", help="comments per LLM request (default LLM_BATCH_SIZE)")
    ap.add_argument("--label-style", choices=["ps", "file", "numeric"], default="ps")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    ap.add_argument("--config", metavar="PATH", help="another base config file (default configs/base.yaml)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("-v", "--verbose", action="store_true")
    g.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("--version", action="store_true", help="print the version and exit")
    return ap


def _early_config(argv: list[str]) -> None:
    """--config must be known before src.config is imported."""
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            os.environ["ECOCONSENSUS_CONFIG"] = argv[i + 1]
        elif a.startswith("--config="):
            os.environ["ECOCONSENSUS_CONFIG"] = a.split("=", 1)[1]


def find_data_file(name: str | None, cli_file: str | None) -> Path:
    from src.data_loading import LoaderError
    if cli_file:
        return Path(cli_file).expanduser().resolve()
    if name:
        p = Path(name)
        return p if p.is_absolute() else TEST_DIR / p
    if not TEST_DIR.exists():
        raise LoaderError(f"{TEST_DIR} does not exist. Create it and put the test file there, or use --file.")
    cands = sorted(p for p in TEST_DIR.iterdir() if p.is_file() and p.suffix.lower() in DATA_EXT
                   and p.name not in NOT_DATA and not p.name.startswith("."))
    if not cands:
        raise LoaderError(f"No data file in {TEST_DIR} (looked for {', '.join(sorted(DATA_EXT))}). Put the file there "
                          f"or use --file.")
    if len(cands) > 1:
        raise LoaderError(f"More than one data file in test/: {[p.name for p in cands]}. Move the extras out, or "
                          f"set data_file / --file.")
    return cands[0]


def effective_config(args) -> dict:
    from src.pipeline import normalise_label_map, parse_label_map_arg
    cfg = dict(TEST_CONFIG)
    for k in ("id_col", "text_col", "label_col", "claim_col", "language_col", "delimiter", "encoding"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    if args.delimiter == "\\t":
        cfg["delimiter"] = "\t"
    if args.claim:
        cfg["fixed_claim"] = args.claim
    if args.no_llm:
        cfg["use_llm"] = False
    if args.llm_budget is not None:
        cfg["llm_budget"] = args.llm_budget
    lm = normalise_label_map(cfg["label_map"])
    if args.label_map:
        lm.update(parse_label_map_arg(args.label_map))
    cfg["label_map"] = lm
    return cfg


# ---------------------------------------------------------------- --inspect
def inspect(p: Path, cfg: dict) -> int:
    import pandas as pd
    from src.data_loading import find_column, read_table
    raw, info = read_table(p, cfg["delimiter"], cfg["encoding"])
    cols = list(raw.columns)
    print(f"File      : {p}")
    print(f"Format    : {p.suffix.lower()}  encoding={info.encoding}  delimiter={info.delimiter!r}")
    print(f"Rows      : {len(raw)}")
    print(f"Columns   : {cols}\n")
    with pd.option_context("display.max_columns", 20, "display.width", 160, "display.max_colwidth", 60):
        print(raw.head(5).to_string(), "\n")
    planned = {}
    for role, key in (("id", "id_col"), ("comment", "text_col"), ("label", "label_col"), ("claim", "claim_col"),
                      ("language", "language_col")):
        if role == "claim" and not cfg[key]:            # never auto-detected: the fixed claim is used
            planned[key] = None
            continue
        try:
            planned[key] = find_column(cols, role, cfg[key])
        except Exception as e:  # noqa: BLE001
            planned[key] = f"ERROR: {e}"
    lab = planned["label_col"]
    if lab and not str(lab).startswith("ERROR") and lab in raw:
        vc = raw[lab].astype(str).str.strip().value_counts()
        print(f"Distinct values in label column '{lab}':")
        for v, n in vc.items():
            target = cfg["label_map"].get(v.lower())
            print(f"  {v!r:>20} x{n:<6} -> {target or 'NOT IN label_map'}")
        if any(v.strip().lstrip("-").isdigit() for v in vc.index) and not any(k.isdigit() for k in cfg["label_map"]):
            print("  Numeric labels and no numeric map: add e.g. --label-map \"0=favour,1=none,2=against\" "
                  "after checking what the numbers mean.")
        print()
    print("Planned configuration:")
    for k in ("id_col", "text_col", "label_col", "claim_col", "language_col"):
        print(f"  {k:<13} {planned[k]}" + ("" if planned[k] or k != "id_col" else "  (row number is used)"))
    print(f"  fixed_claim   {cfg['fixed_claim']}")
    print(f"  use_llm       {cfg['use_llm']}  llm_budget {cfg['llm_budget']}")
    if planned["text_col"] is None:
        print("\nNo comment column found: set text_col in TEST_CONFIG or pass --text-col NAME.")
    return 0


# ---------------------------------------------------------------- report
def build_report(df, res, models, cfg, args, data_path, info, llm_used: list[str], ts: str):
    import numpy as np
    import pandas as pd
    from src.config import LABELS
    from src.ensembling import LLMS, TRANSFORMERS
    from src.metrics import bootstrap_macro_f1, compute_metrics, confusion_md
    from src.pipeline import DISPLAY, VERSION, meta_weights_md
    from src.reporting import Report, md_table
    from src.utils import env_info

    rep = Report("EcoConsensus test report")
    for w in info.warnings:
        rep.warn(w)
    scored = df["label"].notna().to_numpy() & ~df["empty"].to_numpy(dtype=bool)
    has_labels = bool(scored.any())
    n_empty = int(df["empty"].sum())
    dist = df.loc[df["label"].notna(), "label"].astype(int).value_counts().reindex(range(3), fill_value=0)
    rep.add("Header", "\n".join([
        f"- Time: {datetime.now():%Y-%m-%d %H:%M:%S}  (test.py {VERSION})",
        f"- Command: `{' '.join(sys.argv)}`",
        f"- File: `{data_path}` (encoding {info.encoding}, delimiter {info.delimiter!r})",
        f"- Columns used: " + ", ".join(f"{k}=`{v}`" for k, v in info.mapping.items() if v),
        f"- Rows read: {len(df)}; empty comments after cleaning: {n_empty} (predicted, but left out of the metrics)",
        f"- Labelled rows scored: {int(scored.sum())}" + ("" if has_labels else " (no labels: prediction-only run)"),
        "- Label counts: " + ", ".join(f"{LABELS[i]} {int(dist[i])}" for i in range(3)),
        f"- LLMs: {'used' if llm_used else 'not used (--no-llm or use_llm False)'}",
    ]))
    rep.add("Data summary", md_table([{"script_group": g, "rows": int(n)} for g, n in
                                      df["script_group"].value_counts().items()]))

    systems = [s for s in ["majority", "tfidf_lr", *TRANSFORMERS, *LLMS, "soft_vote", "majority_vote", "stacked"]
               if s in res.preds]
    metrics_out = {}
    if has_labels:
        y_all = df["label"].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
        groups = {"script_group": df["script_group"].to_numpy()}
        if df["language"].notna().any():
            groups["language"] = df["language"].astype(str).to_numpy()

        def score(pred):
            ok = scored & (pred >= 0)
            y, p = y_all[ok].astype(int), pred[ok]
            m = compute_metrics(y, p, groups={k: v[ok] for k, v in groups.items()})
            m["coverage"] = float(ok.sum() / max(1, scored.sum()))
            if ok.any():
                m["bootstrap"] = bootstrap_macro_f1(y, p)
            return m
        for s in systems:
            metrics_out[s] = score(res.preds[s])

        def best(names):
            c = [s for s in names if s in metrics_out and metrics_out[s]["coverage"] > 0]
            return max(c, key=lambda s: metrics_out[s]["macro_f1"]) if c else None
        best_tr, best_llm = best(TRANSFORMERS), best(LLMS)
        rows = []
        order = ["majority", "tfidf_lr", "muril", "indicbert", "best_transformer", "llm_a", "llm_b", "best_llm",
                 "soft_vote", "majority_vote", "stacked"]
        for s in order:
            src = {"best_transformer": best_tr, "best_llm": best_llm}.get(s, s)
            if src is None or src not in metrics_out:
                continue
            m = metrics_out[src]
            b = m.get("bootstrap")
            name = (f"Best transformer ({DISPLAY[src]})" if s == "best_transformer" else
                    f"Best LLM ({DISPLAY[src]})" if s == "best_llm" else DISPLAY[s])
            rows.append(dict(system=name, n=m["n"], coverage=m["coverage"], macro_f1=m["macro_f1"],
                             ci95=f"{b['low']:.3f}-{b['high']:.3f}" if b else "", accuracy=m["accuracy"],
                             weighted_f1=m["weighted_f1"], mcc=m["mcc"], kappa=m["kappa"],
                             **{f"f1_{c}": m["per_class"][c]["f1"] for c in LABELS}))
        text = md_table(rows)
        text += ("\n\n\"Best transformer\" and \"Best LLM\" are picked by Macro-F1 **on this same file**, so they are "
                 "optimistic. `coverage` < 1 means some rows had no answer from that system (LLM unavailable); they "
                 "are left out of its score, not counted as wrong. `ci95`: bootstrap 95% interval (1,000 resamples).")
        if "stacked" in metrics_out:
            singles = {s: metrics_out[s]["macro_f1"] for s in ["tfidf_lr", *TRANSFORMERS, *LLMS]
                       if s in metrics_out and metrics_out[s]["coverage"] >= 0.999}
            ens = metrics_out["stacked"]["macro_f1"]
            if singles:
                top = max(singles, key=singles.get)
                verdict = ("beats" if ens > singles[top] else "does NOT beat")
                text += (f"\n\n**The ensemble (Macro-F1 {ens:.4f}) {verdict} the best single system with full coverage, "
                         f"{DISPLAY[top]} ({singles[top]:.4f}).**")
        rep.add("System comparison", text)
        for s in systems:
            m = metrics_out[s]
            rep.add("Per-class results", f"**{DISPLAY[s]}**\n\n" +
                    md_table([{"class": k, **v} for k, v in m["per_class"].items()]))
            rep.add("Confusion matrices", f"**{DISPLAY[s]}** (rows = true, columns = predicted)\n\n" +
                    confusion_md(m["confusion"]))
        for key in groups:
            rows = [{"system": DISPLAY[s], **{g: v["macro_f1"] for g, v in metrics_out[s][f"by_{key}"].items()}}
                    for s in systems]
            rep.add("Results by script group", f"Macro-F1 by `{key}`\n\n" + md_table(rows))
    else:
        rows = []
        for s in systems:
            p = res.preds[s]
            rows.append({"system": DISPLAY[s], **{LABELS[i]: int((p == i).sum()) for i in range(3)},
                         "no answer": int((p < 0).sum()),
                         "agrees with ensemble": float((p == res.preds["stacked"]).mean()) if "stacked" in res.preds else None})
        rep.add("Prediction summary", "No labels in the file, so no metrics. Label counts per system:\n\n" + md_table(rows))

    # ensemble details
    if res.llm_info:
        rep.add("LLM usage", md_table([dict(component=DISPLAY[c], provider=i["provider"], model=i["model"],
                                            prompt=i["prompt_version"], status=i["status"], requests=i["requests"],
                                            cache_hits=i["cache_hits"], answered=i["answered"],
                                            unavailable=i["unavailable"], parse_failures=i["parse_failures"])
                                       for c, i in res.llm_info.items()]))
        for c, i in res.llm_info.items():
            if i["stopped"]:
                rep.warn(f"{DISPLAY[c]} {i['status']}: {i['stopped']}")
            if i["errors"]:
                rep.add("LLM usage", f"{DISPLAY[c]} problems: " + ", ".join(f"{k} x{v}" for k, v in i["errors"].items()))
    vc = pd.Series(res.variant).fillna("none").value_counts()
    rep.add("Fallback usage", "Meta-model variant used per comment (the first whose components all answered):\n\n" +
            md_table([{"variant": k, "rows": int(v)} for k, v in vc.items()]))
    if vc.drop(labels=["meta_all4"], errors="ignore").sum() and cfg["use_llm"]:
        rep.warn(f"{int(vc.drop(labels=['meta_all4'], errors='ignore').sum())} comments used a fallback meta-model "
                 f"(an LLM had no answer for them)")
    w = meta_weights_md(models, "meta_all4")
    if w:
        rep.add("Explainability", "Learned meta-model weights (`meta_all4`; rows = features, columns = classes). A large "
                                  "positive weight from a component's class-k feature to class k means the ensemble "
                                  "trusts that component for that class.\n\n" + w)
    if models.meta_info:
        cv = [dict(variant=v, C=d.get("C"), rows=d.get("n_rows"), cv_macro_f1=d.get("cv_macro_f1"))
              for v, d in models.meta_info.get("variants", {}).items()]
        rep.add("Explainability", "Meta-model training (grouped 5-fold CV on the `meta` slice):\n\n" + md_table(cv))

    if has_labels and "stacked" in res.preds:
        y = df["label"].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
        p = res.preds["stacked"]
        wrong = scored & (p != y)
        pairs = pd.Series([f"{LABELS[int(a)]} -> {LABELS[int(b)]}" for a, b in zip(y[wrong], p[wrong])]).value_counts()
        text = (f"Ensemble errors: {int(wrong.sum())} of {int(scored.sum())}.\n\nMost common confusions "
                f"(true -> predicted):\n\n" + md_table([{"confusion": k, "rows": int(v)} for k, v in pairs.head(6).items()]))
        ex = []
        for i in np.flatnonzero(wrong)[:20]:
            row = dict(id=df["id"].iloc[i], comment=str(df["comment"].iloc[i])[:120], true=LABELS[int(y[i])],
                       ensemble=LABELS[int(p[i])])
            for c in [*TRANSFORMERS, *LLMS]:
                if c in res.preds:
                    row[c] = LABELS[int(res.preds[c][i])] if res.preds[c][i] >= 0 else "-"
            row["variant"] = res.variant[i]
            ex.append(row)
        text += "\n\nFirst 20 misclassified comments with each component's vote:\n\n" + md_table(ex)
        rep.add("Error analysis", text)

    rep.add("Environment", md_table([env_info()]))
    shown = {k: v for k, v in cfg.items()}
    shown.update(device=args.device, batch_size=args.batch_size, label_style=args.label_style,
                 require_llm=args.require_llm, allow_unknown_labels=args.allow_unknown_labels)
    rep.add("Configuration", "```json\n" + json.dumps(shown, indent=2, ensure_ascii=False, default=str) + "\n```")
    rep.add("Warnings", "- Scores on translated or romanised text describe machine-translated data.\n"
                        "- The transformers and the meta-model were trained on Pol-NLI; on other domains they are "
                        "less reliable than the LLMs.")
    rep.add("References", "Models, datasets and libraries: `docs/REFERENCES.md`.")
    return rep, metrics_out


# ---------------------------------------------------------------- main
def run(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    _early_config(argv)
    sys.path.insert(0, str(ROOT))
    args = build_parser().parse_args(argv)
    from src.pipeline import VERSION
    if args.version:
        print(f"test.py {VERSION}")
        return 0

    import logging

    import numpy as np
    import pandas as pd
    from src.config import out_path
    from src.data_loading import LabelError, LoaderError, load_file
    from src.ensembling import LLMS, TRANSFORMERS
    from src.pipeline import (EXIT_CONFIG, EXIT_DATA, EXIT_LLM, EXIT_MODELS, LLMRequired, ModelFilesMissing,
                              file_label_spelling, format_labels, load_models, run_pipeline)
    from src.reporting import write_latest
    from src.utils import get_logger, safe_write, set_seed, timestamp, to_jsonable

    try:
        cfg = effective_config(args)
        data_path = find_data_file(cfg["data_file"], args.file)
        if not data_path.exists():
            print(f"ERROR: file not found: {data_path}", file=sys.stderr)
            return EXIT_DATA
        if args.inspect:
            return inspect(data_path, cfg)
    except (LoaderError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_CONFIG

    out_root = OUT_DIR
    log = get_logger("test", "test", log_dir=(out_root / "logs") if out_root else None,
                     console_level=logging.WARNING if args.quiet else logging.INFO)
    set_seed(42)
    ts = timestamp()
    log.info("data file: %s", data_path)

    try:
        df, info = load_file(data_path, id_col=cfg["id_col"], text_col=cfg["text_col"], label_col=cfg["label_col"],
                             claim_col=cfg["claim_col"], language_col=cfg["language_col"], label_map=cfg["label_map"],
                             delimiter=cfg["delimiter"], encoding=cfg["encoding"], fixed_claim=cfg["fixed_claim"],
                             allow_unknown_labels=args.allow_unknown_labels, detect_claim=False)
    except LoaderError as e:
        log.error("%s", e)
        return EXIT_CONFIG
    except LabelError as e:
        unknown = getattr(e, "unknown", {})
        if unknown and all(str(v).strip().lstrip("-").replace(".", "", 1).isdigit() for v in unknown):
            log.error("Numeric labels found and no map. %s Check what the numbers mean, then pass e.g. "
                      "--label-map \"0=favour,1=none,2=against\".", e)
        else:
            log.error("%s", e)
        return EXIT_CONFIG
    except Exception as e:  # noqa: BLE001  (unreadable file)
        log.error("Could not read %s: %s: %s", data_path, type(e).__name__, e)
        return EXIT_DATA
    if len(df) == 0 or df["empty"].all():
        log.error("%s has no comments (rows %d, empty after cleaning %d)", data_path, len(df), int(df["empty"].sum()))
        return EXIT_DATA
    log.info("rows %d, empty %d, labelled %d, columns %s", len(df), info.n_empty, int(df["label"].notna().sum()),
             {k: v for k, v in info.mapping.items() if v})
    for w in info.warnings:
        log.warning("%s", w)

    components = TRANSFORMERS + (LLMS if cfg["use_llm"] else [])
    try:
        models = load_models(components)
    except ModelFilesMissing as e:
        log.error("%s", e)
        return EXIT_MODELS
    try:
        res = run_pipeline(df, models, components, device=args.device, llm_batch_size=args.batch_size,
                           llm_budget=cfg["llm_budget"], require_llm=args.require_llm, log=log)
    except LLMRequired as e:
        log.error("%s (--require-llm set)", e)
        return EXIT_LLM
    except ValueError as e:              # e.g. --device cuda without a GPU
        log.error("%s", e)
        return EXIT_CONFIG

    llm_used = [c for c in LLMS if c in res.llm_info]
    rep, metrics_out = build_report(df, res, models, cfg, args, data_path, info, llm_used, ts)
    text = rep.render()

    test_dir = TEST_DIR
    file_map = file_label_spelling(df) if args.label_style == "file" else None
    if args.label_style == "file" and not file_map:
        log.warning("--label-style file: the file has no labels; writing Favour / None / Against instead")
    pred_csv = pd.DataFrame({"ID": df["id"], "Predicted_Label": format_labels(res.preds["stacked"], args.label_style,
                                                                              file_map)}).to_csv(index=False)
    result_p, arch1 = write_latest(test_dir / "result.txt", text, test_dir / "archive")
    pred_p, arch2 = write_latest(test_dir / "predictions.csv", pred_csv, test_dir / "archive")
    md_p = safe_write(test_dir / "reports" / f"result_{ts}.md", text, add_timestamp=False)
    for a in (arch1, arch2):
        if a:
            log.info("previous file archived: %s", a)

    detail = df[["id", "comment_raw", "comment", "script_group", "label"]].copy()
    for c in [*TRANSFORMERS, *LLMS]:
        if f"{c}_available" in res.table and res.table[f"{c}_available"].any():
            cols = [x for x in res.table.columns if x.startswith(c + "_") and not x.endswith("_label")]
            for x in cols:
                detail[x] = res.table[x].to_numpy()
    for s, p in res.preds.items():
        detail[f"{s}_pred"] = p
    detail["meta_variant"] = res.variant
    if res.stacked_probs is not None:
        for j in range(3):
            detail[f"stacked_p{j}"] = np.round(res.stacked_probs[:, j], 6)
    pdir = (out_root / "predictions") if out_root else out_path("test", "predictions", create=True)
    mdir = (out_root / "metrics") if out_root else out_path("test", "metrics", create=True)
    det_p = safe_write(pdir / f"test_{ts}_detailed.csv", detail.to_csv(index=False), add_timestamp=False)
    met_p = safe_write(mdir / f"test_{ts}.json", json.dumps(to_jsonable(dict(
        created=datetime.now().isoformat(timespec="seconds"), file=str(data_path), rows=len(df),
        config={k: v for k, v in cfg.items()}, llm=res.llm_info,
        variants=pd.Series(res.variant).fillna("none").value_counts().to_dict(), metrics=metrics_out)),
        indent=2, ensure_ascii=False), add_timestamp=False)

    if metrics_out and not args.quiet:
        print("\nMacro-F1 on", data_path.name)
        for s, m in sorted(metrics_out.items(), key=lambda kv: -kv[1]["macro_f1"]):
            print(f"  {s:<14} {m['macro_f1']:.4f}  (acc {m['accuracy']:.4f}, coverage {m['coverage']:.0%})")
    print(f"\nReport      : {result_p}\nPredictions : {pred_p}\nMarkdown    : {md_p}\nDetails     : {det_p}\n"
          f"Metrics     : {met_p}")
    for w in rep.warnings:
        log.warning("%s", w)
    return 0


if __name__ == "__main__":
    sys.exit(run())
