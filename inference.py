"""Classify comments toward the claim "Climate change and global warming is a serious concern."
-> Favour / None / Against. One script for a single comment and for a whole (test) file.

    python inference.py -t "ग्लोबल वार्मिंग सिर्फ एक झूठ है।"   # one comment, printed
    python inference.py --inspect     # the one data file in data/test/: columns, 5 rows, label values, planned config
    python inference.py               # full run on that file -> outputs/inference/result.txt, predictions.csv
    python inference.py -f my_comments.csv [--text-col "Comment Text" --id-col "No"]
    python inference.py --no-llm      # transformers + baselines only (no API calls)
    python inference.py -t "some comment" --mode muril --json

A file run scores every system (baselines, both transformers, both LLMs, votes, stacked ensemble) and, when the file
has labels, writes the full comparison. Edit CONFIG below or pass flags (flags win over the block, the block wins over
auto-detection). Full guide: INFERENCE.md (next to this file).
"""

# ============================ CONFIG: edit the values you need ============================
CONFIG = {
    "data_file":    None,        # None = the only data file in data/test/; or e.g. "comments.csv"
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
# ==========================================================================================

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "test"     # where the test file is looked for when no -f is given
DATA_EXT = {".csv", ".tsv", ".json", ".jsonl", ".xlsx", ".xls", ".txt"}


def build_parser() -> argparse.ArgumentParser:
    from src.pipeline import MODES
    ap = argparse.ArgumentParser(prog="inference.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("-t", "--text", help="a single comment to classify")
    src.add_argument("-f", "--file", help="a file of comments instead of the one in data/test/")
    ap.add_argument("--inspect", action="store_true", help="show the file and the planned config, then stop")
    ap.add_argument("-c", "--claim", metavar="TEXT", help="another fixed claim (models were trained for the default)")
    for k in ("id", "text", "label", "claim", "language"):
        ap.add_argument(f"--{k}-col", dest=f"{k}_col", metavar="NAME")
    ap.add_argument("--label-map", metavar='"A=favour,B=none,C=against"', help="adds to the CONFIG map")
    ap.add_argument("--allow-unknown-labels", action="store_true", help="unknown label values -> unlabeled rows")
    ap.add_argument("--delimiter", metavar="CHAR")
    ap.add_argument("--encoding", metavar="NAME")
    ap.add_argument("-m", "--mode", choices=MODES, default="ensemble", help="which system gives the final label")
    ap.add_argument("--no-llm", action="store_true", help="call no LLM; the ensemble uses its transformers-only variant")
    ap.add_argument("--require-llm", action="store_true", help="exit 5 if an LLM is unavailable instead of falling back")
    ap.add_argument("--llm-budget", type=int, metavar="N", help="at most N requests per provider in this run")
    ap.add_argument("--batch-size", type=int, metavar="N",
                    help="comments per LLM request for a file (default LLM_BATCH_SIZE)")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    ap.add_argument("--config", metavar="PATH", help="another base config file (default configs/base.yaml)")
    ap.add_argument("-o", "--output-dir", metavar="DIR", help="where results go: DIR/inference/ (default outputs/inference/)")
    ap.add_argument("--label-style", choices=["ps", "file", "numeric"], default="ps")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("-v", "--verbose", action="store_true",
                   help="more output (one comment: also probabilities and the progress log)")
    g.add_argument("-q", "--quiet", action="store_true",
                   help="less output (one comment: only the label; file: only the output paths)")
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
        return p if p.is_absolute() else DATA_DIR / p
    if not DATA_DIR.exists():
        raise LoaderError(f"{DATA_DIR} does not exist. Create it and put the test file there, or use --file.")
    cands = sorted(p for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() in DATA_EXT
                   and not p.name.startswith("."))
    if not cands:
        raise LoaderError(f"No data file in {DATA_DIR} (looked for {', '.join(sorted(DATA_EXT))}). Put the file there "
                          f"or use --file.")
    if len(cands) > 1:
        raise LoaderError(f"More than one data file in data/test/: {[p.name for p in cands]}. Move the extras out, "
                          f"or set data_file / --file.")
    return cands[0]


def effective_config(args) -> dict:
    from src.pipeline import TEXT_LABEL_MAP, normalise_label_map, parse_label_map_arg
    cfg = dict(CONFIG)
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
    lm = dict(TEXT_LABEL_MAP)                       # the loader's text spellings, then the CONFIG map on top
    lm.update(normalise_label_map(cfg["label_map"]))
    if args.label_map:
        lm.update(parse_label_map_arg(args.label_map))
    cfg["label_map"] = lm
    return cfg


# ---------------------------------------------------------------- --inspect
def inspect(p: Path, cfg: dict) -> int:
    import pandas as pd
    from src.data_loading import map_columns, read_table
    raw, info = read_table(p, cfg["delimiter"], cfg["encoding"])
    cols = list(raw.columns)
    print(f"File      : {p}")
    print(f"Format    : {p.suffix.lower()}  encoding={info.encoding}  delimiter={info.delimiter!r}")
    print(f"Rows      : {len(raw)}")
    print(f"Columns   : {cols}\n")
    with pd.option_context("display.max_columns", 20, "display.width", 160, "display.max_colwidth", 60):
        print(raw.head(5).to_string(), "\n")
    roles = {"id": "id_col", "comment": "text_col", "label": "label_col", "claim": "claim_col",
             "language": "language_col"}
    try:
        found = map_columns(cols, {r: cfg[k] for r, k in roles.items()})
    except Exception as e:  # noqa: BLE001  (a named column that does not exist)
        print(f"ERROR: {e}")
        return 2
    planned = {k: found[r] for r, k in roles.items()}
    claim_seen = None
    if not cfg["claim_col"] and planned["claim_col"]:     # found, but only used when named
        claim_seen, planned["claim_col"] = planned["claim_col"], None
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
    if claim_seen:
        vals = raw[claim_seen].astype(str).str.strip().value_counts()
        print(f"\nClaim / statement column '{claim_seen}' found ({len(vals)} distinct value(s), e.g. "
              f"{vals.index[0][:60]!r}). It is NOT used: every comment is paired with fixed_claim, the claim the "
              f"models were trained on. Pass --claim-col {claim_seen!r} to use the column instead.")
    if planned["text_col"] is None:
        print("\nNo comment column found: set text_col in CONFIG or pass --text-col NAME.")
    elif planned["label_col"]:
        print(f"\nRun type: EVALUATION. Stance labels from '{planned['label_col']}'; the run scores every system "
              f"and writes outputs/inference/evaluation.csv and result.txt.")
    else:
        print("\nRun type: PREDICTION ONLY (no stance / label column found). If the file has one, pass "
              "--label-col NAME to get the evaluation.")
    return 0


# ---------------------------------------------------------------- report (file runs)
def build_report(df, res, models, cfg, args, data_path, info, llm_used: list[str], final_source):
    import numpy as np
    import pandas as pd
    from src.config import LABELS
    from src.ensembling import LLMS, TRANSFORMERS
    from src.metrics import bootstrap_macro_f1, compute_metrics, confusion_md
    from src.pipeline import DISPLAY, VERSION, meta_weights_md
    from src.reporting import Report, md_table
    from src.utils import env_info

    rep = Report("EcoConsensus inference report")
    for w in info.warnings:
        rep.warn(w)
    scored = df["label"].notna().to_numpy() & ~df["empty"].to_numpy(dtype=bool)
    has_labels = bool(scored.any())
    n_empty = int(df["empty"].sum())
    dist = df.loc[df["label"].notna(), "label"].astype(int).value_counts().reindex(range(3), fill_value=0)
    rep.add("Header", "\n".join([
        f"- Time: {datetime.now():%Y-%m-%d %H:%M:%S}  (inference.py {VERSION})",
        f"- Command: `{' '.join(sys.argv)}`",
        f"- File: `{data_path}` (encoding {info.encoding}, delimiter {info.delimiter!r})",
        f"- Columns used: " + ", ".join(f"{k}=`{v}`" for k, v in info.mapping.items() if v),
        f"- Claim: fixed claim \"{df['claim'].iloc[0]}\"" + (f" (column `{info.claim_col_ignored}` found and not used; "
                                                         f"--claim-col uses it)" if info.claim_col_ignored else ""),
        f"- Rows read: {len(df)}; empty comments after cleaning: {n_empty} (predicted, but left out of the metrics)",
        f"- Labelled rows scored: {int(scored.sum())}" + ("" if has_labels else " (no labels: prediction-only run)"),
        "- Label counts: " + ", ".join(f"{LABELS[i]} {int(dist[i])}" for i in range(3)),
        f"- LLMs: {'used' if llm_used else 'not used (--no-llm, use_llm False or a mode without LLMs)'}",
        f"- Prediction file label from: `--mode {args.mode}` (" +
        ", ".join(f"{k} {int(v)}" for k, v in pd.Series(final_source).value_counts().items()) + ")",
    ]))
    rep.add("Data summary", md_table([{"script_group": g, "rows": int(n)} for g, n in
                                      df["script_group"].value_counts().items()]))

    # a system that answered no row (an LLM not configured or out of quota) has nothing to score
    systems = [s for s in ["majority", "tfidf_lr", *TRANSFORMERS, *LLMS, "soft_vote", "majority_vote", "stacked"]
               if s in res.preds and (res.preds[s] >= 0).any()]
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
    if vc.drop(labels=["meta_all4"], errors="ignore").sum() and cfg["use_llm"] and args.mode == "ensemble":
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
    shown.update(mode=args.mode, device=args.device, batch_size=args.batch_size, label_style=args.label_style,
                 require_llm=args.require_llm, allow_unknown_labels=args.allow_unknown_labels)
    rep.add("Configuration", "```json\n" + json.dumps(shown, indent=2, ensure_ascii=False, default=str) + "\n```")
    rep.add("Warnings", "- Scores on translated or romanised text describe machine-translated data.\n"
                        "- The transformers and the meta-model were trained on Pol-NLI; on other domains they are "
                        "less reliable than the LLMs.")
    rep.add("References", "Models, datasets and libraries: `docs/REFERENCES.md`.")
    return rep, metrics_out


def evaluation_table(metrics_out: dict):
    """One row per system, best Macro-F1 first: what `outputs/inference/evaluation.csv` holds."""
    import pandas as pd
    from src.config import LABELS
    from src.pipeline import DISPLAY
    def r4(v):                                   # undefined metrics (e.g. kappa when one class only) stay empty
        return None if v is None else round(v, 4)
    rows = []
    for s, m in metrics_out.items():
        b = m.get("bootstrap") or {}
        row = dict(system=DISPLAY[s], key=s, n=m["n"], coverage=r4(m["coverage"]),
                   macro_f1=r4(m["macro_f1"]), ci95_low=r4(b.get("low")), ci95_high=r4(b.get("high")),
                   accuracy=r4(m["accuracy"]), weighted_f1=r4(m["weighted_f1"]), mcc=r4(m["mcc"]),
                   kappa=r4(m["kappa"]))
        for c in LABELS:
            for k in ("precision", "recall", "f1"):
                row[f"{k}_{c}"] = r4(m["per_class"][c][k])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False, kind="stable")


# ---------------------------------------------------------------- main
def run(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    _early_config(argv)
    sys.path.insert(0, str(ROOT))
    for stream in (sys.stdout, sys.stderr):           # Hindi / Bengali on Windows consoles
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = build_parser()
    args = ap.parse_args(argv)
    from src.pipeline import VERSION
    if args.version:
        print(f"inference.py {VERSION}")
        return 0
    if args.text is not None and args.inspect:
        ap.error("--inspect looks at a file; it cannot be combined with -t")

    import logging

    from src.config import FIXED_CLAIM, out_path
    from src.data_loading import LabelError, LoaderError, load_file
    from src.pipeline import (EXIT_CONFIG, EXIT_DATA, EXIT_LLM, EXIT_MODELS, LLMRequired, ModelFilesMissing,
                              load_models, needed_components, rows_from_texts, run_pipeline)
    from src.utils import get_logger, set_seed

    try:
        cfg = effective_config(args)
        data_path = None
        if args.text is None:
            data_path = find_data_file(cfg["data_file"], args.file)
            if not data_path.exists():
                print(f"ERROR: file not found: {data_path}", file=sys.stderr)
                return EXIT_DATA
            if args.inspect:
                return inspect(data_path, cfg)
    except (LoaderError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_CONFIG
    if args.mode in ("llm-a", "llm-b") and not cfg["use_llm"]:
        print(f"ERROR: --mode {args.mode} needs an LLM; drop --no-llm (or set use_llm True)", file=sys.stderr)
        return EXIT_CONFIG

    out_root = (Path(args.output_dir).expanduser().resolve() / "inference") if args.output_dir else out_path("inference")
    one = args.text is not None
    level = (logging.INFO if args.verbose else logging.WARNING) if one else \
            (logging.WARNING if args.quiet or args.json else logging.INFO)
    log = get_logger("inference", "inference", log_dir=out_root / "logs", console_level=level, stream=sys.stderr)
    log_file = next((h.baseFilename for h in log.handlers if isinstance(h, logging.FileHandler)), "")
    set_seed(42)

    # ---- input
    if one:
        df, info = rows_from_texts([args.text], cfg["fixed_claim"]), None
    else:
        log.info("data file: %s", data_path)
        try:
            df, info = load_file(data_path, id_col=cfg["id_col"], text_col=cfg["text_col"], label_col=cfg["label_col"],
                                 claim_col=cfg["claim_col"], language_col=cfg["language_col"],
                                 label_map=cfg["label_map"], delimiter=cfg["delimiter"], encoding=cfg["encoding"],
                                 fixed_claim=cfg["fixed_claim"], allow_unknown_labels=args.allow_unknown_labels,
                                 detect_claim=False)
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
        for w in info.warnings:
            log.warning("%s", w)
    if len(df) == 0 or df["empty"].all():
        log.error("No comment to classify (rows %d, all empty after cleaning).", len(df))
        return EXIT_DATA
    if df["claim"].iloc[0] != FIXED_CLAIM:
        log.warning("Claim differs from the one the models were trained for; results are not reliable.")
    log.info("input: %s, %d rows, mode %s, llm %s", "text" if one else data_path, len(df), args.mode,
             "on" if cfg["use_llm"] else "off")

    # ---- models and run
    components = needed_components(args.mode, use_llm=cfg["use_llm"])
    try:
        models = load_models(components, need_meta=args.mode in ("ensemble", "transformers"))
    except ModelFilesMissing as e:
        log.error("%s", e)
        return EXIT_MODELS
    try:
        res = run_pipeline(df, models, components, device=args.device, llm_batch_size=1 if one else args.batch_size,
                           llm_budget=cfg["llm_budget"], require_llm=args.require_llm, log=log)
    except LLMRequired as e:
        log.error("%s (--require-llm set)", e)
        return EXIT_LLM
    except ValueError as e:              # e.g. --device cuda without a GPU
        log.error("%s", e)
        return EXIT_CONFIG
    if one:
        return _print_one(args, df, res, log, log_file)
    return _write_file_results(args, cfg, df, info, res, models, data_path, out_root, log, log_file)


def _votes_and_probs(res):
    import numpy as np
    from src.ensembling import LLMS, TRANSFORMERS
    from src.pipeline import DISPLAY, PS_LABELS

    def votes(i: int) -> dict:
        return {DISPLAY[c]: PS_LABELS[res.preds[c][i]] if res.preds[c][i] >= 0 else "unavailable"
                for c in [*TRANSFORMERS, *LLMS] if c in res.preds}

    def probs(i: int) -> dict:
        out = {DISPLAY[c]: {PS_LABELS[j]: round(float(res.table[f"{c}_p{j}"].iloc[i]), 4) for j in range(3)}
               for c in TRANSFORMERS if c in res.preds}
        if res.stacked_probs is not None and np.isfinite(res.stacked_probs[i]).all():
            out["Ensemble"] = {PS_LABELS[j]: round(float(res.stacked_probs[i, j]), 4) for j in range(3)}
        return out
    return votes, probs


def _print_one(args, df, res, log, log_file) -> int:
    from src.pipeline import DISPLAY, final_prediction, format_labels
    pred, source = final_prediction(res, args.mode)
    for c, i in res.llm_info.items():
        if i["status"] != "ok":
            log.warning("%s %s%s; the ensemble used a fallback", DISPLAY[c], i["status"],
                        f" ({i['stopped']})" if i["stopped"] else "")
    final = format_labels(pred, "numeric" if args.label_style == "numeric" else "ps")
    votes, probs = _votes_and_probs(res)
    claim = df["claim"].iloc[0]
    log.info("result: %s via %s; votes %s", final[0], source[0], votes(0))
    if args.json:
        obj = dict(comment=args.text, comment_clean=df["comment"].iloc[0], claim=claim, final=final[0],
                   final_id=int(pred[0]), mode=args.mode, used=source[0], votes=votes(0), log=log_file)
        if args.verbose:
            obj["probabilities"] = probs(0)
        print(json.dumps(obj, ensure_ascii=False, indent=2))
    elif args.quiet:
        print(final[0])
    else:
        print(f"Comment : {args.text}\nClaim   : {claim}\nFinal   : {final[0]}")
        v = votes(0)
        if v:
            print("Votes   : " + "  ".join(f"{k}={x}" for k, x in v.items()))
        print(f"Mode    : {source[0]}" + (f" ({args.mode})" if args.mode != "ensemble" else ""))
        if args.verbose:
            for k, p in probs(0).items():
                print(f"  {k:<10} " + "  ".join(f"{l} {x:.3f}" for l, x in p.items()))
        print(f"Log     : {log_file}")
    return 0


def _write_file_results(args, cfg, df, info, res, models, data_path, out_root, log, log_file,
                        file_map: dict[int, str] | None = None) -> int:
    """file_map: label id -> output spelling for --label-style file (the web API passes the user's own mapping);
    default: the most common spelling of each class in the file's label column."""
    import numpy as np
    import pandas as pd
    from src.ensembling import LLMS, TRANSFORMERS
    from src.pipeline import DISPLAY, file_label_spelling, final_prediction, format_labels
    from src.reporting import write_latest
    from src.utils import safe_write, timestamp, to_jsonable

    ts = timestamp()
    pred, source = final_prediction(res, args.mode)
    for c, i in res.llm_info.items():
        if i["status"] != "ok":
            log.warning("%s %s%s; the ensemble used a fallback for those comments", DISPLAY[c], i["status"],
                        f" ({i['stopped']})" if i["stopped"] else "")
    llm_used = [c for c in LLMS if c in res.llm_info]
    rep, metrics_out = build_report(df, res, models, cfg, args, data_path, info, llm_used, source)
    text = rep.render()

    file_map = (file_map or file_label_spelling(df)) if args.label_style == "file" else None
    if args.label_style == "file" and not file_map:
        log.warning("--label-style file: the file has no labels; writing Favour / None / Against instead")
    final = format_labels(pred, args.label_style, file_map)
    pred_csv = pd.DataFrame({"ID": df["id"], "Predicted_Label": final}).to_csv(index=False)
    result_p, arch1 = write_latest(out_root / "result.txt", text, out_root / "archive")
    pred_p, arch2 = write_latest(out_root / "predictions.csv", pred_csv, out_root / "archive")
    md_p = safe_write(out_root / "reports" / f"result_{ts}.md", text, add_timestamp=False)
    eval_p, arch3 = (write_latest(out_root / "evaluation.csv", evaluation_table(metrics_out).to_csv(index=False),
                                  out_root / "archive") if metrics_out else (None, None))
    if not metrics_out:
        log.warning("No stance / label column found, so this was a PREDICTION-ONLY run (no evaluation). If the file "
                    "has labels, pass --label-col NAME (python inference.py --inspect shows the columns).")
    for a in (arch1, arch2, arch3):
        if a:
            log.info("previous file archived: %s", a)

    detail = df[["id", "comment_raw", "comment", "script_group", "label"]].copy()
    for c in [*TRANSFORMERS, *LLMS]:
        if f"{c}_available" in res.table and res.table[f"{c}_available"].any():
            for x in [x for x in res.table.columns if x.startswith(c + "_") and not x.endswith("_label")]:
                detail[x] = res.table[x].to_numpy()
    for s, p in res.preds.items():
        detail[f"{s}_pred"] = p
    detail["meta_variant"] = res.variant
    if res.stacked_probs is not None:
        for j in range(3):
            detail[f"stacked_p{j}"] = np.round(res.stacked_probs[:, j], 6)
    detail["final_pred"] = pred
    detail["final_source"] = source
    det_p = safe_write(out_root / "predictions" / f"inference_{ts}_detailed.csv", detail.to_csv(index=False),
                       add_timestamp=False)
    met_p = safe_write(out_root / "metrics" / f"inference_{ts}.json", json.dumps(to_jsonable(dict(
        created=datetime.now().isoformat(timespec="seconds"), file=str(data_path), rows=len(df), mode=args.mode,
        config={k: v for k, v in cfg.items()}, llm=res.llm_info,
        variants=pd.Series(res.variant).fillna("none").value_counts().to_dict(), metrics=metrics_out)),
        indent=2, ensure_ascii=False), add_timestamp=False)
    for w in rep.warnings:
        log.warning("%s", w)

    paths = dict(report=str(result_p), predictions=str(pred_p), evaluation=str(eval_p) if eval_p else None,
                 markdown=str(md_p), details=str(det_p), metrics=str(met_p), log=log_file)
    if args.json:
        summary = dict(file=str(data_path), rows=len(df), empty_rows=int(df["empty"].sum()), mode=args.mode,
                       evaluated=bool(metrics_out), columns={k: v for k, v in info.mapping.items() if v},
                       label_counts=pd.Series(final).value_counts().to_dict(),
                       used=pd.Series(source).value_counts().to_dict(),
                       macro_f1={s: m["macro_f1"] for s, m in metrics_out.items()}, **paths)
        print(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2, default=str))
        return 0
    if metrics_out and not args.quiet:
        print("\nMacro-F1 on", data_path.name)
        for s, m in sorted(metrics_out.items(), key=lambda kv: -kv[1]["macro_f1"]):
            print(f"  {s:<14} {m['macro_f1']:.4f}  (acc {m['accuracy']:.4f}, coverage {m['coverage']:.0%})")
    print(f"\nReport      : {result_p}\nPredictions : {pred_p}" + (f"\nEvaluation  : {eval_p}" if eval_p else
          "\nEvaluation  : none (no labels in the file)") + f"\nMarkdown    : {md_p}\nDetails     : {det_p}\n"
          f"Metrics     : {met_p}\nLog         : {log_file}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
