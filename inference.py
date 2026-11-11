"""Classify one comment, or a file of comments, toward the claim
"Climate change and global warming is a serious concern."  ->  Favour / None / Against.

    python inference.py -t "ग्लोबल वार्मिंग सिर्फ एक झूठ है।"
    python inference.py -f my_comments.csv [--text-col "Comment Text" --id-col "No"]
    python inference.py -t "some comment" --no-llm          # transformers only, no API calls
    python inference.py -t "some comment" --mode muril --json

Sends one comment per LLM request. To score a labelled file with batching and the full comparison, use test.py.
Full guide: INFERENCE.md (next to this file).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    from src.pipeline import MODES
    ap = argparse.ArgumentParser(prog="inference.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("-t", "--text", help="a single comment to classify")
    src.add_argument("-f", "--file", help="a file of comments (.csv .tsv .json .jsonl .xlsx .txt)")
    ap.add_argument("-c", "--claim", help="another claim (the models were trained for the default one)")
    for k in ("id", "text", "claim", "label"):
        ap.add_argument(f"--{k}-col", dest=f"{k}_col", metavar="NAME")
    ap.add_argument("--label-map", metavar='"A=favour,B=none,C=against"')
    ap.add_argument("--allow-unknown-labels", action="store_true")
    ap.add_argument("--delimiter", metavar="CHAR")
    ap.add_argument("--encoding", metavar="NAME")
    ap.add_argument("-m", "--mode", choices=MODES, default="ensemble")
    ap.add_argument("--no-llm", action="store_true", help="call no LLM; the ensemble uses its transformers-only variant")
    ap.add_argument("--require-llm", action="store_true", help="exit 5 if an LLM is unavailable instead of falling back")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    ap.add_argument("--config", metavar="PATH", help="another base config file (default configs/base.yaml)")
    ap.add_argument("-o", "--output-dir", metavar="DIR", help="where logs and predictions go (default outputs/inference/)")
    ap.add_argument("--label-style", choices=["ps", "file", "numeric"], default="ps")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("-v", "--verbose", action="store_true", help="also print probabilities and the meta-model variant")
    g.add_argument("-q", "--quiet", action="store_true", help="print only the final label(s)")
    ap.add_argument("--version", action="store_true", help="print the version and exit")
    return ap


def _early_config(argv: list[str]) -> None:
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            os.environ["ECOCONSENSUS_CONFIG"] = argv[i + 1]
        elif a.startswith("--config="):
            os.environ["ECOCONSENSUS_CONFIG"] = a.split("=", 1)[1]


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
    if not args.text and not args.file:
        ap.error("give a comment with -t TEXT or a file with -f PATH")

    import logging

    import numpy as np
    import pandas as pd
    from src.config import FIXED_CLAIM, LABELS, out_path
    from src.data_loading import LabelError, LoaderError, load_file
    from src.ensembling import LLMS, TRANSFORMERS
    from src.metrics import compute_metrics, confusion_md
    from src.pipeline import (DISPLAY, EXIT_CONFIG, EXIT_DATA, EXIT_LLM, EXIT_MODELS, PS_LABELS, TEXT_LABEL_MAP,
                              LLMRequired, ModelFilesMissing, file_label_spelling, final_prediction, format_labels,
                              load_models, needed_components, parse_label_map_arg, rows_from_texts, run_pipeline)
    from src.utils import get_logger, safe_write, set_seed, timestamp

    out_root = Path(args.output_dir).expanduser().resolve() / "inference" if args.output_dir else None
    log = get_logger("inference", "inference", log_dir=(out_root / "logs") if out_root else None,
                     console_level=logging.INFO if args.verbose else logging.WARNING, stream=sys.stderr)
    log_file = next((h.baseFilename for h in log.handlers if isinstance(h, logging.FileHandler)), "")
    set_seed(42)

    # ---- input
    try:
        if args.text is not None:
            df = rows_from_texts([args.text], args.claim)
            info = None
        else:
            lm = dict(TEXT_LABEL_MAP)
            if args.label_map:
                lm.update(parse_label_map_arg(args.label_map))
            delim = "\t" if args.delimiter == "\\t" else args.delimiter
            df, info = load_file(args.file, id_col=args.id_col, text_col=args.text_col, label_col=args.label_col,
                                 claim_col=args.claim_col, label_map=lm, delimiter=delim, encoding=args.encoding,
                                 fixed_claim=args.claim, allow_unknown_labels=args.allow_unknown_labels,
                                 detect_claim=False)
            for w in info.warnings:
                log.warning("%s", w)
    except (LoaderError, LabelError, ValueError) as e:
        if isinstance(e, LoaderError) and str(e).startswith("File not found"):
            log.error("%s", e)
            return EXIT_DATA
        log.error("%s", e)
        return EXIT_CONFIG
    except Exception as e:  # noqa: BLE001
        log.error("Could not read %s: %s: %s", args.file, type(e).__name__, e)
        return EXIT_DATA
    if len(df) == 0 or df["empty"].all():
        log.error("No comment to classify (the text is empty after cleaning).")
        return EXIT_DATA
    claim = df["claim"].iloc[0]
    if claim != FIXED_CLAIM:
        log.warning("Claim differs from the one the models were trained for; results are not reliable.")
    log.info("input: %s, %d rows, mode %s, llm %s", "text" if args.text is not None else args.file, len(df),
             args.mode, "off" if args.no_llm else "on")

    # ---- models and run
    components = needed_components(args.mode, use_llm=not args.no_llm)
    if args.mode in ("llm-a", "llm-b") and args.no_llm:
        log.error("--mode %s needs an LLM; drop --no-llm", args.mode)
        return EXIT_CONFIG
    try:
        models = load_models(components, need_meta=args.mode in ("ensemble", "transformers"))
    except ModelFilesMissing as e:
        log.error("%s", e)
        return EXIT_MODELS
    try:
        res = run_pipeline(df, models, components, device=args.device, llm_batch_size=1,
                           require_llm=args.require_llm, log=log)
    except LLMRequired as e:
        log.error("%s (--require-llm set)", e)
        return EXIT_LLM
    except ValueError as e:
        log.error("%s", e)
        return EXIT_CONFIG
    pred, source = final_prediction(res, args.mode)
    for c, i in res.llm_info.items():
        if i["status"] != "ok":
            log.warning("%s %s%s; the ensemble used a fallback for those comments", DISPLAY[c], i["status"],
                        f" ({i['stopped']})" if i["stopped"] else "")
    file_map = file_label_spelling(df) if args.label_style == "file" else None
    final = format_labels(pred, args.label_style, file_map)

    def votes(i: int) -> dict:
        out = {}
        for c in [*TRANSFORMERS, *LLMS]:
            if c in res.preds:
                out[DISPLAY[c]] = PS_LABELS[res.preds[c][i]] if res.preds[c][i] >= 0 else "unavailable"
        return out

    def probs(i: int) -> dict:
        out = {}
        for c in TRANSFORMERS:
            if c in res.preds:
                out[DISPLAY[c]] = {PS_LABELS[j]: round(float(res.table[f"{c}_p{j}"].iloc[i]), 4) for j in range(3)}
        if res.stacked_probs is not None and np.isfinite(res.stacked_probs[i]).all():
            out["Ensemble"] = {PS_LABELS[j]: round(float(res.stacked_probs[i, j]), 4) for j in range(3)}
        return out

    # ---- one comment
    if args.text is not None:
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

    # ---- a file
    ts = timestamp()
    out = pd.DataFrame({"ID": df["id"], "Predicted_Label": final})
    for c in [*TRANSFORMERS, *LLMS]:
        if c in res.preds:
            out[DISPLAY[c]] = [PS_LABELS[x] if x >= 0 else "" for x in res.preds[c]]
    out["Source"] = source
    pdir = (out_root / "predictions") if out_root else out_path("inference", "predictions", create=True)
    pp = safe_write(pdir / f"inference_{ts}.csv", out.to_csv(index=False), add_timestamp=False)
    log.info("predictions: %s", pp)
    if args.quiet:
        print("\n".join(final))
        return 0
    counts = pd.Series(final).value_counts().to_dict()
    summary = dict(file=str(args.file), rows=len(df), empty_rows=int(df["empty"].sum()), label_counts=counts,
                   predictions=str(pp), log=log_file, used=pd.Series(source).value_counts().to_dict())
    scored = df["label"].notna().to_numpy() & ~df["empty"].to_numpy(dtype=bool)
    if scored.any():
        y = df["label"].astype("Float64").to_numpy(dtype=float, na_value=np.nan)[scored].astype(int)
        m = compute_metrics(y, pred[scored])
        summary["metrics"] = dict(n=m["n"], macro_f1=m["macro_f1"], accuracy=m["accuracy"], per_class=m["per_class"],
                                  confusion=m["confusion"])
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0
    print(f"File        : {args.file}\nRows read   : {len(df)} (empty comments: {summary['empty_rows']})")
    print("Labels      : " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    print("Used        : " + ", ".join(f"{k} {v}" for k, v in summary["used"].items()))
    if "metrics" in summary:
        m = summary["metrics"]
        print(f"\nMacro-F1 {m['macro_f1']:.4f}   accuracy {m['accuracy']:.4f}   ({m['n']} labelled rows)")
        for k, v in m["per_class"].items():
            print(f"  {k:<8} precision {v['precision']:.3f}  recall {v['recall']:.3f}  F1 {v['f1']:.3f}  n {v['support']}")
        print("\n" + confusion_md(m["confusion"]))
    print(f"\nPredictions : {pp}\nLog         : {log_file}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
