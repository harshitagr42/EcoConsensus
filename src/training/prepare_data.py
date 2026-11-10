"""Build the frozen, PREPROCESSED splits and the upload bundle for Colab.

    python -m src.training.prepare_data                      # first run: check, clean, split, write, zip
    python -m src.training.prepare_data                      # later runs: reuse frozen splits, re-zip only
    python -m src.training.prepare_data --overwrite-splits   # rebuild the splits (changes every downstream result)

Steps
  1. Read the 7 versions (English + 3 Hindi + 3 Bengali) of Pol-NLI train, Pol-NLI validation and SemEval from the
     explicit file list below (the Bagheli folder is a wrong-language translation and is never read).
  2. Check every translated file against its English file: same row count, same labels (and same IDs for SemEval).
  3. Clean every comment with src.preprocessing.preprocess (the one shared cleaning function). The training
     notebooks read the cleaned `comment` column as it is and do no cleaning of their own.
  4. Pol-NLI train: drop rows whose English text equals a val/SemEval comment (78 of the 90 val comments also occur
     in train) and keep one source row per identical English text (some sentences repeat ~100 times).
  5. Split: Pol-NLI train -> train_base (~85%) + meta (~15%), grouped by source row and stratified by label;
     Pol-NLI validation -> val; SemEval -> semeval_test (both kept as they are).
  6. Write data/processed/{train_base,meta,val,semeval_test}.csv, manifest.json, split_report.md, a data summary,
     and data/ecoconsensus_data.zip (upload it to MyDrive/EcoConsensus/ on each Colab account).

Columns in every split file
  id, source_id, group_id, dataset, language, variant, claim, comment (cleaned, model input), comment_raw,
  label (0 favour / 1 neutral / 2 against), label_name, script_group
"""
from __future__ import annotations

import argparse
import re
import sys
import zipfile
from datetime import datetime

import pandas as pd

from src.config import FIXED_CLAIM, L2I, LABELS, ROOT, SEED, BASE, path
from src.data_loading import read_table
from src.preprocessing import is_empty, preprocess, script_group
from src.splitting import (assign_groups, check_complete_groups, check_no_leakage, split_groups, split_report)
from src.utils import file_sha256, get_logger, safe_write, write_json

# ============================== CONFIG ==============================
PN = "data/raw/original/pol_nli_data/final_processed"
PN_HI = "data/raw/translated/Hindi/pol_nli_data"
PN_BN = "data/raw/translated/Bengali/pol_nli_data"
SE = "data/raw/original/SemEval_2016_Task6"
SE_HI = "data/raw/translated/Hindi/SemEval_2016_Task6"
SE_BN = "data/raw/translated/Bengali/SemEval_2016_Task6"

POLNLI_LABELS = {"0": "favour", "1": "neutral", "2": "against"}                 # `entailment` column
SEMEVAL_LABELS = {"favor": "favour", "favour": "favour", "none": "neutral", "against": "against",
                  "0": "favour", "1": "neutral", "2": "against"}                 # `Stance` column


def _polnli(part: str) -> dict:
    return dict(comment="premise", label="entailment", id=None, label_map=POLNLI_LABELS, files={
        ("english", "original"):          f"{PN}/Pol_NLI_{part}.csv",
        ("hindi", "pure_hindi"):          f"{PN_HI}/Pol_NLI_{part}_pure_hindi.csv",
        ("hindi", "hindi_code_mix"):      f"{PN_HI}/Pol_NLI_{part}_code_mix.csv",
        ("hindi", "hinglish"):            f"{PN_HI}/Pol_NLI_{part}_hinglish.csv",
        ("bengali", "pure_bengali"):      f"{PN_BN}/Pol_NLI_{part}_pure_bengali.csv",
        ("bengali", "bengali_code_mix"):  f"{PN_BN}/Pol_NLI_{part}_bengali_code_mix.csv",
        ("bengali", "banglish"):          f"{PN_BN}/Pol_NLI_{part}_banglish.csv",
    })


SOURCES = {
    "polnli_train": _polnli("train"),
    "polnli_val": _polnli("validation"),
    "semeval": dict(comment="Tweet", label="Stance", id="ID", label_map=SEMEVAL_LABELS, files={
        ("english", "original"):          f"{SE}/final_processed.csv",
        ("hindi", "pure_hindi"):          f"{SE_HI}/final_processed_pure_hindi.csv",
        ("hindi", "hindi_code_mix"):      f"{SE_HI}/final_processed_hindi_code_mix.csv",
        ("hindi", "hinglish"):            f"{SE_HI}/final_processed_hinglish.csv",
        ("bengali", "pure_bengali"):      f"{SE_BN}/final_processed_pure_bengali.csv",
        ("bengali", "bengali_code_mix"):  f"{SE_BN}/final_processed_bengali_code_mix.csv",
        ("bengali", "banglish"):          f"{SE_BN}/final_processed_banglish.csv",
    }),
}
N_VERSIONS = 7
SPLIT_FILES = ["train_base", "meta", "val", "semeval_test"]
COLUMNS = ["id", "source_id", "group_id", "dataset", "language", "variant", "claim", "comment", "comment_raw",
           "label", "label_name", "script_group"]
# =====================================================================

_EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF]")
_HASHTAG = re.compile(r"#\w+")


def _col(df: pd.DataFrame, name: str) -> str:
    for c in df.columns:
        if str(c).strip().lower().lstrip("﻿") == name.lower():
            return c
    raise KeyError(f"column '{name}' not found; columns are {list(df.columns)}")


def load_source(name: str, spec: dict, log) -> tuple[pd.DataFrame, list[str]]:
    """All 7 versions of one dataset, row-aligned with the English file. Returns (rows, problems)."""
    problems, frames, english = [], [], None
    for (lang, variant), rel in spec["files"].items():
        p = ROOT / rel
        if "bagheli" in rel.lower():
            raise ValueError(f"{rel}: Bagheli files must never be used")
        if not p.exists():
            problems.append(f"{name}: missing file {rel}")
            continue
        raw, _ = read_table(p)
        comment_col, label_col = _col(raw, spec["comment"]), _col(raw, spec["label"])
        labels = [spec["label_map"].get(str(v).strip().lower().removesuffix(".0")) for v in raw[label_col]]
        bad = sum(l is None for l in labels)
        if bad:
            problems.append(f"{name}/{lang}/{variant}: {bad} rows with unknown labels "
                            f"{sorted({str(v) for v, l in zip(raw[label_col], labels) if l is None})[:5]}")
        df = pd.DataFrame({
            "row": range(len(raw)),
            "language": lang, "variant": variant,
            "comment_raw": raw[comment_col].astype(object),
            "label_name": labels,
            "orig_id": raw[_col(raw, spec["id"])].astype(str).values if spec["id"] else None,
        })
        if english is None:
            if (lang, variant) != ("english", "original"):
                raise ValueError("the English file must be listed first")
            english = df
        else:
            if len(df) != len(english):
                problems.append(f"{name}/{lang}/{variant}: {len(df)} rows, English has {len(english)} "
                                f"-> not row-aligned, file skipped")
                continue
            mism = (df["label_name"].values != english["label_name"].values).sum()
            if mism:
                problems.append(f"{name}/{lang}/{variant}: {mism} labels differ from English (English label used)")
                df["label_name"] = english["label_name"].values
            if spec["id"] and (df["orig_id"].values != english["orig_id"].values).any():
                problems.append(f"{name}/{lang}/{variant}: IDs differ from English -> not row-aligned, file skipped")
                continue
        frames.append(df)
        log.info("  %-14s %-8s %-17s %5d rows  %s", name, lang, variant, len(df), rel)
    rows = pd.concat(frames, ignore_index=True)
    rows["dataset"] = name
    rows["source_id"] = [f"{name}_{r:05d}" for r in rows["row"]]
    rows["id"] = rows["source_id"] + "__" + rows["language"] + "__" + rows["variant"]
    return rows, problems


def clean(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["comment"] = [preprocess(t) for t in rows["comment_raw"]]
    rows["claim"] = FIXED_CLAIM
    rows["script_group"] = [script_group(t) for t in rows["comment"]]
    rows["label"] = rows["label_name"].map(L2I)
    return rows


def data_summary(splits: dict[str, pd.DataFrame]) -> str:
    """Lengths, hashtags, emojis, handles/URLs and duplicates: read once before training."""
    allrows = pd.concat([d.assign(split=n) for n, d in splits.items()], ignore_index=True)
    allrows["lv"] = allrows["language"] + "/" + allrows["variant"]
    raw = allrows["comment_raw"].fillna("").astype(str)
    allrows["chars"] = allrows["comment"].str.len()
    allrows["words"] = allrows["comment"].str.split().str.len()
    allrows["hashtags"] = raw.str.count(_HASHTAG.pattern)
    allrows["emojis"] = raw.str.count(_EMOJI.pattern)
    allrows["handles"] = allrows["comment"].str.count("@USER")
    allrows["urls"] = allrows["comment"].str.count(r"\bURL\b")
    out = ["## Comment length (words, cleaned) per language/variant", "",
           "| language/variant | mean | median | p95 | max |", "|---|---|---|---|---|"]
    for lv, g in allrows.groupby("lv"):
        w = g["words"]
        out.append(f"| {lv} | {w.mean():.1f} | {w.median():.0f} | {w.quantile(.95):.0f} | {w.max()} |")
    out += ["", "## Hashtags, emojis, handles, URLs (rows containing at least one)", "",
            "| split | rows | hashtags | emojis | @USER | URL |", "|---|---|---|---|---|---|"]
    for s, g in allrows.groupby("split", sort=False):
        out.append(f"| {s} | {len(g)} | {(g.hashtags > 0).sum()} | {(g.emojis > 0).sum()} | "
                   f"{(g.handles > 0).sum()} | {(g.urls > 0).sum()} |")
    eng = allrows[allrows["language"] == "english"]
    top = pd.Series([h.lower() for t in eng["comment_raw"].fillna("") for h in _HASHTAG.findall(str(t))]).value_counts()
    out += ["", "## Most frequent hashtags (English, raw)", "",
            ", ".join(f"{h} ({c})" for h, c in top.head(25).items()) or "_none_", ""]
    dup = eng.groupby("split")["comment"].apply(lambda s: int(s.duplicated().sum()))
    out += ["## Exact duplicate English comments inside a split (kept; grouped together)", "",
            ", ".join(f"{k}: {v}" for k, v in dup.items()), ""]
    return "\n".join(out)


def eng_key(d: pd.DataFrame) -> set:
    return set(d.loc[d["language"] == "english", "comment"].str.casefold().str.strip())


def build(args, log) -> tuple[dict[str, pd.DataFrame], list[str], dict]:
    problems = []
    data = {}
    log.info("Reading raw files")
    for name, spec in SOURCES.items():
        rows, prob = load_source(name, spec, log)
        problems += prob
        data[name] = clean(rows)

    # rows whose cleaned comment is empty or whose label is unknown are dropped (and listed)
    dropped = {}
    for name, df in data.items():
        bad = df["comment"].map(is_empty) | df["label"].isna()
        if bad.any():
            dropped[name] = df.loc[bad, ["id", "comment_raw", "label_name"]]
            problems.append(f"{name}: {int(bad.sum())} rows dropped (empty comment or unknown label)")
        data[name] = df[~bad].copy()
        data[name]["label"] = data[name]["label"].astype(int)
        incomplete = check_complete_groups(data[name], N_VERSIONS)
        if incomplete:
            problems.append(f"{name}: {len(incomplete)} source rows do not have all {N_VERSIONS} versions "
                            f"(e.g. {incomplete[:3]})")

    # group ids: identical cleaned English text -> one group (no duplicate on both sides of a split)
    for name, df in data.items():
        eng = df[df["language"] == "english"][["source_id", "comment"]].drop_duplicates("source_id")
        gmap = assign_groups(eng)
        df["group_id"] = df["source_id"].map(gmap).fillna(df["source_id"])
        n_merged = (gmap.index != gmap.values).sum()
        if n_merged:
            log.info("  %s: %d source rows share their English text with another row (grouped)", name, n_merged)

    # Pol-NLI train rows whose English text also appears in val or SemEval would leak into evaluation
    train = data["polnli_train"]
    held_text = eng_key(data["polnli_val"]) | eng_key(data["semeval"])
    tr_eng = train[train["language"] == "english"]
    leak_groups = set(tr_eng.loc[tr_eng["comment"].str.casefold().str.strip().isin(held_text), "group_id"])
    if leak_groups:
        msg = f"polnli_train: {len(leak_groups)} groups have English text identical to a val/SemEval comment"
        if args.keep_cross_duplicates:
            problems.append(msg + " (kept: --keep-cross-duplicates)")
        else:
            problems.append(msg + " -> removed from training data")
            train = train[~train["group_id"].isin(leak_groups)]

    # Pol-NLI repeats some sentences up to 100 times (same label, same translations). Keep one source row per
    # group so training is not dominated by a few templates (--keep-duplicates to disable).
    if not args.keep_duplicates:
        before = train["source_id"].nunique()
        train = train[train["source_id"] == train["group_id"]]
        removed = before - train["source_id"].nunique()
        if removed:
            problems.append(f"polnli_train: {removed} duplicate source rows removed "
                            f"(identical English text and label; {train['source_id'].nunique()} unique rows kept)")

    # train_base / meta: grouped by group_id, stratified by the group's (English) label
    g = (train[train["language"] == "english"].groupby("group_id")["label"]
         .agg(lambda s: s.mode().iloc[0]).rename("label").reset_index())
    keep, held = split_groups(g, args.meta_fraction, args.seed)
    splits = {
        "train_base": train[train["group_id"].isin(keep)],
        "meta": train[train["group_id"].isin(held)],
        "val": data["polnli_val"],
        "semeval_test": data["semeval"],
    }
    # group ids are dataset-prefixed, so cross-dataset clashes are impossible; the check still guards the code
    check_no_leakage(splits)
    for p in problems:
        log.warning(p)
    return {k: v[COLUMNS].sort_values(["source_id", "language", "variant"]).reset_index(drop=True)
            for k, v in splits.items()}, problems, dropped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overwrite-splits", action="store_true", help="rebuild splits even if they exist")
    ap.add_argument("--meta-fraction", type=float, default=BASE["splits"]["meta_fraction"])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--keep-cross-duplicates", action="store_true",
                    help="keep Pol-NLI train rows whose text equals a val/SemEval comment")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="keep repeated Pol-NLI train comments (by default one source row per identical English text)")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args(argv)
    log = get_logger("prepare_data")

    out_dir = path("processed", create=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [s for s in SPLIT_FILES if (out_dir / f"{s}.csv").exists()]

    if existing and not args.overwrite_splits:
        if len(existing) != len(SPLIT_FILES):
            log.error("Only some split files exist (%s). Re-run with --overwrite-splits.", existing)
            return 2
        log.info("Frozen splits found in %s; reusing them (use --overwrite-splits to rebuild).", out_dir)
        splits = {s: pd.read_csv(out_dir / f"{s}.csv", dtype={"id": str, "source_id": str, "group_id": str},
                                 keep_default_na=False) for s in SPLIT_FILES}
        check_no_leakage(splits)
    else:
        splits, problems, dropped = build(args, log)
        for s, df in splits.items():
            df.to_csv(out_dir / f"{s}.csv", index=False, encoding="utf-8")
            log.info("wrote %-13s %6d rows (%d source rows)", s, len(df), df["source_id"].nunique())

        report = ["# Split report", "", f"Created {datetime.now():%Y-%m-%d %H:%M:%S}, seed {args.seed}, "
                  f"meta fraction {args.meta_fraction}. Grouped by source row (English duplicates merged), "
                  f"stratified by label. No source/group id is in two splits (checked).", "",
                  "## Data check", ""]
        report += [f"- {p}" for p in problems] or ["- All 7 versions present and row-aligned for every dataset."]
        report += ["", split_report(splits), "# Data summary", "", data_summary(splits)]
        for name, d in dropped.items():
            report += [f"## Dropped rows: {name}", "", "```", d.head(30).to_string(index=False), "```", ""]
        text = "\n".join(report)
        (out_dir / "split_report.md").write_text(text, encoding="utf-8")
        rp = safe_write(path("reports", create=True) / "split_report.md", text)
        log.info("split report: %s and %s", out_dir / "split_report.md", rp)

        manifest = dict(
            created=datetime.now().isoformat(timespec="seconds"), seed=args.seed, meta_fraction=args.meta_fraction,
            labels={i: l for i, l in enumerate(LABELS)}, fixed_claim=FIXED_CLAIM,
            splits={s: dict(rows=len(d), source_rows=int(d["source_id"].nunique()),
                            classes=d["label_name"].value_counts().reindex(LABELS, fill_value=0).to_dict(),
                            sha256=file_sha256(out_dir / f"{s}.csv")) for s, d in splits.items()},
            sources={f"{n}/{l}/{v}": dict(file=f, sha256=file_sha256(ROOT / f))
                     for n, spec in SOURCES.items() for (l, v), f in spec["files"].items() if (ROOT / f).exists()},
            problems=problems, preprocessing=BASE.get("preprocessing"),
        )
        write_json(out_dir / "manifest.json", manifest)

    # every split must hold the 3 classes; train_base/meta near-equal ratios
    for s, df in splits.items():
        pct = df["label_name"].value_counts(normalize=True).reindex(LABELS, fill_value=0) * 100
        log.info("%-13s %s", s, "  ".join(f"{l} {pct[l]:5.1f}%" for l in LABELS))

    if not args.no_zip:
        zp = path("bundle")
        files = [f"{s}.csv" for s in SPLIT_FILES] + ["manifest.json", "split_report.md"]
        tmp = zp.with_suffix(".zip.tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                if (out_dir / f).exists():
                    zf.write(out_dir / f, arcname=f)
        tmp.replace(zp)
        log.info("bundle: %s (%.1f MB). Upload it to MyDrive/EcoConsensus/ on BOTH Colab accounts.",
                 zp, zp.stat().st_size / 2**20)
    return 0


if __name__ == "__main__":
    sys.exit(main())
