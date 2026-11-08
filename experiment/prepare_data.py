"""Build the SMALL upload bundle (experiment_data.zip) used only by experiment.ipynb.

The experiment uses ONLY SemEval-2016 Task 6 (climate rows): English + the three Hindi variants
(pure, Hinglish, code-mixed) + the three Bengali variants (pure, Banglish, code-mixed) = 7 rows per tweet.
This script does no training. It samples a small subset of tweets (SAMPLE_SOURCE_ROWS), stratified by
label, and keeps all 7 language/variant rows of every sampled tweet.

Run this on your Mac, then upload the zip to Google Drive (see SETUP.md).

    python experiment/prepare_data.py                # uses the CONFIG block below
    python experiment/prepare_data.py --rows 200     # different subset size (source tweets)
    python experiment/prepare_data.py --root /path/to/EcoConsensus

What it does
  1. Reads the English, Hindi and Bengali SemEval CSVs listed in DATASETS.
  2. Maps labels to the project ids (0 = favour, 1 = neutral, 2 = against) and checks that every
     translated file lines up row-for-row with the English one (so splits can be grouped by tweet).
  3. Samples SAMPLE_SOURCE_ROWS tweets: all tweets of KEEP_ALL_CLASSES (default: every 'against' tweet),
     the rest stratified by label. All 7 rows of a tweet are kept together, so every language/variant
     has exactly the same class ratio.
  4. Writes semeval.csv + manifest.json and zips them into experiment/experiment_data.zip.

Only needs pandas. Text is NOT cleaned here; the notebook applies the shared preprocessing.
"""
import argparse
import glob
import hashlib
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

# ============================== CONFIG ==============================
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent                      # EcoConsensus/ (override with --root)
OUT_DIR = HERE / "experiment_data"
ZIP_PATH = HERE / "experiment_data.zip"

# Column names are the same in English and translated files. Matching is case-insensitive.
SEMEVAL_COLUMNS = dict(id="ID", claim="Target", comment="Tweet", label="Stance")

# Project label ids (final): 0 = favour, 1 = neutral, 2 = against. Label NAMES are written to the CSV;
# the notebook turns them into these ids. Numeric keys follow the same mapping.
SEMEVAL_LABEL_MAP = {"favor": "favour", "favour": "favour", "none": "neutral", "neutral": "neutral",
                     "irrelevant": "neutral", "against": "against",
                     "0": "favour", "1": "neutral", "2": "against"}

# Paths are relative to PROJECT_ROOT; globs are allowed. Every file matched is one "variant".
# Files are listed explicitly so nothing else (e.g. the Bagheli folder) can slip in.
SEMEVAL_DIR = "data/raw/original/SemEval_2016_Task6"
DATASETS = {
    "semeval": dict(
        columns=SEMEVAL_COLUMNS, label_map=SEMEVAL_LABEL_MAP, id_prefix="semeval",
        stem_prefix="final_processed",
        files={
            "english": [f"{SEMEVAL_DIR}/final_processed.csv"],
            "hindi": [
                "data/raw/translated/Hindi/SemEval_2016_Task6/final_processed_pure_hindi.csv",
                "data/raw/translated/Hindi/SemEval_2016_Task6/final_processed_hinglish.csv",
                "data/raw/translated/Hindi/SemEval_2016_Task6/final_processed_hindi_code_mix.csv",
            ],
            "bengali": [
                "data/raw/translated/Bengali/SemEval_2016_Task6/final_processed_pure_bengali.csv",
                "data/raw/translated/Bengali/SemEval_2016_Task6/final_processed_banglish.csv",
                "data/raw/translated/Bengali/SemEval_2016_Task6/final_processed_bengali_code_mix.csv",
            ],
        }),
}
# How many SOURCE tweets to keep. Each one is kept in English + 3 Hindi + 3 Bengali variants,
# so rows in the bundle = N x 7. None = keep all 564 tweets.
SAMPLE_SOURCE_ROWS = 300
# Classes whose tweets are ALL kept (SemEval has only 26 'against' tweets; a plain stratified sample of 300
# keeps 14, leaving 2-3 per evaluation split). The remaining quota is filled stratified from the other classes.
# This raises the share of these classes above their share in the full dataset; [] = plain stratified sample.
KEEP_ALL_CLASSES = ["against"]
SAMPLE_SEED = 2024
# A translated file whose labels disagree with English more than this is treated as misaligned.
MAX_LABEL_MISMATCH = 0.02
# =====================================================================

UNIFIED_COLS = ["source_id", "language", "variant", "claim", "comment", "label"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_col(df: pd.DataFrame, wanted: str, path: Path) -> str:
    lookup = {c.strip().lower(): c for c in df.columns}
    if wanted.strip().lower() not in lookup:
        sys.exit(f"[error] {path}: column '{wanted}' not found. Columns present: {list(df.columns)}\n"
                 f"        Fix the column names in the CONFIG block of prepare_data.py.")
    return lookup[wanted.strip().lower()]


def map_label(value, label_map: dict, path: Path) -> str:
    key = str(value).strip().lower()
    try:                                    # "2.0" -> "2"
        key = str(int(float(key)))
    except ValueError:
        pass
    if key not in label_map:
        sys.exit(f"[error] {path}: unknown label value {value!r}. Known: {sorted(label_map)}")
    return label_map[key]


def variant_name(path: Path, stem_prefix: str, language: str) -> str:
    stem = path.stem
    if stem.lower().startswith(stem_prefix.lower()):
        stem = stem[len(stem_prefix):]
    stem = stem.strip("_")
    return stem or ("original" if language == "english" else "default")


def expand(project_root: Path, patterns: list) -> list:
    found = []
    for pat in patterns:
        hits = sorted(glob.glob(str(project_root / pat)))
        if not hits:
            sys.exit(f"[error] No file matches '{pat}' under {project_root}. Check DATASETS paths.")
        found += [Path(h) for h in hits]
    return found


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    except Exception as e:                  # noqa: BLE001 - show a readable message
        sys.exit(f"[error] Cannot read {path}: {e}")


def build_dataset(name: str, spec: dict, root: Path):
    cols, lmap, tag = spec["columns"], spec["label_map"], spec["id_prefix"]
    frames, files_info, mismatch = [], [], {}

    # ---- English first: it defines source ids and the reference labels ----
    eng_paths = expand(root, spec["files"]["english"])
    if len(eng_paths) != 1:
        sys.exit(f"[error] {name}: expected exactly one English file, got {len(eng_paths)}")
    ep = eng_paths[0]
    edf = read_csv(ep)
    ec = {k: find_col(edf, v, ep) for k, v in cols.items() if v}
    if cols["id"]:
        src_ids = [f"{tag}:{v}" for v in edf[ec["id"]]]
    else:
        src_ids = [f"{tag}:{i}" for i in range(len(edf))]
    if len(set(src_ids)) != len(src_ids):
        sys.exit(f"[error] {ep}: duplicate ids in column '{cols['id']}'")
    english = pd.DataFrame({
        "source_id": src_ids, "language": "english", "variant": "original",
        "claim": edf[ec["claim"]], "comment": edf[ec["comment"]],
        "label": [map_label(v, lmap, ep) for v in edf[ec["label"]]],
    })
    ref_label = dict(zip(english["source_id"], english["label"]))
    frames.append(english)
    files_info.append(dict(language="english", variant="original",
                           path=str(ep.relative_to(root)), rows=len(english), sha256=sha256(ep)))

    # ---- translated groups ----
    for lang, patterns in spec["files"].items():
        if lang == "english":
            continue
        for p in expand(root, patterns):
            df = read_csv(p)
            c = {k: find_col(df, v, p) for k, v in cols.items() if v}
            if cols["id"] and cols["id"].strip().lower() in {x.strip().lower() for x in df.columns}:
                ids = [f"{tag}:{v}" for v in df[find_col(df, cols["id"], p)]]
                if set(ids) != set(src_ids):
                    sys.exit(f"[error] {p}: ids do not match the English file "
                             f"({len(set(ids) ^ set(src_ids))} ids differ).")
            else:
                if len(df) != len(english):
                    sys.exit(f"[error] {p}: {len(df)} rows but English has {len(english)} and there is no id "
                             f"column to align on. Row-aligned files are required so that splits can be grouped.")
                ids = src_ids
            out = pd.DataFrame({
                "source_id": ids, "language": lang,
                "variant": variant_name(p, spec["stem_prefix"], lang),
                "claim": df[c["claim"]], "comment": df[c["comment"]],
                "label": [map_label(v, lmap, p) for v in df[c["label"]]],
            })
            mm = float((out["label"] != out["source_id"].map(ref_label)).mean())
            mismatch[str(p.relative_to(root))] = round(mm, 4)
            if mm > MAX_LABEL_MISMATCH:
                sys.exit(f"[error] {p}: {mm:.1%} of labels differ from the English row with the same id/position. "
                         f"The file is probably not row-aligned with the English one.")
            if mm > 0:
                print(f"[warn] {p.name}: {mm:.2%} labels differ from English.")
            frames.append(out)
            files_info.append(dict(language=lang, variant=out["variant"].iloc[0],
                                   path=str(p.relative_to(root)), rows=len(out), sha256=sha256(p)))

    full = pd.concat(frames, ignore_index=True)[UNIFIED_COLS]
    return full, files_info, mismatch


def sample_sources(df: pd.DataFrame, n, seed: int, keep_all=()) -> pd.DataFrame:
    """Keep n source rows with all their language/variant rows.

    Every source row of a class in keep_all is kept; the rest of the quota is filled from the other
    classes, stratified by label (largest-remainder rounding)."""
    src = df.drop_duplicates("source_id")[["source_id", "label"]]
    if n is None or n >= len(src):
        return df
    forced = src[src["label"].isin(keep_all)]
    if len(forced) >= n:
        sys.exit(f"[error] KEEP_ALL_CLASSES {list(keep_all)} already has {len(forced)} tweets, more than the {n} requested.")
    rest, n_rest = src[~src["label"].isin(keep_all)], n - len(forced)
    share = rest["label"].value_counts(normalize=True)
    quota = (share * n_rest).apply(int)
    for lab in (share * n_rest - quota).sort_values(ascending=False).index[: n_rest - int(quota.sum())]:
        quota[lab] += 1
    keep = set(forced["source_id"])
    for lab, q in quota.items():
        keep |= set(rest[rest["label"] == lab].sample(int(q), random_state=seed)["source_id"])
    return df[df["source_id"].isin(keep)].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=PROJECT_ROOT, help="EcoConsensus project root")
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="folder for the unified CSVs")
    ap.add_argument("--zip", type=Path, default=ZIP_PATH, help="zip file to create")
    ap.add_argument("--rows", type=int, default=SAMPLE_SOURCE_ROWS,
                    help="source tweets to keep (0 = all); default from SAMPLE_SOURCE_ROWS")
    args = ap.parse_args()
    n_rows = args.rows or None
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    manifest = dict(sample_source_rows=n_rows, keep_all_classes=KEEP_ALL_CLASSES, label_ids={"favour": 0, "neutral": 1, "against": 2}, sample_seed=SAMPLE_SEED, created_at=datetime.now().isoformat(timespec="seconds"), project_root=str(root),
                    datasets={}, notes=[])
    csv_hashes = []
    for name, spec in DATASETS.items():
        df, files_info, mismatch = build_dataset(name, spec, root)
        full_sources = df["source_id"].nunique()
        df = sample_sources(df, n_rows, SAMPLE_SEED, KEEP_ALL_CLASSES)
        ratios = pd.crosstab([df["language"], df["variant"]], df["label"], normalize="index").round(4)
        assert (ratios.nunique() == 1).all(), "class ratio differs between language/variant groups"
        csv_path = out / f"{name}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")
        csv_hashes.append(sha256(csv_path))
        empty = int((df["comment"].str.strip() == "").sum())
        manifest["datasets"][name] = dict(
            rows=len(df), source_rows=int(df["source_id"].nunique()), source_rows_available=int(full_sources),
            empty_comments=empty,
            label_counts_english=df[df.language == "english"]["label"].value_counts().to_dict(),
            rows_by_language=df["language"].value_counts().to_dict(),
            rows_by_variant={f"{lg}/{v}": int(n) for (lg, v), n in df.groupby(["language", "variant"]).size().items()},
            files=files_info, label_mismatch_vs_english=mismatch,
            label_map=spec["label_map"], columns=spec["columns"])
        print(f"\n== {name}: kept {df['source_id'].nunique()} of {full_sources} source rows -> {len(df)} rows ==")
        print(pd.crosstab([df["language"], df["variant"]], df["label"]).to_string())
        print("class share (identical in every language/variant):",
              ratios.iloc[0].rename(index=lambda x: x).to_dict())
        if empty:
            print(f"[warn] {empty} rows have an empty comment (the notebook drops them).")

    for name, d in manifest["datasets"].items():
        variants = sorted(d["rows_by_variant"])
        manifest["notes"].append(f"{name}: {d['source_rows']} of {d['source_rows_available']} tweets, "
                                 f"{len(variants)} language/variant groups: {', '.join(variants)}")
    manifest["data_hash"] = hashlib.sha256("".join(csv_hashes).encode()).hexdigest()[:16]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as z:
        for fname in [f"{n}.csv" for n in DATASETS] + ["manifest.json"]:
            z.write(out / fname, arcname=fname)
    print(f"\nWrote {args.zip}  ({args.zip.stat().st_size / 1e6:.1f} MB)")
    print("Next: upload that zip to Google Drive as described in experiment/SETUP.md.")


if __name__ == "__main__":
    main()