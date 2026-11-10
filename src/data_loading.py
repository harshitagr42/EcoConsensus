"""Flexible loader: any reasonable CSV/TSV/JSON/JSONL/XLSX/TXT file -> the internal record format.

Internal columns: id, source_id, dataset, language, variant, claim, comment, comment_raw, label (int or <NA>),
label_name, script_group, empty. Only `claim` and `comment` are ever model inputs.

    df, info = load_file("test/my_file.csv", label_col="Stance")
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.config import BASE, FIXED_CLAIM, L2I
from src.preprocessing import is_empty, preprocess, script_group

ENCODINGS = ("utf-8-sig", "utf-8", "utf-16", "cp1252", "latin-1")
ALIASES: dict[str, list[str]] = {k: [a.lower() for a in v] for k, v in BASE["aliases"].items()}
DEFAULT_LABEL_MAP: dict[str, str] = {str(k).strip().lower(): v for k, v in BASE["label_map"].items()}


class LoaderError(Exception):
    """Configuration/column problem (test.py exit code 2)."""


class LabelError(Exception):
    """Unknown label values (test.py exit code 3)."""


@dataclass
class LoadInfo:
    path: str
    encoding: str | None = None
    delimiter: str | None = None
    columns_found: list[str] = field(default_factory=list)
    mapping: dict[str, str | None] = field(default_factory=dict)
    n_rows: int = 0
    n_empty: int = 0
    has_labels: bool = False
    unknown_labels: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- reading
def _read_text(p: Path, encoding: str | None) -> tuple[str, str]:
    raw = p.read_bytes()
    for enc in ([encoding] if encoding else ENCODINGS):
        try:
            return raw.decode(enc), enc
        except (UnicodeDecodeError, LookupError):
            continue
    raise LoaderError(f"Could not decode {p} with any of {ENCODINGS}")


def read_table(p: str | Path, delimiter: str | None = None, encoding: str | None = None) -> tuple[pd.DataFrame, LoadInfo]:
    """Read a file by extension into a DataFrame of strings (no column mapping yet)."""
    p = Path(p)
    info = LoadInfo(path=str(p))
    if not p.exists():
        raise LoaderError(f"File not found: {p}")
    ext = p.suffix.lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(p, dtype=str)
    else:
        text, info.encoding = _read_text(p, encoding)
        if ext == ".jsonl":
            df = pd.DataFrame([json.loads(l) for l in text.splitlines() if l.strip()])
        elif ext == ".json":
            obj = json.loads(text)
            if isinstance(obj, dict):                     # {"data": [{...}, ...]} or {column: [values]}
                records = [v for v in obj.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
                if len(records) == 1:
                    obj = records[0]
            df = pd.DataFrame(obj)
        elif ext == ".txt" and delimiter is None:
            df = pd.DataFrame({"comment": [l for l in text.splitlines() if l.strip()]})
        else:
            if delimiter is None:
                if ext == ".tsv":
                    delimiter = "\t"
                else:
                    try:
                        delimiter = csv.Sniffer().sniff(text[:20000], delimiters=",\t;|").delimiter
                    except csv.Error:
                        delimiter = ","
            info.delimiter = delimiter
            df = pd.read_csv(io.StringIO(text), sep=delimiter, dtype=str, keep_default_na=False)
    df = df.astype(object).where(df.notna(), None)
    info.columns_found = [str(c) for c in df.columns]
    return df, info


# ---------------------------------------------------------------- column + label mapping
def _norm(c) -> str:
    return str(c).strip().lower().lstrip("﻿")


def find_column(columns: list[str], role: str, override: str | None = None) -> str | None:
    norm = {_norm(c): c for c in columns}
    if override:
        if _norm(override) in norm:
            return norm[_norm(override)]
        raise LoaderError(f"Column '{override}' (for {role}) not found. Columns: {list(columns)}")
    for alias in ALIASES.get(role, []):
        if alias in norm:
            return norm[alias]
    return None


def normalise_label(value, label_map: dict[str, str] | None = None) -> str | None:
    """Raw label -> 'favour'/'neutral'/'against', or None if unknown. Numeric strings like '1.0' are accepted."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    lm = label_map or DEFAULT_LABEL_MAP
    v = str(value).strip().lower()
    if v in lm:
        return lm[v]
    try:
        f = float(v)
        if f.is_integer() and str(int(f)) in lm:
            return lm[str(int(f))]
    except ValueError:
        pass
    return None


def load_file(p: str | Path, *, id_col=None, text_col=None, label_col=None, claim_col=None, language_col=None,
              label_map: dict | None = None, delimiter=None, encoding=None, fixed_claim: str | None = None,
              allow_unknown_labels: bool = False, dataset: str = "evaluator") -> tuple[pd.DataFrame, LoadInfo]:
    """Load + map + preprocess one file. Raises LoaderError / LabelError with a readable message."""
    raw, info = read_table(p, delimiter, encoding)
    cols = list(raw.columns)
    m = {
        "id": find_column(cols, "id", id_col),
        "comment": find_column(cols, "comment", text_col),
        "claim": find_column(cols, "claim", claim_col),
        "label": find_column(cols, "label", label_col),
        "language": find_column(cols, "language", language_col),
    }
    info.mapping = m
    if m["comment"] is None:
        raise LoaderError(f"No comment column found in {p}. Columns: {cols}. Set text_col / --text-col.")
    lm = {str(k).strip().lower(): v for k, v in (label_map or DEFAULT_LABEL_MAP).items()}

    out = pd.DataFrame()
    out["id"] = raw[m["id"]].astype(str) if m["id"] else [str(i) for i in range(len(raw))]
    out["source_id"] = out["id"]
    out["dataset"] = dataset
    out["language"] = raw[m["language"]].astype(str) if m["language"] else None
    out["variant"] = None
    out["comment_raw"] = raw[m["comment"]]
    out["comment"] = [preprocess(t) for t in raw[m["comment"]]]
    claim_text = fixed_claim or FIXED_CLAIM
    out["claim"] = [preprocess(c) if c else claim_text for c in raw[m["claim"]]] if m["claim"] else claim_text
    out["empty"] = [is_empty(t) for t in out["comment"]]
    out["script_group"] = [script_group(t) for t in out["comment"]]

    if m["label"]:
        names = [normalise_label(v, lm) for v in raw[m["label"]]]
        unknown = pd.Series([str(v) for v, n in zip(raw[m["label"]], names) if n is None]).value_counts()
        info.unknown_labels = {str(k): int(v) for k, v in unknown.items()}
        if info.unknown_labels and not allow_unknown_labels:
            raise LabelError(f"Unknown label values in {p}: {info.unknown_labels}. Add them to label_map "
                             f"or pass --allow-unknown-labels to score those rows as unlabeled.")
        out["label_name"] = names
        out["label"] = pd.array([L2I[n] if n else pd.NA for n in names], dtype="Int64")
        info.has_labels = out["label"].notna().any()
    else:
        out["label_name"] = None
        out["label"] = pd.array([pd.NA] * len(out), dtype="Int64")
    info.n_rows = len(out)
    info.n_empty = int(out["empty"].sum())
    if info.n_empty:
        info.warnings.append(f"{info.n_empty} empty comments after preprocessing")
    return out, info


def load_split(name: str) -> pd.DataFrame:
    """Read a frozen split written by src/training/prepare_data.py."""
    from src.config import path
    p = path("processed") / f"{name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing. Run: python -m src.training.prepare_data")
    df = pd.read_csv(p, dtype={"id": str, "source_id": str, "group_id": str}, keep_default_na=False)
    df["label"] = df["label"].astype(int)
    return df
