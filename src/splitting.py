"""Grouped, stratified splits and leakage checks.

A *source row* is one row of an English file; all 7 language/variant versions share its `source_id`.
`group_id` additionally merges source rows whose cleaned English text is identical, so duplicates can never
land on both sides of a split.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import LABELS


class LeakageError(Exception):
    pass


def assign_groups(english: pd.DataFrame) -> pd.Series:
    """english: one row per source_id with a cleaned `comment`. Returns group_id per source_id."""
    key = english["comment"].str.casefold().str.strip()
    first = english.groupby(key)["source_id"].transform("first")
    return pd.Series(first.values, index=english["source_id"].values, name="group_id")


def split_groups(groups: pd.DataFrame, test_fraction: float, seed: int) -> tuple[set, set]:
    """groups: one row per group_id with a `label`. Stratified split of groups -> (keep_ids, held_out_ids)."""
    counts = groups["label"].value_counts()
    strat = groups["label"] if counts.min() >= 2 else None
    keep, held = train_test_split(groups["group_id"], test_size=test_fraction, random_state=seed, stratify=strat)
    return set(keep), set(held)


def check_no_leakage(splits: dict[str, pd.DataFrame], cols=("source_id", "group_id")) -> None:
    """Raise if any source/group id appears in two splits."""
    names = list(splits)
    for col in cols:
        for i, a in enumerate(names):
            if col not in splits[a]:
                continue
            for b in names[i + 1:]:
                if col not in splits[b]:
                    continue
                both = set(splits[a][col]) & set(splits[b][col])
                if both:
                    raise LeakageError(f"{len(both)} {col}s are in both '{a}' and '{b}', e.g. {sorted(both)[:5]}")


def check_complete_groups(df: pd.DataFrame, n_versions: int) -> list[str]:
    """source_ids that do not have all n_versions rows."""
    c = df.groupby("source_id").size()
    return sorted(c[c != n_versions].index)


def _pct_table(df: pd.DataFrame, by: str) -> str:
    t = pd.crosstab(df[by], df["label_name"]).reindex(columns=LABELS, fill_value=0)
    t["total"] = t.sum(axis=1)
    lines = [f"| {by} | " + " | ".join(LABELS) + " | total |", "|" + "---|" * (len(LABELS) + 2)]
    for idx, row in t.iterrows():
        cells = [f"{row[l]} ({100 * row[l] / max(1, row['total']):.1f}%)" for l in LABELS]
        lines.append(f"| {idx} | " + " | ".join(cells) + f" | {row['total']} |")
    return "\n".join(lines)


def split_report(splits: dict[str, pd.DataFrame]) -> str:
    """Markdown: counts and class percentages per split, per language and per variant."""
    allrows = pd.concat([d.assign(split=n) for n, d in splits.items()], ignore_index=True)
    allrows["lang_variant"] = allrows["language"] + "/" + allrows["variant"]
    parts = ["## Rows and class balance per split", "", _pct_table(allrows, "split"), ""]
    src = allrows.drop_duplicates("source_id")
    parts += ["## Source rows (one per original comment) per split", "", _pct_table(src, "split"), ""]
    for name, d in splits.items():
        d = d.assign(lang_variant=d["language"] + "/" + d["variant"])
        parts += [f"## `{name}` by language/variant", "", _pct_table(d, "lang_variant"), ""]
    return "\n".join(parts)
