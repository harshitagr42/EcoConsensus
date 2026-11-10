"""Stacking: feature table, meta-model variants, per-row fallback selection, voting baselines, weight tables.

Features per comment (12): muril_p0..p2, indicbert_p0..p2 (probabilities), llm_a_{favour,none,against},
llm_b_{favour,none,against} (one-hot). `<component>_available` columns say which components answered.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import LABELS

TRANSFORMERS = ["muril", "indicbert"]
LLMS = ["llm_a", "llm_b"]
COMPONENTS = TRANSFORMERS + LLMS
LLM_ONEHOT = ["favour", "none", "against"]          # names from docs/STACKING_ENSEMBLE.md (none = neutral, id 1)

VARIANTS: dict[str, list[str]] = {                   # order = preference when several are possible
    "meta_all4": ["muril", "indicbert", "llm_a", "llm_b"],
    "meta_no_llm_a": ["muril", "indicbert", "llm_b"],
    "meta_no_llm_b": ["muril", "indicbert", "llm_a"],
    "meta_transformers": ["muril", "indicbert"],
}


def feature_cols(component: str) -> list[str]:
    if component in TRANSFORMERS:
        return [f"{component}_p{j}" for j in range(3)]
    return [f"{component}_{n}" for n in LLM_ONEHOT]


def variant_cols(variant: str) -> list[str]:
    return [c for comp in VARIANTS[variant] for c in feature_cols(comp)]


def add_transformer(table: pd.DataFrame, comp: str, probs: np.ndarray | None) -> pd.DataFrame:
    """probs: (n, 3) aligned with table rows, or None if the component is missing."""
    for j, c in enumerate(feature_cols(comp)):
        table[c] = probs[:, j] if probs is not None else np.nan
    table[f"{comp}_available"] = (np.isfinite(probs).all(axis=1) if probs is not None
                                  else np.zeros(len(table), dtype=bool))
    return table


def add_llm(table: pd.DataFrame, comp: str, labels: list[int | None] | None) -> pd.DataFrame:
    """labels: label id per row, None where the LLM gave no usable answer (unavailable / not cached)."""
    labels = labels if labels is not None else [None] * len(table)
    onehot = np.zeros((len(table), 3))
    avail = np.array([l is not None for l in labels])
    for i, l in enumerate(labels):
        if l is not None:
            onehot[i, int(l)] = 1.0
    for j, c in enumerate(feature_cols(comp)):
        table[c] = onehot[:, j]
    table[f"{comp}_available"] = avail
    table[f"{comp}_label"] = pd.array([l if l is not None else pd.NA for l in labels], dtype="Int64")
    return table


def available(table: pd.DataFrame, comp: str) -> np.ndarray:
    col = f"{comp}_available"
    return table[col].astype(bool).to_numpy() if col in table else np.zeros(len(table), dtype=bool)


def rows_for_variant(table: pd.DataFrame, variant: str) -> np.ndarray:
    m = np.ones(len(table), dtype=bool)
    for comp in VARIANTS[variant]:
        m &= available(table, comp)
    return m


def choose_variant(table: pd.DataFrame, fitted: dict | None = None) -> np.ndarray:
    """Per row: the first variant (in VARIANTS order) whose components are all available and that is fitted.
    None where no variant fits (both transformers missing) -> the caller falls back to TF-IDF."""
    out = np.full(len(table), None, dtype=object)
    for v in VARIANTS:
        if fitted is not None and v not in fitted:
            continue
        m = rows_for_variant(table, v) & (out == None)  # noqa: E711
        out[m] = v
    return out


def make_meta_model(C: float, seed: int = 42) -> LogisticRegression:
    """Regularised multinomial logistic regression with balanced class weights."""
    return LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=seed)


def predict_with_fallback(table: pd.DataFrame, models: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """models: variant -> fitted model. Returns (pred ids with -1 where no variant applies, probs, variant used)."""
    used = choose_variant(table, models)
    pred = np.full(len(table), -1)
    probs = np.full((len(table), 3), np.nan)
    for v, model in models.items():
        m = used == v
        if m.any():
            p = model.predict_proba(table.loc[m, variant_cols(v)].to_numpy(dtype=float))
            probs[m] = p
            pred[m] = p.argmax(1)
    return pred, probs, used


# ---------------------------------------------------------------- single components and voting
def component_pred(table: pd.DataFrame, comp: str) -> np.ndarray:
    """Label id per row, -1 where the component is unavailable."""
    a = available(table, comp)
    p = table[feature_cols(comp)].to_numpy(dtype=float)
    out = np.where(a, np.nan_to_num(p, nan=-1).argmax(1), -1)
    return out.astype(int)


def soft_vote(table: pd.DataFrame, comps=TRANSFORMERS) -> np.ndarray:
    m = np.ones(len(table), dtype=bool)
    for c in comps:
        m &= available(table, c)
    p = sum(np.nan_to_num(table[feature_cols(c)].to_numpy(dtype=float)) for c in comps)
    return np.where(m, p.argmax(1), -1)


def majority_vote(table: pd.DataFrame, comps=COMPONENTS) -> np.ndarray:
    """Hard vote of the available components; ties broken by the summed transformer probabilities."""
    votes = np.zeros((len(table), 3))
    for c in comps:
        pred = component_pred(table, c)
        ok = pred >= 0
        votes[np.flatnonzero(ok), pred[ok]] += 1
    tb = sum(np.nan_to_num(table[feature_cols(c)].to_numpy(dtype=float)) for c in TRANSFORMERS
             if feature_cols(c)[0] in table) * 1e-3
    out = (votes + tb).argmax(1)
    out[votes.sum(1) == 0] = -1
    return out


def weights_table(model: LogisticRegression, cols: list[str]) -> pd.DataFrame:
    """Rows = features, columns = classes: the meta-model's learned weights (explainability)."""
    coef = model.coef_
    if coef.shape[0] == 1:                         # binary edge case (should not happen with 3 classes)
        coef = np.vstack([-coef, coef])
    t = pd.DataFrame(coef.T, index=cols, columns=[LABELS[c] for c in model.classes_])
    t.loc["(intercept)"] = model.intercept_
    return t


def weight_sanity(model: LogisticRegression, cols: list[str]) -> list[str]:
    """Each component should have a positive weight from its own class-k feature to class k."""
    t = weights_table(model, cols)
    issues = []
    for col in cols:
        k = int(col[-1]) if col[-2:] in ("p0", "p1", "p2") else LLM_ONEHOT.index(col.rsplit("_", 1)[1])
        if t.loc[col, LABELS[k]] <= 0:
            issues.append(f"{col} -> {LABELS[k]} weight is {t.loc[col, LABELS[k]]:.3f} (expected > 0)")
    return issues
