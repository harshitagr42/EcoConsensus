"""Metrics: Macro-F1 first. Labels are fixed to the 3 classes, so a class that is never predicted scores 0."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_fscore_support)

from src.config import LABELS
from src.utils import to_jsonable

IDS = [0, 1, 2]


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, labels=IDS, average="macro", zero_division=0))


def compute_metrics(y_true, y_pred, groups: dict[str, "np.ndarray | list"] | None = None) -> dict:
    """Full metric set. `groups` maps a breakdown name (e.g. 'script_group') to a per-row array;
    each gets Macro-F1, accuracy and n per value."""
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=IDS, zero_division=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = dict(
            n=int(len(y_true)),
            macro_f1=macro_f1(y_true, y_pred),
            accuracy=float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
            weighted_f1=float(f1_score(y_true, y_pred, labels=IDS, average="weighted", zero_division=0)),
            mcc=float(matthews_corrcoef(y_true, y_pred)) if len(y_true) else 0.0,
            kappa=float(cohen_kappa_score(y_true, y_pred, labels=IDS)) if len(y_true) else 0.0,
            per_class={LABELS[i]: dict(precision=float(p[i]), recall=float(r[i]), f1=float(f[i]), support=int(s[i]))
                       for i in IDS},
            confusion=confusion_matrix(y_true, y_pred, labels=IDS).tolist(),
        )
    for name, g in (groups or {}).items():
        g = np.asarray(g)
        out[f"by_{name}"] = {str(v): dict(n=int((g == v).sum()), macro_f1=macro_f1(y_true[g == v], y_pred[g == v]),
                                          accuracy=float((y_true[g == v] == y_pred[g == v]).mean()))
                             for v in sorted(set(g.tolist()))}
    return to_jsonable(out)


def bootstrap_macro_f1(y_true, y_pred, group_ids=None, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05) -> dict:
    """95% interval for Macro-F1. With group_ids (source_id), whole groups are resampled, because the
    7 versions of one comment are not independent."""
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    rng = np.random.default_rng(seed)
    if group_ids is None:
        idx_by_group = [np.array([i]) for i in range(len(y_true))]
    else:
        g = np.asarray(group_ids)
        _, inv = np.unique(g, return_inverse=True)
        order = np.argsort(inv, kind="stable")
        bounds = np.flatnonzero(np.diff(inv[order])) + 1
        idx_by_group = np.split(order, bounds)
    n = len(idx_by_group)
    scores = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n, n)
        idx = np.concatenate([idx_by_group[i] for i in pick])
        scores[b] = macro_f1(y_true[idx], y_pred[idx])
    lo, hi = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return dict(point=macro_f1(y_true, y_pred), low=float(lo), high=float(hi), n_boot=n_boot,
                grouped=group_ids is not None)


def confusion_md(conf: list[list[int]]) -> str:
    head = "| true \\ pred | " + " | ".join(LABELS) + " |"
    lines = [head, "|" + "---|" * (len(LABELS) + 1)]
    for i, row in enumerate(conf):
        lines.append(f"| {LABELS[i]} | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)
