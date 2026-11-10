import numpy as np

from src.metrics import bootstrap_macro_f1, compute_metrics


def test_perfect_and_missing_class():
    m = compute_metrics([0, 1, 2, 0], [0, 1, 2, 0])
    assert m["macro_f1"] == 1.0 and m["confusion"][2][2] == 1
    m = compute_metrics([0, 1, 2], [0, 1, 1])          # 'against' never predicted -> its F1 is 0
    assert m["per_class"]["against"]["f1"] == 0.0
    assert abs(m["macro_f1"] - (1 + 2 / 3 + 0) / 3) < 1e-9


def test_groups_breakdown():
    m = compute_metrics([0, 1, 2, 2], [0, 1, 2, 0], groups={"script_group": ["latin", "latin", "bengali", "bengali"]})
    assert m["by_script_group"]["latin"]["accuracy"] == 1.0
    assert m["by_script_group"]["bengali"]["n"] == 2


def test_bootstrap_grouped():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    p = np.where(rng.random(300) < 0.7, y, rng.integers(0, 3, 300))
    b = bootstrap_macro_f1(y, p, group_ids=np.repeat(np.arange(100), 3), n_boot=200)
    assert b["low"] <= b["point"] <= b["high"] and b["grouped"]
