import numpy as np
import pandas as pd

from src.ensembling import (VARIANTS, add_llm, add_transformer, choose_variant, majority_vote, make_meta_model,
                            predict_with_fallback, variant_cols, weight_sanity)


def make_table(n=300, seed=0, drop_llm_a=(), drop_llm_b=()):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    t = pd.DataFrame({"label": y})
    for comp in ("muril", "indicbert"):
        logits = rng.normal(0, 1, (n, 3))
        logits[np.arange(n), y] += 2
        add_transformer(t, comp, np.exp(logits) / np.exp(logits).sum(1, keepdims=True))
    for comp, drop in (("llm_a", drop_llm_a), ("llm_b", drop_llm_b)):
        lab = [int(v) if rng.random() < 0.8 else int(rng.integers(0, 3)) for v in y]
        add_llm(t, comp, [None if i in drop else l for i, l in enumerate(lab)])
    return t


def test_onehot_and_availability():
    t = make_table(5, drop_llm_a={1})
    assert t.loc[1, ["llm_a_favour", "llm_a_none", "llm_a_against"]].sum() == 0
    assert t.loc[0, ["llm_a_favour", "llm_a_none", "llm_a_against"]].sum() == 1
    assert t["llm_a_available"].tolist() == [True, False, True, True, True]


def test_fallback_per_row():
    t = make_table(4, drop_llm_a={1, 3}, drop_llm_b={2, 3})
    assert choose_variant(t).tolist() == ["meta_all4", "meta_no_llm_a", "meta_no_llm_b", "meta_transformers"]
    t.loc[0, "muril_available"] = False
    assert choose_variant(t)[0] is None                      # -> TF-IDF fallback in inference.py


def test_predict_with_fallback_and_weights():
    train = make_table(600, seed=1)
    models = {v: make_meta_model(1.0).fit(train[variant_cols(v)].to_numpy(), train["label"]) for v in VARIANTS}
    for v, m in models.items():
        assert weight_sanity(m, variant_cols(v)) == []
    test = make_table(50, seed=2, drop_llm_a=set(range(10)))
    pred, probs, used = predict_with_fallback(test, models)
    assert (pred >= 0).all() and np.allclose(probs.sum(1), 1)
    assert set(used[:10]) == {"meta_no_llm_a"} and set(used[10:]) == {"meta_all4"}
    assert (majority_vote(test) >= 0).all()
