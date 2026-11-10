import pandas as pd
import pytest

from src.config import path
from src.splitting import LeakageError, check_no_leakage, split_groups

SPLITS = ["train_base", "meta", "val", "semeval_test"]


def test_leakage_detected():
    a = pd.DataFrame({"source_id": ["s1", "s2"], "group_id": ["s1", "s2"]})
    b = pd.DataFrame({"source_id": ["s2"], "group_id": ["s2"]})
    with pytest.raises(LeakageError):
        check_no_leakage({"a": a, "b": b})


def test_split_groups_stratified():
    g = pd.DataFrame({"group_id": [f"g{i}" for i in range(200)], "label": [i % 3 for i in range(200)]})
    keep, held = split_groups(g, 0.15, 42)
    assert not keep & held and len(keep | held) == 200
    assert 25 <= len(held) <= 35


@pytest.fixture(scope="module")
def processed():
    d = path("processed")
    if not all((d / f"{s}.csv").exists() for s in SPLITS):
        pytest.skip("run python -m src.training.prepare_data first")
    return {s: pd.read_csv(d / f"{s}.csv", dtype=str, keep_default_na=False) for s in SPLITS}


def test_frozen_splits_no_leakage(processed):
    check_no_leakage(processed)


def test_seven_versions_per_source(processed):
    for s, df in processed.items():
        assert (df.groupby("source_id").size() == 7).all(), s
        assert (df.groupby("source_id")["label"].nunique() == 1).all(), s


def test_train_meta_class_ratio(processed):
    a = processed["train_base"]["label"].value_counts(normalize=True)
    b = processed["meta"]["label"].value_counts(normalize=True)
    assert (a - b).abs().max() < 0.03


def test_comments_are_preprocessed(processed):
    from src.preprocessing import preprocess
    sample = processed["semeval_test"].head(300)
    assert (sample["comment"] == sample["comment"].map(preprocess)).all()
    assert not sample["comment"].str.contains("#SemST").any()
