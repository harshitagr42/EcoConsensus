import numpy as np
import pandas as pd
import pytest

from src.config import OUTPUT_COMPONENTS, ROOT, out_path


def test_out_path_is_per_component():
    assert out_path("muril", "predictions") == ROOT / "outputs" / "muril" / "predictions"
    assert out_path("evaluate") == ROOT / "outputs" / "evaluate"
    for c in ("prepare_data", "baselines", "muril", "indicbert", "llm", "ensemble", "evaluate", "inference"):
        assert c in OUTPUT_COMPONENTS


def test_out_path_rejects_unknown_names():
    with pytest.raises(KeyError):
        out_path("not_a_component")
    with pytest.raises(KeyError):
        out_path("muril", "plots")


def test_evaluate_score_leaves_unanswered_rows_out():
    from src.evaluate import score, summary_row
    df = pd.DataFrame(dict(label=[0, 1, 2, 0], script_group=["latin"] * 4, language=["hindi"] * 4,
                           variant=["pure_hindi"] * 4, source_id=["a", "a", "b", "b"]))
    m = score(df, np.array([0, 1, 2, -1]), bootstrap=False)
    assert m["n"] == 3 and m["coverage"] == 0.75 and m["macro_f1"] == 1.0
    row = summary_row("x", "val", m)
    assert row["ci95"] == "" and row["f1_against"] == 1.0
