"""Pipeline and inference.py (one comment and a file) end to end on a few rows, with stub transformers and a stub LLM (no torch,
no network, no model files)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.pipeline as pl
from src.baselines import Majority, make_tfidf_lr
from src.ensembling import VARIANTS, make_meta_model, variant_cols
from src.llm_api import FatalProviderError, LLMComponent

ROOT = Path(__file__).resolve().parent.parent
ROWS = [("1", "Climate change is a serious threat, act now", "FAVOR"),
        ("2", "global warming is a hoax", "AGAINST"),
        ("3", "I had pasta for lunch", "NONE"),
        ("4", "जलवायु परिवर्तन एक hoax है", "AGAINST"),
        ("5", "জলবায়ু পরিবর্তন serious বিষয়", "FAVOR"),
        ("6", "   ", "NONE")]


class StubLLM:
    """Keyword answers in the reply format the prompt asks for."""
    def __init__(self):
        self.calls = 0

    def chat(self, system, user, max_tokens):
        self.calls += 1
        items = json.loads(user[user.index("["):])
        lab = lambda c: "AGAINST" if "hoax" in c else "FAVOUR" if "serious" in c else "NONE"  # noqa: E731
        return json.dumps([{"id": it["id"], "label": lab(it["comment"])} for it in items])


def fake_models() -> pl.Models:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    feats = pd.DataFrame(index=range(300))
    for c in ("muril", "indicbert"):
        p = rng.dirichlet([1, 1, 1], 300)
        p[np.arange(300), y] += 1
        p /= p.sum(1, keepdims=True)
        for j in range(3):
            feats[f"{c}_p{j}"] = p[:, j]
    for c in ("llm_a", "llm_b"):
        for j, n in enumerate(("favour", "none", "against")):
            feats[f"{c}_{n}"] = (y == j).astype(float)
    meta = {v: make_meta_model(1.0).fit(feats[variant_cols(v)].to_numpy(), y) for v in VARIANTS}
    tfidf = make_tfidf_lr(1.0).fit(["serious threat [SEP] a", "hoax lie [SEP] b", "lunch pasta [SEP] c"] * 3,
                                   [0, 2, 1] * 3)
    return pl.Models(Majority(0), tfidf, meta, {"variants": {}})


def stub_transformers(names, df, device="auto", log=None):
    out = {}
    for n in names:
        p = np.full((len(df), 3), 0.2)
        for i, c in enumerate(df["comment"]):
            p[i, 2 if "hoax" in c else 0 if "serious" in c else 1] = 0.6
        out[n] = p
    return out


def stub_factory(tmp_path, missing=()):
    def make(slot, claim, batch_size, max_requests):
        if slot in missing:
            raise FatalProviderError(f"llm_{slot} not configured")
        return LLMComponent(slot, client=StubLLM(), cache_dir=tmp_path / "cache", batch_size=batch_size or 4,
                            sleep=lambda s: None)
    return make


@pytest.fixture
def evaluator_file(tmp_path):
    p = tmp_path / "eval.csv"
    pd.DataFrame(ROWS, columns=["ID", "Tweet", "Stance"]).assign(Target="Climate Change").to_csv(p, index=False)
    return p


@pytest.fixture
def stubs(monkeypatch, tmp_path):
    monkeypatch.setattr(pl, "load_models", lambda components, need_meta=True: fake_models())
    monkeypatch.setattr(pl, "transformer_probs", stub_transformers)
    monkeypatch.setattr(pl, "make_llm", stub_factory(tmp_path))
    for k in ("PROVIDER", "MODEL", "API_KEY", "BASE_URL", "DAILY_LIMIT"):
        monkeypatch.setenv(f"LLM_A_{k}", "")
        monkeypatch.setenv(f"LLM_B_{k}", "")


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"ecoconsensus_{name}", ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- pipeline
def test_label_maps_and_formats():
    assert pl.parse_label_map_arg("0=favour, 1=none,2=against") == {"0": "favour", "1": "neutral", "2": "against"}
    with pytest.raises(ValueError):
        pl.parse_label_map_arg("x=maybe")
    assert not any(k.isdigit() for k in pl.TEXT_LABEL_MAP)           # numbers are never mapped by default
    assert pl.format_labels([0, 1, 2, -1]) == ["Favour", "None", "Against", ""]
    assert pl.format_labels([2], "numeric") == ["2"]
    assert pl.format_labels([0, 1], "file", {0: "FAVOR"}) == ["FAVOR", "None"]


def test_pipeline_all_components_and_fallback(tmp_path, evaluator_file):
    from src.data_loading import load_file
    df, _ = load_file(evaluator_file, label_map=pl.TEXT_LABEL_MAP, detect_claim=False)
    assert (df["claim"] == pl.FIXED_CLAIM).all()                     # `Target` is not taken as the claim
    comps = ["muril", "indicbert", "llm_a", "llm_b"]
    res = pl.run_pipeline(df, fake_models(), comps, transformer_fn=stub_transformers,
                          llm_factory=stub_factory(tmp_path, missing=("a",)))
    assert res.llm_info["llm_a"]["status"] == "not configured"
    assert res.llm_info["llm_b"]["answered"] == 5                    # the empty comment is never sent
    assert list(res.variant[:5]) == ["meta_no_llm_a"] * 5
    assert res.variant[5] == "meta_transformers"
    assert (res.preds["stacked"] >= 0).all()
    assert set(res.preds) >= {"majority", "tfidf_lr", "muril", "indicbert", "llm_b", "soft_vote", "majority_vote"}
    with pytest.raises(pl.LLMRequired):
        pl.run_pipeline(df, fake_models(), comps, transformer_fn=stub_transformers, require_llm=True,
                        llm_factory=stub_factory(tmp_path, missing=("a",)))


def test_llms_run_at_the_same_time(tmp_path):
    """Each stub reply waits until the other LLM is also mid-request; run one after the other, this would time out."""
    import threading
    barrier = threading.Barrier(2, timeout=10)

    class MeetingLLM(StubLLM):
        def chat(self, system, user, max_tokens):
            barrier.wait()
            return super().chat(system, user, max_tokens)

    def make(slot, claim, batch_size, max_requests):
        return LLMComponent(slot, client=MeetingLLM(), cache_dir=tmp_path / f"cache_{slot}", batch_size=2,
                            sleep=lambda s: None)

    df = pl.rows_from_texts(["global warming is a hoax", "climate change is serious", "pasta", "more hoax"])
    res = pl.run_pipeline(df, fake_models(), ["llm_a", "llm_b"], llm_factory=make)
    for c in ("llm_a", "llm_b"):
        assert res.llm_info[c]["status"] == "ok" and res.llm_info[c]["answered"] == 4
        assert list(res.preds[c]) == [2, 0, 1, 2]


def test_cancel_stops_an_llm_like_ctrl_c(tmp_path):
    import threading
    comp = LLMComponent("a", client=StubLLM(), cache_dir=tmp_path / "cache", batch_size=2, sleep=lambda s: None)
    comp.cancel = threading.Event()
    comp.cancel.set()
    out = comp.run(["global warming is a hoax", "pasta"])
    assert comp.interrupted and comp.client.calls == 0
    assert all(r["status"] == "missing" for r in out)


def test_final_prediction_modes(tmp_path):
    df = pl.rows_from_texts(["global warming is a hoax"])
    res = pl.run_pipeline(df, fake_models(), ["llm_a"], llm_factory=stub_factory(tmp_path, missing=("a",)))
    pred, src = pl.final_prediction(res, "llm-a")                     # LLM-A has no answer -> TF-IDF
    assert src[0] == "tfidf_lr fallback" and pred[0] in (0, 1, 2)
    assert "stacked" not in res.preds


# ---------------------------------------------------------------- inference.py on a file
def test_file_run_end_to_end(tmp_path, evaluator_file, stubs, monkeypatch, capsys):
    inf = _load_script("inference")
    monkeypatch.setattr(inf, "DATA_DIR", tmp_path / "data_test")
    out = tmp_path / "out" / "inference"
    assert inf.run(["--file", str(evaluator_file), "--inspect"]) == 0
    assert "Tweet" in capsys.readouterr().out
    assert inf.run(["--file", str(evaluator_file), "-q", "-o", str(tmp_path / "out")]) == 0
    result = (out / "result.txt").read_text(encoding="utf-8")
    for section in ("System comparison", "Per-class results", "Confusion matrices", "LLM usage", "Fallback usage",
                    "Error analysis", "Explainability", "Configuration"):
        assert f"## {section}" in result
    pred = pd.read_csv(out / "predictions.csv", dtype=str, keep_default_na=False)        # "None" is a label
    assert pred.columns.tolist() == ["ID", "Predicted_Label"] and pred["ID"].tolist() == [r[0] for r in ROWS]
    assert set(pred["Predicted_Label"]) <= {"Favour", "None", "Against"}
    assert inf.run(["--file", str(evaluator_file), "-q", "--no-llm", "--label-style", "file",
                    "-o", str(tmp_path / "out")]) == 0
    assert len(list((out / "archive").iterdir())) == 3                  # first result, predictions, evaluation kept
    assert len(list((out / "reports").iterdir())) == 2
    assert set(pd.read_csv(out / "predictions.csv", keep_default_na=False)["Predicted_Label"]) <= {"FAVOR", "NONE", "AGAINST"}
    assert len(list((out / "predictions").glob("inference_*_detailed.csv"))) == 2
    assert len(list((out / "metrics").glob("inference_*.json"))) == 2


@pytest.mark.parametrize("columns, labelled", [
    (["Comments"], False),                                              # comments only -> prediction-only
    (["Comment Text", "Stance Label"], True),                           # comments + stance, unusual headers
    (["Tweet ID", "Statement", "Tweet", "Stance"], True),               # + a statement (claim) column
])
def test_file_shapes(tmp_path, stubs, capsys, columns, labelled):
    inf = _load_script("inference")
    data = {"Comments": [r[1] for r in ROWS], "Comment Text": [r[1] for r in ROWS], "Tweet": [r[1] for r in ROWS],
            "Stance Label": [r[2] for r in ROWS], "Stance": [r[2] for r in ROWS], "Tweet ID": [r[0] for r in ROWS],
            "Statement": ["Climate change and global warming is a serious concern."] * len(ROWS)}
    f = tmp_path / "shape.csv"
    pd.DataFrame({c: data[c] for c in columns}).to_csv(f, index=False)
    assert inf.run(["-f", str(f), "--inspect"]) == 0
    shown = capsys.readouterr().out
    assert ("Run type: EVALUATION" in shown) == labelled and ("PREDICTION ONLY" in shown) == (not labelled)
    if "Statement" in columns:
        assert "Claim / statement column 'Statement' found" in shown
    assert inf.run(["-f", str(f), "--json", "-o", str(tmp_path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["evaluated"] == labelled
    assert summary["columns"]["comment"] in ("Comments", "Comment Text", "Tweet")
    if labelled:
        ev = pd.read_csv(summary["evaluation"])
        assert {"MuRIL", "IndicBERT", "LLM-A", "LLM-B", "Ensemble (stacked)"} <= set(ev["system"])
        assert ev["macro_f1"].is_monotonic_decreasing and "f1_against" in ev and "ci95_low" in ev
    else:
        assert summary["evaluation"] is None
    if "Tweet ID" in columns:
        assert summary["columns"] == {"id": "Tweet ID", "comment": "Tweet", "label": "Stance"}  # Statement not used
        assert pd.read_csv(summary["predictions"], dtype=str)["ID"].tolist() == [r[0] for r in ROWS]


def test_statement_column_with_another_claim_is_reported(tmp_path, stubs):
    inf = _load_script("inference")
    f = tmp_path / "s.csv"
    pd.DataFrame({"comment": [r[1] for r in ROWS], "statement": "Climate Change is a Real Concern",
                  "stance": [r[2] for r in ROWS]}).to_csv(f, index=False)
    assert inf.run(["-f", str(f), "-q", "-o", str(tmp_path)]) == 0
    result = (tmp_path / "inference" / "result.txt").read_text(encoding="utf-8")
    assert "claim column 'statement' is not used" in result and "--claim-col" in result


def test_file_run_default_folder_and_json(tmp_path, evaluator_file, stubs, monkeypatch, capsys):
    inf = _load_script("inference")
    data_dir = tmp_path / "data_test"
    monkeypatch.setattr(inf, "DATA_DIR", data_dir)
    data_dir.mkdir()
    (data_dir / "eval.csv").write_bytes(evaluator_file.read_bytes())
    assert inf.run(["--json", "--mode", "muril", "-o", str(tmp_path)]) == 0       # the one file in data/test/
    summary = json.loads(capsys.readouterr().out)
    assert summary["rows"] == 6 and summary["mode"] == "muril" and "muril" in summary["macro_f1"]
    assert set(summary["used"]) <= {"muril", "tfidf_lr fallback"}
    assert len(pd.read_csv(summary["predictions"])) == 6
    (data_dir / "b.csv").write_text("comment\ny\n")
    assert inf.run(["-q", "-o", str(tmp_path)]) == 2                    # two data files in data/test/


def test_file_run_errors(tmp_path, stubs, monkeypatch):
    inf = _load_script("inference")
    monkeypatch.setattr(inf, "DATA_DIR", tmp_path / "missing_dir")
    o = ["-o", str(tmp_path)]
    num = tmp_path / "num.csv"
    num.write_text("comment,label\nhoax,2\nserious,0\n", encoding="utf-8")
    assert inf.run(["--file", str(num), "-q", *o]) == 2                 # numeric labels need a map
    assert inf.run(["--file", str(num), "-q", "--label-map", "0=favour,1=none,2=against", *o]) == 0
    assert inf.run(["--file", str(num), "-q", "--text-col", "nope", *o]) == 2
    assert inf.run(["-f", str(tmp_path / "missing.csv"), *o]) == 3
    assert inf.run(["-q", *o]) == 2                                     # no data/test/ folder
    assert inf.run(["-f", str(num), "--mode", "llm-a", "--no-llm", *o]) == 2


# ---------------------------------------------------------------- inference.py on one comment
def test_inference_one_comment(tmp_path, stubs, capsys):
    inf = _load_script("inference")
    assert inf.run(["-t", "global warming is a hoax", "--json", "-o", str(tmp_path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["final"] in ("Favour", "None", "Against") and out["used"] == "meta_all4"
    assert set(out["votes"]) == {"MuRIL", "IndicBERT", "LLM-A", "LLM-B"}
    assert inf.run(["-t", "x", "--no-llm", "-q", "-o", str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip() in ("Favour", "None", "Against")
    assert not (tmp_path / "inference" / "result.txt").exists()        # one comment writes only the log
