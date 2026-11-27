"""Web API (api/) end to end with stub transformers and stub LLMs: upload -> mapping -> job -> results and files,
label mapping (numbers are never guessed, a user's 0 can mean neutral), limits and key rotation."""
import json
import time

import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")
from starlette.testclient import TestClient  # noqa: E402

import src.pipeline as pl  # noqa: E402
from api import app as app_mod  # noqa: E402
from api import jobs, settings  # noqa: E402
from src.llm_api import LLMComponent  # noqa: E402
from tests.test_pipeline import StubLLM, fake_models, stub_transformers  # noqa: E402

# the professor's possible format: numeric labels in a column called "prediction", where 0 means NONE
ROWS = [("a1", "Climate change is a serious threat, act now", "1"),
        ("a2", "global warming is a hoax", "2"),
        ("a3", "I had pasta for lunch", "0"),
        ("a4", "जलवायु परिवर्तन एक hoax है", "2"),
        ("a5", "জলবায়ু পরিবর্তন serious বিষয়", "1")]


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(pl, "load_models", lambda components, need_meta=True: fake_models())
    monkeypatch.setattr(pl, "transformer_probs", stub_transformers)

    def make(slot, claim, batch_size, max_requests, key_index=0):
        return LLMComponent(slot, client=StubLLM(), cache_dir=tmp_path / "cache", batch_size=4,
                            sleep=lambda s: None, key_index=key_index)
    monkeypatch.setattr(pl, "make_llm", make)
    for s in "AB":
        for k, v in dict(PROVIDER="stub", MODEL=f"m{s}", API_KEY="", API_KEYS="k1,k2", BASE_URL="http://x",
                         DAILY_LIMIT="").items():
            monkeypatch.setenv(f"LLM_{s}_{k}", v)
    monkeypatch.setattr(jobs, "JOB_LIMIT", jobs.RateLimiter(2, 10))
    monkeypatch.setattr(jobs, "UPLOAD_LIMIT", jobs.RateLimiter(50))
    app_mod.STATE.update(models="ready", error=None)
    jobs.JOBS.clear()
    with TestClient(app_mod.app) as c:
        app_mod.STATE.update(models="ready", error=None)
        yield c


def upload(client, rows=ROWS, name="prof.csv", cols=("No", "Comment Text", "prediction")):
    csv = pd.DataFrame(rows, columns=list(cols)).to_csv(index=False).encode()
    return client.post("/inspect", files={"file": (name, csv, "text/csv")})


def wait(client, job_id, timeout=60):
    t = time.time()
    while time.time() - t < timeout:
        st = client.get(f"/jobs/{job_id}").json()
        if st["status"] in ("done", "error"):
            return st
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_inspect_guesses_columns_but_never_numeric_labels(client):
    r = upload(client)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["rows"] == 5 and d["columns"] == ["No", "Comment Text", "prediction"]
    assert d["guess"]["comment"] == "Comment Text" and d["guess"]["label"] == "prediction"
    assert d["label_guess"]["prediction"] == {"1": None, "2": None, "0": None}      # numbers: the user decides


def test_target_column_with_labels_is_guessed_as_label(client):
    d = upload(client, [(t, "FAVOR" if i % 2 else "AGAINST") for i, (_, t, _) in enumerate(ROWS)],
               cols=("Tweet", "Target")).json()
    assert d["guess"]["label"] == "Target" and d["guess"]["claim"] is None
    assert d["label_guess"]["Target"] == {"AGAINST": "against", "FAVOR": "favour"}


def test_full_job_with_user_label_mapping(client):
    up = upload(client).json()
    body = dict(upload_id=up["upload_id"], comment_col="Comment Text", label_col="prediction", id_col="No",
                label_map={"0": "neutral", "1": "favour", "2": "against"}, output_style="file")
    r = client.post("/jobs", json=body)
    assert r.status_code == 200, r.text
    st = wait(client, r.json()["id"])
    assert st["status"] == "done", st
    assert [s["state"] for s in st["stages"]] == ["done"] * 7
    res = client.get(f"/jobs/{st['id']}/results").json()["results"]
    assert res["labelled"] == 5 and len(res["preview"]) == 5
    first = res["preview"][0]
    assert first["gold"] == "1" and first["gold_class"] == "favour"
    assert first["llm_a"] == "1" and first["final"] in {"0", "1", "2"}       # written in the file's own values
    stacked = next(m for m in res["metrics"] if m["key"] == "llm_a")
    assert stacked["macro_f1"] == 1.0                     # stub LLM is right on every row once 0 = neutral is mapped
    pred = client.get(f"/jobs/{st['id']}/files/predictions.csv")
    assert pred.status_code == 200 and "attachment" in pred.headers["content-disposition"]
    df = pd.read_csv(pd.io.common.StringIO(pred.text), dtype=str)
    assert set(df["Predicted_Label"]) <= {"0", "1", "2"}
    assert client.get(f"/jobs/{st['id']}/files/all.zip").status_code == 200
    assert client.get(f"/jobs/{st['id']}/files/../../etc").status_code == 404


def test_no_label_needs_confirmation_and_unmapped_values_are_rejected(client):
    up = upload(client).json()
    r = client.post("/jobs", json=dict(upload_id=up["upload_id"], comment_col="Comment Text"))
    assert r.status_code == 400 and "confirm" in r.json()["detail"]
    r = client.post("/jobs", json=dict(upload_id=up["upload_id"], comment_col="Comment Text", label_col="prediction",
                                       label_map={"0": "maybe"}))
    assert r.status_code == 400
    r = client.post("/jobs", json=dict(upload_id=up["upload_id"], comment_col="Comment Text", confirm_no_labels=True))
    st = wait(client, r.json()["id"])
    res = client.get(f"/jobs/{st['id']}/results").json()["results"]
    assert st["status"] == "done" and res["metrics"] == [] and res["output_style"] == "numeric"


def test_limits_are_enforced_on_the_server(client, monkeypatch):
    lim = client.get("/limits").json()
    assert lim["max_rows"] == settings.MAX_ROWS and lim["you"]["left_this_hour"] == 2
    up = upload(client).json()
    body = dict(upload_id=up["upload_id"], comment_col="Comment Text", confirm_no_labels=True)
    for _ in range(2):
        wait(client, client.post("/jobs", json=body).json()["id"])
    r = client.post("/jobs", json=body)
    assert r.status_code == 429 and int(r.headers["retry-after"]) > 0
    monkeypatch.setattr(settings, "MAX_ROWS", 3)
    assert upload(client).status_code == 400
    assert client.post("/inspect", files={"file": ("x.xlsx", b"abc", "application/octet-stream")}).status_code == 400


def test_llm_switches_to_second_key_on_daily_quota(tmp_path, monkeypatch):
    from src.llm_api import DailyLimitReached

    class QuotaLLM(StubLLM):
        def chat(self, system, user, max_tokens):
            raise DailyLimitReached("quota", category="DAILY QUOTA USED UP (provider limit; resume tomorrow)")
    for k, v in dict(PROVIDER="stub", MODEL="m", API_KEYS="k1,k2", BASE_URL="http://x", DAILY_LIMIT="").items():
        monkeypatch.setenv(f"LLM_A_{k}", v)
    used = []

    def make(slot, claim, batch_size, max_requests, key_index=0):
        used.append(key_index)
        return LLMComponent(slot, client=QuotaLLM() if key_index == 0 else StubLLM(), cache_dir=tmp_path,
                            batch_size=4, sleep=lambda s: None, key_index=key_index)
    df = pl.rows_from_texts([t for _, t, _ in ROWS])
    events = []
    labels, info = pl.llm_labels("a", df, factory=make, progress=events.append)
    assert used == [0, 1] and info["key_used"] == 2 and info["status"] == "ok"
    assert all(l is not None for l in labels) and info["key_switches"]
    assert any(e["event"] == "key_switch" for e in events)
    # each key has its own daily counter
    assert (tmp_path / "usage__stub__m__key2.json").exists() and (tmp_path / "usage__stub__m.json").exists()
    json.loads((tmp_path / "usage__stub__m__key2.json").read_text())


def test_numbered_keys(monkeypatch):
    from src.llm_api import api_keys
    for k in ("API_KEYS", "API_KEY", "API_KEY_1", "API_KEY_2", "API_KEY_3"):
        monkeypatch.delenv(f"LLM_A_{k}", raising=False)
    monkeypatch.setenv("LLM_A_API_KEY", "single")
    assert api_keys("a") == ["single"]
    monkeypatch.setenv("LLM_A_API_KEY_1", "k1")
    monkeypatch.setenv("LLM_A_API_KEY_2", "k2")
    assert api_keys("a") == ["k1", "k2"]                 # numbered keys win over the single one
    monkeypatch.setenv("LLM_A_API_KEYS", "x,y")
    assert api_keys("a") == ["x", "y"]


def test_visitor_ip_comes_from_cloudflare(client):
    a = {"CF-Connecting-IP": "203.0.113.1", "X-Forwarded-For": "1.2.3.4"}
    up = client.post("/inspect", headers=a, files={"file": ("p.csv", pd.DataFrame(ROWS, columns=["No", "Comment Text", "prediction"]).to_csv(index=False).encode(), "text/csv")}).json()
    body = dict(upload_id=up["upload_id"], comment_col="Comment Text", confirm_no_labels=True)
    wait(client, client.post("/jobs", json=body, headers=a).json()["id"])
    assert client.get("/limits", headers=a).json()["you"]["left_this_hour"] == 1
    other = {"CF-Connecting-IP": "203.0.113.2", "X-Forwarded-For": "203.0.113.1"}     # spoofed XFF is ignored
    assert client.get("/limits", headers=other).json()["you"]["left_this_hour"] == 2


def test_evaluator_output_file(client):
    """input.csv (Id, text) -> output_04.csv (Id, label): same Ids in the same order, labels 0/1/2 by default,
    another spelling on request, and a label key."""
    rows = [("x9", "global warming is a hoax"), ("007", "Climate change is a serious threat, act now"),
            ("a1", "I had pasta for lunch"), ("b2", "   ")]
    csv = pd.DataFrame(rows, columns=["Id", "text"]).to_csv(index=False).encode()
    up = client.post("/inspect", files={"file": ("input.csv", csv, "text/csv")}).json()
    assert up["guess"]["id"] == "Id" and up["guess"]["comment"] == "text" and up["guess"]["label"] is None
    for style, allowed in (("numeric", {"0", "1", "2"}), ("upper", {"FAVOR", "NONE", "AGAINST"})):
        body = dict(upload_id=up["upload_id"], comment_col="text", id_col="Id", confirm_no_labels=True)
        if style != "numeric":
            body["output_style"] = style                         # numeric is the default
        st = wait(client, client.post("/jobs", json=body).json()["id"])
        assert st["status"] == "done", st
        r = client.get(f"/jobs/{st['id']}/files/{settings.OUTPUT_FILE}")
        assert r.status_code == 200 and settings.OUTPUT_FILE in r.headers["content-disposition"]
        out = pd.read_csv(pd.io.common.StringIO(r.text), dtype=str, keep_default_na=False)
        assert list(out.columns) == ["Id", "label"]
        assert out["Id"].tolist() == ["x9", "007", "a1", "b2"]          # unchanged, same order, empty text included
        assert set(out["label"]) <= allowed and all(out["label"])
        key = client.get(f"/jobs/{st['id']}/files/label_key.txt").text
        assert "favour" in key and "against" in key
        res = client.get(f"/jobs/{st['id']}/results").json()["results"]
        assert res["output_file"] == settings.OUTPUT_FILE and set(res["label_key"].values()) == allowed
