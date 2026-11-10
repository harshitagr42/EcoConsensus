import json

import pytest

from src.llm_api import (BudgetGuard, DailyLimitReached, LLMComponent, TokenLimitError, cache_key, load_prompt,
                         parse_reply)

WORDS = {"hoax": "AGAINST", "serious": "FAVOUR"}


class Stub:
    """Fake provider: labels by keyword; can drop ids, return junk, or raise a token-limit error."""

    def __init__(self, drop_first=0, token_limit_above=None, junk=False):
        self.calls, self.sizes = 0, []
        self.drop_first, self.token_limit_above, self.junk = drop_first, token_limit_above, junk

    def chat(self, system, user, max_tokens):
        items = json.loads(user[user.index("["):])
        self.calls += 1
        self.sizes.append(len(items))
        if self.token_limit_above and len(items) > self.token_limit_above:
            raise TokenLimitError("tokens per minute", retry_after=0)
        if self.junk:
            return "sorry, I cannot help"
        out = [{"id": it["id"], "label": next((v for k, v in WORDS.items() if k in it["comment"]), "NONE")}
               for it in items]
        if self.drop_first and len(out) > 1:
            self.drop_first -= 1
            out = out[1:]
        return "Here you go:\n```json\n" + json.dumps(out) + "\n```"


@pytest.fixture(autouse=True)
def no_env(monkeypatch):
    for k in ("PROVIDER", "MODEL", "API_KEY", "BASE_URL", "DAILY_LIMIT"):
        monkeypatch.setenv(f"LLM_A_{k}", "")


def comp(tmp_path, client, bs=4, **kw):
    return LLMComponent("a", client=client, batch_size=bs, cache_dir=tmp_path, sleep=lambda s: None, **kw)


def test_parse_reply_variants():
    assert parse_reply('[{"id": 1, "label": "FAVOR"}, {"id": 2, "label": "none"}]', 2) == {1: 0, 2: 1}
    assert parse_reply('```json\n[{"id": "1", "label": "Against"}]\n``` done', 1) == {1: 2}
    assert parse_reply('{"results": [{"id": 1, "label": "neutral"}]}', 1) == {1: 1}
    assert parse_reply('[{"id": 1, "label": "FAVOUR"}, {"id": 1, "label": "NONE"}]', 1) == {}   # duplicate id
    assert parse_reply('[{"id": 3, "label": "FAVOUR"}, {"id": 1, "label": "maybe"}]', 2) == {}  # bad id/label
    assert parse_reply('[{"id": 1, "label": "AGAINST"}, {"id": 2, "label": "FAV', 2) == {1: 2}  # truncated
    assert parse_reply("", 3) == {}


def test_prompt_loading():
    p = load_prompt()
    assert p.version.startswith("v") and "{CLAIM}" not in p.system and "{FEW_SHOT}" not in p.system
    assert "Climate change and global warming is a serious concern." in p.system
    msg = p.user_message(['he said "hi"', "एक"])
    assert "Classify these 2 comments." in msg and json.loads(msg[msg.index("["):])[1]["comment"] == "एक"
    fs = load_prompt(few_shot=[("it is a hoax", 2)])
    assert fs.version != p.version and "EXAMPLES" in fs.system and "-> AGAINST" in fs.system


def test_run_and_cache_key_ignores_batch_size(tmp_path):
    comments = [f"comment {i} serious" for i in range(7)] + ["a hoax", "hello"]
    c1 = comp(tmp_path, Stub(), bs=2)
    res = c1.run(comments)
    assert [r["label"] for r in res] == [0] * 7 + [2, 1]
    c2 = comp(tmp_path, s2 := Stub(), bs=5)                  # different batch size, same cache
    res2 = c2.run(comments)
    assert s2.calls == 0 and c2.stats["cache_hits"] == len(comments) and res2 == res
    assert cache_key("c", "x  y") == cache_key("c", "x y")


def test_missing_ids_are_resent(tmp_path):
    c = comp(tmp_path, Stub(drop_first=2), bs=4)
    res = c.run(["a hoax", "serious 1", "b", "serious 2"])
    assert all(r["status"] == "ok" for r in res)
    assert [r["label"] for r in res] == [2, 0, 1, 0]


def test_unparseable_becomes_unavailable_then_stops(tmp_path):
    c = comp(tmp_path, Stub(junk=True), bs=2)
    res = c.run(["x", "y", "z", "w", "v", "u"])
    assert c.stopped and "no usable answer" in c.stopped
    assert all(r["status"] in ("unavailable", "missing") for r in res)


def test_budget_guard_stops_and_resumes(tmp_path):
    comments = [f"serious {i}" for i in range(10)]
    c = comp(tmp_path, Stub(), bs=2)
    c.budget.limit = 2
    res = c.run(comments)
    assert c.stopped and sum(r["status"] == "ok" for r in res) == 4
    assert sum(r["status"] == "missing" for r in res) == 6
    c2 = comp(tmp_path, Stub(), bs=2)                        # "tomorrow": higher limit, resumes from cache
    c2.budget.limit = 100
    assert all(r["status"] == "ok" for r in c2.run(comments)) and c2.stats["cache_hits"] == 4


def test_budget_file(tmp_path):
    b = BudgetGuard("p", "m", 1, tmp_path)
    b.check()
    b.record()
    with pytest.raises(DailyLimitReached):
        BudgetGuard("p", "m", 1, tmp_path).check()


def test_token_limit_shrinks_batch(tmp_path):
    stub = Stub(token_limit_above=3)
    c = comp(tmp_path, stub, bs=8)
    c.min_batch = 2
    res = c.run([f"serious {i}" for i in range(8)])
    assert all(r["status"] == "ok" for r in res)
    assert c.batch_size <= 4 and c.stats["token_backoffs"] >= 1
