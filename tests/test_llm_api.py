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


# ---------------------------------------------------------------- error categories and empty-reply diagnosis
class FakeResponse:
    def __init__(self, status, body, headers=None):
        self.status_code, self._body, self.headers = status, body, headers or {}
        self.text = json.dumps(body)

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, *responses):
        self.responses, self.bodies = list(responses), []

    def post(self, url, json=None, **kw):
        self.bodies.append(json)
        return self.responses.pop(0)


def http_client(*responses):
    from src.llm_api import OpenAICompatibleClient, ProviderConfig
    cfg = ProviderConfig("b", "groq", "m", "k", "https://x/v1", None, {"reasoning_effort": "low"})
    return OpenAICompatibleClient(cfg, session=FakeSession(*responses))


@pytest.mark.parametrize("status,message,error,category", [
    (401, "Invalid API Key", "FatalProviderError", "AUTH ERROR"),
    (404, "model not found", "FatalProviderError", "NOT FOUND"),
    (429, "on tokens per minute (TPM): Limit 8000, Used 5505, Requested 2886.", "RateLimited", "RATE LIMIT (tokens"),
    (429, "on tokens per minute (TPM): Limit 8000, Used 10, Requested 9000.", "TokenLimitError", "REQUEST TOO LARGE"),
    (429, "Rate limit reached on requests per minute (RPM)", "RateLimited", "RATE LIMIT (requests"),
    (429, "Rate limit reached on requests per day (RPD)", "DailyLimitReached", "DAILY QUOTA"),
    (503, "overloaded", "TransientError", "PROVIDER ERROR"),
    (400, "unknown parameter reasoning_effort", "FatalProviderError", "BAD REQUEST"),
])
def test_http_errors_are_categorised(status, message, error, category):
    c = http_client(FakeResponse(status, {"error": {"message": message, "type": "x"}}))
    with pytest.raises(Exception) as ei:
        c.chat("s", "u", 10)
    assert type(ei.value).__name__ == error
    assert ei.value.category.startswith(category) and message in str(ei.value)


def test_extra_body_is_sent_and_reply_details_kept():
    ok = {"choices": [{"message": {"content": "hi", "reasoning": "abc"}, "finish_reason": "stop"}],
          "usage": {"completion_tokens": 5}}
    c = http_client(FakeResponse(200, ok))
    assert c.chat("s", "u", 10) == "hi"
    assert c.session.bodies[0]["reasoning_effort"] == "low"
    assert c.last_info["finish_reason"] == "stop" and c.last_info["reasoning_chars"] == 3


def test_explain_unusable():
    from src.llm_api import explain_unusable
    cut = explain_unusable("", {"finish_reason": "length", "max_tokens": 110,
                                "usage": {"completion_tokens": 110, "completion_tokens_details": {"reasoning_tokens": 108}}},
                           3, "LLM_B")
    assert cut.startswith("OUTPUT CUT OFF") and "108 of them reasoning" in cut and "LLM_B_EXTRA_BODY" in cut
    assert explain_unusable("", {"finish_reason": "content_filter"}, 3, "LLM_B").startswith("BLOCKED")
    assert explain_unusable("", {"finish_reason": "stop"}, 3, "LLM_B").startswith("EMPTY REPLY")
    assert explain_unusable("I think...", {"finish_reason": "stop"}, 3, "LLM_B").startswith("UNPARSEABLE")


class Thinker(Stub):
    """Reasoning model: the reply is cut off until max_tokens is large enough."""

    def __init__(self, need):
        super().__init__()
        self.need, self.limits = need, []

    def chat(self, system, user, max_tokens):
        self.limits.append(max_tokens)
        if max_tokens < self.need:
            self.last_info = {"finish_reason": "length", "usage": {"completion_tokens": max_tokens}}
            return ""
        self.last_info = {"finish_reason": "stop"}
        return super().chat(system, user, max_tokens)


def test_cut_off_replies_raise_max_tokens(tmp_path):
    t = Thinker(need=400)
    c = comp(tmp_path, t, bs=4)
    res = c.run(["a hoax", "serious", "x", "y"])
    assert all(r["status"] == "ok" for r in res)
    assert t.limits[0] < 400 <= t.limits[-1] and c.stats["parse_failures"] == 0
    assert "OUTPUT CUT OFF (max_tokens raised)" in c.stats["errors"]


def test_stop_reason_names_the_cause(tmp_path):
    c = comp(tmp_path, Stub(junk=True), bs=2)
    c.run(["a", "b", "c", "d", "e", "f", "g", "h"])
    assert c.stopped and c.stopped.startswith("UNPARSEABLE REPLY") and "sorry" in c.stopped


class Busy(Stub):
    """Per-minute rate limit: the first `n` requests are rejected with a 429 that the batch fits into."""

    def __init__(self, n):
        super().__init__()
        self.n = n

    def chat(self, system, user, max_tokens):
        from src.llm_api import RateLimited
        if self.n:
            self.n -= 1
            raise RateLimited("TPM: Limit 8000, Used 7000, Requested 2000", retry_after=3, category="RATE LIMIT")
        return super().chat(system, user, max_tokens)


def test_rate_limit_waits_and_resends_same_batch(tmp_path):
    waits = []
    stub = Busy(3)
    c = LLMComponent("a", client=stub, batch_size=4, cache_dir=tmp_path, sleep=waits.append)
    res = c.run(["a hoax", "serious", "x", "y", "z"])
    assert all(r["status"] == "ok" for r in res)
    assert c.batch_size == 4 and stub.sizes == [4, 1]           # batch never shrunk, nothing split
    assert c.stats["requests"] == 2 and c.budget.today == 2      # rejected 429s are not counted
    assert c.stats["rate_limit_waits"] == 3 and waits == [3.5, 3.5, 3.5]


class Window(Stub):
    """Reports a tokens-per-minute window in last_info like the real client."""

    def chat(self, system, user, max_tokens):
        self.last_info = {"tokens_left": 100, "tokens_reset_sec": 12.0, "usage": {"prompt_tokens": 500}}
        return super().chat(system, user, max_tokens)


def test_paces_when_window_is_nearly_used(tmp_path):
    waits = []
    c = LLMComponent("a", client=Window(), batch_size=2, cache_dir=tmp_path, sleep=waits.append)
    c.run(["a", "b", "c", "d"])
    assert waits == [12.5] and c.stats["rate_limit_waits"] == 1   # waited once before the second request


def test_duration_parsing():
    from src.llm_api import _duration
    assert _duration("3") == 3 and _duration("54.187ms") == 0.054187 and _duration("1m30.5s") == 90.5
    assert _duration("5h24m0s") == 5 * 3600 + 24 * 60 and _duration(None) is None
