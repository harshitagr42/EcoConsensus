"""Provider-agnostic LLM stance classifier: OpenAI-compatible chat API, batching with local ids 1..N,
reply validation and repair, per-comment append-only cache, daily budget guard, retries and token-limit backoff.

Configuration: .env (LLM_A_*, LLM_B_*, LLM_BATCH_SIZE) + configs/llm.yaml + prompts/stance_system_prompt.md.
Several keys per slot (e.g. two free accounts): LLM_A_API_KEY_1, LLM_A_API_KEY_2 (or LLM_A_API_KEYS=key1,key2); the pipeline
moves to the next key when one stops on its daily quota or an auth error (src/pipeline.py llm_labels).

    comp = LLMComponent("a")
    results = comp.run(comments)          # list of {"label": 0/1/2 or None, "status": ..., "cached": bool}

Cache: outputs/llm/cache/<provider>__<model>__<prompt_version>.jsonl, one JSON record per comment, keyed by
SHA-256 of the preprocessed claim + comment. The batch size is not part of the key.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import requests

from src.config import CFG, FIXED_CLAIM, LABELS, ROOT, env, path
from src.preprocessing import preprocess
from src.utils import stable_hash, write_json

LLM_CFG = CFG["llm"]
log = logging.getLogger("llm_api")

# LLM answer -> label id (0 favour, 1 neutral, 2 against)
ANSWER_TO_ID = {"favour": 0, "favor": 0, "none": 1, "neutral": 1, "against": 2}
STATUS_OK, STATUS_UNAVAILABLE, STATUS_MISSING = "ok", "unavailable", "missing"

_DAILY_HINTS = ("per day", "perday", "daily", "requests per day", "rpd", "tokens per day", "tpd",
                "quota exceeded", "exceeded your current quota", "insufficient_quota", "free-models-per-day")
_MINUTE_HINTS = ("per minute", "perminute", "rpm", "tpm", "requests per min", "tokens per min")
_TOKEN_HINTS = ("tokens per minute", "tpm", "token limit", "too many tokens", "context length", "context_length",
                "maximum context", "request too large", "reduce the length", "max_tokens")


# ---------------------------------------------------------------- errors
class LLMError(Exception):
    """`category` says what went wrong and on whose side, e.g. "RATE LIMIT (requests per minute)"."""

    def __init__(self, msg, retry_after: float | None = None, category: str | None = None):
        super().__init__(msg)
        self.retry_after = retry_after
        self.category = category or type(self).__name__


class StopRun(LLMError):
    """Stop the run cleanly; everything answered so far is cached and the same command resumes later."""


class DailyLimitReached(StopRun):
    """Budget guard or the provider's daily quota."""


class ProviderUnavailable(StopRun):
    """Repeated network/5xx failures or a run of unparseable replies."""


class TokenLimitError(LLMError):
    pass


class TransientError(LLMError):
    pass


class RateLimited(TransientError):
    """HTTP 429 for a per-minute window (requests or tokens) that this request fits into: wait, resend unchanged.
    The provider does not count a rejected request against the daily quota, so neither does the budget guard."""


class FatalProviderError(LLMError):
    """Bad key, unknown model, bad URL: retrying will not help."""


# ---------------------------------------------------------------- config
@dataclass
class ProviderConfig:
    slot: str                      # "a" or "b"
    provider: str | None
    model: str | None
    api_key: str | None
    base_url: str | None
    daily_limit: int | None
    extra_body: dict | None = None  # LLM_<S>_EXTRA_BODY (JSON), e.g. {"reasoning_effort": "low"}

    key_index: int = 0              # which of LLM_<S>_API_KEYS (0 = the first, or LLM_<S>_API_KEY)

    @classmethod
    def from_env(cls, slot: str, key_index: int = 0) -> "ProviderConfig":
        s = f"LLM_{slot.upper()}_"
        keys = api_keys(slot)
        if key_index >= max(1, len(keys)):
            raise FatalProviderError(f"{s}API_KEYS has {len(keys)} key(s); key {key_index + 1} does not exist",
                                     category="CONFIG ERROR (your side)")
        lim = env(s + "DAILY_LIMIT")
        extra = env(s + "EXTRA_BODY")
        try:
            extra_body = json.loads(extra) if extra else None
        except json.JSONDecodeError as e:
            raise FatalProviderError(f"{s}EXTRA_BODY in .env is not valid JSON ({e}): {extra!r}",
                                     category="CONFIG ERROR (your side)") from None
        return cls(slot=slot.lower(), provider=env(s + "PROVIDER"), model=env(s + "MODEL"),
                   api_key=keys[key_index] if keys else None, base_url=env(s + "BASE_URL"),
                   daily_limit=int(lim) if lim else None, extra_body=extra_body, key_index=key_index)

    @property
    def name(self) -> str:                         # component name used in features and reports
        return f"llm_{self.slot}"

    def missing(self) -> list[str]:
        need = {"PROVIDER": self.provider, "MODEL": self.model, "API_KEY": self.api_key, "BASE_URL": self.base_url}
        return [f"LLM_{self.slot.upper()}_{k}" for k, v in need.items() if not v]

    def __repr__(self) -> str:                     # never print the key
        return f"ProviderConfig(slot={self.slot}, provider={self.provider}, model={self.model}, " \
               f"base_url={self.base_url}, daily_limit={self.daily_limit}" + \
               (f", extra_body={self.extra_body})" if self.extra_body else ")")


def api_keys(slot: str) -> list[str]:
    """The keys of one LLM slot, in the order they are used. Keys are never logged. First that is set:
    LLM_<S>_API_KEYS (comma-separated), LLM_<S>_API_KEY_1, _2, ... (numbered, from 1, stops at the first gap),
    LLM_<S>_API_KEY."""
    s = f"LLM_{slot.upper()}_"
    many = [k.strip() for k in (env(s + "API_KEYS") or "").split(",") if k.strip()]
    if many:
        return many
    numbered, i = [], 1
    while env(f"{s}API_KEY_{i}"):
        numbered.append(env(f"{s}API_KEY_{i}").strip())
        i += 1
    if numbered:
        return numbered
    one = env(s + "API_KEY")
    return [one] if one else []


def default_batch_size() -> int:
    return int(env("LLM_BATCH_SIZE") or LLM_CFG.get("batch_size", 50))


# ---------------------------------------------------------------- prompt
def _code_block_after(text: str, heading: str) -> str:
    i = text.find(heading)
    if i < 0:
        raise ValueError(f"heading '{heading}' not found in prompt file")
    m = re.search(r"```text\n(.*?)\n```", text[i:], re.S)
    if not m:
        raise ValueError(f"no ```text block after '{heading}'")
    return m.group(1)


@dataclass
class Prompt:
    version: str
    system: str
    user_template: str

    def user_message(self, comments: list[str]) -> str:
        items = [{"id": i, "comment": c} for i, c in enumerate(comments, 1)]
        return self.user_template.replace("{N}", str(len(comments))).replace(
            "{ITEMS}", json.dumps(items, ensure_ascii=False, indent=0))


def load_prompt(prompt_file: str | Path | None = None, claim: str = FIXED_CLAIM,
                few_shot: list[tuple[str, int]] | None = None) -> Prompt:
    """few_shot: (preprocessed comment, label id) pairs from train_base. They change the version (and cache key)."""
    p = ROOT / (prompt_file or LLM_CFG["prompt_file"])
    text = p.read_text(encoding="utf-8")
    m = re.search(r"^prompt_version:\s*(\S+)", text, re.M)
    if not m:
        raise ValueError(f"{p}: no 'prompt_version:' line")
    version = m.group(1)
    system = _code_block_after(text, "## System prompt")
    block = ""
    if few_shot:
        names = {0: "FAVOUR", 1: "NONE", 2: "AGAINST"}
        lines = "\n".join(f'Comment: {json.dumps(c, ensure_ascii=False)} -> {names[int(y)]}' for c, y in few_shot)
        block = _code_block_after(text, "## Few-shot block").replace("{EXAMPLES}", lines) + "\n"
        version = f"{version}-fs{stable_hash([(c, int(y)) for c, y in few_shot], 6)}"
    system = system.replace("{CLAIM}", claim).replace("{FEW_SHOT}", block)
    return Prompt(version=version, system=system, user_template=_code_block_after(text, "## User message template"))


def few_shot_from_config() -> list[tuple[str, int]] | None:
    ids = LLM_CFG.get("few_shot_ids") or []
    if not ids:
        return None
    import pandas as pd
    tb = pd.read_csv(path("processed") / "train_base.csv", dtype={"id": str}, keep_default_na=False)
    rows = tb.set_index("id").reindex(ids)
    if rows["comment"].isna().any():
        bad = [i for i, c in zip(ids, rows["comment"]) if not isinstance(c, str)]
        raise ValueError(f"few_shot_ids not found in train_base (few-shot examples must come from train_base): {bad}")
    return list(zip(rows["comment"], rows["label"].astype(int)))


# ---------------------------------------------------------------- reply parsing
def parse_reply(text: str, n: int) -> dict[int, int]:
    """Parse an LLM reply for a request with ids 1..n -> {local id: label id}. Items with an invalid label,
    an unknown id, or an id that appears more than once are left out (the caller re-sends them)."""
    if not text:
        return {}
    t = re.sub(r"```(?:json)?", "", text).strip()
    items = None
    s, e = t.find("["), t.rfind("]")
    if s >= 0 and e > s:
        try:
            items = json.loads(t[s:e + 1])
        except json.JSONDecodeError:
            items = None
    if items is None and t.startswith("{"):
        try:
            obj = json.loads(t)
            lists = [v for v in obj.values() if isinstance(v, list)] if isinstance(obj, dict) else []
            items = lists[0] if lists else ([obj] if isinstance(obj, dict) else None)
        except json.JSONDecodeError:
            items = None
    if items is None:      # last resort: pick {"id": .., "label": ..} pairs out of broken JSON
        items = [{"id": a, "label": b} for a, b in
                 re.findall(r'"id"\s*:\s*"?(\d+)"?\s*,\s*"label"\s*:\s*"([A-Za-z]+)"', t)]
    seen: dict[int, int | None] = {}
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        try:
            i = int(str(it.get("id")).strip())
        except (TypeError, ValueError):
            continue
        lab = ANSWER_TO_ID.get(str(it.get("label", "")).strip().lower())
        if i < 1 or i > n:
            continue
        seen[i] = None if i in seen else lab          # duplicates are invalid
    return {i: l for i, l in seen.items() if l is not None}


# ---------------------------------------------------------------- cache and budget
def cache_key(claim: str, comment: str) -> str:
    """Independent of provider files, batch size and request position: only the cleaned texts."""
    return hashlib.sha256((preprocess(claim) + "\n" + preprocess(comment)).encode("utf-8")).hexdigest()


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s)).strip("-")


_FILE_LOCKS: dict[Path, threading.Lock] = {}
_FILE_LOCKS_GUARD = threading.Lock()


def _file_lock(p: Path) -> threading.Lock:
    """One lock per cache/usage file, shared by every component in this process. LLM-A and LLM-B run in parallel
    threads and use the same files when they are set to the same provider and model."""
    with _FILE_LOCKS_GUARD:
        return _FILE_LOCKS.setdefault(Path(p).resolve(), threading.Lock())


class LLMCache:
    """Append-only JSONL. The last record for a key wins (so a later 'ok' replaces an 'unavailable')."""

    def __init__(self, provider: str, model: str, prompt_version: str, cache_dir: Path | None = None):
        d = Path(cache_dir) if cache_dir else path("cache", create=True)
        d.mkdir(parents=True, exist_ok=True)
        self.path = d / f"{_slug(provider)}__{_slug(model)}__{_slug(prompt_version)}.jsonl"
        self.meta = dict(provider=provider, model=model, prompt_version=prompt_version)
        self.records: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                    self.records[r["key"]] = r
                except (json.JSONDecodeError, KeyError):
                    continue                           # a half-written last line after a crash

    def get(self, key: str) -> dict | None:
        return self.records.get(key)

    def put_many(self, items: list[tuple[str, int | None, str]]) -> None:
        """items: (key, label id or None, status)."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with _file_lock(self.path), open(self.path, "a", encoding="utf-8") as f:
            for key, label, status in items:
                r = dict(key=key, label=LABELS[label] if label is not None else None, status=status,
                         timestamp=now, **self.meta)
                self.records[key] = r
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()


class BudgetGuard:
    """Counts requests per provider/model per (local) day in outputs/llm/cache/usage__<provider>__<model>.json."""

    def __init__(self, provider: str, model: str, daily_limit: int | None, cache_dir: Path | None = None,
                 key_index: int = 0):
        """Each API key has its own counter (key 1 keeps the original file name)."""
        d = Path(cache_dir) if cache_dir else path("cache", create=True)
        suffix = f"__key{key_index + 1}" if key_index else ""
        self.path = d / f"usage__{_slug(provider)}__{_slug(model)}{suffix}.json"
        self.limit = daily_limit
        self.usage = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    @property
    def today(self) -> int:
        return int(self.usage.get(date.today().isoformat(), 0))

    def remaining(self) -> int | None:
        return None if self.limit is None else max(0, self.limit - self.today)

    def check(self) -> None:
        if self.limit is not None and self.today >= self.limit:
            raise DailyLimitReached(f"daily budget of {self.limit} requests used ({self.path.name})",
                                    category="DAILY BUDGET USED UP (your LLM_<slot>_DAILY_LIMIT in .env; resume tomorrow)")

    def record(self, n: int = 1) -> None:
        k = date.today().isoformat()
        with _file_lock(self.path):            # re-read: another component may share this file (same provider/model)
            if self.path.exists():
                self.usage = json.loads(self.path.read_text(encoding="utf-8"))
            self.usage[k] = int(self.usage.get(k, 0)) + n
            write_json(self.path, self.usage)


# ---------------------------------------------------------------- HTTP client
class OpenAICompatibleClient:
    """POST {base_url}/chat/completions. Errors are mapped onto the LLMError classes above."""

    def __init__(self, cfg: ProviderConfig, session: requests.Session | None = None):
        self.cfg = cfg
        self.session = session or requests.Session()
        self.url = cfg.base_url.rstrip("/") + ("" if cfg.base_url.rstrip("/").endswith("/chat/completions")
                                               else "/chat/completions")

    def chat(self, system: str, user: str, max_tokens: int) -> str:
        """Return the reply text. Details of the last reply (finish_reason, token usage, reasoning) are kept in
        `self.last_info` so an empty or cut-off reply can be explained. Errors are raised with a `category`."""
        body = {"model": self.cfg.model, "temperature": LLM_CFG.get("temperature", 0), "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if LLM_CFG.get("json_mode"):
            body["response_format"] = {"type": "json_object"}
        body.update(LLM_CFG.get("extra_body") or {})
        body.update(self.cfg.extra_body or {})
        self.last_info = {}
        slot = f"LLM_{self.cfg.slot.upper()}"
        t0 = time.perf_counter()
        try:
            r = self.session.post(self.url, json=body, timeout=LLM_CFG.get("request_timeout_sec", 120),
                                  headers={"Authorization": f"Bearer {self.cfg.api_key}",
                                           "Content-Type": "application/json"})
        except requests.Timeout:
            raise TransientError(f"no answer within {LLM_CFG.get('request_timeout_sec', 120)} s",
                                 category="TIMEOUT (network or slow provider)") from None
        except requests.ConnectionError as e:
            raise TransientError(f"could not connect to {self.cfg.base_url}: {str(e)[:150]}",
                                 category="NETWORK ERROR (your connection or the provider is down)") from None
        limits = _limit_headers(r)
        self.last_info = dict(tokens_left=_int(r.headers.get("x-ratelimit-remaining-tokens")),
                              tokens_reset_sec=_duration(r.headers.get("x-ratelimit-reset-tokens")))
        if r.status_code == 200:
            try:
                data = r.json()
                choice = data["choices"][0]
                msg = choice.get("message") or {}
            except (ValueError, KeyError, IndexError, TypeError):
                raise TransientError(f"unexpected 200 body: {r.text[:200]}",
                                     category="BAD RESPONSE (provider side)") from None
            reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
            self.last_info.update(finish_reason=choice.get("finish_reason"), usage=data.get("usage") or {},
                                  reasoning_chars=len(reasoning), refusal=msg.get("refusal"),
                                  max_tokens=max_tokens, seconds=round(time.perf_counter() - t0, 1), limits=limits)
            return msg.get("content") or ""
        detail = _provider_message(r)
        low = (detail + " " + r.text[:2000]).lower()
        ra = _retry_after(r)
        where = f"HTTP {r.status_code}: {detail}" + (f" [{limits}]" if limits else "")
        if r.status_code in (401, 403):
            raise FatalProviderError(where, category=f"AUTH ERROR (your side: check {slot}_API_KEY, and that the key "
                                                     f"belongs to {self.cfg.base_url})")
        if r.status_code == 404:
            raise FatalProviderError(where, category=f"NOT FOUND (your side: check {slot}_MODEL and {slot}_BASE_URL)")
        if r.status_code == 429:
            per_minute = any(h in low for h in _MINUTE_HINTS)
            if not per_minute and any(h in low for h in _DAILY_HINTS):
                raise DailyLimitReached(where, ra, category="DAILY QUOTA USED UP (provider limit; resume tomorrow)")
            if any(h in low for h in _TOKEN_HINTS) or "token" in low:
                m = re.search(r"limit\D{0,3}(\d+)\D+used\D{0,3}(\d+)\D+requested\D{0,3}(\d+)", low)
                if m and int(m.group(3)) > int(m.group(1)):     # this request alone is bigger than the whole window
                    raise TokenLimitError(where, ra, category="REQUEST TOO LARGE for the tokens-per-minute limit "
                                                              "(batch gets smaller)")
                raise RateLimited(where, ra, category="RATE LIMIT (tokens per minute; waiting, then resending the "
                                                      "same batch)")
            raise RateLimited(where, ra, category="RATE LIMIT (requests per minute; waiting, then resending)")
        if r.status_code == 413 or (r.status_code == 400 and any(h in low for h in _TOKEN_HINTS)):
            raise TokenLimitError(where, ra, category="REQUEST TOO LARGE (batch gets smaller)")
        if r.status_code >= 500 or r.status_code == 408:
            raise TransientError(where, ra, category="PROVIDER ERROR (their side; retrying)")
        if r.status_code in (400, 422):
            raise FatalProviderError(where, category=f"BAD REQUEST (your side: a parameter the provider rejects; check "
                                                     f"{slot}_MODEL and {slot}_EXTRA_BODY / extra_body in llm.yaml)")
        raise FatalProviderError(where, category=f"HTTP {r.status_code} (unexpected)")


def _provider_message(r) -> str:
    """The provider's own error text, e.g. Groq's {"error": {"message": ...}} or Gemini's [{"error": {...}}]."""
    try:
        j = r.json()
    except ValueError:
        return r.text[:300].strip() or "(empty body)"
    if isinstance(j, list) and j:
        j = j[0]
    e = j.get("error", j) if isinstance(j, dict) else j
    if isinstance(e, dict):
        parts = [str(e.get(k)) for k in ("code", "status", "type") if e.get(k)]
        return (" / ".join(parts) + ": " if parts else "") + str(e.get("message", e))[:400]
    return str(e)[:400]


def _limit_headers(r) -> str:
    """Remaining quota as the provider reports it (Groq / OpenAI-style x-ratelimit-* headers)."""
    keys = [("x-ratelimit-remaining-requests", "requests left"), ("x-ratelimit-remaining-tokens", "tokens left"),
            ("x-ratelimit-reset-requests", "requests reset in"), ("x-ratelimit-reset-tokens", "tokens reset in"),
            ("retry-after", "retry after")]
    return ", ".join(f"{label} {r.headers[h]}" for h, label in keys if r.headers.get(h))


def explain_unusable(text: str, info: dict, n: int, slot: str) -> str:
    """Why a 200 reply gave no usable answer, in words, with what to change."""
    fr, usage = info.get("finish_reason"), info.get("usage") or {}
    ct = usage.get("completion_tokens")
    rt = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    think = (f", {rt} of them reasoning" if rt else "") + \
            (f", {info['reasoning_chars']} characters of reasoning text" if info.get("reasoning_chars") else "")
    fix_reasoning = (f"This is usually a reasoning model thinking until the limit: set {slot}_EXTRA_BODY="
                     f'{{"reasoning_effort": "low"}} in .env, or raise max_tokens_cap in configs/llm.yaml.')
    if fr == "length":
        return (f"OUTPUT CUT OFF (provider stopped at max_tokens={info.get('max_tokens')} for {n} comments; "
                f"{ct} completion tokens{think}). {fix_reasoning if (rt or info.get('reasoning_chars')) or not text.strip() else 'Raise max_tokens_per_comment in configs/llm.yaml.'}")
    if fr in ("content_filter", "safety", "recitation") or info.get("refusal"):
        return f"BLOCKED BY THE PROVIDER'S SAFETY FILTER (finish_reason={fr}, refusal={info.get('refusal')!r})."
    if not text.strip():
        extra = f" {fix_reasoning}" if (rt or info.get("reasoning_chars")) else ""
        return f"EMPTY REPLY (finish_reason={fr}, {ct} completion tokens{think}).{extra}"
    return (f"UNPARSEABLE REPLY (finish_reason={fr}): no JSON list of {{id, label}} found. "
            f"Reply starts: {text[:200]!r}")


def _duration(v) -> float | None:
    """Seconds from "2", "2.5s", "54.187ms", "1m30.5s", "5h24m0s"."""
    if not v:
        return None
    m = re.match(r"^\s*([\d.]+)\s*(ms|s)?\s*$", str(v))
    if m:
        return float(m.group(1)) / (1000 if m.group(2) == "ms" else 1)
    m = re.findall(r"([\d.]+)(ms|h|m|s)", str(v))
    if m:
        return sum(float(x) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[u] for x, u in m)
    return None


def _int(v) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _retry_after(r) -> float | None:
    """retry-after first (it is the exact wait for this request), then the window resets."""
    for h in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        d = _duration(r.headers.get(h))
        if d is not None:
            return d
    return None


class _NoRequests:
    def chat(self, *a, **k):
        raise FatalProviderError("this component was created with lookup_only=True")


# ---------------------------------------------------------------- the component
class LLMComponent:
    """One LLM (slot a or b). `run()` is all most callers need."""

    def __init__(self, slot: str, prompt: Prompt | None = None, batch_size: int | None = None,
                 client=None, cache_dir: Path | None = None, sleep=time.sleep, max_requests: int | None = None,
                 lookup_only: bool = False, key_index: int = 0):
        """lookup_only: only PROVIDER and MODEL are needed (to find the cache file); no request can be sent.
        key_index: which of LLM_<S>_API_KEYS to use (the cache is shared by all keys of a provider/model)."""
        self.cfg = ProviderConfig.from_env(slot, key_index)
        if lookup_only:
            if not (self.cfg.provider and self.cfg.model):
                raise FatalProviderError(f"{self.cfg.name}: set LLM_{slot.upper()}_PROVIDER and _MODEL in .env "
                                         f"to read its cache")
            client = client or _NoRequests()
        if client is None and self.cfg.missing():
            raise FatalProviderError(f"{self.cfg.name} not configured; set {', '.join(self.cfg.missing())} in .env")
        self.prompt = prompt or load_prompt(few_shot=few_shot_from_config())
        self.batch_size = int(batch_size or default_batch_size())
        self.min_batch = int(LLM_CFG.get("min_batch_size", 5))
        self.client = client or OpenAICompatibleClient(self.cfg)
        provider, model = self.cfg.provider or "stub", self.cfg.model or "stub"
        self.cache = LLMCache(provider, model, self.prompt.version, cache_dir)
        self.budget = BudgetGuard(provider, model, self.cfg.daily_limit, cache_dir, key_index)
        self.sleep = sleep
        self.max_requests = max_requests          # optional cap for this run (on top of the daily budget)
        self.stats = dict(requests=0, cache_hits=0, sent=0, ok=0, unavailable=0, parse_failures=0,
                          missing_ids=0, token_backoffs=0, retries=0, rate_limit_waits=0, errors={})
        self.reasoning_tokens = int(LLM_CFG.get("reasoning_tokens", 0))   # grows when replies are cut off
        self._tokens_left: int | None = None      # tokens-per-minute window, from the last reply's headers
        self._tokens_reset: float | None = None
        self._last_cost: int | None = None        # prompt + max_tokens of the last accepted request
        self.last_problem: str | None = None
        self.stopped: str | None = None           # reason if the run stopped early
        # why: "quota" (daily quota / budget), "cap" (max_requests), "fatal" (auth, model, URL), "unavailable",
        # "interrupted". "quota" and "fatal" are worth retrying with another API key.
        self.stop_kind: str | None = None
        self.on_progress = None                   # optional callback(done, total) after each batch (web API)
        self.interrupted = False                  # Ctrl+C: answers so far are cached, the caller stops
        self.cancel: threading.Event | None = None  # set by the caller (another thread) to stop like Ctrl+C
        self._empty_streak = 0
        self._partial: dict[str, int | None] = {}

    @property
    def name(self) -> str:
        return self.cfg.name

    # -- one HTTP request with retries; returns {local id: label}
    def _max_tokens(self, n: int) -> int:
        """Answer tokens per comment + a fixed extra + a reasoning allowance (0 until a reply is cut off)."""
        base = LLM_CFG.get("max_tokens_per_comment", 20) * n + LLM_CFG.get("max_tokens_extra", 50)
        return int(min(LLM_CFG.get("max_tokens_cap", 8192), base + self.reasoning_tokens))

    def _check_cancel(self) -> None:
        """Ctrl+C reaches only the main thread; when this component runs in a worker thread, the main thread sets
        `cancel` and the worker stops here as if it had been interrupted (answers so far are cached)."""
        if self.cancel is not None and self.cancel.is_set():
            raise KeyboardInterrupt

    def _wait(self, seconds: float) -> None:
        """self.sleep, but woken early (and stopped) when `cancel` is set."""
        self._check_cancel()
        if self.cancel is not None and self.sleep is time.sleep:
            self.cancel.wait(seconds)
        else:
            self.sleep(seconds)
        self._check_cancel()

    def _pace(self) -> None:
        """Wait for the tokens-per-minute window when the last reply said too few tokens are left for another
        request of the same size. Avoids sending requests that would only come back as 429."""
        if self._tokens_left is None or self._last_cost is None or self._tokens_left >= self._last_cost:
            return
        wait = min(65.0, (self._tokens_reset or 10.0) + 0.5)
        log.info("%s: %d tokens left in this minute, the next request needs ~%d; waiting %.0fs", self.name,
                 self._tokens_left, self._last_cost, wait)
        self.stats["rate_limit_waits"] += 1
        self._wait(wait)
        self._tokens_left = None

    def _ask(self, comments: list[str]) -> dict[int, int]:
        if self.max_requests is not None and self.stats["requests"] >= self.max_requests:
            raise DailyLimitReached(f"--max-requests {self.max_requests} reached", category="RUN CAP (--max-requests)")
        self.budget.check()
        user = self.prompt.user_message(comments)
        slot = f"LLM_{self.cfg.slot.upper()}"
        attempt = waits = 0
        max_waits = int(LLM_CFG.get("max_rate_limit_waits", 30))
        cap = LLM_CFG.get("max_tokens_cap", 8192)
        while True:
            max_tokens = self._max_tokens(len(comments))
            self._pace()
            self._check_cancel()
            try:
                self.budget.record()
                self.stats["requests"] += 1
                text = self.client.chat(self.prompt.system, user, max_tokens)
                info = dict(getattr(self.client, "last_info", None) or {}, max_tokens=max_tokens)
                self._tokens_left, self._tokens_reset = info.get("tokens_left"), info.get("tokens_reset_sec")
                prompt_tokens = (info.get("usage") or {}).get("prompt_tokens")
                self._last_cost = prompt_tokens + max_tokens if prompt_tokens else None
                got = parse_reply(text, len(comments))
                if len(got) < len(comments) and info.get("finish_reason") == "length" and max_tokens < cap:
                    self.reasoning_tokens = max(256, 2 * self.reasoning_tokens)
                    self._count("OUTPUT CUT OFF (max_tokens raised)")
                    log.warning("%s: %s Adding %d tokens for reasoning (max_tokens %d for this batch) and resending.",
                                self.name, explain_unusable(text, info, len(comments), slot), self.reasoning_tokens,
                                self._max_tokens(len(comments)))
                    continue
                if got:
                    self._empty_streak = 0
                else:
                    self.stats["parse_failures"] += 1
                    self._empty_streak += 1
                    why = explain_unusable(text, info, len(comments), slot)
                    self.last_problem = why
                    self._count(why.split(" (")[0].split(":")[0])
                    log.warning("%s: no usable answer (%d in a row, stop at 6): %s", self.name, self._empty_streak, why)
                    if self._empty_streak >= 6:
                        stop = ProviderUnavailable(f"6 replies in a row had no usable answer. Last cause: {why}",
                                                   category=why.split(" (")[0].split(":")[0])
                        stop.counted = True
                        raise stop
                self.stats["missing_ids"] += len(comments) - len(got)
                return got
            except RateLimited as e:
                # rejected, so not a used request (the provider does not count it either); wait and resend as is
                self.budget.record(-1)
                self.stats["requests"] -= 1
                self.stats["rate_limit_waits"] += 1
                self._count(e.category)
                waits += 1
                if waits > max_waits:
                    raise ProviderUnavailable(f"still rate-limited after {max_waits} waits. Last error: {e}",
                                              category=e.category) from None
                wait = min(65.0, (e.retry_after or 10.0) + 0.5)
                log.info("%s: %s; waiting %.1fs. (%s)", self.name, e.category, wait, e)
                self._wait(wait)
                self._tokens_left = None
                self.budget.check()
            except TransientError as e:
                attempt += 1
                self._count(e.category)
                if attempt > LLM_CFG.get("max_retries", 5):
                    raise ProviderUnavailable(f"gave up after {attempt - 1} retries. Last error: {e}",
                                              category=e.category) from None
                self.stats["retries"] += 1
                wait = e.retry_after or min(LLM_CFG.get("backoff_cap_sec", 90),
                                            LLM_CFG.get("backoff_base_sec", 2) * 2 ** (attempt - 1))
                log.warning("%s: %s — %s; retry %d/%d in %.0fs", self.name, e.category, e, attempt,
                            LLM_CFG.get("max_retries", 5), wait)
                self._wait(wait)
                self.budget.check()
            except TokenLimitError as e:
                self.budget.record(-1)            # rejected before any work
                self.stats["requests"] -= 1
                self._count(e.category)
                raise
            except (StopRun, FatalProviderError) as e:
                if not getattr(e, "counted", False):
                    self._count(e.category)
                raise

    def _count(self, category: str) -> None:
        self.stats["errors"][category] = self.stats["errors"].get(category, 0) + 1

    def classify_batch(self, items: list[tuple[str, str]]) -> list[dict]:
        """items: (key, preprocessed comment). Returns [{id: key, label, status}] for every item.
        Missing/invalid answers are re-sent in halves, then alone, then marked unavailable.
        StopRun propagates; the answers collected so far stay in `self._partial` for the caller to cache."""
        out: dict[str, int | None] = {}
        self._partial = out

        def solve(batch: list[tuple[str, str]], retried: bool = False):
            try:
                got = self._ask([c for _, c in batch])
            except TokenLimitError as e:
                # lower the batch size for the rest of this run, wait, then send this batch in smaller pieces
                self.stats["token_backoffs"] += 1
                if self.batch_size > self.min_batch:
                    self.batch_size = max(self.min_batch, self.batch_size // 2)
                    log.warning("%s: %s — %s; batch size -> %d", self.name, e.category, e, self.batch_size)
                self._wait(e.retry_after or 20)
                if len(batch) == 1:
                    if retried:
                        out[batch[0][0]] = None
                    else:
                        solve(batch, True)
                    return
                size = max(1, min(self.batch_size, (len(batch) + 1) // 2))
                for i in range(0, len(batch), size):
                    solve(batch[i:i + size])
                return
            for i, (k, _) in enumerate(batch, 1):
                if i in got:
                    out[k] = got[i]
            missing = [it for i, it in enumerate(batch, 1) if i not in got]
            if not missing:
                return
            if len(missing) == len(batch) and not retried:     # nothing usable: same batch once more
                return solve(batch, True)
            if len(missing) == 1:
                if len(batch) > 1 or not retried:
                    return solve(missing, len(batch) == 1)
                out[missing[0][0]] = None
                return
            half = (len(missing) + 1) // 2
            solve(missing[:half])
            solve(missing[half:])

        solve(items)
        return [dict(id=k, label=out.get(k), status=STATUS_OK if out.get(k) is not None else STATUS_UNAVAILABLE)
                for k, _ in items]

    def lookup(self, comments: list[str], claim: str = FIXED_CLAIM) -> list[dict]:
        """Cache only, no requests: [{label, status}] with status 'missing' for uncached comments."""
        res = []
        for c in comments:
            r = self.cache.get(cache_key(claim, c))
            if r is None:
                res.append(dict(label=None, status=STATUS_MISSING))
            else:
                res.append(dict(label=LABELS.index(r["label"]) if r.get("label") else None, status=r["status"]))
        return res

    def run(self, comments: list[str], claim: str = FIXED_CLAIM, retry_unavailable: bool = False,
            progress_every: int = 1) -> list[dict]:
        """Classify comments (preprocessed again, which is idempotent). Cached answers are reused, the rest is sent
        in batches. On a daily limit the run stops cleanly; unsent comments get status 'missing'."""
        texts = [preprocess(c) for c in comments]
        keys = [cache_key(claim, t) for t in texts]
        todo, seen = [], set()
        for k, t in zip(keys, texts):
            r = self.cache.get(k)
            if r and (r["status"] == STATUS_OK or not retry_unavailable):
                self.stats["cache_hits"] += 1
                continue
            if k not in seen:
                seen.add(k)
                todo.append((k, t))
        n_batches = 0
        i = 0
        try:
            while i < len(todo):
                batch = todo[i:i + self.batch_size]
                try:
                    res = self.classify_batch(batch)
                except (StopRun, KeyboardInterrupt) as e:
                    partial = [(k, l, STATUS_OK) for k, l in self._partial.items() if l is not None]
                    self.cache.put_many(partial)
                    self.stats["ok"] += len(partial)
                    if isinstance(e, KeyboardInterrupt):
                        self.interrupted = True
                        self.stopped = "INTERRUPTED (Ctrl+C)"
                        self.stop_kind = "interrupted"
                    else:
                        self.stopped = f"{e.category}: {e}"
                        self.stop_kind = ("cap" if str(e.category).startswith("RUN CAP") else
                                          "quota" if isinstance(e, DailyLimitReached) else "unavailable")
                    break
                self.cache.put_many([(r["id"], r["label"], r["status"]) for r in res])
                self.stats["sent"] += len(batch)
                self.stats["ok"] += sum(r["status"] == STATUS_OK for r in res)
                self.stats["unavailable"] += sum(r["status"] != STATUS_OK for r in res)
                i += len(batch)
                n_batches += 1
                if self.on_progress:
                    self.on_progress(i, len(todo))
                if progress_every and n_batches % progress_every == 0:
                    log.info("%s: %d/%d new comments done (batch %d, requests %d, rate-limit waits %d, today %d%s)",
                             self.name, i, len(todo), self.batch_size, self.stats["requests"],
                             self.stats["rate_limit_waits"], self.budget.today,
                             f"/{self.budget.limit}" if self.budget.limit else "")
        except FatalProviderError as e:
            self.stopped = f"{e.category}: {e}"
            self.stop_kind = "fatal"
        if self.stopped:
            left = sum(1 for k, _ in todo if self.cache.get(k) is None)
            log.error("%s (%s / %s) STOPPED.\n    Reason: %s\n    %d comments still to do; everything answered so far "
                      "is cached. Fix the cause if it is on your side, then rerun the same command to resume.",
                      self.name, self.cfg.provider, self.cfg.model, self.stopped, left)
        return self.lookup(texts, claim)
