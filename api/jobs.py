"""Uploads, rate limits and the job queue of the web API. One worker thread runs one job at a time (the free Space
has 2 CPUs); every job writes into its own folder under WORK_DIR/jobs/<id>/ with the same files as inference.py.

A job: the uploaded CSV + the user's column and label mapping -> a canonical CSV (id, comment[, claim][, label])
-> src.data_loading.load_file -> src.pipeline.run_pipeline (progress events become stages) -> inference.py's
report and prediction files -> a JSON summary for the website.
"""
from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
import uuid
import zipfile
from collections import deque
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from api import settings

LABEL_CLASSES = ["favour", "neutral", "against"]          # our ids 0 / 1 / 2
CHOICES = LABEL_CLASSES + ["ignore"]
OUTPUT_STYLES = {"file": "same values as the uploaded file", "ps": "Favour / None / Against",
                 "numeric": "our ids 0 / 1 / 2"}
PREVIEW_ROWS = 50
# Column names that probably hold the gold stance although the loader would not pick them (the professor's file may
# call it predict / prediction / target).
EXTRA_LABEL_NAMES = ["predict", "prediction", "predictions", "predicted", "target", "gold_label", "true_label",
                     "answer", "output"]
log = logging.getLogger("api")


class UserError(Exception):
    """A problem with the request (shown to the user as is, HTTP 400)."""


# ---------------------------------------------------------------- rate limits
class RateLimiter:
    """Sliding one-hour window per key and a per-day total; in memory (a restart resets it)."""

    def __init__(self, per_hour: int, per_day: int | None = None):
        self.per_hour, self.per_day = per_hour, per_day
        self.hits: dict[str, deque] = {}
        self.day, self.day_count = date.today(), 0
        self.lock = threading.Lock()

    def _prune(self, key: str, now: float) -> deque:
        q = self.hits.setdefault(key, deque())
        while q and now - q[0] > 3600:
            q.popleft()
        if date.today() != self.day:
            self.day, self.day_count = date.today(), 0
        return q

    def status(self, key: str) -> dict:
        with self.lock:
            now = time.time()
            q = self._prune(key, now)
            left = max(0, self.per_hour - len(q))
            reset = int(3600 - (now - q[0])) + 1 if q and not left else 0
            out = dict(left_this_hour=left, retry_after=reset)
            if self.per_day is not None:
                out["left_today"] = max(0, self.per_day - self.day_count)
            return out

    def take(self, key: str) -> None:
        """Count one use or raise RateLimited."""
        with self.lock:
            now = time.time()
            q = self._prune(key, now)
            if self.per_day is not None and self.day_count >= self.per_day:
                tomorrow = time.mktime(date.fromordinal(date.today().toordinal() + 1).timetuple())
                raise RateLimited(f"The daily limit of {self.per_day} prediction runs (all visitors together) is "
                                  f"used up. It resets at midnight server time.", int(tomorrow - now) + 1)
            if len(q) >= self.per_hour:
                wait = int(3600 - (now - q[0])) + 1
                raise RateLimited(f"Limit reached: {self.per_hour} per hour from your network. Try again in "
                                  f"{max(1, round(wait / 60))} min.", wait)
            q.append(now)
            self.day_count += 1

    def refund(self, key: str) -> None:
        """A run that failed on the server's side does not count."""
        with self.lock:
            q = self.hits.get(key)
            if q:
                q.pop()
            self.day_count = max(0, self.day_count - 1)


class RateLimited(Exception):
    def __init__(self, msg: str, retry_after: int):
        super().__init__(msg)
        self.retry_after = retry_after


JOB_LIMIT = RateLimiter(int(settings.JOBS_PER_IP_PER_HOUR), int(settings.JOBS_PER_DAY))
UPLOAD_LIMIT = RateLimiter(int(settings.UPLOADS_PER_IP_PER_HOUR))


# ---------------------------------------------------------------- uploads (inspect)
def _dir(kind: str, id_: str) -> Path:
    return settings.WORK_DIR / kind / id_


def save_upload(filename: str, data: bytes) -> dict:
    """Store the file and describe it for the mapping step: columns, 5 preview rows, guessed roles, value counts of
    every column with few distinct values (any of them can be chosen as the stance column) and label guesses."""
    from src.data_loading import LoaderError, map_columns, read_table
    from src.pipeline import TEXT_LABEL_MAP
    ext = Path(filename or "").suffix.lower()
    if ext not in settings.ALLOWED_EXT:
        raise UserError(f"Only {', '.join(settings.ALLOWED_EXT)} files are accepted (got '{ext or 'no extension'}').")
    if len(data) > settings.MAX_FILE_MB * 1024 * 1024:
        raise UserError(f"File is {len(data) / 1048576:.1f} MB; the limit is {settings.MAX_FILE_MB:g} MB.")
    if not data.strip():
        raise UserError("The file is empty.")
    uid = uuid.uuid4().hex
    d = _dir("uploads", uid)
    d.mkdir(parents=True)
    p = d / f"input{ext}"
    p.write_bytes(data)
    try:
        raw, info = read_table(p)
    except (LoaderError, Exception) as e:  # noqa: BLE001  (any parser error is the file's problem)
        shutil.rmtree(d, ignore_errors=True)
        raise UserError(f"Could not read the CSV: {e}") from None
    if len(raw) == 0:
        shutil.rmtree(d, ignore_errors=True)
        raise UserError("The CSV has a header but no rows.")
    if len(raw) > settings.MAX_ROWS:
        shutil.rmtree(d, ignore_errors=True)
        raise UserError(f"The file has {len(raw)} rows; the limit is {settings.MAX_ROWS}. Split it into smaller files.")
    cols = [str(c) for c in raw.columns]
    raw.columns = cols
    if len(set(cols)) != len(cols):
        shutil.rmtree(d, ignore_errors=True)
        raise UserError("Two columns have the same name; rename one and upload again.")

    values = {}
    for c in cols:
        vc = raw[c].fillna("").astype(str).str.strip().value_counts()
        if len(vc) <= 30:
            values[c] = [dict(value=str(v), count=int(n)) for v, n in vc.items()]
    guess = map_columns(cols, {})
    if guess["label"] is None:
        low = {c.strip().lower(): c for c in cols}
        guess["label"] = next((low[n] for n in EXTRA_LABEL_NAMES if n in low and low[n] in values), None)
        if guess["label"] and guess["claim"] == guess["label"]:
            guess["claim"] = None
    if guess["claim"] and guess["claim"] in values and guess["label"] is None and len(values[guess["claim"]]) <= 6:
        # e.g. a "target" column holding 0/1/2 or FAVOR/AGAINST: a label column, not a claim
        if all(_label_guess(v["value"], TEXT_LABEL_MAP) or v["value"].lstrip("-").isdigit()
               for v in values[guess["claim"]]):
            guess["label"], guess["claim"] = guess["claim"], None
    label_guess = {c: {v["value"]: _label_guess(v["value"], TEXT_LABEL_MAP) for v in vals}
                   for c, vals in values.items()}
    meta = dict(upload_id=uid, filename=Path(filename).name, rows=len(raw), columns=cols,
                encoding=info.encoding, delimiter=info.delimiter)
    (d / "meta.json").write_text(pd.Series(meta).to_json(), encoding="utf-8")
    preview = raw.head(5).fillna("").astype(str).to_dict(orient="records")
    return dict(meta, preview=preview, guess=guess, values=values, label_guess=label_guess)


def _label_guess(value: str, text_map: dict) -> str | None:
    """Text labels only (FAVOR -> favour, ...). Numbers are never guessed: 0 means different things in different
    files. Empty cells -> ignore."""
    v = str(value).strip().lower()
    if v == "":
        return "ignore"
    return text_map.get(v)


def upload_meta(uid: str) -> dict:
    import json
    if not uid or not uid.isalnum():
        raise UserError("Unknown upload id.")
    p = _dir("uploads", uid) / "meta.json"
    if not p.exists():
        raise UserError("This upload has expired; upload the file again.")
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- jobs
@dataclass
class Job:
    id: str
    ip: str
    upload_id: str
    filename: str
    params: dict
    created: float = field(default_factory=time.time)
    status: str = "queued"                 # queued / running / done / error
    stages: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    started: float | None = None
    finished: float | None = None
    results: dict | None = None

    @property
    def dir(self) -> Path:
        return _dir("jobs", self.id)

    def stage(self, key: str) -> dict:
        return next(s for s in self.stages if s["key"] == key)

    def public(self) -> dict:
        now = time.time()
        stages = []
        for s in self.stages:
            t0, t1 = s.get("t0"), s.get("t1")
            stages.append({k: v for k, v in s.items() if k not in ("t0", "t1")} |
                          dict(seconds=round((t1 or now) - t0, 1) if t0 else None))
        return dict(id=self.id, status=self.status, filename=self.filename, rows=self.params.get("rows"),
                    position=queue_position(self.id), stages=stages, warnings=self.warnings, error=self.error,
                    elapsed=round((self.finished or now) - self.started, 1) if self.started else 0,
                    has_results=self.results is not None)


JOBS: dict[str, Job] = {}
_QUEUE: "queue.Queue[str]" = queue.Queue()
_LOCK = threading.Lock()
_WAITING: list[str] = []
_WORKER: threading.Thread | None = None
LLM_STATE: dict[str, dict] = {}            # slot -> last known quota state (shown by /health)


def queue_position(job_id: str) -> int | None:
    with _LOCK:
        return _WAITING.index(job_id) + 1 if job_id in _WAITING else None


def llm_names() -> dict[str, str]:
    from src.config import env
    out = {}
    for c in ("llm_a", "llm_b"):
        s = c[-1].upper()
        prov, model = env(f"LLM_{s}_PROVIDER"), env(f"LLM_{s}_MODEL")
        out[c] = f"LLM-{s}" + (f" ({prov} · {model})" if prov and model else "")
    return out


def validate(params: dict) -> dict:
    """Check the mapping the user confirmed; returns the cleaned parameters. Raises UserError."""
    meta = upload_meta(params.get("upload_id", ""))
    cols = meta["columns"]

    def col(name, required=False):
        v = params.get(name)
        if v in (None, ""):
            if required:
                raise UserError(f"Choose the {name.replace('_col', '')} column.")
            return None
        if v not in cols:
            raise UserError(f"Column '{v}' is not in the file.")
        return v
    comment, claim, label, id_ = col("comment_col", True), col("claim_col"), col("label_col"), col("id_col")
    chosen = [c for c in (comment, claim, label, id_) if c]
    if len(set(chosen)) != len(chosen):
        raise UserError("One column is used for two roles; each role needs its own column.")
    lm = {}
    if label:
        lm = {str(k): str(v).lower() for k, v in (params.get("label_map") or {}).items()}
        bad = {k: v for k, v in lm.items() if v not in CHOICES}
        if bad:
            raise UserError(f"Label mapping must be one of {CHOICES}; got {bad}.")
        if not any(v in LABEL_CLASSES for v in lm.values()):
            raise UserError("The stance column is chosen but no value is mapped to favour / neutral / against.")
    elif not params.get("confirm_no_labels"):
        raise UserError("No stance column chosen: confirm that you want predictions without metrics.")
    style = params.get("output_style") or ("file" if label else "ps")
    if style not in OUTPUT_STYLES:
        raise UserError(f"output_style must be one of {list(OUTPUT_STYLES)}.")
    if style == "file" and not label:
        style = "ps"
    return dict(upload_id=meta["upload_id"], filename=meta["filename"], rows=meta["rows"], comment_col=comment,
                claim_col=claim, label_col=label, id_col=id_, label_map=lm, output_style=style)


def submit(params: dict, ip: str) -> Job:
    p = validate(params)
    with _LOCK:
        if len(_WAITING) >= settings.QUEUE_MAX:
            raise QueueFull(f"{len(_WAITING)} files are already waiting; try again in a few minutes.")
    JOB_LIMIT.take(ip)
    job = Job(id=uuid.uuid4().hex, ip=ip, upload_id=p["upload_id"], filename=p["filename"], params=p)
    names = llm_names()
    job.stages = [dict(key="read", label="Reading and cleaning the file", state="pending"),
                  dict(key="muril", label="MuRIL (fine-tuned transformer)", state="pending"),
                  dict(key="indicbert", label="IndicBERT v2 (fine-tuned transformer)", state="pending"),
                  dict(key="llm_a", label=names["llm_a"], state="pending"),
                  dict(key="llm_b", label=names["llm_b"], state="pending"),
                  dict(key="meta", label="Stacked meta-model (final prediction)", state="pending"),
                  dict(key="report", label="Metrics, report and files", state="pending")]
    with _LOCK:
        JOBS[job.id] = job
        _WAITING.append(job.id)
    _ensure_worker()
    _QUEUE.put(job.id)
    return job


class QueueFull(Exception):
    pass


def _ensure_worker() -> None:
    global _WORKER
    with _LOCK:
        if _WORKER is None or not _WORKER.is_alive():
            _WORKER = threading.Thread(target=_worker, name="job-worker", daemon=True)
            _WORKER.start()


def _worker() -> None:
    while True:
        job_id = _QUEUE.get()
        with _LOCK:
            if job_id in _WAITING:
                _WAITING.remove(job_id)
            job = JOBS.get(job_id)
        if job is None:
            continue
        try:
            run_job(job)
        except Exception as e:  # noqa: BLE001  (never kill the worker)
            log.exception("job %s failed", job_id)
            job.status, job.error = "error", f"{type(e).__name__}: {e}"
            JOB_LIMIT.refund(job.ip)
        finally:
            job.finished = time.time()
            for s in job.stages:
                if s["state"] == "running":
                    s["state"], s["t1"] = "error", time.time()


# ---------------------------------------------------------------- running one job
def _set(job: Job, key: str, state: str, detail: str | None = None, **kw) -> None:
    s = job.stage(key)
    if state == "running" and not s.get("t0"):
        s["t0"] = time.time()
    if state in ("done", "warning", "skipped", "error"):
        s.setdefault("t0", time.time())
        s["t1"] = time.time()
    s["state"] = state
    if detail is not None:
        s["detail"] = detail
    s.update(kw)


def _job_logger(job: Job) -> tuple[logging.Logger, str]:
    """A log file per job; the llm_api module logger writes into it too (one job runs at a time)."""
    d = job.dir / "logs"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "job_log.txt"
    lg = logging.getLogger(f"job.{job.id}")
    lg.setLevel(logging.INFO)
    lg.propagate = False
    fh = logging.FileHandler(f, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    lg.handlers = [fh]
    lib = logging.getLogger("llm_api")
    lib.handlers, lib.propagate = [fh], False
    lib.setLevel(logging.INFO)
    return lg, str(f)


def build_input(job: Job) -> Path:
    """The uploaded file reduced to the chosen columns under fixed names, so the loader cannot pick another column
    (e.g. a 'stance' column the user said to ignore)."""
    from src.data_loading import read_table
    p = job.params
    src = next(_dir("uploads", job.upload_id).glob("input.*"))
    raw, _ = read_table(src)
    raw.columns = [str(c) for c in raw.columns]
    out = pd.DataFrame({"id": raw[p["id_col"]] if p["id_col"] else [str(i + 1) for i in range(len(raw))],
                        "comment": raw[p["comment_col"]].fillna("")})
    if p["claim_col"]:
        out["claim"] = raw[p["claim_col"]].fillna("")
    if p["label_col"]:
        out["label"] = raw[p["label_col"]].fillna("").astype(str).str.strip()
    job.dir.mkdir(parents=True, exist_ok=True)
    dst = job.dir / "input_mapped.csv"
    out.to_csv(dst, index=False)
    return dst


def run_job(job: Job) -> None:
    import inference as inf
    from src.config import FIXED_CLAIM
    from src.data_loading import LabelError, LoaderError, load_file
    from src.ensembling import LLMS, TRANSFORMERS
    from src.pipeline import DISPLAY, format_labels, load_models, needed_components, run_pipeline
    from src.utils import set_seed

    job.status, job.started = "running", time.time()
    p = job.params
    lg, log_file = _job_logger(job)
    lg.info("job %s: %s, %d rows, mapping %s", job.id, job.filename, p["rows"],
            {k: p[k] for k in ("comment_col", "claim_col", "label_col", "id_col", "output_style")})
    set_seed(42)

    # ---- read
    _set(job, "read", "running")
    lm = {k.strip().lower(): v for k, v in p["label_map"].items() if v in LABEL_CLASSES}
    try:
        df, info = load_file(build_input(job), id_col="id", text_col="comment", label_col="label" if p["label_col"]
                             else None, claim_col="claim" if p["claim_col"] else None, label_map=lm,
                             allow_unknown_labels=True, detect_claim=False, fixed_claim=FIXED_CLAIM)
    except (LoaderError, LabelError) as e:
        _set(job, "read", "error", str(e))
        job.status, job.error = "error", str(e)
        return
    info.mapping = dict(id=p["id_col"], comment=p["comment_col"], claim=p["claim_col"], label=p["label_col"],
                        language=None)
    if len(df) == 0 or df["empty"].all():
        msg = "No comment to classify: every comment is empty after cleaning."
        _set(job, "read", "error", msg)
        job.status, job.error = "error", msg
        return
    n_lab = int(df["label"].notna().sum())
    ignored = int(len(df) - n_lab) if p["label_col"] else 0
    detail = f"{len(df)} rows, {int(df['empty'].sum())} empty comments"
    if p["label_col"]:
        detail += f", {n_lab} with a stance label" + (f" ({ignored} ignored)" if ignored else "")
    if (df["claim"] != FIXED_CLAIM).any():
        job.warnings.append("Some rows use a claim other than the default one; the models were trained on the "
                            "default claim only, so those predictions are less reliable.")
    _set(job, "read", "done", detail)

    # ---- models
    components = needed_components("ensemble", use_llm=True)
    models = load_models(components, need_meta=True)

    def progress(ev: dict) -> None:
        c, e = ev["component"], ev["event"]
        if c not in {s["key"] for s in job.stages}:
            return
        if e == "start":
            _set(job, c, "running", "loading model" if c in TRANSFORMERS else "starting", total=ev.get("total"))
        elif e == "rows":
            _set(job, c, "running", f"{ev['done']}/{ev['total']} comments", done=ev["done"], total=ev["total"])
        elif e == "key_switch":
            job.warnings.append(f"{DISPLAY[c]}: {ev['message']}")
            _set(job, c, "running", ev["message"])
        elif e == "done" and c in TRANSFORMERS + ["meta"]:
            _set(job, c, "done", f"{ev.get('total')} rows")
        elif e == "done":
            ok = ev["status"] == "ok"
            _set(job, c, "done" if ok else "warning",
                 f"{ev['answered']}/{ev['total']} answered" + ("" if ok else f" — {ev.get('stopped') or ev['status']}"),
                 done=ev["answered"], total=ev["total"])
    n = int((~df["empty"].to_numpy(dtype=bool)).sum())
    from src.llm_api import default_batch_size
    budget = 4 * (n // max(1, default_batch_size()) + 1) + 20       # per provider and key: a safety cap per run
    res = run_pipeline(df, models, components, device=settings.DEVICE, llm_budget=budget, log=lg,
                       progress=progress)
    for c in LLMS:                       # not configured: llm_labels returns before its "done" event
        i = res.llm_info.get(c, {})
        st = job.stage(c)
        if st["state"] in ("pending", "running"):
            _set(job, c, "warning", f"unavailable — {i.get('stopped') or i.get('status')}")
        if i.get("status") not in (None, "ok"):
            job.warnings.append(f"{DISPLAY[c]} answered {i.get('answered', 0)} of {n} comments "
                                f"({i.get('stopped') or i.get('status')}). Those rows use a meta-model without it.")
            quota = "DAILY" in str(i.get("stopped") or "")
            LLM_STATE[c] = dict(day=date.today().isoformat(), quota_reached=quota, keys_tried=i.get("key_used"),
                                keys=i.get("keys"), message=i.get("stopped"))
        else:
            LLM_STATE.pop(c, None)

    # ---- report and files (inference.py writes the same files as a command-line run)
    _set(job, "report", "running")
    file_map = None
    if p["output_style"] == "file":
        file_map = {}
        counts = pd.Series(df["label_raw"]).value_counts() if "label_raw" in df else pd.Series(dtype=int)
        # most frequent spelling of each class first, then the other values the user mapped
        for raw_v in list(counts.index) + list(p["label_map"]):
            cls = lm.get(str(raw_v).strip().lower())
            if cls and LABEL_CLASSES.index(cls) not in file_map:
                file_map[LABEL_CLASSES.index(cls)] = str(raw_v)
        missing = [LABEL_CLASSES[i] for i in range(3) if i not in file_map]
        if missing:
            job.warnings.append(f"Your file has no value for {', '.join(missing)}; predictions of that class are "
                                f"written as Favour / None / Against.")
    args = SimpleNamespace(mode="ensemble", device=settings.DEVICE, batch_size=None, label_style=p["output_style"],
                           require_llm=False, allow_unknown_labels=True, json=True, quiet=True)
    cfg = dict(data_file=job.filename, id_col=p["id_col"], text_col=p["comment_col"], label_col=p["label_col"],
               claim_col=p["claim_col"], language_col=None, label_map=p["label_map"], fixed_claim=FIXED_CLAIM,
               use_llm=True, llm_budget=budget)
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):         # the JSON summary it prints is not needed here
        inf._write_file_results(args, cfg, df, info, res, models, Path(job.filename), job.dir, lg, log_file,
                                file_map=file_map)
    from src.utils import to_jsonable
    make_zip(job)
    job.results = to_jsonable(collect_results(job, df, res, file_map, n_lab))     # NaN -> null
    _set(job, "report", "done", "ready")
    job.status = "done"


def collect_results(job: Job, df: pd.DataFrame, res, file_map, n_lab: int) -> dict:
    import json

    from src.config import LABELS
    from src.ensembling import LLMS, TRANSFORMERS
    from src.pipeline import DISPLAY, final_prediction, format_labels
    style = job.params["output_style"]
    pred, source = final_prediction(res, "ensemble")
    systems = [s for s in [*TRANSFORMERS, *LLMS, "soft_vote", "majority_vote", "tfidf_lr", "stacked"]
               if s in res.preds]
    rows = []
    for i in range(min(PREVIEW_ROWS, len(df))):
        r = dict(id=str(df["id"].iloc[i]), comment=str(df["comment_raw"].iloc[i] or "")[:300])
        if job.params["label_col"]:
            r["gold"] = str(df["label_raw"].iloc[i]) if df["label_raw"].iloc[i] is not None else ""
            r["gold_class"] = LABELS[int(df["label"].iloc[i])] if pd.notna(df["label"].iloc[i]) else ""
        for s in systems:
            v = int(res.preds[s][i])
            r[s] = format_labels([v], style, file_map)[0] if v >= 0 else "—"
        r["final"] = format_labels([pred[i]], style, file_map)[0]
        r["final_class"] = LABELS[int(pred[i])]
        r["final_source"] = str(source[i])
        rows.append(r)
    metrics, conf = [], None
    mfile = next(iter(sorted((job.dir / "metrics").glob("*.json"))), None)
    if mfile:
        m = json.loads(mfile.read_text(encoding="utf-8")).get("metrics") or {}
        ev = job.dir / "evaluation.csv"
        if ev.exists():
            metrics = pd.read_csv(ev).where(lambda x: x.notna(), None).to_dict(orient="records")
        if "stacked" in m:
            conf = dict(labels=LABELS, matrix=m["stacked"].get("confusion"))
    dist = {DISPLAY[s]: {LABELS[k]: int((res.preds[s] == k).sum()) for k in range(3)} | {
        "no answer": int((res.preds[s] < 0).sum())} for s in systems}
    variants = pd.Series(res.variant).fillna("none").value_counts().to_dict()
    llm = {c: {k: i.get(k) for k in ("provider", "model", "status", "answered", "unavailable", "requests",
                                      "cache_hits", "stopped", "keys", "key_used", "key_switches")}
           for c, i in res.llm_info.items()}
    report = (job.dir / "result.txt").read_text(encoding="utf-8") if (job.dir / "result.txt").exists() else ""
    return dict(columns=[("id", "ID"), ("comment", "Comment")] +
                ([("gold", "Your label")] if job.params["label_col"] else []) +
                [(s, DISPLAY[s]) for s in systems] + [("final", "Final")],
                preview=rows, rows=len(df), labelled=n_lab, empty=int(df["empty"].sum()), metrics=metrics,
                confusion=conf, distribution=dist, variants={str(k): int(v) for k, v in variants.items()},
                llm=llm, report=report, output_style=style, output_style_text=OUTPUT_STYLES[style],
                label_map=job.params["label_map"], mapping={k: job.params[k] for k in
                                                            ("comment_col", "claim_col", "label_col", "id_col")},
                files=[f for f in FILES if file_path(job, f)])


# downloadable files: name -> (path pattern inside the job folder, description)
FILES = {"predictions.csv": ("predictions.csv", "ID + final prediction"),
         "evaluation.csv": ("evaluation.csv", "metrics of every system"),
         "report.md": ("result.txt", "full report (Markdown)"),
         "details.csv": ("predictions/*_detailed.csv", "every system's prediction and probabilities per row"),
         "metrics.json": ("metrics/*.json", "all metrics as JSON"),
         "log.txt": ("logs/job_log.txt", "run log"),
         "all.zip": ("all.zip", "everything above")}


def file_path(job: Job, name: str) -> Path | None:
    if name not in FILES:
        return None
    return next(iter(sorted(job.dir.glob(FILES[name][0]))), None)


def make_zip(job: Job) -> None:
    with zipfile.ZipFile(job.dir / "all.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in FILES:
            if name == "all.zip":
                continue
            fp = file_path(job, name)
            if fp:
                z.write(fp, name)


# ---------------------------------------------------------------- cleanup
def cleanup(now: float | None = None) -> None:
    now = now or time.time()
    ttl = settings.RESULT_TTL_HOURS * 3600
    for kind in ("uploads", "jobs"):
        base = settings.WORK_DIR / kind
        for d in base.glob("*") if base.exists() else []:
            if now - d.stat().st_mtime > ttl and d.name not in _WAITING:
                job = JOBS.get(d.name)
                if job and job.status in ("queued", "running"):
                    continue
                shutil.rmtree(d, ignore_errors=True)
                JOBS.pop(d.name, None)


def start_cleanup_thread() -> None:
    def loop():
        while True:
            time.sleep(600)
            try:
                cleanup()
            except Exception:  # noqa: BLE001
                log.exception("cleanup failed")
    threading.Thread(target=loop, name="cleanup", daemon=True).start()
