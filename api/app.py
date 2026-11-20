"""EcoConsensus web API (runs on a Hugging Face Space; the website in web/ calls it).

    GET  /health                 models ready?, LLMs configured, quota state, queue
    GET  /limits                 the limits below + how many runs this visitor has left
    POST /inspect                upload a CSV -> columns, preview, guessed mapping, label values
    POST /jobs                   start a run with the confirmed mapping
    GET  /jobs/{id}              stages and progress
    GET  /jobs/{id}/results      first 50 predictions per model, metrics, report
    GET  /jobs/{id}/files/{name} predictions.csv, evaluation.csv, report.md, details.csv, metrics.json, log.txt, all.zip

Every limit is enforced here (the website only displays it). Local run:
    uvicorn api.app:app --port 7860
"""
from __future__ import annotations

import logging
import sys
import threading
from contextlib import asynccontextmanager
from datetime import date

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from api import settings

sys.path.insert(0, str(settings.ROOT))

from api import jobs  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("api")

STATE = dict(models="loading", error=None)


@asynccontextmanager
async def lifespan(_app):
    settings.WORK_DIR.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=_prepare_models, name="weights", daemon=True).start()
    jobs.start_cleanup_thread()
    yield


app = FastAPI(title="EcoConsensus API", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.ALLOWED_ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type"], expose_headers=["Retry-After", "Content-Disposition"],
                   max_age=3600)


def _prepare_models() -> None:
    from api.weights import ensure_weights
    try:
        ensure_weights()
        STATE.update(models="ready", error=None)
    except Exception as e:  # noqa: BLE001
        log.exception("model files")
        STATE.update(models="error", error=str(e))


def client_ip(request: Request) -> str:
    """The visitor's address as the Space's proxy reports it (first X-Forwarded-For entry)."""
    xff = request.headers.get("x-forwarded-for", "")
    return xff.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def error(status: int, msg: str, retry_after: int | None = None) -> JSONResponse:
    headers = {"Retry-After": str(retry_after)} if retry_after else None
    return JSONResponse({"detail": msg, "retry_after": retry_after}, status_code=status, headers=headers)


def llm_status() -> dict:
    """Per LLM: configured?, number of keys, requests used today per key (budget counter), last quota state."""
    from src.config import env
    from src.llm_api import BudgetGuard, api_keys
    out = {}
    for c, name in jobs.llm_names().items():
        s = c[-1].upper()
        prov, model, url = env(f"LLM_{s}_PROVIDER"), env(f"LLM_{s}_MODEL"), env(f"LLM_{s}_BASE_URL")
        keys = api_keys(c[-1])
        lim = env(f"LLM_{s}_DAILY_LIMIT")
        d = dict(name=name, configured=bool(prov and model and url and keys), keys=len(keys))
        if d["configured"]:
            d["used_today"] = [BudgetGuard(prov, model, int(lim) if lim else None, key_index=k).today
                               for k in range(len(keys))]
            d["daily_limit_per_key"] = int(lim) if lim else None
        st = jobs.LLM_STATE.get(c)
        if st and st["day"] == date.today().isoformat():
            d["last_problem"] = st["message"]
            d["quota_reached"] = st["quota_reached"] and st["keys_tried"] == st["keys"]
        out[c] = d
    return out


@app.get("/health")
def health():
    running = next((j.id for j in jobs.JOBS.values() if j.status == "running"), None)
    return dict(status="ok", models=STATE["models"], models_error=STATE["error"], llm=llm_status(),
                queue=len(jobs._WAITING), running=bool(running))


@app.get("/limits")
def limits(request: Request):
    ip = client_ip(request)
    from src.config import FIXED_CLAIM, LABELS
    return dict(settings.public(), you=jobs.JOB_LIMIT.status(ip), default_claim=FIXED_CLAIM,
                label_ids={i: l for i, l in enumerate(LABELS)}, output_styles=jobs.OUTPUT_STYLES)


@app.post("/inspect")
async def inspect(request: Request, file: UploadFile = File(...)):
    ip = client_ip(request)
    try:
        jobs.UPLOAD_LIMIT.take(ip)
    except jobs.RateLimited as e:
        return error(429, str(e), e.retry_after)
    data = await file.read(int(settings.MAX_FILE_MB * 1024 * 1024) + 1)
    try:
        return jobs.save_upload(file.filename or "", data)
    except jobs.UserError as e:
        return error(400, str(e))


@app.post("/jobs")
async def create_job(request: Request):
    if STATE["models"] != "ready":
        return error(503, "The models are still loading (about a minute after the server starts). Try again shortly."
                     if STATE["models"] == "loading" else f"Model files are missing on the server: {STATE['error']}",
                     30)
    try:
        params = await request.json()
    except ValueError:
        return error(400, "Request body must be JSON.")
    try:
        job = jobs.submit(params if isinstance(params, dict) else {}, client_ip(request))
    except jobs.UserError as e:
        return error(400, str(e))
    except jobs.RateLimited as e:
        return error(429, str(e), e.retry_after)
    except jobs.QueueFull as e:
        return error(503, str(e), 120)
    return job.public()


def _job(job_id: str):
    job = jobs.JOBS.get(job_id)
    if job is None:
        return None, error(404, f"No such job (results are kept for {settings.RESULT_TTL_HOURS:g} hours, and a "
                                f"server restart clears them).")
    return job, None


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job, err = _job(job_id)
    return err or job.public()


@app.get("/jobs/{job_id}/results")
def job_results(job_id: str):
    job, err = _job(job_id)
    if err:
        return err
    if job.status == "error":
        return error(409, f"The run failed: {job.error}")
    if job.results is None:
        return error(409, "The run has not finished.")
    return dict(job.public(), results=job.results, files={k: v[1] for k, v in jobs.FILES.items()
                                                          if k in job.results["files"]})


@app.get("/jobs/{job_id}/files/{name}")
def job_file(job_id: str, name: str):
    job, err = _job(job_id)
    if err:
        return err
    p = jobs.file_path(job, name)
    if p is None:
        return error(404, f"No file '{name}' for this job.")
    stem = job.filename.rsplit(".", 1)[0][:60]
    return FileResponse(p, filename=f"{stem}_{name}")
