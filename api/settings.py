"""Limits and deployment settings of the web API. Every value can be overridden by a line in the server's .env
(e.g. JOBS_PER_DAY=30). GET /limits sends them to the website, so what the page shows is what is enforced here."""
from __future__ import annotations

import os
from pathlib import Path

API_DIR = Path(__file__).resolve().parent
ROOT = API_DIR.parent

try:                                       # the server's .env (mounted into the container) before reading any value
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass


def _num(name: str, default: float) -> float:
    v = os.environ.get(name)
    try:
        return type(default)(v) if v not in (None, "") else default
    except ValueError:
        return default


MAX_ROWS = _num("MAX_ROWS", 2000)                          # rows per CSV
MAX_FILE_MB = _num("MAX_FILE_MB", 5.0)                     # upload size
JOBS_PER_IP_PER_HOUR = _num("JOBS_PER_IP_PER_HOUR", 5)     # prediction runs per visitor per hour
JOBS_PER_DAY = _num("JOBS_PER_DAY", 20)                    # prediction runs per day, all visitors together
UPLOADS_PER_IP_PER_HOUR = _num("UPLOADS_PER_IP_PER_HOUR", 30)   # file previews (cheap, no model or LLM)
QUEUE_MAX = _num("QUEUE_MAX", 5)                           # waiting jobs; one runs at a time
RESULT_TTL_HOURS = _num("RESULT_TTL_HOURS", 6.0)           # uploads and results are deleted after this
ALLOWED_EXT = (".csv",)
# The evaluator's prediction file: output_<group id>.csv with the columns Id and label
GROUP_ID = os.environ.get("GROUP_ID", "04")
OUTPUT_FILE = f"output_{GROUP_ID}.csv"

ALLOWED_ORIGINS = [o.strip().rstrip("/") for o in os.environ.get(
    "ALLOWED_ORIGINS", "https://ecoconsensus.harshitagr42.com").split(",") if o.strip()]
# local testing of web/ (python -m http.server 8080 inside web/)
ALLOWED_ORIGINS += ["http://localhost:8080", "http://127.0.0.1:8080"]

WEIGHTS_REPO = os.environ.get("WEIGHTS_REPO")              # e.g. <hf-user>/ecoconsensus-weights (server .env)
WORK_DIR = Path(os.environ.get("WORK_DIR", "/tmp/ecoconsensus"))
DEVICE = os.environ.get("DEVICE", "cpu")


def public() -> dict:
    return dict(max_rows=MAX_ROWS, max_file_mb=MAX_FILE_MB, jobs_per_ip_per_hour=JOBS_PER_IP_PER_HOUR,
                jobs_per_day=JOBS_PER_DAY, uploads_per_ip_per_hour=UPLOADS_PER_IP_PER_HOUR, queue_max=QUEUE_MAX,
                result_ttl_hours=RESULT_TTL_HOURS, file_types=list(ALLOWED_EXT), output_file=OUTPUT_FILE)
