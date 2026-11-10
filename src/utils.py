"""Seeding, logging, hashing and never-overwrite file writing shared by every script."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import platform
import random
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def stable_hash(obj, n: int = 10) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


def file_sha256(p: Path, n: int | None = 16) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n] if n else h.hexdigest()


def to_jsonable(o):
    """numpy -> python, NaN/inf -> None, Paths -> str, recursively."""
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


# ---------------------------------------------------------------- never-overwrite writing
def unique_path(p: Path, add_timestamp: bool = True) -> Path:
    """`name_<ts>.ext`, then `name_<ts>_2.ext`, `_3`, ... until the name is free."""
    p = Path(p)
    stem = f"{p.stem}_{timestamp()}" if add_timestamp else p.stem
    cand = p.with_name(f"{stem}{p.suffix}")
    k = 2
    while cand.exists():
        cand = p.with_name(f"{stem}_{k}{p.suffix}")
        k += 1
    return cand


def safe_write(p: Path, content: str | bytes, add_timestamp: bool = True) -> Path:
    """Write to a NEW file and return its path. Never overwrites: create-only open ('x') on a unique name."""
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    while True:
        target = unique_path(p, add_timestamp)
        try:
            mode = "xb" if isinstance(content, bytes) else "x"
            kw = {} if isinstance(content, bytes) else {"encoding": "utf-8"}
            with open(target, mode, **kw) as f:
                f.write(content)
            return target
        except FileExistsError:            # lost a race; pick the next name
            continue


def write_json(p: Path, obj, overwrite: bool = True) -> Path:
    """Atomic JSON write for state files (run_info, manifests). Use safe_write for reports."""
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_jsonable(obj), indent=2, ensure_ascii=False)
    if not overwrite:
        return safe_write(p, text, add_timestamp=False)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(p)
    return p


# ---------------------------------------------------------------- logging
def get_logger(run_type: str, log_dir: Path | None = None) -> logging.Logger:
    """Console + `outputs/logs/<run_type>_<YYYYMMDD_HHMMSS>.txt`. The command line is logged first."""
    from src.config import path
    log_dir = Path(log_dir) if log_dir else path("logs", create=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(run_type)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    log_file = unique_path(log_dir / f"{run_type}.txt")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.info("command: %s", " ".join(sys.argv))
    logger.info("log file: %s", log_file)
    return logger


def env_info() -> dict:
    def ver(pkg):
        try:
            return metadata.version(pkg)
        except Exception:
            return None
    pkgs = ("numpy", "pandas", "scikit-learn", "torch", "transformers", "peft", "mlflow", "requests")
    return {"python": platform.python_version(), "platform": platform.platform(),
            **{f"lib_{p}": ver(p) for p in pkgs if ver(p)}}


# ---------------------------------------------------------------- MLflow (optional everywhere)
def mlflow_start(experiment: str, run_name: str, logger: logging.Logger | None = None):
    """Return an active mlflow run, or None if mlflow is missing/broken (never stops a script)."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    try:
        import mlflow
        from src.config import BASE, ROOT, path
        path("metrics", create=True)
        uri = BASE["mlflow"]["tracking_uri"]
        if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
            uri = f"sqlite:///{ROOT / uri[len('sqlite:///'):]}"
        mlflow.set_tracking_uri(uri)
        if mlflow.get_experiment_by_name(experiment) is None:
            mlflow.create_experiment(experiment, artifact_location=(path("metrics") / "mlartifacts").as_uri())
        mlflow.set_experiment(experiment)
        return mlflow.start_run(run_name=run_name)
    except Exception as e:  # noqa: BLE001
        if logger:
            logger.warning("MLflow disabled: %s: %s", type(e).__name__, str(e)[:200])
        return None
