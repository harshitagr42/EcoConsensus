"""Load the YAML configs and .env once, and resolve project paths.

    from src.config import CFG, ROOT, path
    path("processed")            # -> <root>/data/processed
    out_path("muril", "predictions")   # -> <root>/outputs/muril/predictions
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"


def _load_yaml(name: str) -> dict:
    p = CONFIG_DIR / name
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}


def load_env() -> None:
    """Read .env into os.environ (existing variables win). Silent if python-dotenv or .env is missing."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env", override=False)


@lru_cache(maxsize=1)
def load_config() -> dict:
    load_env()
    return {
        "base": _load_yaml("base.yaml"),
        "transformers": _load_yaml("transformers.yaml"),
        "llm": _load_yaml("llm.yaml"),
    }


CFG = load_config()
BASE = CFG["base"]
LABELS: list[str] = list(BASE["labels"])           # ["favour", "neutral", "against"]
L2I = {l: i for i, l in enumerate(LABELS)}
FIXED_CLAIM: str = BASE["fixed_claim"]
SEED: int = int(BASE["seed"])


def path(key: str, create: bool = False) -> Path:
    """Absolute path for a key of base.yaml `paths`."""
    p = ROOT / BASE["paths"][key]
    if create:
        (p if not p.suffix else p.parent).mkdir(parents=True, exist_ok=True)
    return p


OUTPUT_COMPONENTS: list[str] = list(BASE["outputs"]["components"])
OUTPUT_KINDS: list[str] = list(BASE["outputs"]["kinds"])


def out_path(component: str, kind: str | None = None, create: bool = False) -> Path:
    """`outputs/<component>/<kind>`: every script keeps its logs, metrics, predictions and reports in its own folder."""
    if component not in OUTPUT_COMPONENTS:
        raise KeyError(f"unknown output component {component!r}; add it to base.yaml outputs.components")
    if kind is not None and kind not in OUTPUT_KINDS:
        raise KeyError(f"unknown output kind {kind!r}; one of {OUTPUT_KINDS}")
    p = ROOT / BASE["paths"]["outputs"] / component
    if kind:
        p = p / kind
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default
