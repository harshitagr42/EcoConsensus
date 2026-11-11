"""The ONE text-cleaning function used everywhere (prepare_data, baselines, LLM calls, inference.py).

Rules (TRD §5): NFC, noise list (#SemST, leading RT) removed, URLs -> URL, @handles -> @USER,
'#' dropped from hashtags (word kept), whitespace collapsed. Emojis, case, punctuation and every script are kept.
"""
from __future__ import annotations

import re
import unicodedata

from src.config import BASE

_P = BASE.get("preprocessing", {})
HANDLE_TOKEN = _P.get("handle_token", "@USER")
URL_TOKEN = _P.get("url_token", "URL")
EMPTY_MARKER = ""

_NOISE = [re.compile(p, re.I) for p in _P.get("noise_patterns", [r"#SemST\b"])]
_RT = re.compile(r"^\s*RT\b[\s:]*", re.I)
_URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
_HANDLE = re.compile(r"(?<![\w@])@\w+")
_HASH = re.compile(r"#(?=[^\s#])")
_WS = re.compile(r"\s+")


def preprocess(text) -> str:
    """Clean one comment. None/NaN -> '' (EMPTY_MARKER); the loader flags such rows."""
    if text is None or (isinstance(text, float) and text != text):
        return EMPTY_MARKER
    t = unicodedata.normalize("NFC", str(text))
    for rx in _NOISE:
        t = rx.sub(" ", t)
    if _P.get("drop_leading_rt", True):
        t = _RT.sub("", t)
    t = _URL.sub(URL_TOKEN, t)
    t = _HANDLE.sub(HANDLE_TOKEN, t)
    if _P.get("strip_hashtag_symbol", True):
        t = _HASH.sub("", t)
    return _WS.sub(" ", t).strip()


def is_empty(text: str) -> bool:
    return not text or not text.strip()


def script_group(text: str) -> str:
    """'devanagari', 'bengali' or 'latin' by the majority of letters (for reporting only, never a feature)."""
    dev = ben = lat = 0
    for ch in str(text):
        o = ord(ch)
        if 0x0900 <= o <= 0x097F or 0xA8E0 <= o <= 0xA8FF:
            dev += 1
        elif 0x0980 <= o <= 0x09FF:
            ben += 1
        elif ch.isalpha():
            lat += 1
    if dev == ben == 0:
        return "latin"
    return "devanagari" if dev >= ben else "bengali"
