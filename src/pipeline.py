"""Shared "run the components on rows" code for test.py and inference.py.

    models = load_models(needed_components("ensemble"))           # baselines, meta-models (exit 4 if missing)
    res = run_pipeline(df, models, components=..., use_llm=True)  # df from data_loading.load_file
    res.preds["stacked"], res.variant, res.llm_info

Same feature table and meta-model fallback as training (src/ensembling.py): the stacked model uses the first variant
whose components all answered for that row (all four -> without LLM-A -> without LLM-B -> transformers only); a row
with no transformer output falls back to TF-IDF. Nothing here imports src/training/ or MLflow.
"""
from __future__ import annotations

import gc
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.baselines import Majority, pair_text
from src.config import CFG, FIXED_CLAIM, LABELS, ROOT, path
from src.data_loading import DEFAULT_LABEL_MAP
from src.ensembling import (LLMS, TRANSFORMERS, VARIANTS, add_llm, add_transformer, component_pred, majority_vote,
                            predict_with_fallback, soft_vote, weights_table)
from src.preprocessing import is_empty, preprocess, script_group

VERSION = "1.0.0"

# exit codes shared by test.py and inference.py (TESTING.md / INFERENCE.md)
EXIT_OK, EXIT_CONFIG, EXIT_DATA, EXIT_MODELS, EXIT_LLM = 0, 2, 3, 4, 5

DISPLAY = {"majority": "Majority", "tfidf_lr": "TF-IDF + LR", "muril": "MuRIL", "indicbert": "IndicBERT",
           "llm_a": "LLM-A", "llm_b": "LLM-B", "soft_vote": "Soft vote (2 transformers)",
           "majority_vote": "Majority vote (4 components)", "stacked": "Ensemble (stacked)"}
PS_LABELS = ["Favour", "None", "Against"]               # problem-statement spelling of ids 0 / 1 / 2

# Text labels only: numeric labels are never mapped by default (0/1/2 mean different things in different files).
TEXT_LABEL_MAP: dict[str, str] = {k: v for k, v in DEFAULT_LABEL_MAP.items() if not k.strip().isdigit()}
TEXT_LABEL_MAP.update({"unrelated": "neutral", "favor": "favour", "none": "neutral"})
_TARGET = {"favour": "favour", "favor": "favour", "none": "neutral", "neutral": "neutral", "against": "against",
           "0": "favour", "1": "neutral", "2": "against"}

MODES = ["ensemble", "transformers", "muril", "indicbert", "llm-a", "llm-b", "tfidf", "majority"]


class ModelFilesMissing(Exception):
    """Exit code 4."""


class LLMRequired(Exception):
    """An LLM was unavailable and --require-llm was set (exit code 5)."""


# ---------------------------------------------------------------- label maps and label output
def normalise_label_map(m: dict) -> dict[str, str]:
    """{'FAVOR': 'favour', 'x': 'none', ...} -> lower-cased keys, targets favour / neutral / against."""
    out = {}
    for k, v in (m or {}).items():
        t = _TARGET.get(str(v).strip().lower())
        if t is None:
            raise ValueError(f"label_map: {k!r} -> {v!r}; the target must be favour, none (neutral) or against")
        out[str(k).strip().lower()] = t
    return out


def parse_label_map_arg(text: str) -> dict[str, str]:
    """'A=favour,B=none,C=against' -> {'a': 'favour', 'b': 'neutral', 'c': 'against'}."""
    pairs = {}
    for part in [p for p in text.split(",") if p.strip()]:
        if "=" not in part:
            raise ValueError(f"--label-map: '{part}' is not VALUE=CLASS")
        k, v = part.rsplit("=", 1)
        pairs[k.strip()] = v.strip()
    return normalise_label_map(pairs)


def file_label_spelling(df: pd.DataFrame) -> dict[int, str] | None:
    """For --label-style file: the most common raw spelling of each class in the input's label column."""
    if "label_raw" not in df or df["label_raw"].isna().all():
        return None
    out = {}
    for i in range(3):
        vals = df.loc[df["label"] == i, "label_raw"].dropna()
        if len(vals):
            out[i] = str(vals.value_counts().index[0])
    return out if out else None


def format_labels(ids, style: str = "ps", file_map: dict[int, str] | None = None) -> list[str]:
    """Label ids -> strings. ps: Favour / None / Against; numeric: 0 / 1 / 2; file: the input file's spelling."""
    res = []
    for i in ids:
        i = int(i)
        if i < 0:
            res.append("")
        elif style == "numeric":
            res.append(str(i))
        elif style == "file" and file_map and i in file_map:
            res.append(file_map[i])
        else:
            res.append(PS_LABELS[i])
    return res


def rows_from_texts(texts: list[str], claim: str | None = None) -> pd.DataFrame:
    """The internal record format for comments typed on the command line (inference.py -t)."""
    claim = preprocess(claim) if claim else FIXED_CLAIM
    comments = [preprocess(t) for t in texts]
    return pd.DataFrame({"id": [str(i) for i in range(len(texts))], "source_id": [str(i) for i in range(len(texts))],
                         "comment_raw": texts, "comment": comments, "claim": claim,
                         "empty": [is_empty(c) for c in comments], "script_group": [script_group(c) for c in comments],
                         "language": None, "label": pd.array([pd.NA] * len(texts), dtype="Int64"),
                         "label_raw": None})


# ---------------------------------------------------------------- what is needed, and loading it
def needed_components(mode: str = "ensemble", use_llm: bool = True) -> list[str]:
    """Components to run for an inference mode. test.py always runs everything ("ensemble")."""
    need = {"ensemble": TRANSFORMERS + LLMS, "transformers": TRANSFORMERS, "muril": ["muril"],
            "indicbert": ["indicbert"], "llm-a": ["llm_a"], "llm-b": ["llm_b"], "tfidf": [], "majority": []}[mode]
    return [c for c in need if use_llm or c not in LLMS]


def check_model_files(components: list[str], need_meta: bool) -> list[str]:
    """Missing model files (relative paths), empty when everything is in place."""
    missing = []
    for c in components:
        if c in TRANSFORMERS:
            d = ROOT / CFG["transformers"]["models"][c]["adapter_dir"]
            for f in ("adapter_config.json", "adapter_model.safetensors"):
                if not (d / f).exists():
                    missing.append(str((d / f).relative_to(ROOT)))
    b = path("baselines")
    for f in ("majority.json", "tfidf_lr.joblib"):
        if not (b / f).exists():
            missing.append(str((b / f).relative_to(ROOT)))
    if need_meta and not (path("meta_models") / "meta_transformers.joblib").exists():
        missing.append(str((path("meta_models") / "meta_transformers.joblib").relative_to(ROOT)))
    return missing


@dataclass
class Models:
    majority: Majority
    tfidf: object
    meta: dict[str, object]                   # variant -> fitted LogisticRegression
    meta_info: dict = field(default_factory=dict)


def load_models(components: list[str], need_meta: bool = True) -> Models:
    """Baselines and meta-models from models/finetuned/. Raises ModelFilesMissing (exit 4) naming the files.
    The joblib files are scikit-learn pickles: only load files you created yourself."""
    missing = check_model_files(components, need_meta)
    if missing:
        raise ModelFilesMissing("Model files missing: " + ", ".join(missing) +
                                ". Restore models/finetuned/ (see src/training/README.md).")
    b, d = path("baselines"), path("meta_models")
    meta = {v: joblib.load(d / f"{v}.joblib") for v in VARIANTS if (d / f"{v}.joblib").exists()} if need_meta else {}
    info_p = d / "meta_info.json"
    info = json.loads(info_p.read_text(encoding="utf-8")) if info_p.exists() else {}
    return Models(Majority.load(b / "majority.json"), joblib.load(b / "tfidf_lr.joblib"), meta, info)


# ---------------------------------------------------------------- transformers
def pick_device(device: str = "auto"):
    import torch
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda: no CUDA GPU found")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("--device mps: Apple MPS is not available")
    return torch.device(device)


class TransformerComponent:
    """Base model from the Hugging Face Hub + the LoRA adapter (and trained classifier/pooler) from models/finetuned/.
    Input: (claim, comment) sentence pair, max_length from configs/transformers.yaml, same as in training."""

    def __init__(self, name: str, device: str = "auto", batch_size: int = 32):
        cfg = CFG["transformers"]["models"][name]
        self.name, self.hf_name = name, cfg["hf_name"]
        self.adapter_dir = ROOT / cfg["adapter_dir"]
        self.max_length = int(CFG["transformers"].get("max_length", 128))
        self.device_arg, self.batch_size = device, batch_size
        self.model = self.tokenizer = self.device = None

    def load(self) -> "TransformerComponent":
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        self.device = pick_device(self.device_arg)
        self.tokenizer = AutoTokenizer.from_pretrained(self.adapter_dir)
        kw = dict(num_labels=3, id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)},
                  ignore_mismatched_sizes=True)      # the classifier is replaced by the trained one in the adapter
        if self.device.type == "mps":
            kw["attn_implementation"] = "eager"
        base = AutoModelForSequenceClassification.from_pretrained(self.hf_name, **kw)
        self.model = PeftModel.from_pretrained(base, self.adapter_dir).to(self.device).eval()
        return self

    def predict_proba(self, claims: list[str], comments: list[str]) -> np.ndarray:
        import torch
        if self.model is None:
            self.load()
        enc = self.tokenizer(list(claims), list(comments), truncation=True, max_length=self.max_length)
        lengths = [len(x) for x in enc["input_ids"]]
        order = np.argsort(lengths)                      # length-sorted batches: less padding
        keys = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc]
        pad = self.tokenizer.pad_token_id or 0
        probs = np.zeros((len(lengths), 3), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, len(order), self.batch_size):
                idx = order[s:s + self.batch_size]
                width = max(lengths[i] for i in idx)
                batch = {k: torch.tensor([enc[k][i] + [pad if k == "input_ids" else 0] * (width - lengths[i])
                                          for i in idx], dtype=torch.long, device=self.device) for k in keys}
                logits = self.model(**batch).logits.float()
                probs[idx] = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def unload(self) -> None:
        self.model = self.tokenizer = None
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def transformer_probs(names: list[str], df: pd.DataFrame, device: str = "auto", log=None) -> dict[str, np.ndarray]:
    """One model in memory at a time (8 GB machines): load, predict every row, free."""
    out = {}
    for n in names:
        if log:
            log.info("%s: loading %s + adapter %s", n, CFG["transformers"]["models"][n]["hf_name"],
                     CFG["transformers"]["models"][n]["adapter_dir"])
        comp = TransformerComponent(n, device).load()
        if log:
            log.info("%s: predicting %d rows on %s", n, len(df), comp.device)
        out[n] = comp.predict_proba(df["claim"].tolist(), df["comment"].tolist())
        comp.unload()
    return out


# ---------------------------------------------------------------- LLMs
def make_llm(slot: str, claim: str, batch_size: int | None, max_requests: int | None):
    from src.llm_api import LLMComponent, few_shot_from_config, load_prompt
    prompt = None if claim == FIXED_CLAIM else load_prompt(claim=claim, few_shot=few_shot_from_config())
    return LLMComponent(slot, prompt=prompt, batch_size=batch_size, max_requests=max_requests)


def llm_labels(slot: str, df: pd.DataFrame, batch_size: int | None = None, max_requests: int | None = None,
               log=None, factory=None) -> tuple[list[int | None], dict]:
    """Label id per row (None = no answer) and a usage summary. Empty comments are never sent.
    Answers come from the cache first; the rest are sent in batches (inference.py uses batch size 1)."""
    from src.llm_api import FatalProviderError
    factory = factory or make_llm
    labels: list[int | None] = [None] * len(df)
    info = dict(component=f"llm_{slot}", status="ok", requests=0, cache_hits=0, answered=0, unavailable=0,
                parse_failures=0, stopped=None, provider=None, model=None, prompt_version=None, errors={})
    send = ~df["empty"].to_numpy(dtype=bool)
    for claim in pd.unique(df.loc[send, "claim"]):
        idx = np.flatnonzero(send & (df["claim"] == claim).to_numpy())
        try:
            comp = factory(slot, claim, batch_size, max_requests)
        except (FatalProviderError, ValueError) as e:
            info.update(status="not configured", stopped=str(e))
            if log:
                log.warning("LLM-%s unavailable: %s", slot.upper(), e)
            return labels, info
        res = comp.run(df["comment"].iloc[idx].tolist(), claim=claim, progress_every=5)
        for i, r in zip(idx, res):
            labels[i] = r["label"] if r["status"] == "ok" else None
        st = comp.stats
        for k in ("requests", "cache_hits", "parse_failures"):
            info[k] += st.get(k, 0)
        for k, v in st.get("errors", {}).items():
            info["errors"][k] = info["errors"].get(k, 0) + v
        info.update(provider=comp.cfg.provider, model=comp.cfg.model, prompt_version=comp.prompt.version)
        if comp.stopped:
            info.update(status="stopped", stopped=comp.stopped)
            if comp.interrupted:
                raise KeyboardInterrupt
    info["answered"] = sum(l is not None for l in labels)
    info["unavailable"] = int(send.sum()) - info["answered"]
    if info["status"] == "ok" and info["unavailable"]:
        info["status"] = "partial"
    return labels, info


# ---------------------------------------------------------------- the run
@dataclass
class PipelineResult:
    table: pd.DataFrame                          # the 12 features + <component>_available per row
    preds: dict[str, np.ndarray]                 # system -> label ids (-1 = no answer)
    variant: np.ndarray                          # meta-model variant used per row ("tfidf_lr fallback" if none)
    stacked_probs: np.ndarray | None
    llm_info: dict[str, dict]


def run_pipeline(df: pd.DataFrame, models: Models, components: list[str], *, device: str = "auto",
                 llm_batch_size: int | None = None, llm_budget: int | None = None, require_llm: bool = False,
                 log=None, transformer_fn=None, llm_factory=None) -> PipelineResult:
    """Run the requested components on preprocessed rows and combine them. Baselines always run (cheap); the stacked
    model runs when meta-models are loaded. Raises LLMRequired when require_llm and an LLM left rows unanswered."""
    transformer_fn = transformer_fn or transformer_probs          # looked up at call time (tests replace them)
    llm_factory = llm_factory or make_llm
    table = pd.DataFrame(index=range(len(df)))
    tr = [c for c in TRANSFORMERS if c in components]
    probs = transformer_fn(tr, df, device, log) if tr else {}
    for c in TRANSFORMERS:
        add_transformer(table, c, probs.get(c))
    llm_info = {}
    for c in LLMS:
        if c in components:
            labels, llm_info[c] = llm_labels(c[-1], df, llm_batch_size, llm_budget, log, llm_factory)
            add_llm(table, c, labels)
            if require_llm and llm_info[c]["unavailable"] + (llm_info[c]["status"] == "not configured") > 0:
                raise LLMRequired(f"{DISPLAY[c]} unavailable ({llm_info[c]['status']}: "
                                  f"{llm_info[c]['stopped'] or str(llm_info[c]['unavailable']) + ' rows without an answer'})")
        else:
            add_llm(table, c, None)

    texts = pair_text(df)
    preds = {"majority": np.asarray(models.majority.predict(texts), dtype=int),
             "tfidf_lr": np.asarray(models.tfidf.predict(texts), dtype=int)}
    for c in TRANSFORMERS + LLMS:
        if c in components:
            preds[c] = component_pred(table, c)
    if all(c in components for c in TRANSFORMERS):
        preds["soft_vote"] = soft_vote(table)
        if any(c in components for c in LLMS):
            preds["majority_vote"] = majority_vote(table)
    variant, stacked_probs = np.full(len(df), None, dtype=object), None
    if models.meta and all(c in components for c in TRANSFORMERS):
        p, stacked_probs, variant = predict_with_fallback(table, models.meta)
        no_variant = p < 0
        p = np.where(no_variant, preds["tfidf_lr"], p)
        variant = np.where(no_variant, "tfidf_lr fallback", variant)
        preds["stacked"] = p.astype(int)
    return PipelineResult(table, preds, variant, stacked_probs, llm_info)


def final_prediction(res: PipelineResult, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """inference.py: the label per row for a --mode and where it came from. A single-LLM mode falls back to TF-IDF
    on rows the LLM did not answer."""
    key = {"ensemble": "stacked", "transformers": "stacked", "muril": "muril", "indicbert": "indicbert",
           "llm-a": "llm_a", "llm-b": "llm_b", "tfidf": "tfidf_lr", "majority": "majority"}[mode]
    p = res.preds[key].copy()
    src = res.variant.copy() if key == "stacked" else np.full(len(p), key, dtype=object)
    miss = p < 0
    if miss.any():
        p[miss] = res.preds["tfidf_lr"][miss]
        src[miss] = "tfidf_lr fallback"
    return p, src


def meta_weights_md(models: Models, variant: str = "meta_all4") -> str:
    from src.ensembling import variant_cols
    m = models.meta.get(variant)
    if m is None:
        return ""
    t = weights_table(m, variant_cols(variant))
    return "```\n" + t.round(3).to_string() + "\n```"
