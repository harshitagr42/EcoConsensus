# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Stance detection (CSET 346 NLP course project): given the fixed claim **"Climate change and global warming is a serious concern."** and a Hindi/Bengali comment, predict `favour` / `neutral` / `against` (ids **0 / 1 / 2**, final — use this mapping everywhere; the LLM prompt says NONE for neutral). Primary metric is **Macro-F1**. The final system is a stacked ensemble: two fine-tuned transformers (MuRIL `google/muril-base-cased`, IndicBERT v2 `ai4bharat/IndicBERTv2-MLM-only`) + two LLMs via API → logistic-regression meta-model.

## Current state (check before assuming code exists)

Phase 1 (experiment, `experiment/`) is done: LoRA chosen for both models. Phase 2 code for Days 1–3 exists
(runbook: `src/training/README.md`):
- `configs/{base,transformers,llm}.yaml`; `src/config.py`, `utils.py` (seed, logging, `safe_write`, MLflow), `preprocessing.py`
  (the one cleaning function), `data_loading.py` (flexible evaluator loader), `splitting.py`, `metrics.py`, `reporting.py`,
  `baselines.py`, `llm_api.py` (OpenAI-compatible adapter, batching, cache, budget guard), `ensembling.py`.
- `src/training/prepare_data.py` builds the frozen, **already cleaned** splits `data/processed/{train_base,meta,val,semeval_test}.csv`
  and `data/ecoconsensus_data.zip` (uploaded to Drive). Pol-NLI train is deduplicated by English text and rows equal to
  val/SemEval comments are removed (TRD decision 21).
- `src/training/train_muril.ipynb`, `train_indicbert.ipynb`: **self-contained** Colab notebooks (no `src/` imports, no cleaning),
  run in parallel on two accounts; they write `<model>_results.zip` with project-relative paths (adapter, `probs_<model>_<split>.csv`).
  The two notebooks are identical except for the model lines in CONFIG; edit both the same way.
- `src/training/run_llm_batch.py`, `train_inference.py` (joins probs + LLM cache → `features_<split>.csv`), `train_meta.py`.
- `tests/` (pytest). Not written yet (Day 4+): `src/pipeline.py`, `test.py`, `inference.py`, README.

Specs: `docs/TRD.md` (decision log §20 — append when a decision changes), `docs/STACKING_ENSEMBLE.md`, `docs/APP_FLOW.md`,
`docs/IMPLEMENTATION_PLAN.md`, `docs/INFERENCE.md`, `docs/TESTING.md`. The LLM prompt the code reads is
`prompts/stance_system_prompt.md` (`prompt_version: v3`); `docs/LLM_PROMPT.md` is its rationale.

## Commands

Local env is a Python 3.13 venv at `.venv` (Mac M3, 8 GB — no local fine-tuning).

```bash
source .venv/bin/activate
python -m src.training.prepare_data [--overwrite-splits]
python -m src.baselines
python -m pytest -q tests
python -m src.training.run_llm_batch --provider a b --split meta val [--check|--pilot|--estimate|--score]
python -m src.training.train_inference
python -m src.training.train_meta
mlflow ui --backend-store-uri sqlite:///outputs/metrics/mlflow.db
```

To smoke-test a training notebook locally, exec its code cells in a separate venv (torch, transformers 5.22.2, peft 0.21.2)
with `MODEL_NAME = "hf-internal-testing/tiny-random-BertForSequenceClassification"`, `INSTALL_DEPS = False`,
`LOCAL_DATA_DIR = "data/processed"`, `LOCAL_OUT_DIR` in a temp folder, `TRAIN_LIMIT` small. Keep notebook outputs empty.
`experiment/prepare_data.py` / `experiment.ipynb` belong to the finished experiment only.

## Architecture invariants (span multiple docs; easy to violate)

- **Data roles are strict** to avoid stacking leakage: transformers train only on Pol-NLI (early stopping on Pol-NLI validation, never SemEval); the meta-model trains only on SemEval **dev**; SemEval **test** is scored once at the end. (The experiment phase is separate: it trains and ranks methods on its own SemEval subset split; only the method choice carries over.)
- **Splits are grouped by `src_id`** (row position in the English file). Every translated file must be row-aligned with its English source so a tweet and all its translations land in the same split/fold.
- **Model inputs are only `(claim, comment)` as a sentence pair**, max_length 128. `lang`, `variant`, `src_id` are never features (evaluator data won't have them).
- **One shared preprocessing function** used everywhere (NFC, keep emojis, `@handle`→`@USER`, URLs→`URL`, drop `#SemST`, strip `#` from hashtags, collapse whitespace; no lowercasing/stopword removal).
- **One transformer per GPU session** (~15 GB Colab VRAM): MuRIL and IndicBERT train in parallel on two Colab accounts. Each pipeline step is resumable and writes its own output.
- **LLM layer**: provider-agnostic `classify_batch`, config from `.env` (`LLM_A_*`, `LLM_B_*`, `LLM_BATCH_SIZE`, set by the pilot) + `configs/llm.yaml`, local ids 1..n per request, append-only JSONL cache at `outputs/cache/<provider>__<model>__<prompt_version>.jsonl` keyed by SHA-256 of preprocessed claim+comment. Daily-limit errors stop cleanly and resume from cache. `semeval_test` is refused until `prompt_frozen: true`.
- **Reports are generated by code, never overwritten**: timestamped names, create-only open, numeric suffix on clash.

## Gotchas

- Label spelling is British `favour`; SemEval raw files say `FAVOR`/`NONE`/`AGAINST` and are mapped by explicit label maps — never rely on implicit ordering.
- `data/raw/translated/Bagheli/` is a mistaken translation (different language) and must never be used; `prepare_data.py` lists its input files explicitly for that reason.
- The LLM prompt is versioned (`v3`, few-shot adds `-fs<hash>`) and the version is part of the LLM cache key: bump it on any prompt change.
- `data/processed/*`, `data/ecoconsensus_data.zip`, `models/`, `outputs/` are build outputs and git-ignored (as are `experiment/experiment_data*`).
- pandas 3: `groupby(...).apply` drops the grouping column; avoid it.
