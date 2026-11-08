# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Stance detection (CSET 346 NLP course project): given the fixed claim **"Climate change and global warming is a serious concern."** and a Hindi/Bengali comment, predict `favour` / `neutral` / `against` (ids **0 / 1 / 2**, final — use this mapping everywhere). Primary metric is **Macro-F1**. The final system is a stacked ensemble: two fine-tuned transformers (MuRIL `google/muril-base-cased`, IndicBERT v2 `ai4bharat/IndicBERTv2-MLM-only`) + two LLMs via API → logistic-regression meta-model.

## Current state (check before assuming code exists)

The project is **docs-first and two-phase**. Only Phase 1 exists:
- `experiment/prepare_data.py` — runs locally, builds `experiment/experiment_data.zip` from **SemEval only**: a small subset of tweets (default 300, every `against` tweet kept), each in English + 3 Hindi + 3 Bengali variants. Translated files must be row-aligned with the English file (checked).
- `experiment/experiment.ipynb` — self-contained Colab notebook (Kaggle as backup) comparing fine-tuning methods (`head_only`, `last_n`, `lora`, `qlora`). It unzips the bundle from `MyDrive/EcoConsensus/`, splits by tweet into train/val/test, and writes results, reports and MLflow (`mlflow.db` + `mlartifacts/`) to `MyDrive/EcoConsensus/experiment_outputs/`. Only cell 1 (CONFIG) is meant to be edited. It does **not** import from `src/`. `experiment/SETUP.md` is the run guide.

`src/` (beyond an empty `__init__.py`), `test.py`, `inference.py`, `requirements*.txt` and tests are **not written yet**. They are specified in `docs/` and are built only after the experiment picks one method per model. When writing them, follow the specs rather than inventing structure:
- `docs/TRD.md` — the authoritative technical spec: repo layout (§2), data contract & evaluator loader + `test.py` CONFIG block (§3), preprocessing rules (§4), splits (§5), training settings (§7), LLM layer/batching/cache (§8), reports (§12), test list (§15), decision log (§16 — append to it when a decision changes).
- `docs/STACKING.md` — feature files, meta-model training, 15 subset fallback models.
- `docs/APP_FLOW.md`, `docs/PRD.md`, `docs/IMPLEMENTATION_PLAN.md` (staged plan with sign-offs/cut-lines), `docs/INFERENCE.md` (CLI usage of `test.py`/`inference.py`).
- `prompts/stance_system_prompt.md` — LLM prompt, versioned (`v2`); bumping the version changes the cache key.

## Commands

Local env is a Python 3.13 venv at `.venv` (Mac M3, 8 GB — no local fine-tuning).

```bash
source .venv/bin/activate
python experiment/prepare_data.py                 # build experiment_data.zip (pandas only)
python experiment/prepare_data.py --rows 0                       # all 564 tweets
python experiment/prepare_data.py --root PATH --out DIR --zip FILE
```

The notebook needs torch/transformers/peft/mlflow, which are not in `.venv`. To smoke-test it locally, run its code cells in a separate venv with a tiny model (e.g. `hf-internal-testing/tiny-random-BertForSequenceClassification`), `INSTALL_DEPS=False`, one seed. Edit the `.ipynb` JSON programmatically (cell sources), keep outputs empty. Planned Phase 2 commands (per TRD/STACKING, not yet implemented):

```bash
python -m src.training.train_transformer --model muril --method <chosen>      # GPU, one model per process
python -m src.training.train_transformer --model indicbert --method <chosen>
python -m src.training.train_inference --splits dev test --components muril indicbert llm_a llm_b
python -m src.training.train_meta                                             # CPU
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Architecture invariants (span multiple docs; easy to violate)

- **Data roles are strict** to avoid stacking leakage: transformers train only on Pol-NLI (early stopping on Pol-NLI validation, never SemEval); the meta-model trains only on SemEval **dev**; SemEval **test** is scored once at the end. (The experiment phase is separate: it trains and ranks methods on its own SemEval subset split; only the method choice carries over.)
- **Splits are grouped by `src_id`** (row position in the English file). Every translated file must be row-aligned with its English source so a tweet and all its translations land in the same split/fold.
- **Model inputs are only `(claim, comment)` as a sentence pair**, max_length 128. `lang`, `variant`, `src_id` are never features (evaluator data won't have them).
- **One shared preprocessing function** used everywhere (NFC, keep emojis, `@handle`→`@USER`, URLs→`URL`, drop `#SemST`, strip `#` from hashtags, collapse whitespace; no lowercasing/stopword removal).
- **Transformers trained/loaded one at a time** (~15 GB Colab VRAM). Each pipeline step is resumable and writes its own output.
- **LLM layer**: provider-agnostic `classify_batch`, config from `.env` (`LLM_A_*`, `LLM_B_*`, `LLM_BATCH_SIZE`), batches of ~20 with local ids 1..n, append-only JSONL cache at `outputs/cache/<provider>__<model>__<prompt_version>.jsonl` keyed by SHA-256 of preprocessed claim+comment. Daily-limit errors stop cleanly and resume from cache.
- **Reports are generated by code, never overwritten**: timestamped names, create-only open, numeric suffix on clash.

## Gotchas

- Label spelling is British `favour`; SemEval raw files say `FAVOR`/`NONE`/`AGAINST` and are mapped by explicit label maps — never rely on implicit ordering.
- `data/raw/translated/Bagheli/` is a mistaken translation (different language) and must never be used; `prepare_data.py` lists its input files explicitly for that reason.
- The LLM prompt is versioned (`v2`) and the version is part of the LLM cache key: bump it on any prompt change.
- `experiment/experiment_data/` and `experiment/experiment_data.zip` are build outputs of `prepare_data.py`.
