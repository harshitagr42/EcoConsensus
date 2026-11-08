# Technical Requirements Document (TRD)

**Project:** EcoConsensus
**Companion to:** PRD.md, STACKING.md, APP_FLOW.md, IMPLEMENTATION_PLAN.md
**Version:** 3

Decided once, written down. Changes after sign-off go in the decision log at the bottom.

---

## 1. Environments

| Environment | Used for | Not used for |
|---|---|---|
| **MacBook Pro M3, 8 GB** | Writing code, tests, baselines, `test.py`, `inference.py`, stacking fit, MLflow UI | Fine-tuning IndicBERT or MuRIL |
| **Google Colab (T4 GPU)** | Experiment notebook, final fine-tuning (`src/training/`) | Unattended long runs (sessions drop) |
| **Kaggle Notebooks (2x T4 or P100)** | Same; two GPUs let more jobs run at once | Same |
| **LLM APIs (2 providers, free tier)** | Translation, stance classification | Anything needing guaranteed uptime |

### Python and libraries

| Item | Decision |
|---|---|
| Python | **3.13 or newer locally.** Colab and Kaggle use the version they ship (for example 3.11 or 3.12); code avoids newer-only syntax so it runs there too. If a dependency lacks 3.13 wheels, fall back to 3.12 and record it here |
| PyTorch | Latest stable. PyTorch has supported Python 3.13 since release 2.6 |
| Hugging Face `transformers` | **Latest release.** The release tracker I checked lists the 5.x line (5.14.x at that time); I could not confirm the exact newest version, so the experiment report records the installed version and `requirements.txt` pins it after the first successful run |
| `peft` (LoRA), `bitsandbytes` (QLoRA, CUDA only), `accelerate` | Latest. Python 3.13 support of `bitsandbytes` and `peft` was **not verified**; the notebook installs them and reports a skipped QLoRA job if that fails |
| `scikit-learn` | Baselines, metrics, grouped stratified splitting, meta-model (the main framework stays PyTorch) |
| `pandas` | Data handling |
| `mlflow` | Experiment tracking, SQLite store |
| `python-dotenv` | Reads `.env` |
| API clients | Provider SDK or `requests`, behind one interface |
| Standard library | `os`, `pathlib`, `json`, `logging`, `argparse`, `hashlib`, `datetime`, `unicodedata`, `re` |

Models: **MuRIL** `google/muril-base-cased`; **IndicBERT v2** `ai4bharat/IndicBERTv2-MLM-only` (278M parameters, MIT licence, BERT architecture; a safetensors file is available). Both are BERT-style, so LoRA targets `query` and `value`.

## 2. Repository layout

```text
EcoConsensus/
├── data/
│   ├── raw/
│   │   ├── original/            # Pol-NLI and SemEval originals (kept as they are)
│   │   └── translated/
│   │       ├── Hindi/           # pure, code_mix, hinglish per dataset
│   │       └── Bengali/         # same names as Hindi (to be translated); Bagheli folder is ignored
│   └── processed/               # frozen splits and exported clean tables
├── experiment/
│   ├── prepare_data.py          # Mac: builds experiment_data.zip (small SemEval subset)
│   ├── experiment.ipynb         # single notebook for Colab / Kaggle; outputs go to Drive
│   ├── SETUP.md
│   └── reports/                 # copy of the experiment report the method choice is based on
├── src/                         # built AFTER the experiment phase
│   ├── config.py  data_loading.py  preprocessing.py  splitting.py
│   ├── baselines.py  transformer_infer.py  llm_api.py  ensemble.py
│   ├── metrics.py  reporting.py  tracking.py  utils.py
│   └── training/                # Colab / Kaggle training code only
│       ├── train_transformer.py  methods.py  checkpointing.py
│       ├── train_inference.py    # builds meta-model input features (transformers + LLMs)
│       └── train_meta.py         # fits the meta-model from the feature files
├── prompts/stance_system_prompt.md
├── test/                        # evaluator drops the dataset here; results are written here
├── models/
│   ├── original/                # untouched base models or pointers
│   └── finetuned/               # <model>__<method>/ checkpoints + run_info.json
├── outputs/
│   ├── logs/  metrics/  predictions/  cache/        # cache = LLM responses
├── docs/                        # PRD, TRD, APP_FLOW, IMPLEMENTATION_PLAN, STACKING, INFERENCE
├── papers/                      # reference papers
├── test.py
├── inference.py
├── .env   .env.example          # .env is git-ignored
├── requirements.txt  requirements-train.txt
└── README.md
```

`src/` and the root scripts are written only after the fine-tuning method is chosen (end of the experiment phase). The notebook is self-contained and does not import from `src/`.

## 3. Data contract

**Internal record:** `src_id`, `lang`, `variant`, `claim`, `comment`, `label`.

- `src_id` = row position in the file. A translated file must keep the English file's row order; the notebook warns when row counts differ.
- Model inputs are `claim` and `comment` only. `lang`, `variant`, `src_id` and `label` are never features (language is used only to report results by language).
- Labels (final): **`favour = 0`, `neutral = 1`, `against = 2`**. Normalisation map: Pol-NLI `entailment` 0/1/2 and SemEval numeric 0/1/2 to favour/neutral/against; strings `favor`, `favour` to favour, `none`, `neutral`, `irrelevant` to neutral, `against` to against (case-insensitive); anything else is reported and excluded from metrics.
- Columns are matched by alias: comment = `premise`, `tweet`, `comment`, `text`; claim = `hypothesis`, `target`, `claim`; label = `entailment`, `stance`, `label`. A file that cannot be mapped fails with the list of columns it found.

### Evaluator file loader (used by `test.py`)

1. Find the file: `--file`, else `TEST_FILE`, else the first `.csv`, `.tsv`, `.json`, `.jsonl`, `.xlsx` in `test/`.
2. Read with the configured encoding and delimiter (sniffed if unset). Excel needs `openpyxl`.
3. Resolve columns in this order: flag, CONFIG value, alias match (case-insensitive, stripped). Language detection for per-language reporting: a language column if configured, otherwise script detection (Devanagari / Bengali / Latin) on the comment.
4. No claim column: fixed claim. No label column: prediction-only mode. No ID column: row number starting at 1.
5. Unknown label values: listed in the log and report, rows excluded from metrics, still predicted.
6. Empty comments: predicted as `neutral` with a warning, listed in the report.

### CONFIG block at the top of `test.py` (the on-the-spot switch)

```python
# ================= EVALUATOR DATASET CONFIG: edit here =================
TEST_FILE        = None   # None = first supported file in test/
ENCODING         = "utf-8-sig"
DELIMITER        = None   # None = sniff
COMMENT_COLUMN   = None   # None = auto-detect via aliases
CLAIM_COLUMN     = None
LABEL_COLUMN     = None
ID_COLUMN        = None   # None = row number
LANGUAGE_COLUMN  = None   # None = detect from script
FIXED_CLAIM      = "Climate change and global warming is a serious concern."
USE_FILE_CLAIM   = False  # True = use CLAIM_COLUMN text instead of FIXED_CLAIM
LABEL_MAP        = {"favor": "favour", "favour": "favour", "support": "favour", "0": "favour",
                    "none": "neutral", "neutral": "neutral", "irrelevant": "neutral", "1": "neutral",
                    "against": "against", "oppose": "against", "2": "against"}
# =======================================================================
```

Every item has a matching command-line flag (see INFERENCE.md); flags win over the block. The block's numeric mapping follows the project ids (0 = favour, 1 = neutral, 2 = against); the evaluator may number them differently, so it is the first thing to check when an evaluator file uses numbers.

## 4. Preprocessing

One function used by the notebook, `src/`, `test.py` and `inference.py`.

| Rule | Detail |
|---|---|
| Unicode | NFC |
| Emojis | Kept |
| `RT @user` prefix | `RT` removed |
| `@handle` | `@USER` |
| URLs | `URL` |
| `#SemST` | Removed (a dataset tag) |
| Other hashtags | `#` dropped, word kept |
| Whitespace | Collapsed |
| Case, punctuation, stop words | Untouched |

## 5. Splits

- **SemEval:** `StratifiedGroupKFold(10)`, 3 folds form **test** (about 30%), 7 form **dev** (about 70%). Stratified on the English label, grouped by `src_id`, fixed seed. The test ids are saved.
- **Experiment notebook only:** `prepare_data.py` takes a small SemEval subset (default 300 of 564 tweets, every `against` tweet kept, the rest stratified) in English + 3 Hindi + 3 Bengali variants. The notebook splits it by tweet, stratified: **train 60% / val 20% / test 20%**. This split is used only to choose fine-tuning methods (see STACKING.md section 3 for the caveat).
- **Pol-NLI:** stratified sample of training sentences (default 35%) with all their translations; Pol-NLI validation (90) kept as a second, in-domain validation set.
- Split sizes and class ratios are written into every report.

## 6. Baselines

| Baseline | Specification |
|---|---|
| Majority | Most frequent training label |
| TF-IDF + LR | Character n-grams (2 to 5, `char_wb`) united with word n-grams (1 to 2), Logistic Regression with balanced class weights |

## 7. Transformers

### Input
`tokenizer(claim, comment, truncation=True, max_length=128)`: a sentence pair; `AutoModelForSequenceClassification(num_labels=3)`. Language is not an input.

### Methods compared (full fine-tuning removed)

| ID | Method | Trains | LR (head / body) |
|---|---|---|---|
| `head` | New head only | Classifier + pooler | 5e-4 / n.a. |
| `last_layers` | Head + last N layers (N = 2) | Classifier, pooler, top N encoder layers | 5e-4 / 3e-5 |
| `lora` | LoRA r=16, alpha=32, dropout 0.1 on `query`, `value` | Adapters + classifier + pooler | 5e-4 / 3e-4 |
| `qlora` | Same adapters on a 4-bit NF4 base | Same | 5e-4 / 3e-4 |

IndicBERT v2 (MLM-only) ships without a trained pooler, so the pooler is treated as part of the new head. QLoRA is expected to give little benefit for models this small; it is included for the comparison and may fail on a given platform, in which case it is reported as failed or skipped.

### Training settings

| Concept | Setting |
|---|---|
| Optimiser | AdamW, weight decay 0.01 (applied to all trainable parameters; the bias / LayerNorm exclusion is a known simplification) |
| Scheduler | 10% linear warm-up, then linear decay |
| Loss | Cross-entropy with balanced class weights from the training data |
| Batch / length | 32 / 128 |
| Mixed precision | fp16 autocast with gradient scaling on CUDA |
| Gradient clipping | 1.0 |
| Dropout | Model default on the head |
| Epochs | 3 in the comparison; up to 10 with early stopping (patience 2, monitored on Pol-NLI validation **not** SemEval dev) for final training |
| Seeds | 42; add more if time allows |

**Why early stopping uses Pol-NLI validation in the final phase:** SemEval `dev` trains the meta-model, so it must not also pick base-model epochs.

### Final training: sequential, one model per run

Colab provides roughly 14.7 to 15.8 GB of VRAM, so the final phase does **not** run the two transformers at once.

1. `python -m src.training.train_transformer --model muril --method <chosen>` trains MuRIL alone, saves `ckpt_last` every epoch and `checkpoint_best`, writes `run_info.json`, then exits (freeing the GPU).
2. The same command with `--model indicbert` runs afterwards, in a fresh process.
3. `python -m src.training.train_inference` loads the saved best checkpoints **one at a time** (never both on the GPU), predicts SemEval `dev` and `test`, calls the two LLMs through the cache, and writes one feature file per component.
4. `python -m src.training.train_meta` fits the meta-model from the feature files (CPU, seconds).

Each step is resumable and writes its own outputs, so a dropped Colab session costs at most one epoch or one LLM batch. Steps 3 and 4 need no GPU once the transformer features exist (the LLM features can be built on any machine with the `.env` keys and the cache).

### Experiment notebook (method comparison only)

- **Data:** the SemEval subset above, uploaded to Google Drive as `experiment_data.zip` and unzipped by the notebook. Pol-NLI is not used in the experiment.
- **Runs:** model x method x seed, executed one after another in the notebook process (no parallel jobs). Each run trains on `train`, picks its best epoch on `val`; methods are ranked on `val` Macro-F1; `test` is reported for the recommended method only.
- **Claim:** the fixed claim (`CLAIM_MODE = "fixed"`), same as `test.py` and `inference.py`.
- **Resume:** splits and finished results are keyed by a hash of all settings and reused; an interrupted run starts again.
- **Outputs (Drive, `MyDrive/EcoConsensus/experiment_outputs/`):** `results/`, `predictions/`, `splits/`, `reports/`, MLflow `mlflow.db` + `mlartifacts/`.

## 8. LLM component

### Interface
One abstract `classify_batch(items) -> list[{id, label, status}]` with one adapter per provider. Provider, model, endpoint, batch size, temperature (0) and daily request limit come from `.env` and `configs/llm.yaml`. Keys are never logged.

### Prompt
`prompts/stance_system_prompt.md` (versioned `v2`; labels `favour`, `neutral`, `against`). Input is a JSON list of numbered comments; output is a JSON array of `{id, label}`. The claim is stated once in the message. `inference.py` and the single-entry path use the same prompt with a list of one.

### Batching
- Training, experiments, `test.py`: batches of 20 uncached comments per request (configurable; reduce if answers get truncated).
- Items in a batch use short local ids (1..n), mapped back to rows by code.
- A batch result with a missing or duplicate id retries the missing items individually once.

### Cache
- Append-only JSON Lines at `outputs/cache/<provider>__<model>__<prompt_version>.jsonl`.
- Key: SHA-256 of the preprocessed claim and comment. Stored: label, raw answer, timestamp.
- Only uncached items go into a batch. A prompt version change creates a new cache file.
- Cache files are git-ignored and may be copied between machines.

### Reliability
Retry with backoff on temporary errors; on a daily-limit error stop cleanly, report how many rows remain, and continue next run from the cache. Unparseable or refused answers become `unavailable` and trigger the ensemble fallback (STACKING.md).

## 9. Ensemble

Full design, request budget and training procedure: **STACKING.md**.

## 10. Tracking (MLflow)

- SQLite store `mlflow.db` (works the same on Colab, Kaggle and the Mac). Open with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
- Experiment notebook: `mlflow.db` and `mlartifacts/` live in the Drive outputs folder. On Colab the database is written to `/content` and copied to Drive after every run (SQLite on the Drive mount can hit locking errors).
- Experiments: `baselines`, `finetune_method_comparison`, `finetune_final`, `llm_prompt_check`, `stacking`, `final_comparison`.
- Run names: `<model>__<method>__seed<k>`.
- Model choices use `dev` and validation metrics only. SemEval `test` numbers appear only in `final_comparison` and in `test.py` output.

## 11. Metrics

One shared module. **Macro-F1** (labels fixed to the three classes), accuracy, per-class precision, recall, F1 and support, confusion matrix, weighted-F1, Matthews correlation coefficient, results by language, and a bootstrap 95% interval for Macro-F1 on small sets.

## 12. Reports

- Generated by code from saved results; no AI. Markdown; `test.py` also writes the `.txt` version.
- Names carry a timestamp. Files are opened in **create-only mode**; if the name exists, a numeric suffix is added. An existing report is never replaced.
- Where: experiment reports in `experiment_outputs/reports/` on Drive (copy the one the choice is based on into `experiment/reports/`); `test.py` reports and prediction file in `test/`; metrics in `outputs/metrics/`; logs in `outputs/logs/`; predictions in `outputs/predictions/`.

## 13. Secrets

`.env` holds keys (git-ignored); `.env.example` is committed with empty values:

```text
LLM_A_PROVIDER=  LLM_A_MODEL=  LLM_A_API_KEY=  LLM_A_BASE_URL=  LLM_A_DAILY_LIMIT=
LLM_B_PROVIDER=  LLM_B_MODEL=  LLM_B_API_KEY=  LLM_B_BASE_URL=  LLM_B_DAILY_LIMIT=
LLM_BATCH_SIZE=20
HF_TOKEN=            # optional, for gated or rate-limited model downloads
```

## 14. Logging

One log per run: `outputs/logs/<run>_<YYYYMMDD_HHMMSS>.txt` with the command, config, library versions, input pointer, preprocessing counts, per-row outputs, fallbacks and warnings, and the metrics summary.

## 15. Testing (on the Mac)

Preprocessing, loader with several fake evaluator formats, label maps, grouped split (no leakage), LLM parser with messy replies and batch id mismatches, cache hit and miss, metrics on hand-made predictions, report writer refuses to overwrite, a tiny forward pass if memory allows, and a full `test.py` run on 3 rows with a stubbed LLM.

## 16. Decision log

| # | Decision | Reason |
|---|---|---|
| 1 | Language and variant are not model inputs | Evaluator data will not have them |
| 2 | Claim and comment as a sentence pair | Standard for two-text input |
| 3 | Emojis kept, handles to `@USER` | Stance signal |
| 4 | Stratified, grouped splits | Equal class ratios, no translation leakage |
| 5 | Class-weighted loss | Imbalance, Macro-F1 |
| 6 | Provider-agnostic LLM layer with cache and batches | Free-tier limits |
| 7 | Full fine-tuning removed | Overfitting risk on small data, user decision |
| 8 | Base models trained on Pol-NLI only; meta-model on SemEval dev | No retraining folds needed, honest meta features (STACKING.md) |
| 9 | Parallel subprocess jobs in one notebook | Faster comparison on one or two GPUs |
| 10 | Bagheli ignored, Bengali to be added | Assignment languages are Hindi and Bengali |
| 11 | Claim mode `fixed` | Evaluator claim is unknown; model sees the same claim at train and test |
| 12 | Experiment phase before `src/` | Method choice decides what `src/` contains |
| 13 | `test.py` writes timestamped results | "Never overwrite" overrides a fixed `result.txt` name |
| 14 | Final transformers trained one at a time; features built by `train_inference.py`; meta-model trained by `train_meta.py` | Colab VRAM is about 15 GB, so no simultaneous training; separate steps are resumable |
| 15 | Label ids fixed: 0 = favour, 1 = neutral, 2 = against; the third class is called `neutral` everywhere (was `none`) | One mapping for data, models, LLM prompt (v2), feature files and reports |
| 16 | Experiment uses only a small SemEval subset (English + 3 Hindi + 3 Bengali variants), split train / val / test by tweet; all `against` tweets kept | Fast method comparison; enough `against` examples for a readable Macro-F1 |
| 17 | Experiment data shipped as a zip on Drive; results, reports and MLflow (SQLite, synced from `/content`) written to Drive | Survives Colab disconnects; one place to collect everything |
