# Technical Requirements Document (TRD)

**Project:** EcoConsensus
**Version:** v2
**Companions:** `PRD.md`, `APP_FLOW.md`, `STACKING_ENSEMBLE.md`, `INFERENCE.md`, `TESTING.md`, `LLM_PROMPT.md`, `REFERENCES.md`

Decided once, written down. Any change after sign-off goes in the decision log (section 20).

---

## 1. Environments

| Environment | Used for | Not used for |
|---|---|---|
| **MacBook Pro M3, 8 GB** | Writing code, unit tests, preprocessing, splits, baselines, `inference.py`, `test.py`, MLflow UI | Training MuRIL or IndicBERT (memory) |
| **Google Colab (T4, about 14.7–15.8 GB)** | Transformer training, feature building | Long unattended runs (sessions disconnect) |
| **Kaggle Notebooks (T4/P100)** | Same as Colab, as a backup | Same |
| **LLM provider APIs (2, free tier)** | Stance classification, translation (done once) | Anything needing guaranteed uptime |

Because of the GPU memory limit, **each transformer is trained on its own GPU**: MuRIL and IndicBERT train **in parallel on two Colab accounts**, each with its own self-contained notebook (decision 19). No session trains two models at once.

## 2. Python and libraries

**Python:** 3.13 or newer is the target (the experiment ran on 3.13.16). A lower version is acceptable only if a dependency forces it; the floor is 3.11. Set `requires-python = ">=3.13"` and say so in the README.

**Versions tested in the experiment (Colab, 2026-11-08):** pin these in `requirements.txt` and update to the latest only after re-running the smoke tests.

| Library | Tested version | Purpose |
|---|---|---|
| `torch` | 2.14.1 | Deep learning (PyTorch is the main framework) |
| `transformers` | 5.22.2 | Models, tokenizers |
| `peft` | 0.21.2 | LoRA adapters |
| `accelerate` | 1.15.0 | Training utilities |
| `bitsandbytes` | 0.50.2 | 4-bit quantisation (Colab/Kaggle only; QLoRA failed, see section 8) |
| `scikit-learn` | 1.9.1 | Baselines, metrics, splitting, meta-model |
| `pandas` | 3.1.1 | Data handling |
| `mlflow` | 3.18.0 | Experiment tracking |
| `python-dotenv` | latest | Read `.env` |
| `pyyaml`, `requests`, `joblib`, `openpyxl` | latest | Config, HTTP, saving sklearn models, XLSX input |

**Versions used for the final training (Colab, 2026-11-10, both notebooks, T4 GPU):** read from each adapter's
`run_info.json` (`env`). Cell 2 of each notebook installs `PINNED` with `pip install --upgrade`; `torchao` was added
and `--upgrade` turned on because the packages Colab ships clashed with the pinned `transformers` / `peft` /
`accelerate` (an older preinstalled `torchao` breaks the import). Keep these pins when re-running the notebooks.

| Library | Version on Colab | How it got there |
|---|---|---|
| Python | 3.13.16 | Colab runtime |
| `torch` | 2.14.1+cu130 | Colab runtime (not reinstalled) |
| `transformers` | 5.22.2 | `PINNED` |
| `peft` | 0.21.2 | `PINNED` |
| `accelerate` | 1.15.0 | `PINNED` |
| `mlflow` | 3.18.0 | `PINNED` |
| `torchao` | `>=0.16.0` (latest at install time; the exact version is not logged) | `PINNED`, added to avoid the clash |
| `scikit-learn` | 1.9.1 | `pip install --upgrade` (unpinned, matched the tested version) |
| `pandas` | 3.1.1 | `pip install --upgrade` (unpinned, matched the tested version) |

`bitsandbytes` is not installed by the final notebooks (LoRA only). `requirements-train.txt` lists the same pins.

Two requirement files: `requirements.txt` (inference, `test.py`, baselines, reports; **no** `bitsandbytes`) and `requirements-train.txt` (adds training extras for Colab and Kaggle).

All sources, creators and licences are listed in `REFERENCES.md`.

## 3. Repository layout

```text
EcoConsensus/
├── .env.example                  # every variable, empty values
├── .env                          # real keys, git-ignored
├── .gitignore
├── README.md
├── requirements.txt
├── requirements-train.txt
├── inference.py                  # one comment or a file
├── test.py                       # evaluator test runner (reads test/)
├── configs/
│   ├── base.yaml                 # paths, seed, labels, fixed claim, aliases, noise list
│   ├── transformers.yaml         # models, LoRA settings, training settings
│   └── llm.yaml                  # providers, batch size, budgets, prompt version
├── docs/                         # PRD, TRD, APP_FLOW, IMPLEMENTATION_PLAN, STACKING_ENSEMBLE,
│                                 # INFERENCE, TESTING, LLM_PROMPT, REFERENCES
├── data/
│   ├── raw/
│   │   ├── original/             # pol_nli_data/, SemEval_2016_Task6/ (as they are now)
│   │   └── translated/
│   │       ├── Hindi/            # pol_nli_data/, SemEval_2016_Task6/ (pure, code-mix, hinglish)
│   │       └── Bengali/          # pol_nli_data/, SemEval_2016_Task6/ (pure, code-mix, banglish)
│   └── processed/                # frozen splits + split report
├── experiment/                   # finished method-comparison work (notebook, report, SETUP.md)
├── models/
│   ├── original/                 # pointers to base models (names, revisions)
│   └── finetuned/
│       ├── muril_lora/           # adapter + run_info.json
│       ├── indicbert_lora/
│       ├── baselines/            # tfidf_lr.joblib, majority.json
│       └── meta/                 # meta_all4, meta_no_llm_a, meta_no_llm_b, meta_transformers
├── outputs/                      # one folder per component (decision 28), all git-ignored
│   ├── <component>/              # prepare_data, baselines, muril, indicbert, llm, ensemble, evaluate, inference, test
│   │   ├── logs/                 # timestamped .txt run logs
│   │   ├── metrics/              # mlflow.db (one per component), metrics JSON/CSV
│   │   ├── predictions/          # prediction / probability / feature CSVs
│   │   └── reports/              # generated .md reports (timestamped)
│   └── llm/cache/                # LLM answers, one record per comment + daily usage counters
├── papers/
├── src/
│   ├── __init__.py
│   ├── config.py                 # load YAML + .env, resolve paths
│   ├── data_loading.py           # flexible loader, column and label mapping
│   ├── preprocessing.py
│   ├── splitting.py
│   ├── metrics.py
│   ├── baselines.py
│   ├── transformer_model.py      # dataset, model builder, LoRA, train loop
│   ├── llm_api.py                # provider adapters, batching, parsing, cache, retry
│   ├── ensembling.py             # feature builder, meta-model, fallback selection
│   ├── reporting.py              # deterministic report writer, never overwrites
│   ├── pipeline.py               # shared "run components on rows" code for inference/test
│   ├── utils.py                  # seeding, logging, safe file writing
│   ├── evaluate.py               # metrics + report for every model that has outputs
│   └── training/                 # prepare_data, Colab notebooks, LLM batch, features, meta-model (section 9)
├── test/                         # evaluator file goes here; results are written here too
│   ├── result.txt                # latest report (created by test.py)
│   ├── predictions.csv           # latest prediction file (created by test.py)
│   ├── reports/                  # timestamped .md copies
│   └── archive/                  # previous result.txt / predictions.csv
└── tests/                        # unit tests (not the evaluator folder)
```

Rules:
- All logic lives in `src/` so the Mac, `inference.py` and `test.py` run the same code. **Exception:** the two Colab training notebooks (`src/training/train_muril.ipynb`, `train_indicbert.ipynb`) are self-contained single files (decision 19); they read text already cleaned by `src/preprocessing.py` through `prepare_data.py`.
- `inference.py` and `test.py` import nothing from `src/training/` and never import MLflow.

## 4. Data contract

**Internal record:** `id`, `source_id`, `dataset`, `language`, `variant`, `claim`, `comment`, `label`.

- Model inputs are `claim` and `comment` only. `id`, `source_id`, `dataset`, `language`, `variant` and `label` are never model features.
- **Labels:** `favour = 0`, `none = 1`, `against = 2` (matches Pol-NLI and the experiment).
- **Claim mode:** `fixed` by default (the English PS claim for every row, every language, as in the experiment). Other modes (`dataset`, `augmented`) exist in config for experiments with the Pol-NLI `hypothesis` and `augmented_hypothesis` columns.
- **Source mapping:**

| Source | Comment | Claim | Label |
|---|---|---|---|
| Pol-NLI | `premise` | fixed (or `hypothesis`) | `entailment`: 0 favour, 1 none, 2 against (the `review` column is dropped) |
| SemEval | `Tweet` | fixed (the `Target` text is ignored by default) | `Stance`: FAVOR → 0, NONE → 1, AGAINST → 2 |
| Evaluator file | Auto-detected or from `TEST_CONFIG` | Fixed unless `claim_col` is set | Through `label_map` |

- **Variants:** `english/original`, `hindi/pure_hindi`, `hindi/hindi_code_mix`, `hindi/hinglish`, `bengali/pure_bengali`, `bengali/bengali_code_mix`, `bengali/banglish`.

**Flexible loader** (`data_loading.py`), in this order:
1. Read the file. Formats by extension: `.csv`, `.tsv`, `.json`, `.jsonl`, `.xlsx`, `.txt` (one comment per line). Delimiter auto-detected. Encodings tried in order: `utf-8-sig`, `utf-8`, `utf-16`, `cp1252`, `latin-1`.
2. Strip and lowercase column names, then apply explicit overrides, then alias lists:
   - id: `id`, `idx`, `index`, `sno`, `s.no`, `serial`, `comment_id`
   - comment: `comment`, `text`, `tweet`, `premise`, `sentence`, `content`, `body`, `review_text`, `yt_comment`
   - claim: `claim`, `target`, `topic`, `hypothesis`, `statement`
   - label: `label`, `stance`, `class`, `category`, `gold`, `y`
   - language: `language`, `lang`
3. Missing comment column → stop and list the columns found. Missing id → row number. Missing claim → fixed claim. Missing label → prediction-only mode.
4. Normalise labels with the `label_map`. Unknown values are printed (value and count) and the run stops unless `--allow-unknown-labels` is given, in which case those rows are scored as unlabeled.
5. Preprocess, flag empty comments, and compute a derived **script group** per comment from Unicode ranges (Devanagari → `devanagari`, Bengali → `bengali`, otherwise `latin`) for reporting, since the evaluator file has no language column.

## 5. Preprocessing specification

One function, used everywhere.

| Rule | Detail |
|---|---|
| Unicode | NFC (important for Devanagari and Bengali) |
| Emojis | Kept |
| `@handle` | → `@USER` |
| URLs | → `URL` |
| Hashtags | Keep the word, drop `#` (confirm after looking at the data) |
| Noise list (config) | Remove `#SemST` and a leading `RT` |
| Whitespace | Collapse spaces, tabs and newlines |
| Case | Preserved for transformers and LLMs; TF-IDF may lowercase |
| Empty text | Return an empty marker; the loader flags the row |

Do not strip punctuation, stop words or non-Latin characters.

## 6. Splits

- **Pol-NLI train (3,517 source rows):** grouped by `source_id`, stratified by label → `train_base` (about 85%) and `meta` (about 15%).
- **Pol-NLI validation (90 source rows):** `val`.
- **SemEval (565 tweets):** `semeval_test`, never split further.
- Each source row appears in 7 versions, all in the same split.
- Seed fixed in `base.yaml`. Splits are written once to `data/processed/` and reloaded by every script. Re-creating them needs an explicit `--overwrite-splits` flag.
- The split report lists counts and percentages per class, split, language and variant. Scripts stop if a source id appears in two splits.

## 7. Baselines

| Baseline | Specification |
|---|---|
| Majority class | Predict the most frequent `train_base` label |
| TF-IDF + logistic regression | Character n-grams (for example 2–5, `char_wb`) plus word n-grams (1–2), `class_weight="balanced"`, regularisation `C` chosen on `val` only. Input is `claim + separator + comment` |

Both are saved in `models/finetuned/baselines/`.

## 8. Transformers

### 8.1 Models
- `google/muril-base-cased` (Google; Khanuja et al., 2021)
- `ai4bharat/IndicBERTv2-MLM-only` (AI4Bharat; Doddapaneni et al., 2023)

Weights download from the Hugging Face Hub on first use. `models/original/` stores only the model names and revisions.

### 8.2 Input
A sentence pair: `tokenizer(claim, comment, truncation=True, max_length=128)`. In the experiment the 99th percentile was 58 tokens and nothing was truncated at 128. `AutoModelForSequenceClassification` with `num_labels=3`.

### 8.3 Methods compared (experiment) and the choice

| Method | Experiment status | Notes |
|---|---|---|
| Head only | Done | Weakest (MuRIL val Macro-F1 0.360 ± 0.109) |
| Head + last 2 layers | Done | |
| Head + last 4 layers | Done | |
| LoRA | Done, **recommended for both models** | 1,182,723 trainable parameters (about 0.4–0.5%) |
| QLoRA | **Failed** in all 4 runs (`AssertionError`) | Reported as not run; needs `bitsandbytes` on CUDA and is not part of the final pipeline |

Final choice: LoRA for both transformers (rank 16, alpha 32, dropout 0.1, target modules `query` and `value`, learning rate 3e-4), picked by the parsimony rule in the experiment report. The method sits in `transformers.yaml`, so switching is a config change.

### 8.4 Training settings (defaults, all in config)

| Item | Setting |
|---|---|
| Optimizer | AdamW (Loshchilov and Hutter, 2019), weight decay 0.01, none on bias and LayerNorm |
| Learning rate | 3e-4 for LoRA; warmup 10% then linear decay |
| Loss | Cross-entropy with class weights from `train_base` |
| Batch size | 16 (use gradient accumulation if memory is short) |
| Epochs | Up to 10 with early stopping, patience 2, monitored on `val` Macro-F1 |
| Dropout (head) | 0.1 |
| Gradient clipping | 1.0 |
| Mixed precision | fp16 on T4 |
| Seeds | 42 and 43 (as in the experiment); report both |

Terms: *regularisation* is anything that stops memorising (weight decay, dropout, early stopping, fewer trainable parameters); *dropout* randomly switches off neurons during training; *weight decay* pulls weights toward zero; *early stopping* ends training when validation stops improving; *class weights* make mistakes on rare classes cost more.

### 8.5 Output used by the ensemble
For every comment each transformer produces **3 probabilities** (softmax over the 3 logits), in the order favour, none, against. `src/training/train_inference.py` stores them as columns `muril_p0, muril_p1, muril_p2` and `indicbert_p0, indicbert_p1, indicbert_p2`.

## 9. Training folder (`src/training/`)

Each script runs alone and can be resumed. (The user's note says `train_infrence.py`; the file name here is spelled `train_inference.py`.)

| Script | What it does | Where it runs |
|---|---|---|
| `prepare_data.py` | Reads the 7 versions of Pol-NLI train/validation and SemEval, checks row alignment, cleans every comment with `src/preprocessing.py`, deduplicates Pol-NLI train, writes the frozen splits and `data/ecoconsensus_data.zip` | Mac |
| `train_muril.ipynb`, `train_indicbert.ipynb` | Self-contained: unzip the bundle from Drive, LoRA on `train_base`, early stopping on `val`, checkpoint every epoch (resume = Run all), save adapter + `run_info.json`, MLflow, write `probs_<model>_<split>.csv` for `meta`, `val`, `semeval_test`, zip the results | Colab (one account per model, in parallel) |
| `run_llm_batch.py --provider a --split meta` | Fills the LLM cache for a split in batches, honours the daily budget, resumes from cache; also `--check`, `--estimate`, `--pilot`, `--score` | Mac (while Colab trains) |
| `train_inference.py` | Joins the downloaded transformer probabilities with the cached LLM answers into `features_<split>.csv` (no GPU, no requests) | Mac |
| `train_meta.py` | Fits the meta-model and its fallback variants on the `meta` feature table and compares with voting | Mac (CPU) |
| `src/evaluate.py` | Scores every model with outputs (baselines, transformers, LLM cache, votes, stacked) on `val`, `meta`, `semeval_test`; report in `outputs/evaluate/` | Mac (CPU) |

Order: `prepare_data.py` (Mac) → upload the zip to both Drives → run both notebooks at the same time while `run_llm_batch.py` fills the cache on the Mac → unzip `<model>_results.zip` in the project root (fills `models/finetuned/<model>_lora/` and `outputs/<model>/`) → `train_inference.py` → `train_meta.py` → `src/evaluate.py`. Checkpoints go to Drive, never only to the temporary disk. Runbook: `src/training/README.md`.

## 10. LLM component

### 10.1 Interface
One abstract interface, `classify_batch(items) -> list[{id, label, status}]`, one adapter per provider (an OpenAI-compatible adapter covers most providers; others get their own). Providers, model names, batch size and daily budgets are set through `.env` and `llm.yaml`.

### 10.2 Variables in `.env` / `.env.example`

```text
LLM_A_PROVIDER=           # e.g. the adapter name
LLM_A_MODEL=
LLM_A_API_KEY=
LLM_A_BASE_URL=           # optional, for OpenAI-compatible endpoints
LLM_A_DAILY_LIMIT=        # requests per day, used by the budget guard
LLM_A_EXTRA_BODY=         # optional JSON added to every request, e.g. {"reasoning_effort": "low"} for reasoning models
LLM_B_PROVIDER=
LLM_B_MODEL=
LLM_B_API_KEY=
LLM_B_BASE_URL=
LLM_B_DAILY_LIMIT=
LLM_B_EXTRA_BODY=
LLM_BATCH_SIZE=50         # comments per request in training runs and test.py
HF_TOKEN=                 # optional, for Hugging Face downloads
```

### 10.3 Batching and caching
- **Batch size** defaults to 50 and is configurable. `inference.py` always sends 1.
- Requests use the system prompt in `LLM_PROMPT.md`. Every comment in a request has a numeric `id`. The reply is a JSON array of `{id, label}`.
- **Cache:** one JSON Lines record per comment. Key = hash of (provider, model, prompt version, claim, preprocessed comment). The batch size is **not** in the key, so changing it never invalidates cached answers. The cache stores the label, status and timestamp only.
- **Validation of each reply:** parse JSON; check that the returned ids equal the sent ids; check that labels are valid. Items that fail are retried in a batch of half the size, then individually, then marked `unavailable`.
- **Retries:** exponential backoff with a cap, honouring `Retry-After` when present.
- **Budget guard:** counts requests per provider per day from a small usage file. When `LLM_x_DAILY_LIMIT` is reached the script stops cleanly, prints how many comments are still missing, and resumes from cache next time.
- **Token limits:** Hindi and Bengali use many more tokens per word than English, so a 50-comment batch can be several thousand input tokens. If a provider returns a token-per-minute error, the script lowers the batch size automatically for that provider and waits.
- **Output length:** `max_tokens` = `max_tokens_per_comment` × comments + `max_tokens_extra` (`configs/llm.yaml`, capped by `max_tokens_cap`). If a reply is cut off (`finish_reason: length`, typical for reasoning models such as `openai/gpt-oss-20b`, which spend the budget on reasoning) the limit doubles for the rest of the run and the request is resent. `LLM_x_EXTRA_BODY={"reasoning_effort": "low"}` avoids most of that.
- **Error reporting:** every failure is logged with a category that says what happened and on whose side: `AUTH ERROR` / `NOT FOUND` / `BAD REQUEST` / `CONFIG ERROR` (your side: key, model, URL, parameters), `RATE LIMIT (requests per minute)` / `RATE LIMIT (tokens per minute)` / `REQUEST TOO LARGE` (waits, retries, shrinks the batch), `DAILY QUOTA USED UP` (provider) or `DAILY BUDGET USED UP` (your `DAILY_LIMIT`), `PROVIDER ERROR` / `TIMEOUT` / `NETWORK ERROR`, and for HTTP 200 replies without an answer `OUTPUT CUT OFF`, `EMPTY REPLY`, `BLOCKED BY THE PROVIDER'S SAFETY FILTER`, `UNPARSEABLE REPLY` (with the provider's message, `finish_reason`, token usage and remaining-quota headers). A stopped run logs the reason and how many comments are left; the counts per category are in the log and the `llm_batch` report.
- **Few-shot examples** (if few-shot wins on `val`) come from `train_base` only.

## 11. Ensembling summary

Features per comment (12 numbers): MuRIL 3 probabilities, IndicBERT 3 probabilities, LLM-A one-hot (3), LLM-B one-hot (3). The meta-model is a regularised multinomial logistic regression with balanced class weights, trained on the `meta` slice. Full design, budget table and schedule: `STACKING_ENSEMBLE.md`.

## 12. Reporting module (`src/reporting.py`)

- Deterministic Python only; no AI call anywhere.
- All files are written through one `safe_write()` that never overwrites: it adds a timestamp, and if the name still exists it adds a counter (`_2`, `_3`).
- `result.txt` and `predictions.csv` in `test/` are "latest" files: the previous one is moved to `test/archive/` (named with its original timestamp) before the new one is written.
- Report sections: header, data summary, system comparison, per-class results, confusion matrices, results by script group, LLM usage, fallback usage, error analysis, explainability, environment, configuration, warnings, references pointer.
- The experiment report format (`experiment_report_<timestamp>.md`) stays as it is.

## 13. `test.py`

Behaviour in short (details in `TESTING.md`):
- Lives in the repo root, reads `test/`, writes into `test/` and `outputs/`.
- A `TEST_CONFIG` block at the top holds: `data_file`, `id_col`, `text_col`, `label_col`, `claim_col`, `language_col`, `label_map`, `delimiter`, `encoding`, `fixed_claim`, `use_llm`, `llm_budget`. Command-line flags override it. `--inspect` prints columns, sample rows and distinct label values without running any model.
- Runs every component, picks the **best single transformer** and **best single LLM** by Macro-F1 on the same file (shown in the report), and applies the stacked ensemble with automatic fallback per row.
- LLM calls are batched (`LLM_BATCH_SIZE`) and cached.
- Exit codes: `0` success, `2` configuration or column error, `3` data error, `4` missing model files, `5` all LLMs unavailable and `--require-llm` set.

## 14. `inference.py`

Behaviour in short (details in `INFERENCE.md`): one comment by default, or a file; one request per LLM; component selection with `--mode`; same loader, preprocessing and meta-models as `test.py`.

## 15. Experiment tracking (MLflow)

- Store: one SQLite file per component at `outputs/<component>/metrics/mlflow.db` (the experiment used `experiment_outputs/mlflow.db`); the Colab runs arrive as `outputs/<model>/metrics/mlflow.db` inside `<model>_results.zip`. Open one with `mlflow ui --backend-store-uri sqlite:///outputs/<component>/metrics/mlflow.db`.
- Experiments: `finetune_method_comparison` (done), `finetune_final`, `baselines`, `llm_prompts`, `stacking`, `evaluation`, `final_test`.
- Per run: parameters, per-epoch metrics, final metrics, trainable parameters, training time, library versions, `run_info.json`.
- Run naming: `<component>__<method>__seed<k>__<hash>`.
- Model selection uses `val` metrics. Test metrics are only logged in `final_test`.

## 16. Metrics (`metrics.py`)

Macro-F1 (primary, labels fixed to the 3 classes so a missing class scores 0), accuracy, weighted-F1, per-class precision, recall, F1 and support, confusion matrix, MCC, Cohen's kappa, all repeated per script group, and a bootstrap 95% interval for Macro-F1 (1,000 resamples, grouped by `source_id` when ids are available because the 7 versions of a comment are not independent).

## 17. Logging

One `.txt` file per run in `outputs/<component>/logs/`: `<run_type>_<YYYYMMDD_HHMMSS>.txt`. Contents: timestamp, command line, config summary and versions, input pointer and row counts, preprocessing summary, per-comment outputs (when small), warnings (parse errors, API failures, dropped rows, fallbacks), metrics. Python `logging` with a file handler and console output. Keys are never logged.

## 18. Testing

Run: `python -m pytest -v tests` (`-rA` adds a pass/fail/skip line per test at the end). GitHub Actions runs the same
suite on every push to every branch (`.github/workflows/tests.yml`, Python 3.13, `requirements.txt` without
`torch`/`transformers`/`peft`). `data/` is not in git, so the tests that read the frozen splits are skipped there.

On the Mac, small and fast (`tests/`):
- Preprocessing (emoji kept, `@USER`, `URL`, `#SemST`, Hindi and Bengali strings, empty text)
- Loader on many fake file formats, column names and label spellings
- Splits (class ratios within tolerance, no source id in two splits)
- LLM reply parser on messy replies, id mismatches and missing items; cache key ignores batch size
- Budget guard stops and resumes
- Metrics on hand-made predictions
- Reporting never overwrites and archives `result.txt`
- Meta-model fallback selection per row
- End-to-end smoke test of `test.py` and `inference.py` on a handful of rows with a stubbed LLM

## 19. Code style, security, reproducibility

- **Comments:** a short header per file saying what it is for, and comments only where the reason is not obvious. No line-by-line narration.
- Seeds for Python, NumPy and PyTorch are set in one `set_seed()`; library versions are recorded in every run.
- Keys only in `.env` (or Colab and Kaggle secrets). `.gitignore` covers `.env`, `outputs/` (cache, logs, predictions, MLflow), `mlruns/`, `mlflow.db`, large weights and `test/*` data files (not the result files' folder structure).
- LoRA adapters (about 5 MB each) may be committed; full base-model weights never are.
- `joblib` files (TF-IDF, meta-models) are scikit-learn pickles. Only load files you created yourself.

## 20. Decision log

| # | Decision | Reason |
|---|---|---|
| 1 | Language is not a model input | Test data will not contain it |
| 2 | Sentence-pair input (claim, comment) | How transformers encode two texts |
| 3 | Emojis kept, handles → `@USER` | Keeps stance signal |
| 4 | Stratified + grouped splits | Equal class ratios, no translation leakage |
| 5 | Class-weighted loss | Class imbalance, Macro-F1 primary |
| 6 | Provider-agnostic LLM layer | Free tiers are unreliable |
| 7 | Hold-out slice for the meta-model | Fits free-tier request limits and needs no extra training runs |
| 8 | Training on Colab or Kaggle; code, `test.py` and inference on the Mac | 8 GB RAM |
| 9 | Pol-NLI trains; SemEval tests only | Quality of labels; SemEval is the own test |
| 10 | Label order favour 0, none 1, against 2 | Matches Pol-NLI and the experiment |
| 11 | Full fine-tuning removed | Plan change; four methods compared |
| 12 | LoRA for both transformers | Parsimony rule on the experiment results |
| 13 | QLoRA recorded as not run | Failed in all four runs |
| 14 | Batch size 50 for LLM calls, configurable | Fewer requests per day; cache is per comment |
| 15 | `test.py` in repo root; `result.txt` latest + archive | User's decision |
| 16 | Python 3.13 or newer | User preference; experiment used 3.13.16 |
| 17 | Bagheli translations discarded; Bengali used | Translation mistake corrected |
| 18 | (Not executed, see 22.) Experiment re-run planned: LoRA r / alpha / target modules / learning rate searched on the experiment `val` (stage 1), QLoRA fixed (pooler kept full precision) and reusing LoRA's pick; the fixed values in 8.3 and 8.4 are replaced by the new report's picks | The old fixed values were never tuned; QLoRA had failed because the 4-bit pooler was wrapped by `modules_to_save` |
| 19 | MuRIL and IndicBERT trained **in parallel on two Colab accounts**, each from one self-contained notebook (`src/training/train_muril.ipynb`, `train_indicbert.ipynb`) that imports nothing from `src/`; replaces `train_transformer.py` + thin notebooks | User decision: saves a day of GPU time; a single file per account is simpler to run |
| 20 | Text is cleaned once, in `src/training/prepare_data.py` (through `src/preprocessing.py`); the split CSVs and the Colab bundle hold the cleaned `comment`; the notebooks do no cleaning | One cleaning function, no copy of it in the notebooks |
| 21 | Pol-NLI train: 1,417 rows whose English text repeats another row are removed (some sentences occur ~100 times, same label and translation), and 64 train groups identical to a `val`/SemEval comment are removed (78 of the 90 `val` comments also occurred in train). Result: `train_base` 1,424 and `meta` 252 source rows (× 7). `--keep-duplicates` / `--keep-cross-duplicates` restore the old behaviour | Duplicates dominated training; overlap with `val` made early stopping optimistic |
| 22 | LoRA values from the 2026-11-08 report (r 16, alpha 32, dropout 0.1, query+value, lr 3e-4); the re-run of decision 18 was not executed | No time for the re-run |
| 23 | One seed (42) per transformer | Plan cut-line; GPU time |
| 24 | Each notebook writes the transformer's probabilities for `meta`, `val`, `semeval_test`; `train_inference.py` only joins them with the LLM cache on the Mac | Adapters live on different Drives; no account needs both |
| 25 | LLM prompt `v3` in `prompts/stance_system_prompt.md`, built from `LLM_PROMPT.md` (labels FAVOUR / AGAINST / NONE → 0 / 2 / 1); the code reads the prompt from that file | `v2` file was dropped; one source of truth for the code |
| 26 | Both LLM components use one OpenAI-compatible adapter; provider, model and URL only in `.env` | User decision; covers most free-tier providers |
| 27 | LLM cache fill and meta-model training run on the Mac (Colab only if memory runs out) | Meta-model is a small logistic regression |
| 28 | Outputs are grouped by component: `outputs/<component>/{logs,metrics,predictions,reports}/` for `prepare_data`, `baselines`, `muril`, `indicbert`, `llm`, `ensemble`, `evaluate`, `inference`, `test`; one MLflow DB per component; LLM cache in `outputs/llm/cache/`. The notebooks' results zip uses the same layout | User decision (2026-11-11): each model's files in one place |
| 29 | `src/evaluate.py` scores every available system (baselines, transformers, LLMs from cache, soft/majority vote, stacked) on `val`, `meta`, `semeval_test` and writes a report; `stacked` is not scored on `meta` (in-sample). No requests, no tuning | User request: one command for the metrics of every model |
