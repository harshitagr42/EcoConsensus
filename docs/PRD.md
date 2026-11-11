# Product Requirements Document (PRD)

**Project:** EcoConsensus — Stance Detection in Media on Global Warming for Indic Languages (CSET 346, NLP)
**Version:** v2 (replaces the v1 draft; updated for the professor's PS, the evaluator test run, the new datasets and the experiment results)
**Timeline:** about 5 days (Mon 9 Nov – Fri 13 Nov 2026). The PS lists all dates as "to be announced", so shift the plan if the real dates differ.
**Companions:** `TRD.md`, `APP_FLOW.md`, `IMPLEMENTATION_PLAN.md`, `STACKING_ENSEMBLE.md`, `../INFERENCE.md` (repo root), `LLM_PROMPT.md`, `REFERENCES.md`

If a feature is not in section 4, it is not in the build.

---

## 1. What this project is

Given a comment and one fixed claim, the system decides whether the comment agrees with the claim, disagrees with it, or says nothing clear about it.

**Fixed claim (from the PS):** "Climate change and global warming is a serious concern."

| Label | Stored as | Meaning (PS wording) |
|---|---|---|
| Favour | `0` | The comment agrees that climate change or global warming is a serious concern |
| None | `1` | Neutral, unrelated, or does not clearly express a stance |
| Against | `2` | The comment disagrees with the claim |

This order (0 = favour, 1 = none, 2 = against) matches the Pol-NLI `entailment` column (0 entailment, 1 neutral, 2 contradiction) and the experiment notebook. It replaces the v1 draft, which used a different order.

**Languages:** Hindi and Bengali (the PS languages), each in three text forms: pure script, code-mixed, and romanized (Hinglish, Banglish). English is kept as a seventh version of every comment, because the source datasets are English.

**Method:** two local transformers (MuRIL, IndicBERT v2) fine-tuned with LoRA, and two LLMs called through APIs, combined by a **stacked meta-model**. It is compared against two baselines (majority class, TF-IDF + logistic regression).

**No UI.** The project is used through `inference.py`: one comment, or a file (by default the evaluator's test file in `data/test/`).

## 2. Who uses it

| User | What they are trying to do |
|---|---|
| **Builder (you)** | Train, compare, pick methods from evidence, and write the 8-page report |
| **Evaluator / professor** | Drop a hidden test file into `data/test/`, run `inference.py`, and read `outputs/inference/result.txt` and the prediction file |
| **Anyone cloning the repo** | Understand and reproduce the work from the README and `docs/` |

## 3. Goals and non-goals

**Goals**
1. Classify stance (3 classes) for Hindi and Bengali comments toward the fixed claim.
2. Beat the TF-IDF baseline on **Macro-F1** and report honestly where the ensemble beats (or does not beat) the best single transformer and the best single LLM.
3. Work on an evaluator test file whose columns and label spellings are unknown, with a configuration that can be changed on the spot.
4. Produce every report by script only, never overwriting an older report.
5. Keep the repo organised, commented where needed, documented, and every external source cited.

**Non-goals (not in this build)**
- No web or desktop UI.
- No claims other than the fixed one (a claim override exists, but the models are trained for this claim only).
- No languages beyond Hindi, Bengali and the English source text.
- No AI-written reports. Reports are produced by deterministic scripts.
- No full fine-tuning of all layers (dropped after the plan change).
- No large hyperparameter search.

## 4. Features

Each feature lists who it is for, what it does, and what counts as done.

### F1. Flexible data loading
- **For:** builder, evaluator
- **What:** One loader reads CSV, TSV, JSON, JSON Lines, XLSX and plain text. It maps whatever columns it finds to the internal fields `id`, `comment`, `claim`, `label`, `language`.
- **Done when:**
  - Auto-detection works from a list of known column names (for example `Tweet`, `text`, `premise` → comment; `Stance`, `label` → label; `ID`, `id` → id).
  - Explicit overrides in the `CONFIG` block of `inference.py` or by command-line flag always win over auto-detection.
  - A missing comment column stops the run with a message that lists the columns found.
  - A missing id column falls back to the 1-based row number. A missing claim column uses the fixed claim. A missing label column switches to prediction-only mode.
  - Label spellings (`FAVOR`, `Favour`, `AGAINST`, `None`, `NONE`, `Neutral`, and so on) are normalised through an editable map. Unknown values are listed and never guessed. Numeric labels are never mapped unless the map says so.

### F2. Datasets and translation
- **For:** builder
- **What:** Fixes what each dataset is used for.

| Dataset | Rows | Quality | Used for |
|---|---|---|---|
| Pol-NLI subset (Burnham et al., 2024) | 3,517 train + 90 validation (English) | Lower | Training the transformers and baselines; the hold-out slice that trains the meta-model |
| SemEval-2016 Task 6, climate target | 565 tweets | High | **Own test set and experiments only**. Never used for training or for the meta-model |
| Evaluator's hidden file | Unknown | Unknown | Final test through `inference.py` only |

  Each English comment also exists in 6 translated versions (Hindi: pure, code-mix, Hinglish; Bengali: pure, code-mix, Banglish), made once with an LLM. That is 7 versions per comment.
- **Done when:**
  - Translated files live in `data/raw/translated/Hindi/` and `data/raw/translated/Bengali/`, grouped by dataset and variant.
  - The earlier Bagheli translations are removed from the data tree and not used anywhere.
  - Every translated row keeps its source row id, so all 7 versions of a comment always land in the same split.
  - The README states that Hindi and Bengali text is machine-translated, names the LLM used, and describes the limitation.
  - Only final clean translated files are kept in the repo (no intermediate files or translation scripts).

### F3. Text preprocessing
- **For:** builder, evaluator (applied identically everywhere)
- **What:** Cleans social-media text without removing stance signals.
- **Done when:**
  - Emojis are kept. `@handle` becomes `@USER`. URLs become `URL`. Unicode is normalised to NFC.
  - Repeated whitespace is collapsed, and empty or whitespace-only comments are flagged, not crashed on.
  - A dataset-specific noise list is removed (for example the `#SemST` tag that ends every SemEval tweet, and a leading `RT`). The list lives in config.
  - Case is preserved for transformers and LLMs. The TF-IDF baseline may lowercase.
  - One function is used for training, evaluation, LLM prompts and `inference.py`.
  - The preprocessing choices are written up in the report as the "novel preprocessing" bonus item (script-aware handling, romanized text, emoji kept).

### F4. Stratified, leak-free splitting
- **For:** builder
- **What:** Builds the frozen splits.

| Split | Source | Used for |
|---|---|---|
| `train_base` | About 85% of Pol-NLI train | Training MuRIL, IndicBERT and the TF-IDF baseline; few-shot examples for the LLM prompt |
| `meta` | About 15% of Pol-NLI train | Training the stacking meta-model only (the transformers never see it) |
| `val` | Pol-NLI validation (90 comments, 7 versions each) | Early stopping, TF-IDF tuning, LLM prompt choice |
| `semeval_test` | All 565 SemEval tweets | Own final test |

- **Done when:**
  - The split is stratified by label and grouped by source id.
  - It is created once with a fixed seed, saved in `data/processed/`, and reused by every script.
  - A split report (counts and percentages per class, split and language) is saved. The script stops if any source id appears in two splits.

### F5. Baselines
- **For:** builder, evaluator
- **What:** Majority class (the floor) and TF-IDF (character + word n-grams) with logistic regression (the one to beat).
- **Done when:** both are trained on `train_base`, tuned on `val` only, saved in `models/finetuned/baselines/`, and scored by `inference.py` next to every other system.

### F6. Transformer fine-tuning methods
- **For:** builder
- **What:** Fine-tunes MuRIL (`google/muril-base-cased`) and IndicBERT v2 (`ai4bharat/IndicBERTv2-MLM-only`) on sentence pairs (claim, comment). **Four methods** were compared in the experiment: head only, head + last N layers (N = 2 and 4), LoRA, QLoRA. Full fine-tuning of all layers is removed.
- **Experiment result (report of 2026-11-08, SemEval subset, 2 seeds):**
  - LoRA was recommended for both models by the parsimony rule (MuRIL val Macro-F1 0.607, IndicBERT 0.645).
  - QLoRA failed in all four runs (`AssertionError`) and is reported as **not run**, never scored as zero.
  - Known weak spot: the `against` class is rare, so its F1 (and therefore Macro-F1) is unstable. IndicBERT/LoRA scored 0.044 F1 on `against` in the held-out check.
- **Done when:**
  - The final models are LoRA for both transformers, trained independently on Colab or Kaggle (one at a time, because of the ~15 GB GPU memory limit).
  - Each run uses class-weighted loss, AdamW, dropout, early stopping and checkpoints, and is logged to MLflow.
  - Language is never a model input.
  - The method choice can be changed in config without touching code.

### F7. LLM stance classifier (provider-agnostic, batched, cached)
- **For:** builder, evaluator
- **What:** Two LLMs from two different providers classify comments. The code talks to one common interface, so providers change by editing config and `.env`.
- **Done when:**
  - Provider, model name and key come from `.env` and config, never from code.
  - Training runs and `inference.py` file runs send **batches of comments per request** (default 50, configurable). `inference.py -t` sends one comment.
  - Every answer is cached on disk **per comment**, so reruns and different batch sizes never spend quota twice.
  - The system prompt in `LLM_PROMPT.md` is versioned, and its version is stored with each cached answer.
  - Unparseable or missing answers are retried in smaller batches, then recorded as `unavailable`, never dropped silently.
  - A daily request budget can be set. When it is reached the run stops cleanly and resumes from cache the next day.

### F8. Stacking ensemble
- **For:** builder, evaluator
- **What:** A small meta-model takes the outputs of MuRIL, IndicBERT, LLM-A and LLM-B and produces the final label. Full design in `STACKING_ENSEMBLE.md`.
- **Done when:**
  - The meta-model is trained on the `meta` slice only: rows the transformers never trained on.
  - It is compared with simple voting as a sanity check.
  - Fallback meta-models exist for missing components (no LLM-A, no LLM-B, transformers only). The output always says which one was used.
  - `semeval_test` and the evaluator file are never used to fit anything.

### F9. Experiment tracking and the experiment report
- **For:** builder
- **What:** MLflow records every training run. The experiment notebook (finished) produced the method-comparison report.
- **Done when:** every final choice in the technical report can be traced to a logged run, and the experiment report is stored under `experiment/experiment_outputs/reports/`.

### F10. Evaluation and report generation (script only)
- **For:** builder, evaluator
- **What:** One shared metrics module scores every system. One reporting module turns the numbers into Markdown and text reports with no AI involved.
- **Done when each report shows:**
  - **Macro-F1 (primary)**, accuracy, weighted-F1, per-class precision, recall and F1, confusion matrix, MCC and Cohen's kappa.
  - Results by script group (Devanagari, Bengali, Latin) and, when available, by dataset variant.
  - Error analysis: the most common confusions and a list of misclassified examples.
  - Explainability: per-component votes for each comment and the meta-model's learned weights.
  - Environment versions, configuration used, row counts, dropped rows and warnings.
  - **A new report never overwrites an old one.** Timestamped files are never reused, and `result.txt` is archived before it is replaced.

### F11. Checkpointing and resume
- **For:** builder (Colab and Kaggle sessions disconnect)
- **Done when:** a checkpoint is saved every epoch to persistent storage (Google Drive or Kaggle output), the best checkpoint by validation Macro-F1 is kept separately, and rerunning the same run name resumes from the last checkpoint.

### F12. `inference.py`
- **For:** evaluator, builder
- **What:** Classifies one comment (default) or a file. Full usage in `INFERENCE.md` (repo root).
- **Done when:** it prints the final label and each component's vote, writes a timestamped log, needs no training code or MLflow, sends exactly one comment to each LLM, and gives a clear message for every failure listed in `APP_FLOW.md`.

### F13. `inference.py` on a file (evaluator test run)
- **For:** evaluator, builder
- **What:** Runs everything on the file in `data/test/` and writes the full comparison. Full guide in `INFERENCE.md` (repo root).
- **Done when:**
  - It finds the single data file in `data/test/`, or takes `--file`.
  - A `CONFIG` block at the top of `inference.py` (and matching command-line flags) lets you set column names, label spellings, delimiter and claim handling in a minute, with `--inspect` showing the file's columns, sample rows and distinct label values first.
  - It compares: majority baseline, TF-IDF baseline, MuRIL, IndicBERT, the **best single transformer**, LLM-A, LLM-B, the **best single LLM**, and **the stacked ensemble**.
  - It writes `outputs/inference/result.txt` (latest) plus a timestamped `.md` copy in `outputs/inference/reports/`; the previous `result.txt` is moved to `outputs/inference/archive/` first.
  - It writes the **prediction file** the PS requires (`ID`, `Predicted_Label`) to `outputs/inference/predictions.csv` (latest, previous one archived). Logs, detailed predictions and metrics go to `outputs/inference/`, the LLM cache to `outputs/llm/cache/`.
  - It still produces predictions and a prediction-only report when the file has no labels.

### F14. Logging and outputs
- **Done when:** every run writes a timestamped `.txt` log in `outputs/<component>/logs/` with the command, input pointer, preprocessing summary, config, per-component outputs, fallbacks and warnings.

### F15. Model storage layout
- **Done when:** `models/original/` holds pointers to the untouched base models, and `models/finetuned/` holds one folder per trained component (LoRA adapters, TF-IDF model, meta-models) with a `run_info.json` (base model, method, config, split id, metrics, MLflow run id). LoRA adapters are small enough for Git (roughly 5 MB each).

### F16. Secrets
- **Done when:** API keys live only in `.env` (git-ignored). `.env.example` lists every variable with empty values and a one-line comment. No key appears in code, logs or cache files.

### F17. Training folder
- **Done when:** `src/training/` holds the scripts for Colab and Kaggle: train one transformer, fill the LLM cache in batches, build the meta-model features, train the meta-model. Each script runs on its own, so a session limit never forces a full restart.

### F18. Documentation and citations
- **Done when:** the README and `docs/` cover setup on macOS, Linux and Windows, data layout, how to run each stage, how to reproduce results, limitations, and cite every library, model, dataset and LLM used (`REFERENCES.md`). Code has short comments where the reason is not obvious, and is not over-commented.

## 5. Mapping to the professor's PS

| PS requirement | Where it is met |
|---|---|
| Preprocessing (cleaning, normalisation, tokenisation, features) | F3, F5, F6 |
| Explore TF-IDF, IndicBERT, MuRIL and other approaches | F5, F6, F9 |
| Explore LLM-based models | F7 |
| Train and evaluate a stance model | F6, F8, F10 |
| Source code on GitHub with a clear README | F18 |
| Technical report (max 8 pages) | Written by you from the generated reports; not generated by AI scripts |
| Prediction file with ID and Predicted Label | F13 |
| Macro-F1, accuracy, precision, recall, confusion matrix | F10 |
| Bonus: novel preprocessing | F3 |
| Bonus: data augmentation | F2 (translation into 6 variants) and optional claim paraphrases from the Pol-NLI `augmented_hypothesis` column |
| Bonus: explainable AI | F10 (component votes, meta-model weights) |
| Bonus: ensemble models | F8 |
| Bonus: comprehensive error analysis | F10 |
| Bonus: well-documented code | F18 |
| Mention external libraries, models, LLMs | `REFERENCES.md` |

## 6. Success criteria

| Criterion | Target |
|---|---|
| Ensemble Macro-F1 vs TF-IDF | Higher on `semeval_test` |
| Majority baseline | Reported, to show why Macro-F1 matters |
| Comparison complete | One table with majority, TF-IDF, both transformers, both LLMs, the best singles and the ensemble |
| Unknown test file | `inference.py` runs on a file with unknown columns after at most a one-line config change |
| Reports | Script-generated, nothing overwritten |
| Reproducible | A fresh clone plus the README reproduces the baselines on a laptop |

If the ensemble does not beat the best single model, that is a valid result, and the report states it with a likely reason.

## 7. Constraints

- **Hardware:** MacBook Pro M3, 8 GB RAM, for code, tests, baselines and inference. Transformer training runs on Colab or Kaggle (T4, about 14.7–15.8 GB), one model at a time.
- **Cost:** no paid APIs. Gemini (unreliable on the free tier) and OpenRouter (too few requests per day) are excluded. Daily request limits drive the batching, caching and hold-out design.
- **Software:** Python 3.13 or newer, the latest Hugging Face `transformers` (see `TRD.md` for the tested versions).
- **Time:** about 5 days.

## 8. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| `against` is rare, so Macro-F1 swings on a handful of comments | Unstable headline score | Class weights in every model and the meta-model; report per-class F1; bootstrap interval on `semeval_test` |
| Training data (political NLI) differs from YouTube comments | Lower real-world scores | Report honestly; test on SemEval and the evaluator file; discuss in error analysis |
| Machine-translated Hindi/Bengali is noisy | Per-language scores partly reflect translation quality | Spot-check samples; report by variant |
| LLM free-tier limits or downtime | Incomplete ensemble | Batching, per-comment cache, daily budget, fallback meta-models |
| Large batches (50+) skip or misalign items | Wrong labels from the LLM | Ids in every item, count check, retry in smaller batches, a small pilot to pick the batch size |
| Unknown evaluator format | `inference.py` fails at the demo | Auto-detection, `CONFIG`, `--inspect`, flags, clear errors |
| QLoRA failed in the experiment | One method not compared | Report as not run; the final choice is LoRA from the remaining three methods |
| SemEval subset was used to choose methods | Slightly optimistic test scores | State it in the report; final models are trained only on Pol-NLI |
| Pol-NLI page states no license | Reuse question | Cite the authors; check with the instructor if the code is published publicly |

## 9. Assumptions

1. The evaluator's file contains at least a comment text column; labels may or may not be present.
2. The evaluator's comments are in Hindi, Bengali or romanized text, but `inference.py` accepts anything.
3. Every one of the 3,517 + 90 Pol-NLI rows has all 7 versions available (the request estimates depend on this).

## 10. Open items

- Which two LLM providers and models (kept in `.env`, so this does not block code).
- Which LLM produced the translations (the README must name it).
- Whether the technical report must follow a course template.
- Confirm the exact `#SemST` and `RT` handling after looking at the cleaned data.
