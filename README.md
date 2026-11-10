# EcoConsensus

Stance detection on global warming for Hindi and Bengali comments (CSET 346 NLP project). Given the claim **"Climate change and global warming is a serious concern."** and a comment, the system predicts **favour** (0), **neutral** (1) or **against** (2). The final system is a stacked ensemble of two fine-tuned transformers (MuRIL, IndicBERT v2) and two LLMs, compared against a majority-class baseline and a TF-IDF + logistic regression baseline. The primary metric is Macro-F1.

## Status

| Part | State |
|---|---|
| Planning documents (`docs/`) | Written |
| Experiment (`experiment/prepare_data.py`, `experiment/experiment.ipynb`) | Done on Colab; LoRA chosen for both models |
| Data, baselines, LLM layer, stacking scripts (`src/`, `src/training/`) | Written (Days 1–3); runbook in `src/training/README.md` |
| MuRIL and IndicBERT (LoRA) | Trained on Colab (2026-11-10); scored with `python -m src.evaluate` |
| `test.py`, `inference.py` | Not written yet (Day 4) |

## Where to start

| I want to... | Read |
|---|---|
| Run the experiments on Colab or Kaggle | `experiment/SETUP.md` |
| Understand the requirements | `docs/PRD.md`, `docs/TRD.md` |
| See how everything connects | `docs/APP_FLOW.md` |
| See the plan and day-by-day sign-offs | `docs/IMPLEMENTATION_PLAN.md` |
| See how the ensemble is built and trained | `docs/STACKING.md` |
| Run `inference.py` or `test.py` on macOS, Linux or Windows | `docs/INFERENCE.md` |
| Use the LLM prompt | `prompts/stance_system_prompt.md` |

## Repository layout

See `docs/TRD.md` section 3. Short version: `data/` (raw, translated, processed), `experiment/` (notebook, setup, generated reports), `src/` (with `src/training/` for data preparation and the Colab notebooks), `models/original` and `models/finetuned`, `outputs/`, `test/` (evaluator data and results), `tests/` (unit tests), `docs/`, `prompts/`.

`outputs/` has one folder per component, each with the same four subfolders:

```text
outputs/
├── prepare_data/   baselines/   muril/   indicbert/   llm/   ensemble/   evaluate/   inference/   test/
│   ├── logs/          # timestamped .txt run logs
│   ├── metrics/       # mlflow.db (one per component), metrics JSON/CSV
│   ├── predictions/   # prediction, probability and feature CSVs
│   └── reports/       # generated Markdown reports
└── llm/cache/         # LLM answers (append-only) and daily usage counters
```

Open one component's MLflow runs with `mlflow ui --backend-store-uri sqlite:///outputs/<component>/metrics/mlflow.db`.

## Library versions

Local (Mac / Linux / Windows): `pip install -r requirements.txt` (Python 3.13). The Colab training notebooks install
their own pins in cell 2: `transformers==5.22.2`, `peft==0.21.2`, `accelerate==1.15.0`, `mlflow==3.18.0` and
`torchao>=0.16.0`, with `pip install --upgrade`, on top of Colab's `torch 2.14.1+cu130` (Python 3.13.16, T4). The
`torchao` pin and `--upgrade` avoid a clash with Colab's preinstalled packages. Full table: `docs/TRD.md` section 2.

## Running the tests

```bash
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -v tests              # one line per test: PASSED / FAILED / SKIPPED
python -m pytest -v -rA tests          # plus a summary of every outcome at the end
python -m pytest -rfE tests            # only list what failed or errored
python -m pytest tests/test_metrics.py -v              # one file
python -m pytest tests -k "leakage" -v                 # tests whose name matches
python -m pytest tests -v --junitxml=pytest-report.xml # machine-readable report
```

Tests in `tests/test_splitting.py` that read the frozen splits are skipped until `python -m src.training.prepare_data`
has been run. GitHub Actions (`.github/workflows/tests.yml`) runs the suite on every push to every branch; the result
per test is on the run's summary page and in the `pytest-report` artifact.

## Configuration and secrets

Copy `.env.example` to `.env` and fill in the two LLM providers' details. `.env` must be in `.gitignore`; keys are never written in code or logs.

## Workflow

1. Experiment (Colab): `prepare_data.py` builds a small SemEval subset as `experiment_data.zip`; upload it to Drive; the notebook compares head, head + last layers, LoRA and QLoRA for both models and writes results, MLflow and reports to Drive; read the report; choose one method per model.
2. `python -m src.training.prepare_data` builds the cleaned splits and `data/ecoconsensus_data.zip`; upload it to Drive.
3. Final training on Colab: MuRIL and IndicBERT **in parallel on two accounts** (`src/training/train_<model>.ipynb`); unzip each `<model>_results.zip` in the project root. Meanwhile `run_llm_batch.py` fills the LLM cache on the Mac.
4. `train_inference.py` joins both transformers' probabilities and both LLMs' cached answers into feature tables; `train_meta.py` trains the meta-model.
5. `python -m src.evaluate` scores every model and writes a report to `outputs/evaluate/`.
6. `test.py` evaluates a dataset placed in `test/`; `inference.py` classifies single comments (Day 4).

## Data

| Dataset | Use |
|---|---|
| Pol-NLI (lower quality, political NLI sentences; about 3.5k train and 90 validation after processing) | Training |
| SemEval-2016 Task 6 climate rows (564 tweets, higher quality) | Experiments (small subset), meta-model training, own testing |
| Evaluator's hidden test set | Final test (format unknown) |

Both datasets are machine-translated into Hindi and Bengali (pure script, code-mixed, romanised) with LLMs, so reported scores for those languages describe translated text. The assignment says the course datasets are shared by email; as they will not be provided, the data above was assembled independently.

## Reports

All reports are Markdown (and `.txt` from `test.py`), written only by script code from saved results, with timestamped names. A report is never overwritten; a name clash adds a numeric suffix.

## Limitations

- Training text is not YouTube comments; translated text differs from natural Hindi and Bengali writing.
- SemEval has few `against` examples, so Macro-F1 is sensitive to small changes.
- Free LLM tiers limit daily requests; results depend on the cache being complete.
- Method comparison used a small number of epochs and seeds.

## References

Bibliographic details marked (checked) were confirmed against the model or dataset card; the others are from memory. **Check each entry before putting it in the report.**

**Models**
- MuRIL: Khanuja, S., Bansal, D., Mehtani, S., et al. (2021). *MuRIL: Multilingual Representations for Indian Languages.* arXiv:2103.10730. Model `google/muril-base-cased`, Google.
- IndicBERT v2: Doddapaneni, S., Aralikatte, R., Ramesh, G., et al. (2023). *Towards Leaving No Indic Language Behind: Building Monolingual Corpora, Benchmark and Models for Indic Languages.* ACL 2023. Model `ai4bharat/IndicBERTv2-MLM-only`, AI4Bharat, MIT licence, 278M parameters (checked).

**Datasets**
- Pol-NLI and Political DEBATE: Burnham, M., Kahn, K., Wang, R. Y., Peng, R. X. (2024). *Political DEBATE: Efficient Zero-shot and Few-shot Classifiers for Political Text.* arXiv:2409.02078. Dataset `mlburnham/Pol_NLI` on Hugging Face (checked). It is assembled from several public sources; see its dataset card for them.
- SemEval-2016 Task 6: Mohammad, S., Kiritchenko, S., Sobhani, P., Zhu, X., Cherry, C. (2016). *SemEval-2016 Task 6: Detecting Stance in Tweets.* Proceedings of SemEval-2016.

**Methods**
- LoRA: Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.
- QLoRA: Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* arXiv:2305.14314.
- AdamW: Loshchilov, I., Hutter, F. (2019). *Decoupled Weight Decay Regularization.* ICLR 2019. arXiv:1711.05101.

**Libraries and platforms**
- Hugging Face Transformers: Wolf, T., et al. (2020). *Transformers: State-of-the-Art Natural Language Processing.* EMNLP 2020 System Demonstrations. Hugging Face, Inc.
- PEFT: Mangrulkar, S., Gugger, S., Debut, L., Belkada, Y., Paul, S. Hugging Face, Inc. github.com/huggingface/peft
- bitsandbytes: Tim Dettmers and contributors. github.com/bitsandbytes-foundation/bitsandbytes
- PyTorch: Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS 2019.
- scikit-learn: Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12.
- pandas: McKinney, W. (2010). *Data Structures for Statistical Computing in Python.* SciPy 2010; The pandas development team.
- MLflow: Zaharia, M., et al. (2018). *Accelerating the Machine Learning Lifecycle with MLflow.* IEEE Data Engineering Bulletin 41(4); Databricks and the MLflow contributors.
- python-dotenv: Saurabh Kumar and contributors.
- Google Colaboratory (Google) and Kaggle Notebooks (Kaggle, Google): training platforms.

**LLMs used (the assignment requires naming every LLM)**

| Purpose | Provider and company | Model and version | Date used |
|---|---|---|---|
| Translation to Hindi and Bengali | _fill in_ | _fill in_ | _fill in_ |
| Stance classification, LLM A | _fill in_ | _fill in_ | _fill in_ |
| Stance classification, LLM B | _fill in_ | _fill in_ | _fill in_ |
| Documentation and code drafting | Anthropic, Claude | Claude Sonnet 5.5 | November 2026 |

Reference papers kept in `papers/` are listed in the technical report, not here.
