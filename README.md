# EcoConsensus

Stance detection on global warming for Hindi and Bengali comments (CSET 346 NLP project). Given the claim **"Climate change and global warming is a serious concern."** and a comment, the system predicts **favour** (0), **neutral** (1) or **against** (2). The final system is a stacked ensemble of two fine-tuned transformers (MuRIL, IndicBERT v2) and two LLMs, compared against a majority-class baseline and a TF-IDF + logistic regression baseline. The primary metric is Macro-F1.

## Status

| Part | State |
|---|---|
| Planning documents (`docs/`) | Written |
| Experiment (`experiment/prepare_data.py`, `experiment/experiment.ipynb`) | Done on Colab; LoRA chosen for both models |
| Data, baselines, LLM layer, stacking scripts (`src/`, `src/training/`) | Written (Days 1–3); runbook in `src/training/README.md` |
| MuRIL and IndicBERT (LoRA) | Trained on Colab (2026-11-10); scored with `python -m src.evaluate` |
| `inference.py` (`src/pipeline.py`): one comment or a test file | Written (Day 4); guide `INFERENCE.md` |

## Where to start

| I want to... | Read |
|---|---|
| Run the experiments on Colab or Kaggle | `experiment/SETUP.md` |
| Understand the requirements | `docs/PRD.md`, `docs/TRD.md` |
| See how everything connects | `docs/APP_FLOW.md` |
| See the plan and day-by-day sign-offs | `docs/IMPLEMENTATION_PLAN.md` |
| See how the ensemble is built and trained | `docs/STACKING_ENSEMBLE.md` |
| Run `inference.py` on macOS, Linux or Windows, or score a test file | `INFERENCE.md` (root) |
| Use the LLM prompt | `prompts/stance_system_prompt.md` |

## Repository layout

See `docs/TRD.md` section 3. Short version: `data/` (raw, translated, processed, `test/` for the evaluator's file), `experiment/` (notebook, setup, generated reports), `src/` (with `src/training/` for data preparation and the Colab notebooks), `models/original` and `models/finetuned`, `outputs/`, `tests/` (unit tests), `docs/`, `prompts/`.

`outputs/` has one folder per component, each with the same four subfolders:

```text
outputs/
├── prepare_data/   baselines/   muril/   indicbert/   llm/   ensemble/   evaluate/   inference/
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
6. `inference.py` classifies one comment, or scores the file placed in `data/test/` (or any `-f` file) with the full comparison report in `outputs/inference/` (`INFERENCE.md`).

## Data

The course datasets were not provided, so the data was assembled independently from two public English datasets.
Full processing notes: `data/raw/readme.md`.

| Dataset | Where to get it | Labels | Use here |
|---|---|---|---|
| Pol-NLI (Burnham et al., 2024) | [Hugging Face `mlburnham/Pol_NLI`](https://huggingface.co/datasets/mlburnham/Pol_NLI) · [paper](https://arxiv.org/abs/2409.02078) | **LLM-derived, lower quality** (see below) | Training: `train_base` (1,424 source rows), `meta` (252) for the meta-model, `val` (90) for early stopping |
| SemEval-2016 Task 6, target "Climate Change is a Real Concern" (Mohammad et al., 2016) | [task page](http://alt.qcri.org/semeval2016/task6/) · [dataset page](https://www.saifmohammad.com/WebPages/StanceDataset.htm) · [paper](https://aclanthology.org/S16-1003/) | Human-annotated FAVOR / NONE / AGAINST | Held-out test (`semeval_test`, 564 tweets); a small subset was used in the method-comparison experiment |
| Evaluator's hidden test set | given on evaluation day | unknown format | Final test with `inference.py` (file in `data/test/`) |

### Data quality and provenance (read before using the scores)

- **Hindi and Bengali data is LLM-translated.** Every English sentence of both datasets was machine-translated once with an
  LLM into six versions: pure Hindi, Hindi code-mixed, Hinglish, pure Bengali, Bengali code-mixed and Banglish. The
  translations may contain errors, and each one keeps the label of its English source. Scores on
  Hindi and Bengali therefore describe machine-translated text, not natural social-media writing. (An earlier batch
  translated into the wrong language, `data/raw/translated/Bagheli/`, is never used.)
- **Pol-NLI labels are low quality.** Pol-NLI only has binary entailment labels for hypotheses such as "Climate change is
  (not) a serious concern". The negated hypotheses were rewritten and their labels flipped, and the three-class stance
  labels (favour / neutral / against) were then produced **with the help of an LLM (Claude)**. They are silver labels, not
  human annotations. This is why Pol-NLI is used only for training and the human-labelled SemEval tweets are kept for testing.
- **Duplicates were removed.** 1,417 Pol-NLI rows repeated another row's English text, and 64 train groups matched a
  `val` or SemEval sentence (`src/training/prepare_data.py`).

## Reports

All reports are Markdown (and `result.txt` from `inference.py`), written only by script code from saved results, with timestamped names. A report is never overwritten; a name clash adds a numeric suffix.

## Limitations

- Training text is not YouTube comments; translated text differs from natural Hindi and Bengali writing.
- All Hindi and Bengali data is LLM-translated, and the Pol-NLI training labels are LLM-derived (silver, lower quality).
- Out of domain the fine-tuned transformers are weak: on SemEval tweets they score below the TF-IDF baseline, and the
  stacked ensemble (Macro-F1 0.650) does not beat the best single LLM (0.779). All components are still kept (TRD decision 32).
- SemEval has few `against` examples, so Macro-F1 is sensitive to small changes.
- Free LLM tiers limit daily requests; results depend on the cache being complete.
- Method comparison used a small number of epochs and seeds.

## References

Plain-text citations for every library, method and LLM are in `docs/REFERENCES.md`. BibTeX for the datasets, the
models and every paper in `papers/` is below. Fields marked `TO_FILL` still need to be checked and completed.

### Datasets

The two dataset entries are copied from `data/raw/readme.md`.

SemEval-2016 Task 6 ([task page](http://alt.qcri.org/semeval2016/task6/), [dataset](https://www.saifmohammad.com/WebPages/StanceDataset.htm)):

```bibtex
@inproceedings{mohammad2016semeval,
  title={SemEval-2016 Task 6: Detecting Stance in Tweets},
  author={Mohammad, Saif and Kiritchenko, Svetlana and Sobhani, Parinaz and Zhu, Xiaodan and Cherry, Colin},
  booktitle={Proceedings of the 10th International Workshop on Semantic Evaluation (SemEval-2016)},
  pages={31--41},
  year={2016},
  publisher={Association for Computational Linguistics}
}
```

Pol-NLI / Political DEBATE ([dataset](https://huggingface.co/datasets/mlburnham/Pol_NLI), [paper](https://arxiv.org/abs/2409.02078)):

```bibtex
@misc{burnham2024politicaldebateefficientzeroshot,
      title={Political DEBATE: Efficient Zero-shot and Few-shot Classifiers for Political Text}, 
      author={Michael Burnham and Kayla Kahn and Ryan Yang Wang and Rachel X. Peng},
      year={2024},
      eprint={2409.02078},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.02078}, 
}
```

Companion article to SemEval-2016 Task 6 (`papers/1605.01655v1.pdf`):

```bibtex
@article{mohammad2017stance,
  title={Stance and Sentiment in Tweets},
  author={Mohammad, Saif M. and Sobhani, Parinaz and Kiritchenko, Svetlana},
  journal={ACM Transactions on Internet Technology},
  volume={17},
  number={3},
  articleno={26},
  year={2017},
  doi={TO_FILL},
  eprint={1605.01655},
  archivePrefix={arXiv}
}
```

### Transformer models

MuRIL, model [`google/muril-base-cased`](https://huggingface.co/google/muril-base-cased) (`papers/2103.10730v2.pdf`):

```bibtex
@misc{khanuja2021muril,
  title={MuRIL: Multilingual Representations for Indian Languages},
  author={Khanuja, Simran and Bansal, Diksha and Mehtani, Sarvesh and Khosla, Savya and Dey, Atreyee and Gopalan, Balaji and Margam, Dilip Kumar and Aggarwal, Pooja and Nagipogu, Rajiv Teja and Dave, Shachi and Gupta, Shruti and Gali, Subhash Chandra Bose and Subramanian, Vish and Talukdar, Partha},
  year={2021},
  eprint={2103.10730},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2103.10730}
}
```

IndicBERT v2, model [`ai4bharat/IndicBERTv2-MLM-only`](https://huggingface.co/ai4bharat/IndicBERTv2-MLM-only) (`papers/2212.05409v3.pdf`):

```bibtex
@inproceedings{doddapaneni2023indicbertv2,
  title={Towards Leaving No Indic Language Behind: Building Monolingual Corpora, Benchmark and Models for Indic Languages},
  author={Doddapaneni, Sumanth and Aralikatte, Rahul and Ramesh, Gowtham and Goyal, Shreya and Khapra, Mitesh M. and Kunchukuttan, Anoop and Kumar, Pratyush},
  booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={TO_FILL},
  year={2023},
  publisher={Association for Computational Linguistics},
  url={TO_FILL},
  eprint={2212.05409},
  archivePrefix={arXiv}
}
```

Fine-tuning method (LoRA):

```bibtex
@inproceedings{hu2022lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2022},
  eprint={2106.09685},
  archivePrefix={arXiv}
}
```

### LLMs used as ensemble components

```bibtex
@misc{google2026gemini35flashlite,
  title={Gemini 3.5 Flash-Lite},
  author={{Google}},
  howpublished={Gemini API, model gemini-3.5-flash-lite},
  year={2026},
  url={TO_FILL},
  note={Accessed November 2026}
}

@misc{openai2025gptoss20b,
  title={gpt-oss-20b},
  author={{OpenAI}},
  howpublished={Open-weight model openai/gpt-oss-20b, served by the Groq API},
  year={TO_FILL},
  url={TO_FILL},
  note={Accessed November 2026}
}
```

### Other papers in `papers/`

LLM ensembling (`papers/2023.acl-long.792.pdf`):

```bibtex
@inproceedings{jiang2023llmblender,
  title={LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion},
  author={Jiang, Dongfu and Ren, Xiang and Lin, Bill Yuchen},
  booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={14165--14178},
  year={2023},
  publisher={Association for Computational Linguistics},
  url={TO_FILL}
}
```

Confidence elicitation in LLMs (`papers/2306.13063v2.pdf`):

```bibtex
@inproceedings{xiong2024llmuncertainty,
  title={Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs},
  author={Xiong, Miao and Hu, Zhiyuan and Lu, Xinyang and Li, Yifei and Fu, Jie and He, Junxian and Hooi, Bryan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024},
  eprint={2306.13063},
  archivePrefix={arXiv}
}
```

Climate-activism stance shared task (`papers/2024.case-1.33.pdf`):

```bibtex
@inproceedings{thapa2024case,
  title={Stance and Hate Event Detection in Tweets Related to Climate Activism - Shared Task at {CASE} 2024},
  author={Thapa, Surendrabikram and Rauniyar, Kritesh and Jafri, Farhan Ahmad and Shiwakoti, Shuvam and Veeramani, Hariram and Jain, Raghav and Kohli, Guneet Singh and H{\"u}rriyeto{\u{g}}lu, Ali and Naseem, Usman},
  booktitle={Proceedings of the 7th Workshop on Challenges and Applications of Automated Extraction of Socio-political Events from Text (CASE 2024)},
  pages={234--247},
  year={2024},
  publisher={Association for Computational Linguistics},
  address={TO_FILL},
  url={TO_FILL}
}
```

Code-mixed Hinglish: fine-tuned models against LLMs (`papers/2602.21933v1.pdf`):

```bibtex
@misc{majumder2026smallwinsbig,
  title={Small Wins Big: Comparing Large Language Models and Domain Fine-Tuned Models for Sarcasm Detection in Code-Mixed Hinglish Text},
  author={Majumder, Bitan and Sen, Anirban},
  year={2026},
  eprint={2602.21933},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2602.21933},
  note={TO_FILL: venue if published}
}
```

The other files in `papers/` are already cited above: `1605.01655v1.pdf` (Mohammad et al., 2017),
`2103.10730v2.pdf` (MuRIL), `2212.05409v3.pdf` (IndicBERT v2), `2409.02078v1.pdf` (Political DEBATE / Pol-NLI).

### LLMs used (the assignment requires naming every LLM)

| Purpose | Provider and company | Model and version | Date used |
|---|---|---|---|
| Translation of both datasets to Hindi and Bengali | _fill in_ | _fill in_ | _fill in_ |
| Refining Pol-NLI labels into favour / neutral / against | Anthropic, Claude | _fill in (model version)_ | _fill in_ |
| Stance classification, LLM A | Google, Gemini API (OpenAI-compatible endpoint) | `gemini-3.5-flash-lite`, prompt `v3`, zero-shot | November 2026 |
| Stance classification, LLM B | OpenAI open-weight model served by Groq | `openai/gpt-oss-20b` (reasoning effort low), prompt `v3`, zero-shot | November 2026 |
| Documentation and code drafting | Anthropic, Claude (Claude Code) | Claude Sonnet 5.5, Claude Opus 5.5 | November 2026 |
