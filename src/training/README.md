# Training runbook (Days 1–3)

All commands from the project root with the venv active (`source .venv/bin/activate`, `pip install -r requirements.txt`).

## Day 1 — data, splits, baselines (Mac)

```bash
python -m src.training.prepare_data          # check 7 versions, clean, dedupe, split, zip -> data/ecoconsensus_data.zip
python -m src.baselines                      # majority + TF-IDF/LR, C on val -> models/finetuned/baselines/
python -m pytest -q tests                    # loader, preprocessing, splits, metrics, reporting, LLM layer, stacking
```

Read `data/processed/split_report.md` once (class balance, lengths, hashtags, emojis, duplicates).
The splits are frozen: re-running `prepare_data` only re-zips them; `--overwrite-splits` rebuilds them (then retrain everything).

## Day 2 — transformers (two Colab accounts, in parallel) + LLM setup (Mac)

1. Upload `data/ecoconsensus_data.zip` to `MyDrive/EcoConsensus/` on **both** accounts.
2. Account 1: open `src/training/train_muril.ipynb` → T4 GPU → Run all.
   Account 2: open `src/training/train_indicbert.ipynb` → T4 GPU → Run all.
   Disconnected? Reconnect and Run all again: it resumes from the last finished epoch.
3. When each finishes, download `MyDrive/EcoConsensus/outputs_<model>/<model>_results.zip` and unzip it in the project root:
   `unzip -o muril_results.zip -d .` (same for indicbert). This fills `models/finetuned/<model>_lora/` and
   `outputs/predictions/probs_<model>_{meta,val,semeval_test}.csv`.

Meanwhile on the Mac (fill `.env` from `.env.example` first):

```bash
python -m src.training.run_llm_batch --provider a b --check                  # 3 comments per provider
python -m src.training.run_llm_batch --provider a --pilot                    # batch sizes 10/25/50/100 on 100 val comments
#   -> set LLM_BATCH_SIZE in .env
python -m src.training.run_llm_batch --provider a b --split val              # zero-shot on val (prints Macro-F1)
#   optional few-shot: put train_base ids in configs/llm.yaml few_shot_ids, rerun --split val, compare
#   -> freeze: prompt_frozen: true in configs/llm.yaml (never edit the prompt afterwards)
```

## Day 3 — LLM cache, features, meta-model (Mac)

```bash
# two terminals, one per provider; rerun the same command on later days if the daily limit stops it
python -m src.training.run_llm_batch --provider a --split meta val
python -m src.training.run_llm_batch --provider b --split meta val
python -m src.training.run_llm_batch --provider a --split semeval_test       # only after prompt_frozen: true
python -m src.training.run_llm_batch --provider b --split semeval_test

python -m src.training.train_inference      # -> outputs/predictions/features_{meta,val,semeval_test}.csv
python -m src.training.train_meta           # -> models/finetuned/meta/*.joblib + meta_info.json + report
```

Useful flags: `--estimate` (requests needed, nothing sent), `--max-requests N`, `--retry-unavailable`,
`--variants english/original hindi/pure_hindi bengali/pure_bengali` (quota cut-line), `--score` (metrics from cache only).

Outputs: logs in `outputs/logs/`, timestamped reports in `outputs/reports/`, MLflow in `outputs/metrics/mlflow.db`
(`mlflow ui --backend-store-uri sqlite:///outputs/metrics/mlflow.db`; the Colab runs are in `outputs/metrics/mlflow_colab_<model>.db`).
