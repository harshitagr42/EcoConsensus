# Experiment setup (Colab, Kaggle as backup)

The experiment compares four fine-tuning methods (head only, head + last N layers, LoRA, QLoRA) for MuRIL and IndicBERT v2, so that one method per model, and for LoRA its hyperparameters, can be chosen for the final training. It runs in two stages: **stage 1** searches LoRA's rank `r`, `alpha`, target modules and learning rate over the grid `LORA_SEARCH` (cell 1) and picks one combination per model on `val`; **stage 2** runs every method with all `SEEDS`, LoRA and QLoRA using that pick. It uses **only a small subset of SemEval-2016 Task 6** (climate tweets) in English, the three Hindi variants and the three Bengali variants.

Files in `experiment/`:

| File | Purpose |
|---|---|
| `prepare_data.py` | Runs on your Mac. Builds the small data bundle `experiment/experiment_data.zip`. No training |
| `experiment.ipynb` | Single notebook you run on Colab (or Kaggle). Unzips the bundle, trains, writes results, MLflow and reports to Drive. Only cell 1 needs editing |
| `SETUP.md` | This file |

Labels everywhere: **0 = favour, 1 = neutral, 2 = against**.

## 1. Build the data bundle (Mac, once)

Run from the project root (Python 3.10+, needs pandas only; the project `.venv` works):

```bash
python experiment/prepare_data.py              # 300 tweets (default)
python experiment/prepare_data.py --rows 200   # a different subset size
python experiment/prepare_data.py --rows 0     # all 564 tweets
```

What it reads (listed explicitly in the `DATASETS` block, so the Bagheli folder is never picked up):

| Language | Files (`final_processed_*.csv`) | Variant names in reports |
|---|---|---|
| English | `data/raw/original/SemEval_2016_Task6/final_processed.csv` | `original` |
| Hindi | `data/raw/translated/Hindi/SemEval_2016_Task6/` | `pure_hindi`, `hinglish`, `hindi_code_mix` |
| Bengali | `data/raw/translated/Bengali/SemEval_2016_Task6/` | `pure_bengali`, `banglish`, `bengali_code_mix` |

What it does:

1. Maps `Stance` to the project labels (`FAVOR` = favour, `NONE` = neutral, `AGAINST` = against; numbers 0/1/2 map the same way).
2. Checks that every translated file is row-aligned with the English file (same `ID`s, labels agree for at least 98% of rows). It stops with a clear message otherwise.
3. Samples `SAMPLE_SOURCE_ROWS` tweets (default 300). **Every `against` tweet is kept** (`KEEP_ALL_CLASSES`), because SemEval has only 26 of them; a plain stratified sample of 300 would keep 14, leaving 2 or 3 per evaluation split. The rest of the quota is sampled stratified by label. This makes `against` about 9% of the subset instead of about 5% of the full data. Set `KEEP_ALL_CLASSES = []` for a plain stratified sample.
4. Keeps all 7 versions of every sampled tweet (1 English + 3 Hindi + 3 Bengali), so 300 tweets give 2,100 rows and every language and variant has the same class ratio (the script asserts this).
5. Prints the counts per language, variant and label, and writes `experiment/experiment_data/semeval.csv`, `manifest.json` and the zip `experiment/experiment_data.zip`.

Columns in `semeval.csv`: `source_id, language, variant, claim, comment, label`. Text is not cleaned here; the notebook applies the shared preprocessing.

## 2. Run on Google Colab (T4)

1. In Google Drive, create the folder `MyDrive/EcoConsensus/` and upload `experiment_data.zip` into it. (A different folder name: change `DRIVE_FOLDER` in cell 1.)
2. Go to colab.research.google.com, then **File > Upload notebook** and pick `experiment/experiment.ipynb`.
3. **Runtime > Change runtime type > T4 GPU > Save.**
4. **Runtime > Run all.** Approve the Google Drive access prompt in cell 2. (If no GPU is attached, cell 2 stops with a message instead of training on CPU.)
5. Cell 4 unzips a fresh copy of the bundle into `/content/experiment_data_unzipped/`, prints the class table, and splits the tweets.
6. Wait for the last cell. Everything is written to `MyDrive/EcoConsensus/experiment_outputs/`:

   | Path | Contents |
   |---|---|
   | `reports/` | `experiment_report_<timestamp>.md` and `_runs.csv`. A new file every time; old ones are never overwritten |
   | `results/` | One JSON per run (metrics, history, params, time, memory) |
   | `predictions/` | Per-row predictions and probabilities of every run (val and test) |
   | `splits/<id>/` | The frozen `train` / `val` / `test` CSVs and a split report |
   | `mlflow.db` | MLflow tracking store (SQLite): params, metrics per epoch, per-class and per-variant F1 |
   | `mlartifacts/` | MLflow artifacts: each run's result JSON and each generated report |

   MLflow note: SQLite on the Drive mount can hit file-locking errors, so the notebook writes `/content/mlflow.db` and copies it to Drive after every run. After a disconnect it restores the Drive copy first, so the store keeps growing across sessions.

If the session disconnects: reconnect and **Run all** again. Splits are reused, finished runs are skipped, and only missing runs are done (a run that was cut off starts again from the beginning; runs take a few minutes).

## 3. Kaggle (backup)

1. **New Notebook**, then **File > Import Notebook** and choose `experiment.ipynb`.
2. **Add Input > Upload > New Dataset**, upload `experiment_data.zip`. Kaggle extracts it; the notebook finds `semeval.csv` either way.
3. Right panel: **Accelerator = GPU T4** (or P100), **Internet = On** (for the models and pip packages).
4. **Run All.** Outputs go to `/kaggle/working/experiment_outputs/` (same layout as above, `mlflow.db` written directly there). They survive only with **Save Version > Save & Run All (Commit)** or if you download the folder before the session ends.

## 4. How the data is split

The notebook splits the subset **by tweet**, stratified by label, so all 7 versions of a tweet are in the same split (no translation of a test tweet is ever trained on):

| Split | Share (default) | Used for |
|---|---|---|
| `train` | 60% (`TRAIN_FRACTION`) | Training. By default all 7 versions of each tweet (`TRAIN_VARIANT_POLICY = "all"`) |
| `val` | 20% (`VAL_FRACTION`) | Picking the best epoch and **ranking the methods** |
| `test` | the remaining 20% | Held out. Shown only for the recommended method per model, never used to choose |

With 300 tweets that is 180 / 60 / 60 tweets (1,260 / 420 / 420 rows). The split is saved under `splits/<id>/` and reused as long as the data and split settings are unchanged.

The claim fed to the model is the project's fixed claim, "Climate change and global warming is a serious concern." (`CLAIM_MODE = "fixed"`), the same text `test.py` and `inference.py` will use. `CLAIM_MODE = "file"` uses the (translated) `Target` column instead.

## 5. Before you press Run all

- [ ] `experiment_data.zip` is in `MyDrive/EcoConsensus/`, and the "Loaded semeval.csv: ..." line in cell 4 shows the rows and 7 language/variant groups you expect.
- [ ] GPU is T4. Cell 2 prints `Device: cuda`.
- [ ] Cell 1 `MODELS`: the IndicBERT checkpoint is the one you want (default `ai4bharat/IndicBERTv2-MLM-only`).
- [ ] For a fast first pass set `SEEDS = [42]`, `EPOCHS = 1` and `SEARCH_MAX_TRIALS = 4` (or `TRAIN_SOURCE_ROWS = 60`). Then go back to the defaults.

Rough time at the defaults (1,260 training rows, 3 epochs, 2 models): stage 1 is 36 LoRA combinations x 1 seed x 2 models = 72 short runs, stage 2 is 5 methods x 2 seeds x 2 models = 20 runs. The previous run measured about 10 to 15 s per epoch on a T4, so **an estimate of roughly 1 to 1.5 hours**, most of it stage 1. `SEARCH_MAX_TRIALS = 12` cuts stage 1 to a third. Both stages resume after a disconnect.

Library versions: cell 2 runs `pip install -U transformers peft accelerate mlflow scikit-learn pandas` (+ `bitsandbytes` on Linux), so you get the latest releases. The installed versions are printed and saved in every report. If a new release breaks something, pin versions in `_pip(...)` in cell 2.

## 6. Points to keep in mind while deciding

1. **Ranking and held-out.** Methods are ranked on `val` Macro-F1. `test` is shown only for the recommended method. Do not change your choice after looking at `test`; that turns it into a second val set.
2. **Small sets, small gaps.** `val` has about 60 tweets (5 `against`), each in 7 versions that are not independent. Gaps of a few Macro-F1 points are noise. The report recommends the method with the fewest trainable parameters among those within noise of the best (the parsimony rule).
3. **`against` is rare.** Its F1 moves Macro-F1 a lot. Check the "val F1 against" column, the per-class F1 and the confusion matrix. If `against` is near zero for every method, more data (`--rows 0`) helps more than a different method.
4. **Check both seeds.** Look at the "per-seed F1" column. A method that wins on one seed and loses on the other is not a winner. Add a seed (`SEEDS = [42, 43, 44]`) for the top two; finished runs are reused.
5. **Compare against TF-IDF.** Section 4 of the report has Majority and TF-IDF + LR on the same splits. If no transformer beats TF-IDF, say so in the write-up.
6. **Languages and variants.** Section 6 shows Macro-F1 per language and per script variant. Every variant column has the same tweets, so differences come from the text form (translation quality, romanisation, tokenizer coverage; the `[UNK]` rates in section 2 help to separate these).
7. **This is in-domain and small.** Train and val are both SemEval tweets, so scores are higher than they will be for training on other data, and the ranking may shift with more data (LoRA and deeper partial fine-tuning usually gain the most). The final training follows `docs/TRD.md`.
8. **QLoRA** needs CUDA and `bitsandbytes`. If it fails, it is recorded as skipped with the reason, never as zero. For models this small it saves little memory and is slower than LoRA; it is there to show the technique. By default it reuses LoRA's picked hyperparameters (`QLORA_HPARAMS = "from_lora"`), so the LoRA vs QLoRA gap is the cost of quantization alone. `QLORA_BITS = [4, 8]` adds LoRA on an 8-bit base as a second quantization level.
8a. **The LoRA search uses `val` too.** Its pick, the best epoch and the method ranking all come from `val`, so LoRA-type `val` scores are a little optimistic compared with head-only and last-N (fixed learning rates). Stage 2 uses different seeds from the search (`SEARCH_SEEDS` vs `SEEDS`), and `test` stays the fair check.
9. **Changing settings.** Results are keyed by a hash of all settings. Change a setting and those runs are recomputed; old results stay in `results/` and are ignored by new reports.
10. **Record the decision.** Note the report file you base the choice on, with its timestamp and the MLflow run names, and add it to the decision log in `docs/TRD.md`.

## 7. After the run

1. Open the newest `reports/experiment_report_*.md` on Drive. Section 1 gives the recommendation per model and the baseline comparison.
2. To browse MLflow on the Mac: download `experiment_outputs/mlflow.db` (and `mlartifacts/` if you want the artifacts), then from that folder run `mlflow ui --backend-store-uri sqlite:///mlflow.db`. Metrics and parameters show up; artifact links point at Drive paths and will not open locally (the same files are in `results/` and `reports/`).
3. Once a method is chosen for each model, move on to `src/` (final training, LLM component, stacking, `inference.py`, `test.py`).

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| "Could not find the dataset" | The zip is not at `MyDrive/EcoConsensus/experiment_data.zip` (Colab) or not added as an input (Kaggle). Check `DRIVE_FOLDER` |
| "... has no semeval.csv" | The zip was built by an older `prepare_data.py`. Rebuild it and upload again |
| "No GPU found" | Change the runtime type to T4 GPU and run again |
| Tokenizer or model download error | Internet must be on (Kaggle). Check the model id. The other model still runs |
| CUDA out of memory | The notebook retries once with half the batch size and 2x accumulation. If it still fails, lower `BATCH_SIZE` or `MAX_LEN` in cell 1 |
| `bitsandbytes` error | QLoRA is skipped. Everything else is unaffected |
| `[mlflow log failed]` / `[mlflow disabled]` | Training and reports are unaffected (they do not depend on MLflow). Check the message; `USE_MLFLOW = False` turns it off |
| `torch.load` / safetensors error while loading a model | Update `torch` and `transformers` (restart the runtime after pip) or use a checkpoint that ships `.safetensors` |
| Pip upgrade seems ignored | Runtime > Restart session, then Run all |

## 9. Sources

Every model, library and dataset used is cited in section 11 of each generated report. Main ones: MuRIL (Google; Khanuja et al., 2021), IndicBERT v2 (AI4Bharat; Doddapaneni et al., ACL 2023), Hugging Face Transformers and PEFT, LoRA (Hu et al., 2021, Microsoft), QLoRA (Dettmers et al., 2023) and bitsandbytes, AdamW (Loshchilov and Hutter, 2019), PyTorch, scikit-learn, MLflow, SemEval-2016 Task 6 (Mohammad et al., 2016).
