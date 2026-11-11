# Using `inference.py`

Classify one comment, or a whole file of comments (for example the evaluator's test file), toward the claim *"Climate change and global warming is a serious concern."* Works on macOS, Linux and Windows.

This page sits next to `inference.py` in the repository root. If the code and this page ever differ, fix this page on the same day.

`inference.py` has two inputs:

- **One comment** (`-t "text"`): prints the label and each component's vote. One request per LLM.
- **A file**: the single data file in `data/test/` (default) or any file given with `-f PATH`. Every system runs (baselines, both transformers, both LLMs in batches, votes, stacked ensemble), and the full report and the prediction file are written to `outputs/inference/`. If the file has labels, the report compares all systems.

LLM answers are cached per comment, so a repeated comment costs nothing.

---

## 1. One-time setup

You need Python 3.13 or newer (3.11 is the lowest that may work), several GB of free disk for PyTorch and the base models, and internet on the first run (the base models download from the Hugging Face Hub and are cached after that).

### macOS and Linux (bash or zsh)

```bash
git clone <repo-url> EcoConsensus
cd EcoConsensus
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then open .env and fill in the keys
```

### Windows (PowerShell)

```powershell
git clone <repo-url> EcoConsensus
cd EcoConsensus
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env   # then open .env and fill in the keys
$env:PYTHONUTF8 = "1"         # needed so Hindi and Bengali print correctly
```

If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope Process RemoteSigned` once in that window.

### Windows (cmd.exe)

```bat
py -3.13 -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
copy .env.example .env
set PYTHONUTF8=1
chcp 65001
```

### The `.env` file

Keys are read only from `.env`; nothing is hard-coded. Variables:

| Variable | Meaning |
|---|---|
| `LLM_A_PROVIDER`, `LLM_A_MODEL`, `LLM_A_API_KEY`, `LLM_A_BASE_URL` | First LLM (base URL only for OpenAI-compatible endpoints) |
| `LLM_B_PROVIDER`, `LLM_B_MODEL`, `LLM_B_API_KEY`, `LLM_B_BASE_URL` | Second LLM |
| `LLM_A_DAILY_LIMIT`, `LLM_B_DAILY_LIMIT` | Requests per day, used by the budget guard |
| `LLM_BATCH_SIZE` | Comments per LLM request for a file run and the training scripts (one comment is always sent alone) |
| `HF_TOKEN` | Optional, for Hugging Face downloads |

No keys? The script still runs with the two transformers only (see `--no-llm`).

### Model files

`models/finetuned/` must contain `muril_lora/`, `indicbert_lora/`, `baselines/` and `meta/`. `models/` is git-ignored, so these folders come from the training run (`muril_results.zip` / `indicbert_results.zip` from Colab, then `python -m src.baselines` and `python -m src.training.train_meta`; see `src/training/README.md`) or from the shared project bundle. The LoRA adapters are small (about 5 MB each); the base models (`google/muril-base-cased`, `ai4bharat/IndicBERTv2-MLM-only`) download from the Hugging Face Hub on the first run. If a file is missing, the script names it and stops (exit code 4).

Only the components a mode needs are loaded: `--mode tfidf` needs no transformer, `--no-llm` sends no request. The two transformers are loaded one after the other, so 8 GB of RAM is enough.

---

## 2. Quick examples

Replace `python` with `python3` if your system needs it. Examples work in every shell unless marked.

```bash
# One comment (English)
python inference.py -t "We can't deny it, it's really happening."

# One comment, Hindi (macOS and Linux, any shell)
python inference.py -t "ग्लोबल वार्मिंग सिर्फ एक झूठ है।"

# The test file in data/test/ (full report)
python inference.py --inspect
python inference.py

# Another file
python inference.py -f my_comments.csv

# A file with different column names
python inference.py -f data.xlsx --text-col "Comment Text" --id-col "No"

# Transformers only (no API calls)
python inference.py -t "some comment" --no-llm

# Machine-readable output
python inference.py -t "some comment" --json
```

**Windows quoting.** In PowerShell use double quotes. In cmd.exe, Hindi and Bengali typed on the command line can arrive garbled even after `chcp 65001`; put the text in a file and use `-f` instead.

---

## 3. Evaluator day: scoring a test file in about a minute

1. **Put the file in `data/test/`** (create the folder if needed). Keep only one data file there.
2. **Look at it:** `python inference.py --inspect`. It prints the file's columns, the first 5 rows, the distinct values in the likely label column, and the configuration it would use. It runs no model.
3. **Fix the configuration if the guess is wrong**, either by editing the `CONFIG` block at the top of `inference.py` or by adding flags to the command (flags win over the block, and the block wins over auto-detection).
4. **Run:** `python inference.py` (add `--no-llm` if the internet or the API keys are not available).
5. **Read** `outputs/inference/result.txt`. The prediction file is `outputs/inference/predictions.csv`.

If something goes wrong, the message names the column, label value or file involved and says what to change.

### The `CONFIG` block

This block sits at the very top of `inference.py`. It applies to file runs (and `fixed_claim` / `use_llm` / `llm_budget` to one comment too). Edit the values you need; leave the rest.

```python
CONFIG = {
    "data_file":    None,        # None = the only data file in data/test/; or e.g. "comments.csv"
    "id_col":       None,        # e.g. "ID"; None = auto-detect, else the row number is used
    "text_col":     None,        # e.g. "Tweet"; the comment text (required, auto-detected if None)
    "label_col":    None,        # e.g. "Stance"; None = auto-detect, no labels found = prediction-only
    "claim_col":    None,        # None = always use the fixed claim below
    "language_col": None,        # optional; if None the script group is detected from the text
    "label_map": {               # file value (any case) -> favour / none / against
        "FAVOR": "favour", "FAVOUR": "favour", "SUPPORT": "favour",
        "AGAINST": "against", "OPPOSE": "against",
        "NONE": "none", "NEUTRAL": "none", "UNRELATED": "none",
    },
    "delimiter":    None,        # None = auto-detect
    "encoding":     None,        # None = try utf-8-sig, utf-8, utf-16, cp1252, latin-1
    "fixed_claim":  "Climate change and global warming is a serious concern.",
    "use_llm":      True,
    "llm_budget":   None,        # max requests per provider this run; None = use the .env daily limit
}
```

Numeric labels are **never** mapped by default, because `0/1/2` can mean different things in different files. If the file uses numbers, add them yourself, for example `{"0": "favour", "1": "none", "2": "against"}` (only after checking what the file means; `--inspect` shows the distinct values).

### Typical test-file shapes and what to do

| What the file looks like | What to set |
|---|---|
| `ID,Target,Tweet,Stance` with FAVOR / AGAINST / NONE | Nothing; auto-detect handles it (`Target` is ignored: the fixed claim is used unless you set `claim_col`) |
| `id,comment,label` with Favour / Against / None | Nothing |
| `comment_id,yt_comment,stance` with 0 / 1 / 2 | `--text-col yt_comment --id-col comment_id --label-col stance --label-map "0=favour,1=none,2=against"` (confirm the meaning first) |
| Only one column of comments, no header | Add `--text-col 0` (a number picks the column by position when there is no header) |
| Labels in Hindi or Bengali words | Add them to `label_map`, for example `"पक्ष": "favour"` |
| No label column | Nothing; the run is prediction-only and the report says so |
| Several sheets in an `.xlsx` | The first sheet is used; save the right sheet as CSV if needed |
| A claim column that differs by row | `--claim-col` only if you want the claim used (the models were trained for the default claim) |

### What the run does

1. Finds the file, reads it (trying delimiters and encodings), applies your configuration.
2. Cleans every comment with the same preprocessing used in training.
3. Runs: majority baseline, TF-IDF baseline, MuRIL, IndicBERT (one transformer in memory at a time; about 1 minute for 4,000 rows on an M3 Mac). Empty comments are predicted too (so every ID gets a label) but are left out of the metrics and never sent to an LLM.
4. Sends the comments to LLM-A and LLM-B in batches (answers are cached per comment, so a rerun costs nothing).
5. Applies the stacked meta-model comment by comment. If an LLM answer is missing, the matching fallback model is used and counted.
6. If labels exist: computes the metrics, picks the **best single transformer** and the **best single LLM** by Macro-F1 on this file, and compares everything.
7. Writes the report and the prediction file (section 5).

---

## 4. All flags

### Input

| Flag | Meaning |
|---|---|
| `-t TEXT`, `--text TEXT` | A single comment to classify |
| `-f PATH`, `--file PATH` | A file of comments (`.csv`, `.tsv`, `.json`, `.jsonl`, `.xlsx`, `.txt`) instead of the one in `data/test/`. Use either `-t` or `-f`, not both |
| *(neither)* | The single data file in `data/test/` (or `CONFIG["data_file"]`) |
| `--inspect` | Show the file's columns, 5 sample rows, distinct label values and the planned config, then stop (runs no model) |
| `-c TEXT`, `--claim TEXT` | Use a different claim. The models are trained for the default claim, so results for other claims are not reliable |
| `--id-col NAME` | Column holding the row id (default: auto-detect; falls back to row number) |
| `--text-col NAME` | Column holding the comment (default: auto-detect) |
| `--claim-col NAME` | Column holding a claim per row (default: none, use the fixed claim; a column named `Target` or `claim` is **not** picked up unless you name it here) |
| `--label-col NAME` | Column holding true labels (default: auto-detect; if found, the report compares all systems) |
| `--language-col NAME` | Optional language column (adds a by-language breakdown to the report) |
| `--label-map "A=favour,B=none,C=against"` | How the file's label values map to the three classes (`favour`, `none`, `against`). Needed for numeric labels |
| `--allow-unknown-labels` | Continue when some label values are not in the map; those rows are treated as unlabeled |
| `--delimiter CHAR` | Force a delimiter for `.csv` / `.tsv` / `.txt` |
| `--encoding NAME` | Force a text encoding (default: tries `utf-8-sig`, `utf-8`, `utf-16`, `cp1252`, `latin-1`) |

### Model selection

| Flag | Meaning |
|---|---|
| `-m MODE`, `--mode MODE` | Which system gives the final label (for a file, the report still shows every system that ran): `ensemble` (default), `transformers` (both transformers combined by the transformers-only meta-model, no LLM), `muril`, `indicbert`, `llm-a`, `llm-b`, `tfidf`, `majority`. In `llm-a` / `llm-b` mode a comment the LLM does not answer falls back to TF-IDF (shown as `tfidf_lr fallback`) |
| `--no-llm` | Do not call any LLM. The stacked model falls back to its transformers-only variant |
| `--require-llm` | Stop with exit code 5 if an LLM is unavailable instead of falling back |
| `--llm-budget N` | Send at most N requests per provider in this run; the rest fall back |
| `--batch-size N` | Comments per LLM request for a file (default `LLM_BATCH_SIZE`, normally 50) |
| `--device DEVICE` | `auto` (default), `cpu`, `cuda` or `mps` |
| `--config PATH` | Use another base config file instead of `configs/base.yaml` (`transformers.yaml` and `llm.yaml` stay in `configs/`) |

### Output

| Flag | Meaning |
|---|---|
| `-o DIR`, `--output-dir DIR` | Where results go: `DIR/inference/` (default `outputs/inference/`) |
| `--label-style STYLE` | How labels are written: `ps` (default: `Favour`, `None`, `Against`), `file` (the spelling used in the input file's label column), or `numeric` (`0`, `1`, `2`) |
| `--json` | Print the result (one comment) or a run summary with Macro-F1 per system and the output paths (file) as JSON; log lines go to stderr, so stdout stays valid JSON |
| `-v`, `--verbose` | One comment: also print the progress log and each transformer's and the ensemble's probabilities |
| `-q`, `--quiet` | One comment: print only the final label. File: print only the output paths |

### Other

| Flag | Meaning |
|---|---|
| `--version` | Print the version and exit |

Flags override the matching `CONFIG` entry; the block overrides auto-detection.
| `-h`, `--help` | Print usage and exit |

---

## 5. What you see

One comment:

```text
Comment : ग्लोबल वार्मिंग सिर्फ एक झूठ है।
Claim   : Climate change and global warming is a serious concern.
Final   : Against
Votes   : MuRIL=Against  IndicBERT=Against  LLM-A=Against  LLM-B=None
Mode    : meta_all4
Log     : outputs/inference/logs/inference_20261112_101500.txt
```

(The values above are an example of the layout, not a real result.) `Mode` is the meta-model variant that produced the label: `meta_all4` when all four components answered, `meta_no_llm_a` / `meta_no_llm_b` when one LLM did not, `meta_transformers` with `--no-llm` or when neither LLM answered.

A file run logs its progress, prints Macro-F1 per system (if the file has labels) and the paths of the files below.

### Files written

One comment writes only its log. A file run writes everything below `outputs/inference/` (or `DIR/inference/` with `-o DIR`):

| File | Content |
|---|---|
| `result.txt` | The full report (latest). The previous one is moved to `archive/` first |
| `reports/result_<time>.md` | The same report as Markdown, timestamped |
| `predictions.csv` | `ID,Predicted_Label` for every row, the problem-statement prediction file (previous one archived) |
| `logs/inference_<time>.txt` | Exact command, configuration, per-step log, warnings |
| `predictions/inference_<time>_detailed.csv` | Every component's output per comment (transformer probabilities, LLM answers, each system's label, meta-model variant, ensemble probabilities, final label and its source) |
| `metrics/inference_<time>.json` | The numbers behind the report |

Reading a CSV with pandas? Pass `keep_default_na=False`, otherwise the label `None` is read as a missing value.

Nothing is ever overwritten without being kept first. All reports are produced by the script; no AI writes them.

### Reading `result.txt`

1. **Header:** time, command, file, rows read, rows dropped (empty comments) and why, label counts.
2. **Comparison table:** Macro-F1 (first, with a bootstrap 95% interval), accuracy, weighted-F1, MCC, kappa and per-class F1 for: majority, TF-IDF, MuRIL, IndicBERT, best transformer, LLM-A, LLM-B, best LLM, soft vote, majority vote, **ensemble**. "Best" is chosen on this same file and the report says so. A closing line states whether the ensemble beats the best single system.
3. **Per-class table:** precision, recall, F1 for Favour, None, Against, for every system.
4. **Confusion matrices:** rows are true labels, columns are predictions.
5. **By script group:** Devanagari, Bengali, Latin (Latin covers English and romanised text).
6. **Ensemble details:** how many comments used each meta-model variant, the learned weights, LLM requests used, cache hits, parse errors and `unavailable` counts.
7. **Error analysis:** the most common confusions and the first 20 misclassified comments with each component's vote.
8. **Environment and configuration:** library versions and the exact configuration used, including your `CONFIG` values.
9. **Warnings and limitations.**

With no labels, sections 2–5 and 7 are replaced by a prediction summary: label counts per system and how often the systems agree.


---

## 6. Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 2 | Bad usage, configuration or column error |
| 3 | Data error (file not found, unreadable, empty) |
| 4 | Model files missing |
| 5 | An LLM was unavailable and `--require-llm` was set |

## 7. Common problems

| Problem | Fix |
|---|---|
| `python` is the wrong version | Use `python3.13` (macOS/Linux) or `py -3.13` (Windows) when creating the virtual environment |
| Hindi or Bengali shows as `?` or boxes | Use a UTF-8 terminal and a font with those scripts; on Windows set `PYTHONUTF8=1` |
| "No API key for LLM-A" | Fill `.env`, or use `--no-llm` |
| First run is slow | The base models download once, then load from the cache |
| "Model files missing" | Check that `models/finetuned/` has the four folders listed above |
| Wrong column picked | Set `--text-col`, `--id-col`, `--label-col` explicitly |
| Numeric labels rejected ("Numeric labels found and no map") | Pass `--label-map "0=favour,1=none,2=against"` (only if that is what the file means) |
| "More than one data file in data/test/" | Move extras out, or use `--file` |
| "Unknown label values: …" | Add them to `label_map`, or `--allow-unknown-labels` |
| "LLM-A unavailable" | Missing key, rate limit or no internet: fill `.env`, wait for the daily reset, or `--no-llm` |
| Many rows dropped | Empty or whitespace-only comments; the report header gives the count |
| Scores look odd | Wrong label column or map: re-run `--inspect` and compare the counts in the report header with the file |

## 8. Sources

Models, libraries and datasets used by this script are cited in `docs/REFERENCES.md`.
