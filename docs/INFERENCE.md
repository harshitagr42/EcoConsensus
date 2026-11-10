# Using `inference.py`

Classify one comment, or a file of comments, toward the claim *"Climate change and global warming is a serious concern."* Works on macOS, Linux and Windows.

> This is the planned interface. The flags below are the specification the code will follow. If the code and this page ever differ, fix this page on the same day.

`inference.py` sends **one comment** to each LLM per call. To score a whole labelled file with batching and a full comparison, use `test.py` (see `TESTING.md`).

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
| `LLM_BATCH_SIZE` | Comments per request for `test.py` and training scripts (ignored by `inference.py`) |
| `HF_TOKEN` | Optional, for Hugging Face downloads |

No keys? The script still runs with the two transformers only (see `--no-llm`).

### Model files

`models/finetuned/` must contain `muril_lora/`, `indicbert_lora/`, `baselines/` and `meta/`. The LoRA adapters are small (about 5 MB each) and come with the repository. If a folder is missing, the script names it and stops (exit code 4).

---

## 2. Quick examples

Replace `python` with `python3` if your system needs it. Examples work in every shell unless marked.

```bash
# One comment (English)
python inference.py -t "We can't deny it, it's really happening."

# One comment, Hindi (macOS and Linux, any shell)
python inference.py -t "ग्लोबल वार्मिंग सिर्फ एक झूठ है।"

# A file
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

## 3. All flags

### Input

| Flag | Meaning |
|---|---|
| `-t TEXT`, `--text TEXT` | A single comment to classify |
| `-f PATH`, `--file PATH` | A file of comments (`.csv`, `.tsv`, `.json`, `.jsonl`, `.xlsx`, `.txt`). Use either `-t` or `-f`, not both |
| `-c TEXT`, `--claim TEXT` | Use a different claim. The models are trained for the default claim, so results for other claims are not reliable |
| `--id-col NAME` | Column holding the row id (default: auto-detect; falls back to row number) |
| `--text-col NAME` | Column holding the comment (default: auto-detect) |
| `--claim-col NAME` | Column holding a claim per row (default: none, use the fixed claim) |
| `--label-col NAME` | Column holding true labels (default: auto-detect; if found, metrics are printed) |
| `--label-map "A=favour,B=none,C=against"` | How the file's label values map to the three classes (`favour`, `none`, `against`). Needed for numeric labels |
| `--allow-unknown-labels` | Continue when some label values are not in the map; those rows are treated as unlabeled |
| `--delimiter CHAR` | Force a delimiter for `.csv` / `.tsv` / `.txt` |
| `--encoding NAME` | Force a text encoding (default: tries `utf-8-sig`, `utf-8`, `utf-16`, `cp1252`, `latin-1`) |

### Model selection

| Flag | Meaning |
|---|---|
| `-m MODE`, `--mode MODE` | `ensemble` (default), `transformers` (both transformers combined, no LLM), `muril`, `indicbert`, `llm-a`, `llm-b`, `tfidf`, `majority` |
| `--no-llm` | Do not call any LLM. The stacked model falls back to its transformers-only variant |
| `--require-llm` | Stop with exit code 5 if an LLM is unavailable instead of falling back |
| `--device DEVICE` | `auto` (default), `cpu`, `cuda` or `mps` |
| `--config PATH` | Use another config file (default `configs/base.yaml`) |

### Output

| Flag | Meaning |
|---|---|
| `-o DIR`, `--output-dir DIR` | Where logs and prediction files go (default `outputs/`) |
| `--label-style STYLE` | How labels are written: `ps` (default: `Favour`, `None`, `Against`), `file` (the spelling used in the input file's label column), or `numeric` (`0`, `1`, `2`) |
| `--json` | Print the result as JSON instead of text |
| `-v`, `--verbose` | Also print each component's probabilities and the meta-model variant |
| `-q`, `--quiet` | Print only the final label (one per line for files) |

### Other

| Flag | Meaning |
|---|---|
| `--version` | Print the version and exit |
| `-h`, `--help` | Print usage and exit |

---

## 4. What you see

One comment:

```text
Comment : ग्लोबल वार्मिंग सिर्फ एक झूठ है।
Claim   : Climate change and global warming is a serious concern.
Final   : Against
Votes   : MuRIL=Against  IndicBERT=Against  LLM-A=Against  LLM-B=None
Mode    : meta_all4
Log     : outputs/inference/logs/inference_20261112_101500.txt
```

(The values above are an example of the layout, not a real result.)

A file prints a short summary (rows read, rows skipped, label counts), the path of the predictions file, and, if the file had labels, Macro-F1, accuracy, per-class precision and recall, and the confusion matrix.

### Files written

| File | Content |
|---|---|
| `outputs/inference/logs/inference_<time>.txt` | Command, input pointer, preprocessing summary, config, each component's output, fallbacks, warnings |
| `outputs/inference/predictions/inference_<time>.csv` | `ID`, `Predicted_Label`, plus each component's vote |

Files are timestamped and never overwritten.

---

## 5. Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 2 | Bad usage, configuration or column error |
| 3 | Data error (file not found, unreadable, empty) |
| 4 | Model files missing |
| 5 | An LLM was unavailable and `--require-llm` was set |

## 6. Common problems

| Problem | Fix |
|---|---|
| `python` is the wrong version | Use `python3.13` (macOS/Linux) or `py -3.13` (Windows) when creating the virtual environment |
| Hindi or Bengali shows as `?` or boxes | Use a UTF-8 terminal and a font with those scripts; on Windows set `PYTHONUTF8=1` |
| "No API key for LLM-A" | Fill `.env`, or use `--no-llm` |
| First run is slow | The base models download once, then load from the cache |
| "Model files missing" | Check that `models/finetuned/` has the four folders listed above |
| Wrong column picked | Set `--text-col`, `--id-col`, `--label-col` explicitly |
| Numeric labels rejected | Pass `--label-map "0=favour,1=none,2=against"` (only if that is what the file means) |

## 7. Sources

Models, libraries and datasets used by this script are cited in `REFERENCES.md`.
