# Running `inference.py` and `test.py`

**Status:** this is the interface specification. The scripts are written after the experiment phase, so flags below are the contract they must follow; update this file if a flag changes.

## 1. Install (once)

You need Python 3.13 or newer if possible (3.11 or 3.12 also work if a dependency lacks 3.13 support). The trained weights must be in `models/finetuned/`, and `.env` must exist (copy `.env.example` and fill in the LLM keys; without keys the system uses fallbacks).

**macOS or Linux (Terminal)**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then edit .env
```

**Windows (PowerShell)**
```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env   # then edit .env
```
If PowerShell blocks activation: `Set-ExecutionPolicy -Scope Process RemoteSigned`.

**Windows (Command Prompt)**: `.venv\Scripts\activate.bat`.

Run commands from the repository root. On Windows use `py` or `python` instead of `python3`, and `\` or `/` in paths (quote paths with spaces).

## 2. `inference.py`: one comment (or a small file)

```bash
python3 inference.py --comment "ये बहुत गंभीर समस्या है"
python3 inference.py -c "জলবায়ু পরিবর্তন একটা মিথ্যা" --mode transformer
python3 inference.py --file my_comments.csv --comment-col text
```

| Flag | Short | Meaning | Default |
|---|---|---|---|
| `--comment TEXT` | `-c` | The comment to classify | none |
| `--file PATH` | `-f` | A csv / tsv / json / jsonl file instead of one comment | none |
| `--comment-col NAME` | | Column holding comments in `--file` | auto-detect |
| `--claim TEXT` | | Override the fixed claim | the fixed claim |
| `--mode MODE` | `-m` | `ensemble`, `transformer` (both transformers, no LLM), `llm`, `baseline` | `ensemble` |
| `--no-llm` | | Never call an LLM API (uses the transformers-only meta-model) | off |
| `--no-cache` | | Ignore the LLM cache for this run | off |
| `--output-dir DIR` | `-o` | Where the log and prediction file go | `outputs/` |
| `--json` | | Print the result as JSON | off |
| `--quiet` | `-q` | Print only the final label | off |
| `--verbose` | `-v` | Also print per-model probabilities and raw LLM answers | off |
| `--help` | `-h` | Show help | |

Exactly one of `--comment` and `--file` is needed. A single comment is sent to each LLM as a batch of one and cached by entry.

Output: the label (`favour`, `neutral`, `against`), each model's vote, which ensemble was used (full, or a fallback subset), and the log path `outputs/logs/inference_<timestamp>.txt`. Exit code 0 on success, 2 on bad usage, 3 when weights are missing.

## 3. `test.py`: evaluator dataset

1. Put the evaluator's file in `test/`.
2. Check the **CONFIG block at the top of `test.py`**: column names, label map, claim mode. If the file looks like the examples, nothing needs changing.
3. Run:
```bash
python3 test.py
python3 test.py --file test/hidden.csv --comment-col Tweet --label-col Stance --id-col ID
```

| Flag | Meaning | Default |
|---|---|---|
| `--file PATH`, `-f` | Dataset file | first supported file in `test/` |
| `--encoding NAME` | File encoding | `utf-8-sig` |
| `--delimiter CHAR` | Separator (`"\t"` for tab) | sniffed |
| `--comment-col NAME` | Comment text column | auto-detect |
| `--claim-col NAME` | Claim column | none (fixed claim) |
| `--label-col NAME` | Gold label column | auto-detect; none means prediction-only |
| `--id-col NAME` | ID column | row number from 1 |
| `--lang-col NAME` | Language column | detect from script |
| `--use-file-claim` | Use the file's claim text instead of the fixed claim | off |
| `--label-map "a=favour,b=neutral,c=against"` | Extra label spellings or numbers | built-in map |
| `--no-llm` | Do not call LLMs | off |
| `--cache-only` | Use only cached LLM answers; missing rows trigger fallbacks | off |
| `--out-dir DIR` | Where results go | `test/` |
| `--quiet`, `-q` / `--verbose`, `-v` | Less or more printing | |
| `--help`, `-h` | Help | |

Flags override the CONFIG block.

Outputs (never overwritten; each run adds timestamped files):
- `test/result_<timestamp>.txt` and `.md`: full report with all metrics and the comparison of Majority, TF-IDF, best transformer, best LLM and the ensemble
- `test/predictions_<timestamp>.csv`: `ID,Predicted Label`
- `outputs/logs/`, `outputs/metrics/`, `outputs/predictions/`

## 4. Common problems

| Message | Fix |
|---|---|
| `Could not find a comment column. Columns found: ...` | Use `--comment-col` or set `COMMENT_COLUMN` |
| `Unknown label values: ...` | Add them with `--label-map` or `LABEL_MAP`; those rows are predicted but not scored |
| `Weights not found in models/finetuned/...` | Run the final build steps or copy the weights there |
| `LLM A unavailable, using fallback` | Check `.env` key and daily limit; results remain valid, report names the fallback |
| Garbled Hindi or Bengali text | Wrong encoding; try `--encoding utf-8` or `utf-16` |
| Terminal shows boxes instead of Indic text | Terminal font; the files are still correct |
