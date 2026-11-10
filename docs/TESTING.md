# Evaluator-day guide: `test.py`

You will get a test file, and you will not know its columns, label spellings or format. `test.py` is built so that you can configure it in about a minute and then get the full comparison.

> Planned interface. The flags and the `TEST_CONFIG` block below are the specification for the code.

---

## 1. The 60-second routine

1. **Put the file in `test/`** (the repo's `test/` folder). Keep only one data file there.
2. **Look at it:** `python test.py --inspect`. It prints the file's columns, the first 5 rows, the distinct values in the likely label column, and the configuration it would use. It runs no model.
3. **Fix the configuration if the guess is wrong**, either by editing the `TEST_CONFIG` block at the top of `test.py` or by adding flags to the command (flags win over the block, and the block wins over auto-detection).
4. **Run:** `python test.py` (add `--no-llm` if the internet or the API keys are not available).
5. **Read** `test/result.txt`. The prediction file is `test/predictions.csv`.

If something goes wrong, the message names the column, label value or file involved and says what to change.

---

## 2. The `TEST_CONFIG` block

This block sits at the very top of `test.py`. Edit the values you need; leave the rest.

```python
TEST_CONFIG = {
    "data_file":    None,        # None = the only data file in test/; or e.g. "comments.csv"
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

---

## 3. Flags

Every flag has the same effect as the matching `TEST_CONFIG` entry and overrides it.

| Flag | Meaning |
|---|---|
| `-f PATH`, `--file PATH` | Use this file instead of the one in `test/` |
| `--inspect` | Show columns, 5 sample rows, distinct label values and the planned config, then stop |
| `--id-col NAME` | Id column |
| `--text-col NAME` | Comment column |
| `--label-col NAME` | True-label column |
| `--claim-col NAME` | Per-row claim column |
| `--language-col NAME` | Language column |
| `--label-map "A=favour,B=none,C=against"` | Label mapping; adds to or replaces the block's map |
| `--allow-unknown-labels` | Keep going when some label values are not in the map (those rows are treated as unlabeled) |
| `--delimiter CHAR` | Force a delimiter |
| `--encoding NAME` | Force an encoding |
| `--claim TEXT` | Use another fixed claim |
| `--no-llm` | Do not call any LLM; the stacked model uses its transformers-only variant |
| `--require-llm` | Stop (exit 5) if an LLM is unavailable instead of falling back |
| `--llm-budget N` | Send at most N requests per provider in this run; the rest fall back |
| `--batch-size N` | Comments per LLM request (default from `LLM_BATCH_SIZE`, normally 50) |
| `--label-style STYLE` | Prediction labels: `ps` (default: `Favour`, `None`, `Against`), `file` (the spelling the test file uses), `numeric` (`0`, `1`, `2`) |
| `--device DEVICE` | `auto`, `cpu`, `cuda`, `mps` |
| `--config PATH` | Another config file |
| `-v`, `--verbose` / `-q`, `--quiet` | More or less console output |
| `--version`, `-h`, `--help` | Version, usage |

Exit codes: `0` success, `2` configuration or column error, `3` data error, `4` model files missing, `5` LLM unavailable with `--require-llm`.

### Setup and OS notes

Setup is identical to `INFERENCE.md` (virtual environment, `pip install -r requirements.txt`, `.env`). Run `python test.py` on macOS and Linux, `python test.py` or `py -3.13 test.py` on Windows. On Windows set `PYTHONUTF8=1` before running.

---

## 4. Typical test-file shapes and what to do

| What the file looks like | What to set |
|---|---|
| `ID,Target,Tweet,Stance` with FAVOR / AGAINST / NONE | Nothing; auto-detect handles it |
| `id,comment,label` with Favour / Against / None | Nothing |
| `comment_id,yt_comment,stance` with 0 / 1 / 2 | `--text-col yt_comment --id-col comment_id --label-col stance --label-map "0=favour,1=none,2=against"` (confirm the meaning first) |
| Only one column of comments, no header | Add `--text-col 0` (a number picks the column by position when there is no header) |
| Labels in Hindi or Bengali words | Add them to `label_map`, for example `"पक्ष": "favour"` |
| No label column | Nothing; the run is prediction-only and the report says so |
| Several sheets in an `.xlsx` | The first sheet is used; save the right sheet as CSV if needed |
| A claim column that differs by row | `--claim-col` only if you want the claim used (the models were trained for the default claim) |

---

## 5. What the run does

1. Finds the file, reads it (trying delimiters and encodings), applies your configuration.
2. Cleans every comment with the same preprocessing used in training.
3. Runs: majority baseline, TF-IDF baseline, MuRIL, IndicBERT.
4. Sends the comments to LLM-A and LLM-B in batches (answers are cached per comment, so a rerun costs nothing).
5. Applies the stacked meta-model comment by comment. If an LLM answer is missing, the matching fallback model is used and counted.
6. If labels exist: computes the metrics, picks the **best single transformer** and the **best single LLM** by Macro-F1 on this file, and compares everything.
7. Writes the report and the prediction file.

## 6. Where the results go

| File | Content |
|---|---|
| `test/result.txt` | The full report (latest). The previous one is moved to `test/archive/` first |
| `test/reports/result_<time>.md` | The same report as Markdown, timestamped |
| `test/predictions.csv` | `ID,Predicted_Label` for every row (previous one archived) |
| `outputs/test/logs/test_<time>.txt` | Exact command, configuration, per-step log, warnings |
| `outputs/test/predictions/test_<time>_detailed.csv` | Every component's output per comment |
| `outputs/test/metrics/test_<time>.json` | The numbers behind the report |

Nothing is ever overwritten without being kept first. All reports are produced by the script; no AI writes them.

## 7. Reading `result.txt`

1. **Header:** time, command, file, rows read, rows dropped (empty comments) and why, label counts.
2. **Comparison table:** Macro-F1 (first), accuracy, weighted-F1, MCC, kappa for: majority, TF-IDF, MuRIL, IndicBERT, best transformer, LLM-A, LLM-B, best LLM, **ensemble**. "Best" is chosen on this same file and the report says so.
3. **Per-class table:** precision, recall, F1 for Favour, None, Against, for every system.
4. **Confusion matrices:** rows are true labels, columns are predictions.
5. **By script group:** Devanagari, Bengali, Latin (Latin covers English and romanised text).
6. **Ensemble details:** how many comments used each meta-model variant, the learned weights, LLM requests used, cache hits, parse errors and `unavailable` counts.
7. **Error analysis:** the most common confusions and the first 20 misclassified comments with each component's vote.
8. **Environment and configuration:** library versions and the exact configuration used, including your `TEST_CONFIG` values.
9. **Warnings and limitations.**

With no labels, sections 2–5 and 7 are replaced by a prediction summary: label counts per system and how often the systems agree.

## 8. Troubleshooting

| Message or symptom | Cause | Fix |
|---|---|---|
| "More than one data file in test/" | Several candidates | Move extras out, or use `--file` |
| "Comment column not found. Columns found: …" | Auto-detect missed it | `--text-col NAME` |
| "Unknown label values: …" | Spellings not in `label_map` | Add them, or `--allow-unknown-labels` |
| "Numeric labels found and no map" | Safety stop | Add a map after checking what the numbers mean |
| Many rows dropped | Empty or whitespace-only comments | Check the file; the report lists the count |
| "LLM-A unavailable" | Missing key, rate limit or no internet | Fill `.env`, wait for the daily reset, or `--no-llm` |
| "Model files missing: …" | `models/finetuned/` incomplete | Restore the named folder |
| Scores look odd | Wrong label column or map | Re-run `--inspect`; compare the counts in the header with the file |

## 9. Sources

All models, libraries and datasets are cited in `REFERENCES.md`.
