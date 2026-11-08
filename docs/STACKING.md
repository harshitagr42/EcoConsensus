# Stacking ensemble: how it is built and trained

**Companion to:** TRD.md. Read this when you build `src/training/train_inference.py` and `src/training/train_meta.py`.

## 1. The idea in one picture

Four **base models** each give an opinion on a comment. A small **meta-model** learns how much to trust each one and gives the final label.

```text
comment + claim ──► MuRIL (fine-tuned)      ──► 3 probabilities ─┐
                ──► IndicBERT (fine-tuned)  ──► 3 probabilities ─┤
                ──► LLM A (API, batched)    ──► 1 label ─────────┼──► meta-model ──► final label
                ──► LLM B (API, batched)    ──► 1 label ─────────┘
```

Think of a panel of four judges and a chair who has seen how often each judge was right on past cases, and weighs them accordingly.

## 2. Why this design fits your limits

| Limit | How the design handles it |
|---|---|
| Colab VRAM is about 15 GB | Transformers are trained **one at a time**; the feature step loads them **one at a time** |
| LLM APIs allow only so many requests per day | LLMs are never trained. They are called once per comment, in **batches**, with a per-entry **cache**; the work resumes next day |
| Little high-quality labelled data | Base models learn on the large lower-quality Pol-NLI set; the small high-quality SemEval set trains only the tiny meta-model |
| Test data is unknown | The meta-model only needs 12 numbers per comment, whatever the input file looked like |

## 3. Data roles (nothing is used twice by mistake)

| Data | Used for | Never used for |
|---|---|---|
| Pol-NLI subset (train) | Fine-tuning each transformer | Meta-model |
| Pol-NLI validation (90 sentences, all variants) | Early stopping of each transformer | Meta-model |
| SemEval **dev** (about 70%) | Meta-model training and its cross-validation | Fine-tuning transformers |
| SemEval **test** (about 30%) | Final comparison, touched once | Anything else |
| Evaluator file | Final prediction in `test.py` | Training |

**Why this avoids the usual stacking trap.** A meta-model must learn from base-model outputs on rows the base models never trained on. Because the transformers learn only from Pol-NLI and the meta-model learns only from SemEval, every feature the meta-model sees is already "out of sample". No cross-fold retraining of transformers is needed, which would not fit on Colab anyway. It also lets the meta-model learn the **domain shift** (Pol-NLI sentences versus tweets and comments): if a transformer is over-confident on tweets, the meta-model learns to discount it.

Caveat: the experiment phase picked fine-tuning methods on a small SemEval subset with its own train / val / test split, so some of those tweets later fall into SemEval dev and test. Only the method choice depends on them (no weights are carried over), so this is a mild selection effect; say so in the report. The evaluator file stays clean.

## 4. The three steps

### Step 1: train each transformer alone (GPU)

```text
python -m src.training.train_transformer --model muril     --method <chosen>
python -m src.training.train_transformer --model indicbert --method <chosen>
```

Run them as **two separate processes**, one after the other, so the first model's memory is released. Each saves `checkpoint_best` (best Pol-NLI validation Macro-F1), `ckpt_last`, and `run_info.json` under `models/finetuned/<model>__<method>/`.

### Step 2: build the meta-model features (`train_inference.py`)

```text
python -m src.training.train_inference --splits dev test --components muril indicbert llm_a llm_b
```

What it does, in order, and each part saves its own file so it can be run on different days or machines:

1. **MuRIL features (GPU):** load the best checkpoint, predict every dev and test row, write `outputs/features/<split>__muril.csv` (3 probabilities), then free the GPU.
2. **IndicBERT features (GPU):** same, one model at a time.
3. **LLM A features:** take rows not in the cache, send them in batches (default 20 comments per request), parse, cache each entry, write `outputs/features/<split>__llm_a.csv`. If the daily limit is hit, it stops cleanly and says how many rows remain; run it again tomorrow and it continues.
4. **LLM B features:** same.

Feature file columns: `row_key` (hash of claim and comment), `src_id`, `lang`, `variant`, `label`, then `p_favour, p_neutral, p_against` (transformers, in label-id order 0, 1, 2) or `llm_label` (LLMs). Rows are matched across files by `row_key`, so files can be built in any order.

### Step 3: train the meta-model (`train_meta.py`, CPU, seconds)

```text
python -m src.training.train_meta
```

1. Load the dev feature files and join them on `row_key`.
2. **Features per row (12 numbers):** MuRIL probabilities (3), IndicBERT probabilities (3), LLM A one-hot label (3), LLM B one-hot label (3).
3. **Model:** multinomial logistic regression with balanced class weights. With a few thousand rows and 12 inputs, anything bigger would overfit.
4. **Choose its regularisation strength (C)** by 5-fold cross-validation **grouped by `src_id`** (a tweet and all its translations stay in the same fold) on dev only. Report the cross-validated Macro-F1.
5. **Sanity comparisons on dev (same cross-validation):** best single transformer, best single LLM, simple average of probabilities, majority vote. If the meta-model does not beat simple voting, say so in the report.
6. **Fit on all dev**, save to `models/finetuned/meta/`.
7. **Fit one meta-model for every non-empty subset of the four components** (15 tiny models). These are the fallbacks.
8. Write a generated Markdown report to `outputs/metrics/` (never overwriting).
9. SemEval **test** is scored once, afterwards, by `test.py` or the final comparison script.

## 5. Fallbacks at test and inference time

If an LLM is missing for a row (no key, daily limit, unparseable answer), the system uses the meta-model fitted **without** that component, chosen from the 15 saved subsets. The output and log state which subset was used. If every LLM is missing, the transformers-only model is used. If everything but one transformer is missing, that transformer's own argmax is used.

## 6. Request budget (estimate; check your provider's current limits)

Assumptions: 565 SemEval source tweets, 70/30 dev/test split, batch size 20. Your real counts may differ slightly.

| Rows included | Dev rows | Test rows | Requests per LLM (dev + test) | Both LLMs |
|---|---|---|---|---|
| Hindi + Bengali variants only (6 per tweet) | about 2,370 | about 1,020 | about 170 | about 340 |
| Plus English (7 per tweet) | about 2,770 | about 1,190 | about 200 | about 400 |

Add roughly 10% for retries and single-item reruns. If one provider's daily allowance is below its share, split the run over several days; the cache makes this safe.

Ways to cut the budget if needed, most harmless first: skip English rows, use fewer SemEval dev rows for LLM features (the meta-model only has 12 inputs, so a few hundred dev rows can be enough), or raise the batch size while checking that answers do not get truncated. `test.py` on the evaluator file needs its own requests: rows divided by batch size, for each LLM.

During `inference.py`, only one entry is sent per call (a batch of one), cached by entry.

## 7. Failure cases

| Problem | What happens |
|---|---|
| LLM daily limit | Stop cleanly, save cache, tell you rows remaining |
| LLM returns invalid JSON | Retry once for the batch, then single items; still failing means `unavailable` |
| A GPU step runs out of memory | Lower batch size in the command; the run resumes with finished rows kept |
| Feature files disagree on row sets | `train_meta.py` lists the rows missing per component and trains the subset models on the common rows |
| Evaluator comment never seen by LLM cache | New request, cached afterwards |

## 8. What the report must show

Dev cross-validated results for meta-model, each single component, simple average and majority vote; SemEval test results for the same; per-class and per-language results; which fallback subsets were used and how often; and a plain statement of whether stacking helped.
