# Stacking Ensemble: how it is built and trained

**Project:** EcoConsensus
**Companions:** `TRD.md` (settings), `LLM_PROMPT.md` (prompt), `APP_FLOW.md` (flows)

The hard limit is the **number of LLM requests per day**. Everything below is designed around it.

---

## 1. The idea in plain words

Think of four students (MuRIL, IndicBERT, LLM-A, LLM-B) and a teacher (the meta-model). Each student gives an answer. The teacher has watched the students on practice questions and learned who to trust for which kind of comment. On a new comment, the teacher reads the four answers and makes the final call.

This is called **stacking** (Wolpert, 1992). The students are the *base models*. The teacher is the *meta-model*.

## 2. What goes in and what comes out

### Base model outputs

| Model | Raw output | Written as features |
|---|---|---|
| MuRIL | 3 probabilities that sum to 1 (favour, none, against) | `muril_p0, muril_p1, muril_p2` |
| IndicBERT | 3 probabilities that sum to 1 | `indicbert_p0, indicbert_p1, indicbert_p2` |
| LLM-A | One label as text, parsed | `llm_a_favour, llm_a_none, llm_a_against` (one of them is 1, the others 0) |
| LLM-B | One label as text, parsed | `llm_b_favour, llm_b_none, llm_b_against` |

LLMs do not give trustworthy probabilities, so each LLM contributes a one-hot label. A one-hot label puts a 1 in the chosen class and 0 in the others.

### Meta-model input: one row of 12 numbers per comment

| Comment | muril p0,p1,p2 | indicbert p0,p1,p2 | llm_a (f,n,a) | llm_b (f,n,a) | true label |
|---|---|---|---|---|---|
| "it's all a hoax" | 0.30, 0.50, 0.20 | 0.20, 0.30, 0.50 | 0,0,1 | 0,0,1 | against |
| "great, more solar" | 0.80, 0.15, 0.05 | 0.60, 0.30, 0.10 | 1,0,0 | 0,1,0 | favour |

### Meta-model output

Three final probabilities (favour, none, against). The largest one is the prediction.

### What the meta-model is

A regularised multinomial **logistic regression** with `class_weight="balanced"`. It is small on purpose: with a few thousand rows a bigger model would memorise. It learns weights like "when IndicBERT says *against* and an LLM agrees, trust *against*". Those weights are also the explainability output in the report.

## 3. Why the meta-model needs its own data

If the teacher watches the students on questions they already memorised, the students look perfect, and the teacher learns "always trust the transformers". On new comments that fails. So the meta-model must learn from rows **the transformers never trained on**.

Three ways to get those rows were considered:

| Option | Idea | Cost | Chosen |
|---|---|---|---|
| **Hold-out slice** | Set aside about 15% of Pol-NLI before training. The transformers never see it | One training run per transformer; LLM calls only on the slice | **Yes** |
| K-fold out-of-fold | Train each transformer K times, each time leaving a part out | K× training; LLM calls on every row | No (too many requests) |
| Part of SemEval | Use some SemEval rows | Cheap, but SemEval stops being a clean test set | No |

Trade-off accepted: the transformers train on 85% of Pol-NLI, not 100%, and are **not retrained** afterwards (retraining would change their outputs and make the meta-model's lessons stale).

## 4. Data roles

| Split | Rows (source rows × 7 versions) | Transformers | LLMs | Meta-model |
|---|---|---|---|---|
| `train_base` | about 2,989 × 7 | **Train** | Few-shot examples only (if used) | Not used |
| `meta` | about 528 × 7 = 3,696 | Predict only | Predict | **Train** |
| `val` | 90 × 7 = 630 | Early stopping, tuning | Prompt comparison, batch-size pilot | Choose meta settings (for example `C`) in a small check |
| `semeval_test` | 565 × 7 = 3,955 | Predict only | Predict | Predict only: **final test** |
| Evaluator file | unknown | Predict only | Predict | Predict only |

Rules that protect the results:
1. All 7 versions of a comment are in the same split (grouped by `source_id`).
2. The prompt is frozen **before** any `semeval_test` or evaluator row is sent to an LLM.
3. Few-shot examples come from `train_base` only.
4. `semeval_test` and the evaluator file never fit anything.

## 5. Build and train: step by step

| Step | Script | Where | What happens | Output |
|---|---|---|---|---|
| S0 | `splitting.py` | Mac | Create `train_base`, `meta`, `val`, `semeval_test`; write split report | `data/processed/` |
| S1 | `train_transformer.py --model muril` | Colab | Train MuRIL with LoRA on `train_base`; early stopping on `val` | `models/finetuned/muril_lora/` |
| S2 | `train_transformer.py --model indicbert` | Colab (new session) | Same for IndicBERT | `models/finetuned/indicbert_lora/` |
| S3 | `run_llm_batch.py` (once per provider) | Anywhere | Prompt choice on `val`, then fill the cache for `meta` in batches. Resumes daily | `outputs/llm/cache/` |
| S4 | `train_inference.py --split meta` (also `val`, `semeval_test`) | Colab | Run both transformers on the split, join cached LLM answers, write the feature table | `outputs/ensemble/predictions/features_<split>.csv` |
| S5 | `train_meta.py` | Mac or Colab | Fit the meta-model with grouped cross-validation inside `meta` to pick `C`; fit the final model on all of `meta`; fit fallback variants; compare with voting | `models/finetuned/meta/` |
| S6 | `test.py` | Mac | Run all components on the test file, apply the meta-model, write reports | `test/` and `outputs/` |

S1 and S2 are independent of S3, so on a day when LLM quota is the bottleneck, run S3 in the background while Colab trains.

Cross-validation inside `meta` is grouped by `source_id` and stratified by label (for example 5 folds). The 7 versions of one comment are strongly related, so splitting them across folds would give a falsely high score.

## 6. Fallback meta-models

An LLM may be unavailable (limit reached, no key, outage). A meta-model trained with all 12 features cannot simply ignore missing ones, so several are trained once, on the same `meta` rows:

| Variant | Features used | Used when |
|---|---|---|
| `meta_all4` | 12 | Everything available |
| `meta_no_llm_a` | 9 | LLM-A unavailable |
| `meta_no_llm_b` | 9 | LLM-B unavailable |
| `meta_transformers` | 6 | Both LLMs unavailable |

`test.py` and `inference.py` pick the variant **per comment** and write which one was used into the report and the log. Both LLMs and one transformer missing (weights not found) is not handled by stacking: the run falls back to the TF-IDF baseline and says so.

## 7. Request budget

All numbers below are estimates for **one provider** (the two providers each need their own quota). They assume 7 versions for every Pol-NLI and SemEval comment, a 15% hold-out (528 source rows), `val` of 90 source rows, and 10% extra requests for retries. The script that computed them is a few lines of Python and the same formula goes in `run_llm_batch.py` as a pre-run estimate.

**Rows to send**

| Purpose | Rows |
|---|---|
| `meta` (3,696) + `val` (630) | 4,326 |
| `semeval_test` | 3,955 |

**Requests per provider**

| Batch size | Training side (`meta` + `val`) | `semeval_test` |
|---|---|---|
| 25 | 192 | 175 |
| 50 | 96 | 88 |
| 100 | 49 | 44 |

**Days needed at a given daily cap per provider**

| Daily cap | Batch 25 | Batch 50 | Batch 100 |
|---|---|---|---|
| 50 requests | 4 days + 4 days | 2 days + 2 days | 1 day + 1 day |
| 200 requests | 1 + 1 | 1 + 1 | 1 + 1 |
| 1,000 requests | 1 + 1 | 1 + 1 | 1 + 1 |

(Each cell is training side + `semeval_test`.) These runs can overlap with GPU training, so most of it fits inside the 5 days even at 50 requests per day, but only just: keep the batch at 50 or more in that case. A small share of each day's quota should stay free for the pilot and debugging.

**Tokens, not just requests.** Many providers also limit tokens per minute or per day. Rough input size per request is the system prompt plus `batch size × tokens per comment`. Hindi and Bengali comments are several times longer in tokens than English ones, so a 50-comment batch may reach several thousand tokens. Check each provider's limits when you pick it, and if you hit token limits, lower the batch size before anything else. The script does this automatically for a provider that returns a token-limit error.

**Ways to cut requests further if needed** (in this order):
1. Raise the batch size (to 75 or 100) after the pilot shows accuracy holds.
2. Send only some versions of each `meta` comment (for example English plus one Hindi and one Bengali variant). This cuts `meta` rows from 3,696 to about 1,584.
3. Shrink the hold-out from 15% to 10%.

## 8. Batch-size pilot (do this once, before the big runs)

Bigger batches save requests but can make an LLM skip items, shift answers by one position, or get lazier on the last items.

1. Take 100 comments from `val` (mixed variants).
2. Send them at batch sizes 10, 25, 50 and 100 to one provider (about 20 requests in total).
3. Compare, against the true labels: Macro-F1, number of missing or mismatched ids, and the accuracy of the last 10 items of each batch versus the first 10.
4. Pick the largest batch whose Macro-F1 is within noise of the smallest and which has no missing ids. Write the choice in `LLM_BATCH_SIZE`.

## 9. Per-comment cache and resuming

- The cache has one record per comment: `key → {label, status, prompt_version, provider, model, timestamp}`. The key hashes (provider, model, prompt version, claim, preprocessed comment).
- The batch size is not in the key, so a different batch size later reuses everything already answered.
- `run_llm_batch.py` first removes cached comments from the work list, then sends only the rest.
- When the daily budget is reached it stops, prints "N comments still missing for provider X", and the same command resumes tomorrow.
- A new prompt version means new keys. Do not edit the prompt after step S3 starts.

## 10. Checks before trusting the meta-model

- Class counts in `meta` are close to those in Pol-NLI train (the split report).
- LLM parse-error and `unavailable` rates are low and reported per provider.
- Meta-model cross-validated Macro-F1 is compared with: best single transformer, best single LLM, majority vote of the four, and soft vote of the two transformers. If the meta-model does not beat simple voting, say so.
- The learned weights are printed and sanity-checked (each component should have a positive weight on its own class).
- No row of `semeval_test` or `val` appears in `meta` (the split script already enforces this; the training script checks again).

## 11. What goes into the final reports

Per comment in `test.py` output: all component outputs, the meta-model variant used, and the final label. Per run: the weights of the meta-model, how often each fallback was used, LLM requests used, cache hits, parse errors.

## 12. Risks

| Risk | Mitigation |
|---|---|
| Meta slice class mix differs from the evaluator's data | Class-balanced weights; report per-class results |
| LLM answers drift if the provider silently updates the model | Model name and date stored in each cache record; rerun is possible because the cache is per comment |
| Few rows of `against` in `meta` | Balanced class weights; report bootstrap intervals |
| Free-tier outages on test day | Fallback variants; evaluator's file can run with `--no-llm` |
| Transformer overfits Pol-NLI wording | Early stopping on `val`; SemEval test shows the real gap |

## 13. References

Wolpert (1992) for stacking; scikit-learn (Pedregosa et al., 2011) for the logistic regression and cross-validation. Full list in `REFERENCES.md`.
