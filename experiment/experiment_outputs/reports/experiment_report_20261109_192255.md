# Fine-tuning method comparison report

Generated 2026-11-09 19:22:55 by the experiment notebook (deterministic script, no AI). Experiment: `finetune_method_comparison`. Settings hash set: 93 runs/baselines.

## 1. Summary

Methods are **ranked on `val`** (validation split of the SemEval subset, Macro-F1, mean ± sample std over seeds). The best epoch inside each run is chosen on `val`. The held-out `test` split is shown only for the recommended methods (section 5) and never used to rank.

- **muril**: highest `val` Macro-F1 = **lora (r16_a32_qkv_lr0.001)** (0.632 ± 0.011). Recommended by the parsimony rule = **lora (r16_a32_qkv_lr0.001)** (0.632 ± 0.011, 1,477,635 trainable parameters). Methods within 0.011 of the best: lora.
- **indicbert**: highest `val` Macro-F1 = **qlora_4bit (r16_a32_qv_lr0.001)** (0.654 ± 0.021). Recommended by the parsimony rule = **lora (r16_a32_qv_lr0.001)** (0.642 ± 0.028, 1,182,723 trainable parameters). Methods within 0.021 of the best: last2, last4, lora, qlora_4bit.
- Reference: TF-IDF + LR scores 0.577 on `val`; the best transformer method (0.654) beats it on this set.

**Parsimony rule:** take the method with the best mean score, then, among all methods of the same model whose mean is within max(1 std of the best, 0.010) of it, recommend the one with the fewest trainable parameters. Differences inside that band are noise on this dataset size.

## 2. Data

SemEval-2016 Task 6 subset: 300 of 564 tweets (all tweets of ['against'] kept, the rest stratified), each in English + 3 Hindi + 3 Bengali variants. Frozen split `1af4afea20` (seed 2024), split by tweet: train 0.6 / val 0.2 / test 0.20; training cap none, variant policy `all`, claim mode `fixed`. Labels: 0 = favour, 1 = neutral, 2 = against. Rows dropped for empty comments: {'semeval': 0}.

| split | language | rows | favour | neutral | against |
|---|---|---|---|---|---|
| train | bengali | 540 | 306 | 186 | 48 |
| train | english | 180 | 102 | 62 | 16 |
| train | hindi | 540 | 306 | 186 | 48 |
| **train** | **all (180 tweets)** | **1260** | **714** | **434** | **112** |
| val | bengali | 180 | 102 | 63 | 15 |
| val | english | 60 | 34 | 21 | 5 |
| val | hindi | 180 | 102 | 63 | 15 |
| **val** | **all (60 tweets)** | **420** | **238** | **147** | **35** |
| test | bengali | 180 | 105 | 60 | 15 |
| test | english | 60 | 35 | 20 | 5 |
| test | hindi | 180 | 105 | 60 | 15 |
| **test** | **all (60 tweets)** | **420** | **245** | **140** | **35** |

> Data note: semeval: 300 of 564 tweets, 7 language/variant groups: bengali/banglish, bengali/bengali_code_mix, bengali/pure_bengali, english/original, hindi/hindi_code_mix, hindi/hinglish, hindi/pure_hindi

Token lengths (sentence pair) and share of `[UNK]` tokens per language group:

| model | checkpoint | p50 | p95 | p99 | max | truncated at 128 | [UNK] rate |
|---|---|---|---|---|---|---|---|
| muril | google/muril-base-cased | 39 | 51 | 58 | 70 | 0.0% | bengali: 0.00%, english: 0.00%, hindi: 0.00% |
| indicbert | ai4bharat/IndicBERTv2-MLM-only | 39 | 51 | 58 | 66 | 0.0% | bengali: 0.00%, english: 0.00%, hindi: 0.00% |

## 3. Method comparison

All numbers: mean ± sample std over seeds [42, 43] (n = seeds that finished). Rank set = `val`.

| model | method | rank | n | val Macro-F1 | val accuracy | val F1 against | trainable params | % trainable | train sec | peak mem MB | best epoch | per-seed F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| muril | lora (r16_a32_qkv_lr0.001) (best) **(recommended)** | 1 | 2 | 0.632 ± 0.011 | 0.793 ± 0.007 | 0.248 ± 0.016 | 1,477,635 | 0.62% | 14 | 1498 | 2.0 | 0.639 / 0.624 |
| muril | qlora_4bit (r16_a32_qkv_lr0.001) | 2 | 2 | 0.614 ± 0.037 | 0.777 ± 0.035 | 0.225 ± 0.018 | 1,477,635 | 0.62% | 24 | 1130 | 2.0 | 0.640 / 0.588 |
| muril | last4 | 3 | 2 | 0.586 ± 0.019 | 0.736 ± 0.034 | 0.257 ± 0.003 | 28,944,387 | 12.18% | 9 | 1381 | 1.5 | 0.572 / 0.600 |
| muril | last2 | 4 | 2 | 0.543 ± 0.004 | 0.680 ± 0.015 | 0.243 ± 0.030 | 14,768,643 | 6.22% | 7 | 1169 | 1.5 | 0.540 / 0.546 |
| muril | head_only | 5 | 2 | 0.360 ± 0.109 | 0.632 ± 0.062 | 0.000 ± 0.000 | 592,899 | 0.25% | 4 | 998 | 2.5 | 0.436 / 0.283 |
| indicbert | qlora_4bit (r16_a32_qv_lr0.001) (best) | 1 | 2 | 0.654 ± 0.021 | 0.812 ± 0.010 | 0.289 ± 0.062 | 1,182,723 | 0.42% | 21 | 1216 | 1.5 | 0.669 / 0.639 |
| indicbert | lora (r16_a32_qv_lr0.001) **(recommended)** | 2 | 2 | 0.642 ± 0.028 | 0.800 ± 0.027 | 0.285 ± 0.037 | 1,182,723 | 0.42% | 12 | 1599 | 2.5 | 0.622 / 0.662 |
| indicbert | last4 | 3 | 2 | 0.642 ± 0.007 | 0.793 ± 0.010 | 0.292 ± 0.059 | 28,944,387 | 10.41% | 9 | 1528 | 1.5 | 0.646 / 0.637 |
| indicbert | last2 | 4 | 2 | 0.640 ± 0.016 | 0.783 ± 0.027 | 0.326 ± 0.011 | 14,768,643 | 5.31% | 6 | 1320 | 1.5 | 0.629 / 0.652 |
| indicbert | head_only | 5 | 2 | 0.596 ± 0.006 | 0.729 ± 0.057 | 0.278 ± 0.048 | 592,899 | 0.21% | 5 | 1152 | 2.5 | 0.601 / 0.592 |

### LoRA / QLoRA hyperparameter search (stage 1)

Grid `{'r': [8, 16, 32], 'alpha_mult': [1, 2], 'targets': ['qv', 'qkv', 'all'], 'lr': [0.0003, 0.001]}` (36 combinations), seeds [1], scored on `val`. alpha = r x alpha_mult. Target sets: `{'qv': ['query', 'value'], 'qkv': ['query', 'key', 'value'], 'all': ['query', 'key', 'value', 'attention.output.dense', 'intermediate.dense', 'output.dense']}`. Pick rule: best mean val Macro-F1, then the fewest trainable parameters within max(1 std, 0.010) of it. Stage 2 (table above) re-runs the pick with seeds [42, 43]. QLoRA hyperparameters: `from_lora`.

**muril / lora**: best `r16_a32_qkv_lr0.001`, picked **`r16_a32_qkv_lr0.001`** (band 0.010).

| combination | r | alpha | targets | lr | trainable params | n | val Macro-F1 | val F1 against |
|---|---|---|---|---|---|---|---|---|
| r16_a32_qkv_lr0.001 **(picked)** | 16 | 32 | qkv | 0.001 | 1,477,635 | 1 | 0.619 | 0.241 |
| r8_a16_qv_lr0.0003 | 8 | 16 | qv | 0.0003 | 887,811 | 1 | 0.599 | 0.250 |
| r8_a16_qkv_lr0.001 | 8 | 16 | qkv | 0.001 | 1,035,267 | 1 | 0.598 | 0.212 |
| r32_a64_qkv_lr0.001 | 32 | 64 | qkv | 0.001 | 2,362,371 | 1 | 0.598 | 0.206 |
| r32_a64_qkv_lr0.0003 | 32 | 64 | qkv | 0.0003 | 2,362,371 | 1 | 0.596 | 0.219 |
| r8_a8_qkv_lr0.001 | 8 | 8 | qkv | 0.001 | 1,035,267 | 1 | 0.595 | 0.222 |
| r8_a8_qv_lr0.0003 | 8 | 8 | qv | 0.0003 | 887,811 | 1 | 0.594 | 0.242 |
| r32_a64_qv_lr0.0003 | 32 | 64 | qv | 0.0003 | 1,772,547 | 1 | 0.594 | 0.229 |
| r8_a16_qv_lr0.001 | 8 | 16 | qv | 0.001 | 887,811 | 1 | 0.592 | 0.195 |
| r8_a16_all_lr0.0003 | 8 | 16 | all | 0.0003 | 1,920,003 | 1 | 0.590 | 0.233 |
| r32_a32_qv_lr0.0003 | 32 | 32 | qv | 0.0003 | 1,772,547 | 1 | 0.588 | 0.242 |
| r32_a32_qkv_lr0.001 | 32 | 32 | qkv | 0.001 | 2,362,371 | 1 | 0.587 | 0.209 |
| r32_a64_all_lr0.0003 | 32 | 64 | all | 0.0003 | 5,901,315 | 1 | 0.587 | 0.226 |
| r16_a16_qkv_lr0.001 | 16 | 16 | qkv | 0.001 | 1,477,635 | 1 | 0.586 | 0.222 |
| r32_a64_qv_lr0.001 | 32 | 64 | qv | 0.001 | 1,772,547 | 1 | 0.586 | 0.189 |
| r16_a16_all_lr0.0003 | 16 | 16 | all | 0.0003 | 3,247,107 | 1 | 0.581 | 0.241 |
| r8_a8_all_lr0.0003 | 8 | 8 | all | 0.0003 | 1,920,003 | 1 | 0.581 | 0.208 |
| r16_a32_qkv_lr0.0003 | 16 | 32 | qkv | 0.0003 | 1,477,635 | 1 | 0.575 | 0.242 |
| r16_a16_qkv_lr0.0003 | 16 | 16 | qkv | 0.0003 | 1,477,635 | 1 | 0.573 | 0.216 |
| r8_a16_qkv_lr0.0003 | 8 | 16 | qkv | 0.0003 | 1,035,267 | 1 | 0.572 | 0.184 |
| r32_a32_qkv_lr0.0003 | 32 | 32 | qkv | 0.0003 | 2,362,371 | 1 | 0.571 | 0.219 |
| r16_a32_all_lr0.0003 | 16 | 32 | all | 0.0003 | 3,247,107 | 1 | 0.570 | 0.209 |
| r32_a32_all_lr0.001 | 32 | 32 | all | 0.001 | 5,901,315 | 1 | 0.567 | 0.203 |
| r16_a32_qv_lr0.0003 | 16 | 32 | qv | 0.0003 | 1,182,723 | 1 | 0.567 | 0.189 |
| r16_a16_qv_lr0.001 | 16 | 16 | qv | 0.001 | 1,182,723 | 1 | 0.565 | 0.161 |
| r8_a16_all_lr0.001 | 8 | 16 | all | 0.001 | 1,920,003 | 1 | 0.565 | 0.159 |
| r8_a8_qv_lr0.001 | 8 | 8 | qv | 0.001 | 887,811 | 1 | 0.564 | 0.182 |
| r16_a32_qv_lr0.001 | 16 | 32 | qv | 0.001 | 1,182,723 | 1 | 0.563 | 0.169 |
| r16_a32_all_lr0.001 | 16 | 32 | all | 0.001 | 3,247,107 | 1 | 0.560 | 0.203 |
| r8_a8_all_lr0.001 | 8 | 8 | all | 0.001 | 1,920,003 | 1 | 0.560 | 0.192 |
| r32_a32_qv_lr0.001 | 32 | 32 | qv | 0.001 | 1,772,547 | 1 | 0.558 | 0.151 |
| r16_a16_qv_lr0.0003 | 16 | 16 | qv | 0.0003 | 1,182,723 | 1 | 0.555 | 0.229 |
| r32_a32_all_lr0.0003 | 32 | 32 | all | 0.0003 | 5,901,315 | 1 | 0.555 | 0.200 |
| r8_a8_qkv_lr0.0003 | 8 | 8 | qkv | 0.0003 | 1,035,267 | 1 | 0.554 | 0.190 |
| r16_a16_all_lr0.001 | 16 | 16 | all | 0.001 | 3,247,107 | 1 | 0.530 | 0.197 |
| r32_a64_all_lr0.001 | 32 | 64 | all | 0.001 | 5,901,315 | 1 | 0.241 | 0.000 |

**indicbert / lora**: best `r16_a32_qkv_lr0.001`, picked **`r16_a32_qv_lr0.001`** (band 0.010).

| combination | r | alpha | targets | lr | trainable params | n | val Macro-F1 | val F1 against |
|---|---|---|---|---|---|---|---|---|
| r16_a32_qkv_lr0.001 | 16 | 32 | qkv | 0.001 | 1,477,635 | 1 | 0.654 | 0.275 |
| r32_a32_qkv_lr0.001 | 32 | 32 | qkv | 0.001 | 2,362,371 | 1 | 0.648 | 0.255 |
| r32_a64_qkv_lr0.001 | 32 | 64 | qkv | 0.001 | 2,362,371 | 1 | 0.645 | 0.286 |
| r16_a32_qv_lr0.001 **(picked)** | 16 | 32 | qv | 0.001 | 1,182,723 | 1 | 0.644 | 0.237 |
| r32_a64_qv_lr0.001 | 32 | 64 | qv | 0.001 | 1,772,547 | 1 | 0.641 | 0.241 |
| r16_a16_all_lr0.001 | 16 | 16 | all | 0.001 | 3,247,107 | 1 | 0.638 | 0.237 |
| r32_a32_qv_lr0.001 | 32 | 32 | qv | 0.001 | 1,772,547 | 1 | 0.637 | 0.237 |
| r16_a32_all_lr0.001 | 16 | 32 | all | 0.001 | 3,247,107 | 1 | 0.630 | 0.250 |
| r16_a16_qv_lr0.001 | 16 | 16 | qv | 0.001 | 1,182,723 | 1 | 0.628 | 0.226 |
| r8_a16_qkv_lr0.001 | 8 | 16 | qkv | 0.001 | 1,035,267 | 1 | 0.623 | 0.219 |
| r8_a16_all_lr0.001 | 8 | 16 | all | 0.001 | 1,920,003 | 1 | 0.620 | 0.215 |
| r16_a32_all_lr0.0003 | 16 | 32 | all | 0.0003 | 3,247,107 | 1 | 0.615 | 0.230 |
| r8_a16_qv_lr0.001 | 8 | 16 | qv | 0.001 | 887,811 | 1 | 0.613 | 0.219 |
| r16_a16_qkv_lr0.0003 | 16 | 16 | qkv | 0.0003 | 1,477,635 | 1 | 0.608 | 0.226 |
| r8_a8_qv_lr0.001 | 8 | 8 | qv | 0.001 | 887,811 | 1 | 0.608 | 0.206 |
| r8_a16_qv_lr0.0003 | 8 | 16 | qv | 0.0003 | 887,811 | 1 | 0.605 | 0.212 |
| r8_a16_qkv_lr0.0003 | 8 | 16 | qkv | 0.0003 | 1,035,267 | 1 | 0.603 | 0.219 |
| r32_a64_qv_lr0.0003 | 32 | 64 | qv | 0.0003 | 1,772,547 | 1 | 0.602 | 0.197 |
| r16_a32_qv_lr0.0003 | 16 | 32 | qv | 0.0003 | 1,182,723 | 1 | 0.602 | 0.192 |
| r16_a16_qkv_lr0.001 | 16 | 16 | qkv | 0.001 | 1,477,635 | 1 | 0.601 | 0.197 |
| r8_a8_qkv_lr0.001 | 8 | 8 | qkv | 0.001 | 1,035,267 | 1 | 0.600 | 0.177 |
| r8_a8_all_lr0.0003 | 8 | 8 | all | 0.0003 | 1,920,003 | 1 | 0.600 | 0.246 |
| r32_a64_all_lr0.0003 | 32 | 64 | all | 0.0003 | 5,901,315 | 1 | 0.597 | 0.215 |
| r32_a32_qv_lr0.0003 | 32 | 32 | qv | 0.0003 | 1,772,547 | 1 | 0.593 | 0.182 |
| r8_a8_qkv_lr0.0003 | 8 | 8 | qkv | 0.0003 | 1,035,267 | 1 | 0.592 | 0.203 |
| r8_a8_qv_lr0.0003 | 8 | 8 | qv | 0.0003 | 887,811 | 1 | 0.592 | 0.184 |
| r16_a16_qv_lr0.0003 | 16 | 16 | qv | 0.0003 | 1,182,723 | 1 | 0.591 | 0.182 |
| r32_a64_qkv_lr0.0003 | 32 | 64 | qkv | 0.0003 | 2,362,371 | 1 | 0.590 | 0.173 |
| r16_a32_qkv_lr0.0003 | 16 | 32 | qkv | 0.0003 | 1,477,635 | 1 | 0.590 | 0.200 |
| r16_a16_all_lr0.0003 | 16 | 16 | all | 0.0003 | 3,247,107 | 1 | 0.585 | 0.173 |
| r32_a32_qkv_lr0.0003 | 32 | 32 | qkv | 0.0003 | 2,362,371 | 1 | 0.583 | 0.189 |
| r8_a16_all_lr0.0003 | 8 | 16 | all | 0.0003 | 1,920,003 | 1 | 0.578 | 0.169 |
| r8_a8_all_lr0.001 | 8 | 8 | all | 0.001 | 1,920,003 | 1 | 0.576 | 0.184 |
| r32_a32_all_lr0.0003 | 32 | 32 | all | 0.0003 | 5,901,315 | 1 | 0.576 | 0.170 |
| r32_a64_all_lr0.001 | 32 | 64 | all | 0.001 | 5,901,315 | 1 | 0.572 | 0.169 |
| r32_a32_all_lr0.001 | 32 | 32 | all | 0.001 | 5,901,315 | 1 | 0.288 | 0.121 |

## 4. Reference baselines (same training split, same sets)

| baseline | settings | val Macro-F1 | val accuracy | val F1 against |
|---|---|---|---|---|
| majority | {'label': 'favour'} | 0.241 | 0.567 | 0.000 |
| tfidf_lr | {'C': 1, 'chosen_on': 'val'} | 0.577 | 0.695 | 0.333 |

## 5. Held-out check on `test` (recommended method per model)

Computed after the choice was made by the rules above. Do not go back and change the choice because of these numbers.

| system | Macro-F1 | accuracy | weighted-F1 | MCC | kappa | F1 favour | F1 neutral | F1 against |
|---|---|---|---|---|---|---|---|---|
| muril / lora | 0.564 ± 0.016 | 0.721 ± 0.007 | 0.711 ± 0.010 | 0.473 ± 0.026 | 0.469 ± 0.030 | 0.798 ± 0.012 | 0.685 ± 0.049 | 0.207 ± 0.010 |
| indicbert / lora | 0.581 ± 0.079 | 0.757 ± 0.003 | 0.736 ± 0.010 | 0.530 ± 0.005 | 0.524 ± 0.004 | 0.818 ± 0.001 | 0.728 ± 0.041 | 0.196 ± 0.277 |
| baseline / majority | 0.246 | 0.583 | 0.430 | 0.000 | 0.000 | 0.737 | 0.000 | 0.000 |
| baseline / tfidf_lr | 0.395 | 0.598 | 0.570 | 0.203 | 0.200 | 0.700 | 0.484 | 0.000 |

Confusion matrix, muril / lora (rows = true, columns = predicted, summed over 2 seed(s)):

| true \ pred | favour | neutral | against |
|---|---|---|---|
| favour | 404 | 66 | 20 |
| neutral | 76 | 190 | 14 |
| against | 42 | 16 | 12 |

Confusion matrix, indicbert / lora (rows = true, columns = predicted, summed over 2 seed(s)):

| true \ pred | favour | neutral | against |
|---|---|---|---|
| favour | 423 | 60 | 7 |
| neutral | 75 | 203 | 2 |
| against | 46 | 14 | 10 |

## 6. Macro-F1 by language and by script variant (`val`)

| model | method | val bengali | val english | val hindi | test bengali (recommended only) | test english (recommended only) | test hindi (recommended only) |
|---|---|---|---|---|---|---|---|
| muril | head_only | 0.390 ± 0.108 | 0.373 ± 0.097 | 0.318 ± 0.109 | - | - | - |
| muril | last2 | 0.540 ± 0.002 | 0.525 ± 0.019 | 0.552 ± 0.002 | - | - | - |
| muril | last4 | 0.583 ± 0.009 | 0.593 ± 0.003 | 0.588 ± 0.038 | - | - | - |
| muril | lora | 0.629 ± 0.015 | 0.652 ± 0.007 | 0.629 ± 0.012 | 0.544 ± 0.019 | 0.575 ± 0.005 | 0.580 ± 0.017 |
| muril | qlora_4bit | 0.619 ± 0.030 | 0.621 ± 0.023 | 0.607 ± 0.048 | - | - | - |
| indicbert | head_only | 0.582 ± 0.001 | 0.615 ± 0.032 | 0.603 ± 0.004 | - | - | - |
| indicbert | last2 | 0.616 ± 0.001 | 0.651 ± 0.027 | 0.661 ± 0.027 | - | - | - |
| indicbert | last4 | 0.629 ± 0.002 | 0.648 ± 0.012 | 0.653 ± 0.012 | - | - | - |
| indicbert | lora | 0.639 ± 0.018 | 0.631 ± 0.046 | 0.650 ± 0.034 | 0.585 ± 0.069 | 0.563 ± 0.050 | 0.582 ± 0.098 |
| indicbert | qlora_4bit | 0.638 ± 0.008 | 0.647 ± 0.020 | 0.672 ± 0.033 | - | - | - |
| baseline | majority | 0.241 | 0.241 | 0.241 | - | - | - |
| baseline | tfidf_lr | 0.605 | 0.547 | 0.557 | - | - | - |

By script variant (`val`; each column has the same tweets, so differences come from the text form):

| model | method | bengali/banglish | bengali/bengali_code_mix | bengali/pure_bengali | english/original | hindi/hindi_code_mix | hindi/hinglish | hindi/pure_hindi |
|---|---|---|---|---|---|---|---|---|
| muril | head_only | 0.370 ± 0.136 | 0.416 ± 0.081 | 0.382 ± 0.110 | 0.373 ± 0.097 | 0.300 ± 0.083 | 0.354 ± 0.160 | 0.287 ± 0.065 |
| muril | last2 | 0.524 ± 0.011 | 0.565 ± 0.024 | 0.531 ± 0.005 | 0.525 ± 0.019 | 0.565 ± 0.001 | 0.523 ± 0.007 | 0.572 ± 0.005 |
| muril | last4 | 0.576 ± 0.013 | 0.611 ± 0.004 | 0.563 ± 0.009 | 0.593 ± 0.003 | 0.612 ± 0.052 | 0.574 ± 0.008 | 0.581 ± 0.058 |
| muril | lora | 0.628 ± 0.017 | 0.646 ± 0.024 | 0.614 ± 0.035 | 0.652 ± 0.007 | 0.647 ± 0.018 | 0.641 ± 0.020 | 0.601 ± 0.037 |
| muril | qlora_4bit | 0.634 ± 0.009 | 0.630 ± 0.031 | 0.598 ± 0.060 | 0.621 ± 0.023 | 0.605 ± 0.046 | 0.644 ± 0.031 | 0.580 ± 0.059 |
| indicbert | head_only | 0.523 ± 0.017 | 0.605 ± 0.029 | 0.611 ± 0.041 | 0.615 ± 0.032 | 0.637 ± 0.044 | 0.551 ± 0.045 | 0.606 ± 0.010 |
| indicbert | last2 | 0.600 ± 0.006 | 0.614 ± 0.005 | 0.634 ± 0.004 | 0.651 ± 0.027 | 0.668 ± 0.036 | 0.651 ± 0.008 | 0.665 ± 0.055 |
| indicbert | last4 | 0.626 ± 0.012 | 0.635 ± 0.005 | 0.626 ± 0.011 | 0.648 ± 0.012 | 0.667 ± 0.013 | 0.662 ± 0.011 | 0.631 ± 0.036 |
| indicbert | lora | 0.657 ± 0.003 | 0.641 ± 0.035 | 0.623 ± 0.020 | 0.631 ± 0.046 | 0.635 ± 0.043 | 0.692 ± 0.017 | 0.626 ± 0.037 |
| indicbert | qlora_4bit | 0.632 ± 0.009 | 0.653 ± 0.002 | 0.632 ± 0.031 | 0.647 ± 0.020 | 0.648 ± 0.022 | 0.716 ± 0.046 | 0.648 ± 0.022 |
| baseline | majority | 0.241 | 0.241 | 0.241 | 0.241 | 0.241 | 0.241 | 0.241 |
| baseline | tfidf_lr | 0.605 | 0.615 | 0.596 | 0.547 | 0.556 | 0.557 | 0.559 |

## 7. Learning curves (first seed of each method)

| model | method | seed | train loss per epoch | val Macro-F1 per epoch | val accuracy per epoch |
|---|---|---|---|---|---|
| muril | head_only | 42 | 1.099 / 1.092 / 1.091 | 0.241 / 0.241 / 0.436 | 0.567 / 0.567 / 0.676 |
| muril | last2 | 42 | 1.082 / 0.612 / 0.311 | 0.540 / 0.516 / 0.496 | 0.669 / 0.652 / 0.629 |
| muril | last4 | 42 | 1.022 / 0.312 / 0.063 | 0.572 / 0.525 / 0.569 | 0.712 / 0.645 / 0.714 |
| muril | lora (r16_a32_qkv_lr0.001) | 42 | 1.015 / 0.244 / 0.039 | 0.615 / 0.639 / 0.639 | 0.760 / 0.798 / 0.798 |
| muril | qlora_4bit (r16_a32_qkv_lr0.001) | 42 | 1.014 / 0.258 / 0.038 | 0.608 / 0.640 / 0.639 | 0.776 / 0.802 / 0.810 |
| indicbert | head_only | 42 | 1.146 / 0.983 / 0.940 | 0.476 / 0.601 / 0.581 | 0.686 / 0.769 / 0.698 |
| indicbert | last2 | 42 | 1.072 / 0.576 / 0.251 | 0.629 / 0.594 / 0.617 | 0.764 / 0.752 / 0.776 |
| indicbert | last4 | 42 | 0.970 / 0.224 / 0.061 | 0.646 / 0.625 / 0.631 | 0.786 / 0.783 / 0.798 |
| indicbert | lora (r16_a32_qv_lr0.001) | 42 | 0.925 / 0.227 / 0.074 | 0.610 / 0.581 / 0.622 | 0.760 / 0.733 / 0.781 |
| indicbert | qlora_4bit (r16_a32_qv_lr0.001) | 42 | 0.924 / 0.257 / 0.063 | 0.669 / 0.606 / 0.638 | 0.819 / 0.748 / 0.805 |

## 8. Skipped / failed runs

None.

## 9. How to read these results

- Train, val and test all come from the same SemEval subset (tweets), so these are in-domain scores. They are for **choosing a method**, not final results: the final models are trained as described in docs/TRD.md and evaluated separately.
- Hindi and Bengali rows are machine translations of the English tweets; each language has 3 script variants (pure, romanised, code-mixed). Per-language and per-variant scores partly reflect translation quality and tokenizer coverage (see the `[UNK]` rates in section 2).
- `val` has 60 tweets (5 labelled against), each in 7 versions, and the 7 versions are not independent. Differences of a few points are within noise; prefer the parsimony pick when scores are close.
- `against` is rare, so its F1, and therefore Macro-F1, moves a lot with a handful of tweets. Check the per-class columns and the confusion matrix.
- The best epoch and the method ranking both use `val`, so its scores are slightly optimistic; `test` (section 5) is the unbiased check.
- LoRA / QLoRA hyperparameters (r, alpha, target modules, learning rate) were picked by the stage-1 search on `val`; head-only and last-N use fixed learning rates. Stage 2 re-runs the pick with new seeds, but since the combination itself was chosen on `val`, LoRA-type scores there are a little optimistic compared with the others. `test` (section 5) is the fair check.

## 10. Environment and configuration

```json
{
  "device": "cuda",
  "gpu": "Tesla T4",
  "gpu_mem_gb": 14.6,
  "lib_accelerate": "1.15.0",
  "lib_bitsandbytes": "0.50.2",
  "lib_mlflow": "3.18.0",
  "lib_pandas": "3.1.1",
  "lib_peft": "0.21.2",
  "lib_scikit-learn": "1.9.1",
  "lib_torch": "2.14.1+cu130",
  "lib_transformers": "5.22.2",
  "platform": "Linux-6.6.159+-x86_64-with-glibc2.39",
  "python": "3.13.16",
  "runtime": "colab"
}
```

```json
{
  "MODELS": {
    "muril": "google/muril-base-cased",
    "indicbert": "ai4bharat/IndicBERTv2-MLM-only"
  },
  "RUN_METHODS": [
    "head_only",
    "last_n",
    "lora",
    "qlora"
  ],
  "LAST_N_LIST": [
    2,
    4
  ],
  "LR": {
    "head_only": {
      "lr": 0.001
    },
    "last_n": {
      "lr": 5e-05,
      "head_lr": 0.0005
    }
  },
  "LORA_SEARCH": {
    "r": [
      8,
      16,
      32
    ],
    "alpha_mult": [
      1,
      2
    ],
    "targets": [
      "qv",
      "qkv",
      "all"
    ],
    "lr": [
      0.0003,
      0.001
    ]
  },
  "LORA_TARGET_SETS": {
    "qv": [
      "query",
      "value"
    ],
    "qkv": [
      "query",
      "key",
      "value"
    ],
    "all": [
      "query",
      "key",
      "value",
      "attention.output.dense",
      "intermediate.dense",
      "output.dense"
    ]
  },
  "LORA_DROPOUT": 0.1,
  "SEARCH_SEEDS": [
    1
  ],
  "SEARCH_MAX_TRIALS": null,
  "QLORA_BITS": [
    4
  ],
  "QLORA_HPARAMS": "from_lora",
  "CLAIM_MODE": "fixed",
  "FIXED_CLAIM": "Climate change and global warming is a serious concern.",
  "TRAIN_FRACTION": 0.6,
  "VAL_FRACTION": 0.2,
  "TRAIN_SOURCE_ROWS": null,
  "TRAIN_VARIANT_POLICY": "all",
  "SPLIT_SEED": 2024,
  "SEEDS": [
    42,
    43
  ],
  "EPOCHS": 3,
  "BATCH_SIZE": 16,
  "MAX_LEN": 128,
  "WEIGHT_DECAY": 0.01,
  "WARMUP_FRACTION": 0.1,
  "GRAD_CLIP": 1.0,
  "HEAD_DROPOUT": 0.1,
  "PATIENCE": 2,
  "EARLY_STOP_MONITOR": "val",
  "RANK_SET": "val"
}
```

Total training+evaluation time of the runs in this report: 4.5 min. MLflow store: `/content/drive/MyDrive/EcoConsensus/experiment_outputs/mlflow.db` (copy it to the Mac and open with `mlflow ui --backend-store-uri sqlite:///mlflow.db`).

## 11. References

- **MuRIL**: Khanuja et al., 2021, *MuRIL: Multilingual Representations for Indian Languages*, Google Research. arXiv:2103.10730. Model: `google/muril-base-cased` (Google).
- **IndicBERT v2**: Doddapaneni et al., 2023, *Towards Leaving No Indic Language Behind: Building Monolingual Corpora, Benchmark and Models for Indic Languages*, ACL 2023 (AI4Bharat). Model: `ai4bharat/IndicBERTv2-MLM-only`.
- **Transformers**: Wolf et al., 2020, *Transformers: State-of-the-Art Natural Language Processing*, EMNLP 2020 (Hugging Face).
- **PEFT** (LoRA/QLoRA adapters): Hugging Face PEFT library (Mangrulkar et al., 2022).
- **LoRA**: Hu et al., 2021, *LoRA: Low-Rank Adaptation of Large Language Models*, Microsoft. arXiv:2106.09685.
- **QLoRA**: Dettmers et al., 2023, *QLoRA: Efficient Finetuning of Quantized LLMs*, NeurIPS 2023; **bitsandbytes** by Tim Dettmers et al.
- **AdamW**: Loshchilov and Hutter, 2019, *Decoupled Weight Decay Regularization*, ICLR 2019.
- **PyTorch**: Paszke et al., 2019, *PyTorch: An Imperative Style, High-Performance Deep Learning Library*, NeurIPS 2019.
- **scikit-learn**: Pedregosa et al., 2011, JMLR 12:2825-2830.
- **MLflow**: Zaharia et al., 2018, *Accelerating the Machine Learning Lifecycle with MLflow*, IEEE Data Engineering Bulletin 41(4) (Databricks).
- **SemEval-2016 Task 6** (experiment data): Mohammad, Kiritchenko, Sobhani, Zhu and Cherry, 2016, *SemEval-2016 Task 6: Detecting Stance in Tweets*, SemEval-2016.
- Hindi and Bengali text is machine-translated from the English tweets with an LLM (see the project README for the model used).
