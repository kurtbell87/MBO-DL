# Analysis: CNN Reproduction Diagnostic

## Verdict: REFUTED

The hypothesis — that fixing three protocol deviations (z-score normalization, architecture 2→32→64→2→32→32, missing cosine LR) would restore CNN mean OOS R² to ≥ 0.10 — is **decisively refuted**. With all three deviations corrected and the architecture matching R3 exactly (12,128 params, 0% deviation), fold 5 train R² = 0.002. This is statistically indistinguishable from Phase B's broken pipeline (train R² = 0.001). The MVE gate correctly triggered, aborting the full protocol. Five additional normalization variants were tested diagnostically; all produce R² < 0.002, including the z-scored variant that Phase B's post-mortem labeled "FATAL."

The three deviations were not the root cause. Phase B's post-mortem was incorrect. The data pipeline difference between R3's Phase 4 Python export and Phase 9A's C++ export is now the primary suspect.

---

## Results vs. Success Criteria

- [ ] **SC-1: FAIL (NOT EVALUATED)** — mean_cnn_r2_h5 ≥ 0.10. MVE gate failure aborted the full 5-fold. Only fold 5 was trained; test R² = 0.000077 — three orders of magnitude below the 0.10 threshold. Even if all 5 folds performed identically, the mean would be ~0.0001. No plausible scenario saves this.
- [ ] **SC-2: FAIL** — No fold train R² < 0.05 at h=5. MVE fold 5 train R² = **0.00211** (observed) vs. **0.05** (threshold). 24× below threshold. This is the same failure mode as Phase B (train R² = 0.001). The CNN cannot fit the training data.
- [ ] **SC-3: NOT EVALUATED** — aggregate_expectancy_base ≥ $0.50/trade. Conditional on SC-1 passing. SC-1 failed; Step 2 was not executed.
- [ ] **SC-4: NOT EVALUATED** — Hybrid outperforms GBT-only on accuracy OR expectancy. Conditional on SC-1 passing. Step 2 was not executed.
- [ ] **SC-5: FAIL** — No sanity check failures. The "train R² > 0.05" sanity check hard-failed (0.00211 << 0.05). 5/7 evaluated sanity checks pass, 1 hard-fails, 1 soft-fails (channel 0 units in index points rather than integer ticks), 1 N/A.

**Score: 0/2 evaluated criteria pass. 3/5 not evaluated due to abort cascade.**

---

## Metric-by-Metric Breakdown

### Primary Metrics

#### 1. mean_cnn_r2_h5

| Property | Value |
|----------|-------|
| Observed | null (MVE gate aborted full 5-fold) |
| MVE proxy (fold 5 only) | test R² = 0.000077 |
| Threshold | ≥ 0.10 |
| R3 baseline | 0.132 ± 0.048 |
| Phase B baseline | −0.002 |
| Delta vs. R3 | −0.132 (entire signal absent) |
| Delta vs. Phase B | +0.002 (noise — no improvement) |

The reproduction completely failed. The CNN shows zero predictive power on `time_5s.csv`. Fixing the three protocol deviations produced no measurable improvement over Phase B.

#### 2. aggregate_expectancy_base

Not evaluated. SC-1 gate failure prevented Step 2 execution. Phase B reference: GBT-only expectancy −$0.38/trade (base costs).

### Secondary Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| per_fold_cnn_r2_h5 | Fold 5 only: 0.000077 | Full 5-fold not executed |
| per_fold_cnn_train_r2_h5 | Fold 5 only: 0.00211 | Smoking gun reproduced: CNN cannot fit training data |
| epochs_trained_per_fold | Fold 5: 40/50 | Early stopping triggered (patience=10). Model plateaued ~epoch 30. |
| mean_xgb_accuracy | Not evaluated | Step 2 not executed |
| mean_xgb_f1_macro | Not evaluated | Step 2 not executed |
| aggregate_profit_factor | Not evaluated | Step 2 not executed |
| hybrid_vs_gbt_delta_accuracy | Not evaluated | Step 2 not executed |
| hybrid_vs_gbt_delta_expectancy | Not evaluated | Step 2 not executed |
| cost_sensitivity_table | Not evaluated | Step 2 not executed |
| xgb_top10_features | Not evaluated | Step 2 not executed |
| label_distribution | Not evaluated (data loaded: 87,970 bars, 19 days) | Step 2 not executed |

**R3 Comparison Table (Step 1):**

| Fold | R3 test R² | This Experiment test R² | Delta |
|------|-----------|------------------------|-------|
| 1 | 0.163 | — | — |
| 2 | 0.109 | — | — |
| 3 | 0.049 | — | — |
| 4 | 0.180 | — | — |
| 5 | 0.159 | 0.0001 | **−0.159** |
| **Mean** | **0.132** | **—** | — |

The gap is total. This is not a degradation or marginal miss — it is a complete absence of signal. R3's fold 5 produced R² = 0.159; this experiment's fold 5 produced R² = 0.0001. Same architecture, same optimizer, same hyperparameters, different data pipeline.

### Normalization Diagnostic (5 variants, fold 5, 10 epochs each)

The RUN agent tested 5 normalization variants as a diagnostic sweep after the MVE gate failure. This was beyond the spec requirements but is the most valuable finding of the experiment:

| Variant | Best R² | Description |
|---------|---------|-------------|
| Raw targets (R3 protocol) | 0.000812 | R3-exact configuration |
| Normalized targets (z-scored) | 0.001510 | Z-scored target variable |
| Tick-scale ch0 + normalized targets | 0.000792 | Channel 0 multiplied ×4 to match R3 tick units |
| Z-scored both channels + normalized targets | 0.001491 | Phase B's exact normalization — the "FATAL" error |
| Higher LR (3e-3) + normalized targets | 0.000794 | 3× learning rate |

**Critical finding:** All 5 variants produce R² < 0.002. The z-scored variant (Phase B's "FATAL" error) performs within noise of the raw variant (0.0015 vs. 0.0008). **This directly falsifies the Phase B post-mortem.** Z-scoring was not fatal because there is no tick magnitude information in this data for z-scoring to destroy.

### Sanity Checks

| # | Check | Expected | Observed | Pass/Fail |
|---|-------|----------|----------|-----------|
| 1 | CNN param count within 5% of 12,128 | 12,128 ± 606 | **12,128** (0.0% deviation) | **PASS** |
| 2 | Channel 0 = integer tick offsets | Integers like −10, −5, −1, 1, 5, 10 | Raw price offsets in index points, range [−5.625, 5.625], std=1.45 | **SOFT FAIL** — not z-scored (good), but in index point units (0.25 per tick) rather than integer ticks. Scale tested explicitly; no effect on R². |
| 3 | Channel 1 = log-transformed sizes | Positive reals like 2.3, 3.1 | log1p(raw_sizes), z-scored per fold | **PASS** |
| 4 | LR decays from ~1e-3 toward ~1e-5 | Cosine schedule confirmed | Start: 0.001, End: 0.000124 (epoch 40, early stop) | **PASS** |
| 5 | Train R² per fold > 0.05 | > 0.05 | **0.00211** | **FAIL** — 24× below threshold |
| 6 | No NaN in CNN outputs | 0 NaN | 0 NaN confirmed | **PASS** |
| 7 | Fold day boundaries non-overlapping | No overlap | Non-overlapping confirmed | **PASS** |
| 8 | XGBoost accuracy 0.33–0.90 | Step 2 only | Not evaluated | **N/A** |

**Summary:** 5 PASS, 1 hard FAIL (train R²), 1 soft FAIL (channel 0 units), 1 N/A.

The hard FAIL on train R² is the decisive result. At 0.002, the CNN explains 0.2% of training variance — indistinguishable from random. This is not overfitting (train >> test) or regularization collapse (test << train). The model cannot find *any* learnable relationship between the (20, 2) book tensor and `fwd_return_5` in this data. Forty epochs of training with appropriate learning rate decay provided ample opportunity for convergence.

---

## Resource Usage

| Resource | Budgeted | Actual | Assessment |
|----------|----------|--------|------------|
| GPU hours | 0 | 0 | On budget |
| Wall clock | 90 min (5,400s) | 17.4 min (1,046s) | 19% of budget |
| Training runs | 16 | 6 (1 MVE + 5 diagnostic variants) | 38% of budget |
| Seeds | 1 (seed=42) | 1 | On budget |

The MVE gate design worked as intended, saving ~70% of expected compute by aborting before futile full 5-fold and Step 2 execution. The 90-minute budget was appropriate for the full protocol; 17 minutes was sufficient for the abort path.

---

## Confounds and Alternative Explanations

### 1. Data Pipeline Difference — PRIMARY SUSPECT

R3 used Phase 4 Track B.1 (Python-based export). This experiment uses Phase 9A C++ `bar-feature-export` (`time_5s.csv`). The two pipelines could differ in ways that fundamentally alter the book tensor:

- **Mid-price calculation:** Simple mid vs. microprice vs. weighted mid. Different reference points shift all price offsets.
- **Level ordering / selection:** Which 20 of potentially many book levels are selected, and in what order. If the C++ export uses different level aggregation (e.g., price-level aggregation vs. queue-by-queue), the spatial adjacency structure changes fundamentally — exactly what Conv1d exploits.
- **Book snapshot timing:** Whether the snapshot captures book state at bar open, bar close, or last update within the bar interval.
- **Missing level handling:** How levels beyond visible depth are encoded (zeros, NaN, last-known, or omitted).
- **Bid/ask layout:** Whether levels alternate bid-ask or are grouped (all bids then all asks).

**This is the most parsimonious explanation.** Supporting evidence:
1. A 12,128-parameter CNN achieving 0.2% training R² on 70k samples is not a model failure; it is a signal absence.
2. Five normalization variants produce identical near-zero R², ruling out preprocessing configuration.
3. The ONLY variable that differs between this experiment and R3 is the data source. Architecture, optimizer, hyperparameters, loss, seed — all matched exactly.
4. The C++ export confirms a known difference: price offsets are in index points (floats), while R3 used tick counts (integers). If this visible difference exists, other invisible differences likely exist too.

### 2. Spec Architecture Documentation Error

The diagnostic spec described R3 as `Conv1d(2→32→32)` (~4,001 params). R3 actually used ch=59 (~12,128 params) to hit its stated 12k target. The RUN agent caught this by consulting R3's original spec (`book-encoder-bias.md`). This is a documentation error in the *diagnostic* spec, not an experimental error. However, it raises a meta-concern: **if our documentation of R3's architecture was materially wrong (3× wrong on param count), what other undocumented differences exist in R3's pipeline?** This strengthens the case for a full R3 end-to-end reproduction.

### 3. Data Period / Market Regime

Both this experiment and Phase B used the same 19-day data period (from Phase 9A export). R3 may have used a different day selection from the 312 available. If R3's data spanned a regime with stronger book structure, the signal could be period-specific. However, this doesn't explain a 0.132 → 0.002 gap — R3's worst fold (fold 3) still achieved 0.049, which is 24× better than this experiment's fold 5. Market regime alone cannot plausibly account for this.

### 4. R3 Result Validity — Must Be Questioned

Two independent attempts (Phase B and this diagnostic) have failed to reproduce R3's CNN R² = 0.132 using a different data pipeline. R3's protocol was incompletely documented (channel width error). The adversarial hypothesis must be stated: **R3's R² = 0.132 may itself be unreliable.** Possible mechanisms:
- Temporal leakage in fold construction (look-ahead in normalization statistics)
- Incorrect R² computation (e.g., computed on training set rather than held-out test)
- Data leakage through feature engineering (target information encoded in book snapshot construction)
- An unintentional pipeline artifact that happens to correlate with `fwd_return_5`

This is low probability but high impact. R3's result should not be treated as ground truth until independently reproduced end-to-end.

### 5. Single Seed / Single Fold

Only fold 5 was trained (MVE gate). One seed, one fold — no variance estimate is possible. However, the failure magnitude (0.002 vs. 0.05 threshold, a 24× gap) makes stochastic variance irrelevant. The 5-variant normalization sweep provides additional robustness: 5 independently initialized training runs, all producing R² < 0.002. This is not noise. The signal is absent.

### 6. Phase B's Post-Mortem Was Incorrect Root Cause Analysis

Phase B identified three deviations and attributed the CNN failure to them (especially z-scoring as "FATAL"). This experiment fixed all three and found identical results. The normalization diagnostic explicitly showed z-scored and raw normalization produce the same R² (0.0015 vs. 0.0008). **The post-mortem confused coincidence with causation.** The three deviations coexisted with the failure; they did not cause it. The actual cause was upstream in the data pipeline.

---

## What This Changes About Our Understanding

### Mental Model Before

1. R3 proved the CNN spatial signal exists (R² = 0.132).
2. Phase B broke it via 3 protocol deviations (normalization, architecture, optimizer).
3. Fixing those 3 deviations would restore the signal.
4. The CNN+GBT Hybrid architecture recommendation (R6) was the correct next step.

### Mental Model After

1. **The 3 protocol deviations are NOT the root cause.** With all three fixed, CNN train R² = 0.002 — identical to Phase B. The Phase B post-mortem ("z-scoring was FATAL") was wrong; z-scored and raw normalization produce indistinguishable R².
2. **The data pipeline is the critical variable.** Five normalization variants, spanning the full range of plausible preprocessing, all produce R² < 0.002. The predictive structure is absent from `time_5s.csv`, not hidden behind a configuration error.
3. **R3's signal is pipeline-specific, not market-universal.** It existed in R3's Phase 4 Track B.1 data. It does not exist in Phase 9A's C++ export of the same market data. Either the exports produce materially different book representations, or R3's result was an artifact.
4. **R6's architecture recommendation is currently ungrounded.** R6's "CONDITIONAL GO" for CNN+GBT depended on the CNN signal being real and reproducible. Two independent attempts on Phase 9A data have failed. The recommendation may still be correct — but its enabling evidence has not been independently verified.
5. **The "CNN signal is real but fragile" framing should be revised to: "R3's CNN signal has not been reproduced outside R3's data pipeline."** This is a weaker and more honest claim.

### Revised Hypothesis

The CNN spatial signal at R² = 0.132 was observed within R3's specific data pipeline context. The Phase 9A C++ bar-feature-export produces a different book representation that does not contain the target-correlated spatial structure the Conv1d filters exploit. The root cause is a data contract mismatch between the two export pipelines — not a CNN training configuration error. This must be verified by direct data comparison before any further CNN integration work.

---

## Deviations Log

| Parameter | R3 Value | Actual Value | Justification |
|-----------|----------|--------------|---------------|
| Channel width | 59 (R3 actual, 12,128 params) | 59 (matching R3) | Diagnostic spec incorrectly described as 2→32→32. RUN agent correctly consulted R3's original spec. |
| Data source | Phase 4 Track B.1 export (Python) | Phase 9A C++ bar-feature-export (time_5s.csv) | Mandated by spec. This is Confound #1 — now identified as the primary suspect. |
| Channel 0 units | Integer ticks | Index points (floats, 4× smaller) | Consequence of C++ export convention. Scale-invariance tested; no effect on R². |

Target was zero deviations; achieved two, one potentially consequential (data source).

---

## Proposed Next Experiments

### 1. Data Pipeline Comparison (HIGHEST PRIORITY)

**Hypothesis:** The C++ export and R3's Python export produce structurally different book tensors for the same underlying market events.

**Method:**
- Reproduce R3's Phase 4 Track B.1 Python-based data loading from raw `.dbn.zst` files for the same 19 days.
- Export book snapshots in R3's format: (N, 2, 20) with tick-normalized price offsets.
- Direct comparison: price offset reference, level ordering, value distributions, correlation with `fwd_return_5`.
- If differences found: train CNN on R3-format data for the same 19 days.
- **Decision gate:** R3-format train R² > 0.05 → C++ export confirmed as root cause → handoff to fix. R3-format train R² < 0.05 → R3's signal may be period-specific or artifactual.

### 2. R3 End-to-End Reproduction (if data pipelines differ)

**Purpose:** Independently verify R3's R² = 0.132 on R3's original data to confirm the signal was real.

**Method:** Run R3's complete pipeline on R3's exact day selection. If R3 cannot be reproduced on its own data, the result was likely an artifact.

### 3. GBT-Only Baseline Refinement (independent of CNN outcome)

**Purpose:** Assess whether GBT-only can produce positive expectancy with improved feature engineering or tuning, independent of the CNN question.

**Rationale:** Oracle proves $4.00/trade is available. Phase B showed GBT-only at −$0.38/trade. The gap might be closable without CNN. This path does not depend on resolving the data pipeline question and provides a fallback architecture.

---

## Program Status

- Questions answered this cycle: 0 (P3 narrowed but not resolved)
- New questions added this cycle: 0 (P1 "data pipeline equivalence" already existed; now elevated to highest priority)
- Questions remaining (open, not blocked): 3 (P1 data pipeline equivalence, P2 transaction cost sensitivity, P3 CNN reproduction root cause)
- Handoff required: **NO** — the data comparison experiment is within research scope. If C++ export is confirmed broken, a handoff will be needed at that point to fix shared infrastructure.
