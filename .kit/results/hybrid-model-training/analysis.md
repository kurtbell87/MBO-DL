# Analysis: CNN+GBT Hybrid Model Training on MES Triple Barrier Labels

## Verdict: REFUTED

The CNN+GBT Hybrid architecture fails to produce actionable trading signals. The CNN component is completely non-functional — it cannot fit even the training data (train R² = 0.001 vs expected > 0.15), producing noise embeddings that degrade XGBoost performance. R3's R²=0.132 does not reproduce in this pipeline. XGBoost on hand-crafted features alone achieves 41% 3-class accuracy (above 33% random), but this directional accuracy is insufficient for profitability under realistic transaction costs. The GBT-only baseline outperforms the hybrid on both accuracy and expectancy, proving the CNN embeddings add negative value. Only under optimistic cost assumptions ($2.49 RT) does the model achieve marginal profitability ($0.81/trade, PF=1.19), and even then, the hybrid underperforms GBT-only.

**Critical caveat**: This is a failed reproduction of R3's CNN signal, not a disproof of it. The CNN pipeline has two known deviations from R3 (normalization and architecture), and the sanity check failure (train R² ≈ 0) proves the pipeline is broken. The integration hypothesis remains untested.

---

## Results vs. Success Criteria

- [ ] **SC-1: FAIL** — mean_cnn_r2_h5 = **-0.002** vs threshold **>= 0.08** (baseline R3: 0.132). CNN R² is negative on every fold. 0% of R3's signal reproduced — not the required 60%.
- [x] **SC-2: PASS** — mean_cnn_r2_h1 = **0.0017** (reported; any value accepted). However, this does NOT resolve the R3 open question — the pipeline is broken, so no valid horizon comparison is possible.
- [x] **SC-3: PASS** — mean_xgb_accuracy = **0.410** vs threshold **>= 0.38** (random: 0.333). XGBoost is 7.7pp above random baseline.
- [ ] **SC-4: FAIL** — neg_fold_count_h5 = **5/5** vs threshold **0/5**. All five folds produce negative CNN R² at h=5.
- [ ] **SC-5: FAIL** — aggregate_expectancy_base = **-$0.44/trade** vs threshold **>= $0.50/trade**. $0.94/trade worse than threshold.
- [ ] **SC-6: FAIL** — aggregate_profit_factor_base = **0.91** vs threshold **>= 1.5**. PF < 1.0 means gross losses exceed gross profits.
- [ ] **SC-7: FAIL** — Hybrid accuracy (0.410) < GBT-only (0.411). Hybrid expectancy (-$0.44) < GBT-only (-$0.38). Fails on BOTH metrics. CNN embeddings are noise that degrades XGBoost.
- [x] **SC-8: PASS** — Hybrid accuracy (0.410) > CNN-only (0.380). Passes on accuracy (+3.1pp).
- [x] **SC-9: PASS** — Cost sensitivity table produced for all 3 scenarios (optimistic, base, pessimistic).
- [ ] **SC-10: FAIL** — 2 sanity checks failed: (1) CNN train R² max = 0.0015, below 0.15 — underfitting confirmed; (2) Folds 4 and 5 R² both negative — late folds do not rescue the signal.

**Score: 4/10 PASS, 6/10 FAIL.** Four primary criteria (SC-1, SC-4, SC-5, SC-6) fail dramatically, not marginally. Clear REFUTED.

---

## Metric-by-Metric Breakdown

### Primary Metrics

#### 1. mean_cnn_r2_h5 = -0.002 (threshold >= 0.08; R3 baseline: 0.132)

| Fold | Train R² (h=5) | Test R² (h=5) | Train Size | Test Size |
|------|---------------|---------------|------------|-----------|
| 1 | -0.0008 | -0.0020 | 18,520 | 13,890 |
| 2 | +0.0015 | -0.0022 | 32,410 | 13,890 |
| 3 | +0.0011 | -0.0026 | 46,300 | 13,890 |
| 4 | +0.0012 | -0.0015 | 60,190 | 13,890 |
| 5 | +0.0007 | -0.0016 | 74,080 | 13,890 |
| **Mean** | **+0.0007** | **-0.0020** | — | — |
| **Std** | — | **0.0004** | — | — |

The CNN achieves essentially zero R² on training data (max 0.0015) and consistently negative R² on test data. R3 reported R²=0.132 +/- 0.048 on the same 19 days with a similar architecture. This is a **67x worse result**. The CNN cannot learn anything from this pipeline's input.

**The smoking gun is the training R².** Train R² ~ 0 means this is not overfitting — it is total underfitting. A ~7.7k parameter CNN cannot fit a regression from 40-dim book input to forward returns. This points to a **pipeline or normalization defect**, not an absence of signal. R3 proved the signal exists.

#### 2. aggregate_expectancy_base = -$0.44/trade (threshold >= $0.50)

Pooled across all 32,004 trades from 5 test folds under base costs ($3.74 RT):

| Metric | Value |
|--------|-------|
| Gross profit | $142,744 |
| Gross loss | $156,933 |
| Net PnL | -$14,189 |
| Profit factor | 0.91 |
| Trade win rate | 50.9% (16,295 wins / 32,004 trades) |
| Breakeven win rate (base costs) | 53.3% |
| **Gap to breakeven** | **-2.4 percentage points** |

The model is 2.4pp short of breakeven trade win rate. This is not a small gap — at 32,004 trades, it represents ~768 additional correct directional calls needed.

### Secondary Metrics

#### CNN R² at h=1

| Fold | Test R² (h=1) |
|------|---------------|
| 1 | -0.0021 |
| 2 | +0.0011 |
| 3 | +0.0018 |
| 4 | +0.0018 |
| 5 | +0.0059 |
| **Mean** | **+0.0017** |

Marginally positive but essentially noise. Folds 2-5 selected h=1 over h=5 during training (h=1 was marginally less bad). This does NOT resolve the R3 open question about CNN at h=1 because the pipeline is broken — if train R² at h=5 ~ 0, no horizon comparison is valid.

#### XGBoost Accuracy and F1

| Fold | Accuracy | F1 Macro | Train Days | Test Period |
|------|----------|----------|------------|-------------|
| 1 | 0.343 | 0.338 | 4 | Mar-Apr 2022 |
| 2 | 0.404 | 0.382 | 7 | May-Jun 2022 |
| 3 | **0.480** | **0.432** | 10 | Jul-Aug 2022 |
| 4 | 0.432 | 0.428 | 13 | Sep-Oct 2022 |
| 5 | 0.393 | 0.382 | 16 | Oct-Dec 2022 |
| **Mean** | **0.410** | **0.392** | — | — |

Accuracy peaks at fold 3 (0.48) then declines. Does NOT increase monotonically with training data. Regime effects dominate data quantity. Fold 1 (0.343) is barely above random (0.333) with only 4 training days.

Fold 3's high accuracy is partially inflated by label distribution: 45.5% class 0 in that fold. Predicting "no trade" on nearly half of bars is an easy accuracy win.

#### Per-Fold PnL (Base Costs)

| Fold | Expectancy | PF | Test Period |
|------|------------|-------|-------------|
| 1 | -$0.28 | 0.942 | Mar-Apr 2022 |
| 2 | -$0.37 | 0.924 | May-Jun 2022 |
| 3 | **-$0.68** | **0.864** | Jul-Aug 2022 |
| 4 | -$0.50 | 0.898 | Sep-Oct 2022 |
| 5 | -$0.46 | 0.906 | Oct-Dec 2022 |

No fold is profitable. Fold 3 has the highest accuracy (0.48) but worst expectancy (-$0.68) — the "accuracy-expectancy paradox." The model predicts class 0 more often (no trade), inflating accuracy, but the trades it does make have worse directional accuracy.

#### Aggregate Sharpe

| Scenario | Sharpe |
|----------|--------|
| Optimistic | +28.2 |
| Base | -14.8 |
| Pessimistic | -39.3 |

The enormous magnitudes reflect annualization over a short, non-contiguous test window (19 days across 5 folds). The sign is informative (negative under base costs); the magnitude is not meaningful.

#### Ablation Comparison

| Model | Accuracy | F1 Macro | Expectancy (base) | PF (base) |
|-------|----------|----------|-------------------|-----------|
| **Hybrid** | 0.410 | 0.392 | -$0.44 | 0.910 |
| **GBT-only** | **0.411** | **0.394** | **-$0.38** | **0.923** |
| CNN-only | 0.380 | 0.286 | -$0.31 | 0.936 |

**Key finding**: GBT-only outperforms Hybrid on every metric. The CNN embeddings are noise that harms XGBoost.

CNN-only achieves the lowest accuracy (0.380) but the best PF (0.936) and best expectancy (-$0.31). This is because CNN-only makes fewer trades, and among those trades, directional accuracy is marginally better. However, all three models lose money under base costs.

| Delta Metric | Value |
|-------------|-------|
| ablation_delta_accuracy (hybrid - max baseline) | -0.001 |
| ablation_delta_expectancy (hybrid - max baseline) | -$0.136 |

The hybrid is $0.14/trade WORSE than the best baseline on expectancy.

#### Cost Sensitivity

| Scenario | RT Cost | Expectancy | PF | Net PnL | Trades |
|----------|---------|------------|----|---------|--------|
| Optimistic | $2.49 | **+$0.81** | **1.188** | +$25,816 | 32,004 |
| Base | $3.74 | -$0.44 | 0.910 | -$14,189 | 32,004 |
| Pessimistic | $6.25 | -$2.95 | 0.519 | -$94,519 | 32,004 |

The model flips from profitable to unprofitable between optimistic and base costs — a $1.25/trade difference (exactly 1 MES tick). Verification: $1.25 x 32,004 = $40,005 PnL swing. Actual: $25,816 - (-$14,189) = $40,005. Exact match.

**The model's edge is thinner than 1 tick per trade.** Profitability is entirely determined by execution quality, not by model accuracy.

#### XGBoost Top-10 Features (by Gain)

| Rank | Feature | Category | Mean Gain |
|------|---------|----------|-----------|
| 1 | volatility_50 | Price Dynamics | 17.25 |
| 2 | message_rate | Microstructure | 6.82 |
| 3 | volatility_20 | Price Dynamics | 6.30 |
| 4 | high_low_range_50 | Price Dynamics | 4.49 |
| 5 | spread | Book Shape | 4.39 |
| 6 | minutes_since_open | Time | 4.16 |
| 7 | time_sin | Time | 3.87 |
| 8 | time_cos | Time | 3.82 |
| 9 | cnn_emb_11 | CNN Embedding | 3.78 |
| 10 | return_20 | Price Dynamics | 3.51 |

Findings:
1. **Volatility features dominate.** volatility_50 at 17.25 is 2.5x the next feature. XGBoost is primarily doing regime identification, not directional prediction.
2. **Time features matter.** 3 of top-10 are time-of-day features (minutes_since_open, time_sin, time_cos). The model uses intraday patterns.
3. **Only 1/16 CNN embeddings in top-10** (cnn_emb_11, rank 9). The other 15 dimensions are below return_20. Confirms CNN embeddings are mostly noise.
4. **return_5 absent from top-10.** No evidence of confound #4 leakage. The forward return feature does not dominate XGBoost's splits.

#### Label Distribution

| Fold | Class -1 (%) | Class 0 (%) | Class +1 (%) |
|------|-------------|-------------|-------------|
| 1 | 4,783 (34.4%) | 4,904 (35.3%) | 4,203 (30.3%) |
| 2 | 5,027 (36.2%) | 4,002 (28.8%) | 4,861 (35.0%) |
| 3 | 3,716 (26.8%) | **6,325 (45.5%)** | 3,849 (27.7%) |
| 4 | 4,813 (34.6%) | 4,740 (34.1%) | 4,337 (31.2%) |
| 5 | 4,644 (33.4%) | 4,865 (35.0%) | 4,381 (31.5%) |

Roughly balanced except fold 3 (Jul-Aug 2022) with 45.5% class 0. This reflects lower volatility or tighter ranges in that period, meaning more bars expire at the vol_horizon without hitting target or stop.

### Sanity Checks

| Check | Expected | Observed | Status |
|-------|----------|----------|--------|
| CNN train R² at h=5 > 0.15 | > 0.15 | Max: 0.0015 | **FAIL (100x below)** |
| CNN test R² fold 4 or 5 > 0 | > 0 | Fold 4: -0.0015, Fold 5: -0.0016 | **FAIL** |
| XGBoost accuracy > 0.33 | > 0.33 | 0.410 | PASS |
| No NaN in CNN output | 0 NaN | 0 NaN | PASS |
| Fold boundaries non-overlapping | Valid | Valid | PASS |
| XGBoost accuracy <= 0.90 | <= 0.90 | 0.410 | PASS |

The two failures are the most important diagnostic. CNN train R² of 0.0015 (100x below the 0.15 threshold) proves the CNN is not learning. The failure is in the CNN pipeline, not in the data or the hypothesis.

---

## Resource Usage

| Resource | Budget | Actual | Status |
|----------|--------|--------|--------|
| GPU-hours | 0 | 0 | Within budget |
| Wall-clock | 2 hours (7,200s) | 32.2 min (1,931s) | Within budget (27%) |
| Training runs | 30 | 30 | Within budget |
| Seeds | 1 | 1 | Within budget |

Budget was appropriate with 6x headroom on wall-clock.

---

## Confounds and Alternative Explanations

### 1. Normalization Deviation (HIGH CONFIDENCE — most likely root cause)

**Spec**: "Price offsets: raw ticks from mid. Sizes: z-score per fold."
**Actual (from RUN notes)**: "both channels z-score normalized."

This is a **protocol deviation**. R3 used raw tick offsets for prices — preserving absolute magnitude. Z-scoring prices removes absolute scale, potentially destroying the signal the CNN learned in R3. If R3's learned filters relied on absolute tick magnitudes (e.g., "a 3-tick spread from mid = X"), z-scoring breaks this mapping.

This is the single most likely explanation for the 67x gap between R3's R²=0.132 and this pipeline's R²=-0.002.

### 2. Architecture Mismatch (MODERATE CONFIDENCE — contributing factor)

| Parameter | R3 | This Pipeline |
|-----------|-----|---------------|
| Conv layers | Conv1d(2->32->32) | Conv1d(2->32->64) |
| Parameters | ~12k | ~7.7k |
| LR schedule | Cosine annealing | Fixed lr=1e-3 |
| Final layers | Linear(32->16) + ReLU + Linear(16->1) | Linear(64->16) |
| Early stop patience | 5 | 10 (deviated from spec's 5) |

Architecture alone should not cause R² to go from +0.132 to -0.002. The spec accounted for this with the 0.08 threshold (60% of R3). But cosine LR + deeper head could matter. Contributing factor, not root cause.

### 3. Book Column Reshaping (MODERATE CONFIDENCE — unverifiable)

Notes say columns are "interleaved as (price, size) pairs, reshaped to (N, 2, 20)." If the deinterleaving was incorrect — mixing prices and sizes across the two CNN channels — the spatial structure would be garbled. Without reading code (protocol forbids), this cannot be confirmed. The total CNN failure is consistent with garbled input.

### 4. h=5 Feature Leakage — NOT OBSERVED

return_5 is absent from XGBoost top-10 features. No evidence of the leakage pathway described in confound #4 of the spec.

### 5. Regime Effects Dominate Data Quantity

Accuracy does not increase monotonically with fold number:
- Fold 1 (4d train): 0.343
- Fold 3 (10d train): **0.480** (peak)
- Fold 5 (16d train): 0.393

Regime matters more than data quantity. Fold 3's Jul-Aug 2022 test period has unusual properties (45.5% class 0) that inflate accuracy while degrading expectancy. This is not a confound for the overall verdict (the model fails everywhere), but it matters for interpreting individual fold results.

### 6. XGBoost is Learning Regime, Not Direction

Top features (volatility_50, message_rate, volatility_20, high_low_range_50, spread) are all regime descriptors — they characterize market state, not direction. The model identifies "what kind of market is this?" not "which direction will it move?" This produces slightly-above-random directional calls (50.9% trade WR) but not enough for profitability at standard costs.

### 7. Could R3's Baseline Have Been Inflated?

R3 used a different training infrastructure (cosine LR, different architecture, raw price normalization). If R3's R²=0.132 was partially driven by these protocol details, the "true reproducible signal" may be lower. However, even at 50% of R3 (R²=0.066), that would be 33x what this pipeline achieved. The gap is too large to explain by baseline inflation alone. R3's signal is real; this pipeline fails to access it.

---

## What This Changes About Our Understanding

### Before this experiment:
- R3 showed CNN R²=0.132 on structured book input — spatial encoder is valuable.
- R6 synthesis recommended CNN+GBT Hybrid with high confidence.
- R7 oracle showed $4.00/trade ceiling — models capturing >12.5% of oracle edge are profitable.
- The path forward was "build it and test it."

### After this experiment:

1. **The CNN signal is real but fragile.** R3's R²=0.132 does not transfer automatically to a new pipeline. Normalization, architecture, and LR schedule matter enormously. The signal is not robust to reasonable variations in training protocol. This is a warning for production: the CNN-based edge may require careful calibration and monitoring.

2. **XGBoost on hand-crafted features has marginal classification power** (41% accuracy, 50.9% trade WR) driven by volatility and time-of-day features. This is regime identification, not directional prediction. The edge is thinner than 1 tick per trade.

3. **The CNN+GBT integration hypothesis is untested.** The CNN pipeline is broken (train R² ~ 0), so we never tested the actual hypothesis. We tested a broken pipeline and correctly observed it was broken. The integration question remains open, contingent on first reproducing R3's CNN signal.

4. **Transaction cost sensitivity is extreme.** The XGBoost model flips from +$0.81/trade to -$0.44/trade with 1 tick of cost difference. Any edge from this model is on the order of 1 tick per trade, making profitability entirely dependent on execution quality.

5. **P1 (CNN at h=1) remains unanswered.** The reported h=1 R²=0.0017 is unreliable because the pipeline cannot achieve h=5 R² > 0 even on training data. No valid horizon comparison is possible from this experiment.

### Updated mental model:
The R6 synthesis recommendation (CNN+GBT Hybrid) still stands as the architecture to pursue. However, the path requires a prerequisite step: **reproduce R3's CNN R²=0.132 in the Python pipeline first**, then re-attempt the integration. The current experiment is a failed reproduction, not a disproof.

---

## Proposed Next Experiments

### 1. CNN Reproduction Diagnostic (HIGHEST PRIORITY)

Isolate and fix the reproduction failure before re-attempting the hybrid integration:
- Use R3's exact normalization: raw tick offsets for prices, z-score for sizes only.
- Use R3's exact architecture: Conv1d(2->32->32), Linear(32->16), ReLU, Linear(16->1).
- Use R3's exact LR schedule: cosine annealing.
- Explicitly verify book column reshaping: print channel 0 and channel 1 for a sample bar and confirm they match expected book structure (prices vs sizes).
- Run on fold 5 only (most data). Pass criterion: train R² > 0.15 and test R² > 0.05.
- If it passes, re-run the full 5-fold hybrid protocol.

### 2. XGBoost Hyperparameter Search (CONDITIONAL — if CNN reproduction fails)

If the CNN cannot be reproduced, determine whether XGBoost alone can close the 2.4pp gap to breakeven:
- Grid search: max_depth {4,6,8}, n_estimators {300,500,1000}, learning_rate {0.01,0.05,0.1}.
- 62-feature hand-crafted set only (no CNN).
- Evaluated on 5-fold expanding-window CV with PnL under base costs.
- Pass criterion: aggregate_expectancy_base >= $0.50.

### 3. Cost-Aware Classification Objective (CONDITIONAL — after Experiment 1 succeeds)

If CNN reproduction succeeds and the hybrid integration works, test whether optimizing XGBoost for PnL (cost-aware loss weighting) improves expectancy over uniform cross-entropy.

---

## Program Status

- Questions answered this cycle: 1 (P1 — INCONCLUSIVE, pipeline broken)
- New questions added this cycle: 1 (CNN reproduction diagnostic)
- Questions remaining (open, not blocked): 2 (P2 — cost sensitivity; new P3 — CNN reproduction)
- Handoff required: NO (all fixes are within research scope — normalization, architecture, reshaping)
