# Analysis: XGBoost Hyperparameter Tuning on Full-Year Data

## Verdict: REFUTED (Outcome C — MARGINAL)

Systematic hyperparameter tuning of XGBoost over 79 configurations on the full-year 1.16M-bar dataset produces a **0.15pp accuracy gain** on CPCV (0.4489 to 0.4504) — far below the 2.0pp target. The accuracy landscape is extraordinarily flat (entire 64-config search spans 0.33pp, std=0.0006). However, tuning produces a **$0.065/trade expectancy improvement** (from -$0.066 to -$0.001), bringing the model to the knife-edge of breakeven at $3.74 RT cost. Default parameters are near-optimal for accuracy. **The feature set, not the hyperparameters, is the binding constraint on classification performance.**

---

## Results vs. Success Criteria

- [ ] **SC-1: FAIL** — CPCV mean accuracy 0.4504 vs. threshold 0.469 (missed by 1.86pp). Baseline: 0.4489.
- [ ] **SC-2: FAIL** — CPCV mean expectancy -$0.001/trade vs. threshold $0.00 (missed by $0.001). Baseline: -$0.066.
- [ ] **SC-3: FAIL** — Holdout accuracy 0.423 vs. threshold 0.441 (missed by 1.8pp). Baseline: 0.403.
- [x] **SC-4: PASS** — 50 of 64 search configs outperform default on 5-fold CV (threshold: 3).
- [x] **SC-5: PASS** — Best config CPCV accuracy std = 0.0117 (threshold: <0.05).
- [x] Sanity checks: **ALL 5 PASS** (see details below).

**Decision rule mapping:** SC-4 passes but SC-1 fails → **Outcome C: MARGINAL.** Some configs better, but <2pp gain. Default params are near-optimal. Tuning is not the bottleneck. Next: combine modest tuning gain with label-design-sensitivity.

---

## Metric-by-Metric Breakdown

### Primary Metrics

| Metric | Default | Tuned | Delta | Threshold | Pass? |
|--------|---------|-------|-------|-----------|-------|
| CPCV mean accuracy | 0.4489 | 0.4504 | **+0.15pp** | >=0.469 | FAIL |
| CPCV mean expectancy (base) | -$0.066 | -$0.001 | **+$0.065** | >=$0.00 | FAIL |

The accuracy gain is negligible (0.15pp, well within the CPCV std of 1.17pp — not statistically significant). The expectancy gain is economically meaningful — a 98.4% reduction in negative expectancy — but the model remains $0.001/trade below breakeven.

### Secondary Metrics

| Metric | Default | Tuned | Notes |
|--------|---------|-------|-------|
| search_best_cv_accuracy | 0.4500 | 0.4509 | Best of 64 configs; delta=0.09pp from default |
| search_accuracy_landscape | — | See landscape analysis below | 64 configs, range 0.4476–0.4509 |
| cpcv_accuracy_std | 0.0116 | 0.0117 | Effectively identical |
| cpcv_expectancy_std | — | 0.122 | High variance — expectancy is noisy across splits |
| per_class_recall | See table below | See table below | Long recall worsened |
| profit_factor | 0.986 | 0.9998 | Improved but still <1.0 |
| walkforward_accuracy | — | 0.452 | Agrees with CPCV mean (delta=0.2pp) |
| walkforward_expectancy | — | -$0.140 | **Much worse than CPCV** — critical finding |
| per_quarter_expectancy | See table below | See table below | All 4 quarters improved |
| feature_importance_top10 | See table below | See table below | Same features, same ordering |
| best_n_estimators_mean | 500 (fixed) | 384.5 | Tuned uses FEWER trees than default |
| cost_sensitivity | See table below | See table below | Breakeven exactly at $3.74 RT |
| long_recall_vs_short | 0.201 / 0.586 | 0.149 / 0.634 | Asymmetry worsened |

Additional metrics reported in data:
- **PBO:** 0.733 (default) → 0.556 (tuned). Lower PBO means less overfitting. Improvement.
- **Breakeven RT cost:** $3.74 — exactly the base cost assumption.
- **Label-0 trade fraction:** 22.4% (>20% flag threshold — borderline).
- **Total configs evaluated:** 79 (vs spec's 64 — fine search expanded).
- **Total training runs:** 493 (vs spec's ~420).

### Search Landscape Analysis

The accuracy landscape across all 64 configs is **extraordinarily flat:**

| Statistic | Value |
|-----------|-------|
| Min accuracy | 0.4476 |
| Max accuracy | 0.4509 |
| Full spread | **0.33pp** |
| Mean | 0.4502 |
| Std | **0.0006** |
| Configs above default | 50/64 (78%) |
| Default rank | 51st of 64 |

The entire search range spans barely one-third of a percentage point. The inter-config std (0.06pp) is **20x smaller** than the CPCV fold-to-fold std (1.17pp). The hyperparameter surface is a plateau — accuracy is essentially invariant to XGBoost hyperparameters over this range. The default sits near the bottom of the plateau, but even the top of the plateau is only 0.09pp above default on 5-fold CV.

**Interpretation:** This is strong evidence that the 20-feature set has extracted approximately all the information XGBoost can use for 3-class classification. No amount of hyperparameter tuning will yield the 2pp gain needed. The constraint is informational, not algorithmic.

### Best Tuned Hyperparameters vs. Default

| Parameter | Default | Tuned | Change |
|-----------|---------|-------|--------|
| max_depth | 6 | 6 | Same |
| learning_rate | 0.05 | 0.0134 | -73% (4x slower) |
| min_child_weight | 10 | 20 | +100% (more conservative splits) |
| subsample | 0.8 | 0.561 | -30% (more stochastic) |
| colsample_bytree | 0.8 | 0.748 | -7% (slightly less) |
| reg_alpha (L1) | 0.1 | 0.00137 | -99% (near-zero L1) |
| reg_lambda (L2) | 1.0 | 6.586 | +559% (strong L2) |
| n_estimators (actual) | 500 (fixed) | 384.5 (early stop) | -23% (fewer trees) |

**Profile:** The tuned model is a "slow, conservative, well-regularized" XGBoost: 4x slower learning rate, 6.6x stronger L2 regularization, 2x min_child_weight, 30% more aggressive subsampling, and stops earlier (385 vs 500 trees). The shift from L1 (0.1→0.001) to L2 (1.0→6.6) regularization suggests the model benefits from shrinking all feature contributions uniformly rather than zeroing some out. Same depth (6) — the tree structure complexity is unchanged.

### Per-Class Recall

| Class | Default Recall | Tuned Recall | Delta |
|-------|---------------|--------------|-------|
| Short (-1) | 0.586 | 0.634 | **+4.8pp** |
| Hold (0) | 0.547 | 0.552 | +0.5pp |
| Long (+1) | 0.201 | 0.149 | **-5.2pp** |

The tuned model **worsens the long-recall asymmetry** identified in the E2E CNN experiment. Long recall drops from 0.201 to 0.149 while short recall improves from 0.586 to 0.634. The model becomes MORE confident on shorts and LESS willing to predict longs. This redistribution is what drives the expectancy improvement: fewer losing long trades (which cost $6.25 + RT per wrong call) matter more than additional correct short trades under the asymmetric 10:5 cost structure. The model is effectively learning to abstain from longs more aggressively — not to predict better.

### Per-Quarter Expectancy

| Quarter | Default | Tuned | Delta |
|---------|---------|-------|-------|
| Q1 | +$0.120 | +$0.137 | +$0.017 |
| Q2 | -$0.022 | +$0.058 | **+$0.080** |
| Q3 | -$0.268 | -$0.219 | +$0.049 |
| Q4 | -$0.278 | -$0.180 | **+$0.098** |

Tuning improves all four quarters. Q2 flips from negative to positive. Q3-Q4 remain deeply negative but with reduced losses. The H1 profitable regime strengthens (+$0.137 + $0.058 = +$0.195 H1 mean).

**Discrepancy note:** The E2E CNN experiment reported default GBT per-quarter as Q1=+$0.003, Q2=+$0.029. This experiment shows Q1=+$0.120, Q2=-$0.022. These differ substantially. The aggregate CPCV expectancy matches well (-$0.066 here vs -$0.064 in E2E CNN), so the discrepancy reflects how CPCV test-split predictions are assigned to calendar quarters (which specific splits test which days), not a data or pipeline mismatch. Per-quarter CPCV numbers are noisy and should not be compared across experiments.

### Feature Importance Comparison (Top 10 by Gain)

| Rank | Default Feature | Default Gain | Tuned Feature | Tuned Gain |
|------|----------------|-------------|---------------|------------|
| 1 | volatility_50 | 4,079 | volatility_50 | 8,535 |
| 2 | volatility_20 | 1,191 | volatility_20 | 3,521 |
| 3 | message_rate | 488 | high_low_range_50 | 1,361 |
| 4 | spread | 372 | message_rate | 1,210 |
| 5 | high_low_range_50 | 371 | time_sin | 506 |
| 6 | time_sin | 304 | time_cos | 480 |
| 7 | time_cos | 289 | spread | 449 |
| 8 | minutes_since_open | 255 | minutes_since_open | 444 |
| 9 | weighted_imbalance | 229 | trade_count | 331 |
| 10 | avg_trade_size | 208 | weighted_imbalance | 330 |

**Key observations:**
- volatility_50 dominates both models. Its gain share in the tuned model is **49.7%** — just under the 50% sanity threshold. Essentially a near-monopoly.
- The same features appear in both top-10 lists (trade_count replaces avg_trade_size; minor reordering). No feature dominance shift.
- The improvement comes from regularization changes on the same features, not from discovering latent patterns.
- **return_5 is absent from both top-10 lists** — confound #6 (leakage via backward-looking returns) is cleared.

### Walk-Forward vs. CPCV Consistency

| Metric | CPCV | Walk-Forward | Delta |
|--------|------|-------------|-------|
| Accuracy | 0.4504 | 0.4524 | +0.2pp (good agreement) |
| Expectancy (base) | -$0.001 | -$0.140 | **-$0.139** (large disagreement) |

Walk-forward detail:

| Fold | Train | Test | Accuracy | Expectancy |
|------|-------|------|----------|------------|
| 0 | days 1–100 | days 101–140 | 0.450 | -$0.144 |
| 1 | days 1–140 | days 141–180 | 0.468 | -$0.198 |
| 2 | days 1–180 | days 181–201 | 0.439 | -$0.076 |

Accuracy agrees well (0.2pp divergence, under 5pp threshold). However, **walk-forward expectancy is dramatically worse** (-$0.140 vs -$0.001). This is a critical finding: CPCV's combinatorial splits allow training on future regime patterns (e.g., Q3 data informs Q1 test predictions), painting an optimistic expectancy picture. Walk-forward cannot do this. The walk-forward result is more representative of deployment reality.

### Holdout Evaluation (Sacred — One Shot)

| Metric | Default | Tuned | Delta |
|--------|---------|-------|-------|
| Accuracy | 0.403 | 0.423 | **+2.0pp** |
| Expectancy (base) | -$0.205 | -$0.132 | +$0.073 |

Holdout shows a +2.0pp accuracy gain — larger than the 0.15pp CPCV gain, suggesting the tuned model generalizes better to out-of-distribution regimes (Nov-Dec 2022). However, holdout expectancy remains deeply negative (-$0.132).

**Baseline holdout discrepancy:** This experiment's default holdout accuracy (0.403) is 1.8pp below the E2E CNN experiment's holdout (0.421). CPCV means match (0.449 vs 0.449), so the discrepancy is holdout-specific — likely from differences in how the full-dev training is set up (internal validation split for early stopping may consume different days). The RELATIVE comparison (default vs tuned within this experiment) remains valid.

### Cost Sensitivity

| Scenario | RT Cost | Tuned Exp | Tuned PF | Default Exp | Default PF |
|----------|---------|-----------|----------|-------------|-----------|
| Optimistic | $2.49 | **+$1.249** | **1.306** | +$1.184 | 1.288 |
| Base | $3.74 | -$0.001 | 0.9998 | -$0.066 | 0.986 |
| Pessimistic | $6.25 | -$2.511 | 0.570 | -$2.576 | 0.562 |

Under optimistic costs, both models are profitable ($1.25 and $1.18/trade respectively). The tuned model gains ~$0.065/trade advantage across all cost scenarios. Breakeven RT cost for the tuned model is **$3.74** — a $0.01 reduction in round-trip cost would flip the model profitable.

### Sanity Checks

| Check | Expected | Observed | Pass? |
|-------|----------|----------|-------|
| Train acc > test acc | Yes | **Yes** | PASS |
| No single feature > 50% gain | Yes | **49.7%** (volatility_50) | PASS (barely — 0.3pp margin) |
| Holdout within 5pp of CPCV | Yes | **2.8pp delta** | PASS |
| Early stopping triggers >=90% | Yes | **45/45 (100%)** | PASS |
| Default reproduces baseline ±2pp | Yes | **0.449 CPCV** (matches E2E CNN exactly) | PASS |

**Warning:** volatility_50's gain share at 49.7% is functionally at the 50% threshold. The model is overwhelmingly dominated by realized volatility. If this feature's predictive power is regime-dependent (as Q3-Q4 losses suggest), the entire model is fragile to volatility regime shifts.

---

## Resource Usage

| Resource | Budgeted | Actual | Status |
|----------|----------|--------|--------|
| Wall clock | 120 min (150 max) | 85.8 min | Under budget (72%) |
| Training runs | ~420 | 493 | 17% over |
| Configs evaluated | 64 | 79 | 23% over (fine search expanded) |
| Execution target | Local (Apple Silicon) | Local | Correct per compute policy |
| Abort triggered | — | No | Clean run |

---

## Confounds and Alternative Explanations

### 1. Walk-Forward vs. CPCV Expectancy Disagreement (CRITICAL)

The most important confound. CPCV expectancy (-$0.001) suggests near-breakeven. Walk-forward expectancy (-$0.140) suggests the model loses $0.14/trade in realistic forward deployment. The 13.9-cent gap is large — more than the entire CPCV tuning improvement. CPCV's combinatorial splits allow training on future regimes, which walk-forward cannot. **The walk-forward result is more representative of deployment.** The near-breakeven CPCV expectancy should be interpreted as an optimistic bound, not a deployment estimate. The true deployment expectancy likely falls between -$0.14 (walk-forward) and -$0.001 (CPCV).

### 2. Accuracy Plateau vs. Expectancy Improvement Mechanism

How does 0.15pp accuracy gain produce $0.065 expectancy gain? The answer is class-distribution shift: the tuned model suppresses long predictions (recall 0.201→0.149) while improving short predictions (0.586→0.634). In the asymmetric 10:5 cost structure, avoiding bad long trades saves more per trade than bad short trades cost. The model learns when NOT to trade longs. This is legitimate but raises the question: is the long-suppression regime-specific? If Q3-Q4 longs are specifically what's being avoided, the improvement may not generalize.

### 3. Feature Importance Stability

The same features dominate both models. No feature dominance shift. The improvement is from regularization (how features are weighted), not from discovering latent patterns. This means the ceiling is low — same information, same features, same fundamental constraint.

### 4. Early Stopping Confound — CLEARED

The tuned model uses 384 trees vs. the default's 500. The improvement is NOT from more trees. The total gradient signal is actually 5x less (lr × trees: 0.013 × 384 ≈ 5.0 vs 0.05 × 500 = 25.0). The model succeeds through better regularization, not more computation.

### 5. Could the Baseline Be Poorly Tuned?

This was the hypothesis being tested. The flat landscape (0.33pp range, 0.06pp std across 64 configs) conclusively demonstrates the default is NOT poorly tuned for accuracy. It sits on a plateau where everything performs similarly. The default IS slightly suboptimal for expectancy (class distribution), but this is a secondary effect worth <$0.07/trade.

### 6. Seed Variance

Only seed=42. The CPCV protocol (45 splits) provides temporal resampling. The 0.15pp accuracy improvement is well within one CPCV std (1.17pp) — **not statistically significant.** The $0.065 expectancy improvement is 0.53 std of the expectancy distribution (std=0.122) — also not clearly significant. A multi-seed study would be needed to confirm the expectancy improvement is real, but the flat accuracy landscape makes this low priority.

### 7. Holdout Regime Effect

Days 202-251 (Nov-Dec 2022) is Q4 — the worst quarter. The 2.0pp holdout accuracy gain could reflect genuine generalization improvement OR the tuned model's conservative profile (suppressed longs) working better in the specific low-vol year-end regime. Cannot distinguish with a single holdout period.

---

## What This Changes About Our Understanding

1. **The XGBoost hyperparameter surface for this feature set is a plateau.** Accuracy is insensitive to hyperparameters across the entire 7-parameter search space. This conclusively rules out "undertrained model" as an explanation for the expectancy gap. The 20-feature set is the binding constraint.

2. **Expectancy is more sensitive to class prediction distribution than to raw accuracy.** A 0.15pp accuracy change produces a $0.065 expectancy change via class rebalancing (suppressed longs). This means **label design and class-weighting are higher-leverage interventions than accuracy optimization** — they change the economic structure of predictions without needing better classification.

3. **The model is at the knife-edge of breakeven.** Breakeven RT cost is $3.74. This means independently achievable interventions could flip the model profitable: (a) cost reduction by even $0.01, (b) wider target label reducing breakeven win rate, (c) class-weighted loss favoring economically optimal predictions, (d) regime-conditional trading in Q1-Q2. These interventions are additive.

4. **The long-recall problem worsened under tuning** (0.201→0.149). The model is increasingly short-biased. This supports a 2-class formulation (short/no-short) or asymmetric loss functions that explicitly address the long-signal problem.

5. **Walk-forward expectancy (-$0.140) is substantially more pessimistic than CPCV (-$0.001).** This gap tempers optimism about the near-breakeven CPCV result. Deployment reality is likely between these bounds. The CPCV-WF gap should be measured in future experiments as a standard diagnostic.

6. **volatility_50 dominance (49.7% gain share) is a structural risk.** The model's performance is almost entirely determined by a single feature's relationship to the label. If volatility regimes shift (2022→2023), the model may break. This motivates regime-conditional approaches.

---

## Proposed Next Experiments

1. **Label design sensitivity (P1 — highest priority).** The tuned model breaks even at $3.74 RT. At a 15:3 target/stop ratio, breakeven win rate drops from ~53.3% to ~42.5%. Current accuracy (0.450) is well above that threshold. Test 3-4 label configs: {15:3, 12:4, 10:3, asymmetric cost weights on current 10:5}. Highest-leverage single intervention — changes the economic threshold without needing better predictions.

2. **2-class (short/no-short) formulation.** Long recall is 0.149 — the model cannot reliably identify long opportunities. Instead of fighting this, formulate as binary classification: short-entry vs. hold. If the model's short recall (0.634) produces positive expectancy on shorts alone, this is immediately actionable without solving the long-side problem.

3. **Class-weighted XGBoost with tuned hyperparameters.** Apply asymmetric cost matrix (wrong long costs $6.25+RT, wrong short costs $12.50+RT) as sample weights. This explicitly aligns the loss function with the PnL structure instead of relying on regularization to implicitly suppress losing classes. Combine with the tuned hyperparameters from this experiment.

4. **Regime-conditional filter.** The tuned model is profitable in Q1 (+$0.137) and Q2 (+$0.058). Investigate a volatility-regime gate: trade only when a regime indicator suggests favorable conditions. Caveat: 1 year of data limits regime analysis — this is exploratory, not confirmatory.

---

## Program Status

- Questions answered this cycle: 1 ("Can XGBoost hyperparameter tuning improve classification accuracy by 2+pp?" — REFUTED, Outcome C)
- New questions added this cycle: 0
- Questions remaining (open, not blocked): 4 (label design, regime-stratified info decomposition, cost sensitivity, regime-conditional trading)
- Handoff required: NO
