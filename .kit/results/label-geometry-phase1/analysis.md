# Analysis: Label Geometry Phase 1 — Model Training at Breakeven-Favorable Ratios

## Verdict: REFUTED

**Executive Summary:** Bidirectional triple barrier labels on time_5s bars produce a fundamentally degenerate classification problem. At the control geometry (10:5), 90.7% of bars are hold-labeled; the model degenerates into a hold-predictor (hold recall=99.86%, directional recall <2%). All three treatment geometries (15:3, 19:7, 20:3) were skipped because >95% of bars are hold (97.1%, 98.6%, 98.9% respectively). CPCV expectancy is -$0.610/trade. The hypothesis that accuracy would persist under bidirectional labels at favorable payoff ratios is falsified — the prior ~45% accuracy baseline was measured on long-perspective labels with a fundamentally different class distribution.

---

## Results vs. Success Criteria

- [x] SC-1: **PASS** — 1,004 files exported (4 geometries x 251 days, 0 failures)
- [ ] SC-2: **FAIL** — CPCV + WF completed only for 10:5 (control). 15:3, 19:7, 20:3 skipped (degenerate labels >95% hold)
- [x] SC-3: **PASS (FORMALLY) / FAIL (SUBSTANTIVELY)** — 10:5 CPCV accuracy 89.98% > BEV WR 53.28% + 2pp = 55.28%. **But this is a measurement artifact: the 89.98% is trivially achieved by predicting hold on a 90.7% hold dataset. Directional accuracy at holdout is 53.65% — only 0.37pp above breakeven, not >2pp.** SC-3 as defined is misleading for hold-dominated distributions.
- [ ] SC-4: **FAIL** — CPCV expectancy = -$0.610/trade (base costs). No geometry produces positive expectancy.
- [x] SC-5: **PASS** — Holdout expectancy = +$0.069/trade (> -$0.10). But economically trivial: $84 total over 50 days ($1.68/day), on a 0.53% trading rate.
- [x] SC-6: **PASS** — Walk-forward reported for 10:5 (treatment geometries reported as "skipped — degenerate labels")
- [x] SC-7: **PASS** — Per-direction oracle and time-of-day analysis reported for 10:5

**Primary criteria failed: SC-2 and SC-4. SC-3 is a trivial artifact, not a genuine pass.**

---

## Metric-by-Metric Breakdown

### Primary Metrics

#### 1. cpcv_accuracy_per_geometry

| Geometry | CPCV Accuracy | Std | BEV WR | BEV Margin | Status |
|----------|--------------|-----|--------|------------|--------|
| 10:5 (control) | 89.98% | 3.92% | 53.28% | +36.70pp | Trained (degenerate hold-predictor) |
| 15:3 | — | — | 33.33% | — | **SKIPPED** (97.1% hold) |
| 19:7 | — | — | 38.46% | — | **SKIPPED** (98.6% hold) |
| 20:3 | — | — | 29.63% | — | **SKIPPED** (98.9% hold) |

The 36.70pp "breakeven margin" at 10:5 is a spurious metric. The model achieves 89.98% accuracy by predicting hold 99.86% of the time on a 90.7% hold-dominated dataset. The per-class recall tells the real story:

| Class | CPCV Recall | Holdout Recall |
|-------|------------|----------------|
| Short (-1) | 1.87% | 3.95% |
| Hold (0) | 99.86% | 99.79% |
| Long (+1) | 1.62% | 5.40% |

**The model has no directional discrimination capability.** At holdout, it predicts directional on 1,662 of 229,520 bars (0.72%). Of those, 443 (26.7%) hit hold-labeled bars. On the 1,219 trades where both prediction and label are directional, the win rate is 53.65% — only 0.37pp above the 53.28% breakeven. This is indistinguishable from noise.

#### 2. cpcv_expectancy_per_geometry

| Geometry | Optimistic ($2.49) | Base ($3.74) | Pessimistic ($6.25) | PF (base) |
|----------|--------------------|--------------|---------------------|-----------|
| 10:5 | +$0.640 | **-$0.610** | -$3.120 | 0.882 |
| 15:3 | — | — | — | — |
| 19:7 | — | — | — | — |
| 20:3 | — | — | — | — |

CPCV expectancy is negative at base costs. Profitable only under the optimistic scenario ($2.49 RT = zero slippage + reduced commission).

#### 3. walkforward_expectancy_per_geometry

| Geometry | Fold 1 (101-150) | Fold 2 (151-201) | Fold 3 (202-251) | Mean | Total Trades |
|----------|-----------------|-----------------|------------------|------|----------|
| 10:5 | +$0.128 | **-$1.071** | -$0.031 | **-$0.325** | 2,837 |
| 15:3–20:3 | — | — | — | — | — |

Walk-forward tells a clear story: only 1/3 folds profitable. Mean expectancy -$0.325/trade. The CPCV-to-WF divergence (-$0.610 vs -$0.325) is narrower than in the XGB tuning experiment (-$0.001 vs -$0.140) because both metrics are firmly negative here.

Walk-forward per-fold detail:

| Fold | Test Period | Accuracy | Base Exp | PF | Trades | Fit Time |
|------|------------|----------|----------|-----|--------|----------|
| 1 | Days 101-150 | 94.12% | +$0.128 | 1.028 | 795 | 16.6s |
| 2 | Days 151-201 | 94.42% | -$1.071 | 0.795 | 822 | 31.5s |
| 3 | Days 202-251 | 93.95% | -$0.031 | 0.993 | 1,220 | 42.9s |

Fold 2 is sharply negative — corresponds to mid-year (June-August), consistent with prior observations of H2 underperformance.

### Secondary Metrics

#### breakeven_margin_per_geometry

| Geometry | Model "Accuracy" | BEV WR | Formal Margin | Directional Accuracy (holdout) | Real Margin |
|----------|-----------------|--------|---------------|-------------------------------|-------------|
| 10:5 | 89.98% | 53.28% | +36.70pp | 53.65% | **+0.37pp** |
| 15:3–20:3 | SKIPPED | 29.6-38.5% | — | — | — |

The formal margin is meaningless. The proper comparison is holdout directional accuracy (53.65%) vs BEV WR (53.28%) = **+0.37pp**. Not economically viable.

#### class_distribution_per_geometry

| Geometry | Short (-1) | Hold (0) | Long (+1) | Status |
|----------|-----------|----------|-----------|--------|
| 10:5 | 4.68% | **90.70%** | 4.61% | Trained (severely imbalanced) |
| 15:3 | ~1.5% | **~97.1%** | ~1.4% | SKIPPED (>95% hold) |
| 19:7 | ~0.7% | **~98.6%** | ~0.7% | SKIPPED (>95% hold) |
| 20:3 | ~0.6% | **~98.9%** | ~0.5% | SKIPPED (>95% hold) |

**This is the central finding.** Bidirectional triple barrier labels at time_5s bars are overwhelmingly hold-dominated at ALL geometries. The 95% abort threshold caught the worst cases, but even the control (90.7% hold) is degenerate for practical classification.

For context: prior experiments with long-perspective labels achieved ~45% 3-class accuracy, implying a far more balanced class distribution. The switch to bidirectional labels fundamentally changed the classification problem.

#### per_class_recall_per_geometry

Only 10:5 trained. Short recall 1.87%, long recall 1.62% (CPCV). The model has learned to predict hold almost exclusively.

#### per_direction_oracle

| Geometry | Long Triggers | Long WR | Long Exp | Short Triggers | Short WR | Short Exp | Both Rate |
|----------|--------------|---------|----------|----------------|----------|-----------|-----------|
| 10:5 | 55,196 | 96.96% | $8.19 | 56,005 | 97.00% | $8.20 | 0.14% |

The oracle achieves near-perfect directional accuracy (~97%) on the ~9.3% of bars that have directional labels. Long and short are highly symmetric (triggers, WR, and expectancy within <1%). Both-triggered rate is negligible (0.14%). **The directional signal exists in the data — the model simply cannot extract it from the overwhelmingly hold-dominated distribution.**

#### both_triggered_rate

0.14% — negligible. Bidirectional races rarely conflict.

#### time_of_day_breakdown

| Band | N Bars | % of Total | Directional Rate | Short % | Hold % | Long % |
|------|--------|-----------|-----------------|---------|--------|--------|
| A: Open (09:30-10:00) | 77,810 | 6.71% | **16.41%** | 8.40% | 83.59% | 8.01% |
| B: Mid (10:00-15:00) | 902,340 | 77.78% | 9.34% | 4.69% | 90.66% | 4.65% |
| C: Close (15:00-15:30) | 90,000 | 7.76% | **5.82%** | 2.99% | 94.18% | 2.83% |

Opening range has 2.8x the directional rate vs close, 1.76x vs mid-session. Higher volatility at the open generates more barrier triggers within 5-second windows. Band C is nearly as degenerate as the treatment geometries (94.18% hold). Model accuracy by band was not reported (only label distributions) — a gap.

#### profit_factor_per_geometry

| Geometry | CPCV PF (base) | Holdout PF (base) |
|----------|---------------|-------------------|
| 10:5 | 0.882 | 1.015 |

CPCV profit factor < 1 (losing). Holdout PF = 1.015 (breakeven within noise). Walk-forward folds range from 0.795 (Fold 2, losing) to 1.028 (Fold 1, marginally winning).

#### feature_importance_shift

| Rank | Feature | Gain | Description |
|------|---------|------|-------------|
| 1 | f11 | 133.5 | volatility_50 |
| 2 | f10 | 58.5 | volatility_20 |
| 3 | f12 | 17.3 | high_low_range_50 |
| 4 | f15 | 7.8 | message_rate |
| 5 | f5 | 6.3 | avg_trade_size |
| 6 | f1 | 6.3 | spread |
| 7 | f16 | 5.9 | — |
| 8 | f17 | 5.3 | — |
| 9 | f19 | 5.2 | — |
| 10 | f18 | 5.2 | — |

Volatility features (f11 + f10 = 192.0 combined gain, ~76% of top-10) dominate overwhelmingly. The model has learned a volatility threshold for "will any barrier trigger at all?" — not a directional signal. This is the rational strategy for minimizing logloss on 91% hold data. No cross-geometry comparison possible (3/4 skipped). SC-S2 passes: f11 ≈ 53% of top-10 gain, under 60%.

#### cost_sensitivity

| Scenario | RT Cost | CPCV Exp | CPCV PF | WF Exp |
|----------|---------|----------|---------|--------|
| Optimistic | $2.49 | +$0.640 | 1.152 | +$0.925 |
| Base | $3.74 | -$0.610 | 0.882 | -$0.325 |
| Pessimistic | $6.25 | -$3.120 | 0.503 | -$2.835 |

Profitable only under optimistic costs (zero slippage). The base-to-optimistic flip occurs at ~$3.11 RT cost.

#### long_recall_vs_short

| Eval | Short Recall | Long Recall | Ratio |
|------|-------------|-------------|-------|
| CPCV | 1.87% | 1.62% | 1.15:1 |
| Holdout | 3.95% | 5.40% | 0.73:1 |

Slight asymmetry (holdout favors longs) but both are <6%, rendering the distinction immaterial.

#### holdout_expectancy

Reported in full in the Holdout section below.

### Sanity Checks

- [x] **SC-S1:** Baseline (10:5) CPCV accuracy 89.98% > 0.40 — **PASS.** Trivially so (90.7% hold class).
- [x] **SC-S2:** No single feature > 60% gain share — **PASS.** f11 (volatility_50) ≈ 53% of top-10 gain.
- [x] **SC-S3:** Re-export produces 152 columns — **PASS.**
- [ ] **SC-S4:** No >95% single class — **FAIL.** 15:3=97.1%, 19:7=98.6%, 20:3=98.9% hold. Even 10:5 (90.7%) is borderline degenerate.

SC-S4 failure is the most consequential result. It reveals a structural problem with bidirectional labels at the time_5s scale.

---

## Holdout Evaluation

**10:5 (only geometry trained) — days 202-251, trained on full dev set (201 days):**

| Metric | Optimistic | Base | Pessimistic |
|--------|-----------|------|-------------|
| Expectancy | +$1.319 | **+$0.069** | -$2.441 |
| Profit Factor | 1.326 | **1.015** | 0.579 |
| N Trades | 1,219 | 1,219 | 1,219 |

Confusion matrix:

```
              pred=-1    pred=0    pred=+1
true=-1:        288      6,665       343     (7,296 true shorts)
true=0:         202    215,002       241     (215,445 true holds)
true=+1:        222      6,191       366     (6,779 true longs)
```

**Derived metrics:**
- Directional predictions: 712 short + 950 long = 1,662 (0.72% of 229,520 bars)
- Hold predictions: 227,858 (99.28% of bars)
- Directional trades (both pred and label nonzero): 654 correct + 565 wrong = **1,219**
- Directional win rate: 654/1,219 = **53.65%** (breakeven = 53.28%, margin = **+0.37pp**)
- Label=0 hit rate on directional predictions: 443/1,662 = **26.7%** (flagged at >25% per spec)
- Total holdout PnL: 654 × $8.76 + 565 × (-$9.99) = +$84.69 over 50 days = **$1.69/day**
- Trading rate: 1,219/229,520 = **0.53%** of bars

The holdout is marginally positive under base costs (+$0.069/trade) but economically trivial. The 0.37pp directional margin above breakeven, on 1,219 trades, is within sampling noise. The 26.7% label-0 hit rate exceeds the 25% reliability threshold — over a quarter of directional predictions land on bars where neither barrier was triggered, representing undefined real-world exits.

---

## Resource Usage

| Resource | Budgeted | Actual | Utilization |
|----------|----------|--------|-------------|
| Wall clock | 240 min (4h) | **27.4 min** | 11.4% |
| CPCV fits | 180 (4×45) | **45** (1×45) | 25% |
| WF fits | 12 (4×3) | **3** (1×3) | 25% |
| Holdout fits | 2 | **1** | 50% |
| Export runs | 1,004 | **1,004** | 100% |
| GPU hours | 0 | 0 | — |

Under-utilization reflects protocol-compliant skipping of degenerate geometries. The budget was appropriate for the planned 4-geometry sweep; the experiment terminated early due to label degeneracy, not resource exhaustion.

---

## Confounds and Alternative Explanations

### 1. Bidirectional vs. Long-Perspective Label Distribution (CRITICAL — ROOT CAUSE)

The prior ~45% accuracy baseline was measured on **long-perspective** labels (`compute_tb_label()`). Those labels apparently had a more balanced class distribution — necessary to support 45% accuracy on a 3-class problem. Bidirectional labels (`compute_bidirectional_tb_label()`) require the price to reach the target without hitting the stop in **either** direction within the bar's lifespan. On 5-second bars, most price paths don't traverse 10 ticks ($2.50) in either direction within 5 seconds, producing ~91% hold at 10:5.

**This is not a confound — it is the finding.** The experiment revealed that the label type fundamentally changes the classification problem. The synthesis-v2 conclusion that "the model's ~45% accuracy would persist under high-ratio geometries" was premised on a label type that produces balanced distributions. Bidirectional labels do not.

### 2. Bar Duration × Barrier Width Mismatch

The 5-second bar window constrains price movement. At 10:5 (target=$2.50, stop=$1.25), MES needs a $2.50 move in ≤5s without first moving $1.25 against — a rare event (~9.3% of bars). At 20:3 (target=$5.00, stop=$0.75), the target is $5.00 in 5s while the stop is within bid-ask noise ($0.75 = 0.6 ticks of spread). The 98.9% hold rate is physically inevitable at this timescale.

**Could longer bars fix this?** Possibly — 30s or 60s bars provide more price movement per window. But this changes the bar type (locked at time_5s since R1/R6) and introduces a new set of trade-offs.

### 3. Hold-Dominated Training Dynamics

XGBoost on 91/4.7/4.6 class splits learns a volatility threshold (predict hold unless volatility is extreme). This minimizes logloss but is useless for trading. Class weighting could force more directional predictions, but at the cost of introducing false directional signals on genuinely flat bars.

### 4. Holdout Flattery

The holdout (+$0.069/trade) is the most favorable evaluation window. CPCV (-$0.610) and walk-forward mean (-$0.325) are both firmly negative. Walk-forward Fold 1 (+$0.128) covers days 101-150 only. The pattern — one favorable period surrounded by losses — is consistent with noise, not a real edge.

### 5. Label=0 Hit Rate at 26.7%

26.7% of directional predictions land on hold-labeled bars (assigned $0 PnL per spec). In practice, these are positions on bars where neither barrier triggers — the exit is undefined. A more realistic PnL model for these trades would likely worsen reported performance.

### 6. Could the Prior ~45% Accuracy on Long-Perspective Labels Be the Anomaly?

The bidirectional labels may be "more correct" (symmetric treatment of longs and shorts) even though they produce degenerate distributions. If so, the prior ~45% accuracy was achieved on a distribution that doesn't reflect the actual market dynamics within 5-second windows. The long-perspective labels may have been generating spurious directional labels (labeling +1 when the price eventually reaches target over multiple bars). This deserves investigation.

---

## What This Changes About Our Understanding

### 1. The Bidirectional Label Premise Was Wrong for 3-Class XGBoost on time_5s

The project invested substantial engineering effort into bidirectional triple barrier labels (PRs #26, #27) under the assumption that symmetric long/short races would produce better labels. For the oracle, they do (97% WR, symmetric long/short). For model training on time_5s bars, they produce a degenerate classification problem where ~91-99% of bars are hold.

**Root cause:** Bidirectional labels are computed within a single bar's price window (5 seconds). Most 5-second windows don't contain enough price movement to trigger any barrier. Long-perspective labels likely computed barriers over a forward-looking multi-bar window, allowing more time for the price to reach targets. This produces balanced distributions suitable for classification.

### 2. The Geometry Hypothesis Remains Untested on Viable Labels

The core hypothesis — that favorable payoff ratios can convert marginal accuracy into positive expectancy — was **never tested**. It was blocked by the degenerate label distribution. The hypothesis should be re-tested on long-perspective labels (or bidirectional labels on longer bars) where the model can learn directional discrimination.

### 3. Volatility Is the Real Feature on This Problem

The model's near-exclusive reliance on volatility features (f11+f10 = 76% of top-10 gain) confirms it learned "will any barrier trigger?" not "which direction?" This is the rational logloss-minimizing strategy for hold-dominated data, but useless for directional trading.

### 4. SC-3 Definition Was Flawed

SC-3 ("CPCV accuracy > BEV WR + 2pp") is meaningless on hold-dominated distributions. Future experiments must define success in terms of **directional accuracy** (accuracy among only directional predictions and directional labels), not overall 3-class accuracy that includes hold predictions.

### 5. The Label-Type Decision Is Upstream of Geometry

Before varying geometry, the label type must produce a viable class distribution. The current bidirectional + time_5s combination is structurally unable to support classification at any geometry tested. Either the label type or the bar duration must change first.

---

## Proposed Next Experiments

### 1. Geometry Sweep on Long-Perspective Labels (HIGHEST PRIORITY)

Re-run the geometry hypothesis using long-perspective labels (`compute_tb_label()`) which produce balanced class distributions. This isolates the geometry effect from the label-type confound. The hypothesis — favorable payoff ratios convert ~45% accuracy into positive expectancy — remains the most promising path and has never been tested.

**Key prerequisite:** Verify that `bar_feature_export --target T --stop S` with `--legacy-labels` produces balanced (not hold-dominated) distributions at varied geometries. Export 1 day at 10:5 and 15:3 with legacy labels and check.

### 2. Characterize Label Distribution by Label Type × Geometry

Before running a full sweep, export a single day at 4-6 geometries under both label types and tabulate class distributions. This is a 30-minute investigation that would prevent another wasted experiment.

### 3. Bidirectional Labels on Longer Bars (LOWER PRIORITY)

If bidirectional symmetry is important, test on 30s or 60s bars where the price has more time to trigger barriers. This requires relaxing the time_5s constraint from R1/R6. Given that R1 REFUTED the subordination hypothesis and R6 recommended time_5s, this change has weak justification unless it is reframed as "choosing the right timescale for the label type."

### 4. 2-Class Formulation (SPECULATIVE)

Collapse to binary: "directional vs hold." The 3-class problem is degenerate at 91%+ hold. A staged approach (first predict if any barrier will trigger, then predict direction among triggered bars) could separate the two tasks. However, this adds pipeline complexity.

---

## Program Status

- Questions answered this cycle: 1 (P0: XGBoost at high-ratio geometries → REFUTED due to degenerate bidirectional labels)
- New questions added this cycle: 2 (long-perspective geometry sweep viability, bidirectional-on-longer-bars viability)
- Questions remaining (open, not blocked): 4 (prior 3 + 1 new)
- Handoff required: **NO** — the fix is within research scope (change label type or bar duration in experiment spec)

---

## Explicit SC-1 through SC-7 Evaluation

| SC | Description | Result | Evidence |
|----|-------------|--------|----------|
| SC-1 | Re-export ≥1,000 files | **PASS** | 1,004 files (4×251), 0 failures |
| SC-2 | CPCV + WF for all 4 geometries | **FAIL** | Only 10:5 trained; 3 skipped (>95% hold) |
| SC-3 | At least one CPCV accuracy > BEV WR + 2pp | **PASS (trivial artifact)** | 89.98% > 55.28%; but hold-prediction on 90.7% hold data; directional acc = 53.65% (+0.37pp) |
| SC-4 | At least one CPCV expectancy > $0.00 | **FAIL** | Best = -$0.610/trade (10:5) |
| SC-5 | Holdout expectancy > -$0.10 | **PASS** | +$0.069/trade; economically trivial ($1.69/day) |
| SC-6 | Walk-forward reported all geometries | **PASS** | 10:5 reported; 3 reported as skipped |
| SC-7 | Per-direction + time-of-day reported | **PASS** | Reported for 10:5 |

**Primary criteria SC-2 and SC-4 FAIL. SC-3 is a measurement artifact.**

---

## Outcome Verdict

**REFUTED.** None of the three pre-committed outcomes match exactly:

- **Not Outcome A** — SC-4 fails (no positive expectancy at any geometry)
- **Not Outcome B as written** — "accuracy stable across geometries" cannot be assessed (3/4 skipped). The model doesn't have a "fixed directional signal" — it has effectively zero directional signal.
- **Not Outcome C as written** — "accuracy drops >10pp at non-baseline" cannot be assessed (3/4 skipped)

The actual outcome is **more fundamental than any of the three**: the bidirectional label type produces a degenerate classification problem at ALL tested geometries on time_5s bars. The hypothesis was **untestable under these conditions.** This is REFUTED because:

1. The mechanism is falsified — model accuracy does NOT persist at ~45% under bidirectional labels; it is trivially inflated to ~90% by hold-prediction on a ~91% hold dataset
2. SC-4 (the economically meaningful criterion) clearly fails
3. The directional accuracy (53.65% holdout) is statistically indistinguishable from breakeven (53.28%)
4. 3/4 geometries are physically degenerate at this timescale
5. The only profitable evaluation window (holdout +$0.069) produces $1.69/day on 0.53% trading rate — economically trivial

**The geometry hypothesis survives in principle** — it was never properly tested. It requires viable labels (long-perspective, or bidirectional on longer bars) where ~45% directional accuracy is achievable. The next step is to verify long-perspective label distributions at varied geometries, then re-run the sweep.
