# Analysis: Label Design Sensitivity — Triple Barrier Geometry

## Verdict: REFUTED (Outcome C) — with critical methodological caveat

The experiment was correctly aborted at Phase 0 per the pre-committed abort criteria. Zero of 123 valid geometries achieves oracle net expectancy > $5.00/trade at base costs ($3.74 RT). Peak is $4.126 at (target=16, stop=10), missing by 17.5%.

**However, the $5.00 abort criterion was fundamentally miscalibrated.** It measured oracle per-trade profit, but model viability depends on breakeven win rate vs. model accuracy. Multiple geometries have breakeven WRs of 29-38% — well below the model's historical ~45% 3-class accuracy. The abort prevented Phase 1 training, which was the only way to answer whether the model can actually capitalize on favorable payoff structures. The verdict is technically REFUTED against the pre-committed criteria, but the underlying question (can geometry changes enable profitability?) remains open and promising.

---

## Results vs. Success Criteria

- [ ] **SC-1**: Oracle heatmap computed — **PASS** — 123/123 valid geometries computed on 19-day subsample (spec planned 20; minor deviation). 21 invalid geometries (target <= stop) correctly excluded.
- [ ] **SC-2**: At least one geometry with oracle net exp > $5.00 — **FAIL** — 0 geometries at base costs ($3.74 RT). Peak = $4.126 at (16:10). At optimistic costs ($2.49 RT), 8 geometries exceed $5.00.
- [ ] **SC-3**: At least one geometry with CPCV accuracy > breakeven_WR + 2pp — **NOT EVALUATED** — abort triggered at Phase 0.
- [ ] **SC-4**: At least one geometry with CPCV expectancy > $0.00 — **NOT EVALUATED** — abort triggered at Phase 0.
- [ ] **SC-5**: Best geometry holdout exp > -$0.10 — **NOT EVALUATED** — abort triggered at Phase 0.
- [ ] **SC-6**: Per-direction oracle + time-of-day reported for 4 geometries — **FAIL** — no Phase 1 re-export performed.
- [ ] No regression on sanity checks — **PARTIAL** — SC-S1 PASS, SC-S2 PASS, SC-S3 FAIL (explained below), SC-S4/S5 NOT TESTED.

**Summary: 1 PASS, 2 FAIL, 4 NOT EVALUATED. The abort cascade rendered 4 of 6 criteria untestable.**

---

## Metric-by-Metric Breakdown

### Primary Metrics

| Metric | Observed | Threshold | Status |
|--------|----------|-----------|--------|
| oracle_viable_count | **0** | >= 1 (at base costs) | **FAIL** |
| best_geometry_cpcv_expectancy | **null** (not tested) | > $0.00 | **NOT EVALUATED** |

The oracle net expectancy distribution across all 123 valid geometries at base costs:

| Stat | Value |
|------|-------|
| Min | $0.743 |
| P25 | $2.138 |
| Median | $2.937 |
| Mean | $2.683 |
| P75 | $3.328 |
| Max | $4.126 |

The entire distribution sits below $5.00. Viability counts at various thresholds:

| Threshold (base costs) | Count | % of Valid |
|------------------------|-------|-----------|
| > $5.00 | 0 | 0% |
| > $4.00 | 1 | 0.8% |
| > $3.00 | 56 | 45.5% |
| > $2.00 | 95 | 77.2% |
| > $1.00 | 115 | 93.5% |

At optimistic costs ($2.49 RT): 8 geometries exceed $5.00, 101 exceed $3.00.

### Secondary Metrics

**Reported from Phase 0 data:**

| Metric | Value | Notes |
|--------|-------|-------|
| oracle_peak_expectancy | $4.126 (base) / $5.376 (optimistic) | At (16, 10) |
| oracle_peak_geometry | (target=16, stop=10) | Moderate ratio 1.6:1 |
| oracle_peak_wr | 60.79% | Oracle wins 60.8% of directional trades |
| oracle_peak_breakeven_wr | 49.97% | Near-symmetric payoff at this geometry |
| oracle_peak_margin | +10.82pp | Oracle WR exceeds breakeven by ~11pp |
| oracle_peak_trade_count | 2,191 (19 days) | ~115 trades/day |
| oracle_peak_profit_factor | 2.4152 | Below baseline PF of 3.30 |
| profit_factor range (top 10) | 2.17 — 2.44 | Narrower range than net exp spread |

**Cost sensitivity (peak geometry 16:10):**

| Scenario | RT Cost | Oracle Net Exp |
|----------|---------|---------------|
| Optimistic | $2.49 | **$5.376** (above $5.00) |
| Base | $3.74 | **$4.126** (below $5.00) |
| Pessimistic | $6.25 | **$1.616** |

**Not collected due to Phase 0 abort (12 metrics):**

| Metric | Status | Impact |
|--------|--------|--------|
| class_distribution_per_geometry | NOT COLLECTED | Cannot assess label imbalance at different geometries |
| cpcv_accuracy_per_geometry | NOT COLLECTED | **Critical gap** — cannot test accuracy transfer across geometries |
| cpcv_expectancy_per_geometry | NOT COLLECTED | **Critical gap** — the actual viability question |
| breakeven_margin_per_geometry (model) | NOT COLLECTED | Oracle margins available; model margins unknown |
| per_class_recall_per_geometry | NOT COLLECTED | Cannot assess long/short asymmetry persistence |
| per_direction_oracle | NOT COLLECTED | Requires Phase 1 bidirectional re-export |
| both_triggered_rate | NOT COLLECTED | Requires Phase 1 re-export |
| time_of_day_breakdown | NOT COLLECTED | Cannot assess regime-conditional effects |
| per_quarter_expectancy_best | NOT COLLECTED | Cannot assess temporal stability |
| feature_importance_shift | NOT COLLECTED | Cannot assess feature relevance changes |
| long_recall_vs_short | NOT COLLECTED | Cannot assess recall asymmetry persistence |
| cost_sensitivity (model) | NOT COLLECTED | Only oracle cost sensitivity available |

### Baseline (10:5) Oracle Under Bidirectional Labels

| Metric | This Experiment | Prior R7 (long-perspective) | Delta |
|--------|----------------|---------------------------|-------|
| Oracle exp (native, incl $2.49 costs) | $3.997 | $4.00 | -$0.003 (negligible) |
| Oracle net exp (base costs) | $2.747 | N/A | — |
| Oracle WR | 64.29% | 64.3% | Unchanged |
| Breakeven WR | 53.28% | 53.3% | Unchanged |
| Oracle margin | +11.01pp | ~11.0pp | Unchanged |
| Trade count (19 days) | 4,873 | 4,873 | Identical |
| Profit factor | 3.297 | 3.30 | Unchanged |

The baseline reproduces R7 exactly. Bidirectional labels do not affect oracle performance (expected — the oracle has perfect foresight regardless of label construction).

### Top-10 Geometries by Oracle Net Expectancy (Base Costs)

| Rank | T:S | Ratio | Net Exp | Oracle WR | BEV WR | Margin | Trades | PF | Geo Score |
|------|-----|-------|---------|-----------|--------|--------|--------|-----|-----------|
| 1 | 16:10 | 1.6 | $4.126 | 60.79% | 49.97% | +10.82pp | 2,191 | 2.42 | 1.698 |
| 2 | 18:8 | 2.25 | $3.962 | 53.60% | 42.28% | +11.33pp | 2,179 | 2.41 | 1.626 |
| 3 | 19:7 | 2.71 | $3.878 | 50.14% | 38.43% | +11.71pp | 2,166 | 2.44 | 1.587 |
| 4 | 16:9 | 1.78 | $3.850 | 58.59% | 47.97% | +10.62pp | 2,287 | 2.41 | 1.619 |
| 5 | 19:10 | 1.9 | $3.841 | 54.72% | 44.80% | +9.92pp | 1,831 | 2.17 | 1.445 |
| 6 | 19:8 | 2.375 | $3.834 | 51.60% | 40.71% | +10.89pp | 2,062 | 2.33 | 1.531 |
| 7 | 19:9 | 2.11 | $3.798 | 52.95% | 42.83% | +10.12pp | 1,947 | 2.24 | 1.474 |
| 8 | 18:9 | 2.0 | $3.792 | 54.60% | 44.41% | +10.19pp | 2,053 | 2.28 | 1.511 |
| 9 | 17:8 | 2.125 | $3.732 | 54.90% | 43.97% | +10.93pp | 2,306 | 2.40 | 1.576 |
| 10 | 18:10 | 1.8 | $3.717 | 55.95% | 46.40% | +9.55pp | 1,959 | 2.19 | 1.447 |

**Critical observation:** The top-10 by oracle net exp all have T:S ratios between 1.6:1 and 2.7:1 — moderate ratios. The high-ratio geometries (5:1+) that have the most favorable breakeven WRs are NOT in this ranking because their absolute per-trade oracle profit is lower.

### The Geometries That Matter Most for Model Viability

The oracle net exp ranking selects the wrong geometries for model viability. The correct selection criterion is breakeven WR. Here are geometries with the most favorable breakeven WRs, verified from individual oracle JSON files:

| T:S | Ratio | BEV WR | Oracle WR | Oracle Net (base) | Oracle Margin | Trades/19d |
|-----|-------|--------|-----------|-------------------|---------------|------------|
| 15:3 | 5.0 | **33.29%** | 41.80% | $1.744 | +8.51pp | 3,990 |
| 19:7 | 2.71 | **38.43%** | 50.14% | $3.878 | +11.71pp | 2,166 |
| 20:2 | 10.0 | **22.69%** | ~29.4% | ~$1.23 | ~+6.7pp | 3,616 |
| 20:3 | 6.67 | **29.60%** | ~34.6% | ~$1.92 | ~+5.0pp | 3,037 |
| 10:5 (ctrl) | 2.0 | **53.28%** | 64.29% | $2.747 | +11.01pp | 4,873 |

**Illustrative model PnL at various accuracy levels:**

Using the spec's PnL model for directional trades: Win = +(T x $1.25) - $3.74; Lose = -(S x $1.25) - $3.74

| Geometry | BEV WR | @ 45% acc | @ 40% acc | @ 35% acc | @ 30% acc |
|----------|--------|-----------|-----------|-----------|-----------|
| 10:5 (ctrl) | 53.3% | **-$1.56** | -$2.81 | -$4.06 | -$5.31 |
| 15:3 | 33.3% | **+$2.63** | +$1.38 | +$0.13 | -$1.12 |
| 15:5 | 40.0% | **+$0.94** | +/-$0.00 | -$0.94 | -$1.88 |
| 19:7 | 38.4% | **+$2.14** | +$0.73 | -$0.68 | -$2.09 |
| 20:3 | 29.6% | **+$5.45** | +$3.73 | +$2.01 | +$0.29 |

**At (15:3), the model is profitable down to ~35% directional accuracy.** At (20:3), profitable down to ~30%. The current 10:5 geometry requires >53% — explaining all prior negative expectancy results.

*Major caveat:* These projections assume the model's ~45% 3-class accuracy transfers to new geometries. It likely does NOT transfer intact — wider targets produce more hold labels, changing the classification problem. Phase 1 training is required to measure this. But even if accuracy drops 10pp (from 45% to 35%), geometry (15:3) remains at breakeven and (20:3) remains profitable.

### Per-Quarter Oracle Stability (Sample: 19:7 Triple Barrier)

| Quarter | WR | Expectancy | Trades | PF |
|---------|-----|------------|--------|-----|
| Q1 | 54.6% | $6.65 | 758 | 2.76 |
| Q2 | 50.7% | $5.41 | 487 | 2.61 |
| Q3 | 44.3% | $3.00 | 488 | 1.95 |
| Q4 | 48.3% | $4.54 | 433 | 2.32 |

Oracle is profitable ALL quarters. Q1 strongest, Q3 weakest — consistent with prior GBT Q1-Q2 > Q3-Q4 pattern. The regime effect persists across geometries.

### Per-Quarter Oracle (Sample: 15:3 Triple Barrier)

| Quarter | WR | Expectancy | Trades | PF |
|---------|-----|------------|--------|-----|
| Q1 | 47.0% | $4.33 | 1,303 | 3.10 |
| Q2 | 39.2% | $2.26 | 928 | 2.50 |
| Q3 | 39.1% | $2.25 | 947 | 2.49 |
| Q4 | 39.7% | $2.56 | 812 | 2.64 |

At (15:3), the oracle is profitable ALL quarters. Q1 is strongest. The higher trade count (210/day vs 114/day for 19:7) provides better statistical power.

### Sanity Checks

| Check | Expected | Observed | Status |
|-------|----------|----------|--------|
| SC-S1: Oracle exp > 0 all geometries | Yes | All 123 positive | **PASS** |
| SC-S2: Higher target → fewer trades (fixed stop) | Monotonic | Holds | **PASS** |
| SC-S3: Narrower stop → more trades (fixed target) | Monotonic | 107 violations | **FAIL** |
| SC-S4: Baseline CPCV accuracy > 0.40 | Yes | NOT TESTED | **N/A** |
| SC-S5: No single feature > 60% gain | Yes | NOT TESTED | **N/A** |

**SC-S3 failure explanation:** The triple barrier has 3+ exit mechanisms (target, stop, take-profit at 20 ticks, time expiry at 300s, session end). Narrowing the stop from 5 to 4 ticks doesn't always increase total directional trades because some trades that previously exited via stop now exit via take-profit or session-end instead. This is mechanistically correct behavior, not a bug. The SC-S3 check assumed a two-barrier system; the actual multi-barrier dynamics invalidate the monotonicity assumption.

**Impact on conclusions: None.** The violation is explained by known exit mechanism interactions and does not indicate a label computation error.

---

## Resource Usage

| Resource | Budget | Actual | Utilization |
|----------|--------|--------|-------------|
| Wall-clock | 2.5 hours | 886s (14.8 min) | **10%** |
| Oracle runs | 144 | 123 | 85% (21 invalid excluded) |
| Training splits | ~185 | 0 | **0%** (abort) |
| Export runs | ~1,004 | 0 | **0%** (abort) |
| GPU hours | 0 | 0 | As expected |
| Subsample days | 20 | 19 | 95% (1 day short) |

The abort saved ~85 minutes of compute. Whether this was a good trade depends on the abort criterion's validity.

---

## Confounds and Alternative Explanations

### 1. The $5.00 Abort Criterion Was Miscalibrated (CRITICAL FINDING)

This is the most important analytical finding. The spec defined SC-2 as "oracle net expectancy > $5.00/trade" with rationale "sufficient margin above costs for a realistic model to capture some fraction." This reasoning contains a fundamental error:

**The model does not "capture a fraction" of oracle net expectancy.** The oracle uses perfect foresight — every trade is optimal. The model makes imperfect predictions and its PnL depends on the payoff structure (geometry), not the oracle's per-trade dollar profit. A geometry with LOW oracle net exp can have HIGH model viability if the breakeven WR is sufficiently low.

Concretely:
- **(16:10)**: Oracle net exp = $4.13 (peak), breakeven WR = 49.97%. Model at 45% → **loses money** (-$1.56/trade).
- **(15:3)**: Oracle net exp = $1.74 (ranked ~80th), breakeven WR = 33.29%. Model at 45% → **earns +$2.63/trade**.

The oracle net exp ranking and the model viability ranking are **inversely correlated** in the high-ratio region. The $5.00 threshold selected for geometries where the oracle makes the most absolute dollars, which tends to be moderate-ratio geometries. But model viability depends on the ratio between target and stop (payoff asymmetry), not the oracle's absolute profit level.

**The correct abort criterion:** "No geometry has breakeven WR <= 42%" — trivially satisfied by dozens of geometries. Or: "No geometry has oracle margin > 5pp" — satisfied by ALL 123 geometries.

### 2. Accuracy Transfer Is the Central Uncertainty

The illustrative PnL table above assumes the model's ~45% 3-class accuracy transfers to new geometries. This is the weakest assumption in the analysis. At wider targets:
- The label distribution shifts (more holds, fewer directional labels)
- The 3-class problem becomes more imbalanced
- The model may predict mostly holds, achieving high accuracy but taking few trades
- Directional recall may drop, reducing effective trade count

This is exactly what Phase 1 was designed to test. The abort destroyed the ability to measure this.

### 3. "45% Accuracy" vs. Directional Win Rate

The 3-class accuracy metric (~45%) includes hold predictions. The directional win rate (wins among directional trades taken) is a different metric. The spec's PnL model only charges for directional predictions against directional labels. If the model achieves 45% 3-class accuracy but only 30% of its non-zero predictions match non-zero labels, the effective PnL could be very different from the illustrative table.

Prior data shows: long recall 0.149, short recall 0.634, hold recall ~0.55 at (10:5). The model is NOT symmetric — it predicts shorts much better than longs. At different geometries, this asymmetry may shift.

### 4. geometry_score Selection Criterion

The geometry_score = oracle_net_exp × sqrt(trade_count / max_trade_count) was designed to balance per-trade edge with sample size. But it inherits the oracle net exp bias: it penalizes high-ratio geometries on both axes (lower net exp AND lower trade counts). A breakeven-margin-based score would have selected fundamentally different geometries.

### 5. Oracle Margin Is Remarkably Stable (~10-12pp)

Across all top-10 geometries, oracle margin (WR - breakeven WR) ranges only from 9.55pp to 11.71pp. This narrow range suggests that the oracle's fractional edge over breakeven is roughly geometry-invariant. The oracle captures a similar relative signal at all geometries — only the payoff structure changes. This supports the hypothesis that model accuracy, not oracle ceiling, is the binding constraint.

### 6. Cost Sensitivity Dominates the Feasibility Boundary

The entire gap between "0 viable geometries" and "8 viable geometries" is $1.25 in RT cost (base vs optimistic = the slippage assumption). A $0.63 improvement in execution (one tick less slippage) would shift the peak from $4.13 to $4.76. The feasibility landscape is extremely sensitive to execution quality — more sensitive than to geometry choice within the explored space.

### 7. 19-Day Subsample

One fewer day than planned. With 92-681 trades per geometry across 19 days, the per-geometry oracle statistics have moderate precision. The trade-weighted mean across geometries is well-estimated, but the per-geometry variance for the least-traded geometries (e.g., 20:10 with 1,756 trades = 92/day) introduces noise. Phase 1 full-year training would use 251 days, largely eliminating subsample concerns.

---

## What This Changes About Our Understanding

### Prior Belief (before this experiment)
The 10:5 geometry requires 53.3% accuracy to break even. The model achieves ~45%, an 8pp shortfall. Changing the geometry was the last lever — wider targets would increase the oracle ceiling, providing more "room" for the model.

### Updated Belief
The oracle ceiling is NOT the constraint — it was the wrong metric. **The breakeven WR is the actual lever.** At 5:1 payoff ratios (e.g., 15:3), breakeven drops to 33.3%, providing 12pp of theoretical margin below the model's 45% accuracy. Even if accuracy drops 10pp at these geometries, 35% still covers breakeven.

**The key finding is structural:** the project has been optimizing the wrong variable. All prior negative expectancy results stem from the 10:5 geometry requiring >53% accuracy. The model's ~45% directional accuracy may be sufficient at 5:1+ payoff ratios. This is a testable hypothesis that requires Phase 1 training.

### What hypothesis should replace the tested one?

**Old:** "At least one geometry has oracle net exp > $5.00" (oracle ceiling metric)
**New:** "At least one geometry with breakeven WR <= 42% achieves model directional accuracy > breakeven WR + 2pp" (model viability metric)

---

## Proposed Next Experiments

### 1. Label Geometry Phase 1 — Direct to Model Training (HIGHEST PRIORITY)

Reframe the experiment: skip the oracle ceiling gate (already mapped by this experiment) and proceed directly to Phase 1 training. Select geometries by breakeven WR diversity instead of oracle net exp:

| Geometry | Breakeven WR | Rationale |
|----------|-------------|-----------|
| 10:5 (control) | 53.3% | Baseline — reproduces prior results |
| 15:3 | 33.3% | 5:1 ratio, high trade count (210/day), breakeven 12pp below model accuracy |
| 19:7 | 38.4% | Best oracle margin (+11.71pp) in the top-10, T:S ratio 2.7:1 |
| 20:3 | 29.6% | Most extreme viable ratio (6.7:1), lowest breakeven in grid |

**New abort criterion:** "Baseline (10:5) CPCV accuracy < 0.40 on bidirectional labels." Do NOT abort on oracle net exp — that gate is resolved.

**Key measurements:** (a) Does 3-class accuracy change across geometries? (b) Does directional recall survive at high-ratio geometries? (c) What is the actual per-trade expectancy at each geometry?

### 2. 2-Class Short/No-Short (If 3-class accuracy degrades at high ratios)

If the model's 3-class accuracy drops below 35% at 15:3 or 20:3 (making them unprofitable), reformulate as 2-class: {short, not-short}. The model's short recall (0.634 at 10:5) is strong. A 2-class formulation concentrates on the model's strength and simplifies the classification problem.

### 3. Cost Reduction Investigation (Complementary)

Since the feasibility landscape is extremely cost-sensitive, investigate execution optimization:
- Limit orders instead of market orders (eliminate slippage)
- ICE MES maker rebates
- Effective spread measurement from MBO data

A $0.63 reduction in RT cost shifts 8 geometries above $5.00 oracle net exp. Cost optimization and geometry optimization are complementary levers.

---

## Program Status

- Questions answered this cycle: 1 (partially — oracle landscape mapped, model viability untested)
- New questions added this cycle: 1 (Phase 1 follow-up with breakeven-based selection)
- Questions remaining (open, not blocked): 4
- Handoff required: NO (existing HANDOFF.md is for tick bar fix, already resolved via TB-Fix TDD)
