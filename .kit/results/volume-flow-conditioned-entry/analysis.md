# Analysis: Volume-Flow Conditioned Entry

## Verdict: REFUTED (Outcome B)

**Executive Summary:** All 5 volume/activity features show massive 20pp diagnostic signal (timeout fraction varies from ~62% in Q1 to ~41% in Q4), yet gating on these features in sequential simulation reduces timeouts by at most 1.76pp — a 10x evaporation. The reason: sequential execution already self-selects for high-activity bars via the 66.1% hold-skip mechanism. Gating provides negligible incremental selection. The stacked configuration (cutoff=270 + message_rate_p25) is the best risk-adjusted option (Calmar 2.59, $33K min account) but delivers only marginal improvement over cutoff=270 alone (+$0.08/trade, -$1K min account). Neither SC-2 ($3.50/trade), SC-3 ($30K), nor SC-5 (5pp timeout reduction) are met. The 41.3% timeout fraction is a structural constant of the volume horizon mechanism — no entry-time gating can change it.

---

## Results vs. Success Criteria

- [x] SC-1: **PASS** — Stage 1 diagnostic produced D1-D5 for all 5 features × 4 quartiles, cross-table populated
- [ ] SC-2: **FAIL** — Optimal config (message_rate_p25) achieves $2.51/trade, well below $3.50 threshold. Even stacked config achieves only $3.10/trade.
- [ ] SC-3: **FAIL** — Optimal config min_acct = $47,500, far above $30K threshold. Stacked config = $33,000 (still above).
- [x] SC-4: **PASS** — 810 simulations completed (5 features × 3 levels × 45 splits + baseline + cutoff_270 + stacked = 18 configs × 45 splits)
- [ ] SC-5: **FAIL** — Maximum timeout reduction = 1.76pp (volatility_50_p75: 39.57% vs 41.33% baseline). "Optimal" gate: 0.12pp. Target was 5pp.
- [x] SC-6: **PASS** — Four-way comparison table fully populated
- [x] SC-7: **PASS** — All 7 output files written to results directory
- [x] SC-8: **PASS** — Bar-level split 0 = $1.0652 vs PR #38 reference $1.065186 (delta < $0.01)

**Sanity Checks:**
- [x] SC-S1: **PASS** — Bar-level split 0: $1.0652 vs $1.065186 (within $0.01)
- [x] SC-S2: **PASS** — Baseline exp: $2.501 vs PR #39 $2.50 (within $0.10)
- [x] SC-S3: **PASS** — Baseline trades/day: 162.21 vs PR #39 162.2 (within 5)
- [x] SC-S4: **PASS** — Baseline timeout: 41.329% vs 41.33% (within 0.5pp)

**Score: 5/8 pass (SC-1/4/6/7/8). SC-2, SC-3, SC-5 fail. All 4 sanity checks pass.**

---

## Metric-by-Metric Breakdown

### Primary Metrics

#### 1. Diagnostic Table (D1-D4)

| Feature | Q1 (low) | Q2 | Q3 | Q4 (high) | Range (pp) | Tier |
|---------|----------|-----|-----|-----------|------------|------|
| trade_count | 61.3% | 44.2% | 42.9% | 41.6% | **19.7** | strong |
| message_rate | 62.2% | 44.2% | 42.7% | 41.0% | **21.3** | strong |
| volatility_50 | 62.1% | 45.0% | 42.6% | 40.4% | **21.7** | strong |
| trade_count_20 | 61.9% | 44.2% | 43.1% | 40.9% | **20.9** | strong |
| message_rate_20 | 62.4% | 44.2% | 42.8% | 40.7% | **21.7** | strong |

All 5 features pass the 5pp strong signal threshold. The range is 19.7–21.7pp — massive by any standard. This is the strongest diagnostic signal in the project's history.

**Critical observation:** The Q1→Q2 step is ~17-18pp for every feature. The Q2→Q4 step is only ~3-4pp. The diagnostic signal is overwhelmingly concentrated in Q1 (the lowest-activity quartile), where timeout fraction jumps to ~62%. Q2-Q4 are compressed into a ~4pp band around 41-45%.

#### 2. Diagnostic Cross-Table (D5)

| | tc_Q1 | tc_Q2 | tc_Q3 | tc_Q4 | Range |
|--|-------|-------|-------|-------|-------|
| **v50_Q1** (low vol) | **71.5%** | 47.0% | 46.9% | 46.8% | **24.7pp** |
| **v50_Q2** | 46.1% | 44.6% | 44.6% | 44.8% | **1.5pp** |
| **v50_Q3** | 44.3% | 42.4% | 42.1% | 42.9% | **2.2pp** |
| **v50_Q4** (high vol) | 42.1% | 41.6% | 40.7% | 39.9% | **2.2pp** |

**The cross-table is the most revealing result in this experiment.** After controlling for volatility:
- **v50_Q1 (low volatility):** trade_count matters enormously (24.7pp range). The v50_Q1 × tc_Q1 cell (71.5%) is the outlier driving all marginal diagnostic signals. Low volatility + low trade count = 71.5% timeout rate.
- **v50_Q2-Q4:** trade_count has almost no independent effect (1.5-2.2pp range). Once volatility is moderate or high, trade count is irrelevant.

**Interpretation:** The diagnostic signal is NOT "high volume → fewer timeouts." It is "extremely quiet periods → dramatically more timeouts." The effect is concentrated in one corner of the feature space (low-volatility, low-activity). Outside that corner, timeout fraction is structurally pinned to ~41-45% regardless of volume or activity.

#### 3. Gate Sweep Table

18 configurations swepted across 45 splits (810 total simulations). Key results by timeout fraction delta from baseline (41.33%):

| Config | Timeout Δ (pp) | Exp/Trade | Trades/Day | Daily PnL | Calmar |
|--------|---------------|-----------|------------|-----------|--------|
| baseline | 0.00 | $2.50 | 162.2 | $413 | 2.16 |
| cutoff_270 | -0.03 | $3.02 | 116.8 | $337 | 2.49 |
| volatility_50_p75 | **-1.76** | $2.97 | 96.9 | $299 | 1.67 |
| message_rate_20_p75 | -1.18 | $1.92 | 93.9 | $222 | 1.39 |
| trade_count_20_p75 | -1.10 | $2.05 | 102.1 | $247 | 1.57 |
| volatility_50_p50 | -0.73 | $2.67 | 137.2 | $381 | 2.03 |
| message_rate_p75 | -0.57 | $2.32 | 133.2 | $320 | 1.79 |
| stacked (msg_rate_p25 + c270) | -0.10 | $3.10 | 114.7 | $338 | 2.59 |
| message_rate_p25 ("optimal") | -0.12 | $2.51 | 158.9 | $405 | 2.14 |

**The maximum timeout reduction is 1.76pp** (volatility_50_p75), achieved by filtering out 79.5% of entry opportunities. This reduces trades/day from 162→97 (-40%), daily PnL from $413→$299 (-28%), and Calmar from 2.16→1.67 (-23%). The timeout reduction is purchased at severe cost and is nowhere near the 5pp target.

**The p75 gates universally degrade performance.** Every p75 gate produces worse Calmar, worse Sharpe, and worse daily PnL than baseline. The most aggressive gates (trade_count_20_p75, message_rate_20_p75) produce the worst results: $247/day and $222/day respectively, with Calmar below 1.6.

**The p25 gates are essentially no-ops.** Gate-skip rates at p25 range from 5.7% (trade_count) to 34.9% (volatility_50), but the gates remove the "wrong" bars — the Q1 bars that the sequential process was already avoiding via hold-skip.

#### 4. Comparison Table (Four-Way)

| Metric | Unfiltered | Cutoff=270 | Volume-Gated | Stacked |
|--------|-----------|------------|-------------|---------|
| Exp/trade | **$2.50** | $3.02 | $2.51 | $3.10 |
| Trades/day | **162.2** | 116.8 | 158.9 | 114.7 |
| Daily PnL | **$413** | $337 | $405 | $338 |
| DD worst | $47,894 | $33,984 | $47,434 | **$32,830** |
| Min acct | $48,000 | $34,000 | $47,500 | **$33,000** |
| Timeout | 41.33% | 41.30% | 41.20% | 41.23% |
| Calmar | 2.16 | 2.49 | 2.14 | **2.59** |
| Sharpe | 2.10 | 2.20 | 2.08 | **2.22** |
| Annual PnL | **$103,605** | $84,530 | $101,579 | $84,941 |

**The volume gate alone (message_rate_p25) is essentially identical to baseline.** Delta: +$0.01/trade, -3.3 trades/day, -$8/day, -$500 min account. This is noise-level improvement.

**The stacked configuration is marginally better than cutoff=270 alone.** Delta from cutoff=270: +$0.08/trade, -2.1 trades/day, +$2/day, -$1K min account, +0.10 Calmar. The volume gate's incremental contribution over the time-of-day cutoff is negligible.

#### 5. Optimal Gate Expectancy

$2.51/trade (message_rate_p25). Selected by Rule 3 (max Calmar) because no gate met SC-2 AND SC-3, and no gate met the relaxed criterion (min_acct <= $35K with max daily PnL). The "optimal" gate is effectively a no-op.

#### 6. Optimal Gate Min Account

$47,500 (message_rate_p25). Virtually unchanged from baseline's $48,000.

### Secondary Metrics

#### 7-10. Optimal Gate Operating Profile

| Metric | Value | Baseline | Delta |
|--------|-------|----------|-------|
| Daily PnL | $405 | $413 | -$8 (-1.9%) |
| Calmar | 2.14 | 2.16 | -0.02 |
| Sharpe | 2.08 | 2.10 | -0.02 |
| Trades/day | 158.9 | 162.2 | -3.3 (-2.0%) |

All metrics WORSE than baseline. The optimal gate degrades performance.

#### 11. Timeout Fraction by Config

Range across all 18 configs: 39.57% to 41.33% = **1.76pp total range**. Compared to the 20pp diagnostic range, this is a **91% evaporation** of the signal when translated from bar-level statistics to sequential simulation.

#### 12. First-100-Bars Entry Percentage

4.1% — well below the 15% threshold. Rolling features (trade_count_20, message_rate_20) are NOT corrupted by staleness at session open. No exclusion needed.

#### 13. Qualifying Features

All 5 features qualify (all >5pp range): trade_count, message_rate, volatility_50, trade_count_20, message_rate_20.

#### 14. Daily PnL Percentiles (Optimal Gate)

| Percentile | Value |
|------------|-------|
| p5 | -$3,870 |
| p25 | -$462 |
| p50 | +$276 |
| p75 | +$1,162 |
| p95 | +$4,714 |

Median positive ($276). ~37% of days are negative (interpolating between p25 and p50). Consistent with baseline profile.

### Sanity Checks

All 4 pass. Baseline perfectly reproduces PR #39 sequential results. Training is identical (bar-level split 0 within $0.0000004 of PR #38). No implementation divergence.

---

## Resource Usage

| Resource | Actual | Budget | Status |
|----------|--------|--------|--------|
| Wall-clock | 3.05 min | 15 min | Well within budget |
| Training runs | 90 | 90 | Exact match (45 splits × 2 stages) |
| Simulations | 810 | 720 | Slightly over (18 configs × 45 splits) |
| GPU-hours | 0 | 0 | CPU-only as designed |

Budget appropriate. Experiment ran in ~1/5 of allocated time.

---

## Confounds and Alternative Explanations

### 1. Sequential Selection Bias (PRIMARY CONFOUND — CONFIRMED)

The 10x evaporation of diagnostic signal (20pp bar-level → 1.76pp simulation) has a clear explanation: **sequential execution already filters out low-activity bars.**

The baseline hold-skip rate is 66.1%. This means 2 out of 3 entry opportunities are skipped because a position is already open. Critically, the hold-skip mechanism is NOT random — entries cluster during volatile periods (shorter barrier races → faster exit → next entry available sooner) and are suppressed during quiet periods (longer races → position held longer → fewer new entries). This was documented in trade-level-risk-metrics: "Sequential entry creates a non-random selection pattern: entries cluster during volatile moments."

The Q1 (low-activity) bars that drive the diagnostic's 20pp range are precisely the bars that sequential execution already avoids. The gate-skip rate at p25 is only 5.7-34.9% — most low-activity bars were already hold-skipped. Gating removes bars that weren't going to be traded anyway.

Evidence: hold-skip rate drops systematically as gate stringency increases (baseline 66.1% → trade_count_p75 43.1% → trade_count_20_p75 18.7%). The gate replaces hold-skip as the dominant filtering mechanism but achieves the same outcome.

### 2. Cross-Table Corner Effect

The D5 cross-table reveals that the entire diagnostic signal is concentrated in one cell: v50_Q1 × tc_Q1 = 71.5% timeout. This is the low-volatility, low-trade-count corner — extreme quiet periods where barriers cannot be hit (low volatility) AND the volume horizon takes forever to fill (low trades). Outside this corner, the feature space is flat (41-45% across 12/16 cells).

This means volume gating is really "avoid extreme quiet periods" gating. But sequential execution already avoids extreme quiet periods through hold-skip mechanics.

### 3. Volume-Volatility Collinearity

All 5 features are highly correlated. The per-feature diagnostic ranges (19.7-21.7pp) are nearly identical, and the cross-table shows almost no independent trade_count effect after controlling for volatility_50. The "5 features" are essentially one signal measured five ways. This explains why sweeping different features produces similar results.

### 4. Gate-as-Hold-Skip-Restructuring

PR #40 showed that cutoff=270 improves via hold-skip restructuring, not timeout avoidance. The same mechanism operates here: the stacked config's marginal improvement ($3.10 vs $3.02) comes from further restructuring which signals get executed, not from reducing timeouts. The timeout fraction barely changes (41.23% stacked vs 41.30% cutoff=270).

---

## What This Changes About Our Understanding

### Before this experiment:
- PR #40 showed timeouts are invariant to time-of-day. Hypothesis: timeouts might respond to volume/activity features that directly relate to the volume horizon consumption mechanism.

### After this experiment:
1. **Timeout fraction (~41.3%) is a structural constant of the 19:7 barrier geometry at 50,000 volume horizon.** Neither time-of-day (PR #40, 0.21pp range across 7 cutoffs) nor volume/activity gating (this experiment, 1.76pp max across 15 gates) can materially change it. Two independent experiments confirm invariance.

2. **The diagnostic paradox is resolved by sequential selection bias.** Bar-level statistics show 20pp variation in timeout fraction, but the sequential execution process already selects for the favorable portion of the distribution. The "entry-time gating" idea was solving a problem that the execution mechanics had already solved.

3. **The stacked configuration (cutoff=270 + message_rate_p25) is the new best risk-adjusted option** at Calmar 2.59, Sharpe 2.22, min_acct $33K. However, the improvement over cutoff=270 alone is marginal (+$0.08/trade, +0.10 Calmar), and the volume gate's contribution is within noise.

4. **Entry-time filtering is exhausted as an intervention class.** Two experiments (time-of-day, volume-flow) have now tried to improve sequential execution by filtering WHEN to enter. Both find: (a) timeout fraction is invariant, (b) expectancy improves only through hold-skip restructuring, and (c) improvements don't reach $3.50/trade or $30K account targets. The next intervention must change the barrier parameters themselves, not the entry timing.

### Updated mental model:
The timeout mechanism operates at the bar-sequence level (post-entry volume horizon consumption), not at the entry-selection level. No entry-time feature can predict whether the post-entry volume horizon will expire before a barrier hit, because the volume horizon (50,000 contracts) represents ~4 hours of volume at median rates — it almost always fills, and whether the barrier hits first depends on the stochastic path, not the entry-time conditions.

---

## Proposed Next Experiments

1. **Barrier geometry re-parameterization (HIGHEST PRIORITY).** Reduce volume horizon from 50,000 to 10,000-25,000 contracts. At 50,000, the race almost always ends at the volume horizon (41.3% timeout). Reducing it trades shorter race windows (more timeouts from time expiry) for faster resolution — the net effect on expectancy is unknown. Alternatively: reduce time horizon from 3,600s to 600-1,200s, which directly caps race duration but changes which barrier events are reachable. This is the only intervention that can mechanically change the timeout rate.

2. **Accept cutoff=270 as the operational config and proceed to paper trading.** cutoff=270 achieves $3.02/trade, $34K min account, Calmar 2.49, Sharpe 2.20. Stacking message_rate_p25 adds +$0.08/trade (noise-level). The operational question is whether these CPCV numbers hold in live execution, not whether entry filters can squeeze more from backtested data.

3. **Regime-conditional trading (Q1-Q2 only).** GBT is marginally profitable in Q1 (+$0.003/trade) and Q2 (+$0.029/trade) under base costs, negative in Q3-Q4. A seasonal filter is a different class of intervention (regime gating, not entry-time gating) that hasn't been tested with sequential execution.

---

## Program Status

- Questions answered this cycle: 1 (volume-flow conditioned entry → REFUTED)
- New questions added this cycle: 0
- Questions remaining (open, not blocked): 5 (long-perspective labels P0, message regime-stratified P2, cost sensitivity P2, regime-conditional P3, multi-contract P2)
- Handoff required: NO

---

## Appendix: Diagnostic-to-Simulation Signal Evaporation

The central finding of this experiment is the **91% evaporation** of diagnostic signal:

| Level | Measure | Range |
|-------|---------|-------|
| Bar-level diagnostic | Timeout fraction by feature quartile | 19.7–21.7pp |
| Sequential simulation (best single gate) | Timeout fraction delta from baseline | 1.76pp |
| Sequential simulation (optimal gate) | Timeout fraction delta from baseline | 0.12pp |

**Evaporation ratio:** 21.7pp (max diagnostic) → 1.76pp (best simulation) = **92% evaporation**.

This is the strongest evidence yet that the sequential execution process creates a powerful self-selection effect. The 66.1% hold-skip rate is not a deficiency — it IS the gating mechanism that entry-time filters were trying to replicate.
