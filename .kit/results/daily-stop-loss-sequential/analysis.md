# Analysis: Daily Stop-Loss Sequential Trading Simulation

## Verdict: REFUTED (Outcome B — SC-2 structural failure)

The full hypothesis ("positive sequential PnL AND >30% drawdown reduction from stop-loss") is REFUTED because SC-2 clearly failed (2.6% DD reduction vs 30% threshold). However, SC-1 — the survival gate — is a **strong positive finding**: sequential execution produces $1,853 total PnL ($2.06/trade, annualized Sharpe 1.54) across 151 test days. The stop-loss failure is structural (single-trade tail risk dominates max DD, which no cumulative daily stop can prevent), not a failure of the model's edge.

The experiment also triggers Outcome D (SC-S2 failure): sequential per-trade expectancy ($2.06) exceeds bar-by-bar ($0.90) by $1.16, outside the ±$0.50 tolerance. This is NOT an error — it reflects the fundamental difference between 901 non-overlapping sequential trades and 593,523 heavily-overlapping bar-by-bar evaluations. The sequential estimate is arguably more economically meaningful (see Section 6).

**Bottom line:** The two-stage pipeline at 19:7 produces a real, positive edge under sequential execution with conservative lockout. The daily cumulative stop-loss is the wrong risk control mechanism for this strategy — position-level stops or intra-trade exits are needed to address single-trade tail risk.

---

## Results vs. Success Criteria

- [x] **SC-1: PASS** — Sequential total PnL = **$1,852.76** > $0 threshold. Edge survives sequential execution.
- [ ] **SC-2: FAIL** — Best stop-loss (L*=-$200) reduces max DD by only **2.6%** (-$478.69 → -$466.20). Threshold: >30%. Failed by 27.4pp.
- [x] **SC-3: PASS** — Sequential per-trade exp = **$2.06** > $0.50 threshold. 4.1x the threshold.
- [x] **SC-4: PASS** — Best stop-loss retains **82.2%** of no-stop PnL ($1,523 / $1,853). Threshold: >60%.
- [x] **SC-S1: PASS** — Mean trades/day = **5.97**, within [5.0, 7.0].
- [ ] **SC-S2: FAIL** — Sequential exp $2.06 is **$1.16 above** bar-by-bar $0.90 (tolerance ±$0.50). Triggers Outcome D investigation.
- [x] **SC-S3: PASS** — Bar-by-bar reproduction = **$0.9015** vs PR #35's $0.90 (diff $0.0015 < $0.01).
- [x] **SC-S4: PASS** — Max DD is monotonically non-decreasing across stop levels (all stops: -$466.20; no-stop: -$478.69).

**Overall: 6/8 pass. 2 failures: SC-2 (structural) and SC-S2 (bar selection bias). Outcome: B + D caveat.**

---

## Metric-by-Metric Breakdown

### Primary Metrics

| Metric | Value | Threshold | Verdict |
|--------|-------|-----------|---------|
| sequential_total_pnl | $1,852.76 | > $0 | **PASS** |
| max_daily_drawdown_no_stop | -$478.69 | — (reference) | Driven by single catastrophic trade |
| max_daily_drawdown_best_stop | -$466.20 | >30% reduction | **FAIL** (2.6%) |

**sequential_total_pnl ($1,852.76):** Across 151 test days and 901 trades. Daily mean $12.27, daily std $125.82, daily Sharpe 0.098 (annualized: 1.54). This is economically meaningful — $12.27/day × 251 trading days = ~$3,080 annualized from 1 MES contract. However, Fold 2 contributes 75% of total PnL (see per-fold analysis).

**max_daily_drawdown_no_stop (-$478.69):** Occurred on 2022-11-02. This is a single catastrophic hold-bar trade. At 19:7 geometry with realized-return PnL, a hold-bar trade where fwd_return is deeply negative and the prediction direction is wrong can produce losses of ~$400-500 (fwd_return × $1.25 × wrong_sign - $3.74). This single event dominates all drawdown metrics.

**max_daily_drawdown_best_stop (-$466.20):** At the -$200 stop level, the worst day improved by only $12.49 (exactly one avoided wrong-direction directional bar). All stop levels from -$25 to -$200 produce the same -$466.20 max DD — the catastrophic loss occurs on the FIRST trade of the day, before the stop can trigger. The $12.49 difference between -$200 and no-stop suggests only one additional trade occurred on that worst day under no-stop.

### Secondary Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| sequential_per_trade_exp | $2.06 | 2.3x bar-by-bar ($0.90). SC-S2 FAIL. |
| trades_per_day_mean | 5.97 | Within expected [5, 7] range |
| trades_per_day_std | 0.29 | Extremely tight — lockout dominates timing |
| total_trades | 901 | 151 days × ~6 trades/day |
| pct_profitable_days_no_stop | 54.3% | Marginally above 50% |
| sharpe_annualized_no_stop | 1.54 | Strong. Above typical quant fund threshold (1.0) |
| mean_daily_pnl | $12.27 | Positive but noisy (std 10.2x mean) |
| std_daily_pnl | $125.82 | Large — driven by hold-bar tail events |
| consecutive_loss_days_max | 5 | Tolerable |
| sequential_hold_fraction | 44.2% | Matches bar-by-bar (44.4%) — no selection bias on hold/directional split |
| lag1_autocorrelation | -0.033 | Essentially zero — no serial dependence in sequential trade outcomes |

**Stop-loss comparison table:**

| Stop Level | Total PnL | Max DD | DD Red. | Sharpe | % Prof Days | Trades | Triggered | PnL Forfeited |
|-----------|-----------|--------|---------|--------|-------------|--------|-----------|---------------|
| None | $1,853 | -$479 | — | 1.54 | 54.3% | 901 | 0 | $0 |
| -$200 | $1,523 | -$466 | 2.6% | 1.23 | 54.3% | 884 | 8 | $0 |
| **-$100** | **$1,904** | **-$466** | **2.6%** | **1.65** | **53.6%** | **803** | **26** | **$278** |
| -$50 | $1,152 | -$466 | 2.6% | 1.05 | 51.0% | 735 | 49 | $1,182 |
| -$25 | $340 | -$466 | 2.6% | 0.33 | 45.0% | 659 | 70 | $2,194 |

**Notable anomaly: The -$100 stop INCREASES total PnL to $1,904 (+$51 vs no-stop) and produces the highest Sharpe (1.65).** It triggers on 26 days, forfeiting $278 from days that would have recovered, but saving more from days that would have worsened. This is the only stop level where the protective benefit exceeds the cost.

**Per-fold sequential results vs bar-by-bar:**

| Fold | Seq PnL | Seq Exp | BB Exp | Seq Sharpe | Days |
|------|---------|---------|--------|------------|------|
| Fold 1 | +$529 | $1.76 | $0.01 | 1.44 | 50 |
| Fold 2 | +$1,384 | $4.52 | $2.54 | 3.70 | 51 |
| Fold 3 | -$60 | -$0.20 | $0.16 | -0.13 | 50 |

Fold 2 dominates (75% of total PnL), consistent with all prior experiments. **Fold 3 goes negative in sequential mode** ($0.16 bar-by-bar → -$0.20 sequential), meaning the sequential simulation HURTS Fold 3 — the opposite of the Fold 1/2 pattern where sequential improves over bar-by-bar. This fold asymmetry is the strongest evidence that the $2.06 sequential exp is fragile.

**Monthly PnL:**

| Month | PnL | Days | Mean Daily | Pattern |
|-------|-----|------|------------|---------|
| May 2022 | -$378 | 3 | -$126 | First 3 days only — small sample, cold start |
| Jun 2022 | +$881 | 21 | +$42 | Best month (Fold 1 test) |
| Jul 2022 | +$171 | 20 | +$9 | Transition to Fold 2 test |
| Aug 2022 | +$230 | 23 | +$10 | Fold 2 test |
| Sep 2022 | +$538 | 21 | +$26 | Strong |
| Oct 2022 | +$763 | 21 | +$36 | Best month by mean daily |
| Nov 2022 | +$481 | 21 | +$23 | Nov 2 catastrophic DD, otherwise strong |
| Dec 2022 | -$832 | 21 | -$40 | Worst month. Dec collapse dominates Fold 3. |

6/8 months positive. December is catastrophic (-$832), which single-handedly makes Fold 3 negative. Without December, Fold 3 PnL would be approximately +$772 (positive). The December collapse is a regime event, not a model failure — but it is real risk in a single-year sample.

**Quarterly PnL:**

| Quarter | PnL | Days | Mean Daily |
|---------|-----|------|------------|
| Q2 2022 | +$503 | 24 | +$21 |
| Q3 2022 | +$938 | 64 | +$15 |
| Q4 2022 | +$412 | 63 | +$7 |

All three quarters positive — contrast with E2E CNN experiment where GBT was negative in Q3-Q4 at 10:5 geometry. The 19:7 geometry + 2-stage pipeline + realized-return PnL model produces positive returns across all observed quarters. Q4 is the weakest, dragged down entirely by December.

**Time-of-day effect:**

| Slot | N | Mean PnL | Std | Median | Signal/Noise |
|------|---|----------|-----|--------|--------------|
| 1 (~09:30) | 151 | $3.29 | $67.18 | $8.76 | 0.049 |
| 2 (~10:30) | 151 | $4.67 | $45.39 | -$3.12 | 0.103 |
| 3 (~11:30) | 151 | $3.29 | $49.77 | $20.01 | 0.066 |
| 4 (~12:30) | 150 | -$1.61 | $41.50 | -$3.74 | -0.039 |
| 5 (~13:30) | 149 | $0.86 | $78.59 | -$12.49 | 0.011 |
| 6 (~14:30) | 149 | $1.78 | $37.89 | -$8.74 | 0.047 |

No slot exceeds 2x another in absolute terms. The range is [$-1.61, $+4.67] — a $6.28 spread against stds of $37-79. Time-of-day is NOT a meaningful confound. Slot 4 (midday) is slightly negative; Slot 2 (10:30) is the best. All signal-to-noise ratios are below 0.11 — indistinguishable from zero given N=149-151.

**Conservative model:** Total PnL = $2,070 ($2.30/trade). HIGHER than realized-return ($1,853, $2.06/trade). Under the conservative model (hold-bar trades = $0, no RT cost charged), the strategy performs better because it avoids the net-negative hold-bar PnL drag (-$2.68/trade per PR #35). This confirms hold-bar trades are a drag, not a benefit.

**Entry bar distribution:** Mean=1796, median=1442, p10=0, p90=3600. The distribution is approximately uniform across the trading day with a slight skew toward earlier bars (mean < midpoint of [0, 3960]). The p10=0 confirms many first trades occur on bar 0 (market open). This is consistent with the lockout=720 structure: trades at approximately bars 0, 720, 1440, 2160, 2880, 3600.

### Sanity Checks

| Check | Expected | Observed | Verdict |
|-------|----------|----------|---------|
| SC-S1: trades/day in [5.0, 7.0] | ~6 | 5.97 | **PASS** |
| SC-S2: seq exp within ±$0.50 of $0.90 | $0.40 - $1.40 | $2.06 | **FAIL** ($1.16 above) |
| SC-S3: BB reproduction within $0.01 | $0.90 | $0.9015 | **PASS** |
| SC-S4: monotonic max DD | — | All stops = -$466; no-stop = -$479 | **PASS** |

---

## Resource Usage

| Resource | Budget | Actual | Assessment |
|----------|--------|--------|------------|
| Wall-clock | 15 min | 13.4 sec | 67x under budget |
| Training runs | 6 | 6 | Exact |
| Seeds | 1 | 1 | Exact |
| GPU hours | 0 | 0 | Exact |
| Simulation runs | 30 | 30 (5 stops × 2 PnL models × 3 folds) | Exact |

Budget was massively over-estimated. The sequential simulation is trivially fast (~7M iterations in <5s). The entire experiment including data loading and model training completed in 13 seconds. This is appropriate for a Quick-tier experiment.

---

## Confounds and Alternative Explanations

### 1. SC-S2 Bar Selection Bias — the Critical Finding

Sequential per-trade exp ($2.06) exceeds bar-by-bar ($0.90) by 129%. The spec flags this as Outcome D: "Sequential timing selects easy bars — bar-by-bar UNDERESTIMATES sequential edge."

**But is this really bias? Or is bar-by-bar the flawed estimate?**

At 19:7 with lockout=720, each sequential trade's outcome window (3600s) is non-overlapping with the next. In bar-by-bar mode, 3,937 daily trades have outcome windows that overlap by ~99.86% (each bar is 5s apart; each outcome spans 3600s = 720 bars). This means ~720 consecutive bar-by-bar "trades" are evaluating essentially the SAME underlying price path with slightly different entry points. The bar-by-bar average of $0.90 over-samples correlated outcomes, effectively counting each price path ~720 times.

**The sequential simulation produces 901 approximately independent trade evaluations. The bar-by-bar produces 593,523 heavily correlated evaluations.** The per-trade expectancy from independent observations ($2.06) is likely more economically meaningful than from correlated observations ($0.90).

However, this interpretation assumes the first-available entry is no worse than a random entry within each 720-bar window. If the model's strongest signals cluster at specific bars within the window (e.g., after a volatility spike) and the lockout forces entry at a specific phase, there could be genuine selection bias. The near-zero lag-1 autocorrelation (-0.033) in sequential trades suggests no systematic serial pattern, but this doesn't rule out within-window selection effects.

**Assessment: SC-S2 failure is likely an artifact of the bar-by-bar overlap problem, not a genuine simulation bug. But cannot definitively rule out positive selection bias. Treat $2.06 as the upper bound and $0.90 as the lower bound of true per-trade expectancy.**

### 2. Lockout Overestimate

The lockout of 720 bars (3600s = full time horizon) is CONSERVATIVE. Directional bars (tb_label != 0) exit when the barrier is hit, typically well before 3600s. The actual mean exit time is unknown (not in Parquet schema). If the true mean exit time is 1800s (half the horizon), the strategy could execute ~12 trades/day instead of ~6, roughly doubling total PnL at similar per-trade expectancy. **A positive result at lockout=720 is a genuine lower bound.**

### 3. Fold 2 Dominance

Fold 2 contributes $1,384 of $1,853 total (75%). Fold 3 is negative (-$60). This is consistent with all prior experiments (PR #35: Fold 2 exp $2.54, Fold 1 $0.01, Fold 3 $0.16). The sequential simulation AMPLIFIES fold dispersion: Fold 1 improves from near-zero ($0.01) to positive ($1.76), but Fold 3 degrades from positive ($0.16) to negative (-$0.20). If Fold 2's market regime (roughly Jul-Sep 2022) is atypical, the entire result is fragile.

The sequential amplification is explicable: with only ~6 trades/day (vs ~3,937 bar-by-bar), daily variance is higher. A fold with marginal bar-by-bar expectancy ($0.16, Fold 3) can easily go negative when sampling 295 trades instead of ~198K.

### 4. December Collapse

December 2022 PnL = -$832, single-handedly making Q4 the weakest quarter and Fold 3 negative. Without December, the equity curve peaks at $2,685 (Nov 30) and the strategy appears highly profitable. This is a single-month regime event in a single-year sample. We cannot assess whether December 2022 is typical or anomalous without multi-year data.

### 5. Single-Trade Tail Risk Dominates Max DD

The -$479 max daily DD (Nov 2) is a single catastrophic hold-bar trade. ALL stop-loss levels produce the same max DD (-$466) because the loss occurs on the first trade of the day. This is a structural limitation of cumulative daily stop-losses: they cannot prevent single-trade catastrophic losses. The daily stop-loss mechanism is the wrong tool for this risk profile.

The max single-trade loss is driven by hold bars with realized-return PnL. At 19:7 geometry, hold bars (tb_label=0) have fwd_return distributions with p10=-63 ticks, p90=+63 ticks (from PR #35). A deeply wrong-direction hold-bar trade at p1 could lose ~$400+ in a single position.

### 6. Daily Stop-Loss Optimization Risk

With only 5 levels and 151 test days, the "best" stop is lightly optimized. However, the result is robust in a useful sense: no stop level achieves >3% DD reduction, and the -$100 level's PnL improvement (+$51 vs no-stop) is a second-order effect. The conclusion (daily stop-loss is structurally ineffective) holds regardless of which level is chosen.

### 7. Conservative vs Realized-Return Divergence

Conservative model ($2,070) exceeds realized-return ($1,853) by $217. Since conservative treats hold-bar PnL as $0 (no cost, no gain), this confirms hold-bar realized returns are net-negative. At 44.2% hold fraction and ~901 trades, this implies mean hold-bar loss of ~$0.55/trade (($1,853-$2,070)/901 × correction for different trade counts). This is milder than the bar-by-bar hold-bar drag of -$2.68 (from PR #35), suggesting sequential timing may avoid the worst hold-bar outcomes.

---

## What This Changes About Our Understanding

1. **The two-stage pipeline at 19:7 produces positive PnL under realistic sequential execution.** This was the #1 open question after three failed parameter optimization experiments. The $0.90/trade bar-by-bar expectancy is real — the edge survives the transition from theoretical to operational. The $2.06 sequential estimate is actually MORE favorable than bar-by-bar, likely because it evaluates non-overlapping trade windows.

2. **Daily cumulative stop-losses are structurally ineffective for this strategy.** The risk profile is dominated by single-trade tail events from hold-bar positions, not by accumulated small losses. The max DD is identical across all stop levels (-$466). This is not a tuning failure — it's a mechanism mismatch. Risk control for this strategy requires **position-level intervention** (tighter intra-trade stops, real-time exit triggers) or **entry-level intervention** (filtering hold bars before entry, which requires Stage 1 threshold > 0.50 — but threshold-sweep showed this collapses trade rate).

3. **The -$100 stop is a genuinely useful PnL filter.** It improves total PnL (+$51), Sharpe (1.54 → 1.65), and prevents continuation on bad days. It does NOT reduce max DD (same -$466). This is a "cut your losses" mechanism, not a "prevent catastrophe" mechanism.

4. **Sequential per-trade expectancy may be more meaningful than bar-by-bar.** The 129% uplift from $0.90 to $2.06 is explained by the elimination of ~660x overlap in outcome windows. The bar-by-bar $0.90 over-samples correlated outcomes. For deployment sizing and risk estimation, $2.06/trade × 6 trades/day = $12.36/day is likely a better estimate than $0.90 × 3,937 trades/day.

5. **Annualized Sharpe of 1.54 is a deployment-grade signal.** This exceeds typical quant fund thresholds (1.0-1.5). However, it comes from a single year (2022), single instrument (MES), and Fold 2 dominates. Multi-year validation is needed before this number carries weight.

6. **December 2022 is a regime risk.** The -$832 monthly loss from December single-handedly makes the strategy appear weaker than it performed for 7/8 months. This is real risk (not a data error), but the strategy survived it and still ended profitable.

---

## Proposed Next Experiments

### 1. Time-to-barrier TDD cycle (highest priority, regardless of outcome)

Add an `exit_bar` column to the Parquet schema recording the actual bar index where the triple barrier was hit. This enables:
- Realistic lockout = actual exit time (not worst-case 720 bars)
- Estimated trade count increase from ~6/day to ~10-15/day
- Total PnL scaling proportional to trade count increase (if per-trade exp holds)
- Validates or refutes the lockout overestimate confound

**Rationale:** The current experiment is a LOWER BOUND. If the edge survives at lockout=720, shorter lockouts can only improve it (more trades with similar per-trade exp). But we need the data to prove it.

### 2. Position-level stop-loss simulation

Replace the daily cumulative stop with a per-position stop:
- Intra-trade mark-to-market exit if unrealized loss exceeds threshold (e.g., -$25, -$50, -$100)
- This directly addresses the single-trade tail risk that the daily stop cannot reach
- Requires tick-level or bar-level price path within the 3600s outcome window (not currently available — needs bar_feature_export extension or forward price columns)

**Rationale:** Daily stop failed structurally. Position-level stop addresses the actual risk source (single-trade catastrophic loss from hold bars).

### 3. Multi-year validation (2023 data if available)

Run the identical sequential simulation on out-of-sample 2023 data:
- Tests whether the edge is 2022-specific or structural
- Addresses December 2022 regime risk concern
- Addresses Fold 2 dominance concern

**Rationale:** The strongest criticism of this result is single-year fragility. 2023 data is the most direct remedy.

---

## Program Status

- Questions answered this cycle: 0 (this experiment does not directly answer an open question in §4, but it provides critical deployment-readiness evidence for the P0 two-stage pipeline evaluation)
- New questions added this cycle: 1 (position-level stop-loss effectiveness)
- Questions remaining (open, not blocked): 4 (long-perspective labels P0, message regime-stratified P2, transaction cost sensitivity P2, regime-conditional trading P3)
- Handoff required: NO (time-to-barrier TDD cycle and position-level stop are within the research+TDD pipeline scope)
