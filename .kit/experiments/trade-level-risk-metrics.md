# Experiment: Trade-Level Risk Metrics for Account Sizing

**Date:** 2026-02-27
**Priority:** P0 — required for account sizing and deployment feasibility
**Parent:** CPCV Corrected Costs (CONFIRMED Outcome A, PR #38)
**Depends on:**
1. CPCV corrected-costs pipeline and results — DONE (PR #38)
2. Label geometry 1h Parquet data (3600s horizon, 152-col) — DONE (PR #33)

**All prerequisites DONE. Re-uses identical CPCV training (45 splits, same model fits). Changes only the test-set evaluation to simulate sequential 1-contract execution.**

---

## Context

CPCV validation (PR #38) confirmed the 2-stage pipeline at $1.81/trade (PBO 6.7%, p<1e-13). But the CPCV computed bar-level predictions on 5-second time bars — each bar gets an independent prediction and an independent barrier race. With ~4,630 bars/day and 87% trade rate, that's ~4,028 bar-level "trades" per day.

**The problem:** These are not sequential trades. A 19-tick barrier race on MES takes ~75 bars (~6.25 min) to resolve on average. With a new prediction every 5 seconds, you'd have ~75 concurrent open positions at any moment. On 1 MES contract, you can only have 1 position open at a time. **The CPCV $1.81/trade is a per-prediction metric, not a per-execution metric.**

Under sequential execution (enter -> wait for barrier -> exit -> enter again), max trades/day ~ 390 min / 6.25 min ~ **62 trades/day**. The per-trade expectancy under sequential execution may differ from the bar-level $1.81 because:
- You select from a subset of signals (the one available at entry time)
- Holding period determines which signals you skip
- Hold bars (43% of predictions) would tie up the contract with no directional edge

**This experiment simulates sequential 1-contract execution**, not bar-level prediction aggregates, to get realistic risk metrics for account sizing.

---

## Hypothesis

Sequential 1-contract execution on the validated 2-stage pipeline (19:7, w=1.0, T=0.50) produces per-trade expectancy ≥ $0.50 under corrected-base costs ($2.49 RT), yielding a minimum viable account size ≤ $5,000 for 1 MES.

**Direction:** Positive sequential per-trade expectancy.
**Magnitude:** ≥ $0.50/trade (≥28% of bar-level $1.81). Structural reasoning: hold-skip removes 43% of bar-level trades that have ~$0.00 directional edge, concentrating on the 57% that are directional predictions with $1.81 average. Sequential signal selection (first-available-in-time, not random sample) may reduce or increase this. The $0.50 threshold is the minimum economically meaningful edge at ~50 sequential trades/day ($25/day = ~$6,300/year on 1 MES).

**Falsification:** If `seq_expectancy_per_trade` < $0.50, the hypothesis is REFUTED. The pipeline may still be positive-expectancy (Outcome B: $0.00-$0.50 range) but below the threshold for viable deployment at small account sizes.

---

## Independent Variables

### Execution mode (primary IV — 2 levels, same trained models)

| Mode | Description |
|------|-------------|
| **Bar-level** | Every bar gets a prediction and a trade (CPCV standard). ~4,028 trades/day. |
| **Sequential** | 1 contract: enter -> wait for barrier resolution -> exit -> enter again. Hold predictions SKIPPED. ~40-80 trades/day. |

### Cost model (FIXED — corrected-base only for sequential analysis)

| Scenario | Commission/side | Spread | Slippage | Total RT |
|----------|----------------|--------|----------|----------|
| **Base** | $0.62 | 1 tick | 0 | **$2.49** |

(Bar-level comparison uses all 3 cost levels from CPCV results already computed.)

### Pipeline configuration (FIXED — identical to CPCV PR #38)

| Parameter | Value |
|-----------|-------|
| Stage 1 weight | 1.0 |
| Stage 1 threshold | 0.50 |
| Geometry | 19:7 (target=19 ticks, stop=7 ticks) |
| Pipeline | Two-stage (reachability + direction) |
| XGB params | Same tuned params as PR #38 |
| Feature set | 20 non-spatial features |

---

## Controls

| Control | Value | Rationale |
|---------|-------|-----------|
| Data source | `.kit/results/label-geometry-1h/geom_19_7/` (152-col Parquet) | Same data as PR #38 |
| CPCV protocol | N=10, k=2, 45 splits, purge=500, embargo=4600 | Identical to PR #38 |
| Training | Same two-stage XGBoost with early stopping | Identical model fits |
| Seed | 42 (per-split: 42 + split_idx) | Identical to PR #38 |
| PnL model | Realized return | Same as PR #38 |
| RT cost (sequential) | $2.49 (corrected-base) | Standard deployment cost |
| Tick value | $1.25 (MES) | |
| Target | 19 ticks = $23.75 | |
| Stop | 7 ticks = $8.75 | |

---

## Baselines

| Baseline | Source | Key Metrics |
|----------|--------|-------------|
| **Bar-level CPCV (PR #38)** | `.kit/results/cpcv-corrected-costs/metrics.json` | Mean exp $1.81/trade (95% CI [$1.46, $2.16]), PBO 0.067, t=10.29, holdout $1.46/trade, break-even RT $4.30, 45 splits |
| **Bar-level per-split** | `.kit/results/cpcv-corrected-costs/cpcv_per_split.csv` | Per-split bar-level expectancy for verification (split 0 must match within $0.05) |

**Reproduction protocol:** The sequential experiment re-runs identical CPCV training (same data, splits, seeds, hyperparameters) and adds the sequential simulation as a post-processing step on each split's test-set predictions. Bar-level metrics from this experiment must match PR #38 within $0.05/trade (MVE check). Any divergence indicates training code was modified and triggers abort.

---

## Sequential Execution Protocol

For each CPCV split's test set (bars ordered by timestamp):

### Step-by-step simulation

1. Start at bar 0 with no position open.
2. At each bar, if no position is open:
   - Get model prediction for this bar (long=+1 / short=-1 / hold=0)
   - If **long or short** -> **ENTER position**. Record: entry bar index, entry timestamp, direction, `minutes_since_open` from feature columns.
   - If **hold (prediction=0)** -> **SKIP** (don't enter, no directional edge on hold bars). Log this bar as a hold-skip.
3. While position is open:
   - The position resolves based on the **entry bar's** triple-barrier outcome:
     - Use `tb_exit_type` column: "upper" or "lower" -> barrier hit
     - Use `tb_bars_held` column: number of bars until barrier resolution
   - PnL computation for the executed trade:
     - If `tb_exit_type` in ("upper", "lower") and `tb_label != 0`:
       - Correct direction (prediction sign matches tb_label sign): PnL = +19 * $1.25 - $2.49 = **+$21.26**
       - Wrong direction (prediction sign != tb_label sign): PnL = -7 * $1.25 - $2.49 = **-$11.24**
     - If `tb_exit_type` == "timeout" (or `tb_label == 0` on the entry bar):
       - PnL = fwd_return_720_ticks * $1.25 * sign(prediction) - $2.49
       - Where fwd_return_720_ticks is the cumulative 720-bar forward return from the entry bar
   - Position is occupied for `tb_bars_held` bars. The next available entry bar is entry_bar_index + tb_bars_held.
4. After position resolves, return to step 2 at the next bar after the exit bar.
5. Log every executed trade: split_idx, entry_timestamp, exit_timestamp, direction, tb_exit_type, pnl, bars_held, minutes_since_open.

**Hold-skip note:** The original CPCV treated hold predictions as "in a position earning realized forward return." Sequential mode skips them entirely — this is a **directional-only** strategy. Report `hold_skip_rate` (fraction of available entry bars where the model predicted hold and we skipped) so the gap between bar-level and sequential strategies is transparent.

### Day boundary handling

- Each test-set group consists of complete trading days.
- At end of each trading day, if a position is open: resolve at the closing bar PnL (use the entry bar's `tb_bars_held` and `tb_exit_type` — the barrier outcome already accounts for the time horizon). If `tb_bars_held` extends beyond the last bar of the day, the position resolves at day end using the forward return up to the last available bar.
- Next day starts fresh (no position carried overnight).

### Important: bars_held must advance correctly

The simulation must advance by `tb_bars_held` bars after entering a position, NOT by a fixed amount. Different barrier races resolve at different speeds. The `tb_bars_held` column in the Parquet data contains the actual resolution time for each bar's barrier race.

---

## Concurrent Position Analysis (Scaling Ceiling)

Independent of the sequential simulation, compute the **concurrent open positions** at each bar in the test set. This answers: "How many contracts would you need to capture ALL bar-level signals simultaneously?"

For each bar in the test set where the model predicts directional (long or short):
- That prediction has a barrier race lasting `tb_bars_held` bars
- The position is "open" from that bar through bar + tb_bars_held - 1

At each bar, count how many bar-level directional predictions have an active barrier race (entered but not yet resolved). Report:
- `concurrent_positions_mean`: average open positions at any bar across the entire test set
- `concurrent_positions_max`: peak concurrent open positions
- `concurrent_positions_p95`: 95th percentile

This is the **theoretical scaling ceiling**: if you ran N MES contracts (or N/5 ES contracts) to capture the full bar-level signal set, N ~ `concurrent_positions_mean`.

**Sanity check:** `concurrent_mean * seq_expectancy_per_trade ~ bar_level_daily_pnl / trading_days_in_test`. If this relationship holds (within 2x), the bar-level and sequential views are consistent.

---

## Metrics (ALL must be reported)

### Primary — Sequential Execution (corrected-base $2.49 RT)

| # | Metric | Description |
|---|--------|-------------|
| 1 | `seq_trades_per_day_mean` | Average sequential trades per RTH session across 45 paths |
| 2 | `seq_trades_per_day_std` | Std dev of daily sequential trade count |
| 3 | `seq_expectancy_per_trade` | Mean PnL per sequential trade (net of $2.49 RT) |
| 4 | `seq_daily_pnl_mean` | Average daily P&L on 1 MES under sequential execution |
| 5 | `seq_daily_pnl_std` | Daily P&L volatility |
| 6 | `seq_max_drawdown_worst` | Worst peak-to-trough drawdown across 45 paths ($, 1 MES) |
| 7 | `seq_max_drawdown_median` | Median max drawdown across 45 paths |
| 8 | `seq_max_consecutive_losses` | Worst consecutive losing streak across 45 paths |
| 9 | `seq_median_consecutive_losses` | Median longest streak across 45 paths |
| 10 | `seq_drawdown_duration_worst` | Trading days peak-to-recovery, worst case |
| 11 | `seq_drawdown_duration_median` | Median recovery duration |
| 12 | `seq_win_rate` | Fraction of sequential trades with PnL > 0 |
| 13 | `seq_win_rate_dir_bars` | Win rate when the entry bar's true label was directional (tb_label != 0) |
| 14 | `seq_hold_skip_rate` | Fraction of available entry bars skipped due to hold prediction |
| 15 | `seq_avg_bars_held` | Average holding period in bars (multiply by 5 for seconds) |

### Secondary — Time Distribution and Concurrent Positions

| # | Metric | Description |
|---|--------|-------------|
| 16 | `time_of_day_distribution` | Sequential entry count by 30-min RTH bucket (using minutes_since_open) |
| 17 | `concurrent_positions_mean` | Average bar-level positions open at any moment |
| 18 | `concurrent_positions_max` | Peak concurrent positions (scaling ceiling) |
| 19 | `concurrent_positions_p95` | 95th percentile concurrent positions |

### Account Sizing

| # | Metric | Description |
|---|--------|-------------|
| 20 | `min_account_survive_all` | Smallest account where ALL 45 paths survive (equity never hits $0) |
| 21 | `min_account_survive_95pct` | Account where <= 5% of paths breach $0 |
| 22 | `calmar_ratio` | Annualized return / worst max drawdown |
| 23 | `daily_pnl_percentiles` | p5, p25, p50, p75, p95 of daily PnL (pooled across all paths) |
| 24 | `annual_expectancy_1mes` | seq_daily_pnl_mean * 251 trading days |

### Sanity Checks

| # | Metric | Expected | Failure meaning |
|---|--------|----------|-----------------|
| S1 | `bar_level_exp_split0` | Within $0.05 of PR #38 split_00 | Training code diverged — ABORT |
| S2 | `seq_hold_skip_rate` | 40-50% (consistent with 43% CPCV hold fraction) | Sequential logic is entering on hold predictions or skipping directional ones |
| S3 | `total_test_bars_processed` | Equal to CPCV total test bars | Data loading or split logic changed |
| S4 | `seq_trades_per_day` ≤ `dir_predictions_per_day` | Always true | Sequential cannot execute MORE trades than there are directional signals |
| S5 | `seq_avg_bars_held` | 50-150 (centered near ~75 bar average from CPCV) | Barrier race duration accounting error |

### Comparison Table (bar-level vs sequential)

Report BOTH perspectives to make the relationship clear:

| Metric | Bar-Level (from CPCV PR #38) | Sequential (1 MES) |
|--------|------------------------------|-------------------|
| Trades/day | ~4,028 | ? |
| Expectancy/trade | $1.81 | ? |
| Daily PnL | ? (theoretical) | ? |
| Annual PnL | ? (theoretical) | ? |
| Hold fraction | 43% | 0% (skipped) |
| Concurrent positions | ? (measured) | 1 |
| Scaling check | concurrent_mean * seq_exp ~ bar_level * count | sanity |

---

## Success Criteria (immutable once RUN begins)

- [ ] **SC-1**: Sequential simulation completes for all 45 CPCV splits without errors
- [ ] **SC-2**: `seq_trades_per_day_mean` is in range [20, 120] (sanity: 390 min / avg_hold_minutes)
- [ ] **SC-3**: `seq_expectancy_per_trade` ≥ $0.50 under corrected-base costs ($2.49 RT)
- [ ] **SC-4**: `min_account_survive_all` ≤ $5,000 (viable for small accounts)
- [ ] **SC-5**: Bar-level vs sequential comparison table is fully populated
- [ ] **SC-6**: Concurrent positions analysis completed (mean, max, p95)
- [ ] **SC-7**: All output files are written to `.kit/results/trade-level-risk-metrics/`
- [ ] **SC-8**: `seq_hold_skip_rate` is in range [35%, 55%] (consistent with CPCV ~43% hold fraction)
- [ ] **SC-S**: No sanity check (S1-S5) fails beyond stated tolerances

---

## Protocol

### Step 0: Adapt the CPCV Pipeline

Start from `.kit/results/cpcv-corrected-costs/run_experiment.py`. The training is identical. Key adaptations:

1. **Keep the entire CPCV training pipeline unchanged** — same groups, purge, embargo, two-stage training, identical model fits.
2. **Add sequential execution simulator** — apply to each split's test set after predictions are computed.
3. **Add concurrent position counter** — count overlapping bar-level barrier races in the test set.
4. **Add risk metric computation** — drawdowns, streaks, account sizing from the sequential equity curves.
5. **Add daily PnL tracking** — aggregate sequential trade PnL by trading day.
6. **Pre-compute fwd_return_720** — same as CPCV (cumulative 720-bar forward return for hold/timeout bars).

### Step 1: Train Models (Identical to CPCV)

For each of 45 splits:
1. Apply purge and embargo (identical to PR #38)
2. Split training fold: 80% train, 20% internal validation
3. Train Stage 1 (is_directional) + Stage 2 (is_long) with early stopping
4. Predict on test set
5. Combine at T=0.50

**These predictions must be IDENTICAL to CPCV PR #38** (same seed, same splits, same data, same hyperparameters).

### Step 2: Sequential Execution Simulation

For each split's test set:
1. Sort test bars by timestamp (they should already be chronological within each day)
2. Group bars by trading day
3. For each day:
   a. Simulate sequential execution per the protocol above
   b. Record each executed trade to trade_log
   c. Track running equity curve
4. Compute per-split metrics: trades/day, expectancy, win rate, max drawdown, max consecutive losses, drawdown duration, hold-skip rate, avg bars held

### Step 3: Concurrent Position Analysis

For each split's test set:
1. For each bar where prediction is directional (non-zero), mark it as "open" for tb_bars_held bars
2. At each bar, count overlapping open positions
3. Compute mean, max, p95 of the concurrent position count

### Step 4: Aggregate Across 45 Splits

1. Pool all sequential trades across splits for overall metrics
2. Compute per-split risk metrics: max drawdown, consecutive losses, drawdown duration
3. Rank across 45 paths for worst/median statistics
4. Account sizing: for each account level ($500 to $10,000 in $100 steps), count how many of 45 paths have equity that never drops below $0. Find the threshold where all 45 survive and where 95% survive.

### Step 5: Comparison Table

1. Pull bar-level metrics from CPCV results (PR #38 metrics.json)
2. Compute bar-level theoretical daily PnL: mean_expectancy * mean_trades_per_day
3. Fill in sequential column from this experiment's results
4. Compute scaling check: concurrent_mean * seq_expectancy ~ bar_level_theoretical_daily_pnl

---

## Minimum Viable Experiment

Verify the sequential simulator on a single CPCV split before running all 45:

1. Run split 0 only
2. Check: seq_trades_per_day is in [20, 120]
3. Check: hold_skip_rate is ~43% (consistent with CPCV hold fraction)
4. Check: bar-level predictions for split 0 match CPCV PR #38 (same expectancy at base cost)

**MVE pass:** Checks 1-3 pass. **ABORT if** bar-level expectancy for split 0 doesn't match PR #38 within $0.01 (training divergence). **ABORT if** sequential trades/day < 5 or > 500 (simulation logic bug).

---

## Abort Criteria

- **Bar-level prediction mismatch:** If split 0's bar-level expectancy differs from PR #38's split_00 by >$0.05. Training code was modified.
- **Zero sequential trades:** If any split produces 0 sequential trades on a test day with predictions available. Simulation logic bug.
- **seq_trades_per_day > 500:** Not possible with ~75-bar average hold. Simulation is not advancing correctly.
- **seq_trades_per_day < 5:** Suspiciously low. Check if simulation is stuck or advancing too far.
- **NaN in PnL:** Computation bug.
- **Wall-clock > 15 min:** Something wrong — CPCV ran in 2.6 min, this adds ~30s of simulation.

---

## Decision Rules

```
OUTCOME A — SC-1 through SC-8 + SC-S all pass (seq_exp ≥ $0.50, account ≤ $5k):
  -> Sequential execution is viable at small account sizes.
  -> Next: Paper trading infrastructure (R|API+ integration, 1 /MES).
  -> If min_account_survive_all <= $2,000: highly feasible.
  -> If min_account_survive_all in [$2,000, $5,000]: feasible with risk awareness.

OUTCOME B — SC-1/2/5/6/7/8/S pass BUT seq_expectancy < $0.50 AND > $0:
  -> Sequential execution is positive but below viable deployment threshold.
  -> Report: decomposition of bar-level ($1.81) → sequential gap.
  -> How much comes from hold-skip? From signal selection bias?
  -> Next: Multi-contract scaling (use concurrent_positions_mean contracts),
     or entry timing optimization (skip certain time-of-day windows).

OUTCOME C — SC-1/2/5/6/7/8/S pass BUT seq_expectancy ≤ $0:
  -> Sequential execution is NOT profitable despite bar-level $1.81.
  -> The gap is structural: hold-skip + signal selection destroy the edge.
  -> Next: Consider entering on hold predictions too, or
     alternative entry timing strategies.

OUTCOME D — Simulation fails (SC-1 or SC-2 fail):
  -> Implementation bug. Debug and retry.
```

---

## Confounds to Watch For

1. **Hold-skip effect on expectancy.** Skipping hold bars (43% of predictions) means we only trade on directional predictions. If directional bars have higher expectancy than the overall $1.81 (because hold bars drag down the mean), sequential expectancy could be HIGHER. Conversely, if the $1.81 includes favorable hold-bar forward returns, sequential could be lower.

2. **Signal selection bias.** Under sequential execution, you enter at the first available signal after the previous trade exits. This is NOT a random sample of all directional signals — it's the first one in time. If signal quality varies within the day (e.g., morning signals are better), sequential execution oversamples certain time windows. The `time_of_day_distribution` metric measures this.

3. **Barrier race duration variance.** Average hold is ~75 bars, but tb_bars_held has high variance. Short barrier races (5-10 bars) allow more trades per day; long races (200+ bars) reduce daily count. Sequential trades/day depends on the DISTRIBUTION of hold times, not just the mean.

4. **Day-boundary truncation.** Positions that enter late in the day may not fully resolve before market close. The simulation uses the Parquet's `tb_bars_held` which may extend beyond the day's available bars. Handle this by using whatever bars remain.

5. **CPCV temporal mixing.** Same confound as PR #38 — CPCV train sets include groups from both before and after the test period. This affects model quality, not the sequential simulation itself. Sequential metrics inherit whatever bias exists in the CPCV predictions.

---

## Resource Budget

**Tier:** Quick

- Max wall-clock time: 15 min (CPCV ran in 2.6 min; simulation adds ~30s per split)
- Max training runs: 90 (45 CPCV splits * 2 stages — identical to PR #38)
- Seeds: 1 (seed=42, per-split: 42 + split_idx)
- COMPUTE_TARGET: local
- GPU hours: 0

**Tier justification:** This is a measurement/simulation experiment on top of an already-validated pipeline (CPCV PR #38). No new model architecture, no hyperparameter search, no multi-seed replication. The XGBoost training is identical; the only new computation is the sequential execution simulator (~30s total). 1 seed is sufficient because the 45 CPCV splits already provide distributional statistics.

### Compute Profile
```yaml
compute_type: cpu
estimated_rows: 1160150
model_type: xgboost
sequential_fits: 90
parallelizable: true
memory_gb: 2
gpu_type: none
estimated_wall_hours: 0.15
```

---

## Deliverables

```
.kit/results/trade-level-risk-metrics/
  metrics.json                # All risk metrics (both bar-level and sequential)
  analysis.md                 # Risk analysis, account sizing, bar-vs-sequential comparison
  run_experiment.py           # CPCV with sequential execution simulator
  spec.md                     # Spec copy
  trade_log.csv               # Per-trade: split_idx, entry_ts, exit_ts, direction, exit_type, pnl, bars_held
  equity_curves.csv           # Per-split cumulative PnL (sequential)
  drawdown_summary.csv        # Per-split: max_dd, max_streak, dd_duration, n_trades
  time_of_day.csv             # Entry distribution by 30-min RTH bucket
  daily_pnl.csv               # Per-day, per-split PnL
  account_sizing.csv          # Account size vs survival rate curve ($500 to $10k in $100 steps)
```

### Required Outputs in analysis.md

1. Executive summary (2-3 sentences: sequential risk metrics, headline account size)
2. **Sequential execution results** — trades/day, expectancy, daily PnL, win rate
3. **Risk metrics** — max drawdown, consecutive losses, drawdown duration across 45 paths
4. **Account sizing** — min_account_survive_all, survive_95pct, survival curve
5. **Time of day distribution** — when do sequential trades cluster?
6. **Concurrent position analysis** — scaling ceiling
7. **Bar-level vs sequential comparison table** — fully populated
8. **Hold-skip analysis** — how much of the expectancy gap comes from skipping hold bars?
9. **Calmar ratio and daily PnL percentiles**
10. **Explicit SC-1 through SC-8 pass/fail**
11. **Outcome verdict (A/B/C/D)**

---

## Exit Criteria

- [ ] Sequential simulation runs on all 45 CPCV splits
- [ ] `seq_trades_per_day_mean` reported — in range [20, 120]
- [ ] `seq_expectancy_per_trade` reported
- [ ] `seq_daily_pnl_mean` and `seq_daily_pnl_std` reported
- [ ] `seq_max_drawdown_worst` and `seq_max_drawdown_median` reported
- [ ] `seq_max_consecutive_losses` and `seq_median_consecutive_losses` reported
- [ ] `seq_drawdown_duration_worst` and `seq_drawdown_duration_median` reported
- [ ] `seq_win_rate` and `seq_win_rate_dir_bars` reported
- [ ] `seq_hold_skip_rate` reported
- [ ] `seq_avg_bars_held` reported
- [ ] `time_of_day_distribution` computed and saved
- [ ] `concurrent_positions_mean`, `concurrent_positions_max`, `concurrent_positions_p95` reported
- [ ] `min_account_survive_all` and `min_account_survive_95pct` reported
- [ ] `calmar_ratio` reported
- [ ] `daily_pnl_percentiles` (p5, p25, p50, p75, p95) reported
- [ ] `annual_expectancy_1mes` reported
- [ ] Bar-level vs sequential comparison table fully populated
- [ ] All output files written to `.kit/results/trade-level-risk-metrics/`
- [ ] metrics.json contains all metrics
- [ ] analysis.md contains all required sections
- [ ] trade_log.csv, equity_curves.csv, drawdown_summary.csv, time_of_day.csv, daily_pnl.csv, account_sizing.csv written
- [ ] Scaling sanity check: concurrent_mean * seq_expectancy ~ bar_level_daily_pnl (within 2x)

---

## Key References

- **CPCV pipeline:** `.kit/results/cpcv-corrected-costs/run_experiment.py` — adapt from this
- **CPCV metrics:** `.kit/results/cpcv-corrected-costs/metrics.json` — bar-level results for comparison
- **CPCV per-split:** `.kit/results/cpcv-corrected-costs/cpcv_per_split.csv` — per-split bar-level expectancy (verify match)
- **CPCV split details:** `.kit/results/cpcv-corrected-costs/split_00.json` through `split_44.json` — per-split predictions if needed
- **Parquet data:** `.kit/results/label-geometry-1h/geom_19_7/` (152-col, 3600s horizon, bidirectional labels)
- **PnL constants:** tick_value=$1.25, tick_size=0.25, target=19 ticks ($23.75), stop=7 ticks ($8.75)
- **Corrected-base cost:** $2.49 RT (AMP volume-tiered, $0.62/side + 1 tick spread)
- **Key Parquet columns:** `tb_label`, `tb_exit_type`, `tb_bars_held`, `fwd_return_1`, `timestamp`, `day`, `is_warmup`, `minutes_since_open`
