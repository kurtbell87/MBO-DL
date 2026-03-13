# Experiment: Daily Stop Loss — Sequential Execution

**Date:** 2026-02-27
**Priority:** P0 — compress drawdowns to reduce minimum account size from $34K
**Parent:** Timeout-Filtered Sequential Execution (CONFIRMED modified-A, PR #40)
**Depends on:**
1. Timeout-filtered sequential pipeline and results — DONE (PR #40, cutoff=270)
2. CPCV corrected-costs pipeline — DONE (PR #38)
3. Label geometry 1h Parquet data — DONE (PR #33)

**All prerequisites DONE. Adds intra-day stop loss to sequential simulation from PR #40 to truncate daily variance.**

---

## Hypothesis

Adding a daily stop loss (DSL) — halting all trading for the remainder of a day when cumulative intra-day P&L drops to -$X — compresses max drawdown enough to reduce min_account_survive_all from $34K to <= $20K while preserving annual PnL >= $50K and Calmar >= 2.0.

**Direction:** Lower worst-path drawdown, lower daily PnL variance (truncated left tail).
**Magnitude:** $20K is a 41% reduction from $34K. $50K annual PnL is ~80% of baseline ~$63K.
**Rationale:** PR #40 (cutoff=270) achieved expectancy $3.02/trade, Sharpe 2.27, Calmar 2.49, but min_account_survive_all = $34K. The drawdown is driven by multi-loss streaks within individual days. Capping daily losses directly truncates the worst-case daily P&L, compressing the equity curve's drawdown profile. This is variance truncation — it sacrifices upside recovery potential on stopped days in exchange for bounded daily losses.

---

## Independent Variables

### Daily Stop Loss threshold (primary IV — 9 levels)

When cumulative intra-day P&L drops to -$X, stop trading for the rest of that day. The triggering trade completes; no new trades are entered after.

| DSL Level | Description |
|-----------|-------------|
| None (no DSL) | Baseline — identical to PR #40 cutoff=270 |
| $5000 | Very loose — only stops catastrophic days |
| $4000 | Loose |
| $3000 | Moderate-loose |
| $2500 | Moderate |
| $2000 | Moderate-aggressive |
| $1500 | Aggressive |
| $1000 | Very aggressive |
| $500 | Extreme — stops after ~2 consecutive losses |

Report ALL 9 levels.

### Pipeline configuration (FIXED — identical to PR #38/#39/#40)

Same two-stage XGBoost, same CPCV splits, same training. Fixed cutoff=270 (from PR #40). Only the daily stop loss mechanism changes.

---

## Controls

| Control | Value | Rationale |
|---------|-------|-----------|
| Data source | `.kit/results/label-geometry-1h/geom_19_7/` (152-col Parquet) | Same data as PR #38/#39/#40 |
| CPCV protocol | N=10, k=2, 45 splits, purge=500, embargo=4600 | Identical |
| Training | Same two-stage XGBoost with early stopping | Identical model fits |
| Seed | 42 (per-split: 42 + split_idx) | Identical |
| RT cost | $2.49 (corrected-base) | Same cost model |
| Sequential execution | Same protocol as PR #39/#40 | Same simulation logic |
| Time cutoff | 270 (fixed) | Best cutoff from PR #40 |
| PnL model | Barrier-hit: WIN_PNL=$23.75, LOSS_PNL=$8.75. Timeout: fwd_return_720 x $1.25 x sign - $2.49 | Identical to PR #40 |

---

## Metrics (ALL must be reported)

### Primary

| # | Metric | Description |
|---|--------|-------------|
| 1 | `dsl_sweep_table` | 9-row table: DSL_level x {trades/day, exp/trade, daily_pnl_mean, daily_pnl_std, daily_pnl_skew, daily_pnl_kurtosis, dd_worst, dd_median, min_acct_all, min_acct_95, calmar, sharpe, annual_pnl, win_rate, dsl_trigger_rate, recovery_sacrifice_rate} |
| 2 | `optimal_dsl_min_account_all` | min_account_survive_all at the recommended DSL threshold |
| 3 | `optimal_dsl_annual_pnl` | Annual PnL at recommended DSL threshold |

### Secondary

| # | Metric | Description |
|---|--------|-------------|
| 4 | `recovery_sacrifice_table` | 8-row table (per non-None DSL): fraction of days where (a) intra-day cumulative P&L dipped below -$X AND (b) day ended positive without DSL. This is the opportunity cost metric. |
| 5 | `intraday_pnl_stats` | Per-day in baseline (DSL=None): intra-day min P&L, max P&L, final P&L, trade count, max consecutive losses within day |
| 6 | `optimal_dsl_calmar` | Calmar ratio at recommended DSL |
| 7 | `optimal_dsl_sharpe` | Annualized Sharpe at recommended DSL |
| 8 | `optimal_dsl_trades_per_day` | Mean sequential trades/day at recommended DSL |
| 9 | `daily_pnl_percentiles_optimal` | p1, p5, p10, p25, p50, p75, p90, p95, p99 of daily PnL at recommended DSL |
| 10 | `dsl_trigger_rate_by_level` | Fraction of trading days where DSL was triggered, per DSL level |

### Sanity Checks

| # | Metric | Expected | Failure meaning |
|---|--------|----------|-----------------|
| SC-S1 | `bar_level_exp_split0` | Within $0.01 of PR #38 split_00 | Training code diverged — ABORT |
| SC-S2 | `dsl_none_exp` | $3.02 +/- $0.10 | DSL=None must reproduce PR #40 cutoff=270 — ABORT |
| SC-S3 | `dsl_none_trades_per_day` | 116.8 +/- 5 | DSL=None must reproduce PR #40 cutoff=270 — ABORT |
| SC-S4 | `dsl_none_min_acct_all` | $34K +/- $1K | DSL=None must reproduce PR #40 cutoff=270 — ABORT |
| SC-S5 | `monotonicity_trades` | Trades/day non-increasing as DSL tightens (None → $500) | Tighter DSL can only remove trades |

---

## Baselines

| Baseline | Source | Key Metrics |
|----------|--------|-------------|
| **Cutoff=270 (PR #40)** | `.kit/results/timeout-filtered-sequential/metrics.json` | Exp $3.02/trade, 116.8 trades/day, daily PnL ~$352, Sharpe 2.27, Calmar 2.49, min_acct_all $34K, win_rate ~50.8% |
| **Sequential unfiltered (PR #39)** | `.kit/results/trade-level-risk-metrics/metrics.json` | Exp $2.50/trade, 162.2 trades/day, DD worst $47,894, min acct $48K |
| **Bar-level CPCV (PR #38)** | `.kit/results/cpcv-corrected-costs/metrics.json` | Exp $1.81/trade, PBO 0.067, break-even RT $4.30 |

**Reproduction protocol:** The DSL=None simulation at cutoff=270 must reproduce PR #40's metrics within tolerances (SC-S2, SC-S3, SC-S4). Training is identical. Divergence > tolerance triggers ABORT.

---

## Success Criteria (immutable once RUN begins)

- [ ] **SC-1**: Min account (all) <= $20,000 at recommended DSL level
- [ ] **SC-2**: Annual PnL >= $50,000 at recommended DSL level
- [ ] **SC-3**: Calmar >= 2.0 at recommended DSL level
- [ ] **SC-4**: Recovery sacrifice rate <= 20% at recommended DSL level
- [ ] **SC-5**: All 9 DSL levels run on all 45 CPCV splits (9 x 45 = 405 simulations)
- [ ] **SC-6**: Per-DSL sweep table fully populated (9 rows x 16 columns)
- [ ] **SC-7**: Recovery sacrifice analysis for all 8 non-baseline DSL levels
- [ ] **SC-8**: All output files written to `.kit/results/daily-stop-loss-sequential/`
- [ ] **SC-9**: Bar-level split 0 matches PR #38 within $0.01 (training unchanged)
- [ ] No sanity check (SC-S1 through SC-S5) fails beyond stated tolerances

---

## Minimum Viable Experiment

Before running the full 405 simulations, run a single-split gate:

1. Train split 0 only (identical to PR #38/#39/#40)
2. Run sequential simulation at DSL=None, cutoff=270 — verify matches PR #40 split 0 within tolerances
3. Run sequential simulation at DSL=$1500, cutoff=270 — verify it produces fewer or equal trades than DSL=None
4. Check bar-level expectancy for split 0 matches PR #38 within $0.01

**MVE pass criteria:**
- Split 0 DSL=None expectancy within $0.10 of PR #40's $3.02
- Split 0 DSL=None trades/day within 5 of PR #40's 116.8
- Split 0 DSL=None min_acct_all within $1K of PR #40's $34K
- Split 0 DSL=$1500 trades/day <= DSL=None trades/day
- Bar-level expectancy match (SC-S1)

**MVE ABORT if:** bar-level mismatch > $0.05 (training diverged), OR DSL=$1500 produces MORE trades than DSL=None (logic bug), OR any NaN in metrics.

If MVE passes, proceed to full 45-split x 9-DSL sweep.

---

## Full Protocol

### Step 0: Fork the Timeout-Filtered Sequential Pipeline

Start from `.kit/results/timeout-filtered-sequential/run_experiment.py`. The training, prediction, and time cutoff are identical. Key adaptations:

1. **Add `daily_stop_loss` parameter** to `simulate_sequential()`. At the start of each while-loop iteration (before hold-skip and time-cutoff checks), check: `if daily_stop_loss is not None and day_pnl_total <= -daily_stop_loss: break`. The triggering trade that pushed P&L below the threshold completes; no new trades are entered after that for the rest of the day.

2. **Track cumulative intra-day P&L** within `simulate_sequential()`. At each trade completion, update `day_pnl_total += trade_pnl`. Reset to 0 at the start of each new day.

3. **Replace `CUTOFF_LEVELS` sweep with `DSL_LEVELS` sweep.** Fixed cutoff=270 for all simulations. `DSL_LEVELS = [None, 5000, 4000, 3000, 2500, 2000, 1500, 1000, 500]`. CPCV loop: train once per split, simulate 9x (one per DSL level).

4. **Recovery sacrifice analysis** — using DSL=None baseline trade logs, compute per-DSL threshold: for each trading day, track the cumulative intra-day P&L path. For each DSL level $X, count:
   - (a) Days where cumulative intra-day P&L dipped below -$X at some point
   - (b) Of those days, how many ended with positive final P&L (i.e., recovered)
   - Recovery sacrifice rate = (b) / (a) — fraction of stopped days that would have recovered

5. **Intra-day P&L path statistics** — for each day in the DSL=None baseline, compute:
   - Intra-day min cumulative P&L (worst point in the day)
   - Intra-day max cumulative P&L (best point in the day)
   - Final P&L (end of day)
   - Trade count
   - Max consecutive losses within the day

### Step 1: Reproduce the Baseline (MVE)

Run split 0 at DSL=None, cutoff=270 and DSL=$1500, cutoff=270. Verify MVE pass criteria above. If ABORT → stop. If PASS → proceed.

### Step 2: Full Sweep

For each of 45 splits:
1. Train two-stage XGBoost (identical to PR #38/#39/#40)
2. For each of 9 DSL levels, run sequential simulation with cutoff=270
3. Record per-split per-DSL: trades/day, expectancy, win rate, daily PnL mean/std/skew/kurtosis, max drawdown, min account, Calmar, Sharpe, annual PnL, DSL trigger rate

### Step 3: Aggregate and Select Optimal DSL

For each DSL level (across 45 splits):
1. Compute all primary metrics (same aggregation as PR #39/#40)
2. Compute account sizing curve ($500 to $50,000 in $500 steps)

**Optimal DSL selection rule:** The recommended DSL is the **loosest (highest dollar) threshold** that achieves ALL of SC-1 (min_acct <= $20K), SC-2 (annual PnL >= $50K), SC-3 (Calmar >= 2.0), and SC-4 (recovery sacrifice <= 20%). "Loosest" means the least restrictive stop — we prefer stopping less often if the targets are already met.

If NO DSL achieves all four criteria: report the DSL that **maximizes Calmar ratio** subject to annual PnL >= $40K (relaxed SC-2). If none meets even the relaxed criterion, report the DSL that minimizes min_account_all.

### Step 4: Detailed Analysis at Recommended DSL

For the recommended DSL, produce the full analysis:
- Equity curves, trade log, drawdown summary, daily PnL distribution, account sizing
- Comparison table: DSL=None (baseline) vs recommended DSL
- Recovery sacrifice analysis for all 8 non-baseline DSL levels
- Intra-day P&L path statistics from baseline

---

## Resource Budget

**Tier:** Quick

- Max GPU-hours: 0
- Max wall-clock time: 15 min
- Max training runs: 90 (identical to PR #38/#39/#40 — 45 splits x 2 stages)
- Simulations: 405 (45 splits x 9 DSL levels, ~0.1s each)
- Max seeds per configuration: 1

### Compute Profile
```yaml
compute_type: cpu
estimated_rows: 1160150
model_type: xgboost
sequential_fits: 90
parallelizable: true
memory_gb: 2
gpu_type: none
estimated_wall_hours: 0.10
```

### Wall-Time Estimation

- XGBoost training: 90 fits on ~1.16M rows ~ 2.6 min (benchmarked from PR #38)
- Sequential simulation: 405 runs at ~0.1s each ~ 40s
- Recovery sacrifice analysis: ~10s (iterate baseline trade logs)
- Data loading + aggregation: ~30s
- **Total estimated: ~4 min.** Budget 15 min as 3.5x safety margin.

---

## Abort Criteria

- **Bar-level mismatch:** Split 0 bar-level expectancy differs from PR #38 by > $0.05. ABORT — training code diverged.
- **DSL=None mismatch:** Must match PR #40 cutoff=270 within tolerances (exp $3.02 +/- $0.10, trades/day 116.8 +/- 5, min_acct $34K +/- $1K). ABORT if not.
- **Wall-clock > 15 min:** Expected ~4 min. 15 min = 3.75x. ABORT.
- **NaN in any metric:** Computation bug. ABORT.
- **DSL=None zero trades on any split:** Simulation logic bug. ABORT.
- **Monotonicity violation on trades/day:** If any tighter DSL produces MORE trades/day than a looser DSL, the DSL logic is inverted. ABORT.

---

## Confounds to Watch For

1. **DSL may disproportionately stop the best recovery days.** If the model's worst intra-day drawdowns tend to occur early on days that eventually have strong recoveries, DSL truncates the high-variance, high-mean days. The recovery sacrifice metric directly measures this.

2. **DSL interacts with sequential execution.** Stopping mid-day changes which trades are executed on subsequent days (no, it doesn't — days are independent in sequential simulation). But within a day, stopping early means later potentially profitable signals are not executed.

3. **DSL threshold in dollar terms depends on the P&L model.** At 19:7 geometry, a single loss is -$8.75 - $2.49 = -$11.24. DSL=$500 triggers after ~44 consecutive losses, which is extremely unlikely. More realistically, a mix of wins and losses. A bad streak of 5 losses in a row = -$56.20, which only triggers DSL=$500 (not $1000+). The effective range of DSL may be narrow.

4. **Correlation between DSL trigger and split quality.** The worst splits (18, 32) may trigger DSL more often, which is desirable — DSL helps most where it's needed most. But if the best splits also trigger often, DSL is too aggressive.

5. **DSL=$500 is nearly meaningless.** At $11.24/loss, even 2 consecutive losses = -$22.48. With 49.2% win rate, 3 consecutive losses are common (~13% of 3-trade sequences). DSL=$500 stops almost every day. Expected to be destructive.

---

## Decision Rules

```
OUTCOME A — SC-1 through SC-9 all pass:
  -> DSL is effective. Bounded daily losses compress drawdown without excessive PnL sacrifice.
  -> Report recommended DSL threshold and full risk profile.
  -> Next: Stack with barrier geometry exploration if further compression needed.

OUTCOME B — Some SC pass (partial effectiveness):
  -> If SC-1 passes but SC-2 fails: drawdown compresses but too much PnL sacrificed.
     -> Report the tradeoff curve (DSL vs annual PnL vs min_account).
     -> Consider accepting higher min_account with better PnL.
  -> If SC-2/SC-3 pass but SC-1 fails: DSL helps PnL stability but doesn't compress
     drawdown enough (losses are spread across days, not concentrated within days).
     -> Drawdown is inter-day, not intra-day. DSL addresses the wrong timescale.
  -> If SC-4 fails (recovery sacrifice > 20%): DSL is cutting off too many recovery days.
     -> Try a looser DSL that sacrifices less recovery.

OUTCOME C — No meaningful effect (min_acct within $2K of baseline at ALL DSL levels):
  -> Daily losses are NOT recoverable intra-day. The drawdown is driven by multi-day
     loss streaks, not intra-day loss clustering.
  -> DSL approach is fundamentally wrong for this problem.
  -> Next: Barrier geometry exploration (Experiment B) — attack the payoff structure
     directly.

OUTCOME D — Simulation failure or sanity check abort:
  -> Implementation bug. Debug and retry.
```

---

## Deliverables

```
.kit/results/daily-stop-loss-sequential/
  run_experiment.py               # Pipeline forked from timeout-filtered-sequential
  metrics.json                    # All metrics: per-DSL summary + optimal DSL details
  analysis.md                     # Full analysis with DSL sweep, recovery sacrifice, comparison
  spec.md                         # Spec copy
  dsl_sweep.csv                   # 9 rows x 16 metrics
  recovery_sacrifice.csv          # 8 rows x sacrifice metrics (one per non-baseline DSL)
  intraday_pnl_stats.csv          # Per-day path statistics from baseline
  optimal_trade_log.csv           # Trade log at recommended DSL
  optimal_equity_curves.csv       # Per-split equity curves at recommended DSL
  optimal_drawdown_summary.csv    # Per-split drawdowns at recommended DSL
  optimal_daily_pnl.csv           # Per-day PnL distribution at recommended DSL
  optimal_account_sizing.csv      # Account size vs survival at recommended DSL ($500-$50K in $500 steps)
  comparison_table.csv            # DSL=None vs recommended DSL side-by-side
```

### Required Outputs in analysis.md

1. Executive summary (2-3 sentences)
2. **DSL sweep results** — 9-row table with all metrics per DSL level
3. **Recovery sacrifice analysis** — for each DSL level, what fraction of stopped days would have recovered?
4. **Intra-day P&L path analysis** — distribution of intra-day min/max/final P&L, trade counts
5. **Optimal DSL selection** — which threshold, why, per the selection rule in Step 3
6. **Detailed results at recommended DSL** — full risk profile
7. **DSL=None vs recommended DSL comparison** — side-by-side table with deltas and % changes
8. **Account sizing at recommended DSL** — survival curve, min account all/95%
9. **Daily PnL distribution** — percentiles, distribution shape comparison
10. **Explicit SC-1 through SC-9 + SC-S1 through SC-S5 pass/fail**
11. **Outcome verdict (A/B/C/D)**

---

## Exit Criteria

- [ ] All 9 DSL levels simulated across all 45 CPCV splits (405 simulations)
- [ ] Per-DSL summary table populated (9 rows x 16+ columns)
- [ ] Recovery sacrifice analysis for all 8 non-baseline DSL levels
- [ ] Intra-day P&L path statistics from baseline computed
- [ ] Optimal DSL identified with rationale per selection rule
- [ ] Full risk metrics at recommended DSL
- [ ] DSL=None vs recommended DSL comparison table populated
- [ ] Account sizing curve at recommended DSL ($500-$50K)
- [ ] min_account_survive_all and min_account_survive_95pct at recommended DSL reported
- [ ] All output files written to `.kit/results/daily-stop-loss-sequential/`
- [ ] metrics.json and analysis.md complete
- [ ] Bar-level split 0 matches PR #38 (training unchanged)
- [ ] SC-S1 through SC-S5 all evaluated and reported

---

## Key References

- **Timeout-filtered sequential pipeline (fork from here):** `.kit/results/timeout-filtered-sequential/run_experiment.py`
- **Timeout-filtered sequential metrics (PR #40 baseline):** `.kit/results/timeout-filtered-sequential/metrics.json`
- **Sequential pipeline (PR #39):** `.kit/results/trade-level-risk-metrics/run_experiment.py`
- **CPCV pipeline (PR #38):** `.kit/results/cpcv-corrected-costs/run_experiment.py`
- **Parquet data:** `.kit/results/label-geometry-1h/geom_19_7/`
- **PnL constants:** tick_value=$1.25, target=19 ticks ($23.75), stop=7 ticks ($8.75), RT=$2.49
- **Key columns:** `tb_label`, `tb_exit_type`, `tb_bars_held`, `fwd_return_1`, `timestamp`, `day`, `minutes_since_open`
