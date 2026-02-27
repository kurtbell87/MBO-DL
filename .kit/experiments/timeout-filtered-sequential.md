# Experiment: Timeout-Filtered Sequential Execution

**Date:** 2026-02-27
**Priority:** P0 — reduce drawdown and improve per-trade expectancy via timeout avoidance
**Parent:** Trade-Level Risk Metrics (REFUTED modified-A, PR #39)
**Depends on:**
1. Trade-level risk metrics pipeline and results — DONE (PR #39)
2. CPCV corrected-costs pipeline — DONE (PR #38)
3. Label geometry 1h Parquet data — DONE (PR #33)

**All prerequisites DONE. Extends the sequential simulation from PR #39 with entry-time filters to avoid timeout-prone trades.**

---

## Hypothesis

Filtering sequential entries to bars where `minutes_since_open <= cutoff` (for some cutoff < 390) increases per-trade expectancy to >= $3.50 (from $2.50 baseline) and reduces min_account_survive_all to <= $30,000 (from $48,000 baseline).

**Direction:** Higher per-trade expectancy, lower worst-path drawdown.
**Magnitude:** $3.50/trade is a 40% improvement over the $2.50 baseline, targeting removal of timeout-diluted late-day trades. $30K account is a 37% reduction from $48K.

**Rationale:** PR #39 showed sequential expectancy of $2.50/trade vs ~$5.00 for pure barrier-hit trades — a 50% dilution from timeout trades. If timeouts cluster late in the RTH session (when insufficient time remains for barrier resolution), a simple time-of-day cutoff can eliminate the worst-quality entries. The sequential avg_bars_held = 28 bars (2.3 min), so most barrier races resolve quickly, but the timeout subset drags the mean from $5 to $2.50.

---

## Independent Variables

### Time cutoff (primary IV — 7 levels)

Sweep entry cutoff as `minutes_since_open` threshold. Do NOT enter a new position if `minutes_since_open > cutoff`.

| Cutoff | RTH Minutes Remaining | Description |
|--------|----------------------|-------------|
| 390 (no filter) | 0 | Baseline — identical to PR #39 |
| 375 | 15 | Conservative — cut last 15 min |
| 360 | 30 | Moderate — cut last 30 min |
| 345 | 45 | Moderate-aggressive |
| 330 | 60 | Aggressive — cut last hour |
| 300 | 90 | Very aggressive — cut last 1.5h |
| 270 | 120 | Extreme — cut last 2h |

Report ALL 7 levels.

### Pipeline configuration (FIXED — identical to PR #38 and #39)

Same two-stage XGBoost, same CPCV splits, same training. Only the sequential simulation entry filter changes.

---

## Controls

| Control | Value | Rationale |
|---------|-------|-----------|
| Data source | `.kit/results/label-geometry-1h/geom_19_7/` (152-col Parquet) | Same data as PR #38/#39 |
| CPCV protocol | N=10, k=2, 45 splits, purge=500, embargo=4600 | Identical |
| Training | Same two-stage XGBoost with early stopping | Identical model fits |
| Seed | 42 (per-split: 42 + split_idx) | Identical |
| RT cost | $2.49 (corrected-base) | Same cost model |
| Sequential execution | Same protocol as PR #39 | Only change is the time cutoff gate |
| PnL model | Barrier-hit: +$21.26 / -$11.24. Timeout: fwd_return × $1.25 × sign - $2.49 | Identical to PR #39 |

---

## Metrics (ALL must be reported)

### Primary

| # | Metric | Description |
|---|--------|-------------|
| 1 | `cutoff_sweep_table` | 7-row table: cutoff × {trades/day, exp/trade, daily_pnl, dd_worst, dd_median, min_acct_all, min_acct_95, win_rate, time_skip_%, hold_skip_%, timeout_fraction} |
| 2 | `optimal_cutoff_expectancy` | Per-trade expectancy at the recommended cutoff |
| 3 | `optimal_cutoff_min_account_all` | min_account_survive_all at the recommended cutoff |

These three directly test the hypothesis: (1) the sweep shows the tradeoff curve, (2)+(3) test the magnitude thresholds.

### Secondary

| # | Metric | Description |
|---|--------|-------------|
| 4 | `optimal_cutoff_daily_pnl` | Mean daily PnL at recommended cutoff |
| 5 | `optimal_cutoff_calmar` | Calmar ratio at recommended cutoff |
| 6 | `optimal_cutoff_sharpe` | Annualized Sharpe at recommended cutoff |
| 7 | `optimal_cutoff_trades_per_day` | Mean sequential trades/day at recommended cutoff |
| 8 | `timeout_fraction_by_cutoff` | Fraction of executed trades that are timeouts, per cutoff level |
| 9 | `barrier_hit_fraction_by_cutoff` | Complement of timeout_fraction — should increase as cutoff tightens |
| 10 | `splits_18_32_comparison` | Expectancy, drawdown, and trade count for the 2 outlier paths (splits 18 & 32) at each cutoff |
| 11 | `daily_pnl_percentiles_optimal` | p5, p25, p50, p75, p95 of daily PnL at recommended cutoff |
| 12 | `annual_expectancy_optimal` | optimal_daily_pnl × 251 |

### Sanity Checks

| # | Metric | Expected | Failure meaning |
|---|--------|----------|-----------------|
| SC-S1 | `bar_level_exp_split0` | Within $0.01 of PR #38 split_00 | Training code diverged — ABORT |
| SC-S2 | `cutoff_390_exp` | Within $0.10 of PR #39's $2.50 | Unfiltered simulation diverged — ABORT |
| SC-S3 | `cutoff_390_trades_per_day` | Within 5 of PR #39's 162.2 | Simulation logic changed — ABORT |
| SC-S4 | `monotonicity_exp` | Expectancy non-decreasing as cutoff tightens (390 → 270) | If exp DECREASES with tighter filter, late-day trades are BETTER than average — invalidates premise |
| SC-S5 | `monotonicity_trades` | Trades/day non-increasing as cutoff tightens (390 → 270) | Impossible violation — tighter filter can only remove trades |

---

## Baselines

| Baseline | Source | Key Metrics |
|----------|--------|-------------|
| **Sequential unfiltered (PR #39)** | `.kit/results/trade-level-risk-metrics/metrics.json` | Exp $2.50/trade, 162.2 trades/day, $412.77/day, DD worst $47,894, DD median $12,917, min acct $48K/$26.6K, Calmar 2.16, Sharpe 2.27, win rate 49.93%, hold-skip 66.1%, avg bars held 28 |
| **Bar-level CPCV (PR #38)** | `.kit/results/cpcv-corrected-costs/metrics.json` | Exp $1.81/trade, PBO 0.067, break-even RT $4.30 |

**Reproduction protocol:** The cutoff=390 (no filter) simulation must reproduce PR #39's sequential metrics within tolerances (SC-S2, SC-S3). Training is identical (same code, same data, same seeds). Divergence > tolerance triggers ABORT.

---

## Success Criteria (immutable once RUN begins)

- [ ] **SC-1**: All 7 cutoff levels run on all 45 CPCV splits (7 × 45 = 315 simulations)
- [ ] **SC-2**: Optimal cutoff achieves `seq_expectancy_per_trade >= $3.50` (40% improvement over $2.50)
- [ ] **SC-3**: Optimal cutoff achieves `min_account_survive_all <= $30,000` (37% reduction from $48K)
- [ ] **SC-4**: Per-cutoff sweep table fully populated (7 rows × 11 columns)
- [ ] **SC-5**: Unfiltered vs filtered comparison table populated at recommended cutoff
- [ ] **SC-6**: Account sizing curve produced for recommended cutoff ($500 to $50K in $500 steps)
- [ ] **SC-7**: All output files written to `.kit/results/timeout-filtered-sequential/`
- [ ] **SC-8**: Bar-level split 0 matches PR #38 within $0.01 (training unchanged)
- [ ] No sanity check (SC-S1 through SC-S5) fails beyond stated tolerances

---

## Minimum Viable Experiment

Before running the full 315 simulations, run a single-split gate:

1. Train split 0 only (identical to PR #38/#39)
2. Run sequential simulation at cutoff=390 (no filter) — verify matches PR #39 split 0 within tolerances
3. Run sequential simulation at cutoff=330 (aggressive, -1h) — verify it produces fewer trades than cutoff=390
4. Check bar-level expectancy for split 0 matches PR #38 within $0.01

**MVE pass criteria:**
- Split 0 cutoff=390 trades/day within 10% of PR #39 split 0
- Split 0 cutoff=330 trades/day < cutoff=390 trades/day (monotonicity)
- Bar-level expectancy match (SC-S1)

**MVE ABORT if:** bar-level mismatch > $0.05 (training diverged), OR cutoff=330 produces MORE trades than cutoff=390 (logic bug), OR any NaN in metrics.

If MVE passes, proceed to full 45-split × 7-cutoff sweep.

---

## Full Protocol

### Step 0: Adapt the Sequential Pipeline

Start from `.kit/results/trade-level-risk-metrics/run_experiment.py`. The training and prediction are identical. Key adaptation:

1. **Add time cutoff parameter** to the sequential simulation function.
2. **At each entry opportunity:** check if `minutes_since_open <= cutoff`. If not, skip (do not enter). Log as "time-filtered skip" (distinct from hold-skip).
3. **Run the sequential simulation 7 times per split** (once per cutoff level). Models are trained ONCE per split; only the simulation loop is repeated with different cutoff values.
4. **Track per-cutoff metrics** separately.
5. **Track `timeout_fraction` per cutoff:** for each executed trade, record whether `tb_exit_type == "timeout"`. Compute fraction of timeouts among executed trades at each cutoff level.

### Step 1: Reproduce the Baseline (MVE)

Run split 0 at cutoff=390 and cutoff=330. Verify MVE pass criteria above. If ABORT → stop. If PASS → proceed.

### Step 2: Full Sweep

For each of 45 splits:
1. Train two-stage XGBoost (identical to PR #38/#39)
2. For each of 7 cutoff levels, run sequential simulation
3. Record per-split per-cutoff: trades/day, expectancy, win rate, daily PnL, max drawdown, consecutive losses, drawdown duration, hold-skip rate, time-filter-skip rate, timeout_fraction, barrier_hit_fraction, avg_bars_held

### Step 3: Aggregate and Select Optimal Cutoff

For each cutoff level (across 45 splits):
1. Compute all primary metrics (same aggregation as PR #39)
2. Compute account sizing curve ($500 to $50,000 in $500 steps)

**Optimal cutoff selection rule:** The recommended cutoff is the **most conservative (highest) cutoff value** that achieves BOTH SC-2 (exp >= $3.50) AND SC-3 (min_acct <= $30K). "Most conservative" means the least filtering needed — we prefer removing fewer trades if the targets are already met.

If NO cutoff achieves both SC-2 and SC-3: report the cutoff that **maximizes daily PnL** subject to min_account_survive_all <= $35K (relaxed SC-3). If none meets even the relaxed criterion, report the cutoff that maximizes Calmar ratio.

### Step 4: Detailed Analysis at Recommended Cutoff

For the recommended cutoff, produce the full analysis:
- Equity curves, trade log, drawdown summary, time-of-day distribution, daily PnL, account sizing
- Comparison table: unfiltered (PR #39 baseline) vs filtered (recommended)
- Effect on outlier paths (splits 18 & 32) — how does filtering affect the group-4 worst cases?

---

## Resource Budget

**Tier:** Quick

- Max GPU-hours: 0
- Max wall-clock time: 15 min
- Max training runs: 90 (identical to PR #38/#39 — 45 splits × 2 stages)
- Simulations: 315 (45 splits × 7 cutoffs, ~0.1s each)
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

- XGBoost training: 90 fits on ~1.16M rows ≈ 2.6 min (benchmarked from PR #38)
- Sequential simulation: 315 runs at ~0.1s each ≈ 30s
- Data loading + aggregation: ~30s
- **Total estimated: ~4 min.** Budget 15 min as 3.5x safety margin.

---

## Abort Criteria

- **Bar-level mismatch:** Split 0 bar-level expectancy differs from PR #38 by > $0.05. ABORT — training code diverged.
- **Cutoff 390 mismatch:** Unfiltered sequential results must match PR #39 within $0.10/trade and 10 trades/day. ABORT if not — simulation logic changed.
- **Wall-clock > 15 min:** Expected ~4 min. 15 min = 3.75x. Something wrong with training or I/O. ABORT.
- **NaN in any metric:** Computation bug. ABORT.
- **Cutoff 390 zero trades on any split:** Simulation logic bug (no-filter should always produce trades). ABORT.
- **Monotonicity violation on trades/day:** If any tighter cutoff produces MORE trades/day than a looser cutoff, the time-filter logic is inverted. ABORT.

Note: cutoff 270 producing zero trades on SOME splits is expected data (those test sets may have few late-day bars). Report it, don't abort.

---

## Confounds to Watch For

1. **Late-day entries may have higher win rate.** If market-on-close dynamics create a directional bias (e.g., MOC imbalance mean-reversion), filtering late entries could remove positive-EV trades, not just timeouts. SC-S4 (monotonicity of expectancy) would catch this — if expectancy DECREASES with tighter cutoff, the premise is wrong.

2. **Cutoff interacts with hold-skip dynamics.** Removing late-day entry windows reduces the pool of available signals. The sequential simulator enters at the FIRST available signal after exit — fewer late-day opportunities may change which signals get executed earlier in the day.

3. **Time cutoff is a proxy, not a direct timeout filter.** A trade entered at minute 300 with a 5-bar (25s) barrier race is fine; a trade entered at minute 200 with a 3600-bar race (5 hours) will timeout regardless. The time cutoff filters by WHEN you enter, not by WHETHER the barrier will resolve. The `timeout_fraction_by_cutoff` metric measures how well this proxy works — if timeout fraction doesn't decrease with tighter cutoffs, the approach is wrong (Outcome C).

4. **Account sizing depends on equity path shape, not just per-trade metrics.** Fewer trades with higher expectancy could have MORE clustered losses (if the good trades were previously interspersed with the filtered-out trades). Watch whether DD_worst decreases proportionally with the expectancy gain.

5. **Splits 18 & 32 sensitivity.** These 2 outlier paths (both containing temporal group 4) drove the $48K account requirement. If their behavior doesn't change under filtering, min_account won't improve even if median drawdown improves.

---

## Decision Rules

```
OUTCOME A — SC-1 through SC-8 all pass:
  -> Timeout filtering is effective. Fewer, richer trades with lower drawdown.
  -> Report recommended cutoff and its full risk profile.
  -> Next: Paper trading at recommended account size.

OUTCOME B — SC-1/4/5/6/7/8 pass BUT SC-2 or SC-3 fail:
  -> Time cutoff helps but not enough to hit both targets.
  -> If SC-2 passes but SC-3 fails: expectancy improves but drawdown is structural
     (driven by loss clustering or outlier paths, not timeout dilution).
  -> If SC-3 passes but SC-2 fails: drawdown compresses but per-trade edge doesn't
     improve (timeout trades weren't the expectancy drag — selection bias is).
  -> Report the best achievable numbers and the tradeoff curve.
  -> Next: Volatility-conditional filtering (skip entries when volatility_50 is below
     a threshold) or accept the account requirement as-is.

OUTCOME C — Filtering has no meaningful effect (expectancy within $0.25 of baseline
            at ALL cutoffs, including cutoff=270):
  -> Timeout trades are NOT concentrated in late-day entries. The dilution is
     uniformly distributed across the RTH session.
  -> The time-of-day filter approach is fundamentally wrong for this problem.
  -> Next: Alternative filtering — predicted barrier duration (using volatility_50
     as a proxy for expected hold time), or regime-conditional entry (Q1-Q2 only).

OUTCOME D — Simulation fails or sanity check abort:
  -> Implementation bug. Debug and retry.
```

---

## Deliverables

```
.kit/results/timeout-filtered-sequential/
  metrics.json                    # All metrics: per-cutoff summary + optimal cutoff details
  analysis.md                     # Full analysis with cutoff sweep, optimal selection, comparison
  run_experiment.py               # Pipeline with time-filtered sequential simulation
  spec.md                         # Spec copy
  cutoff_sweep.csv                # 7 cutoff levels x all metrics
  optimal_trade_log.csv           # Trade log at recommended cutoff
  optimal_equity_curves.csv       # Per-split equity curves at recommended cutoff
  optimal_drawdown_summary.csv    # Per-split drawdowns at recommended cutoff
  optimal_daily_pnl.csv           # Per-day PnL at recommended cutoff
  optimal_account_sizing.csv      # Account size vs survival at recommended cutoff ($500-$50K in $500 steps)
  optimal_time_of_day.csv         # Entry distribution at recommended cutoff
  comparison_table.csv            # Unfiltered vs filtered side-by-side
```

### Required Outputs in analysis.md

1. Executive summary (2-3 sentences)
2. **Cutoff sweep results** — 7-row table with all metrics per cutoff
3. **Timeout fraction analysis** — what fraction of trades are timeouts at each cutoff? Does tighter filtering actually reduce timeouts?
4. **Optimal cutoff selection** — which cutoff, why, per the selection rule in Step 3
5. **Detailed results at recommended cutoff** — full risk profile (all 24 metrics from PR #39 + timeout_fraction + time_filter_skip_rate + barrier_hit_fraction)
6. **Unfiltered vs filtered comparison** — side-by-side table with deltas and % changes
7. **Account sizing at recommended cutoff** — survival curve, min account all/95%
8. **Outlier path analysis (splits 18 & 32)** — how does filtering affect the group-4 worst cases?
9. **Daily PnL distribution** — percentiles, distribution shape comparison
10. **Explicit SC-1 through SC-8 + SC-S1 through SC-S5 pass/fail**
11. **Outcome verdict (A/B/C/D)**

---

## Exit Criteria

- [ ] All 7 cutoff levels simulated across all 45 CPCV splits (315 simulations)
- [ ] Per-cutoff summary table populated (7 rows × 11+ columns)
- [ ] Timeout fraction per cutoff reported (does filtering reduce timeouts?)
- [ ] Optimal cutoff identified with rationale per selection rule
- [ ] Full risk metrics at recommended cutoff (all 24 from PR #39 + timeout_fraction + time_filter_skip_rate + barrier_hit_fraction)
- [ ] Unfiltered vs filtered comparison table populated
- [ ] Account sizing curve at recommended cutoff ($500-$50K)
- [ ] min_account_survive_all and min_account_survive_95pct at recommended cutoff reported
- [ ] Effect on outlier paths (splits 18 & 32) reported
- [ ] All output files written to `.kit/results/timeout-filtered-sequential/`
- [ ] metrics.json and analysis.md complete
- [ ] Bar-level split 0 matches PR #38 (training unchanged)
- [ ] SC-S1 through SC-S5 all evaluated and reported

---

## Key References

- **Sequential pipeline:** `.kit/results/trade-level-risk-metrics/run_experiment.py` — adapt from this
- **Sequential metrics:** `.kit/results/trade-level-risk-metrics/metrics.json` — baseline for comparison
- **Sequential drawdowns:** `.kit/results/trade-level-risk-metrics/drawdown_summary.csv` — per-split baselines
- **CPCV pipeline:** `.kit/results/cpcv-corrected-costs/run_experiment.py` — training reference
- **CPCV metrics:** `.kit/results/cpcv-corrected-costs/metrics.json` — bar-level reference
- **Parquet data:** `.kit/results/label-geometry-1h/geom_19_7/`
- **PnL constants:** tick_value=$1.25, target=19 ticks ($23.75), stop=7 ticks ($8.75), RT=$2.49
- **Key columns:** `tb_label`, `tb_exit_type`, `tb_bars_held`, `fwd_return_1`, `timestamp`, `day`, `minutes_since_open`
