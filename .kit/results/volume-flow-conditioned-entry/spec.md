# Experiment: Volume-Flow Conditioned Entry

**Date:** 2026-02-27
**Priority:** P0 — reduce timeout fraction via volume/activity-based entry gating
**Parent:** Timeout-Filtered Sequential (REFUTED Outcome B, PR #40)
**Depends on:**
1. Trade-level risk metrics pipeline and results — DONE (PR #39)
2. CPCV corrected-costs pipeline — DONE (PR #38)
3. Label geometry 1h Parquet data — DONE (PR #33)
4. Timeout-filtered sequential results — DONE (PR #40)

**All prerequisites DONE. PR #40 proved timeout fraction is invariant at 41.3% regardless of time-of-day — timeouts are driven by the volume horizon (50,000 contracts), not clock time. This experiment attacks the timeout mechanism directly by conditioning entry on volume/activity features observable at entry time.**

---

## Two-Stage Architecture

This experiment uses a diagnostic-first design. PR #40 ran 315 simulations (~4 min) before discovering invariance that could have been detected in a 30-second bar-level analysis. This experiment inverts that: a zero-cost diagnostic stage first, simulation only if signal exists.

---

## Hypothesis

**Direction: UNCERTAIN** — the diagnostic resolves this empirically.

- **Hypothesis A:** High volume → faster volume horizon consumption, but also higher volatility → higher per-bar barrier-hit probability. Net effect: fewer timeouts during high-activity periods.
- **Hypothesis B:** High volume → volume horizon consumed quickly → shorter race window → more timeouts despite higher volatility.

The D5 cross-table disentangles: does volume flow reduce timeouts after controlling for volatility?

---

## Independent Variables

### Entry-time features (5 features, 4 quartile levels each)

For each bar across all 45 CPCV test sets, measure timeout fraction by quartile of these 5 entry-time features:

| Feature | Description | Source |
|---------|-------------|--------|
| `trade_count` | Per-bar trade event count | Parquet column |
| `message_rate` | Per-bar MBO event rate | Parquet column |
| `volatility_50` | 50-bar high-low range | Parquet column (dominant XGBoost feature, 49.7% gain) |
| `trade_count_20` | Rolling 20-bar mean of trade_count | Computed at analysis time |
| `message_rate_20` | Rolling 20-bar mean of message_rate | Computed at analysis time |

### Gate levels for Stage 2 sweep (3 levels per qualifying feature)

For each qualifying feature (>= 3pp range in Stage 1), sweep 3 gate levels: skip entry if feature < {p25, p50, p75} from the training fold.

### Pipeline configuration (FIXED — identical to PR #38, #39, #40)

Same two-stage XGBoost, same CPCV splits, same training. Only the sequential simulation entry gate changes.

---

## Controls

| Control | Value | Rationale |
|---------|-------|-----------|
| Data source | `.kit/results/label-geometry-1h/geom_19_7/` (152-col Parquet) | Same data as PR #38/#39/#40 |
| CPCV protocol | N=10, k=2, 45 splits, purge=500, embargo=4600 | Identical |
| Training | Same two-stage XGBoost with early stopping | Identical model fits |
| Seed | 42 (per-split: 42 + split_idx) | Identical |
| RT cost | $2.49 (corrected-base) | Same cost model |
| Sequential execution | Same protocol as PR #39 | Only change is the entry gate |
| PnL model | Barrier-hit: +$21.26 / -$11.24. Timeout: fwd_return × $1.25 × sign - $2.49 | Identical to PR #39 |

---

## Protocol

### Stage 1: Diagnostic (~30 seconds, no training)

For each bar across all 45 CPCV test sets:
1. Compute quartile assignments for each of the 5 features
2. Measure timeout fraction (from `tb_exit_type`) per quartile per feature
3. Report diagnostics D1-D5:

**D1-D3: Per-feature timeout fraction by quartile**
- D1: `trade_count` — 4 quartiles × timeout fraction
- D2: `message_rate` — 4 quartiles × timeout fraction
- D3: `volatility_50` — 4 quartiles × timeout fraction

**D4: Rolling features**
- `trade_count_20` — 4 quartiles × timeout fraction
- `message_rate_20` — 4 quartiles × timeout fraction

**D5: Cross-table (disentangles volume from volatility)**
- `volatility_50` quartile × `trade_count` quartile → timeout fraction (4×4 = 16 cells)

**Diagnostic tiers:**
- **>= 5pp range** (max quartile - min quartile): Feature qualifies for Stage 2 sweep (strong signal)
- **3-5pp range**: Feature flagged as "marginal" — include in Stage 2 but note lower confidence. Marginal features may compound meaningfully when stacked with cutoff=270.
- **< 3pp range**: Feature does not qualify (noise-level, same invariance as time-of-day)

**ABORT (Outcome C) if** all 5 features show < 3pp variation. This is the strongest possible null: neither time-of-day (PR #40) nor volume/volatility predict timeouts. If any feature is in the 3-5pp marginal zone, proceed to Stage 2 with that feature.

**First-100-bars diagnostic:** Report what % of sequential entries (from PR #39 trade log) fall in the first 100 bars of each session (~8 min). If >15% of entries are in the rolling-feature staleness window, exclude `trade_count_20` and `message_rate_20` from gating or use raw features only for early-session bars.

### Stage 2: Sequential Simulation Sweep (~4 min, conditional on Stage 1)

**Only runs if at least one feature shows >= 3pp range in Stage 1.**

For each qualifying feature (>= 3pp range):
1. Sweep 3 gate levels: skip entry if feature < {p25, p50, p75} from training fold
2. Up to (5 features × 3 levels + 1 baseline) × 45 splits = 720 simulations
3. Models trained ONCE per split (same 90 fits as PR #38/#39/#40)
4. At each entry opportunity: check if the entry-time feature >= gate threshold. If not, skip (do not enter). Log as "volume-gated skip" (distinct from hold-skip and time-skip).

**Stacked configuration:** At the optimal single-feature gate, also run with cutoff=270 stacked (time gate + volume gate together). PR #40's cutoff=270 improved via hold-skip restructuring; if volume gating reduces timeouts through a different mechanism, the effects may compound rather than overlap.

### Stage 3: Aggregate and Report

For each configuration (across 45 splits):
1. Compute all primary metrics (same aggregation as PR #39/#40)
2. Four-way comparison table:
   - Unfiltered (baseline, cutoff=390, no gate)
   - Cutoff=270 only (PR #40 best)
   - Volume-gated only (best single-feature gate)
   - Stacked: cutoff=270 + volume-gated

---

## Metrics (ALL must be reported)

### Primary

| # | Metric | Description |
|---|--------|-------------|
| 1 | `diagnostic_table` | 5 features × 4 quartiles: timeout fraction, with pp range per feature |
| 2 | `diagnostic_cross_table` | 4×4 volatility_50 × trade_count cross-table with timeout fractions |
| 3 | `gate_sweep_table` | N configs × {feature, gate_level, trades/day, exp/trade, daily_pnl, dd_worst, dd_median, min_acct_all, min_acct_95, win_rate, gate_skip_%, hold_skip_%, timeout_fraction, calmar, sharpe, annual_pnl} |
| 4 | `comparison_table` | Four-way: unfiltered vs cutoff=270 vs volume-gated vs stacked |
| 5 | `optimal_gate_expectancy` | Per-trade expectancy at the recommended gate |
| 6 | `optimal_gate_min_account_all` | min_account_survive_all at the recommended gate |

### Secondary

| # | Metric | Description |
|---|--------|-------------|
| 7 | `optimal_gate_daily_pnl` | Mean daily PnL at recommended gate |
| 8 | `optimal_gate_calmar` | Calmar ratio at recommended gate |
| 9 | `optimal_gate_sharpe` | Annualized Sharpe at recommended gate |
| 10 | `optimal_gate_trades_per_day` | Mean sequential trades/day at recommended gate |
| 11 | `timeout_fraction_by_config` | Fraction of executed trades that are timeouts, per config |
| 12 | `first_100_bars_entry_pct` | % of sequential entries in first 100 bars per session |
| 13 | `qualifying_features` | List of features with >= 3pp diagnostic range |
| 14 | `daily_pnl_percentiles_optimal` | p5, p25, p50, p75, p95 of daily PnL at recommended gate |

### Sanity Checks

| # | Metric | Expected | Failure meaning |
|---|--------|----------|-----------------|
| SC-S1 | `bar_level_exp_split0` | Within $0.01 of PR #38 split_00 ($1.065) | Training code diverged — ABORT |
| SC-S2 | `baseline_exp` | Within $0.10 of PR #39's $2.50 | Unfiltered simulation diverged — ABORT |
| SC-S3 | `baseline_trades_per_day` | Within 5 of PR #39's 162.2 | Simulation logic changed — ABORT |
| SC-S4 | `baseline_timeout_fraction` | Within 0.5pp of 41.33% | Timeout tracking changed — ABORT |

---

## Baselines

| Baseline | Source | Key Metrics |
|----------|--------|-------------|
| **Sequential unfiltered (PR #39)** | `.kit/results/trade-level-risk-metrics/metrics.json` | Exp $2.50/trade, 162.2 trades/day, $412.77/day, DD worst $47,894, DD median $12,917, min acct $48K/$26.6K, Calmar 2.16, Sharpe 2.10, win rate 49.93%, hold-skip 66.1%, timeout 41.33%, avg bars held 28 |
| **Cutoff=270 (PR #40)** | `.kit/results/timeout-filtered-sequential/metrics.json` | Exp $3.02/trade, 116.8 trades/day, $336.77/day, DD worst $33,984, DD median $8,687, min acct $34K/$25.5K, Calmar 2.49, Sharpe 2.20, timeout 41.30% |
| **Bar-level CPCV (PR #38)** | `.kit/results/cpcv-corrected-costs/metrics.json` | Exp $1.81/trade, PBO 0.067, break-even RT $4.30 |

**Reproduction protocol:** The unfiltered (no gate) simulation must reproduce PR #39's sequential metrics within tolerances (SC-S2, SC-S3, SC-S4). Training is identical. Divergence > tolerance triggers ABORT.

---

## Success Criteria (immutable once RUN begins)

- [ ] **SC-1**: Stage 1 diagnostic produces D1-D5 with pass/fail per feature (all 5 features × 4 quartiles)
- [ ] **SC-2**: Optimal config achieves `seq_expectancy_per_trade >= $3.50` (40% improvement over $2.50)
- [ ] **SC-3**: Optimal config achieves `min_account_survive_all <= $30,000` (37% reduction from $48K)
- [ ] **SC-4**: If Stage 2 runs: all qualifying feature × gate level × 45 splits complete
- [ ] **SC-5**: Timeout fraction reduction >= 5pp (from 41.3% to <= 36.3%)
- [ ] **SC-6**: Four-way comparison table fully populated (unfiltered / cutoff=270 / volume-gated / stacked)
- [ ] **SC-7**: All output files written to `.kit/results/volume-flow-conditioned-entry/`
- [ ] **SC-8**: Bar-level split 0 matches PR #38 within $0.01 (training unchanged)
- [ ] No sanity check (SC-S1 through SC-S4) fails beyond stated tolerances

---

## Minimum Viable Experiment (Stage 1 IS the MVE)

Stage 1 diagnostic IS the MVE. It runs in ~30 seconds with no training and determines whether Stage 2 is warranted.

**MVE pass criteria:**
- All 5 features × 4 quartiles computed without NaN
- D5 cross-table (4×4) populated
- At least one feature has >= 3pp timeout fraction range across quartiles

**MVE ABORT (Outcome C) if:**
- All 5 features show < 3pp variation → timeout fraction is structurally invariant to ALL observable entry-time features
- Any NaN in diagnostic metrics

If MVE passes (at least one feature >= 3pp), proceed to Stage 2 full simulation sweep.

---

## Full Protocol

### Step 0: Load Data and Compute Diagnostics (Stage 1)

1. Load all 45 CPCV test set predictions and bar features from Parquet
2. For each bar, look up `tb_exit_type` to determine timeout vs barrier-hit
3. Compute rolling features: `trade_count_20` = rolling 20-bar mean of `trade_count`, `message_rate_20` = rolling 20-bar mean of `message_rate`
4. For each of 5 features, assign quartile (Q1-Q4) and compute timeout fraction per quartile
5. Compute D5 cross-table: `volatility_50` quartile × `trade_count` quartile timeout fraction
6. Report first-100-bars diagnostic: what % of PR #39 sequential entries fall in first 100 bars

### Step 1: Evaluate Diagnostic and Decide Stage 2

- If ALL 5 features show < 3pp range → ABORT (Outcome C). Write results, exit.
- If any feature shows >= 3pp → list qualifying features, proceed to Step 2.
- If first-100-bars entry % > 15%, exclude rolling features from gating OR use raw features only for early-session bars.

### Step 2: Train and Simulate (Stage 2, conditional)

For each of 45 splits:
1. Train two-stage XGBoost (identical to PR #38/#39/#40) — 90 fits total
2. For baseline (no gate): run sequential simulation
3. For each qualifying feature × {p25, p50, p75}: run sequential simulation with gate
4. For optimal single-feature gate: run with cutoff=270 stacked
5. Record per-split per-config: trades/day, expectancy, win rate, daily PnL, max drawdown, consecutive losses, hold-skip rate, gate-skip rate, timeout_fraction, barrier_hit_fraction, avg_bars_held

### Step 3: Aggregate and Select Optimal Gate

**Optimal gate selection rule:** The recommended gate is the **most conservative (lowest gate threshold)** that achieves BOTH SC-2 (exp >= $3.50) AND SC-3 (min_acct <= $30K). "Most conservative" = least filtering needed.

If NO gate achieves both SC-2 and SC-3: report the gate that **maximizes daily PnL** subject to min_account_survive_all <= $35K (relaxed SC-3). If none meets even the relaxed criterion, report the gate that maximizes Calmar ratio.

### Step 4: Four-Way Comparison

Produce the comparison table:
1. Unfiltered (PR #39 baseline, no gate, cutoff=390)
2. Cutoff=270 only (PR #40 best)
3. Volume-gated only (best single-feature gate from Step 3)
4. Stacked: cutoff=270 + best volume gate

---

## Resource Budget

**Tier:** Quick (CPU-only, local)

- Max GPU-hours: 0
- Max wall-clock time: 15 min
- Max training runs: 90 (identical to PR #38/#39/#40 — 45 splits × 2 stages)
- Stage 1 simulations: 0 (diagnostic only, no training)
- Stage 2 simulations: up to 720 (5 features × 3 levels + 1 baseline) × 45 splits
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

- Stage 1 diagnostic: ~30 seconds (bar-level analysis, no training)
- XGBoost training (if Stage 2): 90 fits on ~1.16M rows ~ 2.6 min
- Sequential simulation (if Stage 2): up to 720 runs at ~0.1s each ~ 72s
- Data loading + aggregation: ~30s
- **Total estimated: ~5 min.** Budget 15 min as 3x safety margin.

---

## Abort Criteria

- **Bar-level mismatch:** Split 0 bar-level expectancy differs from PR #38 by > $0.05. ABORT — training code diverged.
- **Baseline mismatch:** Unfiltered sequential results must match PR #39 within $0.10/trade and 5 trades/day. ABORT if not.
- **Baseline timeout mismatch:** Timeout fraction must be within 0.5pp of 41.33%. ABORT if not.
- **Wall-clock > 15 min:** Expected ~5 min. 15 min = 3x. ABORT.
- **NaN in any metric:** Computation bug. ABORT.
- **Stage 1 abort:** All 5 features < 3pp → Outcome C (not a bug, a valid result).

---

## Confounds to Watch For

1. **Volume-volatility correlation** — gate may just proxy volatility_50 (already in model). D5 cross-table disambiguates: if timeout fraction varies with trade_count WITHIN a volatility_50 quartile, volume provides independent information.

2. **Hold-skip composition effect** — same as PR #40's cutoff=270 improvement. Gating changes which signals get executed, restructuring hold-skip dynamics. Track hold-skip rate per config.

3. **Rolling feature staleness at day start** — first 20 bars have partial windows for rolling features. If >15% of sequential entries fall in the first 100 bars, this corrupts quartile assignments. Diagnostic reports the fraction and either excludes rolling features or falls back to raw per-bar features for early-session entries.

4. **Training-fold quartile leakage** — gate thresholds (p25/p50/p75) must be computed from training fold only, not test fold. Otherwise quartile boundaries leak test distribution.

---

## Decision Rules

```
OUTCOME A — SC-1 through SC-8 all pass:
  -> Volume-flow gating works. Fewer timeouts, richer trades, lower drawdown.
  -> Report recommended gate and its full risk profile.
  -> Next: Paper trade with volume gate (+ optional cutoff=270 stack).

OUTCOME B — SC-1/4/6/7/8 pass BUT SC-2 or SC-3 or SC-5 fail:
  -> Volume gating helps but not enough alone.
  -> If stacked config (cutoff=270 + volume gate) passes all SC → deploy stacked.
  -> If stacked still fails → report best achievable numbers and tradeoff.
  -> Next: Accept best config or change barrier geometry (volume horizon / time horizon).

OUTCOME C — Stage 1 diagnostic aborts (all features < 3pp):
  -> Timeouts are structurally invariant to ALL observable entry-time features.
  -> This is the strongest possible null: neither time-of-day (PR #40) nor
     volume/volatility predict timeouts. The 41.3% rate is a structural constant
     of the volume horizon mechanism.
  -> Next: Change barrier geometry (reduce volume horizon, add time horizon,
     or adjust target/stop ratio) rather than filtering entries.

OUTCOME D — Implementation bug or sanity check abort:
  -> Debug and retry.
```

---

## Deliverables

```
.kit/results/volume-flow-conditioned-entry/
  metrics.json                    # All metrics: diagnostic + gate sweep + comparison
  analysis.md                     # Full analysis: diagnostic results, gate sweep (if any), comparison
  run_experiment.py               # Pipeline with diagnostic + conditional gate sweep
  spec.md                         # Spec copy
  diagnostic_table.csv            # 5 features x 4 quartiles: timeout fraction
  diagnostic_cross_table.csv      # 4x4 volatility_50 x trade_count cross-table
  gate_sweep.csv                  # N configs x all metrics (if Stage 2 runs)
  comparison_table.csv            # Four-way comparison (if Stage 2 runs)
```

### Required Outputs in analysis.md

1. Executive summary (2-3 sentences)
2. **Stage 1 diagnostic results** — D1-D5 tables, per-feature pp range, qualify/marginal/fail classification
3. **First-100-bars diagnostic** — entry fraction in staleness window, action taken
4. **D5 cross-table interpretation** — does volume predict timeouts after controlling for volatility?
5. **Stage 2 results** (if applicable) — gate sweep table, optimal gate selection
6. **Four-way comparison** — unfiltered vs cutoff=270 vs volume-gated vs stacked (if Stage 2 runs)
7. **Explicit SC-1 through SC-8 + SC-S1 through SC-S4 pass/fail**
8. **Outcome verdict (A/B/C/D)**

---

## Exit Criteria

- [ ] Stage 1 diagnostic runs and produces D1-D5 tables
- [ ] Per-feature pp range computed and tier assigned (>= 5pp / 3-5pp / < 3pp)
- [ ] D5 cross-table (volatility_50 × trade_count) populated
- [ ] First-100-bars entry fraction reported
- [ ] Outcome C check performed: all 5 features < 3pp? If yes, abort Stage 2 and report.
- [ ] If Stage 2 warranted: all qualifying features × 3 gate levels × 45 splits simulated
- [ ] If Stage 2 warranted: stacked config (cutoff=270 + best volume gate) simulated
- [ ] Baseline reproduces PR #39 within tolerances (SC-S2, SC-S3, SC-S4)
- [ ] Bar-level split 0 matches PR #38 within $0.01 (SC-S1/SC-8)
- [ ] Four-way comparison table populated (unfiltered / cutoff=270 / volume-gated / stacked)
- [ ] All SC-1 through SC-8 evaluated with explicit pass/fail
- [ ] metrics.json and analysis.md complete
- [ ] All output files written to `.kit/results/volume-flow-conditioned-entry/`
- [ ] Outcome verdict (A/B/C/D) rendered

---

## Key References

- **Sequential pipeline:** `.kit/results/trade-level-risk-metrics/run_experiment.py` — adapt from this
- **Timeout-filtered pipeline:** `.kit/results/timeout-filtered-sequential/run_experiment.py` — time-filter reference
- **Sequential metrics:** `.kit/results/trade-level-risk-metrics/metrics.json` — baseline for comparison
- **Timeout-filtered metrics:** `.kit/results/timeout-filtered-sequential/metrics.json` — cutoff=270 reference
- **CPCV pipeline:** `.kit/results/cpcv-corrected-costs/run_experiment.py` — training reference
- **CPCV metrics:** `.kit/results/cpcv-corrected-costs/metrics.json` — bar-level reference
- **Parquet data:** `.kit/results/label-geometry-1h/geom_19_7/`
- **PnL constants:** tick_value=$1.25, target=19 ticks ($23.75), stop=7 ticks ($8.75), RT=$2.49
- **Key columns:** `tb_label`, `tb_exit_type`, `tb_bars_held`, `fwd_return_1`, `timestamp`, `day`, `minutes_since_open`, `trade_count`, `message_rate`, `volatility_50`
