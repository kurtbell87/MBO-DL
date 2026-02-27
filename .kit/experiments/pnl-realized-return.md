# Experiment: PnL Realized Return — Corrected Hold-Bar Economics

**Date:** 2026-02-26
**Priority:** P0 — resolves dominant uncertainty from 2-class directional experiment
**Parent:** 2-Class Directional (CONFIRMED with PnL caveat, PR #34)
**Depends on:**
1. 2-Class directional results at 19:7 and 10:5 — DONE (PR #34)
2. Label geometry 1h Parquet data (3600s horizon, 152-col) — DONE (PR #33)
3. 2-class `run_experiment.py` — DONE (`.kit/results/2class-directional/run_experiment.py`)

**All prerequisites DONE. Same data, same model architecture — only the PnL computation changes.**

---

## Context

The 2-class directional experiment (PR #34) achieved its structural goal: trade rate liberated from 0.28% to 85.2% at 19:7. However, the reported per-trade expectancy ($3.775) is ~8x inflated because the PnL model assigns full barrier payoffs (+target or -stop) to ALL trades, including the 44.4% that land on hold-labeled bars.

The analysis identified two correction approaches:
- **Conservative ($0 for hold bars):** Estimated corrected expectancy ~$0.44/trade (CI [$0.04, $0.84])
- **Realized return (THIS EXPERIMENT):** Close at time horizon expiry, take actual price movement as PnL

**Why this matters:** The truth lies between two bounds, but NOT necessarily between the conservative ($0.44) and inflated ($3.78) estimates (see first-principles analysis below).

### First-Principles Prior (CRITICAL — read before evaluating hypothesis)

The conservative model assigns $0 PnL to hold-bar trades — they effectively don't count (removed from both numerator and denominator). The realized model DOES count them: each hold-bar trade incurs $3.74 in round-trip costs plus whatever the forward price return × $1.25 per tick.

**Hold-bar forward return bounds depend on barrier race outcomes:**

For bidirectional hold bars (tb_label = 0) at 19:7 geometry, the label means NEITHER the +19 target NOR the -19 target was reached within 3600s. This gives two sub-populations:

| Sub-population | Condition | Forward return bound | Expected magnitude |
|---------------|-----------|---------------------|-------------------|
| **Calm holds** | Both races NOT triggered (tb_long_triggered=0, tb_short_triggered=0) | (-7, +7) ticks | Small — price oscillated in narrow range |
| **Choppy holds** | One or both stops hit, but no target hit | (-19, +19) ticks | Potentially larger — price moved but not to target |

The fraction of calm vs choppy holds is unknown a priori but measurable from the data (tb_long_triggered and tb_short_triggered columns exist in the 152-col schema).

**Cost analysis for calm holds (bounded in (-7, +7)):**
- Max gross payoff per hold-bar trade: ±7 ticks × $1.25 = ±$8.75
- Round-trip cost: $3.74
- Break-even requires mean directional return > $3.74/$1.25 = 3.0 ticks per trade
- With random direction (50% accuracy): E[gross] = 0, E[PnL] = -$3.74 (pure cost drag)
- With perfect direction prediction: E[PnL] = E[|fwd_return|] × $1.25 - $3.74
  - If E[|fwd_return|] ≈ 2-3 ticks (typical for range-bound bars): E[PnL] ≈ -$1.24 to -$0.49 (STILL NEGATIVE even with perfect direction)

**Consequence for total expectancy:**
- Conservative: per-trade exp = $0.44 (N_dir-only denominator)
- Realized: per-trade exp = (PnL_dir + PnL_hold) / N_total
  = $0.44 × (N_dir/N_total) + hold_mean_pnl × (N_hold/N_total)
  = $0.44 × 0.556 + hold_mean_pnl × 0.444
  = $0.245 + hold_mean_pnl × 0.444
- For realized > $0: need hold_mean_pnl > -$0.55
- For realized > conservative ($0.44): need hold_mean_pnl > +$0.44 (requires profitable hold-bar trades)

**Prior expectation:** Realized PnL is LIKELY BELOW the conservative estimate ($0.44) because the conservative model doesn't count hold-bar trades at all, while the realized model counts them with their actual (likely negative) PnL. The central question is: HOW MUCH below? If hold-bar mean PnL ≈ -$1.50, realized ≈ -$0.42 (NEGATIVE). If hold-bar mean PnL ≈ -$0.50, realized ≈ +$0.02 (MARGINAL).

This experiment measures the actual magnitude. The result directly determines whether threshold optimization (reducing hold-bar exposure) can recover viability.

---

## Hypothesis

Under a corrected PnL model that uses actual realized forward returns on hold-bar trades (instead of full barrier payoffs), the two-stage pipeline at 19:7 geometry achieves walk-forward per-trade expectancy > $0.00 after base costs ($3.74 RT).

**Direction:** Positive corrected expectancy.
**Magnitude:** Expected range based on first-principles analysis: [-$1.50, +$0.50]. The hypothesis (> $0) succeeds only if hold-bar mean PnL > -$0.55, which requires either (a) hold-bar forward returns have meaningful magnitude AND (b) the model's directional prediction on hold bars is meaningfully better than random. Both conditions must hold simultaneously.

**This is designed to be a hard-to-pass gate.** The prior is skeptical (~35% probability of SC-1 passing). A pass would be a strong positive signal. A fail definitively closes the 2-class-at-19:7 approach and redirects toward threshold optimization.

---

## Independent Variables

### PnL Model (primary IV — 3 levels for comparison)

| PnL Model | Hold-Bar Treatment | Source |
|-----------|-------------------|--------|
| **Realized Return (NEW)** | Close at horizon: PnL = fwd_return_ticks × $1.25 × sign(prediction) - RT_cost | This experiment |
| **Conservative** | Hold-bar trades = $0 gross (trade doesn't count) | Analytical estimate from PR #34 analysis |
| **Full Barrier (INFLATED)** | Hold-bar trades = full target/stop payoff based on sign match | Original 2-class experiment (buggy) |

### Geometry (2 levels — same as 2-class experiment)

| Geometry | Hold Rate | BEV WR | Payoff Ratio | Role |
|----------|-----------|--------|--------------|------|
| **19:7** | 47.4% | 38.4% | 2.71:1 | Primary |
| **10:5** | 32.6% | 53.3% | 2:1 | Control |

---

## Controls

| Control | Value | Rationale |
|---------|-------|-----------|
| Pipeline architecture | Two-stage (Stage 1: reachability, Stage 2: direction) | Identical to 2-class experiment |
| XGB params | Same tuned params (LR=0.0134, L2=6.586, depth=6, mcw=20, subsample=0.561, colsample=0.748, reg_alpha=0.0014) | No re-tuning |
| Feature set | 20 non-spatial features | Identical |
| Data source | `.kit/results/label-geometry-1h/geom_19_7/` and `geom_10_5/` | Same Parquet |
| Walk-forward folds | 3 expanding-window (same splits) | Direct comparison |
| Stage 1 threshold | 0.5 | Same |
| Seed | 42 | Reproducibility |

---

## Corrected PnL Model (THE KEY CHANGE)

```
tick_value = $1.25 per tick (MES)
tick_size = 0.25 (MES price per tick)

For geometry (target=T, stop=S):

  1. Model predicts hold (pred == 0): $0 (no trade)

  2. Model predicts direction AND true_label != 0 (directional bar):
     Correct (pred == true_label): PnL = +(T × tick_value) - RT_cost
     Wrong (pred != true_label):   PnL = -(S × tick_value) - RT_cost

  3. Model predicts direction AND true_label == 0 (HOLD BAR — the change):
     Compute forward return at time horizon expiry:
       fwd_return_ticks = (close_price[bar_index + horizon_bars] - close_price[bar_index]) / tick_size
       Where horizon_bars = 3600s / 5s = 720 bars (for time_5s bars)

     PnL = fwd_return_ticks × tick_value × sign(prediction) - RT_cost

     BOUNDS NOTE: Hold bars (tb_label=0) only guarantee that neither target
     (+T or -T) was reached during the 3600s window. The forward return at
     the endpoint is bounded by (-T, +T) = (-19, +19) at 19:7. A tighter
     bound of (-S, +S) = (-7, +7) applies ONLY to calm holds where BOTH
     tb_long_triggered=0 AND tb_short_triggered=0 (neither race resolved).
     Choppy holds (one or both stops hit) can have larger forward returns.

     Edge cases:
     - If bar_index + 720 exceeds the trading day: use the LAST available bar of
       the day as the exit price (forced close at session end). This is conservative.
     - If close prices are not available: fall back to $0 for that bar and log a warning.
     - The RUN agent should identify the correct price column from the 152-col
       schema (e.g., close_price, mid_price, or bid/ask midpoint).

Per-trade expectancy = total PnL / number of trades (bars where pred != 0)
```

**Additionally compute and report:**
- Mean absolute realized return (ticks) on hold bars — characterizes typical hold-bar price movement
- Distribution of hold-bar realized returns: p10, p25, median, p75, p90 in ticks
- Fraction of hold-bar trades that are profitable (before costs)
- Fraction of hold-bar trades that are profitable (after costs)
- Hold-bar mean PnL vs directional-bar mean PnL — decompose total expectancy by bar type
- **Stratify by hold sub-population:** calm holds (both_triggered=0) vs choppy holds (any triggered=1)

---

## Baselines

| Baseline | Value | Source |
|----------|-------|--------|
| 2-class WF expectancy (inflated) | $3.775 | 2-class experiment (full barrier payoff — BUGGY) |
| 2-class WF expectancy (conservative) | ~$0.44 | Analytical estimate ($0 for hold bars) |
| 2-class trade rate | 85.18% | 2-class experiment |
| 2-class dir accuracy | 50.05% | 2-class experiment |
| Hold-bar trade fraction | 44.39% | 2-class experiment (label0_hit_rate) |
| First-principles floor | -$1.50 | From prior analysis: if hold_mean_pnl ≈ -$3.74 |
| First-principles ceiling | +$0.50 | From prior analysis: if hold_mean_pnl ≈ +$0.57 |

---

## Metrics (ALL must be reported)

### Primary

1. **realized_wf_expectancy_19_7**: Walk-forward mean per-trade expectancy ($) at 19:7 under base costs using the realized-return PnL model. THE metric.
2. **hold_bar_mean_pnl_19_7**: Mean PnL ($) per hold-bar trade at 19:7 (before and after costs). Decomposes the hold-bar contribution. The single number that determines whether realized > or < conservative.

### Secondary

| Metric | Description |
|--------|-------------|
| hold_bar_mean_fwd_return_ticks_19_7 | Mean realized forward return (ticks) on hold bars where model predicted direction |
| hold_bar_directional_accuracy_19_7 | Fraction of hold-bar trades where sign(prediction) matches sign(fwd_return). **Key diagnostic:** >52% = model has signal on hold bars, ≤50% = no signal |
| realized_wf_expectancy_10_5 | Corrected expectancy at 10:5 (control) |
| directional_bar_mean_pnl | Mean PnL on directional bars only (MUST match 2-class within $0.01) |
| hold_bar_pnl_distribution | p10, p25, median, p75, p90 of hold-bar per-trade PnL |
| hold_bar_fwd_return_distribution | p10, p25, median, p75, p90 of fwd_return_ticks on hold bars |
| hold_bar_win_rate_gross | Fraction of hold-bar trades profitable before costs |
| hold_bar_win_rate_net | Fraction of hold-bar trades profitable after base costs |
| calm_vs_choppy_decomposition | Stratify hold bars by (tb_long_triggered, tb_short_triggered): fwd_return distribution, mean PnL, count for each sub-population |
| pnl_decomposition | Total expectancy = (frac_directional × dir_exp) + (frac_hold × hold_exp) |
| per_fold_realized_expectancy | Per-fold breakdown (3 folds) — check consistency |
| cost_sensitivity_realized | 3 cost scenarios × 2 geometries with realized PnL model |
| comparison_3_pnl_models | Side-by-side: inflated vs conservative ($0) vs realized return |

### Sanity Checks

| Check | Expected | Failure means |
|-------|----------|---------------|
| SC-S1: Directional-bar PnL unchanged | Within $0.01 of 2-class experiment | Bug in PnL code — directional logic changed |
| SC-S2: Trade rate unchanged | Within 0.1pp of 85.18% at 19:7 | Pipeline changed — should be identical |
| SC-S3: Dir accuracy unchanged | Within 0.1pp of 50.05% at 19:7 | Pipeline changed |
| SC-S4: Hold-bar fwd_return bounded | All within (-19, +19) ticks at 19:7 | Forward return computation error or label bug |
| SC-S5: Mean |fwd_return| on hold bars > 0.5 ticks | Yes — price moves at least half a tick in 3600s | Data or computation bug; also validates that the realized-return approach adds information beyond $0 |

---

## Success Criteria (immutable once RUN begins)

- [ ] **SC-1**: Realized-return WF mean per-trade expectancy > $0.00 at 19:7 (base costs $3.74 RT)
- [ ] **SC-2**: Hold-bar directional accuracy > 52% at 19:7 (model predicts direction on hold bars better than random by at least 2pp — necessary condition for hold-bar trades to contribute positively)
- [ ] **SC-3**: Hold-bar mean realized PnL (gross, before costs) is > $0.00 — confirms the model captures directional price movement on hold bars, not just noise
- [ ] **SC-4**: Per-fold realized expectancy is positive in at least 2 of 3 folds — shows robustness

---

## Minimum Viable Experiment

Run a single walk-forward fold (Fold 1: train days 1-100, test days 101-150) at 19:7 geometry only, with all three PnL models. This verifies:

1. Forward return lookup works (720-bar lookahead exists, no NaN)
2. Hold-bar fwd_return is bounded within (-19, +19) ticks (sanity check SC-S4)
3. Directional-bar PnL matches 2-class experiment (sanity check SC-S1)
4. Trade rate and dir accuracy unchanged (SC-S2, SC-S3)
5. Realized expectancy is computable (no NaN, no missing data)

**MVE pass criteria:** SC-S1 (dir PnL within $0.01), SC-S4 (bounded returns), trade rate within 1pp of 85.18%, forward return lookup succeeds for >90% of hold bars. If ANY fail, STOP — do not proceed to full protocol.

**Expected MVE wall-clock:** <30 seconds (2 XGBoost fits + PnL computation).

---

## Full Protocol

### Step 0: Adapt the 2-Class Experiment Script

Start from `.kit/results/2class-directional/run_experiment.py`. The ONLY change is the PnL model for hold-bar trades. Specifically:

1. After computing combined predictions (Stage 1 filter + Stage 2 direction), identify hold-bar trades: bars where `pred != 0` AND `true_label == 0`.
2. For these bars, compute `fwd_return_ticks` by looking 720 bars ahead in the same day's data (close price difference / tick_size). The RUN agent should identify the appropriate price column from the Parquet schema.
3. Compute PnL = fwd_return_ticks × tick_value × sign(prediction) - RT_cost.
4. For directional bars (true_label != 0): keep the existing barrier-payoff PnL (UNCHANGED).
5. Report ALL three PnL models side-by-side for comparison.

### Step 1: Walk-Forward at 19:7 (Primary)

Same 3 expanding-window folds as 2-class experiment:

| Fold | Train Days | Test Days |
|------|-----------|-----------|
| 1 | 1-100 | 101-150 |
| 2 | 1-150 | 151-201 |
| 3 | 1-201 | 202-251 (holdout) |

Per fold: Train Stage 1 + Stage 2 (identical to 2-class), combine predictions, evaluate with ALL THREE PnL models.

### Step 2: Walk-Forward at 10:5 (Control)

Same protocol using 10:5 data.

### Step 3: Hold-Bar Analysis (NEW — the core diagnostic)

Dedicated analysis of hold-bar trades at 19:7:

1. **Forward return distribution:** histogram of fwd_return_ticks on hold bars, p10/p25/median/p75/p90
2. **Stratify by hold sub-population:**
   - Calm holds: tb_long_triggered=0 AND tb_short_triggered=0 (price stayed in (-7, +7))
   - Choppy holds: at least one race triggered (stop hit but no target reached)
   - Report count, mean fwd_return, mean PnL, directional accuracy for each sub-population
3. **Mean, median, std of hold-bar gross PnL** (before and after costs)
4. **Win rate** before and after costs
5. **Hold-bar directional accuracy:** fraction where sign(prediction) matches sign(fwd_return)
   - If >52%: model has some directional signal on hold bars
   - If ≤50%: no signal — hold-bar trades are pure cost + noise
6. **Correlation between Stage 2 P(long) confidence and hold-bar realized return** — does higher model confidence correlate with larger correct returns?

### Step 4: Three-Way PnL Comparison

| PnL Model | 19:7 Expectancy | 19:7 Daily PnL | 10:5 Expectancy |
|-----------|-----------------|----------------|-----------------|
| Full Barrier (inflated) | $3.775 (from 2-class) | $8,241 | -$0.513 |
| **Realized Return (NEW)** | ? | ? | ? |
| Conservative ($0) | ~$0.44 (estimated) | ~$960 | ? |

### Step 5: PnL Decomposition

Break total realized expectancy into components:

```
Total exp = frac_directional × dir_mean_pnl + frac_hold × hold_mean_pnl

Where:
  frac_directional = N_dir / N_total (expected ~0.556 at 19:7)
  frac_hold = N_hold / N_total (expected ~0.444 at 19:7)
  dir_mean_pnl = mean PnL per directional-bar trade
  hold_mean_pnl = mean PnL per hold-bar trade
```

This decomposition reveals whether the total is driven by directional-bar edge, hold-bar drag, or both.

### Step 6: Cost Sensitivity

3 scenarios × 2 geometries with realized-return PnL model:

| Scenario | RT Cost |
|----------|---------|
| Optimistic | $2.49 |
| Base | $3.74 |
| Pessimistic | $6.25 |

Also report: the **break-even RT cost** for realized model (the cost at which realized expectancy = $0). This directly answers "how cheap must execution be for the strategy to work?"

---

## Resource Budget

**Tier:** Quick

- Max wall-clock time: 15 min
- Max training runs: 14 (3 folds × 2 stages × 2 geometries + 2 MVE — same as 2-class)
- Max seeds: 1 (seed=42)
- COMPUTE_TARGET: local
- GPU hours: 0

The 2-class experiment ran in 32 seconds. This experiment has identical training + slightly more PnL computation (720-bar lookahead per hold bar). Expected: <2 minutes for RUN phase.

### Compute Profile
```yaml
compute_type: cpu
estimated_rows: 1160150
model_type: xgboost
sequential_fits: 14
parallelizable: false
memory_gb: 2
gpu_type: none
estimated_wall_hours: 0.03
```

---

## Abort Criteria

- **Data loading failure:** STOP. Run `orchestration-kit/tools/artifact-store hydrate` and retry.
- **Forward return computation fails for >10% of hold bars:** STOP. Investigate bar data availability (missing close price column, insufficient forward bars).
- **Directional-bar PnL differs from 2-class by >$0.10:** STOP. Bug in PnL code — directional logic should be unchanged.
- **Trade rate differs from 2-class by >1pp:** STOP. Pipeline bug — predictions should be identical.
- **Hold-bar fwd_return outside (-19, +19) at 19:7:** STOP. Forward return computation or label semantics are wrong.
- **Wall-clock > 30 min:** ABORT (2× Quick-tier budget). Report partial results.

---

## Decision Rules

```
OUTCOME A — SC-1 AND SC-2 pass (positive realized expectancy + hold-bar directional signal):
  -> CONFIRMED. Two-stage pipeline at 19:7 is economically viable under realistic PnL.
  -> If realized exp > $1.00: STRONG signal. Proceed to CPCV (45 splits) for
     statistical power. High priority.
  -> If $0.00 < realized exp < $1.00: MARGINAL signal. Proceed to CPCV for
     confidence intervals AND threshold optimization (Priority #3 from NEXT_STEPS)
     to improve economics by reducing hold-bar exposure.
  -> Record: realized exp, hold-bar decomposition, break-even cost.

OUTCOME B — SC-1 fails but SC-2 passes (hold-bar dir accuracy > 52% but negative total exp):
  -> PARTIAL. Model has directional signal on hold bars but costs dominate.
  -> Threshold optimization is the highest-priority next step: raising Stage 1
     threshold from 0.5 to 0.7+ reduces hold-bar trade fraction, concentrating
     on higher-precision directional predictions.
  -> Also viable: intermediate geometry (14:6, 15:5) where the payoff-to-cost
     ratio may be more favorable.

OUTCOME C — SC-2 fails (hold-bar dir accuracy ≤ 52%):
  -> REFUTED. Model has zero directional signal on hold bars.
  -> Hold-bar trades are pure cost drag (~-$3.74 each).
  -> Realized expectancy is definitively below conservative estimate.
  -> Next: Stage 1 threshold optimization (MUST-DO — reduce hold-bar exposure
     to <10% of trades). At threshold 0.8+, nearly all trades hit directional
     bars, and the conservative PnL estimate ($0.44) becomes the realized estimate.
  -> This is the MOST LIKELY outcome (prior ~55%). A useful outcome because it
     definitively quantifies the hold-bar drag and motivates threshold work.

OUTCOME D — Sanity checks fail (directional-bar PnL or trade rate mismatch):
  -> INVALID. Pipeline bug. Debug and re-run.
```

---

## Confounds to Watch For

1. **End-of-day truncation.** Bars near session close may not have 720 bars of forward data. Using last-available-bar exit is conservative but may systematically understate hold-bar returns (forced early exit limits both upside and downside). Log the fraction of hold bars with truncated forward lookups and their mean fwd_return separately.

2. **Time-bar alignment.** 720 bars × 5s = 3600s is exact for time_5s bars, but only if bars are perfectly aligned. Minor concern for regular 5s bars.

3. **Hold-bar direction accuracy is structurally disadvantaged.** Hold bars are bars where neither target was reached — price oscillated or drifted without conviction. Direction prediction on these low-information bars is expected to be harder than on directional bars. The model's ~50% overall direction accuracy is measured on ALL bars; on hold bars specifically, accuracy may be systematically worse. **If hold-bar directional accuracy is <50%, the model's direction predictions on hold bars are ANTI-correlated with realized returns — the model predicts the opposite of what the price does on range-bound bars (possible momentum-on-hold-bars bias, since the model trained on directional bars where momentum is informative).**

4. **Calm vs choppy hold distinction matters.** Calm holds (price in (-7, +7)) have small, likely symmetric forward returns. Choppy holds (a stop was hit) have larger returns with directional bias (the stop hit reveals direction information). The overall hold-bar statistics may mask very different dynamics in these sub-populations. Report stratified results.

5. **Survivorship in forward return.** For bars near the end of the dataset (last 720 bars of day 251), forward returns may be truncated. These should be a small fraction of total bars (~1-2% of daily bars at the boundary) and should be logged and excluded from hold-bar analysis.

6. **Conservative model denominator difference.** The conservative model's per-trade expectancy uses N_dir as denominator (hold-bar trades don't count), while the realized model uses N_total. This means realized < conservative is possible even with hold-bar mean PnL = $0 (realized = $0.44 × 0.556 / 1.0 = $0.245). The comparison should normalize: report BOTH per-trade (different denominators) AND total PnL (same basis) for apples-to-apples.

7. **PnL model mismatch with trading reality.** The realized-return model assumes you hold for the full 3600s horizon regardless of what happens. In practice, a trader using the triple barrier would exit when a barrier is hit. The hold-bar realized return represents a different strategy (hold-to-horizon) than the directional-bar PnL (barrier-exit). This mixed-strategy PnL model is a simplification. Acknowledge this in the analysis.

---

## Deliverables

```
.kit/results/pnl-realized-return/
  metrics.json                     # All SC statuses + corrected metrics
  analysis.md                      # Three-way PnL comparison, hold-bar analysis, verdict
  run_experiment.py                # Adapted from 2-class with corrected PnL model
  spec.md                          # Local copy of spec
  walkforward_results.csv          # Per-fold × per-geometry: all 3 PnL models
  hold_bar_analysis.csv            # Hold-bar return distribution, win rates, decomposition
  cost_sensitivity.csv             # 3 cost scenarios × 2 geometries (realized PnL)
```

### Required Outputs in analysis.md

1. Executive summary (2-3 sentences)
2. **Three-way PnL comparison table** (inflated vs realized vs conservative × 2 geometries) — the headline result
3. **Hold-bar analysis** — mean realized return, distribution, directional accuracy, win rate
4. **Calm vs choppy hold stratification** — sub-population counts, mean fwd_return, mean PnL
5. **PnL decomposition** — directional-bar contribution vs hold-bar contribution to total expectancy
6. **Per-fold consistency** (3 folds × 2 geometries × 3 PnL models)
7. **Cost sensitivity** (3 scenarios × 2 geometries with realized PnL) + break-even RT cost
8. **Explicit SC-1 through SC-4 pass/fail**
9. **Outcome verdict (A/B/C/D)**
10. **Implication for threshold optimization:** if OUTCOME C, compute estimated expectancy at Stage 1 thresholds [0.6, 0.7, 0.8] by extrapolating from the PnL decomposition

---

## Exit Criteria

- [ ] Walk-forward completed for 19:7 (primary) — 3 folds × 2 stages, all 3 PnL models
- [ ] Walk-forward completed for 10:5 (control) — 3 folds × 2 stages, all 3 PnL models
- [ ] Hold-bar analysis: fwd_return distribution, mean PnL, win rates, directional accuracy
- [ ] Calm vs choppy hold stratification reported
- [ ] Three-way PnL comparison table (inflated vs realized vs conservative)
- [ ] PnL decomposition (directional-bar vs hold-bar contribution)
- [ ] Cost sensitivity computed (3 scenarios × 2 geometries, realized PnL model)
- [ ] Break-even RT cost reported
- [ ] Sanity checks: directional-bar PnL, trade rate, dir accuracy unchanged from 2-class
- [ ] All metrics reported in metrics.json
- [ ] analysis.md written with comparison tables and verdict
- [ ] SC-1 through SC-4 explicitly evaluated

---

## Key References

- **2-Class experiment script:** `.kit/results/2class-directional/run_experiment.py` — starting point, adapt PnL model
- **2-Class results:** `.kit/results/2class-directional/metrics.json` — baseline metrics for sanity checks
- **2-Class analysis:** `.kit/results/2class-directional/analysis.md` — PnL model critique (§PnL Model Validity), hold-bar fraction, per-fold decomposition
- **Parquet data:** `.kit/results/label-geometry-1h/geom_19_7/` and `geom_10_5/` — 152-col schema with tb_long_triggered, tb_short_triggered columns
- **Tuned XGB params:** LR=0.0134, L2=6.586, depth=6, mcw=20, subsample=0.561, colsample=0.748, reg_alpha=0.0014
- **PnL constants:** tick_value=$1.25, tick_size=0.25 (MES). RT costs: optimistic $2.49, base $3.74, pessimistic $6.25
