# Experiment: Barrier Geometry Exploration

**Date:** 2026-02-27
**Priority:** P1 — structurally change payoff to compress drawdowns
**Parent:** Timeout-Filtered Sequential Execution (CONFIRMED modified-A, PR #40)
**Depends on:**
1. Timeout-filtered sequential pipeline and results — DONE (PR #40, cutoff=270)
2. CPCV corrected-costs pipeline — DONE (PR #38)
3. Label geometry 1h Parquet data — DONE (PR #33, has geom_19_7 baseline + geom_10_5, geom_15_3, geom_20_3)
4. C++ `bar_feature_export` with `--target`/`--stop`/`--max-time-horizon`/`--volume-horizon` CLI flags — DONE
5. Daily stop loss experiment — RUN FIRST (Experiment A)

**All prerequisites DONE. Re-exports labels with alternative TP/SL geometries and volume horizons, then retrains the full two-stage CPCV pipeline. Changes the fundamental payoff structure.**

---

## Research Question

Does an alternative barrier geometry (TP/SL tick distances) or volume horizon improve risk-adjusted performance over the 19:7 cutoff=270 baseline?

---

## Hypothesis

Re-exporting labels with narrower TP/SL distances (lower absolute risk per trade) or different volume horizons, then retraining the two-stage XGBoost pipeline, can achieve min_account_survive_all <= $15K with annual PnL >= $40K and Calmar >= 2.5 — a structural improvement over the 19:7 baseline ($34K min account).

**Direction:** Lower per-trade risk (narrower barriers) → lower drawdowns at the cost of lower per-trade profit.
**Magnitude:** $15K is a 56% reduction from $34K. $40K annual PnL is ~63% of baseline ~$63K.
**Rationale:** The 19:7 geometry has WIN_PNL=$23.75, LOSS_PNL=$8.75, breakeven win rate 34.6%. A narrower geometry like 10:4 has WIN_PNL=$12.50, LOSS_PNL=$5.00, breakeven 33.3% — similar edge ratio but lower absolute P&L per trade. If the model's accuracy is geometry-independent (same features predict direction regardless of barrier width), narrower barriers reduce per-trade variance, which directly compresses drawdowns.

---

## Independent Variables

### TP/SL Geometry (primary IV — 6 levels including baseline)

| Geometry | Target (ticks) | Stop (ticks) | WIN_PNL | LOSS_PNL | Breakeven WR | Reward:Risk |
|----------|---------------|-------------|---------|----------|-------------|-------------|
| **19:7 (baseline)** | 19 | 7 | $23.75 | $8.75 | 26.9% | 2.71:1 |
| 15:5 | 15 | 5 | $18.75 | $6.25 | 25.0% | 3.00:1 |
| 15:7 | 15 | 7 | $18.75 | $8.75 | 31.8% | 2.14:1 |
| 12:5 | 12 | 5 | $15.00 | $6.25 | 29.4% | 2.40:1 |
| 10:4 | 10 | 4 | $12.50 | $5.00 | 28.6% | 2.50:1 |
| 10:7 | 10 | 7 | $12.50 | $8.75 | 41.2% | 1.43:1 |

5 new geometries + baseline 19:7 = 6 total.

### Volume Horizon (secondary IV — 5 levels including baseline)

| Volume Horizon | Contracts | Description |
|---------------|-----------|-------------|
| **50K (baseline)** | 50,000 | Current default (effectively unlimited for MES) |
| 25K | 25,000 | Tighter volume cap |
| 35K | 35,000 | Moderate-tight |
| 75K | 75,000 | Moderate-loose |
| 100K | 100,000 | Very loose |

4 new volume horizons + baseline 50K = 5 total. All at 19:7 geometry.

### Cutoff Sweep (coarse, per geometry — 3 levels)

| Cutoff | Minutes Remaining | Description |
|--------|------------------|-------------|
| 390 | 0 | No filter |
| 330 | 60 | Moderate |
| 270 | 120 | Aggressive (baseline optimal) |

Narrower targets resolve faster → may tolerate aggressive cutoffs better.

### Pipeline configuration (FIXED)

Two-stage XGBoost (NOT 3-class — the 3-class pipeline collapses at non-10:5 geometries). Same CPCV splits, same features, same training. Only the label data and PnL constants change per geometry.

---

## Controls

| Control | Value | Rationale |
|---------|-------|-----------|
| Data source | `.kit/results/label-geometry-1h/geom_19_7/` for baseline; new exports for other geometries | Same raw MBO data |
| CPCV protocol | N=10, k=2, 45 splits, purge=500, embargo=4600 | Identical |
| Training | Same two-stage XGBoost with early stopping | Identical model architecture |
| Seed | 42 (per-split: 42 + split_idx) | Identical |
| RT cost | $2.49 (corrected-base) | Same cost model |
| Sequential execution | Same protocol as PR #39/#40 | Same simulation logic |
| C++ binary | `./build/bar_feature_export` | Same feature extraction |
| Max time horizon | 3600 seconds (1 hour) | Same as label-geometry-1h |
| Feature set | 20 non-spatial features | Identical |

---

## Metrics (ALL must be reported)

### Primary

| # | Metric | Description |
|---|--------|-------------|
| 1 | `tpsl_sweep_table` | 6-row table (one per TP/SL geometry) at best cutoff: {geometry, cutoff, trades/day, exp/trade, daily_pnl, dd_worst, dd_median, min_acct_all, min_acct_95, calmar, sharpe, annual_pnl, win_rate, pbo, hold_rate, timeout_fraction} |
| 2 | `volume_sweep_table` | 5-row table (one per volume horizon) at best cutoff with 19:7 geometry: same columns as above |
| 3 | `combined_sweep_table` | 2-3 rows: best TP/SL geometry + best volume horizon combinations |

### Secondary

| # | Metric | Description |
|---|--------|-------------|
| 4 | `pbo_comparison` | PBO (fraction of 45 splits with negative expectancy) for each configuration |
| 5 | `breakeven_analysis` | Per-geometry: observed win rate vs breakeven win rate, edge margin |
| 6 | `hold_rate_by_geometry` | Fraction of bars labeled hold (tb_label=0) per geometry — must be < 80% |
| 7 | `resolution_speed` | Per-geometry: median bars_held, p95 bars_held, timeout fraction — narrower barriers should resolve faster |
| 8 | `geometry_cutoff_interaction` | Full 6x3 grid of geometry x cutoff results |
| 9 | `best_config_equity_curves` | Per-split equity curves at the best overall configuration |
| 10 | `best_config_account_sizing` | Account sizing at best configuration ($500-$50K in $500 steps) |

### Sanity Checks

| # | Metric | Expected | Failure meaning |
|---|--------|----------|-----------------|
| SC-S1 | `baseline_19_7_exp` | $3.02 +/- $0.10 at cutoff=270 | Baseline must reproduce PR #40 — ABORT |
| SC-S2 | `baseline_19_7_trades_per_day` | 116.8 +/- 5 at cutoff=270 | Baseline must reproduce PR #40 — ABORT |
| SC-S3 | `baseline_19_7_min_acct_all` | $34K +/- $1K at cutoff=270 | Baseline must reproduce PR #40 — ABORT |
| SC-S4 | `parquet_column_count` | 152 for all new exports | Export schema must match baseline — ABORT |
| SC-S5 | `hold_rate_below_80pct` | < 80% for all geometries | If hold rate >= 80%, geometry is degenerate — SKIP that geometry |

---

## Baselines

| Baseline | Source | Key Metrics |
|----------|--------|-------------|
| **19:7 cutoff=270 (PR #40)** | `.kit/results/timeout-filtered-sequential/metrics.json` | Exp $3.02/trade, 116.8 trades/day, Sharpe 2.27, Calmar 2.49, min_acct_all $34K |
| **Existing geometry exports** | `.kit/results/label-geometry-1h/` | `geom_10_5`, `geom_15_3`, `geom_19_7`, `geom_20_3` at volume_horizon=50K |

**Reusable exports:** `geom_10_5` (target=10, stop=5) is already exported. It can be reused directly — no re-export needed. `geom_15_3` and `geom_20_3` are not in the sweep (different stop distances) so they are not reused.

---

## Success Criteria (immutable once RUN begins)

- [ ] **SC-1**: Min account (all) <= $15,000 at best configuration
- [ ] **SC-2**: Annual PnL >= $40,000 at best configuration
- [ ] **SC-3**: Calmar >= 2.5 at best configuration
- [ ] **SC-4**: PBO <= 0.10 at best configuration
- [ ] **SC-5**: All 5 new TP/SL geometries exported (251 days each) and validated
- [ ] **SC-6**: All 4 new volume horizons exported (251 days each) and validated
- [ ] **SC-7**: Full CPCV pipeline (45 splits x 2 stages x 3 cutoffs) run for each configuration
- [ ] **SC-8**: Comparative analysis across all configurations
- [ ] **SC-9**: All output files written to `.kit/results/barrier-geometry-exploration/`
- [ ] **SC-10**: 19:7 @ 50K baseline reproduces PR #40 metrics

---

## Full Protocol

### Phase 1: Label Re-Export

For each new geometry, run C++ `bar_feature_export` on all 251 trading days:

```bash
./build/bar_feature_export --target T --stop S --max-time-horizon 3600 --volume-horizon V <input.dbn.zst> <output.parquet>
```

**TP/SL exports (5 new geometries x 251 days = 1,255 exports):**
- geom_15_5: `--target 15 --stop 5 --volume-horizon 50000`
- geom_15_7: `--target 15 --stop 7 --volume-horizon 50000`
- geom_12_5: `--target 12 --stop 5 --volume-horizon 50000`
- geom_10_4: `--target 10 --stop 4 --volume-horizon 50000`
- geom_10_7: `--target 10 --stop 7 --volume-horizon 50000`

**Volume horizon exports (4 new configs x 251 days = 1,004 exports):**
- geom_19_7_vol25k: `--target 19 --stop 7 --volume-horizon 25000`
- geom_19_7_vol35k: `--target 19 --stop 7 --volume-horizon 35000`
- geom_19_7_vol75k: `--target 19 --stop 7 --volume-horizon 75000`
- geom_19_7_vol100k: `--target 19 --stop 7 --volume-horizon 100000`

**Note:** `geom_10_5` already exists in `.kit/results/label-geometry-1h/geom_10_5/` — reuse if it has 152 columns and was exported with `--max-time-horizon 3600`.

**Parallelism:** Use ThreadPoolExecutor with 8 workers (pattern from `.kit/results/label-geometry-1h/run_experiment.py`).

**Validation per export:**
- 152 columns (bidirectional schema)
- Hold rate (tb_label=0) < 80%
- Row count 3,000-6,000 per day (typical range)
- No NaN in feature columns

**Estimated time:** ~20 min for TP/SL exports, ~15 min for volume horizons (with 8-way parallelism).

### Phase 2: CPCV Training + Simulation

Per geometry configuration:
1. Load Parquet data from the geometry-specific directory
2. Update WIN_PNL and LOSS_PNL per geometry:
   - `WIN_PNL = target_ticks * $1.25`
   - `LOSS_PNL = stop_ticks * $1.25`
3. Run 45 CPCV splits x 2 stages = 90 model fits
4. Run 45 splits x 3 cutoffs = 135 sequential simulations
5. Compute PBO = fraction of 45 splits with negative expectancy

**Estimated time:** ~3.5 min per geometry, ~32 min for 9 new configs.

### Phase 3: Combined Configurations

Take the best TP/SL geometry and best volume horizon. If they're different from baseline, test 2-3 combined configs:
- Best TP/SL + best volume horizon
- Best TP/SL + baseline volume horizon
- Baseline TP/SL + best volume horizon

### Phase 4: Comparative Analysis

All configs vs baseline. Rank by Calmar ratio, min_account, annual PnL, PBO.

---

## Resource Budget

**Tier:** Medium

- Max GPU-hours: 0
- Max wall-clock time: 4 hours (safety margin for ~82 min estimated)
- Max training runs: ~900 (10 configs x 90 fits/config)
- Simulations: ~1,350 (10 configs x 135 sims/config)
- Exports: ~2,259 (9 new configs x 251 days)
- Max seeds per configuration: 1

### Compute Profile
```yaml
compute_type: cpu
estimated_rows: 1160150
model_type: xgboost
sequential_fits: 900
parallelizable: true
memory_gb: 4
gpu_type: none
estimated_wall_hours: 1.37
```

### Wall-Time Estimation

- Label re-export: ~35 min (2,259 exports, 8-way parallelism, ~7.5s/export)
- XGBoost training: ~29 min (900 fits on ~1.16M rows, ~2 sec/fit)
- Sequential simulation: ~2.2 min (1,350 runs at ~0.1s each)
- Data loading + aggregation: ~5 min
- Phase 3 combined: ~10 min
- **Total estimated: ~82 min.** Budget 4 hours.

---

## Abort Criteria

- **Baseline mismatch:** 19:7 @ 50K cutoff=270 must match PR #40 within tolerances. ABORT if not.
- **Export validation failure:** If any geometry produces 0 rows, NaN features, or column count != 152. SKIP that geometry (do not abort entire experiment).
- **Wall-clock > 4 hours:** Something wrong with export or training. ABORT.
- **All geometries hold rate >= 80%:** Export parameters wrong. ABORT.
- **NaN in any metric:** Computation bug. ABORT.

---

## Confounds to Watch For

1. **Model accuracy IS geometry-dependent.** The XGBoost model is retrained per geometry — different labels → different decision boundaries → different accuracy. A geometry with lower absolute P&L per trade but higher accuracy could outperform.

2. **Narrower barriers resolve faster.** 10:4 (10 ticks target, 4 ticks stop) should resolve in fewer bars than 19:7. This means more trades per day, which could amplify or dampen returns depending on accuracy.

3. **Volume horizon affects hold rate.** Tighter volume horizons cap the barrier race duration, increasing timeout rate. If the model is accurate directionally, more timeouts dilute per-trade expectancy (same issue as PR #39's baseline).

4. **PnL model must be geometry-specific.** Each geometry has different WIN_PNL/LOSS_PNL. The breakeven win rates range from 25.0% (15:5) to 41.2% (10:7). A geometry-naive analysis would be wrong.

5. **Splits 18 & 32 sensitivity.** These outlier paths (temporal group 4) may be structurally unfixable — geometry changes cannot fix temporal regime problems. Report min_acct_95 alongside min_acct_all.

6. **Existing 10:5 export may need re-validation.** Check if it was exported with `--max-time-horizon 3600` and bidirectional labels (152 columns). If not, re-export.

---

## Decision Rules

```
OUTCOME A — SC-1 through SC-10 all pass:
  -> Alternative geometry is structurally superior to 19:7.
  -> Report best configuration and full risk profile.
  -> Next: Stack with daily stop loss from Experiment A for further compression.

OUTCOME B — Some SC pass (partial improvement):
  -> If SC-1 or SC-3 pass but others fail: geometry helps some metrics but not all.
     -> Report the Pareto frontier of configurations.
  -> If PBO > 0.10: geometry overfits differently. Model edge is geometry-specific.
  -> Report best achievable numbers and tradeoff curves.
  -> Next: Consider stacking best geometry + daily stop loss.

OUTCOME C — No geometry meaningfully improves over 19:7:
  -> The 19:7 geometry is already near-optimal for this model and data.
  -> The drawdown is structural (temporal regime failure), not payoff-dependent.
  -> Next: Accept $34K account requirement or explore regime-conditional trading.

OUTCOME D — Export or training failure:
  -> Implementation bug. Debug and retry.
```

---

## Deliverables

```
.kit/results/barrier-geometry-exploration/
  run_experiment.py               # Full pipeline: export + train + simulate + analyze
  metrics.json                    # All metrics across all configurations
  analysis.md                     # Full comparative analysis
  spec.md                         # Spec copy
  geom_15_5/                      # Parquet exports (251 days each)
  geom_15_7/
  geom_12_5/
  geom_10_4/
  geom_10_7/
  geom_19_7_vol25k/
  geom_19_7_vol35k/
  geom_19_7_vol75k/
  geom_19_7_vol100k/
  tpsl_sweep_table.csv            # 6 rows (geometries) x metrics at best cutoff
  volume_sweep_table.csv          # 5 rows (volume horizons) x metrics at best cutoff
  combined_sweep_table.csv        # 2-3 rows: combined best configs
  pbo_comparison.csv              # PBO per configuration
  geometry_cutoff_grid.csv        # Full 6x3 grid of geometry x cutoff
  account_sizing_best.csv         # Account sizing at best config
  equity_curves_best.csv          # Per-split equity curves at best config
```

### Required Outputs in analysis.md

1. Executive summary (2-3 sentences)
2. **TP/SL geometry sweep results** — 6-row table with all metrics per geometry
3. **Volume horizon sweep results** — 5-row table with all metrics per volume horizon
4. **Combined configuration results** — best TP/SL + best volume horizon
5. **PBO analysis** — per-configuration PBO, edge stability across splits
6. **Resolution speed analysis** — bars_held, timeout fraction per geometry
7. **Hold rate analysis** — label distribution per geometry
8. **Best configuration selection** — which config, why
9. **Baseline comparison** — best config vs 19:7 cutoff=270 side-by-side
10. **Account sizing at best config** — survival curve, min account all/95%
11. **Outlier path analysis (splits 18 & 32)** — are they fixable by geometry?
12. **Explicit SC-1 through SC-10 + SC-S1 through SC-S5 pass/fail**
13. **Outcome verdict (A/B/C/D)**

---

## Exit Criteria

- [ ] All 5 new TP/SL geometries exported and validated (1,255 exports)
- [ ] All 4 new volume horizons exported and validated (1,004 exports)
- [ ] 19:7 @ 50K baseline reproduces PR #40 metrics
- [ ] Full CPCV pipeline run for each new configuration (45 splits x 2 stages x 3 cutoffs)
- [ ] TP/SL sweep table populated (6 rows x 16+ columns)
- [ ] Volume horizon sweep table populated (5 rows x 16+ columns)
- [ ] Combined sweep table populated (2-3 rows)
- [ ] PBO computed per configuration
- [ ] Best configuration identified with rationale
- [ ] Account sizing curve at best configuration ($500-$50K)
- [ ] Equity curves at best configuration
- [ ] Comparative analysis complete (all configs vs baseline)
- [ ] All output files written to `.kit/results/barrier-geometry-exploration/`
- [ ] metrics.json and analysis.md complete

---

## Key References

- **Timeout-filtered sequential pipeline:** `.kit/results/timeout-filtered-sequential/run_experiment.py`
- **Label export pipeline (fork for Phase 1):** `.kit/results/label-geometry-1h/run_experiment.py`
- **C++ binary:** `./build/bar_feature_export` (flags: --target, --stop, --max-time-horizon, --volume-horizon)
- **Existing 19:7 Parquet:** `.kit/results/label-geometry-1h/geom_19_7/`
- **Existing 10:5 Parquet:** `.kit/results/label-geometry-1h/geom_10_5/` (verify 152 columns + 1h horizon)
- **Raw MBO data:** `DATA/GLBX-20260207-L953CAPU5B/` (312 daily .dbn.zst files, ~251 RTH days)
- **PnL constants per geometry:** WIN_PNL = target_ticks x $1.25, LOSS_PNL = stop_ticks x $1.25, RT = $2.49
