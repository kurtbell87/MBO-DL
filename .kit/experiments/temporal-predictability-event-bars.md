# Phase R4b: Temporal Predictability on Event-Driven Bars [Research]

**Spec**: TRAJECTORY.md §2.4 (entropy rate theory), §2.1 (subordination)
**Depends on**: Phase R1 (AR R² per bar type), Phase R4 (temporal predictability on time_5s), Phase 4 (feature pipeline), TDD sub-cycle for `bar_feature_export` tool.
**Unlocks**: Revised R6 synthesis (if temporal signal found) or higher confidence in current architecture (if no signal).
**GPU budget**: 0 hours (CPU only). **Max runs**: 1 per bar type (deterministic).

---

## Motivation

R4 concluded "NO TEMPORAL SIGNAL" on `time_5s` bars and recommended dropping the SSM/temporal encoder. However, R4 only tested one bar type. R1's own data shows that **dollar bars have significantly higher AR R²** than time bars:

| Bar Type | AR R² (R1) | Relative to time_5s |
|----------|-----------|---------------------|
| dollar_25k | 0.01131 | 33× higher |
| dollar_50k | 0.00967 | 28× higher |
| dollar_100k | 0.00713 | 21× higher |
| time_5s | 0.00034 | baseline |
| volume_100 | 0.00047 | 1.4× higher |

R1 noted this as evidence against subordination theory (which predicts more IID returns from event bars) but never explored its implications for temporal architecture. The original TRAJECTORY.md §R4 spec called for testing "for each bar type" — the executed R4 deviated by only running `time_5s`.

Finding no temporal structure in uniformly-sampled clock-time returns from a non-uniform process is expected, not informative. TRAJECTORY.md §3 states the core prior: price changes are caused by orders, not time.

This experiment closes the gap by re-running the full R4 protocol on event-driven bars.

---

## Hypothesis

**H1**: Dollar bars (and possibly volume bars) contain exploitable autoregressive structure that is absent from time bars.

**H2**: Temporal feature augmentation of static features improves prediction on event-driven bars, even though it fails on time_5s.

**Null**: The R4 "NO TEMPORAL SIGNAL" conclusion generalizes across bar types. MES returns are martingale regardless of sampling scheme.

---

## Bar Types

| Bar Type | BarFactory Args | Expected Bars (19 days) | R1 AR R² | Priority |
|----------|----------------|------------------------|----------|----------|
| `dollar_25k` | `("dollar", 25000.0)` | ~3.1M (~165K/day) | 0.01131 | **Primary** — strongest prior for temporal signal |
| `volume_100` | `("volume", 100.0)` | ~116K (~6K/day) | 0.00047 | **Secondary** — fast negative confirmation expected |

Reference: `time_5s` (R4 baseline) had 87,970 bars, AR R² = 0.00034.

---

## Data

### Feature Export (Prerequisite)

No feature CSVs exist for event-driven bars. A new C++ tool `bar_feature_export` must be built (TDD sub-cycle, spec: `.kit/docs/bar-feature-export.md`).

**Pipeline**: StreamingBookBuilder → BarFactory → BarFeatureComputer → FeatureExporter CSV
**Days**: Same 19 `SELECTED_DAYS` as R2's `info_decomposition_export.cpp` (lines 261-264):
```
20220103, 20220121, 20220211, 20220304, 20220331, 20220401, 20220422,
20220513, 20220603, 20220630, 20220701, 20220722, 20220812, 20220902,
20220930, 20221003, 20221024, 20221114, 20221205
```
**CSV schema**: Identical to `info-decomposition/features.csv` — 5 metadata + 62 Track A + 40 book_snap + 33 msg_summary + 4 forward returns.

### Warmup

Discard `is_warmup = true` bars (first 50 per day). Additionally discard the first `L` bars per day for configs requiring lookback of depth `L` (no cross-day lookback).

---

## Horizon Mismatch Caveat

R4 horizons {1, 5, 20, 100} are in **bars**, not clock time. Average bar durations differ drastically:

| Bar Type | Avg Duration | h=1 | h=5 | h=20 | h=100 |
|----------|-------------|-----|-----|------|-------|
| time_5s | 5s | 5s | 25s | 100s | 500s |
| volume_100 | ~3.9s | ~4s | ~20s | ~78s | ~390s |
| dollar_25k | ~0.14s | ~0.14s | ~0.7s | ~2.8s | ~14s |

Dollar_25k h=100 (~14s) ≈ time_5s h=3. Direct R² comparisons at the same bar-horizon are not apples-to-apples in clock time. The comparison analysis must document this.

**Executability**: Dollar_25k bars fire every ~140ms. Finding temporal signal at this timescale doesn't mean it's tradeable with MES retail infrastructure. Any temporal signal found in dollar_25k operates in genuine microstructure territory (quote revisions, inventory management, HFT patterns). This informs the downstream decision fork (see Decision Framework).

---

## Protocol

Identical to R4 (`.kit/experiments/temporal-predictability.md`). Replicated here for self-containment.

### Tier 1: Pure Return Autoregression (36 configs per bar type)

| Lookback | Input Dim | Models | Horizons |
|----------|-----------|--------|----------|
| AR-10 | 10 | Linear, Ridge, GBT | h ∈ {1, 5, 20, 100} |
| AR-50 | 50 | Linear, Ridge, GBT | h ∈ {1, 5, 20, 100} |
| AR-100 | 100 | Linear, Ridge, GBT | h ∈ {1, 5, 20, 100} |

**GBT config**: `XGBRegressor(max_depth=4, n_estimators=200, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=20, random_state=42)`.

### Tier 2: Temporal Feature Augmentation (24 configs per bar type)

Same 5 feature configs, 21 temporal features, GBT model — see R4 spec for details.

### Temporal Features (21 dimensions)

`lag_return_{1..10}`, `rolling_vol_{5,20,100}`, `vol_ratio`, `momentum_{5,20,100}`, `mean_reversion_20`, `abs_return_lag1`, `signed_vol`.

Constructed per-day from `return_1` — no cross-day lookback, no lookahead.

### Cross-Validation

5-fold expanding-window time-series CV (identical to R4):

| Fold | Train days | Test days |
|------|-----------|-----------|
| 1 | days 1–4 | days 5–8 |
| 2 | days 1–8 | days 9–11 |
| 3 | days 1–11 | days 12–14 |
| 4 | days 1–14 | days 15–17 |
| 5 | days 1–17 | days 18–19 |

Z-score normalization using training fold statistics only.

### Statistical Framework

- Paired t-test (or Wilcoxon if Shapiro-Wilk p < 0.05) on 5-fold R² differences.
- Holm-Bonferroni within each gap family.
- **Dual threshold**: Δ > 20% of baseline R² AND corrected p < 0.05.
- Report: point estimate, 95% CI, raw p, corrected p.

---

## Decision Framework

### Decision Rules (per bar type)

Same 4 rules as R4. See `.kit/experiments/temporal-predictability.md` for full decision rule text.

- **Rule 1**: Any AR GBT R² > 0 at h > 1 with corrected p < 0.05?
- **Rule 2**: Δ_temporal_book passes dual threshold?
- **Rule 3**: Temporal-Only R² > 0 at corrected p < 0.05?
- **Rule 4**: Reconciliation with R4 time_5s results.

### Cross-Bar-Type Decision

- **Both bar types fail all rules** → R4 conclusion is robust across bar types. Proceed with CNN+GBT, no temporal encoder, with higher confidence. Gap in experimental chain is closed.

- **Dollar_25k passes Rule 2 (temporal augmentation helps)** → Temporal signal exists in event time but not clock time. Two downstream paths:

  **Path A — Full pipeline switch to dollar bars**:
  Re-run oracle expectancy on dollar_25k, re-validate features, rebuild architecture with SSM on dollar-bar sequences.
  - *Cost*: High — re-validates everything on a different timescale.
  - *Risk*: Executability — 140ms bars may not be tradeable with retail MES infrastructure.
  - *Justified only if*: Path B shows large improvement AND oracle expectancy is re-validated on dollar bars.

  **Path B — Cross-scale temporal feature aggregation** (recommended first move):
  Keep `time_5s` as primary bar type (oracle expectancy already validated). Compute dollar-bar-scale temporal features (AR residuals, rolling vol in dollar time, momentum in dollar time), aggregate them to the time_5s bar grid (e.g., for each time_5s bar, compute temporal stats from the ~35 dollar_25k bars in that 5s window), and test whether they improve the existing CNN+GBT on time_5s.
  - *Cost*: Low — one additional feature engineering step + Tier 2 augmentation test.
  - *Advantage*: Tests whether dollar-bar temporal signal is exploitable at the time_5s execution timescale without rebuilding the pipeline.
  - *If Path B passes dual threshold*: Temporal encoder justified as feature engineering. Dollar-bar temporal features become part of the time_5s input vector.
  - *If Path B fails*: Temporal signal exists in dollar time but doesn't transfer to the time_5s execution timescale. Proceed without temporal encoder.

- **Volume_100 passes but dollar_25k fails** → Unexpected given R1 priors. Investigate carefully before acting.

- **If signal is found, also report**: Is it linear (Ridge ≈ GBT) or nonlinear (GBT >> Ridge)? Linear → simple feature augmentation suffices. Nonlinear → SSM or learned encoder warranted.

---

## Compute Budget

### Feature Export (C++ tool)

| Bar Type | Est. Bars | Est. CSV Size | Est. Wall-Clock |
|----------|----------|--------------|-----------------|
| volume_100 | ~116K | ~50 MB | ~1.5 hr |
| dollar_25k | ~3.1M | ~1.3 GB | ~2.5 hr |

### R4 Analysis (Python)

| Bar Type | Training Rows | GBT Fits | Est. Wall-Clock |
|----------|--------------|----------|-----------------|
| volume_100 | ~93K (after warmup) | ~480 | ~40 min |
| dollar_25k | ~2.5M (after warmup) | ~480 | ~6-8 hr |

Dollar_25k dominates: XGBoost on 2.5M rows (max_depth=4, n_estimators=200) takes ~30-60s per fit vs ~3s for 70K rows.

**Total**: volume_100 ~2 hr, dollar_25k ~9-11 hr.
**Strategy**: Run volume_100 first (fast negative confirmation). Then dollar_25k in background overnight.
**GPU hours**: 0.

---

## Implementation

### Step 1: TDD sub-cycle — `bar_feature_export` tool

Spec: `.kit/docs/bar-feature-export.md`
Phases: `red` → `green` → `refactor` → `ship`

Tool: `tools/bar_feature_export.cpp`
CLI: `./bar_feature_export --bar-type <type> --bar-param <threshold> --output <csv_path>`
Pipeline: StreamingBookBuilder → BarFactory → BarFeatureComputer → FeatureExporter CSV

Reuse from `info_decomposition_export.cpp`: StreamingBookBuilder, SELECTED_DAYS, MES_CONTRACTS, format_float, date_to_string. Replace hardcoded `BarFactory::create("time", 5.0)` with CLI-parameterized call.

### Step 2: Export feature CSVs + sanity check

```bash
./build/bar_feature_export --bar-type volume --bar-param 100 \
  --output .kit/results/temporal-predictability-event-bars/volume_100/features.csv

./build/bar_feature_export --bar-type dollar --bar-param 25000 \
  --output .kit/results/temporal-predictability-event-bars/dollar_25k/features.csv
```

**Sanity check**: Verify row counts match R1 expectations before proceeding. See Verification section.

### Step 3: Parameterize R4 Python script

Add `argparse` to `research/R4_temporal_predictability.py`:
- `--input-csv` (default: `.kit/results/info-decomposition/features.csv`)
- `--output-dir` (default: `.kit/results/temporal-predictability`)
- `--bar-label` (default: `"time_5s"`)

~15 lines of changes. Backward compatible.

### Step 4: Run R4 on volume_100

```bash
python research/R4_temporal_predictability.py \
  --input-csv .kit/results/temporal-predictability-event-bars/volume_100/features.csv \
  --output-dir .kit/results/temporal-predictability-event-bars/volume_100 \
  --bar-label volume_100
```

~40 min. Quick negative confirmation expected (R1 AR R² for volume_100 ≈ time_5s).

### Step 5: Run R4 on dollar_25k

```bash
python research/R4_temporal_predictability.py \
  --input-csv .kit/results/temporal-predictability-event-bars/dollar_25k/features.csv \
  --output-dir .kit/results/temporal-predictability-event-bars/dollar_25k \
  --bar-label dollar_25k
```

~6-8 hr. Run in background. This is the critical test.

### Step 6: Comparison analysis + state updates

Write `.kit/results/temporal-predictability-event-bars/analysis.md`:
- Cross-bar-type comparison table: bar_type × horizon → best Tier 1 R², best Tier 2 Δ
- Reference R1 AR R² numbers as prior expectation
- Decision rule evaluation per bar type
- Horizon mismatch analysis (clock-time equivalences)
- If signal found: linear vs nonlinear assessment, Path A vs Path B recommendation
- Executability assessment

Update `.kit/RESEARCH_LOG.md` and `CLAUDE.md` Current State section.

---

## Output Structure

```
.kit/results/temporal-predictability-event-bars/
  volume_100/
    features.csv       ← C++ export (Step 2)
    metrics.json       ← R4 analysis (Step 4)
    analysis.md        ← R4 analysis (Step 4)
  dollar_25k/
    features.csv       ← C++ export (Step 2)
    metrics.json       ← R4 analysis (Step 5)
    analysis.md        ← R4 analysis (Step 5)
  analysis.md          ← cross-bar comparison + decision (Step 6)
```

---

## Verification

### Day Selection Sanity Check (Step 2)

Before running R4, verify that the 19 selected days produce bar counts consistent with R1's numbers. R1 used `select_stratified_days(5)` in `subordination_test.cpp` which deterministically picks the same 19 days. `info_decomposition_export.cpp` hardcodes the same list (line 260: "Same 19 days from R1"). However, verify:

- Dollar_25k produces ~3.1M bars total (~165K/day median). If significantly different, the "33× higher AR R²" prior from R1 may not hold.
- Volume_100 produces ~116K bars total (~6K/day median).
- If counts diverge >10% from R1, document why before proceeding. Possible causes: R1 used 20 days (not 19), rollover exclusion differences, RTH boundary handling.

### CSV Schema Check

Diff the header line of the new feature CSVs against `.kit/results/info-decomposition/features.csv`. Column names, order, and count must match exactly. The R4 Python script relies on specific column names: `bar_index` (sorting), `return_1` (lag construction), `fwd_return_{1,5,20,100}` (targets), `book_snap_{0..39}` (Track B.1), `is_warmup`, `day`.

### Post-Analysis Checks

- metrics.json has same schema as `.kit/results/temporal-predictability/metrics.json`
- All 4 decision rules evaluated per bar type
- analysis.md includes explicit comparison to time_5s R4 results

---

## Resume Protocol

This task spans multiple sessions (~13 hours total compute). If resuming with a cleared context:

1. Read this spec first.
2. Check which outputs already exist:

| Check | File | If exists → |
|-------|------|-------------|
| TDD tool built? | `build/bar_feature_export` exists and is executable | Skip Step 1 |
| Volume_100 features exported? | `.kit/results/temporal-predictability-event-bars/volume_100/features.csv` | Skip Step 2a |
| Dollar_25k features exported? | `.kit/results/temporal-predictability-event-bars/dollar_25k/features.csv` | Skip Step 2b |
| R4 script parameterized? | `research/R4_temporal_predictability.py` accepts `--input-csv` arg | Skip Step 3 |
| Volume_100 R4 complete? | `.kit/results/temporal-predictability-event-bars/volume_100/metrics.json` | Skip Step 4 |
| Dollar_25k R4 complete? | `.kit/results/temporal-predictability-event-bars/dollar_25k/metrics.json` | Skip Step 5 |
| Comparison analysis done? | `.kit/results/temporal-predictability-event-bars/analysis.md` | Skip Step 6, review for completeness |

3. Pick up from the earliest incomplete step.
4. If a feature export or R4 analysis is running in background, check for the process before re-launching.

---

## Exit Criteria

- [x] `bar_feature_export` tool built and tested (TDD sub-cycle complete)
- [x] Feature CSVs exported for volume_100 and dollar_25k
- [x] Bar counts sanity-checked against R1 expectations
- [x] CSV column schema validated against info-decomposition/features.csv
- [x] R4 Python script parameterized (backward compatible)
- [ ] Tier 1 AR R² computed for all 36 cells × 2 bar types
- [ ] Tier 2 augmentation R² computed for all 24 cells × 2 bar types
- [ ] Temporal features correctly constructed (no lookahead, no cross-day leakage)
- [ ] Paired statistical tests with Holm-Bonferroni correction
- [ ] Feature importance extracted from GBT for interpretability
- [ ] Decision rules evaluated per bar type
- [ ] Cross-bar-type comparison with time_5s results documented
- [ ] Horizon mismatch and executability caveats documented
- [ ] If signal found: downstream path (A vs B) recommended with justification
- [ ] Results written to `.kit/results/temporal-predictability-event-bars/`
- [ ] Summary entry ready for `.kit/RESEARCH_LOG.md`
- [ ] `CLAUDE.md` Current State updated
