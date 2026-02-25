# TDD Spec: Wire Bidirectional Labels into bar_feature_export

**Date:** 2026-02-25
**Priority:** P0 — blocks full-year Parquet re-export with corrected labels
**Parent:** Bidirectional Label Export TDD (PR #26), Label Design Sensitivity experiment
**Depends on:** `compute_bidirectional_tb_label()` in `triple_barrier.hpp` (DONE)

---

## Problem

`bar_feature_export.cpp` calls `compute_tb_label()` (long-perspective only) to produce the `tb_label`, `tb_exit_type`, and `tb_bars_held` Parquet columns. The bidirectional labeling function (`compute_bidirectional_tb_label()`) now exists but isn't wired into the export tool. The Parquet schema also lacks the new diagnostic columns (`tb_both_triggered`, `tb_long_triggered`, `tb_short_triggered`).

## Design

### Changes to bar_feature_export.cpp

1. **Replace `compute_tb_label()` call with `compute_bidirectional_tb_label()`** — the config already has `bidirectional = true` by default, so the default export behavior changes to bidirectional.
2. **Add 3 new Parquet columns** to the Arrow schema:
   - `tb_both_triggered` (float64: 0.0 or 1.0)
   - `tb_long_triggered` (float64: 0.0 or 1.0)
   - `tb_short_triggered` (float64: 0.0 or 1.0)
3. **Add CLI flag `--legacy-labels`** that sets `bidirectional = false` for backward compatibility. Default is bidirectional (no flag needed).
4. **Existing columns unchanged:** `tb_label` (float64), `tb_exit_type` (string), `tb_bars_held` (float64) — same names, new semantics under bidirectional mode.

### Schema Change

Old schema: ... `tb_label`, `tb_exit_type`, `tb_bars_held` (3 columns)
New schema: ... `tb_label`, `tb_exit_type`, `tb_bars_held`, `tb_both_triggered`, `tb_long_triggered`, `tb_short_triggered` (6 columns)

Total columns: 149 → 152.

## Tests

### T1: Default Export Uses Bidirectional Labels
- Run `bar_feature_export` on a small test dataset (single day) with no flags
- Verify Parquet output has 152 columns (149 + 3 new)
- Verify `tb_both_triggered`, `tb_long_triggered`, `tb_short_triggered` columns exist
- Verify `tb_label` values are in {-1.0, 0.0, 1.0}

### T2: Legacy Flag Produces Old-Style Labels
- Run `bar_feature_export --legacy-labels` on same test dataset
- Verify Parquet output has 149 columns (no new columns)
- Verify `tb_label` values match old behavior (long-perspective only)

### T3: New Columns Have Correct Values
- Run on test data where we know expected labels:
  - A bar where only the long race triggers → `tb_long_triggered=1`, `tb_short_triggered=0`, `tb_both_triggered=0`, `tb_label=1`
  - A bar where only the short race triggers → `tb_long_triggered=0`, `tb_short_triggered=1`, `tb_both_triggered=0`, `tb_label=-1`
  - A bar where both trigger → `tb_long_triggered=1`, `tb_short_triggered=1`, `tb_both_triggered=1`, `tb_label=0`
  - A bar where neither triggers → all three = 0, `tb_label=0`

### T4: Label Distribution Shift Under Bidirectional
- Run both modes on same day of data
- Verify bidirectional mode produces MORE `tb_label=0` bars than legacy mode (many old -1 labels become 0)
- Verify bidirectional mode produces FEWER `tb_label=-1` bars than legacy mode

### T5: No Regression on Existing Columns
- Run bidirectional export on test data
- Verify all 149 original columns (features, metadata) are unchanged
- Specifically verify `close_mid`, `volatility_50`, `spread`, `time_sin` columns contain same values regardless of label mode

### T6: Schema Column Order
- Verify the 3 new columns appear AFTER `tb_bars_held` in the Parquet schema
- Verify column names are exact: `tb_both_triggered`, `tb_long_triggered`, `tb_short_triggered`

## Build

Follow existing project build conventions. Tests in `tests/bidirectional_export_test.cpp` or extend `tests/bar_feature_export_test.cpp`. Use GTest. Build with CMake. Tests that require raw MBO data should be marked as integration tests (LABEL integration) — unit tests use synthetic bar data only.

**NOTE:** The actual full-year re-export (251 days, 49GB raw data) will be run on AWS EC2, NOT locally. This TDD cycle only covers the code changes and unit/integration tests on small data.

## Exit Criteria

- [ ] `bar_feature_export` defaults to bidirectional labels
- [ ] 3 new Parquet columns present in output schema
- [ ] `--legacy-labels` flag produces old-style 149-column output
- [ ] All tests T1-T6 pass
- [ ] No regression on existing bar_feature_export_test tests
- [ ] No regression on existing triple_barrier_test tests
