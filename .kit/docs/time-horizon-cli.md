# TDD Spec: Triple Barrier Time Horizon CLI Flags

**Date:** 2026-02-26
**Priority:** P0 — blocks label geometry experiment re-run
**Parent:** Label Geometry Phase 1 (REFUTED — time horizon too short)

---

## Problem

`max_time_horizon_s` (300 seconds / 5 minutes) and `volume_horizon` (500 contracts) are hardcoded in both `bar_feature_export.cpp` and `oracle_expectancy.cpp`. The 300-second cap causes 90.7-98.9% of bars to be labeled "hold" because MES price doesn't traverse the barrier distance within 5 minutes on most bars.

The intended trading strategy holds positions for seconds to up to 1 hour. The time horizon must be configurable to at least 3600 seconds (1 hour) to produce meaningful label distributions.

---

## Requirements

### R1: `--max-time-horizon <seconds>` flag for `bar_feature_export`

Add a CLI flag `--max-time-horizon <seconds>` to `bar_feature_export`:
- Default: 3600 (1 hour) — **CHANGED from 300**
- Minimum: 1 second
- Maximum: 86400 (24 hours)
- Applied to `TripleBarrierConfig.max_time_horizon_s` before label computation
- Works with both `--legacy-labels` and bidirectional (default) labels

### R2: `--max-time-horizon <seconds>` flag for `oracle_expectancy`

Add the same flag to `oracle_expectancy`:
- Default: 3600 (1 hour) — **CHANGED from 300**
- Same validation as R1
- Applied to `OracleConfig.max_time_horizon_s`

### R3: `--volume-horizon <contracts>` flag for `bar_feature_export`

Add a CLI flag `--volume-horizon <contracts>` to `bar_feature_export`:
- Default: 50000 (effectively unlimited for MES at 1-hour windows)
- Minimum: 1
- Applied to `TripleBarrierConfig.volume_horizon`

### R4: `--volume-horizon <contracts>` flag for `oracle_expectancy`

Same flag for `oracle_expectancy`:
- Default: 50000
- Same validation as R3
- Applied to `OracleConfig.volume_horizon`

### R5: Updated defaults in config structs

Change the default values in the config structs themselves:
- `TripleBarrierConfig.max_time_horizon_s`: 300 → 3600
- `TripleBarrierConfig.volume_horizon`: 500 → 50000
- `OracleConfig.max_time_horizon_s`: 300 → 3600
- `OracleConfig.volume_horizon`: 500 → 50000

---

## Implementation Notes

### Files to modify

1. **`tools/bar_feature_export.cpp`**
   - Add `--max-time-horizon` and `--volume-horizon` to CLI parsing (near existing `--target`/`--stop` parsing)
   - Update lines ~712-714 where `tb_cfg.volume_horizon` and `tb_cfg.max_time_horizon_s` are assigned
   - Update `usage()` function

2. **`tools/oracle_expectancy.cpp`**
   - Add `--max-time-horizon` and `--volume-horizon` to CLI parsing
   - Update lines ~418-419 where config values are assigned
   - Update `usage()` function

3. **`src/backtest/triple_barrier.hpp`**
   - Change default `max_time_horizon_s` from 300 to 3600
   - Change default `volume_horizon` from 500 to 50000

4. **`src/backtest/oracle_replay.hpp`**
   - Change default `max_time_horizon_s` from 300 to 3600
   - Change default `volume_horizon` from 500 to 50000

### Pattern to follow

The `--target` and `--stop` flags added in PR #28 provide the exact pattern:
- CLI parsing with `std::stoi()`
- Validation with clear error messages
- Applied to config before use

### Validation

- `--max-time-horizon 0` → error: "max-time-horizon must be >= 1"
- `--max-time-horizon -5` → error: "max-time-horizon must be >= 1"
- `--max-time-horizon 3600` → accepted (1 hour)
- `--volume-horizon 0` → error: "volume-horizon must be >= 1"
- Existing tests should still pass with the new defaults (tests set their own config values)

---

## Test Plan

### T1: Default values changed
- Construct `TripleBarrierConfig` with no arguments → `max_time_horizon_s == 3600`, `volume_horizon == 50000`
- Construct `OracleConfig` with no arguments → same

### T2: bar_feature_export accepts --max-time-horizon
- Run `bar_feature_export --bar-type time --bar-param 5 --max-time-horizon 3600 --output /tmp/test.parquet` on 1 file
- Output has 152 columns, non-zero directional labels

### T3: bar_feature_export accepts --volume-horizon
- Run with `--volume-horizon 50000` → valid output

### T4: oracle_expectancy accepts --max-time-horizon
- Run `oracle_expectancy --max-time-horizon 3600 --output /tmp/test_oracle.json` on 1 file
- JSON output has trade_count > 0

### T5: oracle_expectancy accepts --volume-horizon
- Run with `--volume-horizon 50000` → valid output

### T6: Invalid values rejected
- `--max-time-horizon 0` → exit non-zero with error
- `--max-time-horizon -1` → exit non-zero with error
- `--volume-horizon 0` → exit non-zero with error

### T7: Backward compatibility
- Running without `--max-time-horizon` uses new default (3600)
- Running without `--volume-horizon` uses new default (50000)
- Existing `--target`, `--stop`, `--legacy-labels` flags still work

### T8: Combined flags
- `--target 15 --stop 3 --max-time-horizon 3600 --volume-horizon 50000` → all applied correctly

---

## Exit Criteria

- [ ] `--max-time-horizon` flag added to `bar_feature_export` and `oracle_expectancy`
- [ ] `--volume-horizon` flag added to both tools
- [ ] Default `max_time_horizon_s` changed to 3600 in both config structs
- [ ] Default `volume_horizon` changed to 50000 in both config structs
- [ ] Invalid values rejected with clear error messages
- [ ] All existing tests pass
- [ ] New tests for the flags pass
- [ ] `--help` output includes the new flags
