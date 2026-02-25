# TDD Spec: Parameterize oracle_expectancy CLI

## Context

`tools/oracle_expectancy.cpp` currently hardcodes triple barrier parameters at line 414-416:
```cpp
fth_cfg.target_ticks = 10;
fth_cfg.stop_ticks = 5;
fth_cfg.take_profit_ticks = 20;
```

The `OracleConfig` struct already accepts these as configurable fields. The label design sensitivity experiment (`.kit/experiments/label-design-sensitivity.md`) needs to run the oracle with 5 different geometries. This requires CLI arg parsing.

Follow the same arg parsing pattern as `bar_feature_export.cpp` (lines 489-520): simple `for` loop over `argv`, string matching on `--flag`, `std::stoi()` conversion.

## Requirements

- R1: `oracle_expectancy` accepts optional `--target <ticks>` flag (default: 10)
- R2: `oracle_expectancy` accepts optional `--stop <ticks>` flag (default: 5)
- R3: `oracle_expectancy` accepts optional `--take-profit <ticks>` flag (default: 20)
- R4: `oracle_expectancy` accepts optional `--output <path>` flag for JSON results (default: stdout only)
- R5: When no flags provided, behavior is identical to current (backward compatible)
- R6: Invalid flag values (negative, zero, non-integer) print usage and exit 1
- R7: `--help` flag prints usage and exits 0

## Tests

- T1 (R5): No args → same output as current (regression test against known metrics: 4,873 TB trades, $4.00/trade, PF=3.30, WR=64.3%)
- T2 (R1-R3): `--target 15 --stop 3` → runs with modified geometry, produces valid metrics
- T3 (R1-R3): `--target 10 --stop 5 --take-profit 20` → identical to default (explicit defaults match implicit)
- T4 (R4): `--output /tmp/test_oracle.json` → writes JSON with fields: target_ticks, stop_ticks, take_profit_ticks, trades, win_rate, expectancy, profit_factor, per_quarter
- T5 (R6): `--target -1` → exits 1 with usage message
- T6 (R6): `--target abc` → exits 1 with usage message
- T7 (R7): `--help` → prints usage, exits 0

## Constraints

- Do NOT change the default behavior or default parameter values
- Do NOT change the OracleConfig struct (it already supports these fields)
- Do NOT change any header files unless necessary for JSON output
- Follow `bar_feature_export.cpp` arg parsing pattern exactly
- JSON output (R4) should use a simple hand-written serializer (no nlohmann or external JSON lib) — same pattern as other tools in this project

## Exit Criteria

- [ ] All 7 tests pass
- [ ] `ctest --label-exclude integration` still passes (no regressions)
- [ ] `--target 15 --stop 3` produces valid output with different metrics than default
- [ ] No-arg invocation produces byte-identical stdout to pre-change binary
