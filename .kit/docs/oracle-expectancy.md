# Oracle Expectancy Extraction — TDD Spec

## Summary

A C++ tool (`tools/oracle_expectancy.cpp`) that runs the `OracleReplay` engine on real MES MBO data to extract oracle expectancy statistics. This resolves the **#1 open question** from the R6 synthesis: "Extract oracle expectancy from Phase 3 C++ test output."

The oracle expectancy is the average net PnL per trade when a perfect-foresight oracle labels every bar. If oracle expectancy <= 0 after transaction costs, **no model can be profitable** and the CONDITIONAL GO becomes a NO-GO.

## Motivation

All existing oracle replay tests use synthetic bar sequences. No code currently runs the oracle on real MES data. This tool closes that gap.

## Architecture

```
DATA/*.dbn.zst → StreamingBookBuilder → BookSnapshot[] → TimeBarBuilder(5s) → Bar[]
    → OracleReplay(FIRST_TO_HIT, default config) → BacktestResult
    → OracleReplay(TRIPLE_BARRIER, default config) → BacktestResult
    → MultiDayRunner::aggregate() → Summary JSON to stdout
```

Reuses the existing `StreamingBookBuilder` pattern from `tools/subordination_test.cpp`.

## Data Contract

### Input
- `.dbn.zst` files from `DATA/GLBX-20260207-L953CAPU5B/`
- MES front-month contracts: MESH2, MESM2, MESU2, MESZ2
- Instrument IDs: 11355, 13615, 10039, 10299

### Output
JSON to stdout with structure:

```json
{
  "config": {
    "bar_type": "time_5s",
    "target_ticks": 10,
    "stop_ticks": 5,
    "take_profit_ticks": 20,
    "volume_horizon": 500,
    "max_time_horizon_s": 300,
    "costs": {
      "commission_per_side": 0.62,
      "fixed_spread_ticks": 1,
      "slippage_ticks": 0,
      "contract_multiplier": 5.0,
      "tick_size": 0.25
    }
  },
  "days_processed": 20,
  "days_skipped": 0,
  "first_to_hit": {
    "total_trades": ...,
    "winning_trades": ...,
    "losing_trades": ...,
    "win_rate": ...,
    "gross_pnl": ...,
    "net_pnl": ...,
    "expectancy": ...,
    "profit_factor": ...,
    "sharpe": ...,
    "max_drawdown": ...,
    "trades_per_day": ...,
    "avg_bars_held": ...,
    "avg_duration_s": ...,
    "hold_fraction": ...,
    "exit_reasons": {
      "target": ...,
      "stop": ...,
      "take_profit": ...,
      "expiry": ...,
      "session_end": ...,
      "safety_cap": ...
    }
  },
  "triple_barrier": { ... same structure ... },
  "per_quarter": {
    "Q1": { "first_to_hit": {...}, "triple_barrier": {...} },
    "Q2": { ... },
    "Q3": { ... },
    "Q4": { ... }
  }
}
```

## OracleConfig (defaults)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `target_ticks` | 10 | 2.50 points entry threshold |
| `stop_ticks` | 5 | 1.25 points stop loss |
| `take_profit_ticks` | 20 | 5.00 points take profit |
| `volume_horizon` | 500 | Forward look volume |
| `max_time_horizon_s` | 300 | 5 minute safety cap |
| `tick_size` | 0.25 | MES tick size |

## ExecutionCosts (defaults)

| Parameter | Value |
|-----------|-------|
| `commission_per_side` | $0.62 |
| `fixed_spread_ticks` | 1 |
| `slippage_ticks` | 0 |
| `contract_multiplier` | 5.0 |
| `tick_size` | 0.25 |
| `tick_value` | $1.25 |

Round-trip cost = 2 x ($0.62 + $0.625) = **$2.49 per trade**.

## Day Selection

Stratified: 5 days per quarter (Q1-Q4 2022), 20 days total. Same `select_stratified_days` logic as `tools/subordination_test.cpp`. Skip rollover-excluded dates.

## Quarterly Contracts

| Symbol | instrument_id | Start | End | Rollover |
|--------|--------------|-------|-----|----------|
| MESH2 | 11355 | 20220103 | 20220318 | 20220318 |
| MESM2 | 13615 | 20220319 | 20220617 | 20220617 |
| MESU2 | 10039 | 20220618 | 20220916 | 20220916 |
| MESZ2 | 10299 | 20220917 | 20221230 | 20221216 |

## Per-Day Processing

1. Load `.dbn.zst` file for date
2. Stream MBO events through `StreamingBookBuilder` (same as subordination_test.cpp)
3. Build time_5s bars via `BarFactory::create("time", 5.0)`
4. Discard first 20 warmup bars
5. Run `OracleReplay` with `FIRST_TO_HIT` config → `BacktestResult`
6. Run `OracleReplay` with `TRIPLE_BARRIER` config → `BacktestResult`
7. Store per-day results, tagged with quarter

## Aggregation

Use same aggregation logic as `MultiDayRunner::aggregate()`:
- Sum trades, wins, losses, gross_pnl, net_pnl across days
- Recompute win_rate, expectancy, profit_factor, sharpe, max_drawdown
- Track per-quarter breakdowns

## Testable Components

The tool itself is a standalone executable reading real data — not directly unit-testable. However, the following **testable extraction layer** is required:

### `OracleExpectancyReport` struct (new, in `src/backtest/oracle_expectancy_report.hpp`)

```cpp
struct OracleExpectancyReport {
    int days_processed = 0;
    int days_skipped = 0;
    BacktestResult first_to_hit;
    BacktestResult triple_barrier;
    std::map<std::string, BacktestResult> fth_per_quarter;
    std::map<std::string, BacktestResult> tb_per_quarter;
};
```

### `oracle_expectancy::to_json(const OracleExpectancyReport&)` — JSON serializer

Returns a JSON string matching the output format above. Must be deterministic and testable.

### `oracle_expectancy::aggregate_day_results(...)` — aggregation logic

```cpp
OracleExpectancyReport aggregate_day_results(
    const std::vector<DayResult>& fth_results,
    const std::vector<DayResult>& tb_results,
    const std::vector<int>& dates  // for quarter assignment
);
```

Wraps `MultiDayRunner::aggregate()` with per-quarter splitting.

## Validation Gates

1. `to_json` output contains all required fields from the output schema
2. `to_json` round-trips numeric values correctly (float precision)
3. `aggregate_day_results` correctly sums trades across days
4. `aggregate_day_results` correctly computes per-quarter splits
5. `aggregate_day_results` with empty input returns zero-initialized report
6. `aggregate_day_results` skips days with `skipped=true`
7. Expectancy = net_pnl / total_trades (consistency check)
8. Win rate = winning_trades / total_trades
9. Exit reason counts sum to total_trades
10. Per-quarter totals sum to overall totals

## CMake Integration

```cmake
# ---------- Oracle Expectancy Extraction Tool ----------
add_executable(oracle_expectancy tools/oracle_expectancy.cpp)
target_include_directories(oracle_expectancy PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(oracle_expectancy PRIVATE databento::databento)
```

Test target:
```cmake
# Add to TEST_TARGETS list:
oracle_expectancy_test
```

## Project Structure

New files:
- `src/backtest/oracle_expectancy_report.hpp` — report struct + `to_json` + `aggregate_day_results`
- `tests/oracle_expectancy_test.cpp` — unit tests for the report layer
- `tools/oracle_expectancy.cpp` — standalone tool (not unit-tested; integration-tested by running on data)
