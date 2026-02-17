# Phase 2: Execution Cost Model & Oracle Replay [Engineering]

**Spec**: TRAJECTORY.md §5 (oracle), §6 (costs), §9.1 (backtest design)
**Depends on**: Phase 1 (bar-construction) — needs `Bar`, `BarBuilder`, bar types.
**Unlocks**: Phase 3 (multi-day-backtest).

---

## Objective

Implement the execution cost model, the event-denominated oracle (both first-to-hit and triple barrier labeling), the oracle replay engine, and the backtest runner. This phase builds the infrastructure for Phase 3's multi-day backtest.

---

## Data Contracts

### ExecutionCosts (§6.1)

```cpp
struct ExecutionCosts {
    float commission_per_side = 0.62f;    // USD
    enum class SpreadModel { FIXED, EMPIRICAL };
    SpreadModel spread_model = SpreadModel::FIXED;
    int   fixed_spread_ticks = 1;
    int   slippage_ticks = 0;
    float contract_multiplier = 5.0f;     // MES: $5 per index point
    float tick_size = 0.25f;
    float tick_value = 1.25f;             // $5.00 × 0.25

    // Compute cost for a single side (entry or exit)
    float per_side_cost(float actual_spread_ticks = 1.0f) const;
    // Compute full round-trip cost
    float round_trip_cost(float entry_spread, float exit_spread) const;
};
```

**Per round-trip**: $1.24 commission + $1.25 spread (1 tick) + slippage = ~$2.49 minimum.

### OracleConfig (§5.1)

```cpp
struct OracleConfig {
    uint64_t volume_horizon = 500;       // look forward until this many contracts trade
    uint32_t max_time_horizon_s = 300;   // safety cap (5 min) — abnormal inactivity only
    int      target_ticks = 10;          // 2.50 points
    int      stop_ticks = 5;             // 1.25 points
    int      take_profit_ticks = 20;     // 5.00 points
    float    tick_size = 0.25f;

    enum class LabelMethod { FIRST_TO_HIT, TRIPLE_BARRIER };
    LabelMethod label_method = LabelMethod::FIRST_TO_HIT;
};
```

### TripleBarrierConfig (§5.2)

```cpp
struct TripleBarrierConfig {
    int      target_ticks = 10;
    int      stop_ticks = 5;
    uint64_t volume_horizon = 500;       // expiry in volume, not time
    int      min_return_ticks = 2;       // at expiry, |return| must exceed this for directional label
    uint32_t max_time_horizon_s = 300;
};
```

**Triple barrier exits**: upper barrier (+target) → label +1, lower barrier (-stop) → label -1, expiry → sign(return) if |return| >= min_return_ticks else HOLD (0).

### TradeRecord (§9.1)

```cpp
struct TradeRecord {
    int64_t entry_ts, exit_ts;
    float   entry_price, exit_price;
    int     direction;                    // +1 long, -1 short
    float   gross_pnl, net_pnl;          // dollars
    int     entry_bar_idx, exit_bar_idx;
    int     bars_held;
    float   duration_s;
    int     exit_reason;                  // 0=target, 1=stop, 2=take_profit,
                                          // 3=expiry, 4=session_end, 5=safety_cap
};
```

### BacktestResult (§9.1)

```cpp
struct BacktestResult {
    std::vector<TradeRecord> trades;

    int   total_trades, winning_trades, losing_trades;
    float win_rate;
    float gross_pnl, net_pnl;
    float avg_win, avg_loss;              // dollars
    float profit_factor;                  // gross_wins / gross_losses
    float expectancy;                     // net_pnl / total_trades
    float sharpe;                         // annualized from per-trade returns
    float max_drawdown;
    float avg_bars_held, avg_duration_s;

    std::map<int, float> pnl_by_hour;    // hour (9-15) → net PnL
    std::map<int, int>   label_counts;   // action → count
    float hold_fraction;
    std::map<int, int>   exit_reason_counts;
    int   safety_cap_triggered_count;
    float safety_cap_fraction;
};
```

### OracleReplay

```cpp
class OracleReplay {
public:
    OracleReplay(const OracleConfig& oracle, const ExecutionCosts& costs);

    // Run oracle on a bar sequence, track positions, compute PnL per trade.
    BacktestResult run(const std::vector<Bar>& bars);
};
```

### BacktestRunner

```cpp
class BacktestRunner {
public:
    BacktestRunner(const BacktestConfig& config);

    // Orchestrate multi-day runs, aggregate results.
    BacktestResult run_all_days();
};
```

---

## Oracle Logic

### First-to-Hit
- From each bar, look forward accumulating volume until `volume_horizon` contracts traded.
- If mid-price crosses `+target_ticks` first → LONG (+1).
- If mid-price crosses `-stop_ticks` first → SHORT (-1).
- If safety cap (`max_time_horizon_s`) reached → HOLD (0). Log with timestamp.

### Triple Barrier
- Three barriers: upper (+target_ticks), lower (-stop_ticks), expiry (volume_horizon).
- First barrier hit determines label.
- At expiry: label = sign(return) if |return| >= min_return_ticks, else HOLD.
- Safety cap applies identically.

### Execution Assumptions (§6.2)
- Entry/exit at mid_price. Spread cost accounts for bid-ask statistically.
- No partial fills (1 contract, fully filled).
- No market impact.
- Commission applied per-side (2× per round-trip).
- Force-close at session end if oracle hasn't exited (exit_reason=4, log warning).

---

## Project Structure

```
src/backtest/
  execution_costs.hpp
  oracle_replay.hpp
  triple_barrier.hpp
  trade_record.hpp
  backtest_runner.hpp

tests/
  execution_costs_test.cpp
  oracle_replay_test.cpp
  triple_barrier_test.cpp
```

---

## Validation Gate

```
Assert: OracleReplay PnL accounting is correct:
        sum(trade.net_pnl) == result.net_pnl
Assert: Every ENTER has a matching EXIT (no open positions at session end)
        (Force-close at session end if oracle hasn't exited; log warning, exit_reason=4)
Assert: Commission is applied per-side (2× per round-trip)
Assert: Spread cost uses actual spread from bar data when spread_model="empirical"
Assert: Trade direction matches oracle label (+1 for ENTER LONG, -1 for ENTER SHORT)
Assert: No trades during first W bars (observation window warmup)
Assert: Triple barrier correctly handles all three exit conditions (target, stop, expiry)
Assert: Triple barrier expiry labels: sign(return) when |return| >= min_return_ticks, else HOLD
Assert: Safety cap triggers are logged with timestamps and counted in BacktestResult
Assert: exit_reason is correctly recorded for every trade
```
