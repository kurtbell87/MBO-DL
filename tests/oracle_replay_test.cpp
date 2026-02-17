// oracle_replay_test.cpp — TDD RED phase tests for OracleReplay engine
// Spec: .kit/docs/oracle-replay.md §OracleConfig, §OracleReplay, §TradeRecord, §BacktestResult
//
// Tests the oracle replay engine: first-to-hit labeling via bar sequence,
// position tracking, PnL accounting, trade record creation, and BacktestResult
// aggregation. Tests are designed around synthetic bar sequences with known
// price trajectories.
//
// No implementation files exist yet — all tests must fail to compile.

#include <gtest/gtest.h>

// Headers that the implementation must provide (spec §Project Structure):
#include "backtest/execution_costs.hpp"
#include "backtest/oracle_replay.hpp"
#include "backtest/trade_record.hpp"

#include "bars/bar.hpp"  // Bar struct (already exists from Phase 1)
#include "test_bar_helpers.hpp"

#include <vector>

// ===========================================================================
// Helpers — oracle-specific config factories
// ===========================================================================
namespace {

using test_helpers::TICK;
using test_helpers::make_bar;
using test_helpers::make_bar_series;
using test_helpers::make_bar_path;

// Default OracleConfig with first-to-hit.
OracleConfig make_default_config() {
    OracleConfig cfg{};
    cfg.volume_horizon = 500;
    cfg.max_time_horizon_s = 300;
    cfg.target_ticks = 10;
    cfg.stop_ticks = 5;
    cfg.take_profit_ticks = 20;
    cfg.tick_size = 0.25f;
    cfg.label_method = OracleConfig::LabelMethod::FIRST_TO_HIT;
    return cfg;
}

// OracleConfig with small horizons for compact test scenarios.
OracleConfig make_small_config() {
    OracleConfig cfg{};
    cfg.volume_horizon = 100;
    cfg.max_time_horizon_s = 10;
    cfg.target_ticks = 4;   // 1.00 point
    cfg.stop_ticks = 2;     // 0.50 point
    cfg.take_profit_ticks = 8;  // 2.00 points
    cfg.tick_size = 0.25f;
    cfg.label_method = OracleConfig::LabelMethod::FIRST_TO_HIT;
    return cfg;
}

}  // namespace

// ===========================================================================
// 1. OracleConfig defaults
// ===========================================================================
class OracleConfigTest : public ::testing::Test {};

TEST_F(OracleConfigTest, DefaultVolumeHorizon) {
    OracleConfig cfg{};
    EXPECT_EQ(cfg.volume_horizon, 500u);
}

TEST_F(OracleConfigTest, DefaultMaxTimeHorizonSeconds) {
    OracleConfig cfg{};
    EXPECT_EQ(cfg.max_time_horizon_s, 300u);
}

TEST_F(OracleConfigTest, DefaultTargetTicks) {
    OracleConfig cfg{};
    EXPECT_EQ(cfg.target_ticks, 10);
}

TEST_F(OracleConfigTest, DefaultStopTicks) {
    OracleConfig cfg{};
    EXPECT_EQ(cfg.stop_ticks, 5);
}

TEST_F(OracleConfigTest, DefaultTakeProfitTicks) {
    OracleConfig cfg{};
    EXPECT_EQ(cfg.take_profit_ticks, 20);
}

TEST_F(OracleConfigTest, DefaultTickSize) {
    OracleConfig cfg{};
    EXPECT_FLOAT_EQ(cfg.tick_size, 0.25f);
}

TEST_F(OracleConfigTest, DefaultLabelMethodIsFirstToHit) {
    OracleConfig cfg{};
    EXPECT_EQ(cfg.label_method, OracleConfig::LabelMethod::FIRST_TO_HIT);
}

// ===========================================================================
// 2. TradeRecord fields
// ===========================================================================
class TradeRecordTest : public ::testing::Test {};

TEST_F(TradeRecordTest, DefaultConstructionZeroInit) {
    TradeRecord tr{};
    EXPECT_EQ(tr.entry_ts, 0);
    EXPECT_EQ(tr.exit_ts, 0);
    EXPECT_FLOAT_EQ(tr.entry_price, 0.0f);
    EXPECT_FLOAT_EQ(tr.exit_price, 0.0f);
    EXPECT_EQ(tr.direction, 0);
    EXPECT_FLOAT_EQ(tr.gross_pnl, 0.0f);
    EXPECT_FLOAT_EQ(tr.net_pnl, 0.0f);
    EXPECT_EQ(tr.entry_bar_idx, 0);
    EXPECT_EQ(tr.exit_bar_idx, 0);
    EXPECT_EQ(tr.bars_held, 0);
    EXPECT_FLOAT_EQ(tr.duration_s, 0.0f);
    EXPECT_EQ(tr.exit_reason, 0);
}

TEST_F(TradeRecordTest, ExitReasonConstants) {
    // Spec: 0=target, 1=stop, 2=take_profit, 3=expiry, 4=session_end, 5=safety_cap
    // These are int values, verify they are in range.
    TradeRecord tr{};
    tr.exit_reason = 0;  // target
    EXPECT_EQ(tr.exit_reason, 0);
    tr.exit_reason = 4;  // session_end
    EXPECT_EQ(tr.exit_reason, 4);
    tr.exit_reason = 5;  // safety_cap
    EXPECT_EQ(tr.exit_reason, 5);
}

// ===========================================================================
// 3. BacktestResult fields
// ===========================================================================
class BacktestResultTest : public ::testing::Test {};

TEST_F(BacktestResultTest, DefaultConstructionEmpty) {
    BacktestResult result{};
    EXPECT_TRUE(result.trades.empty());
    EXPECT_EQ(result.total_trades, 0);
    EXPECT_EQ(result.winning_trades, 0);
    EXPECT_EQ(result.losing_trades, 0);
    EXPECT_FLOAT_EQ(result.win_rate, 0.0f);
    EXPECT_FLOAT_EQ(result.gross_pnl, 0.0f);
    EXPECT_FLOAT_EQ(result.net_pnl, 0.0f);
}

TEST_F(BacktestResultTest, HasPnlByHourMap) {
    BacktestResult result{};
    EXPECT_TRUE(result.pnl_by_hour.empty());
}

TEST_F(BacktestResultTest, HasLabelCountsMap) {
    BacktestResult result{};
    EXPECT_TRUE(result.label_counts.empty());
}

TEST_F(BacktestResultTest, HasExitReasonCountsMap) {
    BacktestResult result{};
    EXPECT_TRUE(result.exit_reason_counts.empty());
}

TEST_F(BacktestResultTest, HasSafetyCapFields) {
    BacktestResult result{};
    EXPECT_EQ(result.safety_cap_triggered_count, 0);
    EXPECT_FLOAT_EQ(result.safety_cap_fraction, 0.0f);
}

TEST_F(BacktestResultTest, HasHoldFraction) {
    BacktestResult result{};
    EXPECT_FLOAT_EQ(result.hold_fraction, 0.0f);
}

TEST_F(BacktestResultTest, HasDrawdownAndSharpe) {
    BacktestResult result{};
    EXPECT_FLOAT_EQ(result.max_drawdown, 0.0f);
    EXPECT_FLOAT_EQ(result.sharpe, 0.0f);
}

TEST_F(BacktestResultTest, HasAvgBarsHeldAndDuration) {
    BacktestResult result{};
    EXPECT_FLOAT_EQ(result.avg_bars_held, 0.0f);
    EXPECT_FLOAT_EQ(result.avg_duration_s, 0.0f);
}

// ===========================================================================
// 4. OracleReplay — construction
// ===========================================================================
class OracleReplayTest : public ::testing::Test {
protected:
    OracleConfig oracle_cfg;
    ExecutionCosts costs;

    void SetUp() override {
        oracle_cfg = make_small_config();
        costs = ExecutionCosts{};
    }
};

TEST_F(OracleReplayTest, ConstructsWithConfigAndCosts) {
    OracleReplay replay(oracle_cfg, costs);
    // Should not throw — just verifies construction compiles and succeeds.
    (void)replay;
}

// ===========================================================================
// 5. OracleReplay — empty / trivial inputs
// ===========================================================================

TEST_F(OracleReplayTest, EmptyBarSequenceReturnsEmptyResult) {
    OracleReplay replay(oracle_cfg, costs);
    std::vector<Bar> bars;
    BacktestResult result = replay.run(bars);
    EXPECT_EQ(result.total_trades, 0);
    EXPECT_TRUE(result.trades.empty());
}

TEST_F(OracleReplayTest, SingleBarProducesNoTrades) {
    OracleReplay replay(oracle_cfg, costs);
    std::vector<Bar> bars = {make_bar(4500.0f, 100, 0)};
    BacktestResult result = replay.run(bars);
    EXPECT_EQ(result.total_trades, 0);
}

// ===========================================================================
// 6. OracleReplay — First-to-Hit: price rises → LONG signal
// ===========================================================================

TEST_F(OracleReplayTest, FirstToHitLongSignalOnPriceRise) {
    // Price rises by target_ticks * tick_size = 4 * 0.25 = 1.00 point.
    // The oracle should generate a LONG entry.
    oracle_cfg.target_ticks = 4;
    oracle_cfg.stop_ticks = 2;
    oracle_cfg.volume_horizon = 200;

    OracleReplay replay(oracle_cfg, costs);

    // 20 bars: price goes from 4500.0 to 4501.5 (6 ticks up = 1.5 points).
    // After sufficient lookahead, target (+1.0) should be hit before stop (-0.5).
    auto bars = make_bar_series(4500.0f, 4501.5f, 20, 50);
    BacktestResult result = replay.run(bars);

    // We expect at least one trade was generated and it was LONG
    EXPECT_GT(result.total_trades, 0);
    if (!result.trades.empty()) {
        EXPECT_EQ(result.trades[0].direction, 1);  // +1 = LONG
    }
}

// ===========================================================================
// 7. OracleReplay — First-to-Hit: price falls → SHORT signal
// ===========================================================================

TEST_F(OracleReplayTest, FirstToHitShortSignalOnPriceFall) {
    oracle_cfg.target_ticks = 4;
    oracle_cfg.stop_ticks = 2;
    oracle_cfg.volume_horizon = 200;

    OracleReplay replay(oracle_cfg, costs);

    // Price falls from 4500.0 to 4498.5 (6 ticks down).
    auto bars = make_bar_series(4500.0f, 4498.5f, 20, 50);
    BacktestResult result = replay.run(bars);

    EXPECT_GT(result.total_trades, 0);
    if (!result.trades.empty()) {
        EXPECT_EQ(result.trades[0].direction, -1);  // -1 = SHORT
    }
}

// ===========================================================================
// 8. OracleReplay — Trade direction matches oracle label
// ===========================================================================

TEST_F(OracleReplayTest, TradeDirectionMatchesOracleLabel) {
    // Validation gate: "Trade direction matches oracle label (+1 for ENTER LONG, -1 for SHORT)"
    oracle_cfg.target_ticks = 4;
    oracle_cfg.stop_ticks = 2;
    oracle_cfg.volume_horizon = 200;

    OracleReplay replay(oracle_cfg, costs);
    auto bars = make_bar_series(4500.0f, 4502.0f, 30, 50);
    BacktestResult result = replay.run(bars);

    for (const auto& trade : result.trades) {
        EXPECT_TRUE(trade.direction == 1 || trade.direction == -1)
            << "Trade direction must be +1 (LONG) or -1 (SHORT), got: " << trade.direction;
    }
}

// ===========================================================================
// 9. PnL accounting: sum(trade.net_pnl) == result.net_pnl
// ===========================================================================

TEST_F(OracleReplayTest, PnlAccountingSumMatchesTotal) {
    // Validation gate: "sum(trade.net_pnl) == result.net_pnl"
    oracle_cfg.target_ticks = 4;
    oracle_cfg.stop_ticks = 2;
    oracle_cfg.volume_horizon = 200;

    OracleReplay replay(oracle_cfg, costs);
    auto bars = make_bar_series(4500.0f, 4502.0f, 50, 50);
    BacktestResult result = replay.run(bars);

    float sum_net_pnl = 0.0f;
    for (const auto& trade : result.trades) {
        sum_net_pnl += trade.net_pnl;
    }
    EXPECT_FLOAT_EQ(sum_net_pnl, result.net_pnl);
}

// ===========================================================================
// 10. Gross PnL computed correctly
// ===========================================================================

TEST_F(OracleReplayTest, GrossPnlLongTrade) {
    // For a LONG trade: gross_pnl = (exit_price - entry_price) * contract_multiplier
    oracle_cfg.target_ticks = 4;
    oracle_cfg.stop_ticks = 2;
    oracle_cfg.volume_horizon = 200;

    OracleReplay replay(oracle_cfg, costs);
    auto bars = make_bar_series(4500.0f, 4502.0f, 30, 50);
    BacktestResult result = replay.run(bars);

    for (const auto& trade : result.trades) {
        float expected_gross = (trade.exit_price - trade.entry_price)
                                * trade.direction * costs.contract_multiplier;
        EXPECT_NEAR(trade.gross_pnl, expected_gross, 0.01f)
            << "Gross PnL mismatch for trade at bar " << trade.entry_bar_idx;
    }
}

// ===========================================================================
// 11. Net PnL = Gross PnL - round_trip_cost
// ===========================================================================

TEST_F(OracleReplayTest, NetPnlAccountsForCosts) {
    oracle_cfg.target_ticks = 4;
    oracle_cfg.stop_ticks = 2;
    oracle_cfg.volume_horizon = 200;

    OracleReplay replay(oracle_cfg, costs);
    auto bars = make_bar_series(4500.0f, 4502.0f, 30, 50);
    BacktestResult result = replay.run(bars);

    for (const auto& trade : result.trades) {
        // net_pnl should be less than gross_pnl (costs are always positive)
        EXPECT_LT(trade.net_pnl, trade.gross_pnl)
            << "Net PnL should be less than gross PnL after execution costs";
    }
}

// ===========================================================================
// 12. Commission applied per-side (2× per round-trip)
// ===========================================================================

TEST_F(OracleReplayTest, CommissionAppliedPerSide) {
    // Validation gate: "Commission is applied per-side (2× per round-trip)"
    float min_rt_commission = 2.0f * costs.commission_per_side;

    oracle_cfg.target_ticks = 4;
    oracle_cfg.stop_ticks = 2;
    oracle_cfg.volume_horizon = 200;

    OracleReplay replay(oracle_cfg, costs);
    auto bars = make_bar_series(4500.0f, 4502.0f, 30, 50);
    BacktestResult result = replay.run(bars);

    for (const auto& trade : result.trades) {
        float cost_deducted = trade.gross_pnl - trade.net_pnl;
        EXPECT_GE(cost_deducted, min_rt_commission)
            << "Round-trip cost must include at least 2× commission_per_side ($"
            << min_rt_commission << "), but deducted only $" << cost_deducted;
    }
}

// ===========================================================================
// 13. Every ENTER has a matching EXIT — no open positions
// ===========================================================================

TEST_F(OracleReplayTest, EveryEntryHasMatchingExit) {
    // Validation gate: "Every ENTER has a matching EXIT (no open positions at session end)"
    oracle_cfg.target_ticks = 4;
    oracle_cfg.stop_ticks = 2;
    oracle_cfg.volume_horizon = 200;

    OracleReplay replay(oracle_cfg, costs);
    auto bars = make_bar_series(4500.0f, 4501.0f, 40, 50);
    BacktestResult result = replay.run(bars);

    for (const auto& trade : result.trades) {
        EXPECT_GT(trade.exit_ts, trade.entry_ts)
            << "Exit timestamp must be after entry timestamp";
        EXPECT_GT(trade.exit_bar_idx, trade.entry_bar_idx)
            << "Exit bar index must be after entry bar index";
        EXPECT_GT(trade.exit_price, 0.0f)
            << "Exit price must be set";
    }
}

// ===========================================================================
// 14. Exit at session end (force-close)
// ===========================================================================

TEST_F(OracleReplayTest, ForceCloseAtSessionEnd) {
    // Validation gate: "Force-close at session end if oracle hasn't exited"
    // Use a flat price series so the oracle never hits target/stop within volume horizon,
    // and bars end (simulating session end).
    oracle_cfg.target_ticks = 40;  // very wide — never hit
    oracle_cfg.stop_ticks = 40;
    oracle_cfg.volume_horizon = 50;

    OracleReplay replay(oracle_cfg, costs);

    // Flat price — oracle generates HOLD; but if it enters, it shouldn't exit
    // via target/stop. We need a setup where it enters then session ends.
    // Small rise to trigger entry, then flat to end of bars.
    std::vector<float> deltas;
    // Rise 0.25 per bar for 5 bars (1.25 points = 5 ticks, would hit target with small config)
    // But target is 40 ticks — this won't trigger exit.
    for (int i = 0; i < 5; ++i) deltas.push_back(0.25f);
    for (int i = 0; i < 10; ++i) deltas.push_back(0.0f);  // flat
    auto bars = make_bar_path(4500.0f, deltas, 50);

    BacktestResult result = replay.run(bars);

    // Any trade that reaches the end of bars should have exit_reason=4 (session_end)
    for (const auto& trade : result.trades) {
        // If the trade's exit is at or near the last bar, it should be session_end
        if (trade.exit_bar_idx >= static_cast<int>(bars.size()) - 2) {
            EXPECT_EQ(trade.exit_reason, 4)
                << "Trade ending near last bar should have exit_reason=4 (session_end)";
        }
    }
}

// ===========================================================================
// 15. exit_reason correctly recorded
// ===========================================================================

TEST_F(OracleReplayTest, ExitReasonTargetHit) {
    // Price rises enough to hit target → exit_reason=0
    oracle_cfg.target_ticks = 4;
    oracle_cfg.stop_ticks = 20;  // wide stop so target is hit first
    oracle_cfg.volume_horizon = 300;

    OracleReplay replay(oracle_cfg, costs);
    // Steady rise: 1.5 points = 6 ticks > target of 4 ticks
    auto bars = make_bar_series(4500.0f, 4501.5f, 30, 50);
    BacktestResult result = replay.run(bars);

    bool found_target_exit = false;
    for (const auto& trade : result.trades) {
        if (trade.exit_reason == 0) {
            found_target_exit = true;
            // For LONG: exit_price > entry_price
            if (trade.direction == 1) {
                EXPECT_GT(trade.exit_price, trade.entry_price);
            }
        }
    }
    // At least one trade should have hit the target
    if (result.total_trades > 0) {
        EXPECT_TRUE(found_target_exit)
            << "With steady price rise and wide stop, at least one trade should exit on target";
    }
}

TEST_F(OracleReplayTest, ExitReasonStopHit) {
    // LONG trade with price falling to hit stop → exit_reason=1
    oracle_cfg.target_ticks = 20;  // wide target
    oracle_cfg.stop_ticks = 4;
    oracle_cfg.volume_horizon = 300;

    OracleReplay replay(oracle_cfg, costs);

    // Price rises slightly then falls sharply
    std::vector<float> deltas;
    for (int i = 0; i < 3; ++i) deltas.push_back(0.25f);   // up 0.75
    for (int i = 0; i < 15; ++i) deltas.push_back(-0.50f);  // down 7.5

    auto bars = make_bar_path(4500.0f, deltas, 50);
    BacktestResult result = replay.run(bars);

    bool found_stop_exit = false;
    for (const auto& trade : result.trades) {
        if (trade.exit_reason == 1) {
            found_stop_exit = true;
        }
    }
    if (result.total_trades > 0) {
        EXPECT_TRUE(found_stop_exit)
            << "With price reversal, at least one trade should exit on stop";
    }
}

// ===========================================================================
// 16. Safety cap (max_time_horizon_s)
// ===========================================================================

TEST_F(OracleReplayTest, SafetyCapTriggeredOnFlatPrice) {
    // Validation gate: "Safety cap triggers are logged with timestamps and counted"
    oracle_cfg.target_ticks = 100;  // never hit
    oracle_cfg.stop_ticks = 100;    // never hit
    oracle_cfg.volume_horizon = 10000;  // huge
    oracle_cfg.max_time_horizon_s = 5;  // 5 second cap

    OracleReplay replay(oracle_cfg, costs);

    // Flat price, many bars (each 1 second apart)
    auto bars = make_bar_series(4500.0f, 4500.0f, 50, 50);
    BacktestResult result = replay.run(bars);

    // The safety cap count should be reflected in the result
    // If no trades are triggered (HOLD), safety_cap_triggered_count may be 0
    // but if trades exist, those hitting the time limit should be counted.
    EXPECT_GE(result.safety_cap_triggered_count, 0);
    // safety_cap_fraction should be in [0, 1]
    EXPECT_GE(result.safety_cap_fraction, 0.0f);
    EXPECT_LE(result.safety_cap_fraction, 1.0f);
}

TEST_F(OracleReplayTest, SafetyCapExitReason) {
    // If safety cap triggers mid-trade, exit_reason should be 5
    oracle_cfg.target_ticks = 100;
    oracle_cfg.stop_ticks = 100;
    oracle_cfg.volume_horizon = 10000;
    oracle_cfg.max_time_horizon_s = 3;

    OracleReplay replay(oracle_cfg, costs);

    // Small rise to trigger entry, then flat (cap will trigger)
    std::vector<float> deltas;
    for (int i = 0; i < 3; ++i) deltas.push_back(0.25f);
    for (int i = 0; i < 20; ++i) deltas.push_back(0.0f);
    auto bars = make_bar_path(4500.0f, deltas, 50);

    BacktestResult result = replay.run(bars);

    for (const auto& trade : result.trades) {
        if (trade.exit_reason == 5) {
            EXPECT_EQ(result.safety_cap_triggered_count,
                      result.exit_reason_counts.count(5) > 0
                          ? result.exit_reason_counts.at(5)
                          : 0);
        }
    }
}

// ===========================================================================
// 17. No trades during warmup
// ===========================================================================

TEST_F(OracleReplayTest, NoTradesDuringWarmup) {
    // Validation gate: "No trades during first W bars (observation window warmup)"
    oracle_cfg.target_ticks = 2;
    oracle_cfg.stop_ticks = 1;
    oracle_cfg.volume_horizon = 50;

    OracleReplay replay(oracle_cfg, costs);
    // Big rise in first few bars to try to trick the oracle into early entry
    std::vector<float> deltas;
    for (int i = 0; i < 30; ++i) deltas.push_back(0.50f);
    auto bars = make_bar_path(4500.0f, deltas, 50);

    BacktestResult result = replay.run(bars);

    // No trade should have entry_bar_idx < some warmup window
    // The warmup window W is typically related to EWMA span (20 bars default from WarmupTracker)
    for (const auto& trade : result.trades) {
        EXPECT_GE(trade.entry_bar_idx, 1)
            << "No trades should occur at bar index 0 (minimum observation window)";
    }
}

// ===========================================================================
// 18. BacktestResult aggregate statistics
// ===========================================================================

TEST_F(OracleReplayTest, WinRateComputation) {
    oracle_cfg.target_ticks = 4;
    oracle_cfg.stop_ticks = 2;
    oracle_cfg.volume_horizon = 200;

    OracleReplay replay(oracle_cfg, costs);
    auto bars = make_bar_series(4500.0f, 4502.0f, 50, 50);
    BacktestResult result = replay.run(bars);

    if (result.total_trades > 0) {
        float expected_win_rate = static_cast<float>(result.winning_trades)
                                  / static_cast<float>(result.total_trades);
        EXPECT_FLOAT_EQ(result.win_rate, expected_win_rate);
    }
}

TEST_F(OracleReplayTest, WinningPlusLosingEqualsTotal) {
    oracle_cfg.target_ticks = 4;
    oracle_cfg.stop_ticks = 2;
    oracle_cfg.volume_horizon = 200;

    OracleReplay replay(oracle_cfg, costs);
    auto bars = make_bar_series(4500.0f, 4502.0f, 50, 50);
    BacktestResult result = replay.run(bars);

    EXPECT_EQ(result.winning_trades + result.losing_trades, result.total_trades);
}

TEST_F(OracleReplayTest, TradeCountConsistency) {
    oracle_cfg.target_ticks = 4;
    oracle_cfg.stop_ticks = 2;
    oracle_cfg.volume_horizon = 200;

    OracleReplay replay(oracle_cfg, costs);
    auto bars = make_bar_series(4500.0f, 4502.0f, 50, 50);
    BacktestResult result = replay.run(bars);

    EXPECT_EQ(result.total_trades, static_cast<int>(result.trades.size()));
}

TEST_F(OracleReplayTest, ProfitFactorComputation) {
    oracle_cfg.target_ticks = 4;
    oracle_cfg.stop_ticks = 2;
    oracle_cfg.volume_horizon = 200;

    OracleReplay replay(oracle_cfg, costs);

    // Mix of up and down moves to generate both wins and losses
    std::vector<float> deltas;
    for (int i = 0; i < 10; ++i) deltas.push_back(0.25f);
    for (int i = 0; i < 10; ++i) deltas.push_back(-0.25f);
    for (int i = 0; i < 10; ++i) deltas.push_back(0.25f);
    for (int i = 0; i < 10; ++i) deltas.push_back(-0.25f);
    auto bars = make_bar_path(4500.0f, deltas, 60);

    BacktestResult result = replay.run(bars);

    if (result.losing_trades > 0 && result.winning_trades > 0) {
        // profit_factor = gross_wins / gross_losses
        EXPECT_GT(result.profit_factor, 0.0f);
    }
}

TEST_F(OracleReplayTest, ExpectancyComputation) {
    oracle_cfg.target_ticks = 4;
    oracle_cfg.stop_ticks = 2;
    oracle_cfg.volume_horizon = 200;

    OracleReplay replay(oracle_cfg, costs);
    auto bars = make_bar_series(4500.0f, 4502.0f, 50, 50);
    BacktestResult result = replay.run(bars);

    if (result.total_trades > 0) {
        float expected_expectancy = result.net_pnl / static_cast<float>(result.total_trades);
        EXPECT_FLOAT_EQ(result.expectancy, expected_expectancy);
    }
}

// ===========================================================================
// 19. PnL by hour
// ===========================================================================

TEST_F(OracleReplayTest, PnlByHourPopulated) {
    oracle_cfg.target_ticks = 4;
    oracle_cfg.stop_ticks = 2;
    oracle_cfg.volume_horizon = 200;

    OracleReplay replay(oracle_cfg, costs);
    auto bars = make_bar_series(4500.0f, 4501.5f, 30, 50);
    BacktestResult result = replay.run(bars);

    if (result.total_trades > 0) {
        EXPECT_FALSE(result.pnl_by_hour.empty())
            << "pnl_by_hour should be populated when there are trades";
        // All hours should be in RTH range (9-15)
        for (const auto& [hour, pnl] : result.pnl_by_hour) {
            EXPECT_GE(hour, 9);
            EXPECT_LE(hour, 15);
        }
    }
}

// ===========================================================================
// 20. Exit reason counts
// ===========================================================================

TEST_F(OracleReplayTest, ExitReasonCountsSumToTotalTrades) {
    oracle_cfg.target_ticks = 4;
    oracle_cfg.stop_ticks = 2;
    oracle_cfg.volume_horizon = 200;

    OracleReplay replay(oracle_cfg, costs);
    auto bars = make_bar_series(4500.0f, 4501.5f, 40, 50);
    BacktestResult result = replay.run(bars);

    int exit_reason_total = 0;
    for (const auto& [reason, count] : result.exit_reason_counts) {
        exit_reason_total += count;
    }
    EXPECT_EQ(exit_reason_total, result.total_trades);
}

// ===========================================================================
// 21. Label counts
// ===========================================================================

TEST_F(OracleReplayTest, LabelCountsPopulated) {
    oracle_cfg.target_ticks = 4;
    oracle_cfg.stop_ticks = 2;
    oracle_cfg.volume_horizon = 200;

    OracleReplay replay(oracle_cfg, costs);
    auto bars = make_bar_series(4500.0f, 4501.5f, 40, 50);
    BacktestResult result = replay.run(bars);

    // label_counts should reflect oracle decisions (action → count)
    if (result.total_trades > 0) {
        EXPECT_FALSE(result.label_counts.empty());
    }
}

// ===========================================================================
// 22. Entry at mid_price
// ===========================================================================

TEST_F(OracleReplayTest, EntryAtMidPrice) {
    // Spec: "Entry/exit at mid_price. Spread cost accounts for bid-ask statistically."
    oracle_cfg.target_ticks = 4;
    oracle_cfg.stop_ticks = 2;
    oracle_cfg.volume_horizon = 200;

    OracleReplay replay(oracle_cfg, costs);
    auto bars = make_bar_series(4500.0f, 4502.0f, 30, 50);
    BacktestResult result = replay.run(bars);

    for (const auto& trade : result.trades) {
        // Entry price should match the mid_price of the entry bar
        int idx = trade.entry_bar_idx;
        if (idx >= 0 && idx < static_cast<int>(bars.size())) {
            EXPECT_FLOAT_EQ(trade.entry_price, bars[idx].close_mid)
                << "Entry price should be the mid_price of the entry bar";
        }
    }
}

// ===========================================================================
// 23. Bars held and duration
// ===========================================================================

TEST_F(OracleReplayTest, BarsHeldComputation) {
    oracle_cfg.target_ticks = 4;
    oracle_cfg.stop_ticks = 2;
    oracle_cfg.volume_horizon = 200;

    OracleReplay replay(oracle_cfg, costs);
    auto bars = make_bar_series(4500.0f, 4501.5f, 30, 50);
    BacktestResult result = replay.run(bars);

    for (const auto& trade : result.trades) {
        EXPECT_EQ(trade.bars_held, trade.exit_bar_idx - trade.entry_bar_idx)
            << "bars_held should equal exit_bar_idx - entry_bar_idx";
        EXPECT_GT(trade.bars_held, 0)
            << "Must hold for at least 1 bar";
    }
}

TEST_F(OracleReplayTest, DurationSeconds) {
    oracle_cfg.target_ticks = 4;
    oracle_cfg.stop_ticks = 2;
    oracle_cfg.volume_horizon = 200;

    OracleReplay replay(oracle_cfg, costs);
    auto bars = make_bar_series(4500.0f, 4501.5f, 30, 50);
    BacktestResult result = replay.run(bars);

    for (const auto& trade : result.trades) {
        float expected_duration = static_cast<float>(trade.exit_ts - trade.entry_ts)
                                  / 1.0e9f;
        EXPECT_NEAR(trade.duration_s, expected_duration, 0.01f)
            << "duration_s should be (exit_ts - entry_ts) in seconds";
    }
}

// ===========================================================================
// 24. Max drawdown
// ===========================================================================

TEST_F(OracleReplayTest, MaxDrawdownNonNegative) {
    oracle_cfg.target_ticks = 4;
    oracle_cfg.stop_ticks = 2;
    oracle_cfg.volume_horizon = 200;

    OracleReplay replay(oracle_cfg, costs);
    auto bars = make_bar_series(4500.0f, 4501.5f, 50, 50);
    BacktestResult result = replay.run(bars);

    EXPECT_GE(result.max_drawdown, 0.0f)
        << "Max drawdown should be non-negative (reported as a magnitude)";
}

// ===========================================================================
// 25. Volume horizon — oracle looks forward by volume, not bar count
// ===========================================================================

TEST_F(OracleReplayTest, VolumeHorizonAccumulatesBarVolumes) {
    // Oracle looks forward until cumulative volume >= volume_horizon.
    // With vol_per_bar=50 and volume_horizon=100, oracle should look 2 bars ahead.
    oracle_cfg.target_ticks = 2;
    oracle_cfg.stop_ticks = 1;
    oracle_cfg.volume_horizon = 100;

    OracleReplay replay(oracle_cfg, costs);

    // Price goes up by 0.25 per bar (1 tick). After 2 bars = 2 ticks = target hit.
    // If volume horizon is 100 and each bar has 50 volume, oracle sees 2 bars ahead.
    std::vector<float> deltas;
    for (int i = 0; i < 20; ++i) deltas.push_back(TICK);
    auto bars = make_bar_path(4500.0f, deltas, 30);

    BacktestResult result = replay.run(bars);
    // We just verify the engine runs and produces trades — the volume logic
    // determines how far forward the oracle looks.
    EXPECT_GT(result.total_trades, 0);
}

// ===========================================================================
// 26. Spread model EMPIRICAL uses bar spread data
// ===========================================================================

TEST_F(OracleReplayTest, EmpiricalSpreadUsesBarData) {
    // Validation gate: "Spread cost uses actual spread from bar data when spread_model=empirical"
    costs.spread_model = ExecutionCosts::SpreadModel::EMPIRICAL;

    oracle_cfg.target_ticks = 4;
    oracle_cfg.stop_ticks = 2;
    oracle_cfg.volume_horizon = 200;

    OracleReplay replay_emp(oracle_cfg, costs);

    // Bars with 2-tick spread (wider than default 1 tick)
    auto bars = make_bar_series(4500.0f, 4502.0f, 30, 50);
    for (auto& bar : bars) {
        bar.spread = 2.0f * TICK;  // 2 ticks wide
    }

    BacktestResult result_emp = replay_emp.run(bars);

    // Now compare with FIXED model
    costs.spread_model = ExecutionCosts::SpreadModel::FIXED;
    costs.fixed_spread_ticks = 1;
    OracleReplay replay_fixed(oracle_cfg, costs);
    // Same bars but fixed model should use fixed spread
    BacktestResult result_fixed = replay_fixed.run(bars);

    // With wider empirical spread, costs should be higher → net_pnl should be lower
    if (result_emp.total_trades > 0 && result_fixed.total_trades > 0) {
        // Empirical with 2-tick spread should produce worse net PnL than fixed 1-tick
        EXPECT_LT(result_emp.net_pnl, result_fixed.net_pnl)
            << "Empirical 2-tick spread should result in worse net PnL than fixed 1-tick";
    }
}

// ===========================================================================
// 27. Hold fraction
// ===========================================================================

TEST_F(OracleReplayTest, HoldFractionInRange) {
    oracle_cfg.target_ticks = 4;
    oracle_cfg.stop_ticks = 2;
    oracle_cfg.volume_horizon = 200;

    OracleReplay replay(oracle_cfg, costs);
    auto bars = make_bar_series(4500.0f, 4501.0f, 40, 50);
    BacktestResult result = replay.run(bars);

    EXPECT_GE(result.hold_fraction, 0.0f);
    EXPECT_LE(result.hold_fraction, 1.0f);
}

// ===========================================================================
// 28. Avg bars held / avg duration consistency
// ===========================================================================

TEST_F(OracleReplayTest, AvgBarsHeldConsistency) {
    oracle_cfg.target_ticks = 4;
    oracle_cfg.stop_ticks = 2;
    oracle_cfg.volume_horizon = 200;

    OracleReplay replay(oracle_cfg, costs);
    auto bars = make_bar_series(4500.0f, 4502.0f, 50, 50);
    BacktestResult result = replay.run(bars);

    if (result.total_trades > 0) {
        float sum_bars_held = 0.0f;
        for (const auto& trade : result.trades) {
            sum_bars_held += static_cast<float>(trade.bars_held);
        }
        float expected_avg = sum_bars_held / static_cast<float>(result.total_trades);
        EXPECT_NEAR(result.avg_bars_held, expected_avg, 0.01f);
    }
}

TEST_F(OracleReplayTest, AvgDurationConsistency) {
    oracle_cfg.target_ticks = 4;
    oracle_cfg.stop_ticks = 2;
    oracle_cfg.volume_horizon = 200;

    OracleReplay replay(oracle_cfg, costs);
    auto bars = make_bar_series(4500.0f, 4502.0f, 50, 50);
    BacktestResult result = replay.run(bars);

    if (result.total_trades > 0) {
        float sum_duration = 0.0f;
        for (const auto& trade : result.trades) {
            sum_duration += trade.duration_s;
        }
        float expected_avg = sum_duration / static_cast<float>(result.total_trades);
        EXPECT_NEAR(result.avg_duration_s, expected_avg, 0.01f);
    }
}
