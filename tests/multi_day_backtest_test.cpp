// multi_day_backtest_test.cpp — TDD RED phase tests for Multi-Day Oracle Backtest
// Spec: .kit/docs/multi-day-backtest.md
//
// Tests the multi-day backtest runner: day scheduling, instrument rollover
// handling, per-configuration execution, result aggregation across days,
// and JSON output with trade-level detail.
//
// Headers below do not exist yet — all tests must fail to compile.

#include <gtest/gtest.h>

// Headers that the implementation must provide:
#include "backtest/multi_day_runner.hpp"    // MultiDayRunner, DaySchedule, BacktestConfig
#include "backtest/rollover.hpp"            // RolloverCalendar, ContractSpec
#include "backtest/backtest_result_io.hpp"  // JSON serialization for BacktestResult

// Already-existing headers from prior phases:
#include "backtest/execution_costs.hpp"
#include "backtest/oracle_replay.hpp"
#include "backtest/trade_record.hpp"
#include "bars/bar.hpp"
#include "test_bar_helpers.hpp"

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <sstream>

// ===========================================================================
// Helpers
// ===========================================================================
namespace {

using test_helpers::TICK;
using test_helpers::NS_PER_SEC;
using test_helpers::NS_PER_HOUR;
using test_helpers::make_bar;
using test_helpers::make_bar_series;
using test_helpers::make_bar_path;

// Helper: build a synthetic single-day bar vector tagged with a date string.
// Uses make_bar_series internally, shifting timestamps to simulate different days.
std::vector<Bar> make_day_bars(float start_mid, float end_mid, int count,
                                uint32_t vol_per_bar, uint64_t day_offset_ns) {
    auto bars = make_bar_series(start_mid, end_mid, count, vol_per_bar);
    for (auto& bar : bars) {
        bar.open_ts += day_offset_ns;
        bar.close_ts += day_offset_ns;
    }
    return bars;
}

// YYYYMMDD-style integer date for test convenience.
constexpr int DATE_20220103 = 20220103;
constexpr int DATE_20220104 = 20220104;
constexpr int DATE_20220315 = 20220315;  // near rollover MESH2→MESM2
constexpr int DATE_20220316 = 20220316;
constexpr int DATE_20220317 = 20220317;
constexpr int DATE_20220318 = 20220318;  // rollover date
constexpr int DATE_20220614 = 20220614;
constexpr int DATE_20220615 = 20220615;
constexpr int DATE_20220616 = 20220616;
constexpr int DATE_20220617 = 20220617;  // rollover date
constexpr int DATE_20220701 = 20220701;  // OOS start
constexpr int DATE_20221230 = 20221230;

}  // namespace

// ===========================================================================
// 1. BacktestConfig — configuration struct for multi-day runner
// ===========================================================================
class BacktestConfigTest : public ::testing::Test {};

TEST_F(BacktestConfigTest, BarTypeField) {
    BacktestConfig cfg{};
    cfg.bar_type = "volume";
    EXPECT_EQ(cfg.bar_type, "volume");
}

TEST_F(BacktestConfigTest, BarParamField) {
    BacktestConfig cfg{};
    cfg.bar_param = 100.0;
    EXPECT_DOUBLE_EQ(cfg.bar_param, 100.0);
}

TEST_F(BacktestConfigTest, OracleConfigField) {
    BacktestConfig cfg{};
    cfg.oracle.target_ticks = 10;
    cfg.oracle.stop_ticks = 5;
    EXPECT_EQ(cfg.oracle.target_ticks, 10);
    EXPECT_EQ(cfg.oracle.stop_ticks, 5);
}

TEST_F(BacktestConfigTest, LabelMethodField) {
    BacktestConfig cfg{};
    cfg.oracle.label_method = OracleConfig::LabelMethod::FIRST_TO_HIT;
    EXPECT_EQ(cfg.oracle.label_method, OracleConfig::LabelMethod::FIRST_TO_HIT);
    cfg.oracle.label_method = OracleConfig::LabelMethod::TRIPLE_BARRIER;
    EXPECT_EQ(cfg.oracle.label_method, OracleConfig::LabelMethod::TRIPLE_BARRIER);
}

TEST_F(BacktestConfigTest, CostsField) {
    BacktestConfig cfg{};
    cfg.costs.commission_per_side = 0.62f;
    EXPECT_FLOAT_EQ(cfg.costs.commission_per_side, 0.62f);
}

TEST_F(BacktestConfigTest, DefaultOracleConfig) {
    // Spec default: target_ticks=10, stop_ticks=5, volume_horizon=500, take_profit_ticks=20
    BacktestConfig cfg{};
    EXPECT_EQ(cfg.oracle.target_ticks, 10);
    EXPECT_EQ(cfg.oracle.stop_ticks, 5);
    EXPECT_EQ(cfg.oracle.volume_horizon, 500u);
    EXPECT_EQ(cfg.oracle.take_profit_ticks, 20);
}

// ===========================================================================
// 2. ContractSpec — quarterly contract definitions
// ===========================================================================
class ContractSpecTest : public ::testing::Test {};

TEST_F(ContractSpecTest, HasSymbolAndInstrumentId) {
    ContractSpec spec{};
    spec.symbol = "MESH2";
    spec.instrument_id = 13614;
    EXPECT_EQ(spec.symbol, "MESH2");
    EXPECT_EQ(spec.instrument_id, 13614u);
}

TEST_F(ContractSpecTest, HasStartAndEndDate) {
    ContractSpec spec{};
    spec.start_date = DATE_20220103;
    spec.end_date = DATE_20220315;  // 3 days before rollover on 3/18
    EXPECT_EQ(spec.start_date, DATE_20220103);
    EXPECT_EQ(spec.end_date, DATE_20220315);
}

TEST_F(ContractSpecTest, HasRolloverDate) {
    ContractSpec spec{};
    spec.rollover_date = DATE_20220318;
    EXPECT_EQ(spec.rollover_date, DATE_20220318);
}

// ===========================================================================
// 3. RolloverCalendar — manages contract transitions
// ===========================================================================
class RolloverCalendarTest : public ::testing::Test {};

TEST_F(RolloverCalendarTest, DefaultConstructionEmpty) {
    RolloverCalendar cal{};
    EXPECT_TRUE(cal.contracts().empty());
}

TEST_F(RolloverCalendarTest, AddContract) {
    RolloverCalendar cal{};
    ContractSpec spec{};
    spec.symbol = "MESH2";
    spec.instrument_id = 13614;
    spec.rollover_date = DATE_20220318;
    cal.add_contract(spec);
    EXPECT_EQ(cal.contracts().size(), 1u);
}

TEST_F(RolloverCalendarTest, AddMultipleContracts) {
    RolloverCalendar cal{};

    ContractSpec mesh2{};
    mesh2.symbol = "MESH2";
    mesh2.instrument_id = 13614;
    mesh2.rollover_date = DATE_20220318;
    cal.add_contract(mesh2);

    ContractSpec mesm2{};
    mesm2.symbol = "MESM2";
    mesm2.instrument_id = 13615;
    mesm2.rollover_date = DATE_20220617;
    cal.add_contract(mesm2);

    EXPECT_EQ(cal.contracts().size(), 2u);
}

TEST_F(RolloverCalendarTest, IsExcludedDateReturnsTrueForRolloverWindow) {
    // Spec: "Exclude final 3 trading days before each rollover date"
    RolloverCalendar cal{};
    ContractSpec mesh2{};
    mesh2.symbol = "MESH2";
    mesh2.instrument_id = 13614;
    mesh2.rollover_date = DATE_20220318;
    cal.add_contract(mesh2);

    // March 15, 16, 17 are the 3 days before March 18 rollover — excluded
    EXPECT_TRUE(cal.is_excluded(DATE_20220315));
    EXPECT_TRUE(cal.is_excluded(DATE_20220316));
    EXPECT_TRUE(cal.is_excluded(DATE_20220317));

    // Rollover date itself — excluded
    EXPECT_TRUE(cal.is_excluded(DATE_20220318));
}

TEST_F(RolloverCalendarTest, IsExcludedDateReturnsFalseForNormalDays) {
    RolloverCalendar cal{};
    ContractSpec mesh2{};
    mesh2.symbol = "MESH2";
    mesh2.instrument_id = 13614;
    mesh2.rollover_date = DATE_20220318;
    cal.add_contract(mesh2);

    // Jan 3 is not near any rollover — should not be excluded
    EXPECT_FALSE(cal.is_excluded(DATE_20220103));
    EXPECT_FALSE(cal.is_excluded(DATE_20220104));
}

TEST_F(RolloverCalendarTest, GetContractForDateReturnsCorrectContract) {
    RolloverCalendar cal{};

    ContractSpec mesh2{};
    mesh2.symbol = "MESH2";
    mesh2.instrument_id = 13614;
    mesh2.start_date = DATE_20220103;
    mesh2.end_date = DATE_20220315;
    mesh2.rollover_date = DATE_20220318;
    cal.add_contract(mesh2);

    ContractSpec mesm2{};
    mesm2.symbol = "MESM2";
    mesm2.instrument_id = 13615;
    mesm2.start_date = DATE_20220318;
    mesm2.end_date = DATE_20220614;
    mesm2.rollover_date = DATE_20220617;
    cal.add_contract(mesm2);

    // Jan 3 should be in MESH2
    auto contract = cal.get_contract_for_date(DATE_20220103);
    EXPECT_TRUE(contract.has_value());
    EXPECT_EQ(contract->symbol, "MESH2");

    // July 1 should be in a later contract (MESU2)
    // This date is beyond MESM2, so without MESU2 added, should return empty
    auto contract_oos = cal.get_contract_for_date(DATE_20220701);
    EXPECT_FALSE(contract_oos.has_value());
}

TEST_F(RolloverCalendarTest, NoTradingAcrossRollovers) {
    // Spec: "Process each quarterly contract independently. Do not trade across rollovers."
    RolloverCalendar cal{};

    ContractSpec mesh2{};
    mesh2.symbol = "MESH2";
    mesh2.instrument_id = 13614;
    mesh2.start_date = DATE_20220103;
    mesh2.end_date = DATE_20220315;
    mesh2.rollover_date = DATE_20220318;
    cal.add_contract(mesh2);

    // Date on rollover: should be excluded from trading
    EXPECT_TRUE(cal.is_excluded(DATE_20220318));
}

TEST_F(RolloverCalendarTest, ExcludedDatesAreLogged) {
    // Spec: "Log excluded dates"
    RolloverCalendar cal{};

    ContractSpec mesh2{};
    mesh2.symbol = "MESH2";
    mesh2.instrument_id = 13614;
    mesh2.rollover_date = DATE_20220318;
    cal.add_contract(mesh2);

    auto excluded = cal.excluded_dates();
    EXPECT_FALSE(excluded.empty());
    // Should contain at least the 3 days before rollover + rollover day
    EXPECT_GE(excluded.size(), 4u);
}

// ===========================================================================
// 4. DaySchedule — what dates to process
// ===========================================================================
class DayScheduleTest : public ::testing::Test {};

TEST_F(DayScheduleTest, InSampleDateRange) {
    // Spec: "In-sample: Jan–Jun 2022"
    DaySchedule schedule{};
    schedule.in_sample_start = DATE_20220103;  // first trading day of 2022
    schedule.in_sample_end = 20220630;
    EXPECT_EQ(schedule.in_sample_start, DATE_20220103);
    EXPECT_EQ(schedule.in_sample_end, 20220630);
}

TEST_F(DayScheduleTest, OutOfSampleDateRange) {
    // Spec: "Out-of-sample: Jul–Dec 2022"
    DaySchedule schedule{};
    schedule.oos_start = DATE_20220701;
    schedule.oos_end = DATE_20221230;
    EXPECT_EQ(schedule.oos_start, DATE_20220701);
    EXPECT_EQ(schedule.oos_end, DATE_20221230);
}

TEST_F(DayScheduleTest, IsInSample) {
    DaySchedule schedule{};
    schedule.in_sample_start = DATE_20220103;
    schedule.in_sample_end = 20220630;
    schedule.oos_start = DATE_20220701;
    schedule.oos_end = DATE_20221230;

    EXPECT_TRUE(schedule.is_in_sample(DATE_20220103));
    EXPECT_TRUE(schedule.is_in_sample(20220630));
    EXPECT_FALSE(schedule.is_in_sample(DATE_20220701));
    EXPECT_FALSE(schedule.is_in_sample(DATE_20221230));
}

TEST_F(DayScheduleTest, IsOutOfSample) {
    DaySchedule schedule{};
    schedule.in_sample_start = DATE_20220103;
    schedule.in_sample_end = 20220630;
    schedule.oos_start = DATE_20220701;
    schedule.oos_end = DATE_20221230;

    EXPECT_FALSE(schedule.is_oos(DATE_20220103));
    EXPECT_TRUE(schedule.is_oos(DATE_20220701));
    EXPECT_TRUE(schedule.is_oos(DATE_20221230));
}

// ===========================================================================
// 5. MultiDayRunner — construction
// ===========================================================================
class MultiDayRunnerTest : public ::testing::Test {
protected:
    BacktestConfig config;
    RolloverCalendar calendar;
    DaySchedule schedule;

    void SetUp() override {
        config.bar_type = "volume";
        config.bar_param = 100.0;
        config.oracle.target_ticks = 10;
        config.oracle.stop_ticks = 5;
        config.oracle.volume_horizon = 500;
        config.oracle.take_profit_ticks = 20;
        config.oracle.label_method = OracleConfig::LabelMethod::FIRST_TO_HIT;
        config.costs = ExecutionCosts{};

        ContractSpec mesh2{};
        mesh2.symbol = "MESH2";
        mesh2.instrument_id = 13614;
        mesh2.start_date = DATE_20220103;
        mesh2.end_date = DATE_20220315;
        mesh2.rollover_date = DATE_20220318;
        calendar.add_contract(mesh2);

        schedule.in_sample_start = DATE_20220103;
        schedule.in_sample_end = 20220630;
        schedule.oos_start = DATE_20220701;
        schedule.oos_end = DATE_20221230;
    }
};

TEST_F(MultiDayRunnerTest, ConstructsWithConfigCalendarSchedule) {
    MultiDayRunner runner(config, calendar, schedule);
    (void)runner;  // compiles and doesn't throw
}

// ===========================================================================
// 6. MultiDayRunner — per-day result aggregation
// ===========================================================================

TEST_F(MultiDayRunnerTest, RunSingleDayReturnsDayResult) {
    // run_day processes a single day and returns a DayResult
    MultiDayRunner runner(config, calendar, schedule);

    auto bars = make_bar_series(4500.0f, 4502.0f, 50, 50);
    auto day_result = runner.run_day(DATE_20220103, bars);

    EXPECT_EQ(day_result.date, DATE_20220103);
    EXPECT_GE(day_result.result.total_trades, 0);
}

TEST_F(MultiDayRunnerTest, DayResultContainsBacktestResult) {
    MultiDayRunner runner(config, calendar, schedule);

    auto bars = make_bar_series(4500.0f, 4502.0f, 50, 50);
    auto day_result = runner.run_day(DATE_20220103, bars);

    // DayResult should contain a BacktestResult with trade-level detail
    EXPECT_EQ(day_result.result.total_trades,
              static_cast<int>(day_result.result.trades.size()));
}

TEST_F(MultiDayRunnerTest, SkipsExcludedDates) {
    // Spec: "Exclude final 3 trading days before each rollover date"
    MultiDayRunner runner(config, calendar, schedule);

    auto bars = make_bar_series(4500.0f, 4502.0f, 50, 50);

    // March 16 is excluded (near MESH2 rollover on March 18)
    auto day_result = runner.run_day(DATE_20220316, bars);
    EXPECT_TRUE(day_result.skipped);
    EXPECT_EQ(day_result.result.total_trades, 0);
}

TEST_F(MultiDayRunnerTest, SkippedDayHasReason) {
    MultiDayRunner runner(config, calendar, schedule);

    auto bars = make_bar_series(4500.0f, 4502.0f, 50, 50);
    auto day_result = runner.run_day(DATE_20220316, bars);

    EXPECT_TRUE(day_result.skipped);
    EXPECT_FALSE(day_result.skip_reason.empty());
}

// ===========================================================================
// 7. MultiDayRunner — multi-day aggregation
// ===========================================================================

TEST_F(MultiDayRunnerTest, AggregateResultCombinesDays) {
    MultiDayRunner runner(config, calendar, schedule);

    // Simulate two days of bars
    auto bars_day1 = make_bar_series(4500.0f, 4502.0f, 50, 50);
    auto bars_day2 = make_bar_series(4502.0f, 4504.0f, 50, 50);

    auto day1 = runner.run_day(DATE_20220103, bars_day1);
    auto day2 = runner.run_day(DATE_20220104, bars_day2);

    std::vector<DayResult> day_results = {day1, day2};
    auto agg = runner.aggregate(day_results);

    // Aggregate should sum trades from both days
    EXPECT_EQ(agg.total_trades, day1.result.total_trades + day2.result.total_trades);
}

TEST_F(MultiDayRunnerTest, AggregateNetPnlSumsAcrossDays) {
    MultiDayRunner runner(config, calendar, schedule);

    auto bars_day1 = make_bar_series(4500.0f, 4502.0f, 50, 50);
    auto bars_day2 = make_bar_series(4502.0f, 4504.0f, 50, 50);

    auto day1 = runner.run_day(DATE_20220103, bars_day1);
    auto day2 = runner.run_day(DATE_20220104, bars_day2);

    std::vector<DayResult> day_results = {day1, day2};
    auto agg = runner.aggregate(day_results);

    float expected_net = day1.result.net_pnl + day2.result.net_pnl;
    EXPECT_NEAR(agg.net_pnl, expected_net, 0.01f);
}

TEST_F(MultiDayRunnerTest, AggregateIncludesAllTradeRecords) {
    // Spec: "Per-config JSON: ... → BacktestResult + all TradeRecords"
    MultiDayRunner runner(config, calendar, schedule);

    auto bars_day1 = make_bar_series(4500.0f, 4502.0f, 50, 50);
    auto bars_day2 = make_bar_series(4502.0f, 4504.0f, 50, 50);

    auto day1 = runner.run_day(DATE_20220103, bars_day1);
    auto day2 = runner.run_day(DATE_20220104, bars_day2);

    std::vector<DayResult> day_results = {day1, day2};
    auto agg = runner.aggregate(day_results);

    int expected_trade_count = static_cast<int>(day1.result.trades.size())
                               + static_cast<int>(day2.result.trades.size());
    EXPECT_EQ(static_cast<int>(agg.trades.size()), expected_trade_count);
}

TEST_F(MultiDayRunnerTest, AggregateSkipsExcludedDays) {
    MultiDayRunner runner(config, calendar, schedule);

    auto bars_day1 = make_bar_series(4500.0f, 4502.0f, 50, 50);
    auto bars_excluded = make_bar_series(4502.0f, 4504.0f, 50, 50);

    auto day1 = runner.run_day(DATE_20220103, bars_day1);
    auto excluded = runner.run_day(DATE_20220316, bars_excluded);  // excluded

    std::vector<DayResult> day_results = {day1, excluded};
    auto agg = runner.aggregate(day_results);

    // Only day1 trades should be in the aggregate (excluded day has 0 trades)
    EXPECT_EQ(agg.total_trades, day1.result.total_trades);
}

// ===========================================================================
// 8. Trades-per-day metric
// ===========================================================================

TEST_F(MultiDayRunnerTest, TradesPerDayAverage) {
    // Spec: "Trade count > 10 per day avg — Statistical validity"
    MultiDayRunner runner(config, calendar, schedule);

    auto bars_day1 = make_bar_series(4500.0f, 4502.0f, 50, 50);
    auto bars_day2 = make_bar_series(4502.0f, 4504.0f, 50, 50);

    auto day1 = runner.run_day(DATE_20220103, bars_day1);
    auto day2 = runner.run_day(DATE_20220104, bars_day2);

    std::vector<DayResult> day_results = {day1, day2};
    auto agg = runner.aggregate(day_results);

    // trades_per_day should be total_trades / number of non-skipped days
    int active_days = 0;
    for (const auto& d : day_results) {
        if (!d.skipped) active_days++;
    }
    if (active_days > 0) {
        float expected_tpd = static_cast<float>(agg.total_trades)
                             / static_cast<float>(active_days);
        EXPECT_NEAR(agg.trades_per_day, expected_tpd, 0.01f);
    }
}

// ===========================================================================
// 9. In-sample / Out-of-sample split
// ===========================================================================

TEST_F(MultiDayRunnerTest, SplitResultsByInSampleAndOOS) {
    // Spec: "In-sample: Jan-Jun 2022, Out-of-sample: Jul-Dec 2022"
    MultiDayRunner runner(config, calendar, schedule);

    auto bars_is = make_bar_series(4500.0f, 4502.0f, 50, 50);
    auto bars_oos = make_bar_series(4510.0f, 4512.0f, 50, 50);

    auto day_is = runner.run_day(DATE_20220103, bars_is);    // IS
    auto day_oos = runner.run_day(DATE_20220701, bars_oos);  // OOS

    std::vector<DayResult> day_results = {day_is, day_oos};
    auto split = runner.split_results(day_results);

    EXPECT_FALSE(split.in_sample.empty());
    EXPECT_FALSE(split.oos.empty());

    // IS day should be in in_sample
    EXPECT_EQ(split.in_sample.size(), 1u);
    EXPECT_EQ(split.in_sample[0].date, DATE_20220103);

    // OOS day should be in oos
    EXPECT_EQ(split.oos.size(), 1u);
    EXPECT_EQ(split.oos[0].date, DATE_20220701);
}

TEST_F(MultiDayRunnerTest, OOSNetPnlComputed) {
    // Spec: "OOS net PnL > 0 — In-sample could be overfit"
    MultiDayRunner runner(config, calendar, schedule);

    auto bars_oos = make_bar_series(4510.0f, 4512.0f, 50, 50);
    auto day_oos = runner.run_day(DATE_20220701, bars_oos);

    std::vector<DayResult> oos_results = {day_oos};
    auto agg_oos = runner.aggregate(oos_results);

    // The aggregate should have a well-defined net_pnl
    // (no assertion on sign — that's the go/no-go check, not the aggregation logic)
    EXPECT_TRUE(std::isfinite(agg_oos.net_pnl));
}

// ===========================================================================
// 10. MultiDayRunner — bar type configurations from spec
// ===========================================================================

TEST_F(MultiDayRunnerTest, VolumeBarConfigurations) {
    // Spec: "volume V ∈ {50, 100, 200}"
    for (int v : {50, 100, 200}) {
        config.bar_type = "volume";
        config.bar_param = static_cast<double>(v);
        MultiDayRunner runner(config, calendar, schedule);
        (void)runner;  // just verifies construction with each config
    }
}

TEST_F(MultiDayRunnerTest, TickBarConfigurations) {
    // Spec: "tick K ∈ {25, 50, 100}"
    for (int k : {25, 50, 100}) {
        config.bar_type = "tick";
        config.bar_param = static_cast<double>(k);
        MultiDayRunner runner(config, calendar, schedule);
        (void)runner;
    }
}

TEST_F(MultiDayRunnerTest, TimeBarConfigurations) {
    // Spec: "time ∈ {1s, 5s, 60s}"
    for (double s : {1.0, 5.0, 60.0}) {
        config.bar_type = "time";
        config.bar_param = s;
        MultiDayRunner runner(config, calendar, schedule);
        (void)runner;
    }
}

// ===========================================================================
// 11. No trades on rollover exclusion days (validation gate)
// ===========================================================================

TEST_F(MultiDayRunnerTest, NoTradesOnRolloverDay) {
    // Spec: "Assert: Instrument rollover handled correctly (no trades on excluded rollover days)"
    MultiDayRunner runner(config, calendar, schedule);

    auto bars = make_bar_series(4500.0f, 4502.0f, 50, 50);

    // March 18 is a rollover date
    auto result = runner.run_day(DATE_20220318, bars);
    EXPECT_TRUE(result.skipped);
    EXPECT_EQ(result.result.total_trades, 0);
}

TEST_F(MultiDayRunnerTest, NoTradesOnThreeDaysBeforeRollover) {
    MultiDayRunner runner(config, calendar, schedule);

    auto bars = make_bar_series(4500.0f, 4502.0f, 50, 50);

    for (int date : {DATE_20220315, DATE_20220316, DATE_20220317}) {
        auto result = runner.run_day(date, bars);
        EXPECT_TRUE(result.skipped)
            << "Date " << date << " should be excluded (3 days before rollover)";
        EXPECT_EQ(result.result.total_trades, 0);
    }
}

// ===========================================================================
// 12. MultiDayRunner — empty day handling
// ===========================================================================

TEST_F(MultiDayRunnerTest, EmptyBarSequenceForDayReturnsZeroTrades) {
    MultiDayRunner runner(config, calendar, schedule);
    std::vector<Bar> empty_bars;
    auto result = runner.run_day(DATE_20220103, empty_bars);
    EXPECT_EQ(result.result.total_trades, 0);
    EXPECT_FALSE(result.skipped);  // not skipped — just no data
}

TEST_F(MultiDayRunnerTest, SingleBarDayReturnsZeroTrades) {
    MultiDayRunner runner(config, calendar, schedule);
    std::vector<Bar> one_bar = {make_bar(4500.0f, 100, 0)};
    auto result = runner.run_day(DATE_20220103, one_bar);
    EXPECT_EQ(result.result.total_trades, 0);
}

// ===========================================================================
// 13. DayResult — metadata fields
// ===========================================================================

TEST_F(MultiDayRunnerTest, DayResultHasContractSymbol) {
    MultiDayRunner runner(config, calendar, schedule);

    auto bars = make_bar_series(4500.0f, 4502.0f, 50, 50);
    auto day_result = runner.run_day(DATE_20220103, bars);

    // Should record which contract was used for this day
    EXPECT_EQ(day_result.contract_symbol, "MESH2");
}

TEST_F(MultiDayRunnerTest, DayResultHasInstrumentId) {
    MultiDayRunner runner(config, calendar, schedule);

    auto bars = make_bar_series(4500.0f, 4502.0f, 50, 50);
    auto day_result = runner.run_day(DATE_20220103, bars);

    EXPECT_EQ(day_result.instrument_id, 13614u);
}

TEST_F(MultiDayRunnerTest, DayResultHasBarCount) {
    MultiDayRunner runner(config, calendar, schedule);

    auto bars = make_bar_series(4500.0f, 4502.0f, 50, 50);
    auto day_result = runner.run_day(DATE_20220103, bars);

    EXPECT_EQ(day_result.bar_count, 50);
}

// ===========================================================================
// 14. JSON serialization of results
// ===========================================================================
class BacktestResultIOTest : public ::testing::Test {};

TEST_F(BacktestResultIOTest, SerializeBacktestResultToJson) {
    // Spec: "Per-config JSON: bar_type, bar_param, oracle_config, label_method
    //        → BacktestResult + all TradeRecords"
    BacktestResult result{};
    result.total_trades = 5;
    result.net_pnl = 100.0f;

    std::string json = backtest_io::to_json(result);
    EXPECT_FALSE(json.empty());
    // Should contain key fields
    EXPECT_NE(json.find("total_trades"), std::string::npos);
    EXPECT_NE(json.find("net_pnl"), std::string::npos);
}

TEST_F(BacktestResultIOTest, SerializeIncludesTradeRecords) {
    // Spec: "Results written to JSON with full trade-level detail including exit_reason"
    BacktestResult result{};
    TradeRecord trade{};
    trade.entry_price = 4500.0f;
    trade.exit_price = 4502.0f;
    trade.direction = 1;
    trade.gross_pnl = 10.0f;
    trade.net_pnl = 8.0f;
    trade.exit_reason = exit_reason::TARGET;
    result.trades.push_back(trade);
    result.total_trades = 1;

    std::string json = backtest_io::to_json(result);
    EXPECT_NE(json.find("trades"), std::string::npos);
    EXPECT_NE(json.find("exit_reason"), std::string::npos);
    EXPECT_NE(json.find("entry_price"), std::string::npos);
}

TEST_F(BacktestResultIOTest, SerializeIncludesConfigMetadata) {
    // Spec: "bar_type, bar_param, oracle_config, label_method"
    BacktestConfig cfg{};
    cfg.bar_type = "volume";
    cfg.bar_param = 100.0;
    cfg.oracle.label_method = OracleConfig::LabelMethod::FIRST_TO_HIT;

    BacktestResult result{};
    result.total_trades = 0;

    std::string json = backtest_io::to_json(result, cfg);
    EXPECT_NE(json.find("bar_type"), std::string::npos);
    EXPECT_NE(json.find("volume"), std::string::npos);
    EXPECT_NE(json.find("label_method"), std::string::npos);
}

TEST_F(BacktestResultIOTest, SerializeDayResultsVector) {
    // Should be able to serialize a vector of DayResults
    std::vector<DayResult> days;
    DayResult d{};
    d.date = DATE_20220103;
    d.result.total_trades = 3;
    d.skipped = false;
    days.push_back(d);

    std::string json = backtest_io::to_json(days);
    EXPECT_NE(json.find("20220103"), std::string::npos);
}

// ===========================================================================
// 15. Aggregate — max drawdown across days
// ===========================================================================

TEST_F(MultiDayRunnerTest, AggregateMaxDrawdownRecomputed) {
    // Max drawdown across multi-day should be computed from the full equity curve
    MultiDayRunner runner(config, calendar, schedule);

    auto bars_day1 = make_bar_series(4500.0f, 4502.0f, 50, 50);
    auto bars_day2 = make_bar_series(4502.0f, 4500.0f, 50, 50);  // down day

    auto day1 = runner.run_day(DATE_20220103, bars_day1);
    auto day2 = runner.run_day(DATE_20220104, bars_day2);

    std::vector<DayResult> day_results = {day1, day2};
    auto agg = runner.aggregate(day_results);

    // Max drawdown should be non-negative
    EXPECT_GE(agg.max_drawdown, 0.0f);
}

TEST_F(MultiDayRunnerTest, AggregateWinRateRecomputed) {
    MultiDayRunner runner(config, calendar, schedule);

    auto bars_day1 = make_bar_series(4500.0f, 4502.0f, 50, 50);
    auto bars_day2 = make_bar_series(4502.0f, 4504.0f, 50, 50);

    auto day1 = runner.run_day(DATE_20220103, bars_day1);
    auto day2 = runner.run_day(DATE_20220104, bars_day2);

    std::vector<DayResult> day_results = {day1, day2};
    auto agg = runner.aggregate(day_results);

    if (agg.total_trades > 0) {
        // win_rate should be recomputed from combined trades
        float expected = static_cast<float>(agg.winning_trades)
                         / static_cast<float>(agg.total_trades);
        EXPECT_FLOAT_EQ(agg.win_rate, expected);
    }
}

TEST_F(MultiDayRunnerTest, AggregateSharpeRecomputed) {
    MultiDayRunner runner(config, calendar, schedule);

    auto bars_day1 = make_bar_series(4500.0f, 4502.0f, 50, 50);
    auto bars_day2 = make_bar_series(4502.0f, 4504.0f, 50, 50);

    auto day1 = runner.run_day(DATE_20220103, bars_day1);
    auto day2 = runner.run_day(DATE_20220104, bars_day2);

    std::vector<DayResult> day_results = {day1, day2};
    auto agg = runner.aggregate(day_results);

    // Sharpe should be finite (not NaN/Inf)
    EXPECT_TRUE(std::isfinite(agg.sharpe));
}

TEST_F(MultiDayRunnerTest, AggregateProfitFactorRecomputed) {
    MultiDayRunner runner(config, calendar, schedule);

    // One up day, one down day to generate both wins and losses
    auto bars_day1 = make_bar_series(4500.0f, 4502.0f, 50, 50);
    auto bars_day2 = make_bar_series(4502.0f, 4500.0f, 50, 50);

    auto day1 = runner.run_day(DATE_20220103, bars_day1);
    auto day2 = runner.run_day(DATE_20220104, bars_day2);

    std::vector<DayResult> day_results = {day1, day2};
    auto agg = runner.aggregate(day_results);

    // profit_factor should be non-negative
    EXPECT_GE(agg.profit_factor, 0.0f);
}

// ===========================================================================
// 16. MultiDayRunner — both labeling methods on identical bars
// ===========================================================================

TEST_F(MultiDayRunnerTest, BothLabelMethodsOnIdenticalBars) {
    // Spec: "Assert: Both labeling methods (first-to-hit, triple barrier) run on identical bar sequences"
    auto bars = make_bar_series(4500.0f, 4502.0f, 50, 50);

    // Run with FIRST_TO_HIT
    config.oracle.label_method = OracleConfig::LabelMethod::FIRST_TO_HIT;
    MultiDayRunner runner_fth(config, calendar, schedule);
    auto result_fth = runner_fth.run_day(DATE_20220103, bars);

    // Run with TRIPLE_BARRIER
    config.oracle.label_method = OracleConfig::LabelMethod::TRIPLE_BARRIER;
    MultiDayRunner runner_tb(config, calendar, schedule);
    auto result_tb = runner_tb.run_day(DATE_20220103, bars);

    // Both ran on the same bars — bar_count should match
    EXPECT_EQ(result_fth.bar_count, result_tb.bar_count);

    // Both should produce valid (possibly different) results
    EXPECT_GE(result_fth.result.total_trades, 0);
    EXPECT_GE(result_tb.result.total_trades, 0);
}

// ===========================================================================
// 17. Safety cap trigger rate validation
// ===========================================================================

TEST_F(MultiDayRunnerTest, SafetyCapTriggerRateTracked) {
    // Spec: "Assert: Safety cap trigger rate < 1% during RTH"
    MultiDayRunner runner(config, calendar, schedule);

    auto bars = make_bar_series(4500.0f, 4502.0f, 50, 50);
    auto day_result = runner.run_day(DATE_20220103, bars);

    // safety_cap_fraction should be available in the day result
    EXPECT_GE(day_result.result.safety_cap_fraction, 0.0f);
    EXPECT_LE(day_result.result.safety_cap_fraction, 1.0f);
}

// ===========================================================================
// 18. Aggregate — per-day PnL tracking for summary
// ===========================================================================

TEST_F(MultiDayRunnerTest, AggregateHasDailyPnlVector) {
    MultiDayRunner runner(config, calendar, schedule);

    auto bars_day1 = make_bar_series(4500.0f, 4502.0f, 50, 50);
    auto bars_day2 = make_bar_series(4502.0f, 4504.0f, 50, 50);

    auto day1 = runner.run_day(DATE_20220103, bars_day1);
    auto day2 = runner.run_day(DATE_20220104, bars_day2);

    std::vector<DayResult> day_results = {day1, day2};
    auto agg = runner.aggregate(day_results);

    // Should track daily PnL for downstream analysis
    EXPECT_EQ(agg.daily_pnl.size(), 2u);
}

// ===========================================================================
// 19. Aggregate — expectancy recomputed from all trades
// ===========================================================================

TEST_F(MultiDayRunnerTest, AggregateExpectancyRecomputed) {
    MultiDayRunner runner(config, calendar, schedule);

    auto bars_day1 = make_bar_series(4500.0f, 4502.0f, 50, 50);
    auto bars_day2 = make_bar_series(4502.0f, 4504.0f, 50, 50);

    auto day1 = runner.run_day(DATE_20220103, bars_day1);
    auto day2 = runner.run_day(DATE_20220104, bars_day2);

    std::vector<DayResult> day_results = {day1, day2};
    auto agg = runner.aggregate(day_results);

    if (agg.total_trades > 0) {
        float expected_exp = agg.net_pnl / static_cast<float>(agg.total_trades);
        EXPECT_NEAR(agg.expectancy, expected_exp, 0.01f);
    }
}
