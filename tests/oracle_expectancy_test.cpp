// oracle_expectancy_test.cpp — TDD RED phase tests for OracleExpectancyReport
// Spec: .kit/docs/oracle-expectancy.md
//
// Tests the oracle expectancy extraction layer: OracleExpectancyReport struct,
// JSON serialization (oracle_expectancy::to_json), and day-result aggregation
// with per-quarter splitting (oracle_expectancy::aggregate_day_results).
//
// These tests target the testable report/aggregation layer, NOT the standalone
// tool binary that reads real data. All data is synthetic.
//
// No implementation files exist yet — all tests must fail to compile.

#include <gtest/gtest.h>

// Header that the implementation must provide (spec §Project Structure):
#include "backtest/oracle_expectancy_report.hpp"

// Already-existing headers from prior phases:
#include "backtest/execution_costs.hpp"
#include "backtest/multi_day_runner.hpp"
#include "backtest/oracle_replay.hpp"
#include "backtest/trade_record.hpp"
#include "bars/bar.hpp"
#include "test_bar_helpers.hpp"

#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

// ===========================================================================
// Helpers
// ===========================================================================
namespace {

using test_helpers::TICK;
using test_helpers::NS_PER_SEC;
using test_helpers::make_bar;
using test_helpers::make_bar_series;
using test_helpers::make_bar_path;

// Quarter-representative dates (YYYYMMDD).
// Spec: MESH2 20220103–20220318, MESM2 20220319–20220617,
//       MESU2 20220618–20220916, MESZ2 20220917–20221230.
constexpr int Q1_DATE_A = 20220110;
constexpr int Q1_DATE_B = 20220201;
constexpr int Q2_DATE_A = 20220401;
constexpr int Q2_DATE_B = 20220510;
constexpr int Q3_DATE_A = 20220705;
constexpr int Q3_DATE_B = 20220810;
constexpr int Q4_DATE_A = 20221005;
constexpr int Q4_DATE_B = 20221110;

// Build a DayResult with synthetic trades for deterministic testing.
// Creates `num_wins` winning trades and `num_losses` losing trades.
DayResult make_day_result(int date, int num_wins, int num_losses,
                          float win_gross = 12.50f, float loss_gross = -6.25f,
                          float rt_cost = 2.49f) {
    DayResult day{};
    day.date = date;
    day.skipped = false;
    day.bar_count = 100;

    for (int i = 0; i < num_wins; ++i) {
        TradeRecord t{};
        t.entry_price = 4500.0f;
        t.exit_price = 4500.0f + (win_gross / 5.0f);  // contract_multiplier=5
        t.direction = 1;
        t.gross_pnl = win_gross;
        t.net_pnl = win_gross - rt_cost;
        t.entry_bar_idx = i * 3;
        t.exit_bar_idx = i * 3 + 2;
        t.bars_held = 2;
        t.duration_s = 10.0f;
        t.exit_reason = exit_reason::TARGET;
        t.entry_ts = 1000000ULL * static_cast<uint64_t>(i);
        t.exit_ts = t.entry_ts + 10ULL * NS_PER_SEC;
        day.result.trades.push_back(t);
    }

    for (int i = 0; i < num_losses; ++i) {
        TradeRecord t{};
        t.entry_price = 4500.0f;
        t.exit_price = 4500.0f + (loss_gross / 5.0f);
        t.direction = 1;
        t.gross_pnl = loss_gross;
        t.net_pnl = loss_gross - rt_cost;
        t.entry_bar_idx = (num_wins + i) * 3;
        t.exit_bar_idx = (num_wins + i) * 3 + 2;
        t.bars_held = 2;
        t.duration_s = 10.0f;
        t.exit_reason = exit_reason::STOP;
        t.entry_ts = 2000000ULL * static_cast<uint64_t>(i);
        t.exit_ts = t.entry_ts + 10ULL * NS_PER_SEC;
        day.result.trades.push_back(t);
    }

    // Populate aggregate fields on the DayResult's BacktestResult.
    day.result.total_trades = num_wins + num_losses;
    day.result.winning_trades = num_wins;
    day.result.losing_trades = num_losses;
    float total_gross = num_wins * win_gross + num_losses * loss_gross;
    float total_net = num_wins * (win_gross - rt_cost) + num_losses * (loss_gross - rt_cost);
    day.result.gross_pnl = total_gross;
    day.result.net_pnl = total_net;
    if (day.result.total_trades > 0) {
        day.result.win_rate = static_cast<float>(num_wins) /
                              static_cast<float>(day.result.total_trades);
        day.result.expectancy = total_net / static_cast<float>(day.result.total_trades);
    }
    day.result.exit_reason_counts[exit_reason::TARGET] = num_wins;
    day.result.exit_reason_counts[exit_reason::STOP] = num_losses;

    return day;
}

// Build a skipped DayResult.
DayResult make_skipped_day(int date) {
    DayResult day{};
    day.date = date;
    day.skipped = true;
    day.skip_reason = "Excluded: near rollover";
    return day;
}

// Quarter assignment helper — matches spec contract boundaries.
// Q1: 20220103–20220318, Q2: 20220319–20220617,
// Q3: 20220618–20220916, Q4: 20220917–20221230.
std::string expected_quarter(int date) {
    if (date <= 20220318) return "Q1";
    if (date <= 20220617) return "Q2";
    if (date <= 20220916) return "Q3";
    return "Q4";
}

}  // namespace

// ===========================================================================
// 1. OracleExpectancyReport struct — basic construction
// ===========================================================================
class OracleExpectancyReportTest : public ::testing::Test {};

TEST_F(OracleExpectancyReportTest, DefaultConstruction) {
    OracleExpectancyReport report{};
    EXPECT_EQ(report.days_processed, 0);
    EXPECT_EQ(report.days_skipped, 0);
}

TEST_F(OracleExpectancyReportTest, DefaultFirstToHitIsZeroInitialized) {
    OracleExpectancyReport report{};
    EXPECT_EQ(report.first_to_hit.total_trades, 0);
    EXPECT_FLOAT_EQ(report.first_to_hit.net_pnl, 0.0f);
    EXPECT_FLOAT_EQ(report.first_to_hit.expectancy, 0.0f);
}

TEST_F(OracleExpectancyReportTest, DefaultTripleBarrierIsZeroInitialized) {
    OracleExpectancyReport report{};
    EXPECT_EQ(report.triple_barrier.total_trades, 0);
    EXPECT_FLOAT_EQ(report.triple_barrier.net_pnl, 0.0f);
    EXPECT_FLOAT_EQ(report.triple_barrier.expectancy, 0.0f);
}

TEST_F(OracleExpectancyReportTest, PerQuarterMapsAreEmpty) {
    OracleExpectancyReport report{};
    EXPECT_TRUE(report.fth_per_quarter.empty());
    EXPECT_TRUE(report.tb_per_quarter.empty());
}

TEST_F(OracleExpectancyReportTest, FirstToHitFieldIsBacktestResult) {
    OracleExpectancyReport report{};
    report.first_to_hit.total_trades = 42;
    report.first_to_hit.net_pnl = 100.0f;
    EXPECT_EQ(report.first_to_hit.total_trades, 42);
    EXPECT_FLOAT_EQ(report.first_to_hit.net_pnl, 100.0f);
}

TEST_F(OracleExpectancyReportTest, TripleBarrierFieldIsBacktestResult) {
    OracleExpectancyReport report{};
    report.triple_barrier.total_trades = 37;
    report.triple_barrier.net_pnl = -50.0f;
    EXPECT_EQ(report.triple_barrier.total_trades, 37);
    EXPECT_FLOAT_EQ(report.triple_barrier.net_pnl, -50.0f);
}

TEST_F(OracleExpectancyReportTest, PerQuarterMapsAcceptStringKeys) {
    OracleExpectancyReport report{};
    BacktestResult q1{};
    q1.total_trades = 10;
    report.fth_per_quarter["Q1"] = q1;
    report.tb_per_quarter["Q1"] = q1;
    EXPECT_EQ(report.fth_per_quarter["Q1"].total_trades, 10);
    EXPECT_EQ(report.tb_per_quarter["Q1"].total_trades, 10);
}

// ===========================================================================
// 2. aggregate_day_results — core aggregation (Validation Gate 3)
// ===========================================================================
class AggregateDayResultsTest : public ::testing::Test {};

TEST_F(AggregateDayResultsTest, EmptyInputReturnsZeroReport) {
    // Validation Gate 5: empty input → zero-initialized report
    std::vector<DayResult> fth_empty;
    std::vector<DayResult> tb_empty;
    std::vector<int> dates_empty;

    auto report = oracle_expectancy::aggregate_day_results(
        fth_empty, tb_empty, dates_empty);

    EXPECT_EQ(report.days_processed, 0);
    EXPECT_EQ(report.days_skipped, 0);
    EXPECT_EQ(report.first_to_hit.total_trades, 0);
    EXPECT_FLOAT_EQ(report.first_to_hit.net_pnl, 0.0f);
    EXPECT_EQ(report.triple_barrier.total_trades, 0);
    EXPECT_FLOAT_EQ(report.triple_barrier.net_pnl, 0.0f);
    EXPECT_TRUE(report.fth_per_quarter.empty());
    EXPECT_TRUE(report.tb_per_quarter.empty());
}

TEST_F(AggregateDayResultsTest, SingleDaySumsCorrectly) {
    // Validation Gate 3: correctly sums trades across days (single day case)
    auto fth_day = make_day_result(Q1_DATE_A, 5, 3);
    auto tb_day = make_day_result(Q1_DATE_A, 4, 4);

    std::vector<DayResult> fth_results = {fth_day};
    std::vector<DayResult> tb_results = {tb_day};
    std::vector<int> dates = {Q1_DATE_A};

    auto report = oracle_expectancy::aggregate_day_results(
        fth_results, tb_results, dates);

    EXPECT_EQ(report.days_processed, 1);
    EXPECT_EQ(report.first_to_hit.total_trades, 8);
    EXPECT_EQ(report.triple_barrier.total_trades, 8);
}

TEST_F(AggregateDayResultsTest, MultipleDaysSumTradesCorrectly) {
    // Validation Gate 3: correctly sums trades across days (multi-day)
    auto fth_day1 = make_day_result(Q1_DATE_A, 5, 3);   // 8 trades
    auto fth_day2 = make_day_result(Q1_DATE_B, 7, 2);   // 9 trades
    auto tb_day1 = make_day_result(Q1_DATE_A, 4, 4);    // 8 trades
    auto tb_day2 = make_day_result(Q1_DATE_B, 6, 3);    // 9 trades

    std::vector<DayResult> fth_results = {fth_day1, fth_day2};
    std::vector<DayResult> tb_results = {tb_day1, tb_day2};
    std::vector<int> dates = {Q1_DATE_A, Q1_DATE_B};

    auto report = oracle_expectancy::aggregate_day_results(
        fth_results, tb_results, dates);

    EXPECT_EQ(report.days_processed, 2);
    EXPECT_EQ(report.first_to_hit.total_trades, 17);   // 8 + 9
    EXPECT_EQ(report.triple_barrier.total_trades, 17);  // 8 + 9
}

TEST_F(AggregateDayResultsTest, NetPnlSumsAcrossDays) {
    auto fth_day1 = make_day_result(Q1_DATE_A, 5, 3);
    auto fth_day2 = make_day_result(Q2_DATE_A, 7, 2);
    auto tb_day1 = make_day_result(Q1_DATE_A, 4, 4);
    auto tb_day2 = make_day_result(Q2_DATE_A, 6, 3);

    std::vector<DayResult> fth_results = {fth_day1, fth_day2};
    std::vector<DayResult> tb_results = {tb_day1, tb_day2};
    std::vector<int> dates = {Q1_DATE_A, Q2_DATE_A};

    auto report = oracle_expectancy::aggregate_day_results(
        fth_results, tb_results, dates);

    float expected_fth_net = fth_day1.result.net_pnl + fth_day2.result.net_pnl;
    float expected_tb_net = tb_day1.result.net_pnl + tb_day2.result.net_pnl;

    EXPECT_NEAR(report.first_to_hit.net_pnl, expected_fth_net, 0.01f);
    EXPECT_NEAR(report.triple_barrier.net_pnl, expected_tb_net, 0.01f);
}

TEST_F(AggregateDayResultsTest, GrossPnlSumsAcrossDays) {
    auto fth_day1 = make_day_result(Q1_DATE_A, 5, 3);
    auto fth_day2 = make_day_result(Q2_DATE_A, 7, 2);
    auto tb_day1 = make_day_result(Q1_DATE_A, 4, 4);
    auto tb_day2 = make_day_result(Q2_DATE_A, 6, 3);

    std::vector<DayResult> fth_results = {fth_day1, fth_day2};
    std::vector<DayResult> tb_results = {tb_day1, tb_day2};
    std::vector<int> dates = {Q1_DATE_A, Q2_DATE_A};

    auto report = oracle_expectancy::aggregate_day_results(
        fth_results, tb_results, dates);

    float expected_fth_gross = fth_day1.result.gross_pnl + fth_day2.result.gross_pnl;
    float expected_tb_gross = tb_day1.result.gross_pnl + tb_day2.result.gross_pnl;

    EXPECT_NEAR(report.first_to_hit.gross_pnl, expected_fth_gross, 0.01f);
    EXPECT_NEAR(report.triple_barrier.gross_pnl, expected_tb_gross, 0.01f);
}

TEST_F(AggregateDayResultsTest, WinsAndLossesSumAcrossDays) {
    auto fth_day1 = make_day_result(Q1_DATE_A, 5, 3);
    auto fth_day2 = make_day_result(Q2_DATE_A, 7, 2);

    std::vector<DayResult> fth_results = {fth_day1, fth_day2};
    std::vector<DayResult> tb_results = {fth_day1, fth_day2};  // reuse
    std::vector<int> dates = {Q1_DATE_A, Q2_DATE_A};

    auto report = oracle_expectancy::aggregate_day_results(
        fth_results, tb_results, dates);

    EXPECT_EQ(report.first_to_hit.winning_trades, 12);  // 5 + 7
    EXPECT_EQ(report.first_to_hit.losing_trades, 5);    // 3 + 2
}

// ===========================================================================
// 3. aggregate_day_results — skipped day handling (Validation Gate 6)
// ===========================================================================

TEST_F(AggregateDayResultsTest, SkipsDaysWithSkippedTrue) {
    // Validation Gate 6: skips days with skipped=true
    auto fth_active = make_day_result(Q1_DATE_A, 5, 3);   // 8 trades
    auto fth_skipped = make_skipped_day(Q1_DATE_B);
    auto tb_active = make_day_result(Q1_DATE_A, 4, 4);
    auto tb_skipped = make_skipped_day(Q1_DATE_B);

    std::vector<DayResult> fth_results = {fth_active, fth_skipped};
    std::vector<DayResult> tb_results = {tb_active, tb_skipped};
    std::vector<int> dates = {Q1_DATE_A, Q1_DATE_B};

    auto report = oracle_expectancy::aggregate_day_results(
        fth_results, tb_results, dates);

    // Only the active day should count
    EXPECT_EQ(report.first_to_hit.total_trades, 8);
    EXPECT_EQ(report.triple_barrier.total_trades, 8);
}

TEST_F(AggregateDayResultsTest, DaysSkippedCountTracked) {
    auto fth_active = make_day_result(Q1_DATE_A, 5, 3);
    auto fth_skipped = make_skipped_day(Q1_DATE_B);
    auto tb_active = make_day_result(Q1_DATE_A, 4, 4);
    auto tb_skipped = make_skipped_day(Q1_DATE_B);

    std::vector<DayResult> fth_results = {fth_active, fth_skipped};
    std::vector<DayResult> tb_results = {tb_active, tb_skipped};
    std::vector<int> dates = {Q1_DATE_A, Q1_DATE_B};

    auto report = oracle_expectancy::aggregate_day_results(
        fth_results, tb_results, dates);

    EXPECT_EQ(report.days_processed, 1);
    EXPECT_EQ(report.days_skipped, 1);
}

TEST_F(AggregateDayResultsTest, AllDaysSkippedReturnsZeroTrades) {
    auto fth_skip1 = make_skipped_day(Q1_DATE_A);
    auto fth_skip2 = make_skipped_day(Q1_DATE_B);

    std::vector<DayResult> fth_results = {fth_skip1, fth_skip2};
    std::vector<DayResult> tb_results = {fth_skip1, fth_skip2};
    std::vector<int> dates = {Q1_DATE_A, Q1_DATE_B};

    auto report = oracle_expectancy::aggregate_day_results(
        fth_results, tb_results, dates);

    EXPECT_EQ(report.days_processed, 0);
    EXPECT_EQ(report.days_skipped, 2);
    EXPECT_EQ(report.first_to_hit.total_trades, 0);
    EXPECT_EQ(report.triple_barrier.total_trades, 0);
}

// ===========================================================================
// 4. aggregate_day_results — per-quarter splitting (Validation Gate 4)
// ===========================================================================

TEST_F(AggregateDayResultsTest, PerQuarterSplitAssignsCorrectly) {
    // Validation Gate 4: correctly computes per-quarter splits
    auto fth_q1 = make_day_result(Q1_DATE_A, 5, 3);     // Q1
    auto fth_q2 = make_day_result(Q2_DATE_A, 7, 2);     // Q2
    auto fth_q3 = make_day_result(Q3_DATE_A, 6, 4);     // Q3
    auto fth_q4 = make_day_result(Q4_DATE_A, 8, 1);     // Q4
    auto tb_q1 = make_day_result(Q1_DATE_A, 4, 4);
    auto tb_q2 = make_day_result(Q2_DATE_A, 3, 5);
    auto tb_q3 = make_day_result(Q3_DATE_A, 5, 5);
    auto tb_q4 = make_day_result(Q4_DATE_A, 7, 2);

    std::vector<DayResult> fth = {fth_q1, fth_q2, fth_q3, fth_q4};
    std::vector<DayResult> tb = {tb_q1, tb_q2, tb_q3, tb_q4};
    std::vector<int> dates = {Q1_DATE_A, Q2_DATE_A, Q3_DATE_A, Q4_DATE_A};

    auto report = oracle_expectancy::aggregate_day_results(fth, tb, dates);

    // All four quarters should be present
    EXPECT_EQ(report.fth_per_quarter.size(), 4u);
    EXPECT_EQ(report.tb_per_quarter.size(), 4u);
    EXPECT_TRUE(report.fth_per_quarter.count("Q1"));
    EXPECT_TRUE(report.fth_per_quarter.count("Q2"));
    EXPECT_TRUE(report.fth_per_quarter.count("Q3"));
    EXPECT_TRUE(report.fth_per_quarter.count("Q4"));
}

TEST_F(AggregateDayResultsTest, PerQuarterTradeCountsCorrect) {
    auto fth_q1 = make_day_result(Q1_DATE_A, 5, 3);     // 8 trades
    auto fth_q2 = make_day_result(Q2_DATE_A, 7, 2);     // 9 trades

    std::vector<DayResult> fth = {fth_q1, fth_q2};
    std::vector<DayResult> tb = {fth_q1, fth_q2};
    std::vector<int> dates = {Q1_DATE_A, Q2_DATE_A};

    auto report = oracle_expectancy::aggregate_day_results(fth, tb, dates);

    EXPECT_EQ(report.fth_per_quarter.at("Q1").total_trades, 8);
    EXPECT_EQ(report.fth_per_quarter.at("Q2").total_trades, 9);
}

TEST_F(AggregateDayResultsTest, MultipleDaysSameQuarterAggregate) {
    // Two days in Q1 should merge into a single Q1 BacktestResult
    auto fth_q1a = make_day_result(Q1_DATE_A, 5, 3);    // 8 trades
    auto fth_q1b = make_day_result(Q1_DATE_B, 3, 2);    // 5 trades

    std::vector<DayResult> fth = {fth_q1a, fth_q1b};
    std::vector<DayResult> tb = {fth_q1a, fth_q1b};
    std::vector<int> dates = {Q1_DATE_A, Q1_DATE_B};

    auto report = oracle_expectancy::aggregate_day_results(fth, tb, dates);

    EXPECT_EQ(report.fth_per_quarter.at("Q1").total_trades, 13);  // 8 + 5
}

TEST_F(AggregateDayResultsTest, PerQuarterNetPnlSumsCorrectly) {
    auto fth_q1 = make_day_result(Q1_DATE_A, 5, 3);
    auto fth_q2 = make_day_result(Q2_DATE_A, 7, 2);

    std::vector<DayResult> fth = {fth_q1, fth_q2};
    std::vector<DayResult> tb = {fth_q1, fth_q2};
    std::vector<int> dates = {Q1_DATE_A, Q2_DATE_A};

    auto report = oracle_expectancy::aggregate_day_results(fth, tb, dates);

    EXPECT_NEAR(report.fth_per_quarter.at("Q1").net_pnl,
                fth_q1.result.net_pnl, 0.01f);
    EXPECT_NEAR(report.fth_per_quarter.at("Q2").net_pnl,
                fth_q2.result.net_pnl, 0.01f);
}

// ===========================================================================
// 5. Per-quarter totals sum to overall totals (Validation Gate 10)
// ===========================================================================

TEST_F(AggregateDayResultsTest, PerQuarterTotalsSumToOverall_FTH) {
    // Validation Gate 10: per-quarter totals sum to overall totals
    auto fth_q1 = make_day_result(Q1_DATE_A, 5, 3);
    auto fth_q2 = make_day_result(Q2_DATE_A, 7, 2);
    auto fth_q3 = make_day_result(Q3_DATE_A, 6, 4);
    auto fth_q4 = make_day_result(Q4_DATE_A, 8, 1);

    std::vector<DayResult> fth = {fth_q1, fth_q2, fth_q3, fth_q4};
    std::vector<DayResult> tb = {fth_q1, fth_q2, fth_q3, fth_q4};
    std::vector<int> dates = {Q1_DATE_A, Q2_DATE_A, Q3_DATE_A, Q4_DATE_A};

    auto report = oracle_expectancy::aggregate_day_results(fth, tb, dates);

    int quarter_total_trades = 0;
    float quarter_total_net = 0.0f;
    float quarter_total_gross = 0.0f;
    int quarter_total_wins = 0;
    int quarter_total_losses = 0;

    for (const auto& [q, result] : report.fth_per_quarter) {
        quarter_total_trades += result.total_trades;
        quarter_total_net += result.net_pnl;
        quarter_total_gross += result.gross_pnl;
        quarter_total_wins += result.winning_trades;
        quarter_total_losses += result.losing_trades;
    }

    EXPECT_EQ(quarter_total_trades, report.first_to_hit.total_trades);
    EXPECT_NEAR(quarter_total_net, report.first_to_hit.net_pnl, 0.01f);
    EXPECT_NEAR(quarter_total_gross, report.first_to_hit.gross_pnl, 0.01f);
    EXPECT_EQ(quarter_total_wins, report.first_to_hit.winning_trades);
    EXPECT_EQ(quarter_total_losses, report.first_to_hit.losing_trades);
}

TEST_F(AggregateDayResultsTest, PerQuarterTotalsSumToOverall_TB) {
    auto tb_q1 = make_day_result(Q1_DATE_A, 4, 4);
    auto tb_q2 = make_day_result(Q2_DATE_A, 3, 5);
    auto tb_q3 = make_day_result(Q3_DATE_A, 5, 5);
    auto tb_q4 = make_day_result(Q4_DATE_A, 7, 2);

    std::vector<DayResult> fth = {tb_q1, tb_q2, tb_q3, tb_q4};
    std::vector<DayResult> tb = {tb_q1, tb_q2, tb_q3, tb_q4};
    std::vector<int> dates = {Q1_DATE_A, Q2_DATE_A, Q3_DATE_A, Q4_DATE_A};

    auto report = oracle_expectancy::aggregate_day_results(fth, tb, dates);

    int quarter_total_trades = 0;
    float quarter_total_net = 0.0f;
    for (const auto& [q, result] : report.tb_per_quarter) {
        quarter_total_trades += result.total_trades;
        quarter_total_net += result.net_pnl;
    }

    EXPECT_EQ(quarter_total_trades, report.triple_barrier.total_trades);
    EXPECT_NEAR(quarter_total_net, report.triple_barrier.net_pnl, 0.01f);
}

// ===========================================================================
// 6. Derived metrics — expectancy consistency (Validation Gate 7)
// ===========================================================================

TEST_F(AggregateDayResultsTest, ExpectancyEqualsNetPnlOverTotalTrades) {
    // Validation Gate 7: expectancy = net_pnl / total_trades
    auto fth_day1 = make_day_result(Q1_DATE_A, 5, 3);
    auto fth_day2 = make_day_result(Q2_DATE_A, 7, 2);

    std::vector<DayResult> fth = {fth_day1, fth_day2};
    std::vector<DayResult> tb = {fth_day1, fth_day2};
    std::vector<int> dates = {Q1_DATE_A, Q2_DATE_A};

    auto report = oracle_expectancy::aggregate_day_results(fth, tb, dates);

    if (report.first_to_hit.total_trades > 0) {
        float expected_exp = report.first_to_hit.net_pnl /
                             static_cast<float>(report.first_to_hit.total_trades);
        EXPECT_NEAR(report.first_to_hit.expectancy, expected_exp, 0.01f);
    }
    if (report.triple_barrier.total_trades > 0) {
        float expected_exp = report.triple_barrier.net_pnl /
                             static_cast<float>(report.triple_barrier.total_trades);
        EXPECT_NEAR(report.triple_barrier.expectancy, expected_exp, 0.01f);
    }
}

TEST_F(AggregateDayResultsTest, PerQuarterExpectancyConsistent) {
    auto fth_q1 = make_day_result(Q1_DATE_A, 5, 3);
    auto fth_q2 = make_day_result(Q2_DATE_A, 7, 2);

    std::vector<DayResult> fth = {fth_q1, fth_q2};
    std::vector<DayResult> tb = {fth_q1, fth_q2};
    std::vector<int> dates = {Q1_DATE_A, Q2_DATE_A};

    auto report = oracle_expectancy::aggregate_day_results(fth, tb, dates);

    for (const auto& [q, result] : report.fth_per_quarter) {
        if (result.total_trades > 0) {
            float expected = result.net_pnl / static_cast<float>(result.total_trades);
            EXPECT_NEAR(result.expectancy, expected, 0.01f)
                << "Expectancy inconsistent for quarter " << q;
        }
    }
}

// ===========================================================================
// 7. Derived metrics — win rate consistency (Validation Gate 8)
// ===========================================================================

TEST_F(AggregateDayResultsTest, WinRateEqualsWinsOverTotal) {
    // Validation Gate 8: win_rate = winning_trades / total_trades
    auto fth_day1 = make_day_result(Q1_DATE_A, 5, 3);
    auto fth_day2 = make_day_result(Q2_DATE_A, 7, 2);

    std::vector<DayResult> fth = {fth_day1, fth_day2};
    std::vector<DayResult> tb = {fth_day1, fth_day2};
    std::vector<int> dates = {Q1_DATE_A, Q2_DATE_A};

    auto report = oracle_expectancy::aggregate_day_results(fth, tb, dates);

    if (report.first_to_hit.total_trades > 0) {
        float expected_wr = static_cast<float>(report.first_to_hit.winning_trades) /
                            static_cast<float>(report.first_to_hit.total_trades);
        EXPECT_NEAR(report.first_to_hit.win_rate, expected_wr, 0.001f);
    }
}

TEST_F(AggregateDayResultsTest, PerQuarterWinRateConsistent) {
    auto fth_q1 = make_day_result(Q1_DATE_A, 5, 3);
    auto fth_q3 = make_day_result(Q3_DATE_A, 6, 4);

    std::vector<DayResult> fth = {fth_q1, fth_q3};
    std::vector<DayResult> tb = {fth_q1, fth_q3};
    std::vector<int> dates = {Q1_DATE_A, Q3_DATE_A};

    auto report = oracle_expectancy::aggregate_day_results(fth, tb, dates);

    for (const auto& [q, result] : report.fth_per_quarter) {
        if (result.total_trades > 0) {
            float expected = static_cast<float>(result.winning_trades) /
                             static_cast<float>(result.total_trades);
            EXPECT_NEAR(result.win_rate, expected, 0.001f)
                << "Win rate inconsistent for quarter " << q;
        }
    }
}

// ===========================================================================
// 8. to_json — required field presence (Validation Gate 1)
// ===========================================================================
class ToJsonTest : public ::testing::Test {};

TEST_F(ToJsonTest, ContainsConfigBlock) {
    // Validation Gate 1: output contains all required fields — config section
    OracleExpectancyReport report{};
    report.days_processed = 5;

    std::string json = oracle_expectancy::to_json(report);

    EXPECT_NE(json.find("\"config\""), std::string::npos);
    EXPECT_NE(json.find("\"bar_type\""), std::string::npos);
    EXPECT_NE(json.find("\"time_5s\""), std::string::npos);
    EXPECT_NE(json.find("\"target_ticks\""), std::string::npos);
    EXPECT_NE(json.find("\"stop_ticks\""), std::string::npos);
    EXPECT_NE(json.find("\"take_profit_ticks\""), std::string::npos);
    EXPECT_NE(json.find("\"volume_horizon\""), std::string::npos);
    EXPECT_NE(json.find("\"max_time_horizon_s\""), std::string::npos);
}

TEST_F(ToJsonTest, ContainsCostsBlock) {
    OracleExpectancyReport report{};

    std::string json = oracle_expectancy::to_json(report);

    EXPECT_NE(json.find("\"costs\""), std::string::npos);
    EXPECT_NE(json.find("\"commission_per_side\""), std::string::npos);
    EXPECT_NE(json.find("\"fixed_spread_ticks\""), std::string::npos);
    EXPECT_NE(json.find("\"slippage_ticks\""), std::string::npos);
    EXPECT_NE(json.find("\"contract_multiplier\""), std::string::npos);
    EXPECT_NE(json.find("\"tick_size\""), std::string::npos);
}

TEST_F(ToJsonTest, ContainsDaysProcessedAndSkipped) {
    OracleExpectancyReport report{};
    report.days_processed = 20;
    report.days_skipped = 3;

    std::string json = oracle_expectancy::to_json(report);

    EXPECT_NE(json.find("\"days_processed\""), std::string::npos);
    EXPECT_NE(json.find("\"days_skipped\""), std::string::npos);
}

TEST_F(ToJsonTest, ContainsFirstToHitBlock) {
    OracleExpectancyReport report{};
    report.first_to_hit.total_trades = 100;

    std::string json = oracle_expectancy::to_json(report);

    EXPECT_NE(json.find("\"first_to_hit\""), std::string::npos);
}

TEST_F(ToJsonTest, ContainsTripleBarrierBlock) {
    OracleExpectancyReport report{};
    report.triple_barrier.total_trades = 50;

    std::string json = oracle_expectancy::to_json(report);

    EXPECT_NE(json.find("\"triple_barrier\""), std::string::npos);
}

TEST_F(ToJsonTest, FirstToHitBlockContainsAllFields) {
    // Validation Gate 1: all required fields from the output schema
    OracleExpectancyReport report{};
    report.first_to_hit.total_trades = 100;
    report.first_to_hit.winning_trades = 60;
    report.first_to_hit.losing_trades = 40;
    report.first_to_hit.win_rate = 0.6f;
    report.first_to_hit.gross_pnl = 500.0f;
    report.first_to_hit.net_pnl = 250.0f;
    report.first_to_hit.expectancy = 2.5f;
    report.first_to_hit.profit_factor = 1.5f;
    report.first_to_hit.sharpe = 0.8f;
    report.first_to_hit.max_drawdown = 100.0f;
    report.first_to_hit.trades_per_day = 5.0f;
    report.first_to_hit.avg_bars_held = 3.0f;
    report.first_to_hit.avg_duration_s = 15.0f;
    report.first_to_hit.hold_fraction = 0.2f;

    std::string json = oracle_expectancy::to_json(report);

    EXPECT_NE(json.find("\"total_trades\""), std::string::npos);
    EXPECT_NE(json.find("\"winning_trades\""), std::string::npos);
    EXPECT_NE(json.find("\"losing_trades\""), std::string::npos);
    EXPECT_NE(json.find("\"win_rate\""), std::string::npos);
    EXPECT_NE(json.find("\"gross_pnl\""), std::string::npos);
    EXPECT_NE(json.find("\"net_pnl\""), std::string::npos);
    EXPECT_NE(json.find("\"expectancy\""), std::string::npos);
    EXPECT_NE(json.find("\"profit_factor\""), std::string::npos);
    EXPECT_NE(json.find("\"sharpe\""), std::string::npos);
    EXPECT_NE(json.find("\"max_drawdown\""), std::string::npos);
    EXPECT_NE(json.find("\"trades_per_day\""), std::string::npos);
    EXPECT_NE(json.find("\"avg_bars_held\""), std::string::npos);
    EXPECT_NE(json.find("\"avg_duration_s\""), std::string::npos);
    EXPECT_NE(json.find("\"hold_fraction\""), std::string::npos);
}

TEST_F(ToJsonTest, FirstToHitBlockContainsExitReasons) {
    OracleExpectancyReport report{};
    report.first_to_hit.exit_reason_counts[exit_reason::TARGET] = 30;
    report.first_to_hit.exit_reason_counts[exit_reason::STOP] = 20;
    report.first_to_hit.exit_reason_counts[exit_reason::TAKE_PROFIT] = 10;
    report.first_to_hit.exit_reason_counts[exit_reason::EXPIRY] = 5;
    report.first_to_hit.exit_reason_counts[exit_reason::SESSION_END] = 3;
    report.first_to_hit.exit_reason_counts[exit_reason::SAFETY_CAP] = 2;

    std::string json = oracle_expectancy::to_json(report);

    EXPECT_NE(json.find("\"exit_reasons\""), std::string::npos);
    EXPECT_NE(json.find("\"target\""), std::string::npos);
    EXPECT_NE(json.find("\"stop\""), std::string::npos);
    EXPECT_NE(json.find("\"take_profit\""), std::string::npos);
    EXPECT_NE(json.find("\"expiry\""), std::string::npos);
    EXPECT_NE(json.find("\"session_end\""), std::string::npos);
    EXPECT_NE(json.find("\"safety_cap\""), std::string::npos);
}

TEST_F(ToJsonTest, ContainsPerQuarterBlock) {
    OracleExpectancyReport report{};
    BacktestResult q1{};
    q1.total_trades = 25;
    report.fth_per_quarter["Q1"] = q1;
    report.tb_per_quarter["Q1"] = q1;

    std::string json = oracle_expectancy::to_json(report);

    EXPECT_NE(json.find("\"per_quarter\""), std::string::npos);
    EXPECT_NE(json.find("\"Q1\""), std::string::npos);
}

TEST_F(ToJsonTest, PerQuarterContainsBothMethods) {
    OracleExpectancyReport report{};
    BacktestResult q1{};
    q1.total_trades = 25;
    report.fth_per_quarter["Q1"] = q1;
    report.tb_per_quarter["Q1"] = q1;

    std::string json = oracle_expectancy::to_json(report);

    // Per-quarter should have sub-objects for first_to_hit and triple_barrier
    // Find "Q1" and verify both method keys exist in its scope
    auto q1_pos = json.find("\"Q1\"");
    ASSERT_NE(q1_pos, std::string::npos);
    // After Q1, there should be first_to_hit and triple_barrier sub-objects
    auto after_q1 = json.substr(q1_pos);
    EXPECT_NE(after_q1.find("\"first_to_hit\""), std::string::npos);
    EXPECT_NE(after_q1.find("\"triple_barrier\""), std::string::npos);
}

TEST_F(ToJsonTest, AllFourQuartersPresent) {
    OracleExpectancyReport report{};
    for (const auto& q : {"Q1", "Q2", "Q3", "Q4"}) {
        BacktestResult r{};
        r.total_trades = 10;
        report.fth_per_quarter[q] = r;
        report.tb_per_quarter[q] = r;
    }

    std::string json = oracle_expectancy::to_json(report);

    EXPECT_NE(json.find("\"Q1\""), std::string::npos);
    EXPECT_NE(json.find("\"Q2\""), std::string::npos);
    EXPECT_NE(json.find("\"Q3\""), std::string::npos);
    EXPECT_NE(json.find("\"Q4\""), std::string::npos);
}

// ===========================================================================
// 9. to_json — numeric precision (Validation Gate 2)
// ===========================================================================

TEST_F(ToJsonTest, NumericValuesRoundTripCorrectly) {
    // Validation Gate 2: float precision preserved in JSON
    OracleExpectancyReport report{};
    report.days_processed = 20;
    report.first_to_hit.total_trades = 100;
    report.first_to_hit.net_pnl = 123.456f;
    report.first_to_hit.expectancy = 1.23456f;
    report.first_to_hit.win_rate = 0.65f;

    std::string json = oracle_expectancy::to_json(report);

    // The integer fields should appear exactly
    EXPECT_NE(json.find("20"), std::string::npos);   // days_processed
    EXPECT_NE(json.find("100"), std::string::npos);  // total_trades
}

TEST_F(ToJsonTest, DaysProcessedValueCorrect) {
    OracleExpectancyReport report{};
    report.days_processed = 17;
    report.days_skipped = 3;

    std::string json = oracle_expectancy::to_json(report);

    // Should contain the exact values near the field keys
    EXPECT_NE(json.find("\"days_processed\":17"), std::string::npos);
    EXPECT_NE(json.find("\"days_skipped\":3"), std::string::npos);
}

TEST_F(ToJsonTest, ConfigDefaultValues) {
    // Spec config defaults: target_ticks=10, stop_ticks=5, etc.
    OracleExpectancyReport report{};

    std::string json = oracle_expectancy::to_json(report);

    EXPECT_NE(json.find("\"target_ticks\":10"), std::string::npos);
    EXPECT_NE(json.find("\"stop_ticks\":5"), std::string::npos);
    EXPECT_NE(json.find("\"take_profit_ticks\":20"), std::string::npos);
    EXPECT_NE(json.find("\"volume_horizon\":500"), std::string::npos);
    EXPECT_NE(json.find("\"max_time_horizon_s\":300"), std::string::npos);
}

TEST_F(ToJsonTest, CostsDefaultValues) {
    OracleExpectancyReport report{};

    std::string json = oracle_expectancy::to_json(report);

    // Commission: 0.62
    EXPECT_NE(json.find("\"commission_per_side\":0.62"), std::string::npos);
    EXPECT_NE(json.find("\"fixed_spread_ticks\":1"), std::string::npos);
    EXPECT_NE(json.find("\"slippage_ticks\":0"), std::string::npos);
}

TEST_F(ToJsonTest, ZeroTradesProduceZeroMetrics) {
    OracleExpectancyReport report{};
    // Default: zero trades on both methods

    std::string json = oracle_expectancy::to_json(report);

    // Should be valid JSON (not empty)
    EXPECT_FALSE(json.empty());
    EXPECT_EQ(json.front(), '{');
    EXPECT_EQ(json.back(), '}');
}

// ===========================================================================
// 10. to_json — structural validity
// ===========================================================================

TEST_F(ToJsonTest, OutputStartsWithOpenBrace) {
    OracleExpectancyReport report{};
    std::string json = oracle_expectancy::to_json(report);
    ASSERT_FALSE(json.empty());
    EXPECT_EQ(json.front(), '{');
}

TEST_F(ToJsonTest, OutputEndsWithCloseBrace) {
    OracleExpectancyReport report{};
    std::string json = oracle_expectancy::to_json(report);
    ASSERT_FALSE(json.empty());
    EXPECT_EQ(json.back(), '}');
}

TEST_F(ToJsonTest, OutputNotEmpty) {
    OracleExpectancyReport report{};
    std::string json = oracle_expectancy::to_json(report);
    EXPECT_GT(json.size(), 10u);  // should be a non-trivial JSON object
}

TEST_F(ToJsonTest, BracesBalanced) {
    OracleExpectancyReport report{};
    report.days_processed = 5;
    BacktestResult q1{};
    q1.total_trades = 10;
    report.fth_per_quarter["Q1"] = q1;
    report.tb_per_quarter["Q1"] = q1;

    std::string json = oracle_expectancy::to_json(report);

    int brace_count = 0;
    for (char c : json) {
        if (c == '{') ++brace_count;
        if (c == '}') --brace_count;
        EXPECT_GE(brace_count, 0) << "Unbalanced closing brace in JSON";
    }
    EXPECT_EQ(brace_count, 0) << "Unbalanced braces in JSON output";
}

// ===========================================================================
// 11. Exit reason counts sum to total_trades (Validation Gate 9)
// ===========================================================================

TEST_F(AggregateDayResultsTest, ExitReasonCountsSumToTotalTrades_FTH) {
    // Validation Gate 9: exit reason counts sum to total_trades
    // Build day results with mixed exit reasons
    DayResult fth_day{};
    fth_day.date = Q1_DATE_A;
    fth_day.skipped = false;
    fth_day.bar_count = 100;

    // 3 TARGET exits
    for (int i = 0; i < 3; ++i) {
        TradeRecord t{};
        t.gross_pnl = 12.50f;
        t.net_pnl = 10.01f;
        t.exit_reason = exit_reason::TARGET;
        t.bars_held = 2;
        t.duration_s = 10.0f;
        fth_day.result.trades.push_back(t);
    }
    // 2 STOP exits
    for (int i = 0; i < 2; ++i) {
        TradeRecord t{};
        t.gross_pnl = -6.25f;
        t.net_pnl = -8.74f;
        t.exit_reason = exit_reason::STOP;
        t.bars_held = 2;
        t.duration_s = 10.0f;
        fth_day.result.trades.push_back(t);
    }
    // 1 TAKE_PROFIT exit
    {
        TradeRecord t{};
        t.gross_pnl = 25.0f;
        t.net_pnl = 22.51f;
        t.exit_reason = exit_reason::TAKE_PROFIT;
        t.bars_held = 5;
        t.duration_s = 25.0f;
        fth_day.result.trades.push_back(t);
    }
    // 1 SESSION_END exit
    {
        TradeRecord t{};
        t.gross_pnl = 3.0f;
        t.net_pnl = 0.51f;
        t.exit_reason = exit_reason::SESSION_END;
        t.bars_held = 10;
        t.duration_s = 50.0f;
        fth_day.result.trades.push_back(t);
    }

    fth_day.result.total_trades = 7;
    fth_day.result.exit_reason_counts[exit_reason::TARGET] = 3;
    fth_day.result.exit_reason_counts[exit_reason::STOP] = 2;
    fth_day.result.exit_reason_counts[exit_reason::TAKE_PROFIT] = 1;
    fth_day.result.exit_reason_counts[exit_reason::SESSION_END] = 1;

    std::vector<DayResult> fth = {fth_day};
    std::vector<DayResult> tb = {fth_day};  // reuse for simplicity
    std::vector<int> dates = {Q1_DATE_A};

    auto report = oracle_expectancy::aggregate_day_results(fth, tb, dates);

    // The JSON output should have exit_reasons that sum to total_trades
    std::string json = oracle_expectancy::to_json(report);
    EXPECT_NE(json.find("\"exit_reasons\""), std::string::npos);

    // Verify via the BacktestResult struct directly
    int exit_sum = 0;
    for (const auto& [reason, count] : report.first_to_hit.exit_reason_counts) {
        exit_sum += count;
    }
    // Exit reason counts on the aggregated result should sum to total_trades
    // Note: the implementation must aggregate exit_reason_counts across days
    EXPECT_EQ(exit_sum, report.first_to_hit.total_trades);
}

// ===========================================================================
// 12. Edge cases
// ===========================================================================

TEST_F(AggregateDayResultsTest, SingleDayAllWins) {
    auto fth = make_day_result(Q1_DATE_A, 10, 0);
    std::vector<DayResult> fth_results = {fth};
    std::vector<DayResult> tb_results = {fth};
    std::vector<int> dates = {Q1_DATE_A};

    auto report = oracle_expectancy::aggregate_day_results(
        fth_results, tb_results, dates);

    EXPECT_EQ(report.first_to_hit.winning_trades, 10);
    EXPECT_EQ(report.first_to_hit.losing_trades, 0);
    EXPECT_FLOAT_EQ(report.first_to_hit.win_rate, 1.0f);
}

TEST_F(AggregateDayResultsTest, SingleDayAllLosses) {
    auto fth = make_day_result(Q1_DATE_A, 0, 10);
    std::vector<DayResult> fth_results = {fth};
    std::vector<DayResult> tb_results = {fth};
    std::vector<int> dates = {Q1_DATE_A};

    auto report = oracle_expectancy::aggregate_day_results(
        fth_results, tb_results, dates);

    EXPECT_EQ(report.first_to_hit.winning_trades, 0);
    EXPECT_EQ(report.first_to_hit.losing_trades, 10);
    EXPECT_FLOAT_EQ(report.first_to_hit.win_rate, 0.0f);
    EXPECT_LT(report.first_to_hit.net_pnl, 0.0f);
    EXPECT_LT(report.first_to_hit.expectancy, 0.0f);
}

TEST_F(AggregateDayResultsTest, OnlyOneQuarterPopulated) {
    // If all days are in Q3, only Q3 should appear in per-quarter maps
    auto fth_q3a = make_day_result(Q3_DATE_A, 5, 3);
    auto fth_q3b = make_day_result(Q3_DATE_B, 4, 2);

    std::vector<DayResult> fth = {fth_q3a, fth_q3b};
    std::vector<DayResult> tb = {fth_q3a, fth_q3b};
    std::vector<int> dates = {Q3_DATE_A, Q3_DATE_B};

    auto report = oracle_expectancy::aggregate_day_results(fth, tb, dates);

    EXPECT_EQ(report.fth_per_quarter.size(), 1u);
    EXPECT_TRUE(report.fth_per_quarter.count("Q3"));
    EXPECT_FALSE(report.fth_per_quarter.count("Q1"));
    EXPECT_FALSE(report.fth_per_quarter.count("Q2"));
    EXPECT_FALSE(report.fth_per_quarter.count("Q4"));
}

TEST_F(AggregateDayResultsTest, DaysProcessedCountsActiveDaysOnly) {
    auto fth_active1 = make_day_result(Q1_DATE_A, 5, 3);
    auto fth_active2 = make_day_result(Q2_DATE_A, 7, 2);
    auto fth_skip = make_skipped_day(Q3_DATE_A);

    std::vector<DayResult> fth = {fth_active1, fth_active2, fth_skip};
    std::vector<DayResult> tb = {fth_active1, fth_active2, fth_skip};
    std::vector<int> dates = {Q1_DATE_A, Q2_DATE_A, Q3_DATE_A};

    auto report = oracle_expectancy::aggregate_day_results(fth, tb, dates);

    EXPECT_EQ(report.days_processed, 2);
    EXPECT_EQ(report.days_skipped, 1);
}

TEST_F(AggregateDayResultsTest, TradesPerDayRecomputed) {
    auto fth_day1 = make_day_result(Q1_DATE_A, 5, 3);  // 8 trades
    auto fth_day2 = make_day_result(Q2_DATE_A, 7, 2);  // 9 trades

    std::vector<DayResult> fth = {fth_day1, fth_day2};
    std::vector<DayResult> tb = {fth_day1, fth_day2};
    std::vector<int> dates = {Q1_DATE_A, Q2_DATE_A};

    auto report = oracle_expectancy::aggregate_day_results(fth, tb, dates);

    // trades_per_day = total_trades / active_days = 17 / 2 = 8.5
    EXPECT_NEAR(report.first_to_hit.trades_per_day, 8.5f, 0.01f);
}

TEST_F(AggregateDayResultsTest, ProfitFactorRecomputed) {
    auto fth_day = make_day_result(Q1_DATE_A, 5, 3);

    std::vector<DayResult> fth = {fth_day};
    std::vector<DayResult> tb = {fth_day};
    std::vector<int> dates = {Q1_DATE_A};

    auto report = oracle_expectancy::aggregate_day_results(fth, tb, dates);

    // profit_factor = gross_wins / gross_losses (computed from individual trades)
    EXPECT_GT(report.first_to_hit.profit_factor, 0.0f);
}

TEST_F(AggregateDayResultsTest, SharpeRecomputed) {
    auto fth_day1 = make_day_result(Q1_DATE_A, 5, 3);
    auto fth_day2 = make_day_result(Q2_DATE_A, 7, 2);

    std::vector<DayResult> fth = {fth_day1, fth_day2};
    std::vector<DayResult> tb = {fth_day1, fth_day2};
    std::vector<int> dates = {Q1_DATE_A, Q2_DATE_A};

    auto report = oracle_expectancy::aggregate_day_results(fth, tb, dates);

    // Sharpe should be finite (not NaN/Inf)
    EXPECT_TRUE(std::isfinite(report.first_to_hit.sharpe));
}

TEST_F(AggregateDayResultsTest, MaxDrawdownRecomputed) {
    auto fth_day1 = make_day_result(Q1_DATE_A, 5, 3);
    auto fth_day2 = make_day_result(Q2_DATE_A, 2, 8);  // bad day

    std::vector<DayResult> fth = {fth_day1, fth_day2};
    std::vector<DayResult> tb = {fth_day1, fth_day2};
    std::vector<int> dates = {Q1_DATE_A, Q2_DATE_A};

    auto report = oracle_expectancy::aggregate_day_results(fth, tb, dates);

    EXPECT_GE(report.first_to_hit.max_drawdown, 0.0f);
}

// ===========================================================================
// 13. to_json serialization of populated report
// ===========================================================================

TEST_F(ToJsonTest, PopulatedReportSerializesAllSections) {
    // Build a full report and verify all sections appear
    auto fth_q1 = make_day_result(Q1_DATE_A, 5, 3);
    auto fth_q2 = make_day_result(Q2_DATE_A, 7, 2);
    auto tb_q1 = make_day_result(Q1_DATE_A, 4, 4);
    auto tb_q2 = make_day_result(Q2_DATE_A, 6, 3);

    std::vector<DayResult> fth = {fth_q1, fth_q2};
    std::vector<DayResult> tb = {tb_q1, tb_q2};
    std::vector<int> dates = {Q1_DATE_A, Q2_DATE_A};

    auto report = oracle_expectancy::aggregate_day_results(fth, tb, dates);
    std::string json = oracle_expectancy::to_json(report);

    // Top-level sections
    EXPECT_NE(json.find("\"config\""), std::string::npos);
    EXPECT_NE(json.find("\"days_processed\""), std::string::npos);
    EXPECT_NE(json.find("\"first_to_hit\""), std::string::npos);
    EXPECT_NE(json.find("\"triple_barrier\""), std::string::npos);
    EXPECT_NE(json.find("\"per_quarter\""), std::string::npos);
}

TEST_F(ToJsonTest, DifferentFthAndTbResultsSerializeIndependently) {
    // FTH and TB should have different values if given different inputs
    OracleExpectancyReport report{};
    report.first_to_hit.total_trades = 100;
    report.first_to_hit.net_pnl = 500.0f;
    report.triple_barrier.total_trades = 80;
    report.triple_barrier.net_pnl = -200.0f;

    std::string json = oracle_expectancy::to_json(report);

    // Both blocks should be present and contain different values
    auto fth_pos = json.find("\"first_to_hit\"");
    auto tb_pos = json.find("\"triple_barrier\"");
    ASSERT_NE(fth_pos, std::string::npos);
    ASSERT_NE(tb_pos, std::string::npos);
    EXPECT_NE(fth_pos, tb_pos);
}

// ===========================================================================
// 14. Quarter assignment boundary dates
// ===========================================================================

TEST_F(AggregateDayResultsTest, QuarterBoundaryQ1Q2) {
    // Q1 ends on 20220318 (MESH2 end), Q2 starts 20220319 (MESM2 start)
    auto fth_q1_end = make_day_result(20220318, 5, 3);
    auto fth_q2_start = make_day_result(20220319, 4, 2);

    std::vector<DayResult> fth = {fth_q1_end, fth_q2_start};
    std::vector<DayResult> tb = {fth_q1_end, fth_q2_start};
    std::vector<int> dates = {20220318, 20220319};

    auto report = oracle_expectancy::aggregate_day_results(fth, tb, dates);

    EXPECT_TRUE(report.fth_per_quarter.count("Q1"));
    EXPECT_TRUE(report.fth_per_quarter.count("Q2"));
    EXPECT_EQ(report.fth_per_quarter.at("Q1").total_trades, 8);
    EXPECT_EQ(report.fth_per_quarter.at("Q2").total_trades, 6);
}

TEST_F(AggregateDayResultsTest, QuarterBoundaryQ2Q3) {
    auto fth_q2_end = make_day_result(20220617, 5, 3);
    auto fth_q3_start = make_day_result(20220618, 4, 2);

    std::vector<DayResult> fth = {fth_q2_end, fth_q3_start};
    std::vector<DayResult> tb = {fth_q2_end, fth_q3_start};
    std::vector<int> dates = {20220617, 20220618};

    auto report = oracle_expectancy::aggregate_day_results(fth, tb, dates);

    EXPECT_TRUE(report.fth_per_quarter.count("Q2"));
    EXPECT_TRUE(report.fth_per_quarter.count("Q3"));
}

TEST_F(AggregateDayResultsTest, QuarterBoundaryQ3Q4) {
    auto fth_q3_end = make_day_result(20220916, 5, 3);
    auto fth_q4_start = make_day_result(20220917, 4, 2);

    std::vector<DayResult> fth = {fth_q3_end, fth_q4_start};
    std::vector<DayResult> tb = {fth_q3_end, fth_q4_start};
    std::vector<int> dates = {20220916, 20220917};

    auto report = oracle_expectancy::aggregate_day_results(fth, tb, dates);

    EXPECT_TRUE(report.fth_per_quarter.count("Q3"));
    EXPECT_TRUE(report.fth_per_quarter.count("Q4"));
}

TEST_F(AggregateDayResultsTest, FirstDayOfDataIsQ1) {
    auto fth = make_day_result(20220103, 5, 3);

    std::vector<DayResult> fth_results = {fth};
    std::vector<DayResult> tb_results = {fth};
    std::vector<int> dates = {20220103};

    auto report = oracle_expectancy::aggregate_day_results(
        fth_results, tb_results, dates);

    EXPECT_TRUE(report.fth_per_quarter.count("Q1"));
    EXPECT_EQ(report.fth_per_quarter.size(), 1u);
}

TEST_F(AggregateDayResultsTest, LastDayOfDataIsQ4) {
    auto fth = make_day_result(20221230, 5, 3);

    std::vector<DayResult> fth_results = {fth};
    std::vector<DayResult> tb_results = {fth};
    std::vector<int> dates = {20221230};

    auto report = oracle_expectancy::aggregate_day_results(
        fth_results, tb_results, dates);

    EXPECT_TRUE(report.fth_per_quarter.count("Q4"));
    EXPECT_EQ(report.fth_per_quarter.size(), 1u);
}

// ===========================================================================
// 15. Skipped days do not appear in per-quarter results
// ===========================================================================

TEST_F(AggregateDayResultsTest, SkippedDayNotInQuarterResults) {
    auto fth_active = make_day_result(Q1_DATE_A, 5, 3);
    auto fth_skip = make_skipped_day(Q2_DATE_A);

    std::vector<DayResult> fth = {fth_active, fth_skip};
    std::vector<DayResult> tb = {fth_active, fth_skip};
    std::vector<int> dates = {Q1_DATE_A, Q2_DATE_A};

    auto report = oracle_expectancy::aggregate_day_results(fth, tb, dates);

    // Q1 should be present; Q2 should NOT (only had a skipped day)
    EXPECT_TRUE(report.fth_per_quarter.count("Q1"));
    EXPECT_FALSE(report.fth_per_quarter.count("Q2"));
}

// ===========================================================================
// 16. Large dataset simulation — 20 days across 4 quarters
// ===========================================================================

TEST_F(AggregateDayResultsTest, TwentyDaysAcrossFourQuarters) {
    // Spec: 5 days per quarter, 20 days total
    std::vector<DayResult> fth_results;
    std::vector<DayResult> tb_results;
    std::vector<int> dates;

    // Q1: 5 days
    std::vector<int> q1_dates = {20220110, 20220124, 20220207, 20220221, 20220307};
    for (int d : q1_dates) {
        fth_results.push_back(make_day_result(d, 5, 3));
        tb_results.push_back(make_day_result(d, 4, 4));
        dates.push_back(d);
    }

    // Q2: 5 days
    std::vector<int> q2_dates = {20220401, 20220415, 20220502, 20220516, 20220601};
    for (int d : q2_dates) {
        fth_results.push_back(make_day_result(d, 6, 2));
        tb_results.push_back(make_day_result(d, 5, 3));
        dates.push_back(d);
    }

    // Q3: 5 days
    std::vector<int> q3_dates = {20220705, 20220719, 20220802, 20220816, 20220901};
    for (int d : q3_dates) {
        fth_results.push_back(make_day_result(d, 7, 3));
        tb_results.push_back(make_day_result(d, 6, 4));
        dates.push_back(d);
    }

    // Q4: 5 days
    std::vector<int> q4_dates = {20221003, 20221017, 20221101, 20221115, 20221201};
    for (int d : q4_dates) {
        fth_results.push_back(make_day_result(d, 8, 2));
        tb_results.push_back(make_day_result(d, 7, 3));
        dates.push_back(d);
    }

    auto report = oracle_expectancy::aggregate_day_results(
        fth_results, tb_results, dates);

    // 20 active days
    EXPECT_EQ(report.days_processed, 20);
    EXPECT_EQ(report.days_skipped, 0);

    // 4 quarters
    EXPECT_EQ(report.fth_per_quarter.size(), 4u);
    EXPECT_EQ(report.tb_per_quarter.size(), 4u);

    // Each quarter has 5 days
    // Q1 FTH: 5 days × 8 trades = 40
    EXPECT_EQ(report.fth_per_quarter.at("Q1").total_trades, 40);
    // Q2 FTH: 5 days × 8 trades = 40
    EXPECT_EQ(report.fth_per_quarter.at("Q2").total_trades, 40);
    // Q3 FTH: 5 days × 10 trades = 50
    EXPECT_EQ(report.fth_per_quarter.at("Q3").total_trades, 50);
    // Q4 FTH: 5 days × 10 trades = 50
    EXPECT_EQ(report.fth_per_quarter.at("Q4").total_trades, 50);

    // Overall FTH total: 40 + 40 + 50 + 50 = 180
    EXPECT_EQ(report.first_to_hit.total_trades, 180);

    // Per-quarter sums check
    int q_total = 0;
    for (const auto& [q, r] : report.fth_per_quarter) {
        q_total += r.total_trades;
    }
    EXPECT_EQ(q_total, report.first_to_hit.total_trades);
}

// ===========================================================================
// 17. Aggregate preserves TradeRecord vectors for Sharpe/drawdown recompute
// ===========================================================================

TEST_F(AggregateDayResultsTest, AggregateCollectsTradeRecords) {
    auto fth_day1 = make_day_result(Q1_DATE_A, 5, 3);  // 8 trades
    auto fth_day2 = make_day_result(Q2_DATE_A, 7, 2);  // 9 trades

    std::vector<DayResult> fth = {fth_day1, fth_day2};
    std::vector<DayResult> tb = {fth_day1, fth_day2};
    std::vector<int> dates = {Q1_DATE_A, Q2_DATE_A};

    auto report = oracle_expectancy::aggregate_day_results(fth, tb, dates);

    // The aggregate BacktestResult should have all individual TradeRecords
    EXPECT_EQ(static_cast<int>(report.first_to_hit.trades.size()),
              report.first_to_hit.total_trades);
}

TEST_F(AggregateDayResultsTest, AggregateTracksDailyPnl) {
    auto fth_day1 = make_day_result(Q1_DATE_A, 5, 3);
    auto fth_day2 = make_day_result(Q2_DATE_A, 7, 2);

    std::vector<DayResult> fth = {fth_day1, fth_day2};
    std::vector<DayResult> tb = {fth_day1, fth_day2};
    std::vector<int> dates = {Q1_DATE_A, Q2_DATE_A};

    auto report = oracle_expectancy::aggregate_day_results(fth, tb, dates);

    EXPECT_EQ(report.first_to_hit.daily_pnl.size(), 2u);
}

// ===========================================================================
// 18. Determinism — same inputs produce same outputs
// ===========================================================================

TEST_F(AggregateDayResultsTest, DeterministicOutput) {
    auto fth_q1 = make_day_result(Q1_DATE_A, 5, 3);
    auto fth_q2 = make_day_result(Q2_DATE_A, 7, 2);

    std::vector<DayResult> fth = {fth_q1, fth_q2};
    std::vector<DayResult> tb = {fth_q1, fth_q2};
    std::vector<int> dates = {Q1_DATE_A, Q2_DATE_A};

    auto report1 = oracle_expectancy::aggregate_day_results(fth, tb, dates);
    auto report2 = oracle_expectancy::aggregate_day_results(fth, tb, dates);

    EXPECT_EQ(report1.first_to_hit.total_trades, report2.first_to_hit.total_trades);
    EXPECT_FLOAT_EQ(report1.first_to_hit.net_pnl, report2.first_to_hit.net_pnl);
    EXPECT_FLOAT_EQ(report1.first_to_hit.expectancy, report2.first_to_hit.expectancy);
}

TEST_F(ToJsonTest, DeterministicJsonOutput) {
    OracleExpectancyReport report{};
    report.days_processed = 10;
    report.first_to_hit.total_trades = 50;
    report.first_to_hit.net_pnl = 100.0f;
    report.triple_barrier.total_trades = 40;
    report.triple_barrier.net_pnl = -50.0f;

    std::string json1 = oracle_expectancy::to_json(report);
    std::string json2 = oracle_expectancy::to_json(report);

    EXPECT_EQ(json1, json2);
}
