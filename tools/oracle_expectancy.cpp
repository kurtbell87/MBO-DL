// oracle_expectancy.cpp — Oracle Expectancy Extraction Tool
// Reads real MES .dbn.zst data, builds time_5s bars, runs OracleReplay
// with both FIRST_TO_HIT and TRIPLE_BARRIER labeling, then aggregates
// via the tested oracle_expectancy_report.hpp APIs to produce JSON output.
//
// Pattern: same StreamingBookBuilder as tools/subordination_test.cpp.
// Library layer: src/backtest/oracle_expectancy_report.hpp (67 unit tests).

#include "backtest/oracle_expectancy_report.hpp"
#include "backtest/execution_costs.hpp"
#include "backtest/oracle_replay.hpp"
#include "backtest/rollover.hpp"
#include "bars/bar_factory.hpp"
#include "time_utils.hpp"

#include <databento/dbn_file_store.hpp>
#include <databento/record.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

// ===========================================================================
// StreamingBookBuilder — emits snapshots incrementally, O(1) committed state
// (Identical to subordination_test.cpp)
// ===========================================================================
class StreamingBookBuilder {
public:
    explicit StreamingBookBuilder(uint32_t instrument_id,
                                   uint64_t rth_open, uint64_t rth_close)
        : instrument_id_(instrument_id),
          rth_open_(rth_open), rth_close_(rth_close),
          next_snap_ts_(rth_open) {}

    void process_event(uint64_t ts_event, uint64_t order_id, uint32_t instrument_id,
                       char action, char side, int64_t price, uint32_t size, uint8_t flags) {
        if (instrument_id != instrument_id_) return;

        constexpr uint8_t F_LAST = 0x80;

        switch (action) {
            case 'A': apply_add(order_id, side, price, size); break;
            case 'C': apply_cancel(order_id); break;
            case 'M': apply_modify(order_id, side, price, size); break;
            case 'T': apply_trade(side, price, size); break;
            case 'F': apply_fill(order_id, size); break;
            case 'R': apply_clear(); break;
        }

        if (flags & F_LAST) {
            last_commit_ts_ = ts_event;
            committed_ = true;

            while (next_snap_ts_ < rth_close_ && next_snap_ts_ <= ts_event) {
                emit_snapshot(next_snap_ts_);
                next_snap_ts_ += SNAPSHOT_INTERVAL;
            }
        }
    }

    std::vector<BookSnapshot>& snapshots() { return snapshots_; }

private:
    static constexpr uint64_t SNAPSHOT_INTERVAL = 100'000'000ULL;  // 100ms

    uint32_t instrument_id_;
    uint64_t rth_open_;
    uint64_t rth_close_;
    uint64_t next_snap_ts_;
    uint64_t last_commit_ts_ = 0;
    bool committed_ = false;

    struct OrderInfo { char side; int64_t price; uint32_t size; };
    std::unordered_map<uint64_t, OrderInfo> orders_;
    std::map<int64_t, uint32_t> bid_levels_;
    std::map<int64_t, uint32_t> ask_levels_;

    struct TradeRecord { float price; float size; float aggressor_side; };
    std::deque<TradeRecord> trades_;

    float last_mid_ = 0.0f;
    float last_spread_ = 0.0f;
    bool ever_had_both_ = false;

    std::vector<BookSnapshot> snapshots_;

    static float fixed_to_float(int64_t fixed) {
        return static_cast<float>(static_cast<double>(fixed) / 1e9);
    }

    void apply_add(uint64_t order_id, char side, int64_t price, uint32_t size) {
        orders_[order_id] = {side, price, size};
        auto& levels = (side == 'B') ? bid_levels_ : ask_levels_;
        levels[price] += size;
    }

    void apply_cancel(uint64_t order_id) {
        auto it = orders_.find(order_id);
        if (it == orders_.end()) return;
        auto& info = it->second;
        auto& levels = (info.side == 'B') ? bid_levels_ : ask_levels_;
        auto lvl = levels.find(info.price);
        if (lvl != levels.end()) {
            if (lvl->second <= info.size) levels.erase(lvl);
            else lvl->second -= info.size;
        }
        orders_.erase(it);
    }

    void apply_modify(uint64_t order_id, char side, int64_t new_price, uint32_t new_size) {
        auto it = orders_.find(order_id);
        if (it != orders_.end()) {
            auto& info = it->second;
            auto& levels = (info.side == 'B') ? bid_levels_ : ask_levels_;
            auto lvl = levels.find(info.price);
            if (lvl != levels.end()) {
                if (lvl->second <= info.size) levels.erase(lvl);
                else lvl->second -= info.size;
            }
        }
        orders_[order_id] = {side, new_price, new_size};
        auto& levels = (side == 'B') ? bid_levels_ : ask_levels_;
        levels[new_price] += new_size;
    }

    void apply_trade(char side, int64_t price, uint32_t size) {
        float agg = (side == 'B') ? 1.0f : -1.0f;
        trades_.push_back({fixed_to_float(price), static_cast<float>(size), agg});
        if (trades_.size() > 50) trades_.pop_front();
    }

    void apply_fill(uint64_t order_id, uint32_t remaining_size) {
        auto it = orders_.find(order_id);
        if (it == orders_.end()) return;
        auto& info = it->second;
        auto& levels = (info.side == 'B') ? bid_levels_ : ask_levels_;
        auto lvl = levels.find(info.price);
        if (lvl != levels.end()) {
            if (lvl->second <= info.size) levels.erase(lvl);
            else lvl->second -= info.size;
        }
        if (remaining_size == 0) {
            orders_.erase(it);
        } else {
            info.size = remaining_size;
            levels[info.price] += remaining_size;
        }
    }

    void apply_clear() {
        orders_.clear();
        bid_levels_.clear();
        ask_levels_.clear();
    }

    void emit_snapshot(uint64_t ts) {
        if (!committed_) return;

        bool has_bid = !bid_levels_.empty();
        bool has_ask = !ask_levels_.empty();

        if (has_bid && has_ask) ever_had_both_ = true;
        if (!has_bid || !has_ask) {
            if (!ever_had_both_) return;
        }

        constexpr int DEPTH = 10;

        BookSnapshot snap{};
        snap.timestamp = ts;

        int idx = 0;
        for (auto it = bid_levels_.rbegin(); it != bid_levels_.rend() && idx < DEPTH; ++it, ++idx) {
            snap.bids[idx][0] = fixed_to_float(it->first);
            snap.bids[idx][1] = static_cast<float>(it->second);
        }

        idx = 0;
        for (auto it = ask_levels_.begin(); it != ask_levels_.end() && idx < DEPTH; ++it, ++idx) {
            snap.asks[idx][0] = fixed_to_float(it->first);
            snap.asks[idx][1] = static_cast<float>(it->second);
        }

        if (has_bid && has_ask) {
            float best_bid = fixed_to_float(bid_levels_.rbegin()->first);
            float best_ask = fixed_to_float(ask_levels_.begin()->first);
            snap.mid_price = (best_bid + best_ask) / 2.0f;
            snap.spread = best_ask - best_bid;
            last_mid_ = snap.mid_price;
            last_spread_ = snap.spread;
        } else {
            snap.mid_price = last_mid_;
            snap.spread = last_spread_;
        }

        // Trades
        size_t count = trades_.size();
        constexpr int TBUF = 50;
        size_t start = TBUF - count;
        for (size_t i = 0; i < count; ++i) {
            snap.trades[start + i][0] = trades_[i].price;
            snap.trades[start + i][1] = trades_[i].size;
            snap.trades[start + i][2] = trades_[i].aggressor_side;
        }

        snap.time_of_day = time_utils::compute_time_of_day(ts);
        snapshots_.push_back(snap);
    }
};

// ===========================================================================
// Constants
// ===========================================================================
namespace {

const std::string DATA_DIR = "DATA/GLBX-20260207-L953CAPU5B";

// Front-month MES contracts for 2022
struct QuarterlyContract {
    std::string symbol;
    uint32_t instrument_id;
    int start_date, end_date, rollover_date;
};

const std::vector<QuarterlyContract> MES_CONTRACTS = {
    {"MESH2", 11355, 20220103, 20220318, 20220318},
    {"MESM2", 13615, 20220319, 20220617, 20220617},
    {"MESU2", 10039, 20220618, 20220916, 20220916},
    {"MESZ2", 10299, 20220917, 20221230, 20221216},
};

struct Quarter {
    int start, end;
    std::string label;
};

const std::vector<Quarter> QUARTERS = {
    {20220103, 20220331, "Q1"},
    {20220401, 20220630, "Q2"},
    {20220701, 20220930, "Q3"},
    {20221003, 20221230, "Q4"},
};

}  // anonymous namespace

// ===========================================================================
// Utility functions (shared with subordination_test.cpp)
// ===========================================================================
std::string date_to_string(int date) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%04d%02d%02d", date / 10000, (date / 100) % 100, date % 100);
    return buf;
}

uint64_t date_to_midnight_ns(int date) {
    int y = date / 10000, m = (date / 100) % 100, d = date % 100;
    struct tm ref_tm = {};
    ref_tm.tm_year = 2022 - 1900; ref_tm.tm_mon = 0; ref_tm.tm_mday = 3; ref_tm.tm_hour = 5;
    struct tm date_tm = {};
    date_tm.tm_year = y - 1900; date_tm.tm_mon = m - 1; date_tm.tm_mday = d; date_tm.tm_hour = 5;
    time_t ref_t = timegm(&ref_tm);
    time_t date_t = timegm(&date_tm);
    int64_t diff_seconds = static_cast<int64_t>(date_t) - static_cast<int64_t>(ref_t);
    return time_utils::REF_MIDNIGHT_ET_NS +
           static_cast<uint64_t>(diff_seconds) * time_utils::NS_PER_SEC;
}

uint32_t get_instrument_id(int date) {
    for (const auto& c : MES_CONTRACTS) {
        if (date >= c.start_date && date <= c.end_date) return c.instrument_id;
    }
    return 13615;  // fallback
}

std::string get_contract_symbol(int date) {
    for (const auto& c : MES_CONTRACTS) {
        if (date >= c.start_date && date <= c.end_date) return c.symbol;
    }
    return "MESM2";
}

RolloverCalendar build_rollover_calendar() {
    RolloverCalendar cal;
    for (const auto& c : MES_CONTRACTS) {
        cal.add_contract({c.symbol, c.instrument_id, c.start_date, c.end_date, c.rollover_date});
    }
    return cal;
}

std::vector<int> get_available_weekdays() {
    std::vector<int> dates;
    for (auto& entry : std::filesystem::directory_iterator(DATA_DIR)) {
        std::string fname = entry.path().filename().string();
        if (fname.find(".mbo.dbn.zst") == std::string::npos) continue;
        size_t pos = fname.find("20");
        if (pos == std::string::npos) continue;
        std::string date_str = fname.substr(pos, 8);
        int date = std::stoi(date_str);
        int y = date / 10000, m = (date / 100) % 100, d = date % 100;
        struct tm tm_date = {};
        tm_date.tm_year = y - 1900; tm_date.tm_mon = m - 1; tm_date.tm_mday = d;
        mktime(&tm_date);
        if (tm_date.tm_wday == 0 || tm_date.tm_wday == 6) continue;
        dates.push_back(date);
    }
    std::sort(dates.begin(), dates.end());
    return dates;
}

std::vector<int> select_stratified_days(int n_per_quarter) {
    auto rollover = build_rollover_calendar();
    auto all_dates = get_available_weekdays();
    std::vector<int> selected;
    for (const auto& q : QUARTERS) {
        std::vector<int> quarter_dates;
        for (int d : all_dates) {
            if (d >= q.start && d <= q.end && !rollover.is_excluded(d))
                quarter_dates.push_back(d);
        }
        if (quarter_dates.empty()) continue;
        int n = static_cast<int>(quarter_dates.size());
        int count = std::min(n_per_quarter, n);
        for (int i = 0; i < count; ++i) {
            int idx = (i * (n - 1)) / (count - 1 > 0 ? count - 1 : 1);
            selected.push_back(quarter_dates[idx]);
        }
    }
    return selected;
}

// ===========================================================================
// Process one day — build bars, run both oracle modes
// ===========================================================================
struct DayOracleResult {
    int date = 0;
    DayResult fth;  // First-to-hit
    DayResult tb;   // Triple barrier
    bool valid = false;
};

DayOracleResult process_day(int date, const ExecutionCosts& costs) {
    DayOracleResult result;
    result.date = date;

    std::string date_str = date_to_string(date);
    std::string filepath = DATA_DIR + "/glbx-mdp3-" + date_str + ".mbo.dbn.zst";

    if (!std::filesystem::exists(filepath)) {
        std::cerr << "  SKIP: " << filepath << " not found\n";
        return result;
    }

    std::cout << "  " << date_str << ": " << std::flush;

    uint64_t midnight = date_to_midnight_ns(date);
    uint64_t rth_open = midnight + 9ULL * time_utils::NS_PER_HOUR + 30ULL * 60 * time_utils::NS_PER_SEC;
    uint64_t rth_close = midnight + 16ULL * time_utils::NS_PER_HOUR;

    uint32_t inst_id = get_instrument_id(date);
    StreamingBookBuilder builder(inst_id, rth_open, rth_close);
    databento::DbnFileStore store{std::filesystem::path(filepath)};

    while (const auto* record = store.NextRecord()) {
        if (const auto* mbo = record->GetIf<databento::MboMsg>()) {
            uint64_t ts_ns = static_cast<uint64_t>(
                mbo->hd.ts_event.time_since_epoch().count());
            uint8_t flags_raw = static_cast<uint8_t>(mbo->flags.Raw());
            builder.process_event(
                ts_ns, mbo->order_id, mbo->hd.instrument_id,
                static_cast<char>(mbo->action), static_cast<char>(mbo->side),
                mbo->price, mbo->size, flags_raw
            );
        }
    }

    auto& snapshots = builder.snapshots();
    std::cout << snapshots.size() << " snaps, " << std::flush;

    if (snapshots.empty()) {
        std::cout << "skipped (no snapshots)\n";
        return result;
    }

    // Build time_5s bars
    auto bar_builder = BarFactory::create("time", 5.0);
    std::vector<Bar> bars;
    for (const auto& snap : snapshots) {
        if (auto bar = bar_builder->on_snapshot(snap)) {
            bars.push_back(*bar);
        }
    }
    if (auto final_bar = bar_builder->flush()) {
        bars.push_back(*final_bar);
    }

    std::cout << bars.size() << " bars, " << std::flush;

    if (bars.size() < 10) {
        std::cout << "skipped (too few bars)\n";
        return result;
    }

    // Run FTH oracle
    OracleConfig fth_cfg;
    fth_cfg.label_method = OracleConfig::LabelMethod::FIRST_TO_HIT;
    fth_cfg.target_ticks = 10;
    fth_cfg.stop_ticks = 5;
    fth_cfg.take_profit_ticks = 20;
    fth_cfg.volume_horizon = 500;
    fth_cfg.max_time_horizon_s = 300;
    fth_cfg.tick_size = 0.25f;

    OracleReplay fth_replay(fth_cfg, costs);
    auto fth_result = fth_replay.run(bars);

    // Run TB oracle
    OracleConfig tb_cfg = fth_cfg;
    tb_cfg.label_method = OracleConfig::LabelMethod::TRIPLE_BARRIER;

    OracleReplay tb_replay(tb_cfg, costs);
    auto tb_result = tb_replay.run(bars);

    std::cout << "FTH=" << fth_result.total_trades << " trades, "
              << "TB=" << tb_result.total_trades << " trades\n";

    // Package into DayResult structs for the aggregation API
    result.fth.date = date;
    result.fth.result = std::move(fth_result);
    result.fth.skipped = false;
    result.fth.contract_symbol = get_contract_symbol(date);
    result.fth.instrument_id = inst_id;
    result.fth.bar_count = static_cast<int>(bars.size());

    result.tb.date = date;
    result.tb.result = std::move(tb_result);
    result.tb.skipped = false;
    result.tb.contract_symbol = get_contract_symbol(date);
    result.tb.instrument_id = inst_id;
    result.tb.bar_count = static_cast<int>(bars.size());

    result.valid = true;
    return result;
}

// ===========================================================================
// Main
// ===========================================================================
int main() {
    std::cout << "=== Oracle Expectancy Extraction ===\n\n";

    // --- Execution costs (spec defaults) ---
    ExecutionCosts costs;
    costs.commission_per_side = 0.62f;
    costs.spread_model = ExecutionCosts::SpreadModel::FIXED;
    costs.fixed_spread_ticks = 1;
    costs.slippage_ticks = 0;
    costs.contract_multiplier = 5.0f;
    costs.tick_size = 0.25f;
    costs.tick_value = 1.25f;

    // --- Day selection: 20 stratified days (5 per quarter) ---
    std::cout << "Selecting 20 stratified days...\n";
    auto selected_days = select_stratified_days(5);
    std::cout << "Selected " << selected_days.size() << " days:";
    for (int d : selected_days) std::cout << " " << date_to_string(d);
    std::cout << "\n\n";

    if (selected_days.size() < 10) {
        std::cerr << "ERROR: Need >= 10 days, got " << selected_days.size() << "\n";
        return 1;
    }

    // --- Process days ---
    std::cout << "Processing days...\n";
    std::vector<DayResult> fth_results;
    std::vector<DayResult> tb_results;
    std::vector<int> dates_processed;

    for (int date : selected_days) {
        auto day = process_day(date, costs);
        if (day.valid) {
            fth_results.push_back(std::move(day.fth));
            tb_results.push_back(std::move(day.tb));
            dates_processed.push_back(date);
        }
    }

    std::cout << "\nProcessed " << dates_processed.size() << " days.\n\n";

    if (dates_processed.size() < 5) {
        std::cerr << "ERROR: Too few valid days (" << dates_processed.size() << ").\n";
        return 1;
    }

    // --- Aggregate using tested API ---
    auto report = oracle_expectancy::aggregate_day_results(
        fth_results, tb_results, dates_processed);

    // --- Print summary ---
    std::cout << "=== Results ===\n\n";

    std::cout << "Days processed: " << report.days_processed
              << ", skipped: " << report.days_skipped << "\n\n";

    auto print_result = [](const char* label, const BacktestResult& r) {
        std::printf("  %s:\n", label);
        std::printf("    Trades: %d (%.1f/day)\n", r.total_trades, r.trades_per_day);
        std::printf("    Win rate: %.1f%%\n", r.win_rate * 100.0f);
        std::printf("    Gross PnL: $%.2f\n", r.gross_pnl);
        std::printf("    Net PnL:   $%.2f\n", r.net_pnl);
        std::printf("    Expectancy: $%.2f per trade\n", r.expectancy);
        std::printf("    Profit factor: %.2f\n", r.profit_factor);
        std::printf("    Sharpe: %.3f\n", r.sharpe);
        std::printf("    Max drawdown: $%.2f\n", r.max_drawdown);
        std::printf("    Avg bars held: %.1f\n", r.avg_bars_held);
        std::printf("    Avg duration: %.1fs\n", r.avg_duration_s);
        std::printf("    Hold fraction: %.1f%%\n", r.hold_fraction * 100.0f);
    };

    print_result("First-to-Hit", report.first_to_hit);
    std::cout << "\n";
    print_result("Triple Barrier", report.triple_barrier);
    std::cout << "\n";

    // Per-quarter summary
    std::cout << "Per-quarter (FTH):\n";
    for (const auto& [q, r] : report.fth_per_quarter) {
        std::printf("  %s: %d trades, expectancy=$%.2f, PF=%.2f\n",
                    q.c_str(), r.total_trades, r.expectancy, r.profit_factor);
    }
    std::cout << "\nPer-quarter (TB):\n";
    for (const auto& [q, r] : report.tb_per_quarter) {
        std::printf("  %s: %d trades, expectancy=$%.2f, PF=%.2f\n",
                    q.c_str(), r.total_trades, r.expectancy, r.profit_factor);
    }

    // --- Success criteria check (from TRAJECTORY §9.4) ---
    std::cout << "\n=== Success Criteria Check ===\n";
    auto check = [](const char* label, const BacktestResult& r) {
        bool exp_ok = r.expectancy > 0.50f;
        bool pf_ok = r.profit_factor > 1.3f;
        bool wr_ok = r.win_rate > 0.45f;
        bool net_ok = r.net_pnl > 0.0f;
        bool dd_ok = (r.expectancy > 0.0f) && (r.max_drawdown < 50.0f * r.expectancy);
        bool tpd_ok = r.trades_per_day > 10.0f;

        std::printf("  %s:\n", label);
        std::printf("    Expectancy > $0.50:      %s ($%.2f)\n", exp_ok ? "PASS" : "FAIL", r.expectancy);
        std::printf("    Profit factor > 1.3:     %s (%.2f)\n", pf_ok ? "PASS" : "FAIL", r.profit_factor);
        std::printf("    Win rate > 45%%:           %s (%.1f%%)\n", wr_ok ? "PASS" : "FAIL", r.win_rate * 100.0f);
        std::printf("    Net PnL > 0:             %s ($%.2f)\n", net_ok ? "PASS" : "FAIL", r.net_pnl);
        std::printf("    Drawdown < 50x expect:   %s ($%.2f < $%.2f)\n",
                    dd_ok ? "PASS" : "FAIL", r.max_drawdown, 50.0f * r.expectancy);
        std::printf("    Trades/day > 10:         %s (%.1f)\n", tpd_ok ? "PASS" : "FAIL", r.trades_per_day);

        bool all_pass = exp_ok && pf_ok && wr_ok && net_ok && dd_ok && tpd_ok;
        std::printf("    Overall: %s\n", all_pass ? "ALL PASS" : "SOME FAIL");
        return all_pass;
    };

    bool fth_pass = check("First-to-Hit", report.first_to_hit);
    bool tb_pass = check("Triple Barrier", report.triple_barrier);

    std::string verdict;
    if (fth_pass || tb_pass) {
        verdict = "GO";
        std::cout << "\n  VERDICT: GO — Oracle has positive expectancy.\n";
    } else if (report.first_to_hit.expectancy > 0.0f || report.triple_barrier.expectancy > 0.0f) {
        verdict = "CONDITIONAL_GO";
        std::cout << "\n  VERDICT: CONDITIONAL GO — Positive expectancy but not all criteria met.\n";
    } else {
        verdict = "NO_GO";
        std::cout << "\n  VERDICT: NO GO — Oracle does not produce positive expectancy.\n";
    }

    // --- Write JSON ---
    std::filesystem::create_directories(".kit/results/oracle-expectancy");

    // Use the tested to_json serializer for the report
    std::string json = oracle_expectancy::to_json(report);

    std::ofstream out(".kit/results/oracle-expectancy/metrics.json");
    out << json;
    out.close();

    // Write a separate summary with days list and verdict
    std::ofstream summary(".kit/results/oracle-expectancy/summary.json");
    summary << "{\n";
    summary << "  \"experiment\": \"oracle_expectancy\",\n";
    summary << "  \"bar_type\": \"time_5s\",\n";
    summary << "  \"n_days\": " << report.days_processed << ",\n";
    summary << "  \"days_used\": [";
    for (size_t i = 0; i < dates_processed.size(); ++i) {
        if (i) summary << ", ";
        summary << dates_processed[i];
    }
    summary << "],\n";
    summary << "  \"fth_expectancy\": " << report.first_to_hit.expectancy << ",\n";
    summary << "  \"fth_net_pnl\": " << report.first_to_hit.net_pnl << ",\n";
    summary << "  \"fth_total_trades\": " << report.first_to_hit.total_trades << ",\n";
    summary << "  \"fth_win_rate\": " << report.first_to_hit.win_rate << ",\n";
    summary << "  \"fth_profit_factor\": " << report.first_to_hit.profit_factor << ",\n";
    summary << "  \"fth_sharpe\": " << report.first_to_hit.sharpe << ",\n";
    summary << "  \"tb_expectancy\": " << report.triple_barrier.expectancy << ",\n";
    summary << "  \"tb_net_pnl\": " << report.triple_barrier.net_pnl << ",\n";
    summary << "  \"tb_total_trades\": " << report.triple_barrier.total_trades << ",\n";
    summary << "  \"tb_win_rate\": " << report.triple_barrier.win_rate << ",\n";
    summary << "  \"tb_profit_factor\": " << report.triple_barrier.profit_factor << ",\n";
    summary << "  \"tb_sharpe\": " << report.triple_barrier.sharpe << ",\n";
    summary << "  \"verdict\": \"" << verdict << "\"\n";
    summary << "}\n";
    summary.close();

    std::cout << "\nResults written to .kit/results/oracle-expectancy/\n";
    std::cout << "=== Oracle expectancy extraction complete. ===\n";
    return 0;
}
