// subordination_test.cpp — Phase R1: Subordination Hypothesis Test
// Spec: .kit/experiments/subordination-test.md
//
// Tests Clark (1973) / Ané & Geman (2000) subordination hypothesis:
// Do event-driven bars (volume, tick, dollar) produce more IID Gaussian
// returns than fixed-time bars for MES microstructure data?
//
// Uses a streaming book builder that emits snapshots on-the-fly to avoid
// the O(n) committed state memory of the standard BookBuilder.

#include "bars/bar_factory.hpp"
#include "analysis/statistical_tests.hpp"
#include "analysis/multiple_comparison.hpp"
#include "backtest/rollover.hpp"
#include "time_utils.hpp"

#include <databento/dbn_file_store.hpp>
#include <databento/record.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// ===========================================================================
// StreamingBookBuilder — emits snapshots incrementally, O(1) committed state
// ===========================================================================
class StreamingBookBuilder {
public:
    explicit StreamingBookBuilder(uint32_t instrument_id,
                                   uint64_t rth_open, uint64_t rth_close)
        : instrument_id_(instrument_id),
          rth_open_(rth_open), rth_close_(rth_close),
          next_snap_ts_(rth_open) {}

    // Process one MBO event; may emit snapshots if time boundaries are crossed
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

            // Emit any 100ms snapshots whose boundary has been passed
            while (next_snap_ts_ < rth_close_ && next_snap_ts_ <= ts_event) {
                emit_snapshot(next_snap_ts_);
                next_snap_ts_ += SNAPSHOT_INTERVAL;
            }
        }
    }

    // Get all emitted snapshots (moved out)
    std::vector<BookSnapshot>& snapshots() { return snapshots_; }

private:
    static constexpr uint64_t SNAPSHOT_INTERVAL = 100'000'000ULL;  // 100ms

    uint32_t instrument_id_;
    uint64_t rth_open_;
    uint64_t rth_close_;
    uint64_t next_snap_ts_;
    uint64_t last_commit_ts_ = 0;
    bool committed_ = false;

    // Book state
    struct OrderInfo { char side; int64_t price; uint32_t size; };
    std::unordered_map<uint64_t, OrderInfo> orders_;
    std::map<int64_t, uint32_t> bid_levels_;
    std::map<int64_t, uint32_t> ask_levels_;

    // Trade buffer
    struct TradeRecord { float price; float size; float aggressor_side; };
    std::deque<TradeRecord> trades_;

    // Carry-forward
    float last_mid_ = 0.0f;
    float last_spread_ = 0.0f;
    bool ever_had_both_ = false;

    // Output
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
        constexpr int TBUF = 50;

        BookSnapshot snap{};
        snap.timestamp = ts;

        // Bids (descending by price)
        int idx = 0;
        for (auto it = bid_levels_.rbegin(); it != bid_levels_.rend() && idx < DEPTH; ++it, ++idx) {
            snap.bids[idx][0] = fixed_to_float(it->first);
            snap.bids[idx][1] = static_cast<float>(it->second);
        }

        // Asks (ascending by price)
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
constexpr int WARMUP_BARS = 20;

// Front-month MES contracts for 2022 with correct instrument IDs from symbology.json
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

uint32_t get_instrument_id(int date) {
    for (const auto& c : MES_CONTRACTS) {
        if (date >= c.start_date && date <= c.end_date) return c.instrument_id;
    }
    return 13615;  // fallback
}

struct BarConfig {
    std::string name;
    std::string type;
    double threshold;
};

const std::vector<BarConfig> BAR_CONFIGS = {
    {"vol_50",      "volume", 50.0},
    {"vol_100",     "volume", 100.0},
    {"vol_200",     "volume", 200.0},
    {"tick_25",     "tick",   25.0},
    {"tick_50",     "tick",   50.0},
    {"tick_100",    "tick",   100.0},
    {"dollar_25k",  "dollar", 25000.0},
    {"dollar_50k",  "dollar", 50000.0},
    {"dollar_100k", "dollar", 100000.0},
    {"time_1s",     "time",   1.0},
    {"time_5s",     "time",   5.0},
    {"time_60s",    "time",   60.0},
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
// Utility functions
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
    RolloverCalendar rollover;
    for (const auto& c : MES_CONTRACTS) {
        rollover.add_contract({c.symbol, c.instrument_id, c.start_date, c.end_date, c.rollover_date});
    }
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
// Process one day using streaming book builder
// ===========================================================================
struct DayResult {
    int date;
    std::map<std::string, std::vector<float>> returns;
    std::map<std::string, int> bar_counts;
};

DayResult process_day(int date) {
    DayResult result;
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

    // Stream MBO events through the streaming builder (per-contract instrument ID)
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
        std::cout << "skipped\n";
        return result;
    }

    // Build bars for each config
    for (const auto& cfg : BAR_CONFIGS) {
        auto bar_builder = BarFactory::create(cfg.type, cfg.threshold);
        if (!bar_builder) continue;

        std::vector<Bar> bars;
        for (const auto& snap : snapshots) {
            if (auto bar = bar_builder->on_snapshot(snap)) {
                bars.push_back(*bar);
            }
        }
        if (auto final_bar = bar_builder->flush()) {
            bars.push_back(*final_bar);
        }

        // Log-returns, discard first WARMUP_BARS
        std::vector<float> returns;
        int start_idx = std::min(WARMUP_BARS, static_cast<int>(bars.size()));
        for (int i = start_idx + 1; i < static_cast<int>(bars.size()); ++i) {
            if (bars[i].close_mid > 0.0f && bars[i - 1].close_mid > 0.0f) {
                float r = std::log(bars[i].close_mid / bars[i - 1].close_mid);
                if (std::isfinite(r)) returns.push_back(r);
            }
        }

        result.returns[cfg.name] = returns;
        result.bar_counts[cfg.name] = static_cast<int>(bars.size());
    }

    std::cout << "done\n";
    return result;
}

// ===========================================================================
// Statistical helpers
// ===========================================================================
struct WilcoxonResult { float statistic; float p_value; int n; };

WilcoxonResult wilcoxon_signed_rank(const std::vector<float>& x,
                                     const std::vector<float>& y) {
    std::vector<float> diffs;
    for (size_t i = 0; i < std::min(x.size(), y.size()); ++i) {
        float d = x[i] - y[i];
        if (std::abs(d) > 1e-12f) diffs.push_back(d);
    }
    int n = static_cast<int>(diffs.size());
    if (n < 2) return {0.0f, 1.0f, n};

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return std::abs(diffs[a]) < std::abs(diffs[b]);
    });

    std::vector<float> ranks(n);
    int i = 0;
    while (i < n) {
        int j = i;
        while (j < n - 1 && std::abs(std::abs(diffs[order[j + 1]]) - std::abs(diffs[order[j]])) < 1e-12f)
            ++j;
        float avg_rank = static_cast<float>(i + j) / 2.0f + 1.0f;
        for (int k = i; k <= j; ++k) ranks[order[k]] = avg_rank;
        i = j + 1;
    }

    float T_plus = 0.0f, T_minus = 0.0f;
    for (int k = 0; k < n; ++k) {
        if (diffs[k] > 0) T_plus += ranks[k]; else T_minus += ranks[k];
    }
    float T = std::min(T_plus, T_minus);
    float nf = static_cast<float>(n);
    float mean_T = nf * (nf + 1.0f) / 4.0f;
    float var_T = nf * (nf + 1.0f) * (2.0f * nf + 1.0f) / 24.0f;
    if (var_T < 1e-12f) return {T, 1.0f, n};
    float z = (T - mean_T) / std::sqrt(var_T);
    float p = 2.0f * detail::normal_cdf(-std::abs(z));
    return {T, std::min(1.0f, p), n};
}

float wilcoxon_one_sided_p(const std::vector<float>& ev, const std::vector<float>& tv, bool test_less) {
    auto result = wilcoxon_signed_rank(ev, tv);
    std::vector<float> diffs;
    for (size_t i = 0; i < std::min(ev.size(), tv.size()); ++i) {
        float d = ev[i] - tv[i];
        if (std::abs(d) > 1e-12f) diffs.push_back(d);
    }
    float sum_neg = 0.0f, sum_pos = 0.0f;
    for (float d : diffs) { if (d < 0) sum_neg += 1.0f; else sum_pos += 1.0f; }
    if (test_less)
        return (sum_neg >= sum_pos) ? result.p_value / 2.0f : 1.0f - result.p_value / 2.0f;
    else
        return (sum_pos >= sum_neg) ? result.p_value / 2.0f : 1.0f - result.p_value / 2.0f;
}

float hodges_lehmann(const std::vector<float>& x, const std::vector<float>& y) {
    std::vector<float> diffs;
    size_t n = std::min(x.size(), y.size());
    for (size_t i = 0; i < n; ++i) diffs.push_back(x[i] - y[i]);
    if (diffs.empty()) return 0.0f;
    std::sort(diffs.begin(), diffs.end());
    size_t m = diffs.size();
    if (m % 2 == 0) return (diffs[m / 2 - 1] + diffs[m / 2]) / 2.0f;
    return diffs[m / 2];
}

float ks_test_vs_normal(const std::vector<float>& data) {
    if (data.size() < 2) return 0.0f;
    float mean = 0.0f;
    for (float v : data) mean += v;
    mean /= static_cast<float>(data.size());
    float var = 0.0f;
    for (float v : data) { float d = v - mean; var += d * d; }
    var /= static_cast<float>(data.size());
    float sd = std::sqrt(var);
    if (sd < 1e-12f) return 0.0f;
    std::vector<float> sorted = data;
    std::sort(sorted.begin(), sorted.end());
    float D = 0.0f, n = static_cast<float>(sorted.size());
    for (size_t i = 0; i < sorted.size(); ++i) {
        float cdf = detail::normal_cdf((sorted[i] - mean) / sd);
        D = std::max(D, std::abs(static_cast<float>(i + 1) / n - cdf));
        D = std::max(D, std::abs(static_cast<float>(i) / n - cdf));
    }
    return D;
}

std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else out += c;
    }
    return out;
}

std::string to_json(float v) {
    if (std::isnan(v) || std::isinf(v)) return "null";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.8g", static_cast<double>(v));
    return buf;
}

// ===========================================================================
// Main
// ===========================================================================
int main() {
    std::cout << "=== Phase R1: Subordination Hypothesis Test ===\n\n";

    // --- Day selection ---
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
    std::vector<DayResult> day_results;
    for (int date : selected_days) {
        auto result = process_day(date);
        if (!result.returns.empty()) day_results.push_back(std::move(result));
    }
    int n_days = static_cast<int>(day_results.size());
    std::cout << "\nProcessed " << n_days << " days.\n\n";

    if (n_days < 5) { std::cerr << "ERROR: Too few days.\n"; return 1; }

    // --- Compute metrics ---
    std::cout << "Computing metrics...\n";

    struct ConfigMetrics {
        std::string name;
        float jb_stat = 0, jb_p = 1;
        float arch_stat = 0, arch_p = 1;
        std::vector<float> acf_abs;
        std::vector<TestResult> ljung_box;
        float ar_r2 = 0, bar_count_cv = 0, ks_stat = 0;
        int total_returns = 0;
        float median_daily_bars = 0;
        std::vector<float> daily_jb, daily_arch, daily_acf1, daily_ar_r2;
    };

    std::vector<ConfigMetrics> all_metrics;

    for (const auto& cfg : BAR_CONFIGS) {
        ConfigMetrics cm;
        cm.name = cfg.name;
        std::vector<float> pooled;
        std::vector<int> daily_counts;

        for (const auto& dr : day_results) {
            auto it = dr.returns.find(cfg.name);
            if (it == dr.returns.end()) continue;
            const auto& rets = it->second;
            pooled.insert(pooled.end(), rets.begin(), rets.end());

            auto bc = dr.bar_counts.find(cfg.name);
            if (bc != dr.bar_counts.end()) daily_counts.push_back(bc->second);

            if (rets.size() >= 20) {
                cm.daily_jb.push_back(jarque_bera_test(rets).statistic);
                cm.daily_arch.push_back(arch_lm_test(rets).statistic);
                std::vector<float> abs_r(rets.size());
                for (size_t i = 0; i < rets.size(); ++i) abs_r[i] = std::abs(rets[i]);
                auto a = compute_acf(abs_r, {1});
                cm.daily_acf1.push_back(a.empty() ? 0.0f : a[0]);
                float ar = compute_ar_r2(rets, 10);
                cm.daily_ar_r2.push_back(std::isnan(ar) ? 0.0f : ar);
            }
        }

        cm.total_returns = static_cast<int>(pooled.size());
        if (!pooled.empty()) {
            auto jb = jarque_bera_test(pooled); cm.jb_stat = jb.statistic; cm.jb_p = jb.p_value;
            auto ar = arch_lm_test(pooled); cm.arch_stat = ar.statistic; cm.arch_p = ar.p_value;
            std::vector<float> abs_p(pooled.size());
            for (size_t i = 0; i < pooled.size(); ++i) abs_p[i] = std::abs(pooled[i]);
            cm.acf_abs = compute_acf(abs_p, {1, 5, 10});
            cm.ljung_box = ljung_box_test(pooled, {1, 5, 10});
            cm.ar_r2 = compute_ar_r2(pooled, 10);
            if (std::isnan(cm.ar_r2)) cm.ar_r2 = 0;
            cm.ks_stat = ks_test_vs_normal(pooled);
        }
        cm.bar_count_cv = compute_bar_count_cv(daily_counts);
        if (!daily_counts.empty()) {
            auto sc = daily_counts; std::sort(sc.begin(), sc.end());
            cm.median_daily_bars = static_cast<float>(sc[sc.size() / 2]);
        }

        std::cout << "  " << cfg.name << ": N=" << cm.total_returns
                  << " JB=" << cm.jb_stat << " ARCH=" << cm.arch_stat
                  << " ACF1=" << (cm.acf_abs.empty() ? 0.f : cm.acf_abs[0])
                  << " AR=" << cm.ar_r2 << " CV=" << cm.bar_count_cv << "\n";

        all_metrics.push_back(std::move(cm));
    }

    // --- Rankings ---
    int nc = static_cast<int>(all_metrics.size());
    auto rank_asc = [&](auto getter) {
        std::vector<int> ord(nc); std::iota(ord.begin(), ord.end(), 0);
        std::sort(ord.begin(), ord.end(), [&](int a, int b) { return getter(all_metrics[a]) < getter(all_metrics[b]); });
        std::vector<int> r(nc); for (int i = 0; i < nc; ++i) r[ord[i]] = i + 1; return r;
    };
    auto rank_desc = [&](auto getter) {
        std::vector<int> ord(nc); std::iota(ord.begin(), ord.end(), 0);
        std::sort(ord.begin(), ord.end(), [&](int a, int b) { return getter(all_metrics[a]) > getter(all_metrics[b]); });
        std::vector<int> r(nc); for (int i = 0; i < nc; ++i) r[ord[i]] = i + 1; return r;
    };

    auto jbR = rank_asc([](const ConfigMetrics& m) { return m.jb_stat; });
    auto arR = rank_asc([](const ConfigMetrics& m) { return m.arch_stat; });
    auto a1R = rank_asc([](const ConfigMetrics& m) { return m.acf_abs.empty() ? 0.f : m.acf_abs[0]; });
    auto a5R = rank_asc([](const ConfigMetrics& m) { return m.acf_abs.size() > 1 ? m.acf_abs[1] : 0.f; });
    auto a10R = rank_asc([](const ConfigMetrics& m) { return m.acf_abs.size() > 2 ? m.acf_abs[2] : 0.f; });
    auto arR2 = rank_desc([](const ConfigMetrics& m) { return m.ar_r2; });
    auto cvR = rank_asc([](const ConfigMetrics& m) { return m.bar_count_cv; });

    std::vector<float> mean_ranks(nc);
    for (int i = 0; i < nc; ++i) {
        mean_ranks[i] = static_cast<float>(jbR[i] + arR[i] + a1R[i] + a5R[i] + a10R[i] + arR2[i] + cvR[i]) / 7.0f;
    }

    int best_idx = 0;
    for (int i = 1; i < nc; ++i) {
        if (mean_ranks[i] < mean_ranks[best_idx]) best_idx = i;
        else if (std::abs(mean_ranks[i] - mean_ranks[best_idx]) < 1.0f &&
                 all_metrics[i].bar_count_cv < all_metrics[best_idx].bar_count_cv)
            best_idx = i;
    }

    std::cout << "\nRankings:\n";
    for (int i = 0; i < nc; ++i)
        std::cout << "  " << all_metrics[i].name << ": " << mean_ranks[i] << "\n";
    std::cout << "Best: " << all_metrics[best_idx].name << " (" << mean_ranks[best_idx] << ")\n\n";

    // --- Pairwise Wilcoxon ---
    struct PairComp {
        std::string ev, tv, metric;
        float median_diff, hl, raw_p, corrected_p;
        bool sig;
    };
    std::vector<PairComp> pw;
    std::map<std::string, std::vector<float>> raw_p_map;
    std::map<std::string, std::vector<size_t>> idx_map;

    auto match_time = [&](int ei) {
        float em = all_metrics[ei].median_daily_bars;
        int bt = 9; float bd = std::abs(em - all_metrics[9].median_daily_bars);
        for (int t = 10; t < 12; ++t) {
            float d = std::abs(em - all_metrics[t].median_daily_bars);
            if (d < bd) { bd = d; bt = t; }
        }
        return bt;
    };

    for (int ei = 0; ei < 9; ++ei) {
        int ti = match_time(ei);
        int nobs = std::min(static_cast<int>(all_metrics[ei].daily_jb.size()),
                            static_cast<int>(all_metrics[ti].daily_jb.size()));
        if (nobs < 5) continue;

        auto trim = [&](const std::vector<float>& v) {
            return std::vector<float>(v.begin(), v.begin() + nobs);
        };

        auto do_test = [&](const std::string& metric, const std::vector<float>& ev_v,
                           const std::vector<float>& tv_v, bool less) {
            auto e = trim(ev_v), t = trim(tv_v);
            float p = wilcoxon_one_sided_p(e, t, less);
            float hl = hodges_lehmann(e, t);
            std::vector<float> diffs(nobs);
            for (int i = 0; i < nobs; ++i) diffs[i] = e[i] - t[i];
            std::sort(diffs.begin(), diffs.end());
            PairComp pc{all_metrics[ei].name, all_metrics[ti].name, metric,
                        diffs[nobs / 2], hl, p, p, false};
            raw_p_map[metric].push_back(p);
            idx_map[metric].push_back(pw.size());
            pw.push_back(pc);
        };

        do_test("JB", all_metrics[ei].daily_jb, all_metrics[ti].daily_jb, true);
        do_test("ARCH", all_metrics[ei].daily_arch, all_metrics[ti].daily_arch, true);
        do_test("ACF1", all_metrics[ei].daily_acf1, all_metrics[ti].daily_acf1, true);
        do_test("AR_R2", all_metrics[ei].daily_ar_r2, all_metrics[ti].daily_ar_r2, false);
    }

    for (auto& [m, rps] : raw_p_map) {
        auto corrected = holm_bonferroni_correct(rps);
        for (size_t i = 0; i < idx_map[m].size(); ++i) {
            pw[idx_map[m][i]].corrected_p = corrected[i];
            pw[idx_map[m][i]].sig = (corrected[i] < 0.05f);
        }
    }

    std::cout << "Pairwise results:\n";
    for (const auto& p : pw) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "  %-12s vs %-8s | %-5s | mdiff=%9.6f | p=%.4f | pc=%.4f | %s",
            p.ev.c_str(), p.tv.c_str(), p.metric.c_str(), p.median_diff,
            p.raw_p, p.corrected_p, p.sig ? "SIG" : "ns");
        std::cout << buf << "\n";
    }

    // --- Quarter robustness ---
    int best_ev = best_idx < 9 ? best_idx : 0;
    if (best_idx >= 9) for (int i = 1; i < 9; ++i) if (mean_ranks[i] < mean_ranks[best_ev]) best_ev = i;
    int match_t = match_time(best_ev);

    std::map<std::string, bool> qr;
    for (const auto& q : QUARTERS) {
        std::vector<float> ejb, tjb;
        for (const auto& dr : day_results) {
            if (dr.date < q.start || dr.date > q.end) continue;
            auto ei = dr.returns.find(all_metrics[best_ev].name);
            auto ti = dr.returns.find(all_metrics[match_t].name);
            if (ei == dr.returns.end() || ti == dr.returns.end()) continue;
            if (ei->second.size() >= 20 && ti->second.size() >= 20) {
                ejb.push_back(jarque_bera_test(ei->second).statistic);
                tjb.push_back(jarque_bera_test(ti->second).statistic);
            }
        }
        bool ok = true;
        if (ejb.size() >= 3) {
            int wins = 0;
            for (size_t i = 0; i < ejb.size(); ++i) if (ejb[i] < tjb[i]) ++wins;
            ok = (wins > static_cast<int>(ejb.size()) / 2);
        }
        qr[q.label] = ok;
    }
    bool any_reversed = false;
    for (auto& [q, c] : qr) if (!c) any_reversed = true;

    // --- Summary ---
    int sig_primary = 0;
    for (const auto& p : pw)
        if (p.ev == all_metrics[best_ev].name && (p.metric == "JB" || p.metric == "ARCH" || p.metric == "ACF1"))
            if (p.sig) ++sig_primary;

    std::string finding = sig_primary == 3 ? "CONFIRMED" : sig_primary > 0 ? "PARTIALLY_CONFIRMED" : "REFUTED";
    std::string rec;
    if (sig_primary == 3)
        rec = "Use " + all_metrics[best_ev].name + " as primary bar type for all subsequent phases.";
    else if (sig_primary > 0)
        rec = "Use " + all_metrics[best_ev].name + " (best-performing), noting incomplete normalization.";
    else
        rec = "Use time bars (simplest) or investigate information-driven bars.";
    if (any_reversed) rec += " WARNING: Effect reverses in some quarters (regime-dependent).";

    std::cout << "\n=== Summary ===\n";
    std::cout << "Finding: " << finding << "\n";
    std::cout << "Best event bar: " << all_metrics[best_ev].name << " (rank=" << mean_ranks[best_ev] << ")\n";
    std::cout << "Best overall: " << all_metrics[best_idx].name << " (rank=" << mean_ranks[best_idx] << ")\n";
    std::cout << "Primary significant: " << sig_primary << "/3\n";
    std::cout << "Recommendation: " << rec << "\n";

    // --- Write JSON ---
    std::filesystem::create_directories(".kit/results/subordination-test");

    std::ostringstream js;
    js << "{\n  \"experiment\": \"R1_subordination_hypothesis_test\",\n";
    js << "  \"n_days\": " << n_days << ",\n  \"n_configs\": " << nc << ",\n";
    js << "  \"days_used\": [";
    for (size_t i = 0; i < day_results.size(); ++i) {
        if (i) js << ", "; js << day_results[i].date;
    }
    js << "],\n\n  \"bar_type_comparison\": [\n";
    for (int i = 0; i < nc; ++i) {
        const auto& m = all_metrics[i];
        js << "    {\n";
        js << "      \"config\": \"" << m.name << "\",\n";
        js << "      \"total_returns\": " << m.total_returns << ",\n";
        js << "      \"median_daily_bars\": " << to_json(m.median_daily_bars) << ",\n";
        js << "      \"jb_stat\": " << to_json(m.jb_stat) << ",\n";
        js << "      \"jb_p\": " << to_json(m.jb_p) << ",\n";
        js << "      \"arch_lm_stat\": " << to_json(m.arch_stat) << ",\n";
        js << "      \"arch_lm_p\": " << to_json(m.arch_p) << ",\n";
        js << "      \"acf_abs_return_lag1\": " << to_json(m.acf_abs.size() > 0 ? m.acf_abs[0] : 0.f) << ",\n";
        js << "      \"acf_abs_return_lag5\": " << to_json(m.acf_abs.size() > 1 ? m.acf_abs[1] : 0.f) << ",\n";
        js << "      \"acf_abs_return_lag10\": " << to_json(m.acf_abs.size() > 2 ? m.acf_abs[2] : 0.f) << ",\n";
        js << "      \"ljung_box_lag1_stat\": " << to_json(m.ljung_box.size() > 0 ? m.ljung_box[0].statistic : 0.f) << ",\n";
        js << "      \"ljung_box_lag1_p\": " << to_json(m.ljung_box.size() > 0 ? m.ljung_box[0].p_value : 1.f) << ",\n";
        js << "      \"ljung_box_lag5_stat\": " << to_json(m.ljung_box.size() > 1 ? m.ljung_box[1].statistic : 0.f) << ",\n";
        js << "      \"ljung_box_lag5_p\": " << to_json(m.ljung_box.size() > 1 ? m.ljung_box[1].p_value : 1.f) << ",\n";
        js << "      \"ljung_box_lag10_stat\": " << to_json(m.ljung_box.size() > 2 ? m.ljung_box[2].statistic : 0.f) << ",\n";
        js << "      \"ljung_box_lag10_p\": " << to_json(m.ljung_box.size() > 2 ? m.ljung_box[2].p_value : 1.f) << ",\n";
        js << "      \"ar_r2\": " << to_json(m.ar_r2) << ",\n";
        js << "      \"bar_count_cv\": " << to_json(m.bar_count_cv) << ",\n";
        js << "      \"ks_stat\": " << to_json(m.ks_stat) << ",\n";
        js << "      \"mean_rank\": " << to_json(mean_ranks[i]) << ",\n";
        js << "      \"jb_rank\": " << jbR[i] << ",\n";
        js << "      \"arch_rank\": " << arR[i] << ",\n";
        js << "      \"acf1_rank\": " << a1R[i] << ",\n";
        js << "      \"ar_rank\": " << arR2[i] << ",\n";
        js << "      \"cv_rank\": " << cvR[i] << "\n";
        js << "    }" << (i < nc - 1 ? "," : "") << "\n";
    }
    js << "  ],\n\n  \"pairwise_tests\": [\n";
    for (size_t i = 0; i < pw.size(); ++i) {
        const auto& p = pw[i];
        js << "    {\n";
        js << "      \"event_config\": \"" << p.ev << "\",\n";
        js << "      \"time_config\": \"" << p.tv << "\",\n";
        js << "      \"metric\": \"" << p.metric << "\",\n";
        js << "      \"median_diff\": " << to_json(p.median_diff) << ",\n";
        js << "      \"hodges_lehmann\": " << to_json(p.hl) << ",\n";
        js << "      \"wilcoxon_p_raw\": " << to_json(p.raw_p) << ",\n";
        js << "      \"wilcoxon_p_corrected\": " << to_json(p.corrected_p) << ",\n";
        js << "      \"significant\": " << (p.sig ? "true" : "false") << "\n";
        js << "    }" << (i < pw.size() - 1 ? "," : "") << "\n";
    }
    js << "  ],\n\n  \"quarter_robustness\": {\n";
    { int qi = 0; for (auto& [q, c] : qr) {
        js << "    \"" << q << "\": " << (c ? "true" : "false") << (qi < static_cast<int>(qr.size()) - 1 ? "," : "") << "\n"; ++qi;
    }}
    js << "  },\n  \"any_quarter_reversed\": " << (any_reversed ? "true" : "false") << ",\n\n";
    js << "  \"summary\": {\n";
    js << "    \"finding\": \"" << finding << "\",\n";
    js << "    \"best_event_bar\": \"" << all_metrics[best_ev].name << "\",\n";
    js << "    \"best_event_bar_mean_rank\": " << to_json(mean_ranks[best_ev]) << ",\n";
    js << "    \"best_overall\": \"" << all_metrics[best_idx].name << "\",\n";
    js << "    \"best_overall_mean_rank\": " << to_json(mean_ranks[best_idx]) << ",\n";
    js << "    \"primary_metrics_significant\": " << sig_primary << ",\n";
    js << "    \"primary_metrics_total\": 3,\n";
    js << "    \"recommendation\": \"" << json_escape(rec) << "\"\n";
    js << "  }\n}\n";

    std::ofstream out(".kit/results/subordination-test/metrics.json");
    out << js.str();
    out.close();

    std::cout << "\nMetrics written to .kit/results/subordination-test/metrics.json\n";
    std::cout << "=== Phase R1 complete. ===\n";
    return 0;
}
