// bar_feature_export.cpp — CLI tool for exporting bar-level feature CSVs
// Spec: .kit/docs/bar-feature-export.md
//
// Parameterized variant of info_decomposition_export.cpp.
// Pipeline: StreamingBookBuilder -> BarFactory -> BarFeatureComputer -> CSV.
//
// Usage: ./bar_feature_export --bar-type <type> --bar-param <threshold> --output <csv_path>

#include "bars/bar_factory.hpp"
#include "backtest/triple_barrier.hpp"
#include "features/bar_features.hpp"
#include "features/raw_representations.hpp"
#include "time_utils.hpp"

#include <databento/dbn_file_store.hpp>
#include <databento/record.hpp>

// Arrow/Parquet for Parquet output
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

// ===========================================================================
// StreamingBookBuilder — emits snapshots and captures MBO events
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

        if (ts_event >= rth_open_ && ts_event < rth_close_) {
            MBOEvent ev{};
            switch (action) {
                case 'A': ev.action = 0; break;
                case 'C': ev.action = 1; break;
                case 'M': ev.action = 2; break;
                case 'T': ev.action = 3; break;
                default: ev.action = -1; break;
            }
            if (ev.action >= 0) {
                ev.price = fixed_to_float(price);
                ev.size = size;
                ev.side = (side == 'B') ? 0 : 1;
                ev.ts_event = ts_event;
                mbo_events_.push_back(ev);
            }
            // Count trade events for tick bar construction
            if (action == 'T') {
                pending_trade_count_++;
            }
        }

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
    std::vector<MBOEvent>& mbo_events() { return mbo_events_; }

private:
    static constexpr uint64_t SNAPSHOT_INTERVAL = 100'000'000ULL;

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

    uint32_t pending_trade_count_ = 0;  // action='T' events since last snapshot

    std::vector<BookSnapshot> snapshots_;
    std::vector<MBOEvent> mbo_events_;

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

        snap.time_of_day = time_utils::compute_time_of_day(ts);

        // Set trade count for tick bar construction
        snap.trade_count = pending_trade_count_;
        pending_trade_count_ = 0;

        // Fill trade buffer (left-padded, matching BookBuilder::fill_trades)
        {
            size_t count = trades_.size();
            size_t start_idx = TRADE_BUF_LEN - count;
            for (size_t i = 0; i < count; ++i) {
                snap.trades[start_idx + i][0] = trades_[i].price;
                snap.trades[start_idx + i][1] = trades_[i].size;
                snap.trades[start_idx + i][2] = trades_[i].aggressor_side;
            }
        }

        snapshots_.push_back(snap);
    }
};

// ===========================================================================
// Constants
// ===========================================================================
namespace {

const std::string DATA_DIR = "DATA/GLBX-20260207-L953CAPU5B";
constexpr int WARMUP_BARS = 50;

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
    return 13615;
}

const std::vector<int> SELECTED_DAYS = {
    20220103, 20220121, 20220211, 20220304, 20220331, 20220401, 20220422,
    20220513, 20220603, 20220630, 20220701, 20220722, 20220812, 20220902,
    20220930, 20221003, 20221024, 20221114, 20221205
};

}  // anonymous namespace

// ===========================================================================
// Utility
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

std::string format_float(float val) {
    if (std::isnan(val)) return "NaN";
    if (std::isinf(val)) return "Inf";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.17g", static_cast<double>(val));
    return buf;
}

// ===========================================================================
// Parquet Writer
// ===========================================================================
void write_parquet_file(
    const std::string& path,
    const std::vector<int64_t>& timestamps,
    const std::vector<std::string>& bar_types,
    const std::vector<std::string>& bar_params,
    const std::vector<int64_t>& days,
    const std::vector<bool>& warmups,
    const std::vector<int64_t>& bar_indices,
    const std::vector<std::vector<double>>& double_cols,
    const std::vector<std::string>& tb_exit_types)
{
    int64_t num_rows = static_cast<int64_t>(timestamps.size());

    // Build schema: 6 metadata + 141 DOUBLE + 1 STRING (tb_exit_type) + 1 DOUBLE = 149 total
    // double_cols contains 142 columns: 62+40+33+4+1+1 before tb_exit_type, then 1 after.
    // Layout: double_cols[0..140] = cols before tb_exit_type, double_cols[141] = tb_bars_held.
    arrow::FieldVector fields;
    fields.push_back(arrow::field("timestamp", arrow::int64()));
    fields.push_back(arrow::field("bar_type", arrow::utf8()));
    fields.push_back(arrow::field("bar_param", arrow::utf8()));
    fields.push_back(arrow::field("day", arrow::int64()));
    fields.push_back(arrow::field("is_warmup", arrow::boolean()));
    fields.push_back(arrow::field("bar_index", arrow::int64()));

    // Track A features (62)
    auto feature_names = BarFeatureRow::feature_names();
    for (const auto& name : feature_names)
        fields.push_back(arrow::field(name, arrow::float64()));

    // Book snapshot (40)
    for (int i = 0; i < 40; ++i)
        fields.push_back(arrow::field("book_snap_" + std::to_string(i), arrow::float64()));

    // Message summary (33)
    for (size_t i = 0; i < MessageSummary::SUMMARY_SIZE; ++i)
        fields.push_back(arrow::field("msg_summary_" + std::to_string(i), arrow::float64()));

    // Forward returns (4)
    fields.push_back(arrow::field("return_1", arrow::float64()));
    fields.push_back(arrow::field("return_5", arrow::float64()));
    fields.push_back(arrow::field("return_20", arrow::float64()));
    fields.push_back(arrow::field("return_100", arrow::float64()));

    // Event count (1)
    fields.push_back(arrow::field("mbo_event_count", arrow::float64()));

    // Triple barrier labels (3): tb_label=DOUBLE, tb_exit_type=STRING, tb_bars_held=DOUBLE
    fields.push_back(arrow::field("tb_label", arrow::float64()));
    fields.push_back(arrow::field("tb_exit_type", arrow::utf8()));
    fields.push_back(arrow::field("tb_bars_held", arrow::float64()));

    auto schema = arrow::schema(fields);

    // Build arrays
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    std::shared_ptr<arrow::Array> arr;

    // timestamp (INT64)
    {
        arrow::Int64Builder b;
        (void)b.AppendValues(timestamps);
        (void)b.Finish(&arr);
        arrays.push_back(arr);
    }
    // bar_type (STRING)
    {
        arrow::StringBuilder b;
        for (const auto& v : bar_types) (void)b.Append(v);
        (void)b.Finish(&arr);
        arrays.push_back(arr);
    }
    // bar_param (STRING)
    {
        arrow::StringBuilder b;
        for (const auto& v : bar_params) (void)b.Append(v);
        (void)b.Finish(&arr);
        arrays.push_back(arr);
    }
    // day (INT64)
    {
        arrow::Int64Builder b;
        (void)b.AppendValues(days);
        (void)b.Finish(&arr);
        arrays.push_back(arr);
    }
    // is_warmup (BOOLEAN)
    {
        arrow::BooleanBuilder b;
        for (bool v : warmups) (void)b.Append(v);
        (void)b.Finish(&arr);
        arrays.push_back(arr);
    }
    // bar_index (INT64)
    {
        arrow::Int64Builder b;
        (void)b.AppendValues(bar_indices);
        (void)b.Finish(&arr);
        arrays.push_back(arr);
    }

    // DOUBLE columns before tb_exit_type (141: 62+40+33+4+1+1 = 141)
    constexpr size_t TB_EXIT_TYPE_SPLIT = 141;  // double_cols[0..140] before, [141] after
    for (size_t c = 0; c < TB_EXIT_TYPE_SPLIT; ++c) {
        arrow::DoubleBuilder b;
        (void)b.AppendValues(double_cols[c].data(),
                              static_cast<int64_t>(double_cols[c].size()));
        (void)b.Finish(&arr);
        arrays.push_back(arr);
    }

    // tb_exit_type (STRING)
    {
        arrow::StringBuilder b;
        for (const auto& v : tb_exit_types) (void)b.Append(v);
        (void)b.Finish(&arr);
        arrays.push_back(arr);
    }

    // tb_bars_held (DOUBLE) — the last double column
    {
        arrow::DoubleBuilder b;
        (void)b.AppendValues(double_cols[TB_EXIT_TYPE_SPLIT].data(),
                              static_cast<int64_t>(double_cols[TB_EXIT_TYPE_SPLIT].size()));
        (void)b.Finish(&arr);
        arrays.push_back(arr);
    }

    auto table = arrow::Table::Make(schema, arrays);

    // Write to Parquet with ZSTD compression
    auto outfile_result = arrow::io::FileOutputStream::Open(path);
    if (!outfile_result.ok()) {
        std::cerr << "Cannot open Parquet output file: " << path << "\n";
        return;
    }
    auto outfile = *outfile_result;

    auto props = parquet::WriterProperties::Builder()
        .compression(parquet::Compression::ZSTD)
        ->build();

    auto status = parquet::arrow::WriteTable(
        *table, arrow::default_memory_pool(), outfile,
        /*chunk_size=*/num_rows, props);

    if (!status.ok()) {
        std::cerr << "Failed to write Parquet: " << status.ToString() << "\n";
    }
}

// ===========================================================================
// Usage
// ===========================================================================
void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --bar-type <type> --bar-param <threshold> --output <path>\n"
              << "\n"
              << "  --bar-type   Bar type: time, volume, dollar, tick\n"
              << "  --bar-param  Bar threshold (e.g., 5.0 for time_5s)\n"
              << "  --output     Output file path (.csv or .parquet)\n";
}

// ===========================================================================
// Main
// ===========================================================================
int main(int argc, char* argv[]) {
    std::string bar_type;
    std::string bar_param_str;
    std::string output_path;

    // Parse CLI args
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--bar-type" && i + 1 < argc) {
            bar_type = argv[++i];
        } else if (arg == "--bar-param" && i + 1 < argc) {
            bar_param_str = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        }
    }

    // Validate required args
    if (bar_type.empty()) {
        std::cerr << "Missing required argument: --bar-type\n";
        print_usage(argv[0]);
        return 1;
    }
    if (bar_param_str.empty()) {
        std::cerr << "Missing required argument: --bar-param\n";
        print_usage(argv[0]);
        return 1;
    }
    if (output_path.empty()) {
        std::cerr << "Missing required argument: --output\n";
        print_usage(argv[0]);
        return 1;
    }

    double bar_param = std::stod(bar_param_str);

    // Validate bar type via BarFactory
    auto test_builder = BarFactory::create(bar_type, bar_param);
    if (!test_builder) {
        std::cerr << "Invalid bar type: '" << bar_type << "'\n";
        print_usage(argv[0]);
        return 1;
    }

    // Detect output format by file extension
    bool use_parquet = false;
    {
        std::string ext = std::filesystem::path(output_path).extension().string();
        if (ext == ".parquet") {
            use_parquet = true;
        } else if (ext != ".csv") {
            std::cerr << "Unsupported output format. Use .csv or .parquet extension.\n";
            return 1;
        }
    }

    // CSV setup (only if CSV output)
    std::ofstream csv;
    if (!use_parquet) {
        csv.open(output_path);
        if (!csv.is_open()) {
            std::cerr << "Cannot open output file: " << output_path << "\n";
            return 1;
        }
        csv << std::setprecision(17);

        // Write header: 6 metadata + 62 Track A + 40 book_snap + 33 msg_summary + 4 returns + 1 event count + 3 tb labels = 149
        csv << "timestamp,bar_type,bar_param,day,is_warmup,bar_index";
        auto feature_names = BarFeatureRow::feature_names();
        for (const auto& name : feature_names) csv << "," << name;
        for (int i = 0; i < 40; ++i) csv << ",book_snap_" << i;
        for (size_t i = 0; i < MessageSummary::SUMMARY_SIZE; ++i) csv << ",msg_summary_" << i;
        csv << ",return_1,return_5,return_20,return_100";
        csv << ",mbo_event_count";
        csv << ",tb_label,tb_exit_type,tb_bars_held";
        csv << "\n";
    }

    // Parquet data collectors (only if Parquet output)
    // 142 DOUBLE columns: 62 Track A + 40 book + 33 msg + 4 returns + 1 event + 1 tb_label + 1 tb_bars_held
    // tb_exit_type is stored separately as STRING
    constexpr int NUM_DOUBLE_COLS = 142;
    std::vector<int64_t> pq_timestamps;
    std::vector<std::string> pq_bar_types;
    std::vector<std::string> pq_bar_params;
    std::vector<int64_t> pq_days;
    std::vector<bool> pq_warmups;
    std::vector<int64_t> pq_bar_indices;
    std::vector<std::vector<double>> pq_double_cols(NUM_DOUBLE_COLS);
    std::vector<std::string> pq_tb_exit_types;

    int total_bars = 0;

    // Triple barrier config — constant across all days/bars
    TripleBarrierConfig tb_cfg;
    tb_cfg.target_ticks = 10;
    tb_cfg.stop_ticks = 5;
    tb_cfg.volume_horizon = 500;
    tb_cfg.min_return_ticks = 2;
    tb_cfg.max_time_horizon_s = 300;
    tb_cfg.tick_size = 0.25f;

    for (int date : SELECTED_DAYS) {
        std::string date_str = date_to_string(date);
        std::string filepath = DATA_DIR + "/glbx-mdp3-" + date_str + ".mbo.dbn.zst";

        if (!std::filesystem::exists(filepath)) {
            std::cerr << "  SKIP: " << filepath << " not found\n";
            continue;
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
        auto& mbo_events = builder.mbo_events();
        std::cout << snapshots.size() << " snaps, " << mbo_events.size() << " events, " << std::flush;

        if (snapshots.empty()) {
            std::cout << "skipped\n";
            continue;
        }

        // Build bars using CLI-specified bar type
        auto bar_builder = BarFactory::create(bar_type, bar_param);
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

        // Assign MBO event indices to bars
        size_t ev_cursor = 0;
        for (auto& bar : bars) {
            bar.mbo_event_begin = static_cast<uint32_t>(ev_cursor);
            while (ev_cursor < mbo_events.size() &&
                   mbo_events[ev_cursor].ts_event <= bar.close_ts) {
                ev_cursor++;
            }
            bar.mbo_event_end = static_cast<uint32_t>(ev_cursor);

            // Recount message types from actual MBO events
            bar.add_count = 0;
            bar.cancel_count = 0;
            bar.modify_count = 0;
            bar.trade_event_count = 0;
            for (uint32_t i = bar.mbo_event_begin; i < bar.mbo_event_end; ++i) {
                switch (mbo_events[i].action) {
                    case 0: bar.add_count++; break;
                    case 1: bar.cancel_count++; break;
                    case 2: bar.modify_count++; break;
                    case 3: bar.trade_event_count++; break;
                }
            }
            bar.cancel_add_ratio = static_cast<float>(bar.cancel_count) /
                                   (static_cast<float>(bar.add_count) + 1e-8f);
            if (bar.bar_duration_s > 0.0f) {
                float total_msgs = static_cast<float>(bar.add_count + bar.cancel_count +
                                                       bar.modify_count + bar.trade_event_count);
                bar.message_rate = total_msgs / bar.bar_duration_s;
            }
        }

        // Compute Track A features
        BarFeatureComputer computer(0.25f);
        auto feature_rows = computer.compute_all(bars);

        // Write bars to CSV or collect for Parquet
        for (size_t i = 0; i < bars.size(); ++i) {
            bool is_warmup = (static_cast<int>(i) < WARMUP_BARS);
            feature_rows[i].is_warmup = is_warmup;
            feature_rows[i].bar_type = bar_type;
            feature_rows[i].bar_param = static_cast<float>(bar_param);
            feature_rows[i].day = date;

            if (is_warmup) continue;

            // Skip bars without valid forward returns
            if (std::isnan(feature_rows[i].fwd_return_1)) continue;

            const auto& row = feature_rows[i];
            const auto& bar = bars[i];

            // Compute derived data needed for both paths
            auto book_flat = BookSnapshotExport::flatten(bar);

            std::vector<float> msg_summary(MessageSummary::SUMMARY_SIZE, 0.0f);
            uint32_t n_events = bar.mbo_event_end - bar.mbo_event_begin;
            if (n_events > 0) {
                float duration = bar.bar_duration_s;
                if (duration <= 0.0f) duration = 1.0f;
                uint64_t bar_start = bar.open_ts;
                uint64_t bar_range = (bar.close_ts > bar.open_ts) ? (bar.close_ts - bar.open_ts) : 1;

                uint32_t first_half_adds = 0, first_half_cancels = 0;
                uint32_t second_half_adds = 0, second_half_cancels = 0;

                for (uint32_t ei = bar.mbo_event_begin; ei < bar.mbo_event_end; ++ei) {
                    const auto& ev = mbo_events[ei];
                    float frac = static_cast<float>(ev.ts_event - bar_start) /
                                 static_cast<float>(bar_range);
                    frac = std::max(0.0f, std::min(frac, 0.999f));
                    int decile = static_cast<int>(frac * 10.0f);

                    int action_offset = -1;
                    if (ev.action == 0) action_offset = 0;
                    else if (ev.action == 1) action_offset = 1;
                    else if (ev.action == 2) action_offset = 2;

                    if (action_offset >= 0) {
                        msg_summary[decile * 3 + action_offset] += 1.0f;
                    }

                    bool first_half = (frac < 0.5f);
                    if (ev.action == 0) {
                        if (first_half) first_half_adds++;
                        else second_half_adds++;
                    } else if (ev.action == 1) {
                        if (first_half) first_half_cancels++;
                        else second_half_cancels++;
                    }
                }

                constexpr float EPS = 1e-8f;
                msg_summary[30] = static_cast<float>(first_half_cancels) /
                                   (static_cast<float>(first_half_adds) + EPS);
                msg_summary[31] = static_cast<float>(second_half_cancels) /
                                   (static_cast<float>(second_half_adds) + EPS);

                float decile_duration = duration / 10.0f;
                float max_rate = 0.0f;
                for (int dc = 0; dc < 10; ++dc) {
                    float decile_total = msg_summary[dc*3] + msg_summary[dc*3+1] + msg_summary[dc*3+2];
                    float rate = decile_total / (decile_duration + EPS);
                    max_rate = std::max(max_rate, rate);
                }
                msg_summary[32] = max_rate;
            }

            auto tb = compute_tb_label(bars, static_cast<int>(i), tb_cfg);

            if (use_parquet) {
                // Collect metadata
                pq_timestamps.push_back(static_cast<int64_t>(row.timestamp));
                pq_bar_types.push_back(bar_type);
                pq_bar_params.push_back(bar_param_str);
                pq_days.push_back(static_cast<int64_t>(date));
                pq_warmups.push_back(false);
                pq_bar_indices.push_back(static_cast<int64_t>(i));

                // Collect 143 DOUBLE columns
                int d = 0;

                // Track A features (62) — use get_feature_value for consistency
                auto fnames = BarFeatureRow::feature_names();
                for (const auto& name : fnames) {
                    pq_double_cols[d++].push_back(
                        static_cast<double>(row.get_feature_value(name)));
                }

                // Book snapshot (40)
                for (int j = 0; j < 40; ++j) {
                    pq_double_cols[d++].push_back(static_cast<double>(book_flat[j]));
                }

                // Message summary (33)
                for (size_t j = 0; j < MessageSummary::SUMMARY_SIZE; ++j) {
                    pq_double_cols[d++].push_back(static_cast<double>(msg_summary[j]));
                }

                // Forward returns (4)
                pq_double_cols[d++].push_back(static_cast<double>(row.fwd_return_1));
                pq_double_cols[d++].push_back(static_cast<double>(row.fwd_return_5));
                pq_double_cols[d++].push_back(static_cast<double>(row.fwd_return_20));
                pq_double_cols[d++].push_back(static_cast<double>(row.fwd_return_100));

                // Event count (1)
                pq_double_cols[d++].push_back(static_cast<double>(n_events));

                // Triple barrier labels: tb_label (DOUBLE), tb_exit_type (STRING), tb_bars_held (DOUBLE)
                pq_double_cols[d++].push_back(static_cast<double>(tb.label));
                pq_tb_exit_types.push_back(tb.exit_type);
                pq_double_cols[d++].push_back(static_cast<double>(tb.bars_held));
            } else {
                // Write CSV row
                csv << row.timestamp;
                csv << "," << bar_type << "," << bar_param_str << "," << date << ",false," << i;

                // Track A features (62)
                csv << "," << row.book_imbalance_1;
                csv << "," << row.book_imbalance_3;
                csv << "," << row.book_imbalance_5;
                csv << "," << row.book_imbalance_10;
                csv << "," << row.weighted_imbalance;
                csv << "," << row.spread;
                for (int j = 0; j < 10; ++j) csv << "," << row.bid_depth_profile[j];
                for (int j = 0; j < 10; ++j) csv << "," << row.ask_depth_profile[j];
                csv << "," << row.depth_concentration_bid;
                csv << "," << row.depth_concentration_ask;
                csv << "," << row.book_slope_bid;
                csv << "," << row.book_slope_ask;
                csv << "," << row.level_count_bid;
                csv << "," << row.level_count_ask;

                csv << "," << row.net_volume;
                csv << "," << row.volume_imbalance;
                csv << "," << row.trade_count;
                csv << "," << row.avg_trade_size;
                csv << "," << row.large_trade_count;
                csv << "," << row.vwap_distance;
                csv << "," << format_float(row.kyle_lambda);

                csv << "," << format_float(row.return_1);
                csv << "," << format_float(row.return_5);
                csv << "," << format_float(row.return_20);
                csv << "," << format_float(row.volatility_20);
                csv << "," << format_float(row.volatility_50);
                csv << "," << row.momentum;
                csv << "," << format_float(row.high_low_range_20);
                csv << "," << format_float(row.high_low_range_50);
                csv << "," << row.close_position;

                csv << "," << row.volume_surprise;
                csv << "," << row.duration_surprise;
                csv << "," << row.acceleration;
                csv << "," << format_float(row.vol_price_corr);

                csv << "," << row.time_sin;
                csv << "," << row.time_cos;
                csv << "," << row.minutes_since_open;
                csv << "," << row.minutes_to_close;
                csv << "," << row.session_volume_frac;

                csv << "," << row.cancel_add_ratio;
                csv << "," << row.message_rate;
                csv << "," << row.modify_fraction;
                csv << "," << row.order_flow_toxicity;
                csv << "," << row.cancel_concentration;

                // Book snapshot (40)
                for (int j = 0; j < 40; ++j) {
                    csv << "," << format_float(book_flat[j]);
                }

                // Message summary (33)
                for (size_t j = 0; j < MessageSummary::SUMMARY_SIZE; ++j) {
                    csv << "," << format_float(msg_summary[j]);
                }

                // Forward returns
                csv << "," << format_float(row.fwd_return_1);
                csv << "," << format_float(row.fwd_return_5);
                csv << "," << format_float(row.fwd_return_20);
                csv << "," << format_float(row.fwd_return_100);

                // Event count
                csv << "," << n_events;

                // Triple barrier label
                csv << "," << tb.label << "," << tb.exit_type << "," << tb.bars_held;

                csv << "\n";
            }
            total_bars++;
        }

        std::cout << "exported\n";
    }

    // Finalize output
    if (use_parquet) {
        write_parquet_file(output_path, pq_timestamps, pq_bar_types, pq_bar_params,
                           pq_days, pq_warmups, pq_bar_indices, pq_double_cols,
                           pq_tb_exit_types);
    } else {
        csv.close();
    }

    std::cout << "\nTotal bars exported: " << total_bars << "\n";
    std::cout << "Output: " << output_path << "\n";
    return 0;
}
