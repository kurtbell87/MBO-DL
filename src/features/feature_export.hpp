#pragma once

#include "features/bar_features.hpp"
#include "features/raw_representations.hpp"
#include "backtest/rollover.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// ExportConfig
// ---------------------------------------------------------------------------
struct ExportConfig {
    std::string output_path;
    bool include_warmup = false;
};

// ---------------------------------------------------------------------------
// FeatureExporter
// ---------------------------------------------------------------------------
class FeatureExporter {
public:
    FeatureExporter() = default;
    explicit FeatureExporter(const ExportConfig& config) : config_(config) {}

    // Header line (CSV).
    std::string header_line() const {
        std::ostringstream ss;
        // Metadata
        ss << "timestamp,bar_type,bar_param,day,is_warmup";

        // Track A feature names
        auto names = BarFeatureRow::feature_names();
        for (const auto& name : names) {
            ss << "," << name;
        }

        // Track B book snapshot (40 flattened values)
        for (int i = 0; i < 40; ++i) {
            ss << ",book_snap_" << i;
        }

        // Track B message summaries
        for (size_t i = 0; i < MessageSummary::SUMMARY_SIZE; ++i) {
            ss << ",msg_summary_" << i;
        }

        // Forward returns
        ss << ",return_1,return_5,return_20,return_100";

        return ss.str();
    }

    // Format a single row as CSV.
    std::string format_row(const BarFeatureRow& row) const {
        std::ostringstream ss;
        ss << row.timestamp;
        ss << "," << row.bar_type;
        ss << "," << row.bar_param;
        ss << "," << row.day;
        ss << "," << (row.is_warmup ? "true" : "false");

        // Cat 1
        ss << "," << row.book_imbalance_1;
        ss << "," << row.book_imbalance_3;
        ss << "," << row.book_imbalance_5;
        ss << "," << row.book_imbalance_10;
        ss << "," << row.weighted_imbalance;
        ss << "," << row.spread;
        for (int i = 0; i < 10; ++i) ss << "," << row.bid_depth_profile[i];
        for (int i = 0; i < 10; ++i) ss << "," << row.ask_depth_profile[i];
        ss << "," << row.depth_concentration_bid;
        ss << "," << row.depth_concentration_ask;
        ss << "," << row.book_slope_bid;
        ss << "," << row.book_slope_ask;
        ss << "," << row.level_count_bid;
        ss << "," << row.level_count_ask;

        // Cat 2
        ss << "," << row.net_volume;
        ss << "," << row.volume_imbalance;
        ss << "," << row.trade_count;
        ss << "," << row.avg_trade_size;
        ss << "," << row.large_trade_count;
        ss << "," << row.vwap_distance;
        ss << "," << format_float(row.kyle_lambda);

        // Cat 3
        ss << "," << format_float(row.return_1);
        ss << "," << format_float(row.return_5);
        ss << "," << format_float(row.return_20);
        ss << "," << format_float(row.volatility_20);
        ss << "," << format_float(row.volatility_50);
        ss << "," << row.momentum;
        ss << "," << format_float(row.high_low_range_20);
        ss << "," << format_float(row.high_low_range_50);
        ss << "," << row.close_position;

        // Cat 4
        ss << "," << row.volume_surprise;
        ss << "," << row.duration_surprise;
        ss << "," << row.acceleration;
        ss << "," << format_float(row.vol_price_corr);

        // Cat 5
        ss << "," << row.time_sin;
        ss << "," << row.time_cos;
        ss << "," << row.minutes_since_open;
        ss << "," << row.minutes_to_close;
        ss << "," << row.session_volume_frac;

        // Cat 6
        ss << "," << row.cancel_add_ratio;
        ss << "," << row.message_rate;
        ss << "," << row.modify_fraction;
        ss << "," << row.order_flow_toxicity;
        ss << "," << row.cancel_concentration;

        // Track B book snapshot placeholder (40 zeros for now, unless populated)
        for (int i = 0; i < 40; ++i) ss << ",0";

        // Track B message summary placeholder
        for (size_t i = 0; i < MessageSummary::SUMMARY_SIZE; ++i) ss << ",0";

        // Forward returns
        ss << "," << format_float(row.fwd_return_1);
        ss << "," << format_float(row.fwd_return_5);
        ss << "," << format_float(row.fwd_return_20);
        ss << "," << format_float(row.fwd_return_100);

        return ss.str();
    }

    // Batch export to CSV.
    void export_csv(const std::vector<BarFeatureRow>& rows) {
        export_csv_filtered(rows, [](const BarFeatureRow&) { return true; });
    }

    // Batch export with rollover exclusion.
    void export_csv(const std::vector<BarFeatureRow>& rows, const RolloverCalendar& cal) {
        export_csv_filtered(rows, [&cal](const BarFeatureRow& row) {
            return !cal.is_excluded(row.day);
        });
    }

    // Streaming export.
    void begin() {
        if (config_.output_path.empty()) return;
        stream_.open(config_.output_path);
        if (!stream_.is_open()) {
            throw std::runtime_error("Cannot open output file: " + config_.output_path);
        }
        stream_ << header_line() << "\n";
    }

    void write_row(const BarFeatureRow& row) {
        if (!stream_.is_open()) return;
        stream_ << format_row(row) << "\n";
    }

    void end() {
        if (stream_.is_open()) {
            stream_.close();
        }
    }

private:
    ExportConfig config_;
    std::ofstream stream_;

    void export_csv_filtered(const std::vector<BarFeatureRow>& rows,
                             std::function<bool(const BarFeatureRow&)> extra_filter) {
        if (config_.output_path.empty()) return;

        auto parent = std::filesystem::path(config_.output_path).parent_path();
        if (!parent.empty() && !std::filesystem::exists(parent)) {
            throw std::runtime_error("Output directory does not exist: " + parent.string());
        }

        std::ofstream file(config_.output_path);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open output file: " + config_.output_path);
        }

        file << header_line() << "\n";
        for (const auto& row : rows) {
            if (!config_.include_warmup && row.is_warmup) continue;
            if (!extra_filter(row)) continue;
            file << format_row(row) << "\n";
        }
    }

    static std::string format_float(float val) {
        if (std::isnan(val)) return "NaN";
        if (std::isinf(val)) return "Inf";
        std::ostringstream ss;
        ss << val;
        return ss.str();
    }
};
