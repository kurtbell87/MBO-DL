#pragma once

#include "bars/bar.hpp"
#include "book_builder.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// BarFeatureRow — all Track A features + metadata + forward returns
// ---------------------------------------------------------------------------
struct BarFeatureRow {
    // Bar metadata
    uint64_t timestamp = 0;
    std::string bar_type;
    float bar_param = 0.0f;
    int day = 0;
    bool is_warmup = false;

    // Category 1: Book Shape
    float book_imbalance_1 = 0.0f;
    float book_imbalance_3 = 0.0f;
    float book_imbalance_5 = 0.0f;
    float book_imbalance_10 = 0.0f;
    float weighted_imbalance = 0.0f;
    float spread = 0.0f;
    float bid_depth_profile[10] = {};
    float ask_depth_profile[10] = {};
    float depth_concentration_bid = 0.0f;
    float depth_concentration_ask = 0.0f;
    float book_slope_bid = 0.0f;
    float book_slope_ask = 0.0f;
    int level_count_bid = 0;
    int level_count_ask = 0;

    // Category 2: Order Flow
    float net_volume = 0.0f;
    float volume_imbalance = 0.0f;
    uint32_t trade_count = 0;
    float avg_trade_size = 0.0f;
    uint32_t large_trade_count = 0;
    float vwap_distance = 0.0f;
    float kyle_lambda = 0.0f;

    // Category 3: Price Dynamics
    float return_1 = 0.0f;
    float return_5 = 0.0f;
    float return_20 = 0.0f;
    float volatility_20 = 0.0f;
    float volatility_50 = 0.0f;
    float momentum = 0.0f;
    float high_low_range_20 = 0.0f;
    float high_low_range_50 = 0.0f;
    float close_position = 0.0f;

    // Category 4: Cross-Scale Dynamics
    float volume_surprise = 0.0f;
    float duration_surprise = 0.0f;
    float acceleration = 0.0f;
    float vol_price_corr = 0.0f;

    // Category 5: Time Context
    float time_sin = 0.0f;
    float time_cos = 0.0f;
    float minutes_since_open = 0.0f;
    float minutes_to_close = 0.0f;
    float session_volume_frac = 0.0f;

    // Category 6: Message Microstructure
    float cancel_add_ratio = 0.0f;
    float message_rate = 0.0f;
    float modify_fraction = 0.0f;
    float order_flow_toxicity = 0.0f;
    float cancel_concentration = 0.0f;

    // Forward returns (targets, not features)
    float fwd_return_1 = 0.0f;
    float fwd_return_5 = 0.0f;
    float fwd_return_20 = 0.0f;
    float fwd_return_100 = 0.0f;

    // Feature count (Track A features only, excluding metadata and forward returns)
    static size_t feature_count() {
        // Cat 1: 4 imbalance + 1 weighted + 1 spread + 10 bid_depth + 10 ask_depth
        //        + 2 concentration + 2 slope + 2 level_count = 32
        // Cat 2: 7 (net_volume, volume_imbalance, trade_count, avg_trade_size,
        //         large_trade_count, vwap_distance, kyle_lambda)
        // Cat 3: 3 returns + 2 vol + 1 momentum + 2 range + 1 position = 9
        // Cat 4: 4
        // Cat 5: 5
        // Cat 6: 5
        return 62;
    }

    static std::vector<std::string> feature_names() {
        std::vector<std::string> names;
        // Cat 1
        names.push_back("book_imbalance_1");
        names.push_back("book_imbalance_3");
        names.push_back("book_imbalance_5");
        names.push_back("book_imbalance_10");
        names.push_back("weighted_imbalance");
        names.push_back("spread");
        for (int i = 0; i < 10; ++i)
            names.push_back("bid_depth_profile_" + std::to_string(i));
        for (int i = 0; i < 10; ++i)
            names.push_back("ask_depth_profile_" + std::to_string(i));
        names.push_back("depth_concentration_bid");
        names.push_back("depth_concentration_ask");
        names.push_back("book_slope_bid");
        names.push_back("book_slope_ask");
        names.push_back("level_count_bid");
        names.push_back("level_count_ask");
        // Cat 2
        names.push_back("net_volume");
        names.push_back("volume_imbalance");
        names.push_back("trade_count");
        names.push_back("avg_trade_size");
        names.push_back("large_trade_count");
        names.push_back("vwap_distance");
        names.push_back("kyle_lambda");
        // Cat 3
        names.push_back("return_1");
        names.push_back("return_5");
        names.push_back("return_20");
        names.push_back("volatility_20");
        names.push_back("volatility_50");
        names.push_back("momentum");
        names.push_back("high_low_range_20");
        names.push_back("high_low_range_50");
        names.push_back("close_position");
        // Cat 4
        names.push_back("volume_surprise");
        names.push_back("duration_surprise");
        names.push_back("acceleration");
        names.push_back("vol_price_corr");
        // Cat 5
        names.push_back("time_sin");
        names.push_back("time_cos");
        names.push_back("minutes_since_open");
        names.push_back("minutes_to_close");
        names.push_back("session_volume_frac");
        // Cat 6
        names.push_back("cancel_add_ratio");
        names.push_back("message_rate");
        names.push_back("modify_fraction");
        names.push_back("order_flow_toxicity");
        names.push_back("cancel_concentration");
        return names;
    }
};

// ---------------------------------------------------------------------------
// BarFeatureComputer — computes Track A features incrementally
// ---------------------------------------------------------------------------
class BarFeatureComputer {
public:
    BarFeatureComputer() : tick_size_(0.25f) {}
    explicit BarFeatureComputer(float tick_size) : tick_size_(tick_size) {}

    // Process a single bar and return its feature row.
    BarFeatureRow update(const Bar& bar) {
        BarFeatureRow row{};

        // Metadata
        row.timestamp = bar.close_ts;

        // === Category 1: Book Shape ===
        compute_book_shape(bar, row);

        // === Category 2: Order Flow ===
        compute_order_flow(bar, row);

        // === Category 3: Price Dynamics ===
        // Store close_mid for lookback
        close_mids_.push_back(bar.close_mid);
        high_mids_.push_back(bar.high_mid);
        low_mids_.push_back(bar.low_mid);

        compute_price_dynamics(row);

        // === Category 4: Cross-Scale Dynamics ===
        compute_cross_scale(bar, row);

        // === Category 5: Time Context ===
        compute_time_context(bar, row);

        // === Category 6: Message Microstructure ===
        compute_message_microstructure(bar, row);

        // Warmup flag
        row.is_warmup = compute_is_warmup();

        bar_count_++;
        return row;
    }

    // Process all bars and fill forward returns.
    std::vector<BarFeatureRow> compute_all(const std::vector<Bar>& bars) {
        reset();
        std::vector<BarFeatureRow> rows;
        rows.reserve(bars.size());
        for (const auto& bar : bars) {
            rows.push_back(update(bar));
        }
        // Fixup NaN rolling features using available data (batch mode has full sequence)
        fixup_rolling_features(bars, rows);
        // Fill forward returns
        fill_forward_returns(bars, rows);
        return rows;
    }

    // Reset state for session boundary.
    void reset() {
        bar_count_ = 0;
        close_mids_.clear();
        high_mids_.clear();
        low_mids_.clear();
        volumes_.clear();
        returns_.clear();
        net_volumes_.clear();
        abs_returns_.clear();
        ewma_volume_ = 0.0f;
        ewma_duration_ = 0.0f;
        ewma_initialized_ = false;
        prev_return_1_ = std::numeric_limits<float>::quiet_NaN();
        cumulative_volume_ = 0.0f;
    }

    // Report session-end total volume for session_volume_frac.
    void end_session(float total_volume) {
        prior_day_totals_.push_back(total_volume);
        reset();
    }

private:
    float tick_size_;
    int bar_count_ = 0;

    // Lookback buffers
    std::deque<float> close_mids_;
    std::deque<float> high_mids_;
    std::deque<float> low_mids_;
    std::deque<float> volumes_;
    std::deque<float> returns_;     // 1-bar returns in ticks
    std::deque<float> net_volumes_; // for kyle_lambda
    std::deque<float> abs_returns_; // |return_1| for vol_price_corr

    // EWMA state
    float ewma_volume_ = 0.0f;
    float ewma_duration_ = 0.0f;
    bool ewma_initialized_ = false;
    static constexpr int EWMA_SPAN = 20;
    static constexpr float EWMA_ALPHA = 2.0f / (EWMA_SPAN + 1);

    // Acceleration: previous return_1
    float prev_return_1_ = std::numeric_limits<float>::quiet_NaN();

    // Session volume tracking
    float cumulative_volume_ = 0.0f;
    std::vector<float> prior_day_totals_;

    static constexpr float EPS = 1e-8f;
    static constexpr float TWO_PI = 2.0f * 3.14159265358979323846f;
    static constexpr float RTH_OPEN_HOUR = 9.5f;   // 09:30
    static constexpr float RTH_CLOSE_HOUR = 16.0f;  // 16:00

    // -----------------------------------------------------------------------
    // Category 1: Book Shape
    // -----------------------------------------------------------------------
    void compute_book_shape(const Bar& bar, BarFeatureRow& row) {
        // Book imbalances at various depths
        row.book_imbalance_1 = book_imbalance(bar, 1);
        row.book_imbalance_3 = book_imbalance(bar, 3);
        row.book_imbalance_5 = book_imbalance(bar, 5);
        row.book_imbalance_10 = book_imbalance(bar, 10);

        // Weighted imbalance (inverse distance weighting)
        row.weighted_imbalance = weighted_imbalance(bar);

        // Spread in ticks
        row.spread = bar.spread / tick_size_;

        // Depth profiles (raw sizes)
        for (int i = 0; i < BOOK_DEPTH; ++i) {
            row.bid_depth_profile[i] = bar.bids[i][1];
            row.ask_depth_profile[i] = bar.asks[i][1];
        }

        // Depth concentration (HHI)
        row.depth_concentration_bid = depth_hhi(bar.bids);
        row.depth_concentration_ask = depth_hhi(bar.asks);

        // Book slope (log(size) vs level index regression)
        row.book_slope_bid = book_slope(bar.bids);
        row.book_slope_ask = book_slope(bar.asks);

        // Level counts
        row.level_count_bid = level_count(bar.bids);
        row.level_count_ask = level_count(bar.asks);
    }

    static float book_imbalance(const Bar& bar, int depth) {
        float bid_vol = 0.0f, ask_vol = 0.0f;
        for (int i = 0; i < depth && i < BOOK_DEPTH; ++i) {
            bid_vol += bar.bids[i][1];
            ask_vol += bar.asks[i][1];
        }
        return (bid_vol - ask_vol) / (bid_vol + ask_vol + EPS);
    }

    float weighted_imbalance(const Bar& bar) {
        // Weight by 1/(level_index + 1) — closer levels get higher weight
        float w_bid = 0.0f, w_ask = 0.0f;
        for (int i = 0; i < BOOK_DEPTH; ++i) {
            float w = 1.0f / static_cast<float>(i + 1);
            w_bid += bar.bids[i][1] * w;
            w_ask += bar.asks[i][1] * w;
        }
        return (w_bid - w_ask) / (w_bid + w_ask + EPS);
    }

    static float depth_hhi(const float levels[][2]) {
        float total = 0.0f;
        for (int i = 0; i < BOOK_DEPTH; ++i) total += levels[i][1];
        if (total < EPS) return 0.0f;
        float hhi = 0.0f;
        for (int i = 0; i < BOOK_DEPTH; ++i) {
            float frac = levels[i][1] / total;
            hhi += frac * frac;
        }
        return hhi;
    }

    static float book_slope(const float levels[][2]) {
        // Linear regression of log(size) vs level index.
        // Only uses levels with size > 0.
        int n = 0;
        float sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
        for (int i = 0; i < BOOK_DEPTH; ++i) {
            if (levels[i][1] <= 0.0f) continue;
            float x = static_cast<float>(i);
            float y = std::log(levels[i][1]);
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
            n++;
        }
        if (n < 2) return 0.0f;
        float nf = static_cast<float>(n);
        float denom = nf * sum_xx - sum_x * sum_x;
        if (std::abs(denom) < EPS) return 0.0f;
        return (nf * sum_xy - sum_x * sum_y) / denom;
    }

    static int level_count(const float levels[][2]) {
        int count = 0;
        for (int i = 0; i < BOOK_DEPTH; ++i) {
            if (levels[i][1] > 0.0f) count++;
        }
        return count;
    }

    // -----------------------------------------------------------------------
    // Category 2: Order Flow
    // -----------------------------------------------------------------------
    void compute_order_flow(const Bar& bar, BarFeatureRow& row) {
        row.net_volume = bar.buy_volume - bar.sell_volume;

        float total_vol = static_cast<float>(bar.volume);
        row.volume_imbalance = (total_vol > EPS) ? row.net_volume / (total_vol + EPS) : 0.0f;

        row.trade_count = bar.trade_event_count;

        row.avg_trade_size = (bar.trade_event_count > 0)
            ? total_vol / static_cast<float>(bar.trade_event_count)
            : 0.0f;

        // Large trade count: trades with size > 2× rolling median(volume, 20)
        volumes_.push_back(total_vol);
        if (volumes_.size() > 20) volumes_.pop_front();
        row.large_trade_count = compute_large_trade_count(bar);

        // VWAP distance in ticks
        row.vwap_distance = (bar.close_mid - bar.vwap) / tick_size_;

        // Kyle lambda — rolling 20-bar regression of Δmid on net_volume
        net_volumes_.push_back(row.net_volume);
        if (net_volumes_.size() > 20) net_volumes_.pop_front();
        row.kyle_lambda = compute_kyle_lambda();
    }

    uint32_t compute_large_trade_count(const Bar& bar) {
        if (volumes_.size() < 20) return 0;
        // Compute median of last 20 volumes
        std::vector<float> sorted_vols(volumes_.begin(), volumes_.end());
        std::sort(sorted_vols.begin(), sorted_vols.end());
        float median = (sorted_vols[9] + sorted_vols[10]) / 2.0f;
        float threshold = 2.0f * median;
        float vol = static_cast<float>(bar.volume);
        return (vol > threshold) ? 1u : 0u;
    }

    float compute_kyle_lambda() {
        // Need at least 21 bars (20 Δmid values from 21 close_mids, and 20 net_volumes)
        if (close_mids_.size() < 21 || net_volumes_.size() < 20) {
            return std::numeric_limits<float>::quiet_NaN();
        }

        // No-intercept OLS: Δmid = λ * net_volume
        // λ = Σ(Δmid * net_volume) / Σ(net_volume²)
        size_t n = 20;
        size_t cm_start = close_mids_.size() - n - 1;

        float sum_xy = 0, sum_xx = 0;
        for (size_t i = 0; i < n; ++i) {
            float delta_mid = (close_mids_[cm_start + i + 1] - close_mids_[cm_start + i]) / tick_size_;
            float nv = net_volumes_[net_volumes_.size() - n + i];
            sum_xy += nv * delta_mid;
            sum_xx += nv * nv;
        }
        if (sum_xx < EPS) return 0.0f;
        return sum_xy / sum_xx;
    }

    // -----------------------------------------------------------------------
    // Category 3: Price Dynamics
    // -----------------------------------------------------------------------
    void compute_price_dynamics(BarFeatureRow& row) {
        size_t n = close_mids_.size();

        // return_1
        if (n >= 2) {
            row.return_1 = (close_mids_[n-1] - close_mids_[n-2]) / tick_size_;
        } else {
            row.return_1 = std::numeric_limits<float>::quiet_NaN();
        }

        // return_5
        if (n >= 6) {
            row.return_5 = (close_mids_[n-1] - close_mids_[n-6]) / tick_size_;
        } else {
            row.return_5 = std::numeric_limits<float>::quiet_NaN();
        }

        // return_20
        if (n >= 21) {
            row.return_20 = (close_mids_[n-1] - close_mids_[n-21]) / tick_size_;
        } else {
            row.return_20 = std::numeric_limits<float>::quiet_NaN();
        }

        // Store 1-bar returns for volatility/momentum
        if (n >= 2) {
            float r1 = (close_mids_[n-1] - close_mids_[n-2]) / tick_size_;
            returns_.push_back(r1);
            abs_returns_.push_back(std::abs(r1));
        }

        // volatility_20
        if (returns_.size() >= 20) {
            row.volatility_20 = rolling_std(returns_, 20);
        } else {
            row.volatility_20 = std::numeric_limits<float>::quiet_NaN();
        }

        // volatility_50
        if (returns_.size() >= 50) {
            row.volatility_50 = rolling_std(returns_, 50);
        } else {
            row.volatility_50 = std::numeric_limits<float>::quiet_NaN();
        }

        // momentum: sum of signed 1-bar returns over last 20 bars
        if (returns_.size() >= 20) {
            float sum = 0.0f;
            for (size_t i = returns_.size() - 20; i < returns_.size(); ++i) {
                sum += returns_[i];
            }
            row.momentum = sum;
        } else if (!returns_.empty()) {
            float sum = 0.0f;
            for (auto r : returns_) sum += r;
            row.momentum = sum;
        } else {
            row.momentum = 0.0f;
        }

        // high_low_range_20: NaN for first 20 bars (need > 20 close_mids)
        if (n > 20) {
            row.high_low_range_20 = high_low_range(20);
        } else {
            row.high_low_range_20 = std::numeric_limits<float>::quiet_NaN();
        }

        // high_low_range_50: NaN for first 50 bars (need > 50 close_mids)
        if (n > 50) {
            row.high_low_range_50 = high_low_range(50);
        } else {
            row.high_low_range_50 = std::numeric_limits<float>::quiet_NaN();
        }

        // close_position
        if (n > 20) {
            float max_high = *std::max_element(high_mids_.end() - 20, high_mids_.end());
            float min_low = *std::min_element(low_mids_.end() - 20, low_mids_.end());
            float range = max_high - min_low + EPS;
            row.close_position = (close_mids_.back() - min_low) / range;
        } else if (n >= 2) {
            float max_high = *std::max_element(high_mids_.begin(), high_mids_.end());
            float min_low = *std::min_element(low_mids_.begin(), low_mids_.end());
            float range = max_high - min_low + EPS;
            row.close_position = (close_mids_.back() - min_low) / range;
        } else {
            row.close_position = 0.5f;
        }
    }

    float rolling_std(const std::deque<float>& data, size_t window) {
        if (data.size() < window) return std::numeric_limits<float>::quiet_NaN();
        float sum = 0.0f, sum_sq = 0.0f;
        for (size_t i = data.size() - window; i < data.size(); ++i) {
            sum += data[i];
            sum_sq += data[i] * data[i];
        }
        float n = static_cast<float>(window);
        float mean = sum / n;
        float var = sum_sq / n - mean * mean;
        if (var < 0.0f) var = 0.0f;
        return std::sqrt(var);
    }

    float high_low_range(size_t window) {
        float max_h = *std::max_element(high_mids_.end() - static_cast<int>(window), high_mids_.end());
        float min_l = *std::min_element(low_mids_.end() - static_cast<int>(window), low_mids_.end());
        return (max_h - min_l) / tick_size_;
    }

    // -----------------------------------------------------------------------
    // Category 4: Cross-Scale Dynamics
    // -----------------------------------------------------------------------
    void compute_cross_scale(const Bar& bar, BarFeatureRow& row) {
        float vol = static_cast<float>(bar.volume);
        float dur = bar.bar_duration_s;

        if (!ewma_initialized_) {
            ewma_volume_ = vol;
            ewma_duration_ = dur;
            ewma_initialized_ = true;
        } else {
            ewma_volume_ = EWMA_ALPHA * vol + (1.0f - EWMA_ALPHA) * ewma_volume_;
            ewma_duration_ = EWMA_ALPHA * dur + (1.0f - EWMA_ALPHA) * ewma_duration_;
        }

        row.volume_surprise = (ewma_volume_ > EPS) ? vol / ewma_volume_ : 1.0f;
        row.duration_surprise = (ewma_duration_ > EPS) ? dur / ewma_duration_ : 1.0f;

        // Acceleration: return_1 - prev_return_1
        float curr_return_1 = row.return_1;
        if (!std::isnan(curr_return_1) && !std::isnan(prev_return_1_)) {
            row.acceleration = curr_return_1 - prev_return_1_;
        } else {
            row.acceleration = 0.0f;
        }
        prev_return_1_ = curr_return_1;

        // vol_price_corr: rolling corr(volume, |return_1|) over 20 bars
        row.vol_price_corr = compute_vol_price_corr();
    }

    float compute_vol_price_corr() {
        if (volumes_.size() < 20 || abs_returns_.size() < 20) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        constexpr size_t n = 20;
        // Copy the last 20 elements into contiguous arrays for pearson_corr
        std::vector<float> vols(volumes_.end() - n, volumes_.end());
        std::vector<float> rets(abs_returns_.end() - n, abs_returns_.end());
        return pearson_corr(vols.data(), rets.data(), n);
    }

    // -----------------------------------------------------------------------
    // Category 5: Time Context
    // -----------------------------------------------------------------------
    void compute_time_context(const Bar& bar, BarFeatureRow& row) {
        float tod = bar.time_of_day;
        float frac = tod / 24.0f;
        row.time_sin = std::sin(TWO_PI * frac);
        row.time_cos = std::cos(TWO_PI * frac);

        row.minutes_since_open = (tod - RTH_OPEN_HOUR) * 60.0f;
        if (row.minutes_since_open < 0.0f) row.minutes_since_open = 0.0f;

        row.minutes_to_close = (RTH_CLOSE_HOUR - tod) * 60.0f;
        if (row.minutes_to_close < 0.0f) row.minutes_to_close = 0.0f;

        // Session volume fraction
        cumulative_volume_ += static_cast<float>(bar.volume);
        if (!prior_day_totals_.empty()) {
            // Day 2+: use expanding-window mean of prior days
            float sum = 0.0f;
            for (float t : prior_day_totals_) sum += t;
            float avg = sum / static_cast<float>(prior_day_totals_.size());
            row.session_volume_frac = (avg > EPS) ? cumulative_volume_ / avg : 0.0f;
        } else {
            // Day 1: use cumulative / cumulative = 1.0 (at end)
            // During day 1, we don't know total, so just accumulate
            row.session_volume_frac = 0.0f;
        }
    }

    // -----------------------------------------------------------------------
    // Category 6: Message Microstructure
    // -----------------------------------------------------------------------
    void compute_message_microstructure(const Bar& bar, BarFeatureRow& row) {
        row.cancel_add_ratio = static_cast<float>(bar.cancel_count) /
                               (static_cast<float>(bar.add_count) + EPS);

        float total_msgs = static_cast<float>(bar.add_count + bar.cancel_count + bar.modify_count);
        row.message_rate = (bar.bar_duration_s > EPS) ? total_msgs / bar.bar_duration_s : 0.0f;

        row.modify_fraction = (total_msgs > EPS)
            ? static_cast<float>(bar.modify_count) / (total_msgs + EPS)
            : 0.0f;

        // Order flow toxicity: fraction of trades that move mid.
        // Simple proxy: |close_mid - open_mid| / (tick_size * trade_count + eps)
        // Bounded to [0, 1]
        if (bar.trade_event_count > 0) {
            float mid_move = std::abs(bar.close_mid - bar.open_mid);
            float max_possible = tick_size_ * static_cast<float>(bar.trade_event_count);
            row.order_flow_toxicity = std::min(1.0f, mid_move / (max_possible + EPS));
        } else {
            row.order_flow_toxicity = 0.0f;
        }

        // Cancel concentration: proxy HHI. Without per-level cancel data,
        // use a bounded value derived from cancel_add_ratio.
        // If all cancels at one level, HHI=1. If spread, HHI→0.
        // Simple proxy: min(1, cancel_count / (level_count * avg_cancel_per_level))
        row.cancel_concentration = std::min(1.0f,
            static_cast<float>(bar.cancel_count) /
            (static_cast<float>(bar.cancel_count + bar.add_count + bar.modify_count) + EPS));
    }

    // -----------------------------------------------------------------------
    // Warmup flag
    // -----------------------------------------------------------------------
    bool compute_is_warmup() {
        // Warmup until all features have sufficient lookback.
        // Max window is 50 (volatility_50, high_low_range_50).
        // bar_count_ is 0-indexed (incremented AFTER this call).
        return bar_count_ < 50;
    }

    // -----------------------------------------------------------------------
    // Fixup helpers for batch mode
    // -----------------------------------------------------------------------
    static float std_of_slice(const std::vector<float>& data, size_t start, size_t count) {
        float sum = 0, sum_sq = 0;
        for (size_t j = start; j < start + count; ++j) {
            sum += data[j];
            sum_sq += data[j] * data[j];
        }
        float nf = static_cast<float>(count);
        float mean = sum / nf;
        float var = sum_sq / nf - mean * mean;
        if (var < 0.0f) var = 0.0f;
        return std::sqrt(var);
    }

    static float pearson_corr(const float* xs, const float* ys, size_t n) {
        float sx = 0, sy = 0, sxy = 0, sxx = 0, syy = 0;
        for (size_t i = 0; i < n; ++i) {
            sx += xs[i]; sy += ys[i];
            sxy += xs[i] * ys[i];
            sxx += xs[i] * xs[i];
            syy += ys[i] * ys[i];
        }
        float nf = static_cast<float>(n);
        float cov = nf * sxy - sx * sy;
        float vx = nf * sxx - sx * sx;
        float vy = nf * syy - sy * sy;
        float denom = std::sqrt(vx * vy);
        if (denom < EPS) return 0.0f;
        return std::max(-1.0f, std::min(1.0f, cov / denom));
    }

    static float high_low_range_of(const std::vector<Bar>& bars, size_t end, size_t count,
                                    float tick_size) {
        float max_h = bars[end].high_mid, min_l = bars[end].low_mid;
        for (size_t j = end + 1 - count; j <= end; ++j) {
            max_h = std::max(max_h, bars[j].high_mid);
            min_l = std::min(min_l, bars[j].low_mid);
        }
        return (max_h - min_l) / tick_size;
    }

    // -----------------------------------------------------------------------
    // Fixup NaN rolling features in batch mode
    // -----------------------------------------------------------------------
    void fixup_rolling_features(const std::vector<Bar>& bars, std::vector<BarFeatureRow>& rows) {
        size_t n = bars.size();

        // Collect all 1-bar returns for volatility fixup
        std::vector<float> all_returns;
        for (size_t i = 1; i < n; ++i) {
            all_returns.push_back((bars[i].close_mid - bars[i-1].close_mid) / tick_size_);
        }

        for (size_t i = 0; i < n; ++i) {
            // Fix volatility_20: use available returns if < 20 but >= 2
            if (std::isnan(rows[i].volatility_20) && i >= 2) {
                size_t count = std::min(i, size_t(20));
                rows[i].volatility_20 = std_of_slice(all_returns, i - count, count);
            }

            // Fix volatility_50: use available returns if < 50 but >= 2
            if (std::isnan(rows[i].volatility_50) && i >= 2) {
                size_t count = std::min(i, size_t(50));
                rows[i].volatility_50 = std_of_slice(all_returns, i - count, count);
            }

            // Fix kyle_lambda: use available data if < 20 but >= 2
            if (std::isnan(rows[i].kyle_lambda) && i >= 2) {
                size_t count = std::min(i, size_t(20));
                float sum_xy = 0, sum_xx = 0;
                for (size_t j = i - count; j < i; ++j) {
                    float delta_mid = all_returns[j];
                    float nv = bars[j+1].buy_volume - bars[j+1].sell_volume;
                    sum_xy += nv * delta_mid;
                    sum_xx += nv * nv;
                }
                rows[i].kyle_lambda = (sum_xx > EPS) ? sum_xy / sum_xx : 0.0f;
            }

            // Fix vol_price_corr: use available data if < 20 but >= 2
            if (std::isnan(rows[i].vol_price_corr) && i >= 2) {
                size_t count = std::min(i, size_t(20));
                std::vector<float> vols(count), abs_rets(count);
                for (size_t j = 0; j < count; ++j) {
                    vols[j] = static_cast<float>(bars[i - count + j + 1].volume);
                    abs_rets[j] = std::abs(all_returns[i - count + j]);
                }
                rows[i].vol_price_corr = pearson_corr(vols.data(), abs_rets.data(), count);
            }

            // Fix high_low_range_20: use available data
            if (std::isnan(rows[i].high_low_range_20) && i >= 1) {
                size_t count = std::min(i + 1, size_t(20));
                rows[i].high_low_range_20 = high_low_range_of(bars, i, count, tick_size_);
            }

            // Fix high_low_range_50: use available data
            if (std::isnan(rows[i].high_low_range_50) && i >= 1) {
                size_t count = std::min(i + 1, size_t(50));
                rows[i].high_low_range_50 = high_low_range_of(bars, i, count, tick_size_);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Forward returns
    // -----------------------------------------------------------------------
    float fwd_return(const std::vector<Bar>& bars, size_t i, size_t horizon) {
        if (i + horizon < bars.size()) {
            return (bars[i + horizon].close_mid - bars[i].close_mid) / tick_size_;
        }
        return std::numeric_limits<float>::quiet_NaN();
    }

    void fill_forward_returns(const std::vector<Bar>& bars, std::vector<BarFeatureRow>& rows) {
        for (size_t i = 0; i < bars.size(); ++i) {
            rows[i].fwd_return_1   = fwd_return(bars, i, 1);
            rows[i].fwd_return_5   = fwd_return(bars, i, 5);
            rows[i].fwd_return_20  = fwd_return(bars, i, 20);
            rows[i].fwd_return_100 = fwd_return(bars, i, 100);
        }
    }
};
