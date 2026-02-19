#pragma once

#include "bars/bar.hpp"
#include "time_utils.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>

// ---------------------------------------------------------------------------
// BarBuilderBase — common accumulation logic for all bar builder types
// ---------------------------------------------------------------------------
class BarBuilderBase : public BarBuilder {
public:
    std::optional<Bar> flush() override {
        if (!active_) return std::nullopt;
        return finalize_bar();
    }

protected:
    bool active_ = false;

    uint64_t open_ts_ = 0;
    uint64_t close_ts_ = 0;
    float open_mid_ = 0.0f;
    float close_mid_ = 0.0f;
    float high_mid_ = 0.0f;
    float low_mid_ = 0.0f;
    uint32_t cumulative_volume_ = 0;
    uint32_t tick_count_ = 0;
    float buy_volume_ = 0.0f;
    float sell_volume_ = 0.0f;
    double vwap_num_ = 0.0;
    double vwap_den_ = 0.0;

    float max_spread_ = 0.0f;
    float min_spread_ = std::numeric_limits<float>::max();
    uint32_t snapshot_count_ = 0;

    uint32_t mbo_event_begin_ = 0;
    uint32_t mbo_event_end_ = 0;

    uint32_t add_count_ = 0;
    uint32_t cancel_count_ = 0;
    uint32_t modify_count_ = 0;
    uint32_t trade_event_count_ = 0;

    BookSnapshot last_snap_{};

    // Extract trade info from the most recent trade slot in a snapshot.
    struct TradeInfo {
        float price;
        float size;
        float aggressor;
        bool has_trade;
    };

    static TradeInfo extract_trade(const BookSnapshot& snap) {
        float price = snap.trades[TRADE_BUF_LEN - 1][0];
        float size  = snap.trades[TRADE_BUF_LEN - 1][1];
        float agg   = snap.trades[TRADE_BUF_LEN - 1][2];
        return {price, size, agg, size > 0.0f};
    }

    void start_bar_at(const BookSnapshot& snap) {
        active_ = true;
        open_ts_ = snap.timestamp;
        open_mid_ = snap.mid_price;
        reset_accumulators(snap.mid_price);
    }

    void start_bar_contiguous(float mid_price) {
        active_ = true;
        open_ts_ = close_ts_;
        open_mid_ = mid_price;
        reset_accumulators(mid_price);
    }

    void update_bar(const BookSnapshot& snap, const TradeInfo& trade) {
        close_ts_ = snap.timestamp;
        close_mid_ = snap.mid_price;
        high_mid_ = std::max(high_mid_, snap.mid_price);
        low_mid_ = std::min(low_mid_, snap.mid_price);

        max_spread_ = std::max(max_spread_, snap.spread);
        min_spread_ = std::min(min_spread_, snap.spread);
        snapshot_count_++;

        if (trade.has_trade) {
            uint32_t size = static_cast<uint32_t>(trade.size);
            cumulative_volume_ += size;
            tick_count_++;

            if (trade.aggressor > 0.0f) {
                buy_volume_ += trade.size;
            } else {
                sell_volume_ += trade.size;
            }

            vwap_num_ += static_cast<double>(trade.price) * static_cast<double>(trade.size);
            vwap_den_ += static_cast<double>(trade.size);

            trade_event_count_++;
        }

        add_count_++;
        mbo_event_end_++;
        last_snap_ = snap;
    }

    std::optional<Bar> finalize_bar() {
        if (!active_) return std::nullopt;

        Bar bar{};
        bar.open_ts = open_ts_;
        bar.close_ts = close_ts_;
        bar.open_mid = open_mid_;
        bar.close_mid = close_mid_;
        bar.high_mid = high_mid_;
        bar.low_mid = low_mid_;
        bar.volume = cumulative_volume_;
        bar.tick_count = tick_count_;
        bar.buy_volume = buy_volume_;
        bar.sell_volume = sell_volume_;

        if (vwap_den_ > 0.0) {
            bar.vwap = static_cast<float>(vwap_num_ / vwap_den_);
        }

        bar.bar_duration_s = static_cast<float>(bar.close_ts - bar.open_ts) /
                             static_cast<float>(time_utils::NS_PER_SEC);
        bar.time_of_day = time_utils::compute_time_of_day(bar.close_ts);

        for (int i = 0; i < BOOK_DEPTH; ++i) {
            bar.bids[i][0] = last_snap_.bids[i][0];
            bar.bids[i][1] = last_snap_.bids[i][1];
            bar.asks[i][0] = last_snap_.asks[i][0];
            bar.asks[i][1] = last_snap_.asks[i][1];
        }
        bar.spread = last_snap_.spread;

        bar.max_spread = max_spread_;
        bar.min_spread = (min_spread_ == std::numeric_limits<float>::max()) ? 0.0f : min_spread_;
        bar.snapshot_count = snapshot_count_;

        bar.mbo_event_begin = mbo_event_begin_;
        bar.mbo_event_end = mbo_event_end_;

        bar.add_count = add_count_;
        bar.cancel_count = cancel_count_;
        bar.modify_count = modify_count_;
        bar.trade_event_count = trade_event_count_;
        bar.cancel_add_ratio = static_cast<float>(bar.cancel_count) /
                               (static_cast<float>(bar.add_count) + 1e-8f);
        if (bar.bar_duration_s > 0.0f) {
            float total_msgs = static_cast<float>(bar.add_count + bar.cancel_count +
                                                   bar.modify_count + bar.trade_event_count);
            bar.message_rate = total_msgs / bar.bar_duration_s;
        }

        active_ = false;
        return bar;
    }

private:
    void reset_accumulators(float mid_price) {
        high_mid_ = mid_price;
        low_mid_ = mid_price;
        cumulative_volume_ = 0;
        tick_count_ = 0;
        buy_volume_ = 0.0f;
        sell_volume_ = 0.0f;
        vwap_num_ = 0.0;
        vwap_den_ = 0.0;
        max_spread_ = 0.0f;
        min_spread_ = std::numeric_limits<float>::max();
        snapshot_count_ = 0;
        mbo_event_begin_ = mbo_event_end_;
        add_count_ = 0;
        cancel_count_ = 0;
        modify_count_ = 0;
        trade_event_count_ = 0;
    }
};
