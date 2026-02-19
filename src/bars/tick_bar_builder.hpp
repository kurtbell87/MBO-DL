#pragma once

#include "bars/bar_builder_base.hpp"

class TickBarBuilder : public BarBuilderBase {
public:
    explicit TickBarBuilder(uint32_t threshold) : threshold_(threshold) {}

    std::optional<Bar> on_snapshot(const BookSnapshot& snap) override {
        auto trade = extract_trade(snap);

        // Effective trade count: use trade_count if set, else fall back to
        // legacy behavior (1 if trade buffer has data, 0 otherwise)
        uint32_t tc = (snap.trade_count > 0)
                          ? snap.trade_count
                          : (trade.has_trade ? 1u : 0u);

        if (!active_) {
            // Don't start a bar unless there are trades (or carry-over)
            if (tc == 0 && carry_ == 0) return std::nullopt;
            BarBuilderBase::start_bar_at(snap);
            tick_count_ = carry_;
            carry_ = 0;
        }

        update_bar(snap, trade);

        // Undo base class tick_count_ increment (+1 per trade snapshot)
        // and replace with actual trade count
        if (trade.has_trade) {
            tick_count_--;
        }
        tick_count_ += tc;

        if (tick_count_ >= threshold_) {
            carry_ = tick_count_ - threshold_;
            tick_count_ = threshold_;
            auto bar = finalize_bar();

            // If carry-over exists, start next bar immediately so flush() works
            if (carry_ > 0) {
                BarBuilderBase::start_bar_at(snap);
                tick_count_ = carry_;
                carry_ = 0;
            }

            return bar;
        }

        return std::nullopt;
    }

    std::optional<Bar> flush() override {
        if (!active_) return std::nullopt;
        // Only emit partial if there are accumulated trades
        if (tick_count_ == 0) {
            active_ = false;
            return std::nullopt;
        }
        return finalize_bar();
    }

private:
    uint32_t threshold_;
    uint32_t carry_ = 0;
};
