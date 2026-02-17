#pragma once

#include "bars/bar_builder_base.hpp"

class TickBarBuilder : public BarBuilderBase {
public:
    explicit TickBarBuilder(uint32_t threshold) : threshold_(threshold) {}

    std::optional<Bar> on_snapshot(const BookSnapshot& snap) override {
        auto trade = extract_trade(snap);

        if (!active_) {
            if (!trade.has_trade) return std::nullopt;
            BarBuilderBase::start_bar_at(snap);
        }

        update_bar(snap, trade);

        if (tick_count_ >= threshold_) {
            return finalize_bar();
        }

        return std::nullopt;
    }

private:
    uint32_t threshold_;
};
