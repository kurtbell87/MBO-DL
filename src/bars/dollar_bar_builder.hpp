#pragma once

#include "bars/bar_builder_base.hpp"

class DollarBarBuilder : public BarBuilderBase {
public:
    explicit DollarBarBuilder(double threshold, float multiplier = 5.0f)
        : threshold_(threshold), multiplier_(multiplier) {}

    std::optional<Bar> on_snapshot(const BookSnapshot& snap) override {
        auto trade = extract_trade(snap);

        if (!active_) {
            if (!trade.has_trade) return std::nullopt;
            BarBuilderBase::start_bar_at(snap);
        }

        update_bar(snap, trade);

        if (trade.has_trade) {
            cumulative_dollar_volume_ += static_cast<double>(trade.price) *
                                          static_cast<double>(trade.size) *
                                          static_cast<double>(multiplier_);
        }

        if (cumulative_dollar_volume_ >= threshold_) {
            cumulative_dollar_volume_ = 0.0;
            return finalize_bar();
        }

        return std::nullopt;
    }

private:
    double threshold_;
    float multiplier_;
    double cumulative_dollar_volume_ = 0.0;
};
