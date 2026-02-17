#pragma once

#include "bars/bar_builder_base.hpp"

class VolumeBarBuilder : public BarBuilderBase {
public:
    explicit VolumeBarBuilder(uint32_t threshold) : threshold_(threshold) {}

    std::optional<Bar> on_snapshot(const BookSnapshot& snap) override {
        auto trade = extract_trade(snap);

        if (!active_) {
            if (!trade.has_trade) return std::nullopt;
            start_bar_at(snap);
        }

        update_bar(snap, trade);

        if (cumulative_volume_ >= threshold_) {
            auto bar = finalize_bar();
            ever_emitted_ = true;
            return bar;
        }

        return std::nullopt;
    }

    std::optional<Bar> flush() override {
        if (!active_) return std::nullopt;
        return finalize_bar();
    }

private:
    uint32_t threshold_;
    bool ever_emitted_ = false;

    void start_bar_at(const BookSnapshot& snap) {
        if (ever_emitted_) {
            BarBuilderBase::start_bar_contiguous(snap.mid_price);
        } else {
            BarBuilderBase::start_bar_at(snap);
        }
    }
};
