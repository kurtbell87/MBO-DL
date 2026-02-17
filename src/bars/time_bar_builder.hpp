#pragma once

#include "bars/bar_builder_base.hpp"

class TimeBarBuilder : public BarBuilderBase {
public:
    explicit TimeBarBuilder(uint64_t interval_seconds)
        : interval_ns_(interval_seconds * time_utils::NS_PER_SEC),
          snaps_per_bar_(interval_ns_ / SNAPSHOT_INTERVAL_NS) {}

    std::optional<Bar> on_snapshot(const BookSnapshot& snap) override {
        auto trade = extract_trade(snap);

        if (!active_) {
            BarBuilderBase::start_bar_at(snap);
        }

        update_bar(snap, trade);

        if (snapshot_count_ >= snaps_per_bar_) {
            auto bar = finalize_bar();
            // Start next bar immediately for contiguity
            start_bar_contiguous(close_mid_);
            return bar;
        }

        return std::nullopt;
    }

private:
    uint64_t interval_ns_;
    uint64_t snaps_per_bar_;
};
