#pragma once

#include "bars/bar.hpp"
#include "book_builder.hpp"
#include "data/day_event_buffer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// B.1: BookSnapshotExport — PriceLadderInput shape (20, 2)
// ---------------------------------------------------------------------------
struct BookSnapshotExport {
    float data[20][2] = {};

    // Create from a Bar using the same deepest-first bid ordering as PriceLadderInput.
    static BookSnapshotExport from_bar(const Bar& bar) {
        BookSnapshotExport pli{};
        float mid = bar.close_mid;

        // Rows 0-9: bids in reverse order (deepest first, best bid at row 9)
        for (int i = 0; i < 10; ++i) {
            int bid_idx = 9 - i;  // row 0 = deepest bid (index 9)
            pli.data[i][0] = bar.bids[bid_idx][0] - mid;
            pli.data[i][1] = bar.bids[bid_idx][1];
        }
        // Rows 10-19: asks in order (best ask at row 10)
        for (int i = 0; i < 10; ++i) {
            pli.data[10 + i][0] = bar.asks[i][0] - mid;
            pli.data[10 + i][1] = bar.asks[i][1];
        }
        return pli;
    }

    // Flatten to a 40-element vector.
    static std::vector<float> flatten(const Bar& bar) {
        auto pli = from_bar(bar);
        std::vector<float> flat;
        flat.reserve(40);
        for (int i = 0; i < 20; ++i) {
            flat.push_back(pli.data[i][0]);
            flat.push_back(pli.data[i][1]);
        }
        return flat;
    }
};

// ---------------------------------------------------------------------------
// B.2: MessageSummary — fixed-length binned action counts
// ---------------------------------------------------------------------------
struct MessageSummary {
    // Fixed-length output: 10 time-decile bins × 3 action types = 30
    //   + 2 (cancel/add first half, cancel/add second half)
    //   + 1 (max instantaneous message rate)
    //   = 33 total
    static constexpr size_t SUMMARY_SIZE = 33;

    static std::vector<float> from_bar(const Bar& bar, const DayEventBuffer& buf) {
        std::vector<float> summary(SUMMARY_SIZE, 0.0f);

        auto events = buf.get_events(bar.mbo_event_begin, bar.mbo_event_end);
        if (events.empty()) {
            return summary;
        }

        // Bin events into 10 time deciles
        // Each decile has 3 action counts: add, cancel, modify
        float duration = bar.bar_duration_s;
        if (duration <= 0.0f) duration = 1.0f;

        uint64_t bar_start = bar.open_ts;
        uint64_t bar_end = bar.close_ts;
        uint64_t bar_range = (bar_end > bar_start) ? (bar_end - bar_start) : 1;

        // Decile bins: [0..29] = 10 deciles × 3 action types
        // [30] = cancel/add rate first half
        // [31] = cancel/add rate second half
        // [32] = max instantaneous message rate

        uint32_t first_half_adds = 0, first_half_cancels = 0;
        uint32_t second_half_adds = 0, second_half_cancels = 0;

        for (const auto& ev : events) {
            // Determine decile (0-9)
            float frac = static_cast<float>(ev.ts_event - bar_start) /
                         static_cast<float>(bar_range);
            frac = std::max(0.0f, std::min(frac, 0.999f));
            int decile = static_cast<int>(frac * 10.0f);

            // Map action to index offset
            int action_offset = -1;
            if (ev.action == 0) action_offset = 0;      // Add
            else if (ev.action == 1) action_offset = 1;  // Cancel
            else if (ev.action == 2) action_offset = 2;  // Modify

            if (action_offset >= 0) {
                summary[decile * 3 + action_offset] += 1.0f;
            }

            // First/second half tracking
            bool first_half = (frac < 0.5f);
            if (ev.action == 0) {  // Add
                if (first_half) first_half_adds++;
                else second_half_adds++;
            } else if (ev.action == 1) {  // Cancel
                if (first_half) first_half_cancels++;
                else second_half_cancels++;
            }
        }

        constexpr float EPS = 1e-8f;
        summary[30] = static_cast<float>(first_half_cancels) /
                       (static_cast<float>(first_half_adds) + EPS);
        summary[31] = static_cast<float>(second_half_cancels) /
                       (static_cast<float>(second_half_adds) + EPS);

        // Max instantaneous message rate: max events in any decile / (duration/10)
        float decile_duration = duration / 10.0f;
        float max_rate = 0.0f;
        for (int d = 0; d < 10; ++d) {
            float decile_total = summary[d*3] + summary[d*3+1] + summary[d*3+2];
            float rate = decile_total / (decile_duration + EPS);
            max_rate = std::max(max_rate, rate);
        }
        summary[32] = max_rate;

        return summary;
    }
};

// ---------------------------------------------------------------------------
// B.3: LookbackBookSequence — (W, 20, 2)
// ---------------------------------------------------------------------------
class LookbackBookSequence {
public:
    LookbackBookSequence() = default;

    static LookbackBookSequence from_bars(const std::vector<Bar>& bars, int W) {
        if (static_cast<int>(bars.size()) < W) {
            throw std::invalid_argument("Insufficient bars for lookback window");
        }

        LookbackBookSequence seq;
        seq.window_size_ = W;
        seq.snapshots_.resize(W);

        // Use last W bars
        size_t start = bars.size() - W;
        for (int i = 0; i < W; ++i) {
            seq.snapshots_[i] = BookSnapshotExport::from_bar(bars[start + i]);
        }
        return seq;
    }

    static std::vector<float> flatten(const std::vector<Bar>& bars, int W) {
        auto seq = from_bars(bars, W);
        std::vector<float> flat;
        flat.reserve(W * 40);
        for (int i = 0; i < W; ++i) {
            for (int r = 0; r < 20; ++r) {
                flat.push_back(seq.snapshots_[i].data[r][0]);
                flat.push_back(seq.snapshots_[i].data[r][1]);
            }
        }
        return flat;
    }

    int window_size() const { return window_size_; }
    int rows_per_bar() const { return 20; }
    int cols_per_row() const { return 2; }

    BookSnapshotExport bar_snapshot(int idx) const {
        return snapshots_[idx];
    }

private:
    int window_size_ = 0;
    std::vector<BookSnapshotExport> snapshots_;
};

// ---------------------------------------------------------------------------
// R2 Tier 2: RawEventSequence — variable-length, max N events, padded
// ---------------------------------------------------------------------------
struct RawEventSequence {
    // Returns a vector of events, each event is a vector of floats.
    // Padded to max_events with zero vectors.
    static std::vector<std::vector<float>> from_bar(const Bar& bar, const DayEventBuffer& buf,
                                                      size_t max_events) {
        constexpr size_t FIELDS_PER_EVENT = 5;  // action, price, size, side, ts_event

        std::vector<std::vector<float>> result;
        result.reserve(max_events);

        auto events = buf.get_events(bar.mbo_event_begin, bar.mbo_event_end);

        // Take up to max_events real events
        size_t real_count = std::min(events.size(), max_events);
        for (size_t i = 0; i < real_count; ++i) {
            result.push_back({
                static_cast<float>(events[i].action),
                events[i].price,
                static_cast<float>(events[i].size),
                static_cast<float>(events[i].side),
                static_cast<float>(events[i].ts_event)
            });
        }

        // Pad to max_events with zero vectors
        while (result.size() < max_events) {
            result.push_back(std::vector<float>(FIELDS_PER_EVENT, 0.0f));
        }

        return result;
    }
};
