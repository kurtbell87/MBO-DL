#pragma once

#include "book_builder.hpp"          // BookSnapshot, BOOK_DEPTH, TRADE_BUF_LEN
#include "data/day_event_buffer.hpp" // DayEventBuffer, MBOEvent

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

// ---------------------------------------------------------------------------
// Bar — aggregated bar struct
// ---------------------------------------------------------------------------
struct Bar {
    // Temporal fields
    uint64_t open_ts = 0;
    uint64_t close_ts = 0;
    float time_of_day = 0.0f;
    float bar_duration_s = 0.0f;

    // OHLCV fields
    float open_mid = 0.0f;
    float close_mid = 0.0f;
    float high_mid = 0.0f;
    float low_mid = 0.0f;
    float vwap = 0.0f;
    uint32_t volume = 0;
    uint32_t tick_count = 0;
    float buy_volume = 0.0f;
    float sell_volume = 0.0f;

    // Book state at bar close
    float bids[BOOK_DEPTH][2] = {};
    float asks[BOOK_DEPTH][2] = {};
    float spread = 0.0f;

    // Spread dynamics
    float max_spread = 0.0f;
    float min_spread = 0.0f;
    uint32_t snapshot_count = 0;

    // MBO event references
    uint32_t mbo_event_begin = 0;
    uint32_t mbo_event_end = 0;

    // Message summary
    uint32_t add_count = 0;
    uint32_t cancel_count = 0;
    uint32_t modify_count = 0;
    uint32_t trade_event_count = 0;
    float cancel_add_ratio = 0.0f;
    float message_rate = 0.0f;
};

// ---------------------------------------------------------------------------
// BarBuilder — abstract interface for bar construction
// ---------------------------------------------------------------------------
class BarBuilder {
public:
    virtual ~BarBuilder() = default;
    virtual std::optional<Bar> on_snapshot(const BookSnapshot& snap) = 0;
    virtual std::optional<Bar> flush() = 0;
};

// ---------------------------------------------------------------------------
// PriceLadderInput — adapter for encoder input
// ---------------------------------------------------------------------------
struct PriceLadderInput {
    float data[20][2] = {};

    static PriceLadderInput from_bar(const Bar& bar, float mid_price) {
        PriceLadderInput input{};
        // Rows 0-9: bids in reverse order (deepest first, best bid at row 9)
        for (int i = 0; i < 10; ++i) {
            int bid_idx = 9 - i;  // reverse: row 0 = deepest bid (index 9)
            input.data[i][0] = bar.bids[bid_idx][0] - mid_price;
            input.data[i][1] = bar.bids[bid_idx][1];
        }
        // Rows 10-19: asks in order (best ask at row 10)
        for (int i = 0; i < 10; ++i) {
            input.data[10 + i][0] = bar.asks[i][0] - mid_price;
            input.data[10 + i][1] = bar.asks[i][1];
        }
        return input;
    }
};

// ---------------------------------------------------------------------------
// MessageSequenceInput — adapter for MBO event sequence
// ---------------------------------------------------------------------------
struct MessageSequenceInput {
    std::vector<std::vector<float>> events;

    static MessageSequenceInput from_bar(const Bar& bar, const DayEventBuffer& buf) {
        MessageSequenceInput input;
        auto span = buf.get_events(bar.mbo_event_begin, bar.mbo_event_end);
        for (const auto& ev : span) {
            input.events.push_back({
                static_cast<float>(ev.action),
                ev.price,
                static_cast<float>(ev.size),
                static_cast<float>(ev.side),
                static_cast<float>(ev.ts_event)
            });
        }
        // If no events from buffer, produce empty events for the range
        if (input.events.empty() && bar.mbo_event_end > bar.mbo_event_begin) {
            for (uint32_t i = bar.mbo_event_begin; i < bar.mbo_event_end; ++i) {
                input.events.push_back({0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
            }
        }
        return input;
    }
};
