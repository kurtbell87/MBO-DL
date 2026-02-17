#pragma once

#include "time_utils.hpp"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <map>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// BookSnapshot — output struct for a single 100ms snapshot
// ---------------------------------------------------------------------------
constexpr int    BOOK_DEPTH       = 10;              // levels per side
constexpr int    TRADE_BUF_LEN    = 50;              // trade buffer length
constexpr uint64_t SNAPSHOT_INTERVAL_NS = 100'000'000ULL; // 100ms in nanoseconds

struct BookSnapshot {
    uint64_t timestamp = 0;
    float bids[BOOK_DEPTH][2] = {};      // levels × (price, size), descending by price
    float asks[BOOK_DEPTH][2] = {};      // levels × (price, size), ascending by price
    float trades[TRADE_BUF_LEN][3] = {}; // (price, size, aggressor_side), left-padded
    float mid_price = 0.0f;
    float spread = 0.0f;
    float time_of_day = 0.0f;            // fractional hours since midnight ET
};

// ---------------------------------------------------------------------------
// BookBuilder — reconstructs an order book from MBO events and emits snapshots
// ---------------------------------------------------------------------------
class BookBuilder {
public:
    explicit BookBuilder(uint32_t instrument_id)
        : instrument_id_(instrument_id) {}

    // Process a single MBO event
    void process_event(uint64_t ts_event, uint64_t order_id, uint32_t instrument_id,
                       char action, char side, int64_t price, uint32_t size, uint8_t flags) {
        // Instrument filtering
        if (instrument_id != instrument_id_) return;

        constexpr uint8_t F_LAST = 0x80;
        constexpr uint8_t F_SNAPSHOT = 0x20;

        // Apply event to pending state
        switch (action) {
            case 'A': apply_add(order_id, side, price, size); break;
            case 'C': apply_cancel(order_id, side, price); break;
            case 'M': apply_modify(order_id, side, price, size); break;
            case 'T': apply_trade(ts_event, side, price, size); break;
            case 'F': apply_fill(order_id, price, size); break;
            case 'R': apply_clear(flags); break;
        }

        // On F_LAST, commit pending state to committed state
        if (flags & F_LAST) {
            commit(ts_event);
        }
    }

    // Emit snapshots at 100ms boundaries within [start_ns, end_ns)
    // Only emits during RTH (09:30:00 - 16:00:00 ET)
    std::vector<BookSnapshot> emit_snapshots(uint64_t start_ns, uint64_t end_ns) {
        std::vector<BookSnapshot> result;

        constexpr uint64_t BOUNDARY = SNAPSHOT_INTERVAL_NS;

        // Clamp to RTH window
        uint64_t rth_open = rth_open_ns(start_ns);
        uint64_t rth_close = rth_close_ns(start_ns);

        uint64_t eff_start = std::max(start_ns, rth_open);
        uint64_t eff_end = std::min(end_ns, rth_close);

        if (eff_start >= eff_end) return result;

        // Align start to the first 100ms boundary (relative to rth_open) >= eff_start
        uint64_t offset = eff_start - rth_open;
        uint64_t aligned_start = rth_open + ((offset + BOUNDARY - 1) / BOUNDARY) * BOUNDARY;

        // Pre-scan committed states UP TO the first boundary to set carry-forward state.
        // This handles cases where both sides existed in pre-market but one was canceled.
        bool both_sides_seen = false;
        float carry_mid = 0.0f;
        float carry_spread = 0.0f;
        for (const auto& cs : committed_states_) {
            if (cs.ts > aligned_start) break;
            if (cs.has_bid && cs.has_ask) {
                auto [mid, sprd] = compute_mid_spread(cs.bid_levels, cs.ask_levels);
                carry_mid = mid;
                carry_spread = sprd;
                both_sides_seen = true;
            }
        }
        last_mid_price_ = carry_mid;
        last_spread_ = carry_spread;
        ever_had_both_sides_ = both_sides_seen;

        for (uint64_t ts = aligned_start; ts < eff_end; ts += BOUNDARY) {
            // Find the committed book state at this timestamp
            auto state = get_committed_state_at(ts);
            if (!state) continue;

            // Update ever_had_both_sides_ based on committed states up to this boundary
            if (state->has_bid && state->has_ask) {
                ever_had_both_sides_ = true;
            }

            // Skip if book doesn't have both bid and ask and never did
            if (!state->has_bid || !state->has_ask) {
                if (!ever_had_both_sides_) continue;
            }

            BookSnapshot snap{};
            snap.timestamp = ts;

            // Fill bids (descending by price)
            int bid_idx = 0;
            for (auto it = state->bid_levels.rbegin();
                 it != state->bid_levels.rend() && bid_idx < BOOK_DEPTH; ++it, ++bid_idx) {
                snap.bids[bid_idx][0] = fixed_to_float(it->first);
                snap.bids[bid_idx][1] = static_cast<float>(it->second);
            }

            // Fill asks (ascending by price)
            int ask_idx = 0;
            for (auto it = state->ask_levels.begin();
                 it != state->ask_levels.end() && ask_idx < BOOK_DEPTH; ++it, ++ask_idx) {
                snap.asks[ask_idx][0] = fixed_to_float(it->first);
                snap.asks[ask_idx][1] = static_cast<float>(it->second);
            }

            // Mid price and spread
            if (state->has_bid && state->has_ask) {
                auto [mid, sprd] = compute_mid_spread(state->bid_levels, state->ask_levels);
                snap.mid_price = mid;
                snap.spread = sprd;
                last_mid_price_ = mid;
                last_spread_ = sprd;
            } else if (ever_had_both_sides_) {
                snap.mid_price = last_mid_price_;
                snap.spread = last_spread_;
            }

            // Fill trades — use full accumulated trade buffer
            fill_trades(snap);

            // Time of day: fractional hours since midnight ET
            snap.time_of_day = compute_time_of_day(ts);

            result.push_back(snap);
        }

        return result;
    }

private:
    uint32_t instrument_id_;

    // Per-order tracking: order_id -> {side, price, size}
    struct OrderInfo {
        char side;
        int64_t price;
        uint32_t size;
    };
    std::unordered_map<uint64_t, OrderInfo> orders_;

    // Aggregated price levels (price -> total size)
    std::map<int64_t, uint32_t> bid_levels_;  // ascending by price (rbegin = best bid)
    std::map<int64_t, uint32_t> ask_levels_;  // ascending by price (begin = best ask)

    // Committed state snapshots (after F_LAST)
    struct CommittedState {
        uint64_t ts;
        std::map<int64_t, uint32_t> bid_levels;
        std::map<int64_t, uint32_t> ask_levels;
        bool has_bid;
        bool has_ask;
    };
    std::vector<CommittedState> committed_states_;

    // Trade buffer (rolling, max TRADE_BUF_LEN)
    struct TradeRecord {
        float price;
        float size;
        float aggressor_side; // +1.0 for 'B', -1.0 for 'A'
    };
    std::deque<TradeRecord> trades_;

    // Carry-forward mid/spread
    float last_mid_price_ = 0.0f;
    float last_spread_ = 0.0f;
    bool ever_had_both_sides_ = false;

    static float fixed_to_float(int64_t fixed) {
        return static_cast<float>(static_cast<double>(fixed) / 1e9);
    }

    // Compute mid_price and spread from bid/ask level maps.
    // Returns {mid_price, spread}. Caller must ensure both sides are non-empty.
    static std::pair<float, float> compute_mid_spread(
        const std::map<int64_t, uint32_t>& bids,
        const std::map<int64_t, uint32_t>& asks) {
        float best_bid = fixed_to_float(bids.rbegin()->first);
        float best_ask = fixed_to_float(asks.begin()->first);
        return {(best_bid + best_ask) / 2.0f, best_ask - best_bid};
    }

    // Helpers for level manipulation
    std::map<int64_t, uint32_t>& levels_for(char side) {
        return (side == 'B') ? bid_levels_ : ask_levels_;
    }

    void add_to_level(char side, int64_t price, uint32_t size) {
        levels_for(side)[price] += size;
    }

    void remove_from_level(const OrderInfo& info) {
        auto& levels = levels_for(info.side);
        auto lvl = levels.find(info.price);
        if (lvl != levels.end()) {
            if (lvl->second <= info.size) {
                levels.erase(lvl);
            } else {
                lvl->second -= info.size;
            }
        }
    }

    void apply_add(uint64_t order_id, char side, int64_t price, uint32_t size) {
        orders_[order_id] = {side, price, size};
        add_to_level(side, price, size);
    }

    void apply_cancel(uint64_t order_id, char /*side*/, int64_t /*price*/) {
        auto it = orders_.find(order_id);
        if (it == orders_.end()) return;
        remove_from_level(it->second);
        orders_.erase(it);
    }

    void apply_modify(uint64_t order_id, char side, int64_t new_price, uint32_t new_size) {
        auto it = orders_.find(order_id);
        if (it != orders_.end()) {
            remove_from_level(it->second);
        }
        orders_[order_id] = {side, new_price, new_size};
        add_to_level(side, new_price, new_size);
    }

    void apply_trade(uint64_t /*ts*/, char side, int64_t price, uint32_t size) {
        float agg = (side == 'B') ? 1.0f : -1.0f;
        trades_.push_back({fixed_to_float(price), static_cast<float>(size), agg});
        if (trades_.size() > static_cast<size_t>(TRADE_BUF_LEN)) {
            trades_.pop_front();
        }
    }

    void apply_fill(uint64_t order_id, int64_t /*price*/, uint32_t remaining_size) {
        auto it = orders_.find(order_id);
        if (it == orders_.end()) return;

        auto& info = it->second;
        remove_from_level(info);

        if (remaining_size == 0) {
            orders_.erase(it);
        } else {
            info.size = remaining_size;
            add_to_level(info.side, info.price, remaining_size);
        }
    }

    void apply_clear(uint8_t /*flags*/) {
        orders_.clear();
        bid_levels_.clear();
        ask_levels_.clear();
    }

    void commit(uint64_t ts) {
        committed_states_.push_back({
            ts,
            bid_levels_,
            ask_levels_,
            !bid_levels_.empty(),
            !ask_levels_.empty()
        });
    }

    const CommittedState* get_committed_state_at(uint64_t ts) const {
        // Binary search: find the latest committed state with ts <= boundary ts.
        // committed_states_ is sorted by ts (events arrive in order).
        auto it = std::upper_bound(
            committed_states_.begin(), committed_states_.end(), ts,
            [](uint64_t boundary, const CommittedState& cs) { return boundary < cs.ts; });
        if (it == committed_states_.begin()) return nullptr;
        return &(*std::prev(it));
    }

    void fill_trades(BookSnapshot& snap) {
        size_t count = trades_.size();
        size_t start_idx = TRADE_BUF_LEN - count;
        for (size_t i = 0; i < count; ++i) {
            snap.trades[start_idx + i][0] = trades_[i].price;
            snap.trades[start_idx + i][1] = trades_[i].size;
            snap.trades[start_idx + i][2] = trades_[i].aggressor_side;
        }
    }

    static uint64_t rth_open_ns(uint64_t ts) {
        return time_utils::rth_open_ns(ts);
    }

    static uint64_t rth_close_ns(uint64_t ts) {
        return time_utils::rth_close_ns(ts);
    }

    static float compute_time_of_day(uint64_t ts) {
        return time_utils::compute_time_of_day(ts);
    }
};
