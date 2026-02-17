# Phase 1: Bar Construction [Engineering]

**Spec**: TRAJECTORY.md §4, §8.6 (warm-up), §10.2–10.3 (project structure, memory model)
**Predecessor**: Model Orchestrator Spec v0.6 (retired). Reuses `BookSnapshot` from `book_builder.hpp`.
**Depends on**: Nothing (foundation phase).
**Unlocks**: Phase 2 (oracle-replay), Phase R1 (subordination-test), Phase 4 (feature-computation).

---

## Objective

Implement event-driven bar construction from the validated `BookSnapshot` stream. Four bar types (volume, tick, dollar, time), a `DayEventBuffer` for raw MBO event storage, encoder input adapters, message summary computation, and warm-up state tracking.

---

## Data Contracts

### Bar Struct (pure data container)

```cpp
struct Bar {
    // Temporal
    int64_t     open_ts;
    int64_t     close_ts;
    float       time_of_day;       // fractional hours ET at bar close
    float       bar_duration_s;

    // OHLCV
    float       open_mid;
    float       close_mid;
    float       high_mid;
    float       low_mid;
    float       vwap;
    uint64_t    volume;
    uint32_t    tick_count;
    float       buy_volume;
    float       sell_volume;

    // Book state at bar close
    float       bids[10][2];       // (price, size) × 10 levels
    float       asks[10][2];
    float       spread;

    // Intra-bar spread dynamics
    float       max_spread;
    float       min_spread;
    uint32_t    snapshot_count;

    // MBO event reference (indices into DayEventBuffer)
    uint64_t    mbo_event_begin;
    uint64_t    mbo_event_end;     // exclusive

    // Message summary statistics
    uint32_t    add_count;
    uint32_t    cancel_count;
    uint32_t    modify_count;
    uint32_t    trade_event_count;
    float       cancel_add_ratio;  // cancel_count / (add_count + eps)
    float       message_rate;      // total messages / bar_duration_s
};
```

The `Bar` is a **pure data container**. Encoder-specific input formats are constructed downstream by adapter classes.

### Encoder Input Adapters

```cpp
struct PriceLadderInput {
    float data[20][2];  // [bid[9]..bid[0], ask[0]..ask[9]] × (price_delta_from_mid, normalized_size)
    static PriceLadderInput from_bar(const Bar& bar, float mid_price);
};

struct MessageSequenceInput {
    std::vector<std::array<float, 5>> events;  // (action_type, price, size, side, time_offset_ns)
    static MessageSequenceInput from_bar(const Bar& bar, const DayEventBuffer& buf);
};
```

### BarBuilder Interface

```cpp
class BarBuilder {
public:
    virtual ~BarBuilder() = default;
    virtual std::optional<Bar> on_snapshot(const BookSnapshot& snap) = 0;
    virtual std::optional<Bar> flush() = 0;
};
```

### DayEventBuffer

```cpp
struct MBOEvent {
    uint8_t     action;        // Add, Cancel, Modify, Trade
    float       price;
    uint32_t    size;
    uint8_t     side;          // Bid, Ask
    int64_t     ts_event;      // exchange timestamp
};

class DayEventBuffer {
public:
    void load(const std::string& dbn_path, uint32_t instrument_id);
    std::span<const MBOEvent> get_events(uint64_t begin, uint64_t end) const;
    size_t size() const;
    void clear();
private:
    std::vector<MBOEvent> events_;
};
```

**Memory**: After instrument_id filtering, typical MES-only event count is 2M–10M per day (~320 MB worst case). Acceptable. Load per day, clear at day boundary.

---

## Bar Types

### VolumeBarBuilder (§4.2)
- Boundary: cumulative trade volume >= V (default V=100 contracts).
- If a single trade crosses the boundary, it belongs to the current bar (no trade splitting).
- Expected: 200–1000 bars per RTH session at V=100.

### TickBarBuilder (§4.3)
- Boundary: cumulative deduplicated trade count >= K (default K=50).
- Count deduplicated trades (action='T', size > 0).

### DollarBarBuilder (§4.4)
- Boundary: cumulative dollar volume >= D (default D=50000.0).
- dollar_volume += price × size × multiplier (multiplier=5.0 for MES).

### TimeBarBuilder (§4.5)
- Boundary: wall-clock time reaches next interval boundary (default 60s).
- Aligned to session clock. Control group.

### BarFactory
- Config-driven instantiation: `BarFactory::create("volume", 100)` → `std::unique_ptr<BarBuilder>`.

---

## Message Summary Computation

During bar construction, accumulate MBO event counts:
- `add_count`, `cancel_count`, `modify_count`, `trade_event_count`
- `cancel_add_ratio = cancel_count / (add_count + 1e-8f)`
- `message_rate = (add_count + cancel_count + modify_count + trade_event_count) / bar_duration_s`

These are computed inline as snapshots are fed, not post-hoc.

---

## Warm-Up State Tracking (§8.6)

```cpp
// warmup.hpp
struct WarmupTracker {
    bool is_warmup(int bar_index, int ewma_span = 20) const;
    // Returns true if bar_index < ewma_span (EWMA features not yet stable)
    // Downstream features with rolling windows have their own NaN policy
};
```

EWMA state resets at session boundaries (each day starts fresh).

---

## Project Structure

```
src/bars/
  bar.hpp                    # Bar struct, BarBuilder interface, adapters
  volume_bar_builder.hpp
  tick_bar_builder.hpp
  dollar_bar_builder.hpp
  time_bar_builder.hpp
  bar_factory.hpp
src/data/
  day_event_buffer.hpp       # DayEventBuffer
src/features/
  warmup.hpp                 # WarmupTracker

tests/
  bar_builder_test.cpp       # All bar type tests
  day_event_buffer_test.cpp
  warmup_test.cpp
```

---

## Validation Gate

```
Assert: VolumeBarBuilder with V=100 produces bars where each bar.volume >= V
        (except possibly the last flushed bar)
Assert: TickBarBuilder with K=50 produces bars where each bar.tick_count >= K
Assert: TimeBarBuilder produces bars aligned to interval boundaries
Assert: All bar types produce identical total volume across a session
        (sum of bar volumes = total trades in snapshot stream)
Assert: Bar OHLC is consistent: low_mid <= open_mid, close_mid <= high_mid
Assert: Bars are non-overlapping and contiguous (no gaps in timestamp coverage)
Assert: flush() returns partial bar at session end
Assert: Message summary fields are consistent:
        add_count + cancel_count + modify_count + trade_event_count = total MBO events in bar
Assert: cancel_add_ratio computed with epsilon guard
Assert: mbo_event_begin < mbo_event_end for all non-empty bars
Assert: DayEventBuffer.get_events() returns correct events for bar's index range
Assert: DayEventBuffer.clear() releases memory (check with allocation counter or similar)
Assert: PriceLadderInput::from_bar() produces (20, 2) output
Assert: MessageSequenceInput::from_bar() produces correct event sequence from buffer
```
