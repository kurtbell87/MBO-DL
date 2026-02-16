# book_builder — TDD Spec

## Summary

C++ module that reconstructs `BookSnapshot` objects at 100ms intervals from raw Databento MBO (L3) events. This is Phase 1 of the data pipeline.

## Input

- A single daily `.dbn.zst` file read via `databento::DbnFileStore`
- Target `instrument_id` (e.g., `13615` for MESM2)
- Session window: RTH 09:30:00.000 – 16:00:00.000 ET (default)

## Output

`std::vector<BookSnapshot>` — one per 100ms boundary during RTH.

```cpp
struct BookSnapshot {
    uint64_t timestamp;          // exchange timestamp (nanoseconds since epoch)
    float bids[10][2];           // L=10 price levels × (price, size)
    float asks[10][2];           // L=10 price levels × (price, size)
    float trades[50][3];         // T=50 trades × (price, size, aggressor_side)
    float mid_price;             // (best_bid + best_ask) / 2
    float spread;                // best_ask - best_bid (in ticks)
    float time_of_day;           // fractional hours since midnight ET
};
```

## Core Data Structures

```cpp
struct Order {
    int64_t  price;      // fixed-point (1e-9 scale)
    uint32_t size;
    char     side;       // 'A' or 'B'
};

// Order map: order_id → Order
std::unordered_map<uint64_t, Order> orders_;

// Bid aggregation: price (descending) → total size
std::map<int64_t, uint32_t, std::greater<>> bids_;

// Ask aggregation: price (ascending) → total size
std::map<int64_t, uint32_t> asks_;

// Rolling trade buffer: last 50 trades
// Each entry: (price_fixed, size, aggressor_side, sequence)
```

## MBO Event Processing Rules

| Action | Rule |
|--------|------|
| `'R'` (Clear) | Clear order map + both aggregation maps. If `F_SNAPSHOT` flag set, begin snapshot sequence. |
| `'A'` (Add) | Insert order, add size to aggregation at price level. |
| `'C'` (Cancel) | Lookup order, subtract size from aggregation (remove level if zero), remove order. |
| `'M'` (Modify) | Lookup order, remove old from aggregation, update price/size, add new to aggregation. |
| `'T'` (Trade) | Append `(price, size, side)` to trade buffer. If `size == 0`, aggressor fully consumed. |
| `'F'` (Fill) | Lookup order, subtract old size. If `mbo.size == 0`, remove order. Else update size, re-add. Do NOT append to trade buffer. |

## Batch Processing

Only update downstream state after processing a message with `F_LAST` set (`flags & 0x80`). This ensures atomic processing of multi-message venue updates.

## Snapshot Emission

- Fixed 100ms boundaries aligned to session clock (09:30:00.000, 09:30:00.100, ...)
- Extract top 10 price levels from each aggregation map
- Convert prices from fixed-point (`int64_t`, 1e-9 scale) to `float` at emission
- Zero-pad if fewer than 10 levels on either side: `(price=0.0, size=0.0)`
- Trade array: rolling buffer of last 50 trades, left-padded with zeros if < 50 trades
- `mid_price` = (best_bid + best_ask) / 2; carry forward if side is empty
- `spread` = best_ask - best_bid; carry forward if side is empty
- `time_of_day` = fractional hours since midnight ET

## Session Filtering

- Process ALL events (including pre-market) to maintain correct book state
- Only emit snapshots during RTH: 09:30:00.000 – 16:00:00.000 ET
- Warm up: process events from at least 09:29:00.000 before first emission
- If book empty at 09:30:00.000, skip forward until at least 1 bid and 1 ask exist; log skipped count

## Gap Handling

- If no events between two 100ms boundaries, carry forward last-known state
- If gap > 5 seconds during RTH with zero events, log WARNING

## Instrument Filtering

- Accept target `instrument_id` parameter
- Process only events matching that ID
- Ignore all other instruments

## Level Ordering Assertions

- `bids[0].price >= bids[1].price` (best bid first, guaranteed by `std::greater<>`)
- `asks[0].price <= asks[1].price` (best ask first, guaranteed by default ordering)
- Assert on first non-empty snapshot

## Dependencies

- `databento-cpp` (via CMake FetchContent)
- C++20
- GTest for tests

## File Layout

```
src/
  book_builder.hpp       # BookSnapshot struct + BookBuilder class declaration
  book_builder.cpp       # Implementation
tests/
  book_builder_test.cpp  # GTest unit tests
CMakeLists.txt           # Top-level build (FetchContent for databento-cpp, GTest)
```

## Test Cases

### Unit Tests (no real data dependency)

1. **Add orders build book** — Feed synthetic Add events, verify bid/ask aggregation maps are correct.
2. **Cancel removes order** — Add then cancel, verify level removed when size reaches zero.
3. **Modify updates price and size** — Add, then modify to new price. Old level removed, new level created.
4. **Trade appends to trade buffer** — Feed Trade events, verify buffer contents and ordering.
5. **Fill reduces passive order** — Add order, fill partially (size > 0), verify reduced size. Fill fully (size == 0), verify removal.
6. **Clear resets book** — Populate book, send Clear, verify empty.
7. **F_LAST batching** — Send multiple events without F_LAST, verify intermediate states not emitted. Set F_LAST on final event, verify state updated.
8. **Snapshot emission at 100ms boundaries** — Feed events with known timestamps, verify snapshots emitted at correct boundaries.
9. **Level padding** — Book with < 10 levels, verify zero-padding to 10 on each side.
10. **Trade buffer left-padding** — Fewer than 50 trades, verify zero-padding on left (oldest slots).
11. **Price conversion** — Verify fixed-point int64 prices convert correctly to float32.
12. **Mid price and spread** — Verify computation from best bid/ask. Verify carry-forward when a side is empty.
13. **Instrument filtering** — Feed events for multiple instruments, verify only target ID processed.
14. **Gap warning** — Simulate > 5s gap during RTH, verify warning logged and snapshot carried forward.
15. **Level ordering assertion** — Verify bids descending, asks ascending on first non-empty snapshot.
16. **Session filtering** — Feed events across pre-market and RTH, verify snapshots only during RTH window.
17. **Time of day** — Verify fractional hours computation for known timestamps.

### Integration Test (requires real data)

18. **Process single day file** — Read `glbx-mdp3-20220103.mbo.dbn.zst` for instrument_id `13615` (MESM2). Verify: non-empty snapshot vector, all snapshots within RTH, mid_price > 0, spread >= 0, bids[0] > asks[0] never (crossed book = error).
