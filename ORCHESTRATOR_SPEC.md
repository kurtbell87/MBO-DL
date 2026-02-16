# Kenoma Labs — Model Orchestrator Spec
## MES Microstructure Model Suite: Build, Overfit, Validate

**Version**: 0.6 (Draft)
**Date**: 2026-02-15
**Author**: Brandon Bell / Kenoma Labs LLC
**Changes from v0.5**: MBO replaces MBP-10 as source schema (data on disk is MBO), C++ for data pipeline and model inference (libtorch, xgboost C API), project restructured as CMake C++ project, paths updated to reference actual MES.FUT 2022 dataset. See Appendix D.

---

## 1. Purpose

Build four model architectures (MLP, SSM, CNN, GBT) that ingest MES limit order book data and output trading actions. Each model is trained to intentionally overfit on a small data slice (N=32-128 samples) to verify end-to-end correctness of: data pipeline, feature encoding, forward pass, loss computation, backward pass, and action decoding.

This is NOT about finding alpha. This is a correctness harness. If a model cannot memorize 32 samples to near-zero loss, something is broken.

---

## 2. Data Contract

### 2.1 Raw Input: Book Snapshot

Each sample is a time-indexed snapshot of the MES order book plus recent trade tape.

**Source schema**: Databento MBO (Market By Order / L3). This is the canonical schema for all raw data ingestion. The raw data contains per-order events (Add, Cancel, Modify, Trade, Fill, Clear) keyed by `order_id`. The book builder (§2.1.1) reconstructs aggregated price-level snapshots from these events. See `DATA/README.md` for the full MBO record schema, enums, and file format details.

```
BookSnapshot:
  timestamp:    int64       # exchange timestamp (nanoseconds since epoch)
  symbol:       str         # "MES" (fixed for now)
  bids:         float32[L, 2]  # L price levels × (price, size)
  asks:         float32[L, 2]  # L price levels × (price, size)
  trades:       float32[T, 3]  # last T trades × (price, size, aggressor_side)
  mid_price:    float32     # (best_bid + best_ask) / 2
  spread:       float32     # best_ask - best_bid (in ticks)
  time_of_day:  float32     # fractional hours since midnight ET (0.0 - 24.0)
```

**Parameters:**
- `L = 10` (top 10 price levels each side)
- `T = 50` (last 50 trades)
- Snapshot interval: 100ms (10 snapshots/second)
- MES tick size: 0.25 index points ($1.25 per tick per contract)

### 2.1.1 Book Builder Reconstruction Rules

The `book_builder` module (C++) reconstructs `BookSnapshot` objects at 100ms intervals from raw Databento MBO events. This is the most complex data engineering step in Phase 1 and must follow these rules precisely. Read MBO events from `.dbn.zst` files using `databento::DbnFileStore` (see `DATA/README.md` for the C++ API).

**Snapshot timing**: Snapshots are generated at fixed 100ms boundaries aligned to the trading session clock (e.g., 09:30:00.000, 09:30:00.100, 09:30:00.200, ...). Each snapshot captures the last-known state of the book at that boundary timestamp.

**Order book state**: Maintain a full order book per instrument:
- An order map: `std::unordered_map<uint64_t, Order>` keyed by `order_id`, where each `Order` stores `{price, size, side}`.
- Bid aggregation: `std::map<int64_t, uint32_t, std::greater<>>` — price level (descending) → total size.
- Ask aggregation: `std::map<int64_t, uint32_t>` — price level (ascending) → total size.

Process MBO events in `ts_event` order. For each `databento::MboMsg`:

| Action | Processing Rule |
|--------|----------------|
| `'R'` (Clear) | Clear the order map and both aggregation maps for this `instrument_id`. If `F_SNAPSHOT` flag is set, this begins a snapshot sequence — subsequent `'A'` events with `F_SNAPSHOT` populate the initial book state. |
| `'A'` (Add) | Insert `{order_id → (price, size, side)}` into the order map. Add `size` to the appropriate aggregation map at the order's price level. |
| `'C'` (Cancel) | Look up `order_id` in the order map. Subtract its `size` from the aggregation map at its price level (remove the level if size reaches zero). Remove the order from the order map. |
| `'M'` (Modify) | Look up `order_id`. Subtract old size from old price level in aggregation. Update the order's price and size. Add new size to new price level. The order loses queue priority. |
| `'T'` (Trade) | This is the aggressor side of a trade. Extract `(price, size, side)` and append to the trade buffer (see trade accumulation below). The aggressor order may or may not remain in the book — check the `size` field: if 0, the aggressor is fully consumed. |
| `'F'` (Fill) | This is the passive side of a trade. Look up `order_id` in the order map. Subtract old size from aggregation. If `mbo.size == 0`, the order is fully filled — remove it. Otherwise update to the new size and re-add to aggregation. Do NOT append to the trade buffer (the corresponding `'T'` event already captured this trade). |

**Batch processing with `F_LAST`**: Multiple MBO messages can arrive for the same `instrument_id` in one venue packet. Only emit/update the aggregated book state after processing a message with `F_LAST` set (`flags & 0x80`). This ensures atomic processing of multi-message venue updates.

**Snapshot emission at 100ms boundaries**: At each 100ms boundary, extract the top 10 price levels from each aggregation map:
- `bids[0..9]`: first 10 entries from the bid map (highest prices first, guaranteed by `std::greater<>` ordering).
- `asks[0..9]`: first 10 entries from the ask map (lowest prices first, guaranteed by default `std::map` ordering).
- Convert prices from fixed-point (`int64_t`, scale 1e-9) to `float32` at emission time: `float_price = static_cast<float>(price) / 1e9f`.

**Session filtering**: Only emit snapshots during Regular Trading Hours (RTH): 09:30:00.000 ET through 16:00:00.000 ET. Process all MBO events (including pre-market) to maintain correct book state, but only emit snapshots within the RTH window. If a different session window is needed (e.g., for globex overnight), this must be an explicit parameter, not the default.

**Gap handling**: If no MBO events arrive between two consecutive 100ms boundaries, emit a snapshot with the last-known book state (carry forward). This is the correct behavior — the book hasn't changed. If a gap exceeds 5 seconds with zero events during RTH, log a WARNING but do not drop the snapshot. Gaps >5s during RTH are rare for MES and may indicate a data feed issue; the warning lets the caller decide whether to trust the data window.

**Empty book at session start**: At 09:30:00.000, the book state may be partially populated if pre-market activity is thin. The builder must process events from at least 1 minute before session start (09:29:00.000) to warm up the book state, then begin emitting snapshots at 09:30:00.000. If the book is still empty at 09:30:00.000 (no bid or ask levels populated), skip forward until at least one bid and one ask level exist, and log the number of skipped snapshots. Note: Each daily `.dbn.zst` file begins with a snapshot sequence (Clear + Adds with `F_SNAPSHOT` flag) that populates the initial book state for the UTC day — this provides warm-start data even before RTH.

**Level ordering**: Bids are ordered best (highest price) first: `bids[0]` is the best bid. Asks are ordered best (lowest price) first: `asks[0]` is the best ask. This is guaranteed by the aggregation map ordering (`std::greater<>` for bids, default ascending for asks). Verify with an assertion on the first non-empty snapshot: `bids[0].price >= bids[1].price` and `asks[0].price <= asks[1].price`.

**Empty levels**: If fewer than 10 levels exist on either side, pad remaining levels with `(price=0.0, size=0.0)`. The feature encoder handles this — price delta from mid will be large and negative/positive for zero-price levels, and log1p(0) = 0 for size. This is an imperfect encoding of "no level" but acceptable for the overfit harness. A sentinel value (e.g., NaN) would propagate through the pipeline; zero is safer.

**Trade accumulation**: Maintain a rolling buffer of the last 50 trades. Append a trade entry on each `action='T'` (Trade) event — this is the aggressor side and captures the trade once. Do NOT append on `action='F'` (Fill) to avoid double-counting. Each trade entry stores `(price, size, aggressor_side)` where `aggressor_side` is the MBO `side` field (+1.0 for Bid/'B' = buy aggressor, -1.0 for Ask/'A' = sell aggressor). At each 100ms boundary, the snapshot's `trades` field is a copy of the current buffer. See §2.1.2 for padding when fewer than 50 trades have occurred.

**mid_price and spread**: Computed at snapshot emission time from the current best bid and best ask. If either side is empty (no levels), `mid_price` and `spread` are carried forward from the last valid computation. If no valid computation exists yet (very start of session), set `mid_price = 0.0` and `spread = 0.0` and log a WARNING — these snapshots will produce degenerate features and should be excluded from sampling.

**Instrument filtering**: The MBO data for `MES.FUT` contains multiple instruments (outrights and calendar spreads — see `DATA/README.md` symbology table). The book builder must accept a target `instrument_id` parameter and process only events matching that ID. For the overfit harness, use a single front-month outright (e.g., `MESM2` = instrument_id `13615` for Q1–Q2 2022, `MESU2` = `10039` for Q3). The `symbology.json` file in the data directory maps symbols to instrument IDs with date ranges.

### 2.1.2 Trade Array Padding

If fewer than `T = 50` trades have occurred since session start (or since the trade buffer was initialized), the trade array is **zero-padded from the left** (oldest slots):

```
Example: Only 12 trades have occurred.
trades[0:38] = (0.0, 0.0, 0.0)   # padding — 38 zero-filled slots
trades[38:50] = actual trades     # 12 real trades, most recent at index 49

Convention:
- price = 0.0 → price_delta from mid will be a large negative number (mid is positive)
- size = 0.0 → log1p(0) = 0.0, z-score will be negative (below window mean)
- aggressor_side = 0.0 → neither buy nor sell (model sees this as a distinct signal)
```

**Why left-pad**: Most recent trades are always at the end of the array regardless of how many real trades exist. The model sees a consistent layout: recent activity on the right, padding (if any) on the left. This avoids index-shifting bugs where the "most recent trade" changes position based on buffer fill level.

**Impact on GBT features**: Zero-padded trades have `size = 0` and `aggressor_side = 0`. The deduplication step (§2.7) will naturally handle these — zero-padded entries are identical across consecutive snapshots and collapse to a single entry (or zero entries if filtered). The `large_trade_flag` median calculation uses only deduplicated trades with `size > 0`. The `trade_arrival_rate_5s` counts only deduplicated trades with `size > 0`. If no real trades exist in a GBT feature computation window, all trade features default to 0.0.

**Impact on z-score normalization**: In early-session windows where most trade slots are zero-padded, the per-window z-score (§2.6) will have a bimodal distribution (cluster at zero for padding, cluster at real values). The epsilon floor prevents division by zero if all values are identical. This is a known imperfection acceptable for the overfit harness — the sampling strategy (§6.2) should avoid the first few minutes of the session where padding dominates.

### 2.2 Feature Encoding

The raw `BookSnapshot` is flattened into a fixed-width feature vector per timestep. This is the canonical encoding used by all models except GBT (which uses hand-crafted features per §5.4).

```
Feature vector layout (per snapshot):

Index range   | Field                | Shape       | Description
-----------   | -----                | -----       | -----------
[0:10]        | bid_prices_delta     | float32[L]  | (bid_price[i] - mid_price) / tick_size → ticks from mid
[10:20]       | bid_sizes_norm       | float32[L]  | log1p(bid_size[i]), z-scored within obs window
[20:30]       | ask_prices_delta     | float32[L]  | (ask_price[i] - mid_price) / tick_size → ticks from mid
[30:40]       | ask_sizes_norm       | float32[L]  | log1p(ask_size[i]), z-scored within obs window
[40:90]       | trade_prices_delta   | float32[T]  | (trade_price[j] - mid_price) / tick_size
[90:140]      | trade_sizes_norm     | float32[T]  | log1p(trade_size[j]), z-scored within obs window
[140:190]     | trade_aggressor      | float32[T]  | -1.0 (sell) / +1.0 (buy)
[190]         | spread_ticks         | float32[1]  | spread / tick_size
[191]         | time_sin             | float32[1]  | sin(2π × fractional_hour / 24)
[192]         | time_cos             | float32[1]  | cos(2π × fractional_hour / 24)
[193]         | position_state       | float32[1]  | -1.0 (short) / 0.0 (flat) / +1.0 (long)
```

**Computed constants:**
```
F = L + L + L + L + T + T + T + 1 + 1 + 1 + 1
  = 10+10+10+10+50+50+50+1+1+1+1
  = 194
```

**`feature_dim = 194`** — this is the `F` in all model input shapes `(B, W, F)`.

**Implementation note**: Export these index ranges as named constants in `feature_encoder.hpp` (e.g., `constexpr int BID_PRICE_BEGIN = 0; constexpr int BID_PRICE_END = 10;`, etc.) so that downstream consumers (especially CNN §5.3) can split the feature vector without hardcoding indices.

### 2.3 Observation Window

Each model receives a rolling window of `W` consecutive encoded snapshots.

- `W = 600` (60 seconds at 100ms resolution)

Shape per sample: `(W, 194)` → `(600, 194)`

### 2.4 Position State and Trajectory Generation

Position state is **part of the observation** (included in the feature vector as `position_state`). This is required because the oracle labeler (§4.1) generates position-dependent labels (e.g., EXIT is only labeled when in a position).

**Trajectory start index**: The trajectory begins at `t = W - 1` (i.e., `t = 599`), the first timestep for which a full observation window `[0 : W]` is available. All timesteps `t < W - 1` are consumed as history for the first window but do not generate labels or training samples. This guarantees that every observation window has exactly `W` snapshots and that all backward-looking indices in GBT feature computation (§5.4) are valid by construction.

**Trajectory stop index**: The trajectory ends at `t = len(snapshots) - horizon - 1` (inclusive), where `horizon` is the oracle's forward lookahead (default 100 snapshots = 10 seconds). This guarantees the oracle always has sufficient future data for its lookahead. The valid trajectory range is:

```
t_start = W - 1                           # = 599
t_stop  = len(snapshots) - horizon - 1     # inclusive
trajectory_length = t_stop - t_start + 1

Assert: t_stop >= t_start, i.e., len(snapshots) >= W + horizon
        (= 700 snapshots minimum = 70 seconds of data)
```

If `len(snapshots) < W + horizon`, the data window is too short to generate any trajectory. Raise an error — do not silently produce an empty trajectory.

**Implication for sampling**: Observation windows cannot be sampled independently. They must be generated sequentially because position state at timestep `t` depends on the oracle's action at timestep `t-1`. The `trajectory_builder` must simulate a trajectory and track entry price for in-position oracle queries:

```
1. Start at t = t_start with position_state = 0.0 (flat), entry_price = NaN
2. For each timestep t from t_start to t_stop (inclusive):
   a. Build observation window [t-W+1 : t+1]  (always exactly W snapshots)
   b. Inject current position_state into each snapshot's feature vector
   c. Query oracle for label at t (pass position_state AND entry_price)
   d. Update position_state and entry_price based on oracle's action:
      - If oracle returns ENTER LONG (1): position_state = +1, entry_price = mid_price[t]
      - If oracle returns ENTER SHORT (2): position_state = -1, entry_price = mid_price[t]
      - If oracle returns EXIT (3): position_state = 0, entry_price = NaN
      - If oracle returns HOLD (0): no change
      - REVERSE (4) is never generated (see §4.1)
   e. Store (window, label) pair
3. From the full trajectory, sample N_overfit windows
```

**entry_price tracking**: The `trajectory_builder` owns `entry_price` as trajectory-level state. The oracle labeler receives it as a parameter (see §4.1). This separation keeps the oracle stateless per call.

For the overfit test, we sample contiguous windows from this trajectory. Random shuffling of windows within the trajectory is fine for training batches — the position state is baked into the features.

### 2.5 Source

- **Development/overfit test**: Databento historical MES MBO data. Data is stored at `DATA/GLBX-20260207-L953CAPU5B/` as daily `.dbn.zst` files covering all of 2022 (312 files, ~49 GB). Pick any single trading day for the overfit window. Recommended: `glbx-mdp3-20220103.mbo.dbn.zst` (first full RTH trading day of 2022, MESM2 instrument_id `13615`).
- **Later**: Live Databento feed or Rithmic

### 2.6 Normalization

- **Prices**: Delta from current mid-price, expressed in ticks (divide by 0.25). Note: MBO prices arrive as fixed-point `int64_t` with scale 1e-9. Convert to `float` before computing deltas.
- **Sizes**: `log1p(size)`, then z-scored using mean/std computed within the current observation window (per-window normalization). **Epsilon floor**: use `std + 1e-8` as the denominator to prevent division by zero when all sizes in a window are identical (plausible in low-volume pre-open or partial windows at trajectory start).
- **Time of day**: Sinusoidal encoding: `sin(2π × h/24)`, `cos(2π × h/24)`
- **Aggressor side**: -1.0 (sell) / +1.0 (buy)
- **Position state**: Raw value (-1.0, 0.0, +1.0), no normalization needed

**Size normalization detail**: The z-score is computed per observation window, not globally. This means each window's size features have zero mean and unit variance within that window (subject to the epsilon floor). This prevents data leakage across time and handles varying volume regimes, but means the model cannot compare absolute size levels across windows. This is an intentional tradeoff for the correctness harness; revisit for full training.

### 2.7 Trade Tape Deduplication

With MBO data, each trade is represented as a pair of events: `action='T'` (aggressor) and `action='F'` (fill on the passive side). The book builder (§2.1.1) only appends to the trade buffer on `action='T'`, so the raw trade buffer already contains one entry per trade — no cross-event duplication.

However, consecutive `BookSnapshot` objects at 100ms intervals share the same rolling trade buffer. A trade that occurred 200ms ago appears in the current snapshot and the two preceding ones.

**For the feature vector (§2.2)**: No deduplication. Each snapshot's trade array is encoded as-is. The model sees 50 trades per timestep, and overlapping trades across timesteps within the observation window are the model's problem to learn. This is intentional — the raw feature vector preserves the snapshot-local view.

**For GBT hand-crafted features (§5.4)**: Trade-based features (`trade_imbalance_1s`, `trade_imbalance_5s`, `trade_arrival_rate_5s`, `large_trade_flag`) require deduplicated trades to avoid double-counting. The deduplication strategy:

```
To compute trade features over a range of snapshots [t_start : t_end]:
1. Collect all trade arrays from snapshots in the range
2. Each trade has a unique identity from the original MBO event:
   - Preferred: use (ts_event, order_id) from the 'T' event — globally unique
   - Fallback: use (timestamp, price, size, aggressor_side) 4-tuple
3. Deduplicate by dropping exact-match identifiers, keeping the first occurrence
4. Filter: discard entries where size == 0 (these are padding per §2.1.2)
5. Compute features on the deduplicated, filtered set

Implementation note: The book builder should store the MBO sequence number
or ts_event with each trade entry to enable efficient deduplication. If
storing only (price, size, aggressor_side), the 4-tuple fallback works but
has a small collision risk for same-price, same-size trades within the same
nanosecond. This is negligible for the overfit harness.
```

The `gbt_features` module must implement this deduplication. The `feature_encoder` module does NOT deduplicate.

---

## 3. Action Space

Discrete action space with 5 actions:

| Action | ID | Meaning | Valid When | Position Effect |
|--------|----|---------|------------|----------------|
| Hold | 0 | No-op | Always | No change |
| Enter Long | 1 | Buy 1 | Flat only | Flat → Long |
| Enter Short | 2 | Sell 1 | Flat only | Flat → Short |
| Exit | 3 | Close position | In position | Long/Short → Flat |
| Reverse | 4 | Flip position | In position | Long → Short or Short → Long |

**Constraints:**
- Max position: ±1 contract
- Actions 1/2 only valid when `position_state == 0`
- Actions 3/4 only valid when `position_state != 0`

### 3.1 Action Masking Semantics

Invalid actions map to Hold (action 0) at execution time. For training and evaluation:

- **Loss computation**: The loss is computed on the **raw oracle label** against the **model's logits for all 5 actions**. No masking in the loss. The model must learn which actions are valid given the position state feature.
- **Accuracy computation**: A prediction is **correct** if `argmax(logits) == oracle_label`. If the model predicts an invalid action that happens to get masked to Hold, and the oracle label is also Hold, this counts as **incorrect** — the model got the right outcome by accident, but predicted the wrong action. The model should learn to predict Hold directly when Hold is the correct action.
- **Rationale**: This keeps the overfit test clean. The model has position state in its input and should learn the validity constraints implicitly. Explicit masking of logits before softmax is deferred to the RL spec.

---

## 4. Reward / Label

### 4.1 For Supervised Overfit Test

Ground truth labels are generated by a **stateless** oracle with future knowledge. The oracle is called per-timestep by `trajectory_builder`, which manages position state and entry price externally (see §2.4).

```cpp
// C++ signature — the oracle is a pure function, no mutable state.
int oracle_label(
    const std::vector<BookSnapshot>& snapshots,
    int t,
    int position_state,       // -1, 0, +1 (managed by trajectory_builder)
    float entry_price,        // mid_price at entry; NaN if flat (managed by trajectory_builder)
    int horizon = 100,        // 10 seconds forward
    int target_ticks = 10,    // 2.50 points (10 × 0.25)
    int stop_ticks = 5,       // 1.25 points
    int take_profit_ticks = 20, // 5.00 points (20 × 0.25) — exit with profit
    float tick_size = 0.25f
);

// Returns action label for timestep t given current position.
//
// Preconditions:
//     - If position_state == 0: entry_price must be NaN
//     - If position_state != 0: entry_price must be a valid float
//     - t + horizon <= snapshots.size() (caller must ensure lookahead is available)
//
// If flat (position_state == 0):
//     assert(std::isnan(entry_price));
//     Look forward `horizon` snapshots from t.
//     Track mid_price movement in ticks from mid_price[t].
//     If price hits +target_ticks before -stop_ticks → return ENTER_LONG (1)
//     If price hits -target_ticks before +stop_ticks → return ENTER_SHORT (2)
//     If neither threshold hit within horizon → return HOLD (0)
//
// If in position (position_state != 0):
//     assert(!std::isnan(entry_price));
//     direction = position_state  // +1 for long, -1 for short
//     current_pnl_ticks = direction * (mid_price[t] - entry_price) / tick_size
//     Look forward `horizon` snapshots:
//         future_pnl_ticks = direction * (mid_price[t+k] - entry_price) / tick_size
//         If future_pnl_ticks >= take_profit_ticks → return EXIT (3) — take profit
//         If future_pnl_ticks <= -stop_ticks → return EXIT (3) — stop loss
//     If neither hit:
//         return HOLD (0) — hold position
//
// Post-condition assert:
//     assert(position_state == 0 || (label == 0 || label == 3));
//     // Oracle never returns ENTER LONG (1) or ENTER SHORT (2) when in position.
//
// Note: REVERSE (4) is never generated by this oracle. It exists in the
// action space for RL but is unused in supervised training. The oracle
// always exits before re-entering. This means the overfit test validates
// only 4 of 5 action classes. This is acceptable — Reverse is a
// composition of Exit + Enter and tests no new pipeline logic.
```

**In-position exit logic**: The oracle generates EXIT on both adverse moves (stop loss at `-stop_ticks`) and large favorable moves (take profit at `take_profit_ticks`). Between stop and take-profit, the oracle labels HOLD regardless of current PnL. The `take_profit_ticks` threshold is set at 2× `target_ticks` by default — the entry threshold looks for 10-tick moves, and the exit takes profit at 20 ticks. This prevents degenerate label distributions where EXIT only correlates with losing trades. Adjust `take_profit_ticks` relative to `target_ticks` if EXIT label frequency is too low or too high in the subsample.

**entry_price contract**: The oracle does not track or mutate `entry_price`. It receives the value and uses it read-only. The `trajectory_builder` (§2.4) is responsible for setting `entry_price = mid_price[t]` when the oracle returns ENTER LONG or ENTER SHORT, and clearing it to `NaN` when the oracle returns EXIT.

**Position-action consistency guard**: The oracle must assert at exit that it never returns an entry action (1 or 2) when `position_state != 0`. This is structurally guaranteed by the branch logic but the assert catches implementation bugs where the branches are incorrectly structured. Similarly, the oracle never returns EXIT (3) when flat — assert `position_state == 0 or label != 3` is redundant with the branch structure but serves as a safety net.

**Label distribution check**: After **subsampling** N_overfit windows from the trajectory (not on the full trajectory), assert:
- At least 3 of 4 used classes (0, 1, 2, 3) are present in the subsampled set
- No single class exceeds 80% of labels in the subsampled set
- REVERSE (class 4) is never present (assert `4 not in labels`)
- If HOLD dominates >70% of the subsampled labels, consider tightening `target_ticks` or widening `horizon` for the overfit test data window specifically

**Why check on the subsample**: A full 30-minute trajectory may have reasonable class distribution overall, but evenly-spaced subsampling of 32 windows can miss rare classes (e.g., EXIT at ~8% → ~2.5 expected in 32 samples). The check must validate what the model actually trains on.

### 4.2 For RL (Future, Not This Spec)

Reward = realized PnL per timestep minus transaction costs. Defer to separate spec.

---

## 5. Model Architectures

All four neural models share the same input/output interface:

```cpp
// C++ protocol (libtorch)
class TradingModel : public torch::nn::Module {
public:
    // x: (batch, 600, 194) observation window
    // returns: (batch, 5) action logits
    virtual torch::Tensor forward(torch::Tensor x) = 0;
};
```

### 5.1 MLP (Baseline)

Flatten the observation window and pass through dense layers.

```
Input: (B, 600, 194) → Flatten → (B, 116400)
→ Linear(116400, 512) → ReLU → Dropout(0.1)
→ Linear(512, 256) → ReLU → Dropout(0.1)
→ Linear(256, 128) → ReLU
→ Linear(128, 5) → logits
```

**Purpose**: Simplest possible model. If this can't overfit, the data pipeline is broken.

**Parameter count**: ~59.7M (dominated by first linear layer: 116400 × 512 ≈ 59.6M)

**Note**: For the overfit test with N=32, this model has ~1.9M parameters per sample. It *must* memorize. If it doesn't, something is wrong with the pipeline, not the model.

**Convergence note**: Despite extreme overparameterization, the first linear layer's gradient signal is diluted across 116,400 input dimensions. The MLP may take 200-400 epochs to reach 99% on N=32 — significantly more than CNN or GBT. This is expected behavior, not a failure signal. If the MLP exceeds 400 epochs on N=32 without reaching 95%, *then* investigate (check §6.5).

### 5.2 SSM (State Space Model)

Mamba-style selective state space over the temporal sequence.

```
Input: (B, 600, 194)
→ Linear(194, D_model) → projection to model dim
→ MambaBlock(D_model) × N_layers
→ Take last hidden state → (B, D_model)
→ Linear(D_model, 5) → logits
```

**Hyperparameters:**
- `D_model = 128`
- `N_layers = 4`
- `D_state = 16` (SSM state dimension)
- `D_conv = 4` (local convolution width)

**Purpose**: Learns temporal dynamics of book evolution. Should capture sequential patterns like sweep-then-reload, quote stuffing rhythms, momentum sequences.

**Implementation**: Use `mamba-ssm~=2.2` (pin minor version; the block API changed between 1.x and 2.x). **This requires CUDA and Python.** There is no C++ or CPU fallback in scope for this spec. The overfit test for SSM runs via a thin Python script (`python/ssm_training.py`) that reads preprocessed tensor data from disk (written by the C++ pipeline). If no GPU is available, skip SSM — the other three models provide sufficient pipeline validation. Do not attempt to implement S4 as a fallback; it's a different architecture with different behavior, and verifying it adds scope without validating the same code path.

**Rationale for Python-only**: Mamba's selective scan has fused CUDA kernels that don't have C++ equivalents outside the Python package. Running a different implementation would not validate the same forward pass, defeating the purpose of the correctness harness.

### 5.3 CNN (Convolutional)

Treat the order book as a structured spatial signal per timestep, then convolve temporally.

**Book layout (per timestep)**: The encoded feature vector (§2.2) stores bid and ask levels separately. The CNN reconstructs a 20-level "price ladder" ordered from deepest bid to deepest ask:

```
Price ladder (20 levels):
  [bid[9], bid[8], ..., bid[1], bid[0], ask[0], ask[1], ..., ask[8], ask[9]]

Each level has 2 channels: (price_delta, size_norm)
→ Per-timestep spatial tensor: (20, 2) = 40 features from book
```

**Feature split indices** (referencing §2.2 layout):

```cpp
// These constants are exported from feature_encoder.hpp
constexpr int BID_PRICE_BEGIN = 0,   BID_PRICE_END = 10;    // bid_prices_delta
constexpr int BID_SIZE_BEGIN  = 10,  BID_SIZE_END  = 20;    // bid_sizes_norm
constexpr int ASK_PRICE_BEGIN = 20,  ASK_PRICE_END = 30;    // ask_prices_delta
constexpr int ASK_SIZE_BEGIN  = 30,  ASK_SIZE_END  = 40;    // ask_sizes_norm
constexpr int TRADE_BEGIN     = 40,  TRADE_END     = 190;   // all trade features (150 values)
constexpr int SCALAR_BEGIN    = 190, SCALAR_END    = 194;   // spread, time_sin, time_cos, position_state

// Price ladder construction (per timestep):
// 1. Extract bid_prices[0:10] and bid_sizes[0:10] → pairs: [(price[0], size[0]), ..., (price[9], size[9])]
// 2. Reverse bid order: [(price[9], size[9]), ..., (price[0], size[0])]
// 3. Extract ask_prices[0:10] and ask_sizes[0:10] → pairs: [(price[0], size[0]), ..., (price[9], size[9])]
// 4. Concatenate: reversed_bids + asks → (20, 2) tensor
//
// This places the deepest bid at index 0, best bid at index 9,
// best ask at index 10, deepest ask at index 19.
```

**Architecture:**

```
Input: (B, 600, 194)

Step 1 — Split features per timestep using index constants:
  book_spatial: (B, 600, 20, 2) — price ladder with (price_delta, size_norm),
                                   constructed via reindex above
  trade_features: (B, 600, 150) — features[40:190] (T×3 trade features)
  scalar_features: (B, 600, 4) — features[190:194] (spread, time_sin, time_cos, position_state)

Step 2 — Spatial convolution (per timestep, shared weights):
  Reshape book_spatial → (B*600, 2, 20) — treat 2 features as channels over 20 levels
  → Conv1d(in=2, out=32, kernel=3, padding=1) → ReLU
  → Conv1d(in=32, out=64, kernel=3, padding=1) → ReLU
  → AdaptiveAvgPool1d(1) → (B*600, 64)
  → Reshape → (B, 600, 64)

Step 3 — Concatenate all per-timestep features:
  → cat(spatial_out, trade_features, scalar_features) → (B, 600, 64+150+4) = (B, 600, 218)

Step 4 — Temporal convolution:
  → Permute → (B, 218, 600) — channels-first for Conv1d
  → Conv1d(in=218, out=128, kernel=5, padding=2) → ReLU
  → Conv1d(in=128, out=256, kernel=5, padding=2) → ReLU
  → AdaptiveAvgPool1d(1) → (B, 256)

Step 5 — Classification:
  → Linear(256, 5) → logits
```

**Purpose**: Learns spatial structure in the book (bid/ask shape, size clustering at levels) via the spatial path, and temporal patterns via the temporal path. The price ladder ordering preserves the spatial relationship between bid and ask sides — the model can learn patterns like "thin best bid relative to deep asks" as a local spatial feature.

### 5.4 GBT (Gradient Boosted Trees)

Not a neural network. Uses hand-crafted features from the observation window.

**Implementation**: XGBoost multiclass classifier (5 classes). Use XGBoost C API only — not LightGBM, not the Python wrapper. One library, one code path. The C++ wrapper should use `XGBoosterCreate`, `XGBoosterSetParam`, `XGBoosterUpdateOneIter`, `XGBoosterPredict`.

**Objective**: Use `multi:softmax` (returns class indices). This is simpler for the overfit harness where we only need predicted classes, not probabilities. If probability calibration or logit-level comparison with neural models is needed later, switch to `multi:softprob` (returns probability vectors of shape `(N, 5)`) and take `argmax` for accuracy. The choice does not affect tree construction — only the output format. Document which objective is active in the GBT wrapper's constructor.

```
Features per sample (all computed from the observation window ending at timestep t,
where t = W - 1, i.e., the last index of the window):

IMPORTANT: t is always the last index of the observation window. Assert
t == W - 1 at the start of feature computation. All backward-looking
indices (t-10, t-50, t-300, t-599) are valid by construction since the
trajectory starts at t = W-1 (see §2.4) and the window always contains
exactly W snapshots.

Book features:
  1. book_imbalance:
     (sum(bid_sizes[:L]) - sum(ask_sizes[:L])) / (sum(bid_sizes[:L]) + sum(ask_sizes[:L]) + 1e-8)
     Range: [-1, 1]. Uses raw sizes (not log-transformed).
     Epsilon guard prevents division by zero if both sides are empty.

  2. spread_ticks:
     Current spread / tick_size. Dimensionless.

  3. book_depth_ratio_5:
     sum(bid_sizes[0:5]) / (sum(ask_sizes[0:5]) + 1e-8)
     Ratio of top-5-level depth. Uses raw sizes.
     Epsilon guard prevents division by zero if ask side has no size at top 5 levels.

  4. top_level_size_ratio:
     bid_size[0] / (ask_size[0] + 1e-8)
     BBO size ratio. Uses raw sizes.
     Epsilon guard prevents division by zero if best ask has zero size.

Price dynamics (all computed from mid_price series within the window):
  5. mid_return_1s:
     (mid_price[t] - mid_price[t-10]) / tick_size
     1-second return in ticks. Uses snapshot at index t vs t-10.

  6. mid_return_5s:
     (mid_price[t] - mid_price[t-50]) / tick_size
     5-second return in ticks.

  7. mid_return_30s:
     (mid_price[t] - mid_price[t-300]) / tick_size
     30-second return in ticks.

  8. mid_return_60s:
     (mid_price[t] - mid_price[t-599]) / tick_size
     Full-window return in ticks. Uses first snapshot in window.

Trade features (all use DEDUPLICATED trades per §2.7):
  9. trade_imbalance_1s:
     sum(trade_size × aggressor_side) for deduplicated trades in snapshots [t-10 : t+1]
     Net signed volume. Positive = net buying.
     If no real trades (all padding), return 0.0.

  10. trade_imbalance_5s:
      Same as above for deduplicated trades in snapshots [t-50 : t+1].
      If no real trades, return 0.0.

  11. trade_arrival_rate_5s:
      count(deduplicated trades with size > 0 in [t-50 : t+1]) / 5.0
      Unique real trades per second over last 5 seconds.
      If no real trades, return 0.0.

  12. large_trade_flag:
      1.0 if any deduplicated trade in [t-10 : t+1] has size > 2 × median(all
      deduplicated trade sizes with size > 0 in full window).
      0.0 otherwise. If no real trades in window, set to 0.0.

VWAP features:
  13. vwap_distance:
      (mid_price[t] - VWAP) / tick_size
      Where VWAP = sum(trade_price × trade_size) / sum(trade_size) over entire
      window, computed on DEDUPLICATED trades with size > 0.
      If no real trades in window, set to 0.0.

Time features:
  14. time_sin: sin(2π × fractional_hour / 24)
  15. time_cos: cos(2π × fractional_hour / 24)

Position:
  16. position_state: -1.0, 0.0, or +1.0

Total: 16 features per sample.
```

**Purpose**: Interpretable baseline. If GBT with these features can overfit, the labels make sense. If it can't, the labeling oracle may be broken or the features are losing critical information. Feature importance rankings inform what the neural models should be learning.

---

## 6. Overfit Validation Protocol

### 6.1 Reproducibility

All overfit tests must be deterministic. Before any training run:

```cpp
// C++ (libtorch)
#include <torch/torch.h>
#include <cstdlib>

constexpr int SEED = 42;

torch::manual_seed(SEED);
if (torch::cuda::is_available()) {
    torch::cuda::manual_seed_all(SEED);
}
// Set via environment before process start:
// CUBLAS_WORKSPACE_CONFIG=:4096:8
// For full determinism, also set at::globalContext().setDeterministicCuDNN(true);
std::srand(SEED);
```

```python
# Python (for SSM training only)
import torch
import numpy as np
import random

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

XGBoost: set `random_state=SEED` via `XGBoosterSetParam(booster, "seed", "42")`.

Two runs with the same seed, same data, and same code must produce identical loss curves. If they don't, there's a non-determinism bug (likely in data loading or CUDA ops).

### 6.2 Data Subset

Select a contiguous 30-minute window from a single trading day. The dataset contains full-year 2022 MES.FUT MBO data. Recommended starting point: `glbx-mdp3-20220103.mbo.dbn.zst` (first full RTH trading day of 2022), instrument_id `13615` (MESM2), window 09:30-10:00 ET. This yields ~18,000 snapshots. Generate the full oracle trajectory per §2.4, then sample `N_overfit` observation windows:

- **N_overfit = 32** (tiny — must achieve ~100% accuracy)
- **N_overfit = 128** (small — must achieve >95% accuracy)

**Trajectory length guard**: After generating the trajectory, assert `trajectory_length >= N_overfit`. If the trajectory is shorter than N_overfit (e.g., due to a short data window, aggressive session filtering, or many skipped snapshots from book builder warmup), raise an error with the actual trajectory length. Do not silently produce fewer samples.

**Sampling method**: Take every `k`th window from the trajectory, where `k = floor(trajectory_length / N_overfit)`. This gives evenly spaced samples across the 30-minute period, ensuring diversity of position states and market conditions. Do not cluster samples.

**Label distribution validation**: Run the distribution checks from §4.1 on the **subsampled** N_overfit windows, not the full trajectory. If the subsample fails distribution checks, shift the sampling offset by `k/2` and retry. If it still fails, choose a different 30-minute window with more activity.

**Sampling window advice**: Avoid the first 60 seconds of RTH (09:30:00–09:31:00) as the primary sampling region. Early-session snapshots have partially padded trade arrays (§2.1.2) and potentially thin books from session warmup. The 30-minute data window should start at 09:30:00 (to provide history for observation windows), but evenly-spaced sampling from a 30-minute trajectory will naturally spread samples across the full window, diluting early-session artifacts.

Use the same N samples for both training AND evaluation. The goal is memorization.

**Evaluation at N=128**: When `N_overfit = 128` and batch size is 32 (4 batches per epoch), accuracy is computed over **all 128 samples** at the end of each epoch, not per-batch. Run a full inference pass over the entire N=128 set after each training epoch to compute the reported accuracy. This ensures accuracy is measured consistently regardless of batch count.

### 6.3 Success Criteria

For each model, the overfit test PASSES if:

| Metric | N=32 Target | N=128 Target | Max Epochs |
|--------|-------------|--------------|------------|
| **MLP** | Train acc ≥ 99% | Train acc ≥ 95% | 500 |
| **SSM** | Train acc ≥ 99% | Train acc ≥ 95% | 500 |
| **CNN** | Train acc ≥ 99% | Train acc ≥ 95% | 500 |
| **GBT** | Train acc ≥ 99% | Train acc ≥ 95% | 1000 rounds |

Additionally, for all neural models:
- Loss must monotonically decrease (on average over 10-epoch windows) for first 100 epochs
- No NaN/Inf in loss or gradients at any point
- Forward pass on a single batch of 32 samples completes in < 100ms on target hardware
- Model can save/load checkpoint and produce bitwise identical outputs on the same input

For GBT:
- No NaN/Inf in predictions
- Model can save/load and produce identical predictions
- Prediction on 128 samples completes in < 10ms

**Collapse detection**: If any model's accuracy plateaus at or near the majority class frequency (e.g., ~45% if HOLD is 45% of labels) for >50 epochs, this is class collapse — the model is predicting the most common class for all inputs. This is a failure to learn, not a capacity issue. See §6.5 item 6.

### 6.4 Training Hyperparameters (Fixed for Overfit Test)

These are not tuned. They're set conservatively to make memorization easy.

**Neural models (MLP, SSM, CNN):**
- Optimizer: Adam
- Learning rate: 1e-3
- Weight decay: 0 (no regularization — we want overfitting)
- Batch size: 32 (= full N_overfit for N=32; for N=128, use 32)
- Loss: CrossEntropyLoss (no class weights)
- Gradient clipping: max_norm=1.0
- Dropout: 0.0 during overfit test (override model defaults)

**GBT (XGBoost):**
- `objective`: `multi:softmax`
- `num_class`: 5
- `max_depth`: 10
- `learning_rate`: 0.1
- `n_estimators`: 1000
- `subsample`: 1.0 (no bagging — we want overfitting)
- `colsample_bytree`: 1.0
- `min_child_weight`: 1
- `random_state`: 42

### 6.5 Failure Diagnosis

If a model fails to overfit:

1. **Loss doesn't decrease at all** → Learning rate wrong, loss function wrong, labels are constant, or gradient is not flowing (check `requires_grad` / parameter registration on all parameters)
2. **Loss decreases then plateaus well above zero** → Model capacity too low, or data encoding is lossy. Check that `feature_dim` matches expected 194. Check for constant features.
3. **Loss oscillates wildly** → Learning rate too high. Try 1e-4. If still oscillating, check normalization — z-score denominators near zero cause explosions. Verify the epsilon floor (1e-8) is applied in the z-score denominator.
4. **NaN after N epochs** → Gradient explosion. Check for division by zero in normalization (verify epsilon floor is active). Verify gradient clipping is active. Print gradient norms per layer.
5. **Accuracy stuck at ~20% (random for 5 classes)** → Model output isn't connected to loss. Check that logits shape matches labels shape. Check that labels are `int64` / `torch::kLong` in [0,4], not one-hot.
6. **Accuracy stuck at majority class %** → Model collapsed to predicting the most common class. Check label distribution in the subsampled set. If HOLD > 60%, the oracle parameters need adjustment for this data window, or resample with a different offset.
7. **GBT fails but neural models pass** → Feature engineering bug. Check for NaN in engineered features (especially VWAP when no trades exist). Verify epsilon guards are active in `book_depth_ratio_5` and `top_level_size_ratio`. Print feature distributions.
8. **Neural models fail but GBT passes** → Likely a tensor shape or encoding bug in the neural pipeline. The raw data is fine but the tensor encoding is lossy or corrupted.
9. **MLP takes >300 epochs on N=32** → Likely normal (see §5.1 convergence note). Only investigate if it exceeds 400 epochs without reaching 95%.

---

## 7. Infrastructure

### 7.1 Dependencies

**C++ (primary — data pipeline, models, inference):**
```
databento-cpp          # MBO .dbn.zst file reading (DbnFileStore, MboMsg)
libtorch >= 2.1        # PyTorch C++ frontend (MLP, CNN model definitions + training)
xgboost >= 2.0         # GBT via C API (XGBoosterCreate, etc.)
nlohmann_json          # JSON parsing (comes with databento-cpp)
zstd                   # Zstandard decompression (comes with databento-cpp)
Catch2 >= 3.0          # Unit testing framework
```

**Python (SSM only — Mamba requires CUDA Python):**
```
torch >= 2.1
mamba-ssm ~= 2.2       # GPU only; skip if no CUDA. Pin minor version.
numpy
```

**Build system**: CMake >= 3.18. Use FetchContent for databento-cpp, Catch2. libtorch is expected as a pre-installed package (download from pytorch.org or install via conda). XGBoost is built from source via FetchContent or found as a system package.

### 7.2 Hardware Requirements

- **Overfit test (MLP, CNN, GBT)**: CPU is fine. GPU speeds it up but not required.
- **Overfit test (SSM)**: GPU required (mamba-ssm needs CUDA). Runs via Python script.
- **Full training (future)**: Single GPU minimum (RTX 3090+ or A10G on AWS via Kenoma Labs account)
- **Inference (live, future)**: CPU. Models must export to ONNX or run via libtorch.

### 7.3 Project Structure

```
kenoma-models/
├── CMakeLists.txt                     # Top-level CMake build
├── include/
│   └── kenoma/
│       ├── data/
│       │   ├── book_builder.hpp       # MBO → L2 book snapshots at 100ms (§2.1.1)
│       │   │                          # Full order book from MBO events, aggregation to price levels,
│       │   │                          # session filtering, warmup, gap handling, trade buffer,
│       │   │                          # instrument filtering, F_LAST batch processing
│       │   ├── feature_encoder.hpp    # BookSnapshot → flat feature vector (F=194)
│       │   │                          # Exports: FEATURE_DIM, all index range constants
│       │   ├── normalizer.hpp         # Delta-price, log-size, z-score (with epsilon), sinusoidal
│       │   ├── oracle_labeler.hpp     # Stateless oracle (receives position_state + entry_price)
│       │   │                          # Includes position-action consistency assert (§4.1)
│       │   ├── trajectory_builder.hpp # Full trajectory generation; owns position_state + entry_price
│       │   │                          # Uses explicit t_start/t_stop bounds (§2.4)
│       │   ├── snapshot_sampler.hpp   # Sample N_overfit windows from trajectory
│       │   │                          # Includes trajectory length guard (§6.2)
│       │   └── gbt_features.hpp       # Hand-crafted GBT feature extraction (16 features)
│       │                              # Includes trade tape deduplication (§2.7)
│       │                              # Filters zero-size padding trades (§2.1.2)
│       ├── models/
│       │   ├── base.hpp               # TradingModel interface, FEATURE_DIM=194, WINDOW_SIZE=600,
│       │   │                          # NUM_ACTIONS=5, count_parameters() utility
│       │   ├── mlp.hpp
│       │   ├── cnn.hpp
│       │   └── gbt.hpp               # XGBoost C API wrapper
│       └── training/
│           ├── overfit_test.hpp       # Overfit validation harness
│           └── config.hpp             # Hyperparameters, SEED, constants
├── src/
│   ├── data/
│   │   ├── book_builder.cpp
│   │   ├── feature_encoder.cpp
│   │   ├── normalizer.cpp
│   │   ├── oracle_labeler.cpp
│   │   ├── trajectory_builder.cpp
│   │   ├── snapshot_sampler.cpp
│   │   └── gbt_features.cpp
│   ├── models/
│   │   ├── mlp.cpp
│   │   ├── cnn.cpp
│   │   └── gbt.cpp
│   └── training/
│       ├── overfit_test.cpp
│       └── run_overfit_suite.cpp      # Main entry point — runs all models, prints summary
├── tests/
│   ├── test_book_builder.cpp          # MBO reconstruction: order tracking, level aggregation,
│   │                                  # F_LAST batching, gap handling, warmup, trade buffer,
│   │                                  # instrument filtering, snapshot sequence handling
│   ├── test_feature_encoder.cpp       # Assert feature_dim=194, no NaN, correct ranges, index coverage
│   ├── test_models.cpp                # Forward pass shape tests for all models
│   ├── test_oracle.cpp                # Oracle labeler: class distribution, position state consistency,
│   │                                  # entry_price contract, REVERSE never generated,
│   │                                  # position-action consistency assert fires on bad input
│   ├── test_gbt_features.cpp          # GBT feature extraction: no NaN, expected ranges, t==W-1 assert,
│   │                                  # trade deduplication correctness, epsilon guards active,
│   │                                  # zero-size trade filtering, empty-trade-window defaults
│   └── test_overfit.cpp               # Automated overfit validation
├── python/
│   └── ssm_training.py                # SSM (Mamba) training — reads preprocessed tensors from disk,
│                                      # trains via mamba-ssm, writes results JSON
├── scripts/
│   └── download_data.py               # Fetch MES MBO data from Databento (Python — API client)
├── data/                              # Symlink or path to DATA/GLBX-20260207-L953CAPU5B/
│   ├── raw/                           # → .dbn.zst files
│   ├── processed/                     # Preprocessed tensor data (written by C++ pipeline)
│   └── overfit/                       # Small trajectory slices for overfit tests
└── README.md
```

### 7.4 Overfit Test Runner

```bash
# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Run all four models through overfit validation (MLP, CNN, GBT in C++)
./build/run_overfit_suite \
    --data ../DATA/GLBX-20260207-L953CAPU5B/glbx-mdp3-20220103.mbo.dbn.zst \
    --instrument-id 13615 \
    --n-samples 32 \
    --max-epochs 500 \
    --seed 42

# SSM (Python, requires CUDA)
python python/ssm_training.py \
    --tensors data/overfit/mes_20220103_0930_1000.pt \
    --n-samples 32 \
    --max-epochs 500 \
    --seed 42

# Expected output:
# Seed: 42
# Data: glbx-mdp3-20220103.mbo.dbn.zst (instrument_id=13615, MESM2)
# Trajectory length: 17301 (from t=599 to t=17899)
# Subsample label distribution: {HOLD: 41%, LONG: 22%, SHORT: 22%, EXIT: 15%, REVERSE: 0%}
# REVERSE class assertion: PASS (0 occurrences)
#
# ┌─────────┬──────────┬───────────┬────────────┬────────┐
# │ Model   │ N=32 Acc │ N=128 Acc │ Epochs/Rds │ Status │
# ├─────────┼──────────┼───────────┼────────────┼────────┤
# │ MLP     │ 100.0%   │ 97.2%     │ 342        │ PASS   │
# │ SSM     │ 100.0%   │ 96.1%     │ 287        │ PASS   │
# │ CNN     │ 100.0%   │ 95.8%     │ 203        │ PASS   │
# │ GBT     │ 100.0%   │ 98.4%     │ 51         │ PASS   │
# └─────────┴──────────┴───────────┴────────────┴────────┘

# If no GPU available (skip SSM):
./build/run_overfit_suite \
    --data ../DATA/GLBX-20260207-L953CAPU5B/glbx-mdp3-20220103.mbo.dbn.zst \
    --instrument-id 13615 \
    --n-samples 32 \
    --max-epochs 500 \
    --seed 42 \
    --skip-ssm
```

---

## 8. Orchestrator Agent Instructions

The orchestrator agent builds this system in phases. Each phase has explicit validation gates. Do not proceed to the next phase until the current phase passes all assertions.

**Language**: All data pipeline and model code is C++ unless stated otherwise. Use CMake as the build system. Use `databento-cpp` for reading MBO data, `libtorch` for neural models, XGBoost C API for GBT. The only Python code is `ssm_training.py` (Mamba requires Python) and `download_data.py` (Databento Python API for downloads).

**Data location**: MBO data is at `DATA/GLBX-20260207-L953CAPU5B/`. Daily files are named `glbx-mdp3-{YYYYMMDD}.mbo.dbn.zst`. See `DATA/README.md` for the complete schema reference, C++ reading examples, and symbology table.

### Phase 1: Data Pipeline (C++)

1. Write `CMakeLists.txt` — top-level CMake build. FetchContent for `databento-cpp` and `Catch2`. Find `libtorch` and `xgboost` as pre-installed packages. Set C++17 standard.
2. Write `book_builder.hpp/cpp` — reconstruct L2 book snapshots at 100ms intervals from raw MBO events per §2.1.1. Read `.dbn.zst` files using `databento::DbnFileStore`. For each `databento::MboMsg`, update the full order book (order map + bid/ask aggregation maps). At 100ms boundaries, emit the top 10 bid/ask levels as a `BookSnapshot`. Implement: instrument filtering by `instrument_id`, `F_LAST` batch processing, snapshot sequence handling (Clear + Adds with `F_SNAPSHOT`), session filtering (RTH only by default), 1-minute warmup before session start, gap handling with carry-forward and 5s warning, empty level padding with `(0.0, 0.0)`, trade buffer (append on `action='T'` only) with left-padding per §2.1.2, mid_price/spread carry-forward when book side is empty, level ordering assertions.
3. Write `normalizer.hpp/cpp` — implement delta-price (in ticks, converting MBO fixed-point prices via `/ 1e9`), log1p + per-window z-score for sizes (with epsilon floor of 1e-8 in denominator), sinusoidal time encoding.
4. Write `feature_encoder.hpp/cpp` — flatten `BookSnapshot` → feature vector of length 194 per §2.2. This is the single source of truth for `F`. Export `constexpr int FEATURE_DIM = 194` and all index range constants (`BID_PRICE_BEGIN/END`, `BID_SIZE_BEGIN/END`, `ASK_PRICE_BEGIN/END`, `ASK_SIZE_BEGIN/END`, `TRADE_BEGIN/END`, `SCALAR_BEGIN/END`) for use by CNN and GBT.
5. Write `oracle_labeler.hpp/cpp` — implement the stateless oracle from §4.1, including take-profit exit logic and position-action consistency assert. Input: snapshots, timestep, position_state, entry_price. Output: action label (int). The oracle must not maintain or mutate any state — it receives position_state and entry_price as read-only parameters. Assert at exit: if `position_state != 0`, label must be in `{0, 3}`.
6. Write `trajectory_builder.hpp/cpp` — chain the above: raw MBO data → snapshots → features → oracle labels → observation windows with position state baked in. This module owns the mutable state: `position_state` and `entry_price`. It updates these based on oracle output per §2.4. Trajectory starts at `t = t_start = W-1` and ends at `t = t_stop = snapshots.size() - horizon - 1` (inclusive). Assert `snapshots.size() >= W + horizon` before starting. Output: vector of `(window: Tensor[W, F], label: int)` pairs.
7. Write `snapshot_sampler.hpp/cpp` — sample N evenly-spaced windows from a trajectory. Assert `trajectory_length >= N_overfit` before sampling. Run label distribution checks (§4.1) on the subsampled set, not the full trajectory. If distribution check fails, retry with shifted offset before erroring.
8. Write `gbt_features.hpp/cpp` — compute the 16 hand-crafted features from §5.4 given an observation window. Assert `t == W - 1` at the start. Implement trade tape deduplication per §2.7 with zero-size trade filtering per §2.1.2. Apply epsilon guards (1e-8) in all ratio features (book_imbalance, book_depth_ratio_5, top_level_size_ratio). Default all trade-based features to 0.0 when no real trades exist in the computation window.

**Phase 1 Validation Gate:**
```
Assert: feature_encoder output shape is (194,) per snapshot
Assert: all index range constants in feature_encoder cover [0, 194) with no gaps or overlaps
Assert: trajectory_builder output windows have shape (600, 194)
Assert: trajectory starts at t = W-1 (first window spans snapshots [0, W))
Assert: trajectory stops at t = snapshots.size() - horizon - 1 (oracle always has lookahead)
Assert: snapshots.size() >= W + horizon is checked before trajectory generation
Assert: trajectory_length >= N_overfit is checked before sampling
Assert: no NaN or Inf in any tensor
Assert: z-score denominators are all >= 1e-8 (epsilon floor is active)
Assert: label dtype is int64 and all values are in {0, 1, 2, 3}
Assert: label distribution on subsampled set has at least 3 of 4 used classes present
Assert: REVERSE (class 4) never appears in labels
Assert: oracle position-action consistency: labels 1,2 never returned when position_state != 0
Assert: position_state values are exactly {-1.0, 0.0, 1.0}
Assert: entry_price is NaN when position_state == 0, and not NaN otherwise
Assert: book_builder level ordering: bids[0].price >= bids[1].price, asks[0].price <= asks[1].price
Assert: book_builder trade array is always length T=50 (left-padded with zeros if needed)
Assert: book_builder emits snapshots only during RTH (or configured session window)
Assert: book_builder processes F_LAST correctly (no intermediate book states emitted)
Assert: book_builder handles MBO snapshot sequences (Clear + Adds with F_SNAPSHOT flag)
Assert: book_builder filters by instrument_id (only events for target instrument processed)
Assert: GBT features have shape (16,) per sample, no NaN
Assert: GBT epsilon guards are active (book_imbalance denominator, book_depth_ratio_5 denominator,
        top_level_size_ratio denominator all >= 1e-8)
Assert: GBT trade features use deduplicated trades (verify count < sum of raw trade array lengths
        for overlapping windows)
Assert: GBT trade features filter zero-size padding trades
Assert: GBT trade features return 0.0 when no real trades exist in computation window
Print: trajectory bounds (t_start, t_stop, trajectory_length)
Print: subsample label distribution, feature statistics (min, max, mean, std per feature)

Build validation: cmake --build build && ctest --test-dir build --output-on-failure
```

### Phase 2: Model Implementations (C++ with libtorch)

9. Write `base.hpp` — define `TradingModel` abstract base class (libtorch `torch::nn::Module`), `FEATURE_DIM = 194`, `WINDOW_SIZE = 600`, `NUM_ACTIONS = 5`. Include a `count_parameters(model)` utility.
10. Write `mlp.hpp/cpp` per §5.1 using libtorch (`torch::nn::Linear`, `torch::relu`).
11. Write `ssm_training.py` per §5.2 — Python script that reads preprocessed tensor data from disk (e.g., `.pt` files saved by the C++ pipeline via `torch::save`), trains Mamba, writes results to JSON. Skip if no CUDA available; add runtime check.
12. Write `cnn.hpp/cpp` per §5.3 using libtorch — import index range constants from `feature_encoder.hpp` for the feature split. Implement price ladder reindexing as specified.
13. Write `gbt.hpp/cpp` per §5.4 — wrapper around XGBoost C API that accepts a `(N, 16)` feature matrix, trains with `multi:softmax` objective, returns predictions. Document objective choice in constructor.

**Phase 2 Validation Gate:**
```
For each model (MLP, CNN via libtorch; GBT via XGBoost C API):
  Assert: forward pass on random tensor (B=4, W=600, F=194) produces shape (4, 5)
  Assert: no NaN in output logits
  Assert: parameter count matches expected order of magnitude
  Print: model name, parameter count, output shape, forward pass time

For CNN specifically:
  Assert: book_spatial tensor has shape (B*W, 2, 20) after split and reindex
  Assert: spatial conv output has shape (B, W, 64) before concatenation

Build validation: cmake --build build && ctest --test-dir build --output-on-failure
```

### Phase 3: Training Harness (C++ + Python for SSM)

14. Write `config.hpp` — all hyperparameters from §6.4, SEED, and constants as `constexpr`.
15. Write `overfit_test.hpp/cpp` — training loop that:
    - Sets reproducibility seeds per §6.1
    - Takes a model, a dataset of (window, label) pairs, and max epochs
    - For neural models: trains with CrossEntropyLoss, Adam, gradient clipping, dropout=0
    - For GBT: trains XGBoost with overfit-friendly params via C API
    - Logs loss and accuracy every 10 epochs
    - For N=128: computes accuracy over all 128 samples at end of each epoch (full inference pass), not per-batch
    - Checks for NaN/Inf every epoch
    - Detects class collapse: if accuracy is within ±2% of majority class frequency for >50 consecutive epochs, log a WARNING
    - Stops early if accuracy ≥ 99%
    - Returns: pass/fail, final accuracy, final loss, epoch count, loss history
16. Write `run_overfit_suite.cpp` — `main()` entry point. Parses CLI args, runs all specified models through the overfit test, prints summary table (including trajectory bounds), saves results to JSON. Includes REVERSE-never-in-labels assertion in the preamble. Also writes preprocessed tensor data to disk for the SSM Python script.
17. Write `ssm_training.py` integration — reads tensors from `data/overfit/`, trains Mamba, writes results JSON that `run_overfit_suite` can merge into the summary table.

**Phase 3 Validation Gate:**
```
Assert: overfit_test runs without error on a synthetic random dataset
Assert: MLP achieves >90% on random 32-sample dataset within 500 epochs
        (random labels + enough capacity = should memorize)
Assert: loss history is logged correctly
Assert: early stopping triggers when accuracy hits threshold
Assert: class collapse detection fires on a synthetic all-same-label dataset
Assert: N=128 accuracy is computed on all 128 samples, not per-batch
Assert: preprocessed tensor files are written correctly for SSM Python script

Build validation: cmake --build build && ctest --test-dir build --output-on-failure
```

### Phase 4: Overfit Execution

17. Run the overfit suite with N=32 on real MES MBO data (`glbx-mdp3-20220103.mbo.dbn.zst`, instrument_id `13615`). All models must pass.
18. If any model fails, diagnose using §6.5 and fix. Re-run from scratch (new seeds are not allowed — use SEED=42).
19. Run the overfit suite with N=128. All models must pass.
20. If any model fails at N=128 but passed N=32: the model likely needs more epochs or has a subtle capacity issue. Check loss curves. Do not change hyperparameters — investigate the root cause.
21. All models pass → checkpoint the validated code. Tag as `v0.6-overfit-validated`.

### Phase 5: Serialization

22. For MLP and CNN: save libtorch checkpoint (`torch::save`), reload, verify bitwise identical forward pass on the N=32 test batch.
23. For MLP and CNN: export to ONNX. Verify ONNX output matches libtorch output within tolerance (`atol=1e-4`, `rtol=1e-4`). Note: `atol=1e-5` is too tight for float32 after multiple conv/linear layers; 1e-4 is standard and prevents false alarms.
24. For SSM (if available): save PyTorch checkpoint, reload, verify identical forward pass. **Do NOT attempt ONNX export** — Mamba's selective scan uses custom CUDA kernels that don't export cleanly. ONNX for SSM is deferred to a future spec that will address custom operator registration or a pure-PyTorch inference path.
25. For GBT: save XGBoost model (`XGBoosterSaveModel`), reload (`XGBoosterLoadModel`), verify identical predictions on the N=32 test batch.

**Phase 5 Validation Gate:**
```
Assert: all save/load round-trips produce identical outputs
Assert: ONNX models (MLP, CNN) match libtorch within tolerance (atol=1e-4, rtol=1e-4)
Assert: SSM checkpoint loads correctly (no ONNX required)
Assert: GBT predictions match after reload
```

---

## 9. What This Spec Does NOT Cover

- Hyperparameter tuning (irrelevant for overfit test)
- Full training on large datasets
- RL training loop and reward shaping
- Live execution / order management
- Position management and risk controls
- Backtesting framework
- Multi-instrument support
- The 9:25 AM supervised strategy (separate spec)
- The dual-architecture fusion model (build after individual models validate)
- SSM ONNX export (blocked by custom CUDA kernels)
- Mamba CPU fallback / S4 alternative (out of scope)
- Multi-day book continuity (each day is processed independently using its snapshot sequence)

---

## 10. Exit Criteria

This spec is COMPLETE when:

- [ ] C++ CMake project builds cleanly with databento-cpp, libtorch, xgboost, Catch2
- [ ] Data pipeline reads real MES MBO data from `.dbn.zst` files using `databento::DbnFileStore` and produces normalized observation windows of shape `(600, 194)`
- [ ] Book builder reconstructs full order book from MBO events per §2.1.1: order map tracking, bid/ask aggregation maps, `F_LAST` batch processing, snapshot sequence handling, instrument filtering, session filtering (RTH), warmup, gap handling, level ordering, empty level padding, trade buffer (`action='T'` only), trade left-padding, mid_price carry-forward
- [ ] Book builder trade arrays are always length T=50 (left-padded with zeros per §2.1.2)
- [ ] Feature encoder produces exactly `F=194` features per snapshot, matching the layout in §2.2, with all index range constants exported and verified
- [ ] Z-score normalization uses epsilon floor (1e-8) with no division-by-zero possible
- [ ] Oracle labeler generates labels with reasonable class distribution on the subsampled set (at least 3 of 4 used classes, no class >80%)
- [ ] Oracle labeler includes take-profit exit logic (`take_profit_ticks`) alongside stop-loss
- [ ] Oracle labeler never generates REVERSE (class 4)
- [ ] Oracle labeler asserts position-action consistency: never returns ENTER LONG/SHORT when position_state != 0
- [ ] All labels are int64 with values in {0, 1, 2, 3}
- [ ] entry_price is correctly tracked by trajectory_builder and passed to oracle as read-only
- [ ] Trajectory starts at `t = W-1` and stops at `t = snapshots.size() - horizon - 1`, guaranteeing full observation windows and oracle lookahead
- [ ] Trajectory builder asserts `snapshots.size() >= W + horizon` before starting
- [ ] Snapshot sampler asserts `trajectory_length >= N_overfit` before sampling
- [ ] Position state is correctly tracked through the trajectory and encoded in feature vectors
- [ ] GBT features (16 per sample) compute without NaN and have expected value ranges, with t==W-1 asserted
- [ ] GBT ratio features use epsilon guards (1e-8) to prevent division by zero
- [ ] GBT trade-based features use deduplicated trades per §2.7, filter zero-size padding, and default to 0.0 when no real trades exist
- [ ] GBT wrapper documents objective choice (`multi:softmax` vs `multi:softprob`) in code
- [ ] CNN correctly splits feature vector using exported index range constants and constructs price ladder
- [ ] All available models pass the N=32 overfit test (≥99% train accuracy)
- [ ] All available models pass the N=128 overfit test (≥95% train accuracy)
- [ ] N=128 accuracy is evaluated on all 128 samples per epoch
- [ ] Two runs with SEED=42 produce identical loss curves
- [ ] Neural model checkpoints save/load with bitwise identical outputs (libtorch for MLP/CNN, PyTorch for SSM)
- [ ] ONNX export works for MLP and CNN (not SSM) with matching outputs (atol=1e-4)
- [ ] GBT model saves/loads with identical predictions (XGBoost C API)
- [ ] No NaN/Inf anywhere in the pipeline
- [ ] Entire overfit suite runs in < 30 minutes on CPU (excluding SSM)
- [ ] All C++ unit tests pass (`ctest --test-dir build --output-on-failure`)

---

## Appendix A: Changelog from v0.2

| Section | Change | Rationale |
|---------|--------|-----------|
| §2.2 | Added explicit index ranges and slice constants for feature vector layout | CNN and GBT need to split the flat 194-vector; implicit indexing was error-prone |
| §2.4 | Added entry_price tracking to trajectory generation; specified trajectory_builder as state owner | Oracle pseudocode referenced entry_price but no component tracked it |
| §2.6 | Added epsilon floor (1e-8) to z-score denominator | Prevents division by zero in low-volume windows; was only mentioned in diagnosis, not prevented by construction |
| §4.1 | Oracle function signature now includes entry_price parameter; added preconditions; oracle is explicitly stateless | Clarifies state ownership between oracle and trajectory_builder |
| §4.1 | Label distribution check moved to subsampled set; added REVERSE-never-in-labels assertion | Full trajectory distribution doesn't guarantee subsample distribution; REVERSE should be explicitly verified absent |
| §5.1 | Added convergence note about MLP's slow gradient propagation through 116K-dim input | Prevents false alarm when MLP takes 300+ epochs |
| §5.2 | Pinned mamba-ssm to ~=2.2 | API changed between major versions; loose pin risks build failures |
| §5.3 | Added explicit feature split indices and price ladder reindexing procedure | Eliminates ambiguity in how flat feature vector maps to CNN's spatial representation |
| §5.4 | Added assertion that t==W-1; clarified all backward indices are valid by construction | Prevents silent bugs if t is not the last window index |
| §5.5 (Phase 5) | Relaxed ONNX tolerance from 1e-5 to 1e-4 | 1e-5 is too tight for float32 after multiple layers; 1e-4 is standard practice |
| §6.3 | Added collapse detection description | Accuracy plateau at majority class frequency is a distinct failure mode from capacity issues |
| §6.5 | Added item 9 for MLP-specific slow convergence; updated item 3 to reference epsilon floor | Reduces false positives in failure diagnosis |
| §7.4 | Updated expected MLP epoch count in example output; added REVERSE assertion to output | Reflects more realistic MLP convergence behavior |
| §10 | Added 5 new exit criteria covering entry_price, epsilon floor, REVERSE assertion, CNN slicing, ONNX tolerance | Closes gaps in verifiable completeness |

---

## Appendix B: Changelog from v0.3

| Section | Change | Rationale |
|---------|--------|-----------|
| §2.4 | Added trajectory start index: `t = W-1`. Pseudocode now starts loop at `W-1` instead of `t=0`. | Prevents negative indexing into observation windows. Windows at `t < W-1` would require padding or reach into invalid indices. Making this explicit eliminates an entire class of bootstrapping bugs. |
| §2.7 | **New section**: Trade tape deduplication strategy. | Consecutive snapshots have overlapping trade arrays (each snapshot stores last 50 trades). Without deduplication, GBT trade features double-count trades. Feature encoder intentionally does NOT deduplicate (raw snapshot view); GBT features MUST deduplicate. The split is now explicit. |
| §4.1 | Added `take_profit_ticks` parameter (default: 20). In-position oracle now exits on both stop loss AND take profit. | Previous oracle only exited on stop loss, creating degenerate label distribution where EXIT correlated exclusively with losing trades. Take-profit exit ensures EXIT labels span both adverse and favorable price movements, improving label diversity. |
| §5.4 | Added `+ 1e-8` epsilon guard to denominators of `book_imbalance`, `book_depth_ratio_5`, and `top_level_size_ratio`. | These features divide by ask-side sizes which can be zero (e.g., empty book levels during fast markets). Previously only mentioned as a diagnosis step in §6.5; now prevented by construction. |
| §5.4 | All trade-based features now reference §2.7 deduplication. Snapshot ranges use inclusive `t+1` upper bound for clarity. | Eliminates ambiguity about whether trade features double-count across overlapping snapshot trade arrays. |
| §6.2 | Added N=128 evaluation clarification: accuracy computed on all 128 samples per epoch, not per-batch. | With batch_size=32 and N=128, per-batch accuracy can fluctuate. Full-set evaluation after each epoch ensures consistent measurement. |
| §6.5 | Updated item 5 to specify int64 dtype. Updated item 7 to reference epsilon guards instead of just diagnosis. | Label dtype was only mentioned in diagnosis, not enforced. GBT denominator guards are now prevention, not just detection. |
| §7.3 | Added deduplication note to `gbt_features.py` in project structure. Added dedup test to `test_gbt_features.py`. | Makes deduplication responsibility visible in project layout. |
| §7.4 | Updated example label distribution to reflect take-profit exit (EXIT now ~15% vs ~8%). | Take-profit exit increases EXIT frequency; example output should match. |
| §8 Phase 1 | Added 4 new assertions: label dtype int64, trajectory start at W-1, GBT epsilon guards, GBT trade dedup. | Closes validation gaps that could allow silent bugs through the gate. |
| §8 Phase 3 | Added assertion that N=128 accuracy uses full-set evaluation. Updated overfit_test.py spec. | Ensures training harness computes accuracy consistently. |
| §10 | Added 6 new exit criteria: take-profit logic, label dtype, trajectory start, GBT epsilon guards, GBT dedup, N=128 eval. | Verifiable completeness for all v0.4 changes. |

---

## Appendix C: Changelog from v0.4

| Section | Change | Rationale |
|---------|--------|-----------|
| §2.1.1 | **New section**: Book builder reconstruction rules. Specifies snapshot timing (100ms boundaries), state accumulation, RTH session filtering, 1-minute warmup, gap handling (carry-forward + 5s warning), empty level padding `(0.0, 0.0)`, trade buffer with left-padding, mid_price/spread carry-forward, and level ordering assertions. | `book_builder.py` was the most complex Phase 1 module but had only one sentence of spec. An agent would make silent assumptions about session boundaries, gap behavior, empty books, and trade buffering that cascade into downstream data quality issues. |
| §2.1.2 | **New section**: Trade array padding convention. Left-pad with zeros when fewer than T=50 trades exist. Documents impact on feature encoding (price delta, z-score), GBT features (zero-size filtering), and normalization (bimodal distribution in early-session windows). | Without a padding convention, the trade array's layout shifts based on buffer fill level, creating index-dependent bugs. Left-padding keeps recent trades at consistent positions. Zero-padded entries also needed explicit handling rules for GBT deduplication and feature computation. |
| §2.4 | Added explicit trajectory stop index: `t_stop = len(snapshots) - horizon - 1`. Added minimum data length assertion: `len(snapshots) >= W + horizon`. Updated pseudocode loop bounds from `t_start to t_stop (inclusive)`. | Previous spec said "to end of data (minus oracle horizon)" without a formula. The oracle's precondition requires `t + horizon <= len(snapshots)` but no code enforced this bound. Explicit stop index and minimum length assertion prevent the oracle from reading past the end of the data array. |
| §2.7 | Added step 4 to deduplication: filter entries where `size == 0` (padding per §2.1.2). | Zero-padded trade entries from §2.1.2 are identical across snapshots and would survive deduplication as a single zero-size "trade". These must be filtered to avoid corrupting trade-based feature calculations (especially median for `large_trade_flag` and count for `trade_arrival_rate_5s`). |
| §4.1 | Added position-action consistency assert at oracle exit: `assert position_state == 0 or label in (0, 3)`. Added paragraph explaining the structural guarantee and why the assert is still valuable. | The oracle's branch structure guarantees it never returns ENTER LONG/SHORT when in a position, but an implementation bug (e.g., wrong branch condition) could violate this silently. The assert catches implementation errors that the branch structure is supposed to prevent. |
| §5.4 | Added GBT objective clarification: use `multi:softmax` for overfit harness (returns class indices), with note on switching to `multi:softprob` if probability calibration or neural model comparison is needed later. Document choice in wrapper `__init__`. | `multi:softmax` and `multi:softprob` produce different output shapes and the wrong choice causes silent shape bugs downstream. Making the choice explicit and documented prevents confusion when the GBT wrapper is reused in a different context. |
| §5.4 | Updated trade features 9-11 to specify "If no real trades, return 0.0". Updated feature 12 (`large_trade_flag`) to filter `size > 0` in median calculation. | With zero-padded trade arrays (§2.1.2), early-session windows may have no real trades after deduplication + filtering. Without explicit defaults, trade features would produce NaN (division by zero in VWAP) or misleading values (median including zeros). |
| §6.2 | Added trajectory length guard: `assert trajectory_length >= N_overfit`. Added sampling window advice to avoid first 60s of RTH for primary sampling. | If the trajectory is shorter than N_overfit (possible with aggressive filtering or short data windows), the sampler would produce fewer samples than expected. The assert catches this before training starts with wrong batch sizes. Sampling advice prevents systematic early-session artifacts in overfit test data. |
| §7.3 | Added `test_book_builder.py` to test suite. Updated annotations on `book_builder.py`, `oracle_labeler.py`, `trajectory_builder.py`, `snapshot_sampler.py`, `gbt_features.py` to reference new sections. | New §2.1.1 rules need dedicated tests. Module annotations in project structure help the orchestrator agent understand what each module is responsible for. |
| §7.4 | Added trajectory bounds to expected output. | Trajectory bounds (t_start, t_stop, length) should be visible in the overfit suite output for debugging data window issues. |
| §8 Phase 1 | Added 8 new assertions: trajectory stop index, minimum data length, trajectory length >= N_overfit, oracle position-action consistency, book builder level ordering, book builder trade array length, book builder session filtering, GBT zero-size trade filtering, GBT empty-window defaults. Updated book_builder.py build instruction to reference §2.1.1. | Phase 1 validation gate must verify all new invariants from §2.1.1, §2.1.2, §2.4, and §4.1. Without these assertions, the new rules are documented but not enforced at the gate. |
| §8 Phase 4 | Updated tag to `v0.5-overfit-validated`. | Version bump. |
| §10 | Added 7 new exit criteria: book builder reconstruction rules, trade array padding, trajectory stop index, minimum data length, trajectory length guard, oracle position-action consistency, GBT objective documentation, GBT zero-size trade handling. | Verifiable completeness for all v0.5 changes. |

---

## Appendix D: Changelog from v0.5

| Section | Change | Rationale |
|---------|--------|-----------|
| §2.1 | **Source schema changed from MBP-10 to MBO**. The raw data on disk is Databento MBO (L3, per-order events). References `DATA/README.md` for schema details. | The actual data downloaded and stored at `DATA/GLBX-20260207-L953CAPU5B/` is MBO, not MBP-10. The previous spec said "Do not use MBO or other schemas without updating this spec." This update fulfills that instruction. MBO is strictly richer than MBP-10 — we reconstruct aggregated price levels from individual order events. |
| §2.1.1 | **Complete rewrite**: Book builder now reconstructs from individual MBO order events instead of pre-aggregated MBP-10 levels. Specifies full order book state (order map + bid/ask aggregation maps), per-action processing rules (Add, Cancel, Modify, Trade, Fill, Clear), `F_LAST` batch processing, snapshot sequence handling, instrument filtering, and trade extraction from `action='T'` events only. | MBO requires maintaining a full order book and processing each order event individually. MBP-10 provided pre-aggregated levels — the builder just tracked the latest update per level. The new builder is fundamentally more complex but produces the same output (`BookSnapshot` with 10 bid/ask levels). This is the biggest architectural change in v0.6. |
| §2.1.1 | Added **instrument filtering** requirement. Book builder accepts target `instrument_id` and processes only matching events. | MES.FUT MBO data contains multiple instruments (9 outrights + calendar spreads). Without filtering, the book builder would co-mingle orders from different instruments in the same book, producing corrupt snapshots. |
| §2.1.1 | Added **`F_LAST` batch processing** rule. | MBO events arrive in multi-message venue packets. Processing and emitting after each individual message would expose intermediate book states (e.g., a cancel followed by an add in the same packet). `F_LAST` marks the end of each atomic venue update. |
| §2.1.1 | Added **snapshot sequence handling** note. Each daily `.dbn.zst` starts with a Clear + Adds sequence (with `F_SNAPSHOT` flag) that populates the initial book. | Without this, the book builder would start from an empty book at the beginning of each file and miss all resting orders. The snapshot sequence provides the warm-start state. |
| §2.1.2 | Trade extraction clarified: trades come from `action='T'` (aggressor) events only. `action='F'` (fill/passive) is NOT appended to avoid double-counting. | With MBO, each trade generates two events (T for aggressor, F for passive). Appending both would double-count every trade. MBP-10 had a single trade event per match. |
| §2.5 | Updated data source path to reference actual dataset: `DATA/GLBX-20260207-L953CAPU5B/`, daily `.dbn.zst` files, full year 2022, recommended file `glbx-mdp3-20220103.mbo.dbn.zst` with instrument_id `13615` (MESM2). | Previous spec referenced generic "any single trading day" without a concrete path. Concrete paths prevent agents from searching for data or making assumptions about file locations. |
| §2.6 | Added note about MBO fixed-point price conversion (`int64_t` with scale 1e-9 → `float`). | MBO prices are fixed-point integers, not floats. The conversion step must happen before delta-price computation. MBP-10 prices were already in the expected format. |
| §2.7 | Updated deduplication for MBO: trade buffer already contains one entry per trade (only `action='T'` events appended), so cross-event duplication is not a concern. Deduplication across overlapping snapshot windows remains necessary for GBT features. Preferred dedup key is `(ts_event, order_id)` from the original MBO event. | MBO provides globally unique identifiers per trade (ts_event + order_id), making deduplication more reliable than the 4-tuple fallback needed with MBP-10 trade messages. |
| §4.1 | Oracle pseudocode converted from Python to C++. Uses `NaN` instead of `None` for unset `entry_price`. | Aligns with C++ implementation. `NaN` is the idiomatic C++ sentinel for "no value" in float context and enables `std::isnan()` checks. |
| §5 | Model protocol changed from Python `Protocol` to C++ `torch::nn::Module` abstract base class (libtorch). | All neural models are implemented in C++ with libtorch. The interface is the same: `(batch, 600, 194) → (batch, 5)`. |
| §5.2 | SSM implementation clarified as Python-only via `python/ssm_training.py`. Reads preprocessed tensor data from disk (written by C++ pipeline). No C++ equivalent. | Mamba's `mamba-ssm` package is Python/CUDA only with no C++ API. The SSM trains on tensors exported by the C++ pipeline, maintaining the C++ data pipeline while accepting Python for this one model. |
| §5.3 | Feature split constants converted from Python `slice()` to C++ `constexpr int BEGIN/END` pairs. | C++ does not have Python's `slice` type. `constexpr` integer pairs are the idiomatic equivalent and enable compile-time bounds checking. |
| §5.4 | GBT implementation changed to XGBoost **C API** (`XGBoosterCreate`, etc.) instead of Python wrapper. | Aligns with C++ codebase. XGBoost C API is well-documented and avoids Python interop overhead. |
| §6.1 | Added C++ reproducibility setup (libtorch `torch::manual_seed`, `srand`). Python section retained for SSM only. | C++ is the primary language; reproducibility seeds must be set in C++. |
| §6.2 | Updated data paths and dates to reference actual dataset: `glbx-mdp3-20220103.mbo.dbn.zst`, instrument_id `13615` (MESM2). | Concrete, actionable data reference instead of placeholder dates. |
| §7.1 | **Complete rewrite**: Dependencies split into C++ (primary) and Python (SSM only). C++ deps: `databento-cpp`, `libtorch`, `xgboost`, `Catch2`. Build system: CMake with FetchContent. | Previous spec was Python-only. C++ dependencies are fundamentally different and require CMake integration. |
| §7.3 | **Complete rewrite**: Project structure is now a CMake C++ project with `include/kenoma/`, `src/`, `tests/` directories. All pipeline and model modules are `.hpp/.cpp`. `python/` directory contains only `ssm_training.py`. `scripts/download_data.py` stays Python (Databento API client). | Previous structure was entirely Python modules. The new layout follows standard C++ project conventions with header/source separation. |
| §7.4 | **Complete rewrite**: Build/run commands use `cmake` and compiled binary (`./build/run_overfit_suite`) instead of `python`. Added `--instrument-id` CLI flag. SSM runs separately via Python script. | C++ executables require a build step. The instrument_id flag is new (required for MBO instrument filtering). |
| §8 | **Major rewrite**: All phase instructions reference C++ modules (`.hpp/.cpp`), CMake builds, `databento::DbnFileStore`, libtorch, XGBoost C API. Phase 1 adds MBO-specific validation: `F_LAST` processing, snapshot sequence handling, instrument filtering, order map correctness. Build validation step (`cmake --build && ctest`) added to each phase gate. | Previous phases assumed Python modules. C++ phases include build verification as a first-class validation step — the code must compile before any runtime assertions. |
| §8 Phase 4 | Updated tag to `v0.6-overfit-validated`. Data file and instrument_id specified. | Version bump + concrete data reference. |
| §8 Phase 5 | Checkpoints use `torch::save` (libtorch) for MLP/CNN, `XGBoosterSaveModel`/`XGBoosterLoadModel` for GBT. | C++ serialization APIs differ from Python. |
| §9 | Added "Multi-day book continuity" to not-covered list. | Each daily `.dbn.zst` file starts with a snapshot sequence that resets the book. Cross-day continuity is not needed for the overfit harness and would add complexity. |
| §10 | Added 4 new exit criteria: CMake project builds cleanly, `F_LAST` processing, snapshot sequence handling, instrument filtering, C++ unit tests pass. Updated existing criteria to reference C++ APIs. | C++ build and test infrastructure must be validated. MBO-specific invariants (F_LAST, snapshot sequences, instrument filtering) are new requirements. |
