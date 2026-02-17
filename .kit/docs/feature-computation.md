# Phase 4: Feature Computation & Export [Engineering]

**Spec**: TRAJECTORY.md §8.2 (returns), §8.3 (feature taxonomy), §8.6 (warm-up/lookahead), §10.2 (structure)
**Depends on**: Phase 1 (bar-construction) — needs `Bar`, bar builders, `DayEventBuffer`.
**Unlocks**: Phase 5 (feature-analysis), Phase R2 (info-decomposition), Phase R3 (book-encoder-bias).

---

## Objective

Implement the full feature taxonomy (Track A hand-crafted + Track B raw representations), compute forward returns at multiple horizons, enforce warm-up and lookahead bias policies, and export everything to Parquet/CSV for Python analysis.

---

## Return Definitions (§8.2)

Returns computed over bars (not clock time), in ticks:

```
return_1   = (bar[i+1].close_mid - bar[i].close_mid) / tick_size
return_5   = (bar[i+5].close_mid - bar[i].close_mid) / tick_size
return_20  = (bar[i+20].close_mid - bar[i].close_mid) / tick_size
return_100 = (bar[i+100].close_mid - bar[i].close_mid) / tick_size
```

Last n bars of each session have undefined forward returns — exclude from analysis. Forward returns are **TARGETS only**, never features.

---

## Track A: Hand-Crafted Features (§8.3 Categories 1–6)

### Category 1: Book Shape (static per bar — snapshot at bar close)

| # | Feature | Description |
|---|---------|-------------|
| 1.1 | book_imbalance_{1,3,5,10} | (bid_vol - ask_vol) / (bid_vol + ask_vol + eps) at top-N levels |
| 1.2 | weighted_imbalance | Same weighted by inverse distance from mid |
| 1.3 | spread | In ticks |
| 1.4 | bid_depth_profile[0..9] | Raw bid sizes at each level (10 features) |
| 1.5 | ask_depth_profile[0..9] | Raw ask sizes at each level (10 features) |
| 1.6 | depth_concentration_{bid,ask} | HHI of sizes across levels |
| 1.7 | book_slope_{bid,ask} | Linear regression slope of log(size) vs level index |
| 1.8 | level_count_{bid,ask} | Number of non-empty levels (0–10) |

### Category 2: Order Flow (aggregated within bar)

| # | Feature | Description |
|---|---------|-------------|
| 2.1 | net_volume | buy_volume - sell_volume |
| 2.2 | volume_imbalance | net_volume / (total_volume + eps) |
| 2.3 | trade_count | Number of trades in bar |
| 2.4 | avg_trade_size | volume / trade_count |
| 2.5 | large_trade_count | Trades with size > 2× rolling median (20-bar window) |
| 2.6 | vwap_distance | (close_mid - vwap) / tick_size |
| 2.7 | kyle_lambda | Regression of Δmid on net_volume over rolling 20-bar window. NaN for first 20 bars of session. |

### Category 3: Price Dynamics (across bars — lookback)

| # | Feature | Description |
|---|---------|-------------|
| 3.1 | return_{1,5,20} | Backward mid-price return over last N bars |
| 3.2 | volatility_{20,50} | std(1-bar returns) over last N bars |
| 3.3 | momentum | sum of signed 1-bar returns over last N bars (path-dependent) |
| 3.4 | high_low_range_{20,50} | (max(high) - min(low)) over last N bars / tick_size |
| 3.5 | close_position | (close - low_N) / (high_N - low_N + eps) |

### Category 4: Cross-Scale Dynamics

| # | Feature | Description |
|---|---------|-------------|
| 4.1 | volume_surprise | Current bar volume / EWMA(volume, span=20) |
| 4.2 | duration_surprise | Current bar duration / EWMA(duration, span=20). Meaningful for event-driven bars only. |
| 4.3 | acceleration | return_1 - return_1[lag=1] |
| 4.4 | vol_price_corr | rolling corr(volume, |return_1|) over 20 bars |

### Category 5: Time Context

| # | Feature | Description |
|---|---------|-------------|
| 5.1 | time_sin, time_cos | Sinusoidal encoding of time-of-day at bar close |
| 5.2 | minutes_since_open | Continuous, from 09:30 |
| 5.3 | minutes_to_close | Continuous, to 16:00 |
| 5.4 | session_volume_frac | Cumulative volume / historical average daily volume. Uses expanding-window mean of prior days (not future data). Day 1: use that day's actual total volume. |

### Category 6: Message Microstructure (intra-bar MBO summary)

| # | Feature | Description |
|---|---------|-------------|
| 6.1 | cancel_add_ratio | cancel_count / (add_count + eps) |
| 6.2 | message_rate | Total MBO messages / bar_duration_s |
| 6.3 | modify_fraction | modify_count / (add + cancel + modify + eps) |
| 6.4 | order_flow_toxicity | Fraction of trades that move mid_price within bar. VPIN proxy — interpret with caution on volume bars (§8.3 circularity note). |
| 6.5 | cancel_concentration | HHI of cancel counts per price level |

**Total Track A**: ~45 features.

---

## Track B: Raw Representations (§8.3)

| # | Representation | Shape | Description |
|---|----------------|-------|-------------|
| B.1 | Book snapshot | (20, 2) per bar | PriceLadderInput: (price_delta_from_mid, size_norm). CNN input. |
| B.2 | Message sequence summary | Fixed-length | Binned action counts per time decile, cancel/add rates in first vs second half, max instantaneous message rate. For R2 Tier 1. |
| B.3 | Lookback book sequence | (W, 20, 2) | Previous W bars' book snapshots. Temporal encoder input proxy. |

For R2 Tier 2: raw event sequences retrieved from `DayEventBuffer` (variable-length, max 500 events, pad shorter).

---

## Warm-Up and Lookahead Bias Policy (§8.6)

1. **EWMA features** (volume_surprise, duration_surprise): Init at bar 0 with first bar's value. Mark first `ewma_span` (20) bars as WARMUP. EWMA resets at session boundaries.
2. **Rolling window features** (volatility, momentum, kyle_lambda, etc.): NaN for first N bars of session (N = window length). GBT handles NaN natively. For MLP: impute with training-set median.
3. **session_volume_frac**: Expanding-window prior-day mean. Day 1: actual total volume (mild lookahead, acceptable).
4. **Forward returns**: Last n bars undefined. Exclude.
5. **CV splits**: Expanding window only. No shuffling. Normalize using training-set stats only.

**Bar-level `is_warmup` flag**: True if ANY feature is in warmup state. Downstream filters on `is_warmup == false`.

---

## Export Format

CSV (or Parquet if available) with columns:
- Bar metadata: `timestamp`, `bar_type`, `bar_param`, `day`, `is_warmup`
- Track A features (all ~45)
- Track B book snapshot (flattened 40 values)
- Track B message summaries
- Forward returns: `return_1`, `return_5`, `return_20`, `return_100`

Rollover transition days excluded.

---

## Project Structure

```
src/features/
  bar_features.hpp          # Track A: Categories 1–6
  raw_representations.hpp   # Track B: book export, message summaries
  feature_export.hpp        # Export to CSV/Parquet
  warmup.hpp                # Warm-up state tracking (shared with Phase 1)

tests/
  bar_features_test.cpp
  raw_representations_test.cpp
  feature_export_test.cpp
  warmup_test.cpp
```

---

## Validation Gate

```
Assert: No unexpected NaN in Track A features (NaN only where documented:
        kyle_lambda first 20 bars, volatility/momentum during lookback warmup)
Assert: Feature count matches taxonomy (verify exact count per category)
Assert: Forward returns computed correctly (no lookahead leakage in feature computation)
Assert: is_warmup flag is set for bars where ANY feature has insufficient lookback
Assert: session_volume_frac uses expanding-window prior-day mean (not future data)
Assert: EWMA state resets at session boundaries
Assert: Track B PriceLadderInput has shape (20, 2) per bar
Assert: Track B message summaries are consistent with Track A Category 6 fields
Assert: Export includes bar metadata (timestamp, bar_type, bar_param, day, is_warmup)
Assert: Rollover transition days excluded from export
```
