# TDD Spec: Fix Tick Bar Construction in bar_feature_export

**Date:** 2026-02-19
**Depends on:** Phase 1 (bar-construction), Phase 8 (bar-feature-export)
**Priority:** Blocking for R3b rerun (event-bar CNN experiment); non-blocking for time_5s CNN+GBT pipeline

---

## Problem

`bar_feature_export --bar-type tick --bar-param N` constructs bars by counting N book snapshots (emitted at fixed 100ms intervals by the book builder), not N actual trade events (action='T' MBO messages). This makes every "tick bar" functionally identical to a time bar at N/10 seconds.

**Evidence:** R3b discovered that tick bars have:
- `bars_per_day_std = 0.0` across all 19 trading days at every threshold tested (tick_100 through tick_1500)
- Within-day duration p10 = p90 (zero variance — every bar has identical duration)
- Bar count per day is exactly `session_seconds / (threshold / 10)` — consistent with counting 10/s snapshots

**Blast radius:** Every experiment that used "tick bars" (R1 tick_25/50/100, R4c tick_50/100/250, R4d tick_500/3000, R3b tick_100/500/1000/1500) actually tested time bars at different frequencies. Dollar and volume bar construction are NOT affected (confirmed genuine via bar_count_cv > 0 in R1 metrics and non-zero within-day duration variance).

---

## Root Cause

The book builder (Phase 1, §2.1.1 of ORCHESTRATOR_SPEC) emits snapshots at fixed 100ms boundaries. It processes MBO events including action='T' (Trade) and maintains a rolling trade buffer (last 50 trades). However, the bar construction layer in `bar_feature_export` does not have access to the per-snapshot trade count. When constructing tick bars, it counts the number of snapshots received rather than the number of trade events that occurred across those snapshots.

The dollar bar builder correctly aggregates dollar volume because it computes cumulative price × size from trade data. The volume bar builder correctly counts trade volume. Only the tick bar builder is broken because it uses snapshot count as a proxy for trade count — a proxy that is always exactly 10 per second.

---

## Fix

### Approach

Add a `trade_count` field to each book snapshot that records the number of action='T' MBO events processed since the previous snapshot emission. The bar construction layer accumulates this count and closes a tick bar when the cumulative trade count reaches the threshold N.

### Detailed Changes

**1. Book builder: emit per-snapshot trade count**

Each 100ms snapshot already processes all MBO events since the previous boundary. The builder already appends to its trade buffer on action='T'. Add a counter that:
- Resets to 0 at each snapshot emission
- Increments by 1 on each action='T' event processed
- Is emitted as part of the snapshot metadata

This counter is NOT the same as the trade buffer length (the buffer is rolling and caps at 50). It is a simple per-interval counter.

**2. Bar construction: use trade_count for tick bars**

Replace the current tick bar logic (which closes a bar every N snapshots) with:
- Maintain a running `cumulative_trades` counter (starts at 0 at session start)
- For each snapshot, add `snapshot.trade_count` to `cumulative_trades`
- When `cumulative_trades >= threshold`, close the current bar and reset `cumulative_trades` to 0 (or carry over the remainder: `cumulative_trades -= threshold`)
- The bar's book snapshot is the snapshot at bar close (same convention as other bar types)

**3. No changes to dollar, volume, or time bar construction**

These are working correctly. The fix must not alter their behavior. This is enforced by regression tests.

---

## Exit Criteria

- [ ] **T1: Trade count field exists on snapshots.** Each book snapshot includes a `trade_count` field (uint32 or equivalent) that counts the number of action='T' MBO events since the previous snapshot.
- [ ] **T2: Tick bars count trades, not snapshots.** `bar_feature_export --bar-type tick --bar-param 100` on a single trading day produces bars whose count varies with market activity, not with clock time.
- [ ] **T3: Daily bar count variance > 0.** Running tick bar export across >= 3 trading days with different activity levels produces non-zero standard deviation in bars-per-day. (Diagnostic: `bars_per_day_std > 0`.)
- [ ] **T4: Within-day duration variance > 0.** For any tick threshold producing >= 200 bars/day, the p10 and p90 of bar durations must differ. Bars should be shorter during high-activity periods (open, close) and longer during low-activity periods (lunch).
- [ ] **T5: Trade count consistency.** For a single-day tick bar export with threshold N: `sum(bar_trade_counts) + remainder == total_trade_events_in_session`. The total trade count from bars must reconcile with the actual trade count from the MBO stream.
- [ ] **T6: Threshold proportionality.** For two thresholds N and 2N on the same day: the bar count ratio is approximately 2:1 (within 20%). Not exact because trade clustering varies, but proportional because both count the same underlying trades.
- [ ] **T7: Regression — time bars unchanged.** `bar_feature_export --bar-type time --bar-param 5` produces identical output before and after the fix. Byte-identical CSV output for at least 1 trading day.
- [ ] **T8: Regression — dollar bars unchanged.** `bar_feature_export --bar-type dollar --bar-param 25000` produces identical output before and after the fix for at least 1 trading day.
- [ ] **T9: Regression — volume bars unchanged.** `bar_feature_export --bar-type volume --bar-param 100` produces identical output before and after the fix for at least 1 trading day.
- [ ] **T10: No empty bars.** Tick bars with `trade_count == 0` must not be emitted. If no trades occur between snapshots, the bar should not close (accumulation continues).
- [ ] **T11: Feature schema unchanged.** The CSV output columns for tick bars must match the existing schema (same columns, same order). The only change is which rows appear (different bar boundaries), not the column structure.
- [ ] **T12: Existing unit tests pass.** All 1003 existing unit tests continue to pass. No disabled or skipped tests introduced by this change.

---

## Test Strategy

### New Unit Tests

**test_tick_bar_counts_trades:**
Construct a synthetic MBO stream with known trade count (e.g., 50 trades in 10 seconds = 100 snapshots). Verify that `--bar-type tick --bar-param 25` produces exactly 2 bars (50 trades / 25 = 2), NOT 4 bars (100 snapshots / 25 = 4).

**test_tick_bar_variable_duration:**
Construct a synthetic MBO stream where the first 5 seconds have 40 trades and the next 5 seconds have 10 trades. With `--bar-param 25`: the first bar should close in ~3.1s (25 trades out of 40 in 5s), the second bar should close in ~6.9s. Verify that bar durations differ.

**test_tick_bar_no_trade_snapshots:**
Construct a synthetic MBO stream with a 2-second gap (20 snapshots) containing zero trades. Verify that the tick bar does not close during this gap — the cumulative trade count does not advance. The bar boundary only moves when trades occur.

**test_tick_bar_daily_variance:**
Run tick bar export on 3 real trading days with known different volumes (e.g., Jan 3 = high volume first day of year, Jul 1 = low volume summer Friday). Verify `bars_per_day_std > 0`. This is the integration-level diagnostic that would have caught the original bug.

**test_tick_bar_trade_reconciliation:**
For a single-day export, verify that the sum of trades across all bars plus any remainder from the last incomplete bar equals the total action='T' event count from the MBO stream for that day.

**test_tick_bar_threshold_proportionality:**
Export tick bars at thresholds 100 and 200 for the same day. Verify that the bar count ratio is within [1.5, 2.5] (approximately 2:1).

### Regression Tests

**test_time_bar_unchanged:**
Export time_5s bars for 1 day before and after the fix. Compare CSV output byte-for-byte. Must be identical.

**test_dollar_bar_unchanged:**
Export dollar_25k bars for 1 day before and after the fix. Compare CSV output byte-for-byte. Must be identical.

**test_volume_bar_unchanged:**
Export volume_100 bars for 1 day before and after the fix. Compare CSV output byte-for-byte. Must be identical.

### Diagnostic Test (guards against future regressions)

**test_event_bar_daily_variance_guard:**
For all event bar types (tick, dollar, volume), verify `bar_count_cv > 0` across >= 3 trading days. This is the master diagnostic that catches any bar type that is secretly a time bar. This test should be labeled as integration (runs on real data, excluded from default ctest).

---

## Scope Constraints

- **DO NOT** change the book builder's snapshot emission timing (100ms boundaries). The fix is in the bar construction layer only, plus adding a `trade_count` metadata field to snapshots.
- **DO NOT** change how dollar or volume bars are constructed. They work correctly.
- **DO NOT** change the feature computation or CSV schema. Only bar boundaries change for tick bars.
- **DO NOT** change the forward return computation. `fwd_return_5` still means "5 bars ahead" — the bar boundaries just move to different timestamps for tick bars.
- **DO** carry over remainder trades to the next bar (if 27 trades occur and threshold is 25, the next bar starts with 2 accumulated). Do NOT discard partial counts.

---

## Validation After Fix

Once this TDD cycle completes (all exit criteria checked), the immediate next step is:

1. **Quick smoke test:** Export tick_100 bars for 3 days. Verify bars_per_day_std > 0 and duration p10 != p90.
2. **R3b rerun:** Re-run the R3b experiment (`.kit/experiments/r3b-event-bar-cnn.md`) with genuine tick bars. This is the experiment that motivated the fix — it will test whether CNN spatial R² on activity-normalized event bars exceeds the time_5s baseline of 0.084.
