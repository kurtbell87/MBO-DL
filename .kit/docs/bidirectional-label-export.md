# TDD Spec: Bidirectional Triple Barrier Labels

**Date:** 2026-02-25
**Priority:** P0 — blocks label-design-sensitivity experiment; existing labels are methodologically flawed
**Parent:** Label Design Sensitivity experiment, bar_feature_export (Phase 8)

---

## Problem

The current `bar_feature_export` Parquet labels are **long-perspective only**. The triple barrier check asks: "if we entered long here, does price hit +target before -stop?"

- **+1 label:** price rose `target_ticks` (10) before falling `stop_ticks` (5) → long wins 10 ticks
- **-1 label:** price fell `stop_ticks` (5) before rising `target_ticks` (10) → long loses 5 ticks

**The flaw:** A -1 label (price dropped 5 ticks) is NOT evidence that a short with a 10-tick target would succeed. The label confirms a 5-tick drop, not a 10-tick drop. When the backtest enters short on a -1 prediction with a 10-tick profit target, it credits 10-tick wins on entries validated for only 5-tick moves. This inflates short-side P/L.

**The fix:** Bidirectional labeling with two independent barrier checks per bar:
- Long race: does price hit +target before -stop? (from entry)
- Short race: does price hit -target before +stop? (from entry, same geometry mirrored)
- Label = +1 if only long wins, -1 if only short wins, 0 if neither or both

## Design

### New Labeling Mode

Add a `bidirectional` mode to the triple barrier system. For each bar:

1. **Long race** (independent): scan forward, check if `price - entry >= target_dist` before `entry - price >= stop_dist`. Record: {triggered: bool, bars_to_hit: int, exit_type: string}.
2. **Short race** (independent): scan forward over the **same window**, check if `entry - price >= target_dist` before `price - entry >= stop_dist`. Record: {triggered: bool, bars_to_hit: int, exit_type: string}.
3. **Label resolution:**
   - Long triggered, short not → **+1**
   - Short triggered, long not → **-1**
   - Neither triggered → **0 (HOLD)**
   - Both triggered → **0 (HOLD)** — high-volatility regime

4. **Diagnostic column:** `tb_both_triggered` (int: 0 or 1). When both long and short race trigger in the same window, this flag is set. Frequency of this flag is a free volatility regime indicator — price traveled ≥ target_ticks in BOTH directions within the barrier window.

### Key constraint: races are independent

The two barrier checks MUST be evaluated independently over the same forward-looking window. They do not interact. The long race does not "consume" price moves that the short race also sees. Both races observe the same price sequence from entry.

### Parquet Schema Changes

Existing columns (keep for backward compatibility):
- `tb_label` (float64) — now computed bidirectionally
- `tb_exit_type` (string) — exit type of the winning race ("long_target", "short_target", "long_expiry", "short_expiry", "both", "neither")
- `tb_bars_held` (float64) — bars to resolution of the winning race

New columns:
- `tb_both_triggered` (float64: 0.0 or 1.0) — 1 if both races triggered within the window
- `tb_long_triggered` (float64: 0.0 or 1.0) — whether long race hit target
- `tb_short_triggered` (float64: 0.0 or 1.0) — whether short race hit target

### Integration Point

`bar_feature_export.cpp` currently calls `compute_tb_label()` from `triple_barrier.hpp`. The new bidirectional mode should be:
- A new function (e.g., `compute_bidirectional_tb_label()`) in `triple_barrier.hpp`
- Called from `bar_feature_export.cpp` instead of `compute_tb_label()`
- Controlled by a config flag (e.g., `TripleBarrierConfig::bidirectional = true`)
- Default to bidirectional=true for new exports; old behavior available via flag for reproducibility

### Reference: FIRST_TO_HIT Logic

The `oracle_replay.hpp` FIRST_TO_HIT method already tracks 4 milestones (long_target, long_stop, short_target, short_stop) independently. The bidirectional label logic is conceptually similar but simpler — it only needs to know which races triggered, not full position P/L.

## Tests

### T1: Symmetric Price Move — Both Races Trigger
- Input: price sequence that rises target_ticks AND falls target_ticks within the window
- Expected: label=0, tb_both_triggered=1, tb_long_triggered=1, tb_short_triggered=1
- Validates: both-triggered detection and HOLD resolution

### T2: Clean Long Signal — Only Long Race Triggers
- Input: price rises target_ticks (10) without first falling stop_ticks (5)
- Expected: label=+1, tb_both_triggered=0, tb_long_triggered=1, tb_short_triggered=0
- Validates: long-only signal correctly labeled

### T3: Clean Short Signal — Only Short Race Triggers
- Input: price falls target_ticks (10) without first rising stop_ticks (5)
- Expected: label=-1, tb_both_triggered=0, tb_long_triggered=0, tb_short_triggered=1
- Validates: short signal requires full target_ticks downward move (not just stop_ticks)

### T4: Neither Race Triggers — Expiry
- Input: price moves less than stop_ticks in either direction within volume/time window
- Expected: label=0, tb_both_triggered=0, tb_long_triggered=0, tb_short_triggered=0
- Validates: HOLD on no-move bars

### T5: Independence of Races
- Input: price drops stop_ticks (5) then rises target_ticks (10) from entry
- Under OLD (long-perspective) labeling: this would be -1 (long stopped out)
- Under NEW (bidirectional) labeling: long race triggers (price eventually hit +target), short race does NOT trigger (price only dropped stop_ticks, not target_ticks)
- Expected: label=+1, tb_long_triggered=1, tb_short_triggered=0
- **This is the critical regression test.** The old system would label this -1; the new system labels it +1. The price DID eventually rise target_ticks from entry — the intermediate dip doesn't disqualify the long.

Wait — this needs clarification. In the old system, the long race STOPS when stop is hit. The long race checks barriers sequentially: if stop is hit first, the race is over (label=-1). The long race doesn't continue to see if price later recovers.

Revised T5: The races use first-to-hit logic. In the long race, if price drops stop_ticks before rising target_ticks, the long race LOSES (stop hit). So:
- Long race: price drops 5 (stop hit) → long loses → triggered=false (long target NOT hit)
- Short race: price drops 5, but target is 10 → short target not yet hit. Then price rises 10 → short stop hit → short loses → triggered=false
- Expected: label=0 (neither race triggered their target), tb_both_triggered=0

### T5 (revised): Long Stopped, Short Not Reached
- Input: price drops stop_ticks (5) then rises target_ticks (10) from LOW (but only +5 from entry)
- Long race: stop hit at -5 → long target NOT triggered
- Short race: price dropped 5 (not 10) then rose → short stop hit at +5 from entry → short target NOT triggered
- Expected: label=0, tb_long_triggered=0, tb_short_triggered=0
- Validates: 5-tick drop is NOT a short signal under bidirectional labeling (was -1 under old labeling)

### T6: Short Signal Requires Full Target Move
- Input: price drops exactly target_ticks (10) before rising stop_ticks (5)
- Long race: stop hit at -5 → long target not triggered
- Short race: target hit at -10 → short triggered
- Expected: label=-1, tb_short_triggered=1, tb_long_triggered=0
- Validates: short label requires the full 10-tick downward move

### T7: Parameterized Geometry — Non-Default Target/Stop
- Input: target=15, stop=3 geometry
- Test that long race checks +15/-3 and short race checks -15/+3
- Expected: correct labels at non-default geometry

### T8: Expiry Behavior — Volume and Time Barriers
- Input: price stays within ±(stop-1) ticks for entire volume/time window
- Expected: label=0, both races report expiry
- Validates: expiry logic works for both races

### T9: Both-Triggered Frequency Diagnostic
- Input: a sequence of bars with known volatility (e.g., 5 bars where both races trigger, 5 where neither does)
- Compute both-triggered rate
- Expected: 50% both-triggered rate for this input
- Validates: diagnostic column correctly counts high-volatility bars

### T10: Backward Compatibility — Old Mode Still Works
- Input: same bars as T2/T3 with bidirectional=false config
- Expected: old-style labels (long-perspective only) match exactly
- Validates: existing behavior preserved when bidirectional flag is off

## Build

Follow existing project build conventions. Tests should be in a new test file (e.g., `tests/bidirectional_tb_test.cpp`) or added to existing `tests/triple_barrier_test.cpp`. Use GTest. Build with CMake.

## Exit Criteria

- [ ] All tests T1-T10 pass
- [ ] `bar_feature_export` uses bidirectional mode by default
- [ ] New Parquet columns (`tb_both_triggered`, `tb_long_triggered`, `tb_short_triggered`) present in output
- [ ] Old mode (bidirectional=false) reproduces existing labels exactly (T10)
- [ ] `compute_bidirectional_tb_label()` function is independent of `compute_tb_label()` (no shared mutable state)
- [ ] No regression in existing triple_barrier_test.cpp tests
