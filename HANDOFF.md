# Handoff: bar_feature_export tick bar construction counts snapshots, not trades

**Date:** 2026-02-19
**Triggered by:** R3b-event-bar-cnn (experiment r3b-event-bar-cnn)
**Question:** Does CNN spatial R² improve on genuine trade-triggered tick bars vs time_5s?

**Reason:** The tick bar construction logic in `bar_feature_export` is shared infrastructure used by multiple experiments. Fixing it requires modifying shared C++ code that other experiments depend on. Per the research scope heuristic: "Would a different experiment break if I did this wrong? If yes, handoff." Yes — any experiment requesting tick/event bars from bar_feature_export currently receives time bars at a different frequency without warning. This is a correctness bug in infrastructure code.

## Context

Experiment R3b tested CNN spatial predictability across 4 tick-bar thresholds (tick_100, tick_500, tick_1000, tick_1500) against the time_5s baseline. The experiment was designed to answer whether activity-normalized event bars improve CNN R² on book snapshots.

The data revealed a critical anomaly: all tick-bar thresholds produce **identical bar counts per day** (std=0.0) and **zero duration variance** (p10=median=p90) across all 19 trading days. For example, tick_100 yields exactly 2,289 bars every single day regardless of market activity level. This is impossible for genuine tick bars — MES daily trade volume varies by 3-5x across these dates.

The bar construction counts fixed-frequency book-state snapshots (arriving at ~10/second from the data feed) rather than actual trade events. The result: tick_100 = time_10s, tick_500 = time_50s, tick_1000 = time_100s, tick_1500 = time_150s. These are time bars at different frequencies, not activity-normalized event bars.

This also retroactively affects R4d's tick_500 and tick_3000 data — those were also time bars presented as tick bars. The R4d temporal results (no temporal signal) remain valid since time-bar frequency is an even less favorable condition for temporal signal than genuine tick bars, but the "tick bar" label was incorrect.

## What Is Needed

The `bar_feature_export` tool's tick bar construction mode must count **actual trade events** (MBO messages with action=Trade or equivalent) rather than fixed-rate book-state snapshots.

Specific requirements:
- **Input:** MBO `.dbn.zst` files, tick threshold N
- **Current (broken) behavior:** Emits a bar every N book-state snapshots (fixed rate, ~10/s)
- **Correct behavior:** Emits a bar every N trade events (variable rate, activity-dependent)
- **Verification criterion:** bars_per_day must vary across trading days (std > 0). A day with 50,000 trades should produce ~5x more tick_100 bars than a day with 10,000 trades.
- **Duration distribution:** bar durations must show variance (p10 << p90). Bars during the open should be shorter than bars during lunch.

The fix is in the C++ bar construction code, likely in the bar aggregation loop that processes MBO events. The condition for "advance to next bar" needs to count trade-type messages, not all book-state updates.

## What Has Been Tried

No workaround was attempted. The defect was discovered during post-experiment analysis when the bar statistics showed zero variance. The research pipeline cannot fix this because:
1. It's in shared C++ infrastructure code (bar_feature_export)
2. Other experiments (R4d) already used this code path — fixing it retroactively could invalidate their data
3. The fix requires understanding the MBO message type taxonomy in the databento C++ API

## Suggested Resolution

1. In `bar_feature_export`'s tick-bar construction loop, filter events to only count messages representing actual trades (likely `action == Action::Trade` or equivalent in the databento MBO schema)
2. Verify that the same bar construction logic works correctly for existing time bars (which should be unaffected — time bars don't count events)
3. Add a regression test: for a known-activity day, verify that tick_100 bar count equals (number of trades / 100) ± 1, and that bars_per_day varies across days with different activity levels
4. Re-export tick_500 bars and verify bars_per_day_std > 0

## After Resolution

The research pipeline will:
1. Re-run R3b with genuine tick bars (4 thresholds calibrated to actual trade-driven durations of ~5s, ~30s, ~2min, ~10min)
2. Compare CNN spatial R² on genuine event bars vs time_5s baseline (R²=0.084)
3. This unblocks the P3 question in QUESTIONS.md: "Does CNN spatial R² improve on genuine trade-triggered tick bars vs time_5s?"
4. If genuine event bars show R² > 0.101 (20% improvement), the full-year export pipeline switches to event bars before scaling to 250 days

**Priority: LOW.** The main pipeline direction (time_5s CNN+GBT) is proceeding regardless. This handoff unblocks a secondary research question. It should be addressed opportunistically, not urgently.
