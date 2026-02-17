# Last Touch — Cold-Start Briefing

## Project Status

**Phase 1 (bar-construction) complete.** Event-driven bar construction from BookSnapshot stream is fully implemented and tested. 87 new unit tests added (307/308 total pass). Phases 2, R1, and 4 are now unblocked.

## What was completed this cycle

- `src/bars/bar.hpp` — Bar struct, BarBuilder interface, PriceLadderInput, MessageSequenceInput adapters
- `src/bars/bar_builder_base.hpp` — Shared base for all bar builders (OHLCV, spread, message counters)
- `src/bars/volume_bar_builder.hpp` — Volume bar (V=100 default)
- `src/bars/tick_bar_builder.hpp` — Tick bar (K=50 default)
- `src/bars/dollar_bar_builder.hpp` — Dollar bar (D=50000 default)
- `src/bars/time_bar_builder.hpp` — Time bar (60s default)
- `src/bars/bar_factory.hpp` — Config-driven BarBuilder instantiation
- `src/day_event_buffer.hpp` — DayEventBuffer for raw MBO event storage
- `src/features/warmup.hpp` — WarmupTracker (EWMA span guard)
- `tests/bar_builder_test.cpp` — 54 tests for all bar types + factory
- `tests/day_event_buffer_test.cpp` — DayEventBuffer tests
- `tests/warmup_test.cpp` — WarmupTracker tests
- Modified: `src/book_builder.hpp`, `src/feature_encoder.hpp`, `tests/feature_encoder_test.cpp`, `tests/gbt_features_test.cpp`, `CMakeLists.txt`

## What exists

A C++20 MES microstructure model suite that reads raw Databento MBO (L3) order data from `.dbn.zst` files. The overfit harness (MLP, CNN, GBT) is validated at N=32 and N=128 on real data. Serialization (checkpoint + ONNX) is shipped. Bar construction (Phase 1 of TRAJECTORY.md) is complete.

## Phase Sequence

| # | Spec | Kit | Status |
|---|------|-----|--------|
| 1 | `.kit/docs/bar-construction.md` | TDD | **Done** |
| 2 | `.kit/docs/oracle-replay.md` | TDD | **Unblocked** |
| 3 | `.kit/docs/multi-day-backtest.md` | TDD | Blocked by 2 |
| R1 | `.kit/experiments/subordination-test.md` | Research | **Unblocked** |
| 4 | `.kit/docs/feature-computation.md` | TDD | **Unblocked** |
| 5 | `.kit/docs/feature-analysis.md` | TDD | Blocked by 4 |
| R2 | `.kit/experiments/info-decomposition.md` | Research | Blocked by 4 |
| R3 | `.kit/experiments/book-encoder-bias.md` | Research | Blocked by 4 |
| R4 | `.kit/experiments/temporal-predictability.md` | Research | Blocked by 1, R1 |
| 6 | `.kit/experiments/synthesis.md` | Research | Blocked by all |

## Test summary

- **307 unit tests** pass, 1 disabled (`BookBuilderIntegrationTest.ProcessSingleDayFile`), 308 total
- **22 integration tests** (14 N=32 + 8 N=128) — labeled `integration`, excluded from default ctest
- Unit test time: ~5 min. Integration: ~20 min.

## What to do next

1. Ship Phase 1 (commit breadcrumbs + changed files).
2. Start Phase 2 (oracle-replay): `source .master-kit.env && ./.kit/tdd.sh red .kit/docs/oracle-replay.md`
3. After Phase 2, proceed sequentially through the phase table above.

## Build commands

```bash
cmake --build build -j12                                                  # build
cd build && ctest --output-on-failure --label-exclude integration         # unit tests (~5 min)
cd build && ctest --output-on-failure --label-regex integration           # integration tests (~20 min)
```

---

Updated: 2026-02-16
