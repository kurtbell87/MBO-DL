# Last Touch — Cold-Start Briefing

## Project Status

**Phase 4 (feature-computation) complete.** Track A hand-crafted features (~45 features across 6 categories), Track B raw representations (book snapshots, message summaries), forward returns at 4 horizons, warm-up/lookahead bias enforcement, and CSV export — all implemented and tested. 173 new unit tests added (726/727 total pass). Phases 5, R2, and R3 are now unblocked.

## What was completed this cycle

- `src/features/bar_features.hpp` — Track A Categories 1–6: book shape, order flow, price dynamics, cross-scale dynamics, time context, message microstructure (~45 hand-crafted features)
- `src/features/raw_representations.hpp` — Track B: PriceLadderInput (20,2) book snapshots, message sequence summaries, lookback book sequences
- `src/features/feature_export.hpp` — CSV export with bar metadata, Track A features, Track B flattened, forward returns
- `src/features/warmup.hpp` — Warm-up state tracking: EWMA reset at session boundaries, rolling window NaN policy, bar-level `is_warmup` flag
- `tests/bar_features_test.cpp` — Track A feature computation tests
- `tests/raw_representations_test.cpp` — Track B representation tests
- `tests/feature_export_test.cpp` — Export format and metadata tests
- `tests/feature_warmup_test.cpp` — Warm-up enforcement and session boundary tests
- Modified: `CMakeLists.txt` (new test targets)

## What exists

A C++20 MES microstructure model suite that reads raw Databento MBO (L3) order data from `.dbn.zst` files. The overfit harness (MLP, CNN, GBT) is validated at N=32 and N=128 on real data. Serialization (checkpoint + ONNX) is shipped. Bar construction (Phase 1), oracle replay (Phase 2), multi-day backtest infrastructure (Phase 3), and feature computation/export (Phase 4) are complete.

## Phase Sequence

| # | Spec | Kit | Status |
|---|------|-----|--------|
| 1 | `.kit/docs/bar-construction.md` | TDD | **Done** |
| 2 | `.kit/docs/oracle-replay.md` | TDD | **Done** |
| 3 | `.kit/docs/multi-day-backtest.md` | TDD | **Done** |
| R1 | `.kit/experiments/subordination-test.md` | Research | **Unblocked** |
| 4 | `.kit/docs/feature-computation.md` | TDD | **Done** |
| 5 | `.kit/docs/feature-analysis.md` | TDD | **Unblocked** |
| R2 | `.kit/experiments/info-decomposition.md` | Research | **Unblocked** |
| R3 | `.kit/experiments/book-encoder-bias.md` | Research | **Unblocked** |
| R4 | `.kit/experiments/temporal-predictability.md` | Research | Blocked by R1 |
| 6 | `.kit/experiments/synthesis.md` | Research | Blocked by all |

## Test summary

- **726 unit tests** pass, 1 disabled (`BookBuilderIntegrationTest.ProcessSingleDayFile`), 727 total
- **22 integration tests** (14 N=32 + 8 N=128) — labeled `integration`, excluded from default ctest
- Unit test time: ~5 min. Integration: ~20 min.

## What to do next

1. Ship Phase 4 (commit breadcrumbs + changed files).
2. Start Phase 5 (feature-analysis): `source .master-kit.env && ./.kit/tdd.sh red .kit/docs/feature-analysis.md`
3. Or start Phase R1 (subordination-test): `source .master-kit.env && ./.kit/experiment.sh survey .kit/experiments/subordination-test.md`
4. Or start Phase R2 (info-decomposition): `source .master-kit.env && ./.kit/experiment.sh survey .kit/experiments/info-decomposition.md`

## Key files (Phase 4)

| File | Purpose |
|------|---------|
| `src/features/bar_features.hpp` | Track A: 6 categories of hand-crafted features |
| `src/features/raw_representations.hpp` | Track B: book snapshots, message summaries |
| `src/features/feature_export.hpp` | CSV export with metadata + features + returns |
| `src/features/warmup.hpp` | Warm-up state tracking, session boundary resets |

## Build commands

```bash
cmake --build build -j12                                                  # build
cd build && ctest --output-on-failure --label-exclude integration         # unit tests (~5 min)
cd build && ctest --output-on-failure --label-regex integration           # integration tests (~20 min)
```

---

Updated: 2026-02-17
