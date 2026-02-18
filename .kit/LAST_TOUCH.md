# Last Touch — Cold-Start Briefing

## Project Status

**14 phases complete (8 engineering + 6 research).** Oracle expectancy extracted on 19 real MES days. **VERDICT: GO.** All research phases complete. Model architecture build spec written.

## What was completed this cycle

- **Hybrid model build spec written**: `.kit/docs/hybrid-model.md` — comprehensive TDD spec for the CNN+GBT Hybrid model pipeline.
  - Phase A: C++ data export extension (add triple barrier labels to `bar_feature_export`)
  - Phase B: Python CNN encoder training (Conv1d on (20,2) book → 16-dim embedding)
  - Phase C: Python XGBoost classification (embeddings + ~20 non-spatial features → TB labels)
  - Phase D: 5-fold expanding window cross-validation
  - Ablation comparisons (GBT-only, CNN-only baselines)
  - Transaction cost sensitivity (3 scenarios)
  - 17 exit criteria, 26 test cases

## What exists

A C++20 MES microstructure model suite with:
- **Infrastructure**: Bar construction, oracle replay, multi-day backtest, feature computation/export, feature analysis, oracle expectancy report, bar feature export (8 TDD phases)
- **Research results**: Subordination test, info decomposition, book encoder bias, temporal predictability (time_5s + event bars + dollar/tick actionable), synthesis, oracle expectancy (8 complete research phases)
- **Architecture decision**: CNN + GBT Hybrid — Conv1d on raw (20,2) book -> 16-dim embedding -> concat with ~20 non-spatial features -> XGBoost
- **Labeling decision**: Triple barrier (preferred over first-to-hit)
- **Temporal verdict**: NO TEMPORAL SIGNAL — confirmed across 7 bar types, 0.14s–300s. Drop SSM/temporal encoder.

## Phase Sequence

| # | Spec | Kit | Status |
|---|------|-----|--------|
| 1 | `.kit/docs/bar-construction.md` | TDD | **Done** |
| 2 | `.kit/docs/oracle-replay.md` | TDD | **Done** |
| 3 | `.kit/docs/multi-day-backtest.md` | TDD | **Done** |
| R1 | `.kit/experiments/subordination-test.md` | Research | **Done** (REFUTED) |
| 4 | `.kit/docs/feature-computation.md` | TDD | **Done** |
| 5 | `.kit/docs/feature-analysis.md` | TDD | **Done** |
| R2 | `.kit/experiments/info-decomposition.md` | Research | **Done** (FEATURES SUFFICIENT) |
| R3 | `.kit/experiments/book-encoder-bias.md` | Research | **Done** (CNN WINS) |
| R4 | `.kit/experiments/temporal-predictability.md` | Research | **Done** (NO SIGNAL) |
| 6 | `.kit/experiments/synthesis.md` | Research | **Done** (CONDITIONAL GO) |
| 7 | `.kit/docs/oracle-expectancy.md` | TDD | **Done** |
| 7b | `tools/oracle_expectancy.cpp` | Research | **Done** (GO) |
| 8 | `.kit/docs/bar-feature-export.md` | TDD | **Done** |
| R4b | `.kit/experiments/temporal-predictability-event-bars.md` | Research | **Done** (NO SIGNAL — robust) |
| R4c | `.kit/experiments/temporal-predictability-completion.md` | Research | **Done** (CONFIRMED — all nulls) |
| R4d | `.kit/experiments/temporal-predictability-dollar-tick-actionable.md` | Research | **Done** (CONFIRMED) |
| **9** | **`.kit/docs/hybrid-model.md`** | **TDD** | **Spec written — ready for implementation** |

## Test summary

- **1003 unit tests** pass, 1 disabled, 1 skipped, 1004 total
- **22 integration tests** (14 N=32 + 8 N=128) — labeled `integration`, excluded from default ctest
- Unit test time: ~14 min. Integration: ~20 min.

## What to do next

### Immediate: Implement hybrid model spec

Run the TDD phases for `.kit/docs/hybrid-model.md`:

1. **Red**: Extend `bar_feature_export.cpp` with triple barrier labels — write failing tests first.
2. **Green**: Make tests pass.
3. **Refactor**: Clean up.
4. **Ship**: Verify existing tests still pass.
5. Then proceed to Python training pipeline (Phases B–D in the spec).

### Implementation order

1. C++ extension: Add TB labels to `bar_feature_export` (tests 1–10)
2. Python CNN encoder + training (tests 11–15)
3. Python data loading + normalization (tests 16–19)
4. Python XGBoost training (tests 20–21)
5. Python evaluation pipeline (tests 22–26)
6. Run full 5-fold CV, collect results
7. Write analysis document

## Key research results

| Experiment | Finding | Key Number |
|-----------|---------|------------|
| R1 | Subordination refuted for MES | 0/3 primary tests significant |
| R2 | Hand-crafted features sufficient | Best R²=0.0067 (1-bar MLP) |
| R3 | CNN best book encoder | R²=0.132, CNN>Attention p=0.042 |
| R4 | No temporal signal (time_5s) | All 36 AR configs negative R² |
| R4b vol100 | No temporal signal (volume bars) | All 36 AR configs negative R² |
| R4b dollar25k | Marginal signal, redundant | AR R²=0.0006 at h=1, augmentation fails |
| R4c | Tick + extended horizons null | 0/54+ passes across 4 bar types |
| R4d | Dollar + tick actionable null | 0/38 passes, 7s–300s coverage |
| Synthesis | CNN + GBT Hybrid | CONDITIONAL GO -> **GO** |
| Oracle Expectancy | TB passes all 6 criteria | $4.00/trade, PF=3.30 |

## Build commands

```bash
cmake --build build -j12                                                  # build
cd build && ctest --output-on-failure --label-exclude integration         # unit tests (~14 min)
cd build && ctest --output-on-failure --label-regex integration           # integration tests (~20 min)
./build/oracle_expectancy                                                  # oracle expectancy extraction
./build/bar_feature_export --bar-type <type> --bar-param <param> --output <csv>  # feature export
```

---

Updated: 2026-02-18 (hybrid model build spec written)
