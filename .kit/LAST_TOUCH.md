# Last Touch — Cold-Start Briefing

## Project Status

**11 phases complete (6 engineering + 5 research).** Oracle expectancy report layer added (Phase 7). Research verdict: **CONDITIONAL GO** — CNN + GBT Hybrid architecture recommended. Oracle expectancy open question partially resolved (testable layer done, standalone tool next).

## What was completed this cycle

- **Phase 7 (oracle-expectancy)** — TDD cycle for `OracleExpectancyReport` struct, `to_json` JSON serializer, and `aggregate_day_results` per-quarter aggregation logic. 67 new unit tests. Resolves synthesis open question #1 (testable layer).
- New: `src/backtest/oracle_expectancy_report.hpp`
- New: `tests/oracle_expectancy_test.cpp`
- New: `.kit/docs/oracle-expectancy.md` (spec)
- Modified: `CMakeLists.txt`, `src/backtest/multi_day_runner.hpp`, `src/backtest/oracle_replay.hpp`, `src/serialization.hpp`, `tests/bar_features_test.cpp`, `tests/test_bar_helpers.hpp`

## What exists

A C++20 MES microstructure model suite with:
- **Infrastructure**: Bar construction, oracle replay, multi-day backtest, feature computation/export, feature analysis, oracle expectancy report (6 TDD phases)
- **Research results**: Subordination test, info decomposition, book encoder bias, temporal predictability, synthesis (5 research phases)
- **Architecture decision**: CNN + GBT Hybrid — Conv1d on raw (20,2) book → 16-dim embedding → concat with ~20 non-spatial features → XGBoost

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

## Test summary

- **953 unit tests** pass, 1 disabled (`BookBuilderIntegrationTest.ProcessSingleDayFile`), 954 total
- **22 integration tests** (14 N=32 + 8 N=128) — labeled `integration`, excluded from default ctest
- Unit test time: ~6 min. Integration: ~20 min.

## What to do next

1. Build `tools/oracle_expectancy.cpp` standalone executable to run oracle on real MES data (20 stratified days)
2. Test CNN at h=1 (R3 only tested h=5)
3. Design CNN+GBT integration pipeline
4. Estimate transaction costs
5. Proceed to model architecture build spec

## Key research results

| Experiment | Finding | Key Number |
|-----------|---------|------------|
| R1 | Subordination refuted for MES | 0/3 primary tests significant |
| R2 | Hand-crafted features sufficient | Best R²=0.0067 (1-bar MLP) |
| R3 | CNN best book encoder | R²=0.132, CNN>Attention p=0.042 |
| R4 | No temporal signal | All 36 AR configs negative R² |
| Synthesis | CNN + GBT Hybrid | CONDITIONAL GO |

## Build commands

```bash
cmake --build build -j12                                                  # build
cd build && ctest --output-on-failure --label-exclude integration         # unit tests (~6 min)
cd build && ctest --output-on-failure --label-regex integration           # integration tests (~20 min)
```

---

Updated: 2026-02-17
