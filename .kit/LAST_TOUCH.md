# Last Touch — Cold-Start Briefing

## Project Status

**13 phases complete (8 engineering + 5 research).** Oracle expectancy extracted on 19 real MES days. **VERDICT: GO.** Triple barrier passes all 6 success criteria. CONDITIONAL GO upgraded to full GO.

## What was completed this cycle

- **bar-feature-export TDD cycle** — CLI tool `tools/bar_feature_export.cpp` that exports bar-level feature CSVs for arbitrary bar types. Parameterized variant of `info_decomposition_export.cpp` with `--bar-type`, `--bar-param`, `--output` CLI args.
- **Refactoring** — Dead code removal (`t_cdf`, `exit_reason_name`, unused `<numeric>` include), `push_back` loops → `vector::insert`, `weighted_imbalance` → `static`.
- New files: `tools/bar_feature_export.cpp`, `tests/bar_feature_export_test.cpp`
- Modified: `CMakeLists.txt`, `src/features/bar_features.hpp`, `src/backtest/multi_day_runner.hpp`, `src/backtest/oracle_expectancy_report.hpp`, `src/analysis/statistical_tests.hpp`

## Key oracle expectancy results

| Method | Trades | Expectancy | Profit Factor | Win Rate | Sharpe | Net PnL | Verdict |
|--------|--------|------------|---------------|----------|--------|---------|---------|
| **First-to-Hit** | 5,369 | $1.56/trade | 2.11 | 53.2% | 0.136 | $8,369 | 5/6 pass (DD fail) |
| **Triple Barrier** | 4,873 | **$4.00/trade** | **3.30** | **64.3%** | **0.362** | **$19,479** | **ALL 6 PASS** |

**Triple barrier is the preferred labeling method.** Passes all criteria from TRAJECTORY §9.4: expectancy>$0.50, PF>1.3, WR>45%, net PnL>0, DD<50×expectancy, trades/day>10.

Per-quarter stability (TB): Q1=$5.39, Q2=$3.16, Q3=$3.41, Q4=$3.39 — positive expectancy in all 4 quarters.

## What exists

A C++20 MES microstructure model suite with:
- **Infrastructure**: Bar construction, oracle replay, multi-day backtest, feature computation/export, feature analysis, oracle expectancy report, bar feature export (8 TDD phases)
- **Research results**: Subordination test, info decomposition, book encoder bias, temporal predictability, synthesis, oracle expectancy (6 research phases)
- **Architecture decision**: CNN + GBT Hybrid — Conv1d on raw (20,2) book → 16-dim embedding → concat with ~20 non-spatial features → XGBoost
- **Labeling decision**: Triple barrier (preferred over first-to-hit)

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

## Test summary

- **1003 unit tests** pass, 1 disabled (`BookBuilderIntegrationTest.ProcessSingleDayFile`), 1 skipped (`MetadataColumnsTest.BarTypeReflectsCLIArgVolume`), 1004 total
- **22 integration tests** (14 N=32 + 8 N=128) — labeled `integration`, excluded from default ctest
- Unit test time: ~14 min. Integration: ~20 min.

## What to do next

**Proceed to model architecture build spec.** All research prerequisites are resolved:
1. Oracle expectancy: **GO** (TB $4.00/trade, all criteria pass)
2. Architecture: CNN + GBT Hybrid (R3 CNN R²=0.132)
3. Bar type: time_5s (R1 refuted event-driven bars)
4. Labeling: Triple barrier (preferred over first-to-hit)
5. Feature set: ~20 non-spatial features + raw (20,2) book (R2 + R3)
6. Temporal: none (R4 no signal)

**Remaining open questions:**
- CNN at h=1 (R3 tested at h=5 only; synthesis flagged this)
- Transaction cost model refinement (current: fixed spread=1 tick, commission=$0.62/side)
- CNN+GBT integration pipeline design (training loop, embedding extraction)

## Key research results

| Experiment | Finding | Key Number |
|-----------|---------|------------|
| R1 | Subordination refuted for MES | 0/3 primary tests significant |
| R2 | Hand-crafted features sufficient | Best R²=0.0067 (1-bar MLP) |
| R3 | CNN best book encoder | R²=0.132, CNN>Attention p=0.042 |
| R4 | No temporal signal | All 36 AR configs negative R² |
| Synthesis | CNN + GBT Hybrid | CONDITIONAL GO → **GO** |
| Oracle Expectancy | TB passes all 6 criteria | $4.00/trade, PF=3.30 |

## Build commands

```bash
cmake --build build -j12                                                  # build
cd build && ctest --output-on-failure --label-exclude integration         # unit tests (~14 min)
cd build && ctest --output-on-failure --label-regex integration           # integration tests (~20 min)
./build/oracle_expectancy                                                  # oracle expectancy extraction
./build/bar_feature_export --bar-type time --bar-param 5.0 --output out.csv  # bar feature export
```

---

Updated: 2026-02-17 (bar-feature-export TDD cycle)
