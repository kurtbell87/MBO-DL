# Last Touch — Cold-Start Briefing

## Project Status

**14 phases complete (8 engineering + 6 research).** Oracle expectancy extracted on 19 real MES days. **VERDICT: GO.** Triple barrier passes all 6 success criteria. R4b confirms NO TEMPORAL SIGNAL across all bar types. All research phases complete.

## What was completed this cycle

- **R4b Phase E**: Dollar_25k R4 analysis run on AWS EC2 spot instance (c7a.8xlarge, ~27 min). All 36 Tier 1 + 24 Tier 2 configs computed on 3.1M rows.
- **R4b Phase F**: Cross-bar comparison analysis written. Dollar_25k has marginal AR R² at h=1 (R²=0.0006) but temporal augmentation fails dual threshold. Temporal-Only has standalone power (R²=0.012) but is redundant with static features.
- **Cloud-run bug fixes**: Fixed spot instance flag being silently overridden by preflight; switched to uv for Python package management; upgraded Docker images to Python 3.12 (EC2) and 3.11 (RunPod).
- **State updates**: RESEARCH_LOG, CLAUDE.md, exit criteria all updated. All 16 exit criteria checked off.

## What exists

A C++20 MES microstructure model suite with:
- **Infrastructure**: Bar construction, oracle replay, multi-day backtest, feature computation/export, feature analysis, oracle expectancy report, bar feature export (8 TDD phases)
- **Research results**: Subordination test, info decomposition, book encoder bias, temporal predictability (time_5s + event bars), synthesis, oracle expectancy (6 complete research phases)
- **Architecture decision**: CNN + GBT Hybrid — Conv1d on raw (20,2) book -> 16-dim embedding -> concat with ~20 non-spatial features -> XGBoost
- **Labeling decision**: Triple barrier (preferred over first-to-hit)
- **Temporal verdict**: NO TEMPORAL SIGNAL — confirmed across time_5s, volume_100, and dollar_25k bars. Drop SSM/temporal encoder.

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

## Test summary

- **1003 unit tests** pass, 1 disabled, 1 skipped, 1004 total
- **22 integration tests** (14 N=32 + 8 N=128) — labeled `integration`, excluded from default ctest
- Unit test time: ~14 min. Integration: ~20 min.

## What to do next

### Immediate: Model architecture build spec

All research phases are complete. Proceed to the CNN + GBT Hybrid model architecture build:
- Conv1d on raw (20,2) book -> 16-dim embedding
- Concat with ~20 non-spatial features
- XGBoost classifier with triple barrier labels
- Bar type: time_5s, horizons h=1 and h=5

## Key research results

| Experiment | Finding | Key Number |
|-----------|---------|------------|
| R1 | Subordination refuted for MES | 0/3 primary tests significant |
| R2 | Hand-crafted features sufficient | Best R²=0.0067 (1-bar MLP) |
| R3 | CNN best book encoder | R²=0.132, CNN>Attention p=0.042 |
| R4 | No temporal signal (time_5s) | All 36 AR configs negative R² |
| R4b vol100 | No temporal signal (volume bars) | All 36 AR configs negative R² |
| R4b dollar25k | Marginal signal, redundant | AR R²=0.0006 at h=1, augmentation fails |
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

Updated: 2026-02-18 (R4b complete, all research phases done)
