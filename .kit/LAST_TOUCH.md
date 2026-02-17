# Last Touch — Cold-Start Briefing

## Project Status

**All 10 phases complete.** 5 engineering phases (TDD) + 5 research phases (R1–R4, synthesis) done. Research verdict: **CONDITIONAL GO** — CNN + GBT Hybrid architecture recommended. All results merged to main.

## What was completed this cycle

- **R1 (subordination-test)** — REFUTED. Clark/Ané-Geman subordination doesn't hold for MES. No bar type superiority. PR #8.
- **R2 (info-decomposition)** — FEATURES SUFFICIENT. Raw book snapshot captures all linear/MLP-extractable signal. No CNN/message/SSM stages justified from flattened features. PR #9.
- **R3 (book-encoder-bias)** — CNN WINS. Conv1d on structured (20,2) book input: R²=0.132, significantly beats Attention (p=0.042). PR #10.
- **R4 (temporal-predictability)** — NO TEMPORAL SIGNAL. MES returns are martingale at 5s bars. All 36 AR configs negative R². Drop temporal encoder. PR #11.
- **Phase 6 (synthesis)** — CONDITIONAL GO. CNN + GBT Hybrid. Spatial encoder included (R3), message/temporal dropped (R2+R4). Bar type: time_5s. Horizons: h=1, h=5. PR #12.
- Fixed broken symlinks: `.kit/experiment.sh`, `.kit/math.sh` (pointed to old `claude-research-kit/`, `claude-mathematics-kit/`).
- Added `.gitignore` entries for large experiment artifacts (`*.bin`, `features.csv`).

## What exists

A C++20 MES microstructure model suite with:
- **Infrastructure**: Bar construction, oracle replay, multi-day backtest, feature computation/export, feature analysis (5 TDD phases)
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

## Test summary

- **886 unit tests** pass, 1 disabled (`BookBuilderIntegrationTest.ProcessSingleDayFile`), 887 total
- **22 integration tests** (14 N=32 + 8 N=128) — labeled `integration`, excluded from default ctest
- Unit test time: ~6 min. Integration: ~20 min.

## What to do next

Synthesis recommends resolving open questions before model build:
1. Extract oracle expectancy from Phase 3 C++ test output
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
