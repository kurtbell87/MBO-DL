# Last Touch — Cold-Start Briefing

## Project Status

**12 phases complete (7 engineering + 5 research).** Oracle expectancy extracted on 19 real MES days. **VERDICT: GO.** Triple barrier passes all 6 success criteria. CONDITIONAL GO upgraded to full GO.

## What was completed this cycle

- **Oracle expectancy extraction** — Wrote and ran `tools/oracle_expectancy.cpp` on 20 stratified days (19 valid, 1 skipped: 20221230 no data). Both FIRST_TO_HIT and TRIPLE_BARRIER oracle modes tested.
- New: `tools/oracle_expectancy.cpp`
- Modified: `CMakeLists.txt` (added oracle_expectancy build target)
- Output: `.kit/results/oracle-expectancy/metrics.json`, `.kit/results/oracle-expectancy/summary.json`

## Key oracle expectancy results

| Method | Trades | Expectancy | Profit Factor | Win Rate | Sharpe | Net PnL | Verdict |
|--------|--------|------------|---------------|----------|--------|---------|---------|
| **First-to-Hit** | 5,369 | $1.56/trade | 2.11 | 53.2% | 0.136 | $8,369 | 5/6 pass (DD fail) |
| **Triple Barrier** | 4,873 | **$4.00/trade** | **3.30** | **64.3%** | **0.362** | **$19,479** | **ALL 6 PASS** |

**Triple barrier is the preferred labeling method.** Passes all criteria from TRAJECTORY §9.4: expectancy>$0.50, PF>1.3, WR>45%, net PnL>0, DD<50×expectancy, trades/day>10.

Per-quarter stability (TB): Q1=$5.39, Q2=$3.16, Q3=$3.41, Q4=$3.39 — positive expectancy in all 4 quarters.

## What exists

A C++20 MES microstructure model suite with:
- **Infrastructure**: Bar construction, oracle replay, multi-day backtest, feature computation/export, feature analysis, oracle expectancy report (7 TDD phases)
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

## Test summary

- **953 unit tests** pass, 1 disabled (`BookBuilderIntegrationTest.ProcessSingleDayFile`), 954 total
- **22 integration tests** (14 N=32 + 8 N=128) — labeled `integration`, excluded from default ctest
- Unit test time: ~6 min. Integration: ~20 min.

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
cd build && ctest --output-on-failure --label-exclude integration         # unit tests (~6 min)
cd build && ctest --output-on-failure --label-regex integration           # integration tests (~20 min)
./build/oracle_expectancy                                                  # oracle expectancy extraction
```

---

Updated: 2026-02-17
