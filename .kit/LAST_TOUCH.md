# Last Touch — Cold-Start Briefing

## CRITICAL: Orchestrator Protocol

**You are the ORCHESTRATOR. You NEVER write code. You NEVER read source files. You NEVER run verification commands.**

- ALL code is written by kit sub-agents (`.kit/tdd.sh`, `.kit/experiment.sh`, `.kit/math.sh`)
- You ONLY: read/write state `.md` files, run kit phases, check exit codes
- If you need code written → delegate to a kit phase
- If you need something verified → the kit phase already verified it (trust exit 0)
- See `CLAUDE.md` §ABSOLUTE RULES for the full list

## Project Status

**21 phases complete (9 engineering + 12 research). CNN signal confirmed (3rd reproduction, R²=0.089) but NOT economically viable under base costs. Branch: `main`.**

## What was completed this cycle

- **Corrected Hybrid Model (9E)** — `.kit/experiments/hybrid-model-corrected.md`
- **Research kit phases** — frame→run→read all exit 0
- **CNN normalization fix VERIFIED (3rd independent reproduction):**
  - Mean R²=0.089 (proper validation) — matches 9D's 0.084 within +0.005
  - All 5 folds within ±0.015 of 9D reference
  - TICK_SIZE division applied (range [-22.5, 22.5], 100% half-tick quantized)
  - Per-day z-scoring verified (all days mean≈0, std=1.0)
  - Architecture: 12,128 params exactly
- **End-to-end pipeline NOT viable under base costs:**
  - XGBoost accuracy = 0.419 (above random 0.333)
  - Expectancy = -$0.37/trade (base $3.74 RT) — FAIL (needed +$0.50)
  - Profit factor = 0.924 — FAIL (needed 1.50)
  - Gross edge $3.37/trade, breakeven RT = $3.37 ($0.37 short of base costs)
  - Profitable ONLY under optimistic costs: +$0.88/trade at $2.49 RT, PF=1.21
- **Hybrid outperforms GBT-only** (small delta: +0.4pp acc, +$0.075 exp vs GBT-nobook)
- **Key insights:**
  - Regression→frozen-embedding→classification loses information at handoff
  - volatility_50 dominates feature importance (19.9 gain, 2.2× next)
  - CNN embeddings outperform raw book features for XGBoost (CNN = denoiser)
  - Win rate 51.3% vs needed 53.3% — only 2pp gap to breakeven
- **Outcome B: REFUTED** — SC 7/9 PASS, 2 FAIL (SC-4 expectancy, SC-5 profit factor)
- All state files updated (CLAUDE.md, RESEARCH_LOG.md, spec exit criteria)

## What exists

A C++20 MES microstructure model suite with:
- **Infrastructure**: Bar construction, oracle replay, multi-day backtest, feature computation/export, feature analysis, oracle expectancy report, bar feature export, tick bar fix (9 TDD phases)
- **Research results**: 12 complete research phases. CNN spatial signal confirmed (proper-validation R²≈0.089). End-to-end Hybrid pipeline not viable under base costs.
- **Architecture decision**: CNN + GBT Hybrid — signal is REAL but pipeline design is the bottleneck. The regression-to-classification gap prevents viable trading.
- **Labeling decision**: Triple barrier (preferred over first-to-hit)
- **Temporal verdict**: NO TEMPORAL SIGNAL — confirmed across 7 bar types, 0.14s–300s

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
| 9A | `.kit/docs/hybrid-model.md` | TDD | **Done** (C++ TB label export) |
| 9B | `.kit/experiments/hybrid-model-training.md` | Research | **Done (REFUTED)** — normalization wrong |
| 9C | `.kit/experiments/cnn-reproduction-diagnostic.md` | Research | **Done (REFUTED)** — deviations not root cause |
| 9D | `.kit/experiments/r3-reproduction-pipeline-comparison.md` | Research | **Done (CONFIRMED/REFUTED)** — root cause resolved |
| R3b | `.kit/experiments/r3b-event-bar-cnn.md` | Research | **Done (INCONCLUSIVE)** — bar defect |
| TB-Fix | `.kit/docs/tick-bar-fix.md` | TDD | **Done** — tick bars fixed |
| **9E** | **`.kit/experiments/hybrid-model-corrected.md`** | **Research** | **Done (REFUTED — Outcome B)** — CNN R²=0.089, exp=-$0.37 |

## Test summary

- **1003 unit tests** pass, 1 disabled, 1 skipped, 1004 total + tick_bar_fix tests
- **22 integration tests** (14 N=32 + 8 N=128) — labeled `integration`, excluded from default ctest
- Unit test time: ~14 min. Integration: ~20 min.

## What to do next

The CNN spatial signal is real but the pipeline doesn't convert it to viable trading. Five options (in priority order):

### 1. End-to-End CNN Classification (HIGHEST PRIORITY)
Train CNN directly on tb_label (3-class cross-entropy) instead of regression→frozen embedding→XGBoost. Eliminates the regression-to-classification bottleneck identified as the key loss point. The CNN's 16-dim penultimate layer would learn class-discriminative features rather than return-prediction features.

### 2. XGBoost Hyperparameter Tuning (LOW-HANGING FRUIT)
Grid search on max_depth, learning_rate, n_estimators, min_child_weight with 5-fold CV. Current hyperparameters inherited from 9B (broken pipeline era). The 2pp win rate gap is small enough that tuning could close it.

### 3. Label Design Sensitivity (ARCHITECTURAL)
Test alternative triple barrier parameters: wider target (15 ticks), narrower stop (3 ticks). At 15:3 ratio, breakeven win rate drops to ~42.5% — well below current 51.3%.

### 4. CNN at h=1 with Corrected Normalization (EXPLORATORY)
R2 showed signal strongest at h=1. Test with corrected normalization to see if shorter horizon improves classification.

### 5. R3b Rerun with Genuine Tick Bars (INDEPENDENT)
Now that tick bar construction is fixed, rerun event-bar CNN experiment.

### Mental Model Update

- **Before:** "CNN signal is real (R²≈0.084). Corrected normalization should yield viable hybrid trading signals."
- **After:** "CNN signal is real and fully verified (R²=0.089, 3rd reproduction). But the regression→frozen-embedding→classification pipeline loses too much information. Gross edge is $3.37/trade — only $0.37 short of breakeven. The model needs either a better architecture (end-to-end classification), better hyperparameters, or different label design to close the gap."

## Key research results

| Experiment | Finding | Key Number |
|-----------|---------|------------|
| R1 | Subordination refuted for MES | 0/3 primary tests significant |
| R2 | Hand-crafted features sufficient | Best R²=0.0067 (1-bar MLP) |
| R3 | CNN best book encoder | R²=0.132 (leaked), proper R²≈0.084 |
| R4 | No temporal signal (time_5s) | All 36 AR configs negative R² |
| R4b | No temporal signal (event bars) | All AR configs negative R² |
| R4c | Tick + extended horizons null | 0/54+ passes across 4 bar types |
| R4d | Dollar + tick actionable null | 0/38 passes, 7s–300s coverage |
| Synthesis | CNN + GBT Hybrid | CONDITIONAL GO → GO |
| Oracle | TB passes all 6 criteria | $4.00/trade, PF=3.30 |
| 9B | CNN normalization wrong | R²=-0.002 |
| 9C | Deviations not root cause | R²=0.002 |
| 9D | R3 reproduced, root cause resolved | R²=0.1317 (leaked) / 0.084 (proper) |
| R3b | Tick bars are time bars | Peak R²=0.057, all < baseline |
| **9E** | **CNN viable, pipeline not** | **R²=0.089, exp=-$0.37/trade** |

## Build commands

```bash
cmake --build build -j12                                                  # build
cd build && ctest --output-on-failure --label-exclude integration         # unit tests (~14 min)
cd build && ctest --output-on-failure --label-regex integration           # integration tests (~20 min)
./build/oracle_expectancy                                                  # oracle expectancy extraction
./build/bar_feature_export --bar-type <type> --bar-param <param> --output <csv>  # feature export
```

---

Updated: 2026-02-19 (Corrected Hybrid Model — REFUTED Outcome B. CNN R²=0.089 confirmed, expectancy -$0.37/trade. Pipeline bottleneck: regression→classification gap.)
