# Last Touch — Cold-Start Briefing

## CRITICAL: Orchestrator Protocol

**You are the ORCHESTRATOR. You NEVER write code. You NEVER read source files. You NEVER run verification commands.**

- ALL code is written by kit sub-agents (`.kit/tdd.sh`, `.kit/experiment.sh`, `.kit/math.sh`)
- You ONLY: read/write state `.md` files, run kit phases, check exit codes
- If you need code written → delegate to a kit phase
- If you need something verified → the kit phase already verified it (trust exit 0)
- See `CLAUDE.md` §ABSOLUTE RULES for the full list

## Project Status

**22 phases complete (10 engineering + 12 research). Full-year Parquet dataset (251 days, 1.16M bars) exported and in S3. Docker/ECR/EBS cloud pipeline verified end-to-end. CNN signal confirmed (R²=0.089) but NOT economically viable under base costs. Branch: `main`.**

## What was completed this cycle

### Full-Year Parquet Export (2026-02-20)
- **251/251 trading days exported** — 1,160,150 total rows, 0 failures, 0 duplicates
- All 10 success criteria PASS (see `.kit/experiments/full-year-export.md`)
- Schema: 149 columns, zstd compression, 255.7 MB total (3.1× compression)
- Quarter balance: Q1=62, Q2=62, Q3=64, Q4=63
- Wall-clock: 77s (11-way parallel)
- 19/19 reference days validated (max rel_err=4.99e-6, float32 precision)
- Results in S3 artifact store (symlinks in `.kit/results/full-year-export/`)

### Docker + ECR + EBS Cloud Pipeline (2026-02-21)
- **Dockerfile rewritten**: multi-stage build (Ubuntu 22.04 builder → python:3.11-slim runtime), library isolation in `/opt/app-libs/`
- **cmake/Findonnxruntime.cmake** created for pre-built ORT tarball
- **ec2-bootstrap.sh** fixed: awk (not bc), docker pull retry, NVMe fallback, log upload on failure
- **ECR repo**: `651323680805.dkr.ecr.us-east-1.amazonaws.com/mbo-dl`
- **EBS snapshot**: `snap-0efa355754c9a329d` (49GB MBO dataset, 316 files)
- **IAM profile**: `cloud-run-ec2`
- **E2E verified**: EC2 → EBS mount (316 files) → ECR pull → docker run → S3 upload → self-terminate

## What exists

A C++20 MES microstructure model suite with:
- **Infrastructure**: Bar construction, oracle replay, multi-day backtest, feature computation/export, feature analysis, oracle expectancy report, bar feature export, tick bar fix, Parquet export (10 TDD phases)
- **Full-year dataset**: 251 Parquet files in S3 artifact store, ready for model training (13× more data than 19-day baseline)
- **Cloud pipeline**: Docker image in ECR, EBS snapshot with data, IAM role — ready for GPU experiments
- **Research results**: 12 complete research phases. CNN spatial signal confirmed (proper-validation R²≈0.089). End-to-end Hybrid pipeline not viable under base costs.
- **Architecture decision**: CNN + GBT Hybrid — signal is REAL but pipeline design is the bottleneck
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
| 9E | `.kit/experiments/hybrid-model-corrected.md` | Research | **Done (REFUTED — Outcome B)** — CNN R²=0.089, exp=-$0.37 |
| R3b-gen | `.kit/experiments/r3b-genuine-tick-bars.md` | Research | **Done (CONFIRMED low confidence)** — tick_100 R²=0.124, p=0.21 |
| FYE | `.kit/experiments/full-year-export.md` | Research | **Done (CONFIRMED)** — 251 days, 1.16M bars, 10/10 SC pass |
| Infra | Dockerfile + ec2-bootstrap | Chore | **Done** — Docker/ECR/EBS pipeline verified E2E |

## Test summary

- **1003 unit tests** pass, 1 disabled, 1 skipped, 1004 total + tick_bar_fix tests
- **22 integration tests** (14 N=32 + 8 N=128) — labeled `integration`, excluded from default ctest
- **28 Parquet export tests** — all pass
- Unit test time: ~14 min. Integration: ~20 min.

## What to do next

**The full-year dataset (13× more data) and cloud pipeline are now ready.** The CNN spatial signal is real but the pipeline doesn't convert it to viable trading. Five options (in priority order):

### 1. End-to-End CNN Classification (HIGHEST PRIORITY)
Train CNN directly on tb_label (3-class cross-entropy) instead of regression→frozen embedding→XGBoost. Eliminates the regression-to-classification bottleneck identified as the key loss point. **Now unblocked by full-year dataset (1.16M bars vs 87K)** — more data should improve generalization. Cloud pipeline ready for GPU training.

### 2. XGBoost Hyperparameter Tuning (LOW-HANGING FRUIT)
Grid search on max_depth, learning_rate, n_estimators, min_child_weight with 5-fold CV. Current hyperparameters inherited from 9B (broken pipeline era). The 2pp win rate gap is small enough that tuning could close it.

### 3. Label Design Sensitivity (ARCHITECTURAL)
Test alternative triple barrier parameters: wider target (15 ticks), narrower stop (3 ticks). At 15:3 ratio, breakeven win rate drops to ~42.5% — well below current 51.3%.

### 4. CNN at h=1 with Corrected Normalization (EXPLORATORY)
R2 showed signal strongest at h=1. Test with corrected normalization to see if shorter horizon improves classification.

### 5. R3b Rerun with Genuine Tick Bars (INDEPENDENT)
Now that tick bar construction is fixed, rerun event-bar CNN experiment. R3b-genuine showed tick_100 R²=0.124 but p=0.21 — needs multi-seed replication.

### Mental Model Update

- **Before:** "CNN signal is real (R²≈0.089). Pipeline not viable. Limited to 19-day dataset."
- **After:** "Full-year dataset (251 days, 1.16M bars) and cloud GPU pipeline ready. End-to-end CNN classification on full year is the highest-priority next experiment — 13× more training data + eliminating the regression-to-classification bottleneck."

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
| 9E | CNN viable, pipeline not | R²=0.089, exp=-$0.37/trade |
| R3b-gen | Genuine tick bars promising | tick_100 R²=0.124, p=0.21 |
| **FYE** | **Full-year export production-ready** | **251 days, 1.16M bars, 10/10 SC** |

## Build commands

```bash
cmake --build build -j12                                                  # build
cd build && ctest --output-on-failure --label-exclude integration         # unit tests (~14 min)
cd build && ctest --output-on-failure --label-regex integration           # integration tests (~20 min)
./build/oracle_expectancy                                                  # oracle expectancy extraction
./build/bar_feature_export --bar-type <type> --bar-param <param> --output <csv>  # feature export
```

---

Updated: 2026-02-21 (Full-year Parquet export CONFIRMED — 251 days, 1.16M bars. Docker/ECR/EBS cloud pipeline verified E2E.)
