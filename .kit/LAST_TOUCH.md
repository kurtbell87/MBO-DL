# Last Touch — Cold-Start Briefing

**Read this file first. Then read `CLAUDE.md` for the full protocol. Do not read source files.**

---

## CRITICAL: Orchestrator Protocol

**You are the ORCHESTRATOR. You NEVER write code. You NEVER read source files. You NEVER run verification commands.**

- ALL code is written by kit sub-agents (`.kit/tdd.sh`, `.kit/experiment.sh`, `.kit/math.sh`)
- You ONLY: read/write state `.md` files, launch kit phases (MCP tools or bash), check exit codes via dashboard
- If you need code written → delegate to a kit phase
- If you need something verified → the kit phase already verified it (trust exit 0)
- See `CLAUDE.md` §ABSOLUTE RULES for the full list of things you must never do

---

## TL;DR — Where We Are and What To Do

We've built a complete MES microstructure research platform over 24 phases. The CNN spatial signal on order book snapshots is **real and reproducible** (R²=0.089, 3 independent reproductions). But the prior pipeline (CNN regression → frozen embeddings → XGBoost classification) loses too much information at the handoff — expectancy is -$0.37/trade, just $0.37 short of breakeven.

**Your job: launch the end-to-end CNN classification experiment.**

The spec is ready: `.kit/experiments/e2e-cnn-classification.md`

This experiment trains the CNN directly on 3-class triple barrier labels (CrossEntropyLoss) instead of going through a regression intermediary. It runs on the full-year dataset (251 days, 1.16M bars — 13× more data than prior experiments) with a rigorous CPCV validation protocol that generates 9 independent backtest paths and computes Probability of Backtest Overfitting.

```bash
# To launch:
source .orchestration-kit.env
# Then use MCP tool:
kit.research_full question="Can end-to-end CNN classification on tb_label close the viability gap?" spec_path=".kit/experiments/e2e-cnn-classification.md"
# Or bash fallback:
.kit/experiment.sh full "Can end-to-end CNN classification on tb_label close the viability gap?" .kit/experiments/e2e-cnn-classification.md
```

---

## Project Status

**24 phases complete (10 engineering + 12 research + 1 data export + 1 infra). Branch: `main`. Working tree: clean.**

### What's Built
- **C++20 data pipeline**: Bar construction, order book replay, multi-day backtest, feature computation/export, oracle expectancy, Parquet export. 1003+ unit tests, 22 integration tests, 28 Parquet tests.
- **Full-year dataset**: 251 Parquet files (time_5s bars, 1,160,150 bars, 149 columns, zstd compression). Stored in S3 artifact store — run `orchestration-kit/tools/artifact-store hydrate` after clone/worktree to restore.
- **Cloud pipeline**: Docker image in ECR (`651323680805.dkr.ecr.us-east-1.amazonaws.com/mbo-dl`), EBS snapshot with 49GB MBO data (`snap-0efa355754c9a329d`), IAM profile (`cloud-run-ec2`). Verified end-to-end on 2026-02-21.
- **Research conclusions**: See "Key Research Results" below.

### The Core Problem Being Solved

MES (Micro E-mini S&P 500 futures) microstructure prediction. The order book has a real spatial signal — a Conv1d CNN on the 20-level bid/ask snapshot predicts 5-bar forward returns with R²=0.089. But converting that regression signal into a profitable 3-class trading decision (long/flat/short via triple barrier labels) has failed so far because the regression→embedding→classification pipeline loses information.

The hypothesis: training the CNN end-to-end on the classification objective (CrossEntropyLoss on tb_label) will learn class-discriminative spatial features instead of return-variance features, closing the 2pp win rate gap needed for breakeven.

---

## The Experiment Spec: What's In It

**`.kit/experiments/e2e-cnn-classification.md`** — 576 lines, fully specified. Key design decisions:

### Validation Protocol (most important part)
- **50-day holdout** (days 202–251): sacred, touched exactly once at the end
- **CPCV** (Combinatorial Purged Cross-Validation): N=10 groups, k=2 → 45 train/test splits → 9 independent backtest paths
- **Purging**: 500 bars at each fold boundary (prevents triple barrier label leakage)
- **Embargo**: 4,600 bars (~1 day) additional buffer for serial correlation
- **Internal val split**: 80/20 within training fold, also purged (prevents the early-stopping leakage bug from R3)
- **PBO + Deflated Sharpe Ratio**: quantifies overfitting probability
- **Walk-forward sanity check**: 4 expanding-window folds to validate CPCV isn't inflating results
- **Per-regime reporting**: Q1-Q4 breakdown

### Model Configs (3 levels)
1. **E2E-CNN** (primary): Conv1d(2→59→59) + BN + ReLU ×2 → Pool → Linear(59→16) → Linear(16→3). CrossEntropyLoss on tb_label.
2. **E2E-CNN + Features**: Same encoder, but concatenate 16-dim embedding with 20 non-spatial features → Linear(36→3).
3. **GBT-only** (baseline): XGBoost on 20 hand-crafted features (reproduces 9E baseline on full-year data).

### Success Criteria (12 total)
- SC-1: accuracy >= 0.42 (above 9E's 0.419)
- SC-2: expectancy >= $0.00/trade (at least breakeven — 9E was -$0.37)
- SC-3: PBO < 0.50 (not overfit)
- SC-4 through SC-12: ablations, holdout, regime analysis, confusion matrix, etc.

### Decision Rules
- **Outcome A** (breaks even): proceed to multi-seed robustness, then paper trading pipeline
- **Outcome B** (learns but doesn't break even): try label design sensitivity (wider target)
- **Outcome C** (fails to learn): signal can't be monetized through classification at these barriers
- **Outcome D** (GBT beats CNN): simplify to GBT-only, CNN adds nothing for classification
- **Outcome E** (overfit): reduce complexity, larger embargo

---

## Normalization Protocol (CRITICAL — institutional memory)

Three prior experiments (9B, 9C, R3b) failed because of normalization errors. The correct protocol, verified in 9D and 9E:

1. **Book prices (channel 0)**: Divide by TICK_SIZE=0.25 → integer tick offsets. Do NOT z-score.
2. **Book sizes (channel 1)**: log1p() → z-score PER DAY (not per-fold, not globally).
3. **Non-spatial features**: z-score using train-fold stats only.

If you see the sub-agent z-scoring channel 0 or using per-fold z-scoring on sizes, that's the bug that cost us 3 experiment cycles.

---

## Key Research Results

| Experiment | Finding | Key Number | Implication |
|-----------|---------|------------|-------------|
| R1 | Subordination refuted | 0/3 significant | Time bars are the baseline |
| R2 | Features sufficient | R²=0.0067 | Book snapshot is sufficient statistic |
| R3 | CNN best encoder | R²=0.132 (leaked) / 0.084 (proper) | Spatial structure matters |
| R4/R4b/R4c/R4d | No temporal signal | 0/168+ passes | Drop SSM/temporal encoder permanently |
| R6 | Synthesis | CONDITIONAL GO → GO | CNN + GBT Hybrid architecture |
| R7 | Oracle expectancy | $4.00/trade, PF=3.30 | Edge exists at oracle level |
| 9B→9D | Normalization root cause | R²: -0.002 → 0.089 | TICK_SIZE + per-day z-score required |
| **9E** | **Pipeline bottleneck** | **exp=-$0.37/trade** | **Regression→classification gap is the limit** |
| FYE | Full-year export | 251 days, 1.16M bars | 13× more data, unblocks next experiment |

---

## Phase Sequence (complete history)

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
| R4b | `.kit/experiments/temporal-predictability-event-bars.md` | Research | **Done** (NO SIGNAL) |
| R4c | `.kit/experiments/temporal-predictability-completion.md` | Research | **Done** (ALL NULLS) |
| R4d | `.kit/experiments/temporal-predictability-dollar-tick-actionable.md` | Research | **Done** (CONFIRMED) |
| 9A | `.kit/docs/hybrid-model.md` | TDD | **Done** |
| 9B | `.kit/experiments/hybrid-model-training.md` | Research | **Done** (REFUTED) |
| 9C | `.kit/experiments/cnn-reproduction-diagnostic.md` | Research | **Done** (REFUTED) |
| 9D | `.kit/experiments/r3-reproduction-pipeline-comparison.md` | Research | **Done** (ROOT CAUSE FOUND) |
| R3b | `.kit/experiments/r3b-event-bar-cnn.md` | Research | **Done** (INCONCLUSIVE) |
| TB-Fix | `.kit/docs/tick-bar-fix.md` | TDD | **Done** |
| 9E | `.kit/experiments/hybrid-model-corrected.md` | Research | **Done** (REFUTED — Outcome B) |
| R3b-gen | `.kit/experiments/r3b-genuine-tick-bars.md` | Research | **Done** (LOW CONFIDENCE) |
| FYE | `.kit/experiments/full-year-export.md` | Research | **Done** (CONFIRMED) |
| Infra | Dockerfile + ec2-bootstrap | Chore | **Done** |
| **→ E2E** | **`.kit/experiments/e2e-cnn-classification.md`** | **Research** | **READY TO LAUNCH** |

---

## State Files (read order)

1. **This file** — you're here
2. **`CLAUDE.md`** — full protocol, absolute rules, current state, institutional memory
3. **`.kit/RESEARCH_LOG.md`** — cumulative findings from all 12+ experiments
4. **`.kit/QUESTIONS.md`** — open and answered research questions
5. **`.kit/experiments/e2e-cnn-classification.md`** — the next experiment spec (576 lines, fully specified)

## Build Commands (for reference — sub-agents use these, not you)

```bash
cmake --build build -j12                                                  # build
cd build && ctest --output-on-failure --label-exclude integration         # unit tests (~14 min)
./build/bar_feature_export --bar-type time --bar-param 5 --output out.parquet  # feature export
```

---

Updated: 2026-02-21. Next action: launch E2E CNN classification experiment (`e2e-cnn-classification.md`).
