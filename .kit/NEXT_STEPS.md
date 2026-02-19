# Next Task: R3b Rerun — CNN on Genuine Tick Bars (Extended Sweep to 10,000)

## FIRST: Create a New Branch

**BEFORE you read ANY files, explore the codebase, or do anything else — create a worktree and branch.**

```bash
git worktree add ../MBO-DL-r3b-genuine-tick -b experiment/r3b-genuine-tick-bars main
cd ../MBO-DL-r3b-genuine-tick
source .orchestration-kit.env
```

Then symlink the kit scripts (worktrees don't inherit them):
```bash
cd .kit
ln -sf ../orchestration-kit/research-kit/experiment.sh experiment.sh
ln -sf ../orchestration-kit/tdd-kit/tdd.sh tdd.sh
ln -sf ../orchestration-kit/mathematics-kit/math.sh math.sh
cd ..
```

**All work happens on this branch. Never touch main directly.**

---

## Context: What Happened and Why We're Doing This

### The Story So Far

1. **R3** proved CNN spatial signal is real on time_5s bars (R²=0.132, later corrected to R²=0.084 with proper validation in 9D).
2. **R3b (original)** attempted to test CNN on tick bars but was **INCONCLUSIVE** — the C++ `bar_feature_export` tick bar construction was broken. It counted fixed-rate book snapshots (10/s), not trade events. So "tick_100" was actually "time_10s", "tick_500" was "time_50s", etc. All tick bar results are void.
3. **TB-Fix TDD cycle** (merged to main, PR #19) fixed tick bar construction. `book_builder.hpp` now emits `trade_count` per snapshot (counts action='T' MBO events). `tick_bar_builder.hpp` accumulates trade counts and closes bars at threshold with remainder carry-over.
4. **9E (hybrid-model-corrected)** confirmed CNN R²=0.089 on time_5s with corrected normalization (TICK_SIZE ÷0.25, per-day z-scoring, proper 80/20 validation). The CNN signal is real but doesn't survive $3.74 RT costs (expectancy=-$0.37). It IS profitable at $2.49 RT (expectancy=+$0.88).

### Why R3b Rerun Matters

The economic viability gap is thin ($1.25/RT = 1 tick of slippage). If genuine tick bars produce CNN R² meaningfully above 0.089, it could flip the hybrid model to profitability even at base costs. The hypothesis: activity-normalized bars (where each bar represents the same amount of market activity) give the CNN a more homogeneous prediction task, potentially boosting spatial signal quality.

**This has never been tested with genuine tick bars.** The original R3b results are void.

---

## What to Do

### 1. Read State Files (after branching)

Read in this order:
- `CLAUDE.md` (project rules — you are the ORCHESTRATOR, you never write code)
- `.kit/LAST_TOUCH.md` (current state)
- `.kit/RESEARCH_LOG.md` (institutional memory)

### 2. Write a New Experiment Spec

Write a new spec at `.kit/experiments/r3b-genuine-tick-bars.md`. **Do NOT reuse the old R3b spec** — it was designed for the broken tick bars and had only 4 thresholds topping out at ~10,000 ticks in the XL bracket.

The new spec should be a **comprehensive sweep of genuine tick bar thresholds** with the following key parameters:

#### Tick Thresholds (8 levels — extended sweep up to 10,000)

| Label | Threshold | Expected Duration | Rationale |
|-------|-----------|-------------------|-----------|
| XS | tick_50 | ~2-5s | Near time_5s equivalent. Lots of bars, noisy book. |
| Small | tick_100 | ~5-10s | First true event-bar scale. |
| Med-Small | tick_250 | ~15-25s | Moderate aggregation. |
| Medium | tick_500 | ~30-60s | 1-minute-scale. Book has settled. |
| Med-Large | tick_1000 | ~1-2 min | Meaningful equilibrium. |
| Large | tick_2000 | ~3-5 min | Typical discretionary trader bar. |
| XL | tick_5000 | ~8-15 min | Upper intraday range. |
| XXL | tick_10000 | ~20-30 min | Tests whether spatial signal survives extreme aggregation. |

**Important**: These duration estimates are from the OLD broken tick bars. Genuine tick bars will have DIFFERENT durations because they count actual trades, not snapshots. The first step (calibration) will determine actual durations. Expect genuine tick bars to produce fewer bars per day than the old broken ones (real trades are sparser than 10/s snapshots).

#### CNN Protocol (MUST match 9E/9D exactly)

- **Architecture**: R3-exact Conv1d(2→59→59), 12,128 params
- **Normalization**: TICK_SIZE ÷0.25 on prices (integer tick offsets), per-day z-scoring on log1p(size)
- **Validation**: 80/20 train/val from TRAIN days only (NOT test-as-validation)
- **Optimizer**: AdamW(lr=1e-3, wd=1e-4), CosineAnnealingLR(T_max=50, eta_min=1e-5)
- **Batch size**: 512, max 50 epochs, patience=10
- **Target**: fwd_return_5 (5 event-bars ahead, NOT fixed clock time)
- **Seed**: 42
- **CV**: 5-fold expanding window on 19 trading days

#### Baseline

- **time_5s CNN R²=0.089** (from 9E with proper validation) — this is the number to beat
- Per-fold reference from 9E: [0.139, 0.086, -0.049, 0.131, 0.140]

#### Key Design Decisions

1. **Calibration first**: Run bar construction only (no features, no CNN) for all 8 thresholds to get actual bar counts and durations. This determines whether any thresholds are too coarse (< 50 bars/day) or too fine (> 100,000 bars/day) with genuine trade-counting bars.

2. **Drop non-viable thresholds**: If calibration shows tick_10000 produces < 50 bars/day, drop it. If tick_50 produces > 100,000 bars/day (sub-second), drop it. Aim for 4-6 viable thresholds after calibration.

3. **Incremental execution**: Run Medium (tick_500) fold 5 first as MVE gate. If train R² < 0.05, investigate before expanding. Then full sweep.

4. **The R² vs bar-size curve is the primary deliverable**: We want to see the shape — does R² rise, peak, and fall? Is it monotonic? Where is the optimum?

5. **Fold 3 diagnostic**: Track fold 3 (Oct 2022) separately. If any tick-bar threshold produces positive R² on fold 3 (where time_5s gives -0.049), that's evidence the fold 3 failure was bar-type-driven, not regime-driven.

#### Decision Framework

| Outcome | Criterion | Action |
|---------|-----------|--------|
| **BETTER** | Peak R² ≥ 0.107 (20%+ above 0.089) | Switch pipeline to optimal tick-bar threshold |
| **COMPARABLE** | 0.071 ≤ all R² ≤ 0.107 | Stick with time_5s (simpler, proven) |
| **WORSE** | All R² < 0.071 | Time_5s definitively vindicated |
| **Peak found** | Inverted-U shape with clear optimum | Adopt peak threshold |
| **Monotonic up** | R² still rising at tick_10000 | Consider even larger thresholds |

#### Abort Criteria

- Any threshold with < 50 bars/day after calibration → drop that threshold
- Train R² < 0.05 on Medium fold 5 → investigate normalization before expanding
- All tested thresholds produce R² < 0.03 → systematic issue with event-bar export
- Data starvation: if the largest thresholds produce < 500 total train bars in fold 1 (4 days), skip them for fold 1 but still run folds 4-5

#### Resource Budget

| Step | Estimate |
|------|----------|
| Calibration (8 thresholds, bar counts only) | ~1-2 hr |
| Feature export (6-8 thresholds × 19 days) | ~6-10 hr |
| CNN training (6-8 thresholds × 5 folds) | ~2-3 hr |
| **Total** | **~10-15 hr** |

Compute profile: CPU only. No GPU needed. CNN is 12k params.

#### Deliverables

```
.kit/results/r3b-genuine-tick-bars/
├── calibration/
│   └── threshold_sweep.json          # threshold → {bars_per_day, median_duration, p10, p90, total_bars}
├── tick_{N}/                         # One per viable threshold
│   ├── fold_results.json             # Per-fold: {train_r2, test_r2, val_r2, epochs, bar_count}
│   ├── bar_statistics.json           # Duration stats, bars_per_day variance
│   └── normalization_verification.txt
├── sweep_summary.json                # All thresholds: mean R², std, fold 3, bar stats
├── r2_vs_barsize_curve.csv           # threshold, mean_duration, mean_r2, std_r2, fold3_r2
├── comparison_table.md               # Side-by-side: all thresholds + time_5s baseline
└── analysis.md                       # Full analysis with decision per framework
```

### 3. Run the Experiment

After writing the spec:
```bash
mkdir -p .kit/results/r3b-genuine-tick-bars
```

Then launch via MCP tool:
```
kit.research_cycle with spec_path=".kit/experiments/r3b-genuine-tick-bars.md"
```

Or bash fallback (more reliable from worktrees):
```bash
source .orchestration-kit.env
.kit/experiment.sh cycle .kit/experiments/r3b-genuine-tick-bars.md
```

**Note on MCP from worktrees**: The MCP tools run from the main repo's PROJECT_ROOT (hardcoded in `.orchestration-kit.env`). If the MCP launch fails or produces a zombie process, use the bash fallback. Copy the spec to the main repo first if needed:
```bash
cp .kit/experiments/r3b-genuine-tick-bars.md /Users/brandonbell/LOCAL_DEV/MBO-DL-02152026/.kit/experiments/
```

### 4. Update State Files When Done

After the experiment completes:
- Update `.kit/RESEARCH_LOG.md` with the new entry
- Update `.kit/LAST_TOUCH.md` with results and next steps
- Update `CLAUDE.md` Current State section
- Commit on the branch
- Report results to the user

---

## Critical Reminders

1. **You are the ORCHESTRATOR. You NEVER write code.** The research kit sub-agents write all code. You write specs and state files only.
2. **Tick bars are NOW GENUINE** (TB-Fix merged, PR #19). `book_builder.hpp` emits `trade_count` per snapshot. `tick_bar_builder.hpp` accumulates trade counts.
3. **The corrected normalization is mandatory**: TICK_SIZE ÷0.25, per-day z-scoring, proper 80/20 validation. Three prior attempts (9B, 9C, R3b-original) failed because of normalization or data bugs.
4. **Expect different bar counts than the old R3b.** The old "tick_100" produced ~4,630 bars/day (same as time_10s because it was counting snapshots). Genuine tick_100 will produce a DIFFERENT number — likely fewer bars, because real trades are sparser than 10/s snapshots.
5. **The baseline is R²=0.089** (from 9E with proper validation), NOT 0.084 (9D) or 0.132 (R3 leaked). Use 0.089 as the comparator.
6. **Dollar and volume bars are already confirmed genuine** (CV=2-6% and 9-10% respectively). This experiment is tick bars only.

---

Written: 2026-02-19 by orchestrator agent on branch `experiment/cnn-gbt-corrected-pipeline`.
