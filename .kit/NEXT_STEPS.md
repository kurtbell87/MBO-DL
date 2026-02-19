# Next Steps — Updated 2026-02-19

## FIRST: Create a New Branch

**BEFORE you do ANY work — create a worktree and branch.**

```bash
git worktree add ../MBO-DL-<slug> -b <branch-type>/<name> main
cd ../MBO-DL-<slug>
source .orchestration-kit.env
```

**All work happens on a branch. Never touch main directly.**

---

## Current State

- **22 phases complete** (9 engineering + 13 research). All on `main`, clean.
- CNN spatial signal confirmed: R²=0.089 on time_5s (3 independent reproductions).
- Genuine tick bars (tick_100): R²=0.124 (+39% vs baseline), but p=0.21 — not actionable without replication.
- End-to-end pipeline NOT viable: expectancy=-$0.37/trade at base $3.74 RT costs.
- Breakeven requires +2.0pp win rate (51.3% → 53.3%) or alternative architecture.

## Priority Options (owner to choose)

### 1. End-to-End CNN Classification (HIGHEST PRIORITY)

Train CNN directly on tb_label (3-class cross-entropy) instead of regression→frozen embedding→XGBoost. Eliminates the regression-to-classification bottleneck identified as the key loss point in 9E.

**Why:** The 16-dim penultimate layer currently learns return-prediction features. Direct classification would learn class-discriminative features — fundamentally different optimization target.

**Branch:** `experiment/e2e-cnn-classification`

### 2. XGBoost Hyperparameter Tuning (LOW-HANGING FRUIT)

Grid search on max_depth, learning_rate, n_estimators, min_child_weight with 5-fold CV. Current hyperparameters inherited from 9B (broken pipeline era). The 2pp win rate gap is small enough that tuning could close it.

**Branch:** `experiment/xgb-hyperparam-tuning`

### 3. Label Design Sensitivity (ARCHITECTURAL)

Test alternative triple barrier parameters: wider target (15 ticks), narrower stop (3 ticks). At 15:3 ratio, breakeven win rate drops to ~42.5% — well below current 51.3%.

**Branch:** `experiment/label-design-sensitivity`

### 4. CNN at h=1 with Corrected Normalization (EXPLORATORY)

R2 showed signal strongest at h=1. Test with corrected normalization to see if shorter horizon improves classification.

**Branch:** `experiment/cnn-h1-corrected`

### 5. Tick_100 Multi-Seed Replication (INDEPENDENT)

R3b-genuine showed tick_100 R²=0.124 (+39%) but p=0.21 — driven by fold 5 outlier. Needs 3 seeds/fold and adjacent thresholds (tick_50, tick_200) to confirm.

**Branch:** `experiment/tick100-replication`

---

## Key Constraints

- **CNN protocol MUST match 9E/9D**: R3-exact Conv1d(2→59→59), 12,128 params, TICK_SIZE ÷0.25, per-day z-scoring, 80/20 train/val, AdamW+CosineAnnealingLR, seed=42
- **Baseline**: time_5s R²=0.089 (9E), XGBoost acc=0.419, expectancy=-$0.37
- **All experiments use 19-day dataset** (DATA/GLBX-20260207-L953CAPU5B/)
- **Orchestrator protocol**: YOU are the orchestrator. You NEVER write code. Delegate via kit phases.

---

Written: 2026-02-19. All branches clean. No worktrees active.
