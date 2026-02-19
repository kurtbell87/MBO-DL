# Next Steps: Implement Hybrid Model

## What happened this session

1. Read all state files (LAST_TOUCH, RESEARCH_LOG, NEXT_STEPS, synthesis analysis, oracle expectancy metrics)
2. Explored existing C++ source tree: bar.hpp, bar_features.hpp, feature_export.hpp, triple_barrier.hpp, raw_representations.hpp, bar_feature_export.cpp, and all existing TDD specs
3. Wrote comprehensive hybrid model build spec: `.kit/docs/hybrid-model.md`
4. Updated all breadcrumbs: LAST_TOUCH.md, CLAUDE.md, this file

## Spec summary: `.kit/docs/hybrid-model.md`

**Two-stage CNN+GBT Hybrid model pipeline.**

### Phase A: C++ Data Export Extension
- Add triple barrier labels (`tb_label`, `tb_exit_type`, `tb_bars_held`) to `bar_feature_export.cpp`
- Position-independent labeling: "if we entered long here, would target or stop hit first?"
- 10 C++ test cases

### Phase B: Python CNN Encoder Training
- Conv1d on (2, 20) permuted book → 16-dim embedding (~7.5k params)
- Trained with linear regression head on fwd_return_h (h=1 and h=5 separately)
- Adam, lr=1e-3, batch=256, max 50 epochs, early stopping patience=5

### Phase C: Python XGBoost Classification
- Freeze CNN, extract 16-dim embeddings
- Concatenate with 20 non-spatial features (selected from 62 Track A)
- XGBoost multi:softmax, 3 classes {−1, 0, +1}, conservative hyperparameters

### Phase D: 5-Fold Expanding Window CV
- Same 19 days, same expanding window protocol as R2/R3/R4
- Per-fold: train CNN → embeddings → XGBoost → evaluate on test fold
- Report: R², accuracy, expectancy, Sharpe, PnL
- Transaction cost sensitivity: 3 scenarios (optimistic, base, pessimistic)
- Ablation: GBT-only baseline, CNN-only baseline

### Key exit criteria (17 total)
- CNN R² at h=5 ≥ 0.08 (mean across folds)
- XGBoost accuracy ≥ 0.38 (above 1/3 random)
- No fold with negative CNN R² at h=5
- Aggregate expectancy (base cost) ≥ $0.50/trade
- Aggregate profit factor ≥ 1.5
- Hybrid outperforms both GBT-only and CNN-only baselines

## Implementation order

1. **C++ extension** — Add TB labels to `bar_feature_export`. Run TDD: red → green → refactor → ship.
2. **Export data** — Run `bar_feature_export --bar-type time --bar-param 5 --output .kit/results/hybrid-model/time_5s.csv` with the updated tool.
3. **Python pipeline** — Create `scripts/hybrid_model/` with all training and evaluation scripts.
4. **Run CV** — Execute 5-fold CV, collect results.
5. **Analysis** — Write `.kit/results/hybrid-model/analysis.md` summarizing findings.

## Key design decisions in the spec

1. **Position-independent TB labels** — No position tracking. At each bar, label = "would long entry hit target first?" Simplifies labeling and avoids serial dependency in labels.
2. **Two separate CNN encoders** — One for h=1, one for h=5. Resolves the R3 open question (CNN at h=1).
3. **Conservative XGBoost** — Lower depth (6 vs 10), higher regularization than overfit-phase GBT. This is a generalization model.
4. **20 non-spatial features** — Excludes per-level book features (redundant with CNN input). Keeps aggregate signals: weighted_imbalance, spread, order flow, price dynamics, time context, microstructure.
5. **CPU-only training** — CNN is ~7.5k params, trains in seconds. No GPU needed.

## Branch

`experiment/temporal-predictability-dollar-tick-actionable` — will need to create a new branch for Phase 9 implementation.
