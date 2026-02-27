# Next Steps — Updated 2026-02-26

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

- **30+ phases complete** (15 engineering + 23 research). Branch: `experiment/2class-directional`.
- **2-Class Directional — CONFIRMED with PnL caveat** (2026-02-26). Two-stage pipeline liberates trade volume (0.28%→85.2% at 19:7). Stage 1 reachability accuracy 58.6%. Stage 2 direction accuracy ~50% (random at 19:7). Reported exp $3.78 but corrected ~$0.44/trade. PR #34 (OPEN).
- **Label Geometry 1h — INCONCLUSIVE** (2026-02-26). Dir acc ~50-51%, hold predictor at wide barriers. PR #33 (merged).
- **Time Horizon CLI — COMPLETE** (2026-02-26). PR #32 (merged).
- **CNN line CLOSED** for classification. GBT beats CNN by 5.9pp accuracy.
- **XGBoost tuning DONE** — 0.33pp plateau. Feature set is the binding constraint.
- **Critical finding:** Reachability detection works (58.6% at 19:7). Direction prediction is random at wide barriers (~50%). Favorable payoff ratio (2.71:1) is the economic driver, not skill.

## Priority Experiments

### 1. Corrected PnL Validation at 19:7 (HIGHEST PRIORITY)

Re-run walk-forward with PnL model that correctly assigns $0 (or actual realized PnL) to hold-bar trades. Same data, same trained models — just fix the economic computation. Resolves the dominant uncertainty: is corrected expectancy truly positive?

**Rationale:** Current $3.78/trade is ~8x inflated. Corrected estimate ~$0.44 (CI [$0.04, $0.84]). A single re-run with correct PnL determines if the 2-class approach is viable.

**Spec:** Not yet created
**Branch:** TBD (may be on `experiment/2class-directional` or new branch)
**Compute:** Local

### 2. CPCV at 19:7 with Corrected PnL (45 splits)

Three WF folds give CI [$0.04, $0.84] — too wide for go/no-go. CPCV with 45 splits gives proper confidence intervals and PBO. Only worth running if #1 confirms positive expectancy.

**Spec:** Not yet created
**Branch:** TBD
**Compute:** Local

### 3. Stage 1 Threshold Optimization

Raise P(directional) threshold from 0.5 to 0.6-0.8. Trades fewer bars but higher fraction hit directional barriers. Reduces sensitivity to hold-bar PnL model. Could improve corrected economics significantly.

**Spec:** Not yet created
**Branch:** TBD
**Compute:** Local

### 4. Intermediate Geometry Exploration (14:6, 15:5)

19:7 has random direction, 10:5 has marginal signal. An intermediate geometry (BEV ~43-48%) might find the sweet spot where directional accuracy exceeds BEV with enough margin AND the payoff ratio amplifies the edge.

**Spec:** Not yet created
**Branch:** TBD
**Compute:** Local

### 5. Feature Engineering for Wider Barriers (HIGHEST EFFORT)

Add rolling VWAP slope, cumulative order flow (50-500 bars), volatility regime markers. Addresses root cause (feature ceiling at wide barriers). Requires TDD cycle + re-export.

**Spec:** Not yet created
**Branch:** `feat/wider-barrier-features`
**Compute:** Local

### 6. Regime-Conditional Trading (INDEPENDENT)

GBT profitable in H1 2022 (+$0.003, +$0.029), negative in H2. Test Q1-Q2-only strategy.

**Spec:** Not yet created
**Branch:** `experiment/regime-conditional`
**Compute:** Local

---

## Completed (This Cycle)

| Experiment | Verdict | Key Finding |
|-----------|---------|-------------|
| **2-Class Directional** | **CONFIRMED (PnL caveat)** | **Trade rate 0.28%→85.2%. Reachability 58.6%. Direction ~50% (random). Corrected exp ~$0.44.** |
| Label Geometry 1h | INCONCLUSIVE (Outcome B) | Dir acc 50.7%, model refuses to trade at high-ratio geom (<0.3% rate). Feature ceiling. |
| Time Horizon CLI TDD | DONE (PR #32) | `--max-time-horizon`/`--volume-horizon` flags. Defaults 300→3600s, 500→50000. |
| Label Geometry Phase 1 | REFUTED | 90.7-98.9% hold — root cause: 300s time cap (now fixed). |
| Synthesis-v2 | GO (55-60% prior) | Breakeven WR is the correct metric. Label geometry is the remaining lever. |
| Label Design Sensitivity P0 | REFUTED (Outcome C) | 123 geometries mapped. $5.00 gate miscalibrated. |
| XGBoost Tuning | REFUTED (Outcome C) | 0.33pp plateau across 64 configs. |
| Bidirectional Re-Export | PASS (312/312) | 152-col schema, S3 backed. |
| bar-feature-export-geometry TDD | DONE (PR #28) | --target/--stop CLI flags, 47 tests. |

---

## Key Constraints

- **Reachability IS learnable**: Stage 1 binary accuracy 58.6% at 19:7 (6pp above majority baseline).
- **Direction is NOT learnable at wide barriers**: Stage 2 accuracy ~50% at 19:7, with no directional features in top-10.
- **Favorable payoff ratio is the economic driver**: 50% accuracy > 38.4% BEV WR at 2.71:1 payoff.
- **PnL model validity is the dominant uncertainty**: Corrected exp ~$0.44/trade (CI [$0.04, $0.84]). Must validate.
- **Hold-bar trades are 44.4% of volume at 19:7**: Their PnL treatment determines viability.
- **Oracle at 3600s**: $3.22-$9.44/trade — perfect foresight IS profitable.
- **Compute preference**: Local for CPU-only (<1GB data). RunPod for GPU. EC2 spot only for large data (>10GB).
- **Orchestrator protocol**: YOU are the orchestrator. You NEVER write code. Delegate via kit phases.

---

Written: 2026-02-26. 2-class directional CONFIRMED (trade rate liberated, corrected exp ~$0.44). PR #34 open. Next: corrected PnL validation.
