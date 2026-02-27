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

- **30+ phases complete** (15 engineering + 25 research). Branch: `experiment/pnl-realized-return`.
- **PnL Realized Return — REFUTED on SC-2 but economically positive** (2026-02-26). Realized expectancy $0.90/trade at 19:7. Directional bars +$2.10, hold bars -$1.19 (57% edge destruction). Hold-bar dir acc 51.04% (<52% threshold). Fold instability (Fold 2 outlier). PR #35 (OPEN).
- **2-Class Directional — CONFIRMED with PnL caveat** (2026-02-26). Trade rate liberated 0.28%→85.2%. Directional-bar edge $3.77/trade is real and stable. PR #34 (merged).
- **CNN line CLOSED**. **XGBoost tuning DONE**.
- **Critical finding:** Directional-bar edge ($3.77) is real. Hold-bar exposure (44.4% of trades) destroys 57% of the edge. Reducing hold-bar fraction is the key lever.

## Priority Experiments

### 1. Stage 1 Threshold Optimization (HIGHEST PRIORITY)

Sweep P(directional) threshold from 0.5→0.9. At each level: trade rate, hold fraction, realized expectancy. PnL decomposition predicts: at 15% hold (threshold ~0.70), expectancy ~$2.81/trade. No re-training — just re-score with different thresholds.

**Rationale:** Hold bars destroy 57% of the directional-bar edge (-$1.19 of +$2.10). Each 10pp reduction in hold-bar fraction recovers ~$0.27/trade. This is the highest-leverage, lowest-cost experiment.

**Spec:** Not yet created
**Branch:** TBD
**Compute:** Local (Quick tier, <2 min)

### 2. CPCV at Optimal Threshold (45 splits)

Once the optimal threshold is identified, validate with 45-split CPCV for proper CI and PBO. 3 WF folds have insufficient power (t-stat ~1.35, p≈0.31). Only worth running if #1 produces robust results.

**Spec:** Not yet created
**Branch:** TBD
**Compute:** Local

### 3. Volume Horizon Investigation

The 50,000-contract volume horizon truncates the barrier race before 3600s, leaving hold-bar returns unbounded (-63 to +63 ticks, not -19 to +19). Options: (a) set volume_horizon to 10^9 (barrier race always runs full 3600s), (b) measure forward return at race-end time instead of 3600s. Requires re-export + re-train.

**Spec:** Not yet created
**Branch:** TBD
**Compute:** Local

### 4. Intermediate Geometry (14:6, 15:5)

19:7 direction is random, 10:5 has marginal signal. Sweet spot may be in between. Now well-understood PnL model makes this more interpretable.

**Spec:** Not yet created
**Branch:** TBD
**Compute:** Local

### 5. Feature Engineering for Wider Barriers (HIGHEST EFFORT)

Rolling VWAP slope, cumulative order flow, volatility regime markers. Addresses root cause. Requires TDD cycle + re-export.

**Spec:** Not yet created
**Branch:** `feat/wider-barrier-features`
**Compute:** Local

### 6. Regime-Conditional Trading (INDEPENDENT)

GBT profitable in H1 2022 (+$0.003, +$0.029). Test Q1-Q2-only strategy.

**Spec:** Not yet created
**Branch:** `experiment/regime-conditional`
**Compute:** Local

---

## Completed (This Cycle)

| Experiment | Verdict | Key Finding |
|-----------|---------|-------------|
| **PnL Realized Return** | **REFUTED SC-2, +$0.90/trade** | **Dir bars +$2.10, hold bars -$1.19. Hold-bar dir acc 51.04%. Fold instability.** |
| **2-Class Directional** | **CONFIRMED (PnL caveat)** | **Trade rate 0.28%→85.2%. Reachability 58.6%. Direction ~50% (random). Dir-bar edge $3.77.** |
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

- **Directional-bar edge is REAL**: $3.77/trade, stable across all 3 folds. 2.71:1 payoff × ~50% accuracy > 38.4% BEV.
- **Hold bars destroy 57% of edge**: -$1.19/trade drag at 44.4% hold fraction. Reducing hold-bar exposure is THE lever.
- **Hold-bar returns are UNBOUNDED**: Volume horizon (50K) truncates barrier race; returns span ±63 ticks, not ±19.
- **Fold instability at 19:7**: std $1.16 on mean $0.90 (CV=129%). Cannot reject null with 3 folds (p≈0.31).
- **10:5 is definitively non-viable**: -$1.65/trade under realized PnL, negative at all cost levels.
- **Stage 1 threshold is the fastest lever**: No re-training needed, just re-score at different thresholds.
- **Compute preference**: Local for CPU-only. RunPod for GPU. EC2 spot only for large data.
- **Orchestrator protocol**: YOU are the orchestrator. You NEVER write code. Delegate via kit phases.

---

Written: 2026-02-26. PnL realized return: $0.90/trade at 19:7, but fold-unstable and hold bars drag -$1.19. PR #35 open. Next: Stage 1 threshold optimization.
