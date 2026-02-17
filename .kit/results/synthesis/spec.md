# Phase 6: Synthesis [Research]

**Spec**: TRAJECTORY.md §11 Phase 6
**Depends on**: ALL prior phases — 1, 2, 3, R1, 4, 5, R2, R3, R4.
**Unlocks**: Model architecture build spec.
**GPU budget**: 0 hours (analysis only — no model training). **Max runs**: 1 (deterministic).

---

## Objective

Collate findings from R1–R4 and the engineering phases into a single decision document that determines (a) whether to proceed with model training, and (b) the exact architecture, bar type, feature set, and labeling method to use.

This is a **pure analysis phase** — no new models are trained. The experiment reads existing results, resolves inter-experiment tensions, and produces the synthesis document.

---

## Prior Findings (Input Summary)

### R1: Subordination Test — REFUTED
- Event-driven bars do not beat time bars on normality/heteroskedasticity.
- 0/3 primary pairwise tests significant after Holm-Bonferroni.
- Dollar bars show *higher* AR R² — opposite of theory's prediction.
- Effect reverses across quarters (regime-dependent).
- **Decision**: `time_5s` is the baseline bar type.

### R2: Information Decomposition — FEATURES SUFFICIENT
- Best R²=0.0067 (MLP on raw book, h=1). All h>1 horizons negative.
- 0/40 tests pass dual threshold (relative >20% AND corrected p<0.05).
- Δ_spatial=+0.003 (p=0.96): raw book vs. hand-crafted — not significant.
- Δ_msg_summary=−0.001, Δ_msg_learned=−0.003: messages add nothing.
- Δ_temporal=−0.006: lookback hurts (overfitting at 845 dims).
- **Decision**: GBT_BASELINE architecture — no CNN/message-encoder/SSM justified by information gaps.

### R3: Book Encoder Inductive Bias — CNN BEST
- CNN mean R²=0.132 ± 0.048 (5-fold, ~12k params, predicting return_5).
- CNN vs. Attention: Δ=+0.047, corrected p=0.042, Cohen's d=1.86 — **significant**.
- CNN vs. MLP: Δ=+0.032, corrected p=0.251 — not significant but CNN never negative.
- CNN 16-dim embedding linear probe R²=0.111 vs. raw 40-dim probe R²=0.027 (retention ratio=4.16x, p=0.012).
- **Decision**: Conv1d spatial encoder with 16-dim output is the best spatial representation; CNN *amplifies* rather than merely compresses signal.

### R4: Temporal Predictability — NO TEMPORAL SIGNAL
- All 36 Tier 1 AR configs produce negative R².
- 0/16 Tier 2 temporal augmentation gaps pass dual threshold.
- Temporal-Only R² ≈ 0 at h=1, negative at all longer horizons.
- Converges with R2 Δ_temporal finding.
- **Decision**: Drop SSM / temporal encoder. Static current-bar features are sufficient.

---

## Key Tension to Resolve

**R2 says "features sufficient" (Δ_spatial not significant) → drop spatial encoder.**
**R3 says CNN R²=0.132 >> R2 MLP R²=0.007 → spatial encoder adds massive value.**

This is not a contradiction — it is a methodological difference:
- R2 compared a **2-layer MLP (64 units)** on raw book (40-dim) vs. hand-crafted features (62-dim). Both achieved R²<0.007. The *gap* between them was not significant.
- R3 compared a **matched-parameter Conv1d (12k params)** vs. Attention vs. MLP on raw book images with structured `(20, 2)` input. The CNN achieved R²=0.132 — **20× higher** than R2's best.

The resolution hinges on two factors:
1. **Representation format**: R2 flattened the book to a 40-element vector (destroying spatial structure). R3 preserved the `(20, 2)` price-ladder layout.
2. **Model capacity**: R2's MLP was a diagnostic tool (~5k params). R3's CNN was purpose-built (~12k params) with inductive bias for adjacent-level patterns.

Synthesis must determine whether the R3 CNN's R²=0.132 represents **genuine exploitable signal** or **in-sample overfitting** that a GBT on hand-crafted features could match.

---

## Questions to Answer

### Q1: Go/No-Go — Proceed with Model Training?

**Sub-questions**:
- Is there positive out-of-sample R² at any horizon? (R2 + R3 results)
- Is the signal magnitude economically meaningful after estimated transaction costs?
- Do we have a labeling method with demonstrated oracle expectancy? (Phase 3 C++ backtest)

**Evidence sources**: R2 metrics.json, R3 metrics.json, Phase 3 oracle replay (C++ test output).

**Decision rule**:
```
If R² > 0 out-of-sample at ≥1 horizon AND oracle expectancy > 0:
  → GO. Proceed to model build.
If R² > 0 but oracle expectancy unknown or ≤ 0:
  → CONDITIONAL GO. Proceed to model build but flag oracle gap.
If R² ≤ 0 everywhere:
  → NO-GO. The data does not support supervised learning on MES at 5s bars.
```

### Q2: Bar Type

**Evidence**: R1 (REFUTED subordination), R4 (no temporal structure in any bar type).
**Expected answer**: `time_5s` — no competing evidence.

### Q3: Architecture — §7.2 Simplification Cascade (Final)

Resolve R2 vs. R3 tension. Fill in the cascade table with final decisions:

| Encoder Stage | R2 Gap | R2 Verdict | R3 Evidence | Final Decision |
|---------------|--------|------------|-------------|----------------|
| Spatial (CNN) | Δ_spatial=+0.003, p=0.96 | Drop | CNN R²=0.132 >> MLP R²=0.100 | **?** |
| Message encoder | Δ_msg=−0.003 | Drop | N/A | Drop |
| Temporal (SSM) | Δ_temporal=−0.006 | Drop | R4 confirms | Drop |

**Decision rule for spatial encoder**:
```
OPTION A — GBT Baseline (R2 recommendation):
  Use hand-crafted 62-dim features → XGBoost.
  Justification: R2's Δ_spatial was not significant.
  Risk: Leaves R3's R²=0.132 CNN signal on the table.

OPTION B — CNN + GBT Hybrid:
  Use Conv1d encoder on raw (20,2) book → 16-dim embedding → concatenate
  with hand-crafted features → XGBoost or MLP head.
  Justification: R3 shows CNN extracts 20× more signal than flat MLP.
  The 16-dim embedding is a sufficient statistic (retention ratio=4.16x).
  Risk: R3 used return_5; R2 found all h>1 horizons negative.
  Need to reconcile horizon discrepancy.

OPTION C — CNN Standalone:
  Use Conv1d encoder on raw (20,2) book → 16-dim → linear head.
  Justification: R3 sufficiency test shows CNN-16d outperforms raw-40d.
  Risk: Ignores hand-crafted features that may capture non-spatial signal
  (order flow, trade imbalance, etc.).
```

**Reconciliation protocol** (no new training — use existing data):
1. Compare R3's CNN return_5 R² (0.132) with R2's MLP return_5 R² (negative in R2).
2. Check R3's target: was `return_5` defined identically to R2's `fwd_return_5`?
3. If definitions match but R² differs by 20×, the difference is architectural (CNN's inductive bias on structured input), not a data artifact.
4. Document whether R3's R²=0.132 is on the same test days / CV folds as R2.

### Q4: Feature Set — GBT Baseline Input Selection

**Evidence sources**:
- R2 Config (a): 62 hand-crafted features → R²=0.0036 (h=1)
- R2 Config (b): 40 raw book features → R²=0.0067 (h=1)
- R4 feature importance: top static features are book_snap_19, book_snap_21 (best bid/ask levels)
- R3 CNN: 16-dim embedding encodes all predictive book structure

**Decision rule**:
```
If OPTION A (GBT Baseline):
  Input = 62 hand-crafted features (Track A).
  Rationale: Simplest, and R2 showed Δ_spatial is not significant.

If OPTION B (CNN + GBT Hybrid):
  Input = CNN 16-dim embedding ⊕ non-spatial hand-crafted features.
  Non-spatial features = Track A minus book-derived features (i.e., trade
  features, order flow features, volatility features — ~20 dims).
  Rationale: CNN already encodes spatial book structure; avoid redundancy.

If OPTION C (CNN Standalone):
  Input = CNN 16-dim embedding only.
```

### Q5: Prediction Horizon

**Evidence**:
- R2: Only h=1 (~5s) has positive R². All longer horizons negative.
- R3: Tested h=5 only (return_5 = 5-bar ≈ 25s forward). Got R²=0.132.
- R4: h=1 is the only horizon where Static-Book GBT R² > 0 (0.0046).

**Tension**: R2 and R4 agree that only h=1 works. R3 used h=5 and got much higher R².
**Resolution**: Document the discrepancy. If R3's return_5 target is identical to R2's fwd_return_5, the CNN's inductive bias explains the difference. Recommend testing both h=1 and h=5 in the model build.

### Q6: Labeling Method

**Evidence**: Phase 3 oracle_labeler implements first-to-hit labeling (target_ticks=10, stop_ticks=5, take_profit_ticks=20). Phase 3 multi-day backtest was a TDD engineering phase — results are in C++ test output, not `.kit/results/`.

**Action**: Extract oracle expectancy from C++ test output if available, or note as an open question to resolve before model build.

### Q7: Statistical Limitations

Compile across all experiments:
- **Single-year data**: 2022 was a bear market + rate hiking cycle. All findings may be regime-specific.
- **Power**: R2 had 5 folds × ~84k bars. With R² < 0.007, the 95% CI on any gap overlaps zero. We cannot detect effect sizes < ~0.003 R² reliably.
- **Failed corrections**: R2 Δ_spatial had uncorrected p=0.025 but corrected p=0.96. R3 CNN vs. MLP had corrected p=0.251. These are suggestive but inconclusive.
- **R3 horizon**: R3 only tested return_5. CNN performance at h=1 is unknown.
- **No regime stratification**: R1 showed quarter-level reversals. No regime-conditional analysis was performed in R2–R4.

---

## Protocol

### Step 1: Load Results

Read the following files (no new computation):
- `.kit/results/subordination-test/metrics.json` — R1 findings
- `.kit/results/info-decomposition/metrics.json` — R2 findings
- `.kit/results/info-decomposition/analysis.md` — R2 detailed analysis
- `.kit/results/book-encoder-bias/metrics.json` — R3 findings
- `.kit/results/book-encoder-bias/model_comparison.csv` — R3 fold-level R²
- `.kit/results/book-encoder-bias/sufficiency_test.csv` — R3 sufficiency test
- `.kit/results/book-encoder-bias/pairwise_tests.csv` — R3 statistical tests
- `.kit/results/temporal-predictability/metrics.json` — R4 findings
- `.kit/results/temporal-predictability/analysis.md` — R4 detailed analysis

### Step 2: Resolve R2–R3 Tension

1. Extract R3's return_5 definition and confirm it matches R2's `fwd_return_5`.
2. Tabulate R2 MLP R² on `fwd_return_5` vs. R3 CNN R² on `return_5` — same test days?
3. Document whether the 20× R² difference is attributable to:
   - (a) Structured (20,2) input vs. flattened 40-dim vector
   - (b) Conv1d inductive bias vs. generic MLP
   - (c) Model capacity (12k params vs. ~5k params in R2)
   - (d) Different data splits or target definitions
4. Produce a **reconciliation table** with side-by-side comparison.

### Step 3: Fill Architecture Cascade

Using the reconciliation from Step 2, make the final architecture decision:

```
Encoder Stage       | Gap Evidence     | R3 Evidence           | Decision
--------------------|------------------|-----------------------|----------
Spatial (CNN)       | R2: not signif.  | R3: CNN R²=0.132,     | [A/B/C]
                    |                  | p=0.042 vs Attention   |
Message encoder     | R2: negative     | —                     | DROP
Temporal (SSM)      | R2+R4: negative  | —                     | DROP
```

### Step 4: Compile Feature Set

Based on the architecture decision, specify the exact input features:
- List features by name and dimension
- Cite the evidence for each inclusion/exclusion
- Specify preprocessing (normalization, encoding format)

### Step 5: Document Limitations

Enumerate all statistical caveats with severity ratings:
- **Critical**: Could invalidate the go/no-go decision
- **Major**: Could change the architecture recommendation
- **Minor**: Noted for future work

### Step 6: Produce Decision Document

Write the synthesis document to `.kit/results/synthesis/analysis.md`.

---

## Document Structure

```
.kit/results/synthesis/analysis.md

1. Executive Summary
   - Go/no-go: proceed with model training? [YES/CONDITIONAL/NO]
   - Recommended architecture
   - Recommended bar type: time_5s
   - Recommended prediction horizon(s)
   - Recommended labeling method (or flag as open)

2. Bar Type Decision (R1 + R4)
   - R1: Subordination REFUTED — time bars win
   - R4: No temporal structure favoring any bar type
   - Decision: time_5s with justification

3. Architecture Decision (R2 + R3 + R4)
   - §7.2 Simplification Cascade — final table
   - R2 vs. R3 reconciliation table
   - Message encoder: DROP (R2 Tier 2)
   - Temporal encoder: DROP (R2 + R4)
   - Spatial encoder: [INCLUDE/DROP] with justification
   - Final architecture diagram

4. Feature Set Specification
   - Exact feature list with dimensions
   - Preprocessing pipeline
   - Warmup and lookahead policy (§8.6)

5. Prediction Horizon Analysis
   - R2: h=1 only has signal (R²=0.007)
   - R3: h=5 CNN achieves R²=0.132
   - Reconciliation and recommendation

6. Oracle Expectancy (Phase 3)
   - Available results from C++ oracle replay
   - Labeling method comparison (if data available)
   - Or: flag as open question

7. Statistical Limitations
   - Single-year caveat (2022 bear market)
   - Power analysis (detectable effect sizes)
   - Failed corrections (suggestive findings)
   - R3 horizon gap (only tested h=5)
   - No regime-conditional analysis

8. Convergence Matrix
   Table: Question × Experiment → Finding
   Shows how each experiment contributes to each decision.

9. Open Questions for Model Build
   - Items that synthesis cannot resolve
   - Recommended experiments for the model build phase
```

---

## Implementation

```
Language: Python (analysis only — no model training)
Entry point: research/R6_synthesis.py

Dependencies:
  - polars (JSON/CSV loading)
  - numpy (basic statistics)

Input:
  - .kit/results/subordination-test/metrics.json
  - .kit/results/info-decomposition/metrics.json
  - .kit/results/info-decomposition/analysis.md
  - .kit/results/book-encoder-bias/metrics.json
  - .kit/results/book-encoder-bias/model_comparison.csv
  - .kit/results/book-encoder-bias/sufficiency_test.csv
  - .kit/results/book-encoder-bias/pairwise_tests.csv
  - .kit/results/temporal-predictability/metrics.json
  - .kit/results/temporal-predictability/analysis.md

Output:
  - .kit/results/synthesis/analysis.md
  - .kit/results/synthesis/convergence_matrix.csv
  - .kit/results/synthesis/architecture_decision.json
```

---

## Compute Budget

| Item | Estimate |
|------|----------|
| Load all result files | ~1 min |
| R2–R3 reconciliation analysis | ~5 min |
| Document generation | ~10 min |
| **Total wall-clock** | **~15 min** |
| **GPU hours** | **0** |
| **Runs** | **1** |

---

## Deliverables

### Convergence Matrix

```
Question               | R1          | R2               | R3                | R4              | Decision
-----------------------|-------------|------------------|-------------------|-----------------|-------------------
Bar type               | time_5s     | time_5s (used)   | —                 | time_5s (used)  | time_5s
Spatial encoder        | —           | DROP (p=0.96)    | CNN (p=0.042)     | —               | [INCLUDE/DROP]
Message encoder        | —           | DROP (Δ<0)       | —                 | —               | DROP
Temporal encoder       | —           | DROP (Δ=−0.006)  | —                 | DROP (Δ≤+0.0004)| DROP
Signal horizon         | —           | h=1 only (R²>0)  | h=5 (R²=0.132)   | h=1 only (R²>0)| [h=1/h=5/both]
Signal magnitude       | —           | R²=0.007 (h=1)   | R²=0.132 (h=5)   | R²=0.005 (h=1) | [reconcile]
Subordination theory   | REFUTED     | —                 | —                 | —               | REFUTED
Book sufficiency       | —           | CONFIRMED         | CNN amplifies     | —               | [reconcile]
Temporal predictability| —           | Δ_temporal<0      | —                 | Martingale      | NONE
```

### Architecture Decision JSON

```json
{
  "go_no_go": "GO|CONDITIONAL|NO_GO",
  "bar_type": "time_5s",
  "bar_param": 5,
  "architecture": "gbt_baseline|cnn_gbt_hybrid|cnn_standalone",
  "spatial_encoder": {
    "include": true|false,
    "type": "conv1d",
    "embedding_dim": 16,
    "justification": "R3 p=0.042 vs attention, retention_ratio=4.16"
  },
  "message_encoder": {
    "include": false,
    "justification": "R2 Δ_msg_learned < 0 at all horizons"
  },
  "temporal_encoder": {
    "include": false,
    "justification": "R2 Δ_temporal = −0.006; R4 confirms martingale"
  },
  "features": {
    "source": "track_a_62|cnn_16d|cnn_16d_plus_non_spatial",
    "dimension": 62|16|36,
    "preprocessing": "z-score per day, warmup=50 bars"
  },
  "prediction_horizons": [1, 5],
  "labeling": {
    "method": "first_to_hit|triple_barrier|open_question",
    "params": {"target_ticks": 10, "stop_ticks": 5, "horizon": 100}
  },
  "limitations": [
    "single_year_2022",
    "r3_only_tested_h5",
    "no_regime_conditioning",
    "power_floor_r2_0.003"
  ]
}
```

### Required Outputs in `analysis.md`

1. Executive summary with go/no-go verdict.
2. Convergence matrix (all experiments × all questions).
3. R2–R3 reconciliation table with side-by-side comparison.
4. §7.2 simplification cascade — final filled table.
5. Exact feature set specification with dimensions and preprocessing.
6. Prediction horizon recommendation with evidence from R2, R3, R4.
7. Statistical limitations with severity ratings.
8. Open questions list for the model build phase.
9. Explicit statement: "Architecture recommendation: [X]" with one-paragraph justification.

---

## Exit Criteria

- [ ] All R1–R4 result files loaded and cross-referenced
- [ ] R2 vs. R3 tension resolved with documented reconciliation
- [ ] §7.2 simplification cascade filled with final decisions for all 3 encoder stages
- [ ] Bar type decision documented (expected: time_5s)
- [ ] Feature set specified (exact dimensions + preprocessing)
- [ ] Prediction horizon recommendation documented with evidence
- [ ] Oracle expectancy status documented (result or flagged as open)
- [ ] Convergence matrix produced showing all experiments × all questions
- [ ] Statistical limitations enumerated with severity ratings
- [ ] Architecture decision JSON produced
- [ ] Go/no-go decision documented with supporting evidence
- [ ] Open questions for model build phase listed
- [ ] Synthesis document produced at `.kit/results/synthesis/analysis.md`
- [ ] Summary entry ready for `.kit/RESEARCH_LOG.md`
