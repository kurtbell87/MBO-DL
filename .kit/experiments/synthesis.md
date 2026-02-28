# Experiment: Full Research Synthesis v2 — Strategic Pivot Assessment

## Hypothesis

A comprehensive synthesis of all 22+ experiments will establish that: (1) the project's overall viability status is **GO** — at least one unexplored high-leverage intervention (label geometry training at breakeven-favorable ratios) has >50% prior probability of producing positive per-trade expectancy; (2) the prior focus on oracle net expectancy as the gate metric was the wrong optimization target — breakeven win rate vs. model accuracy is the binding constraint; and (3) the single highest-priority next experiment is Phase 1 label geometry training at 4 geometries (10:5, 15:3, 19:7, 20:3) selected by breakeven WR diversity.

**Falsifiable direction:** If the synthesis finds that ALL plausible interventions have been tested or that the accuracy-transfer uncertainty makes the label geometry lever <50% likely to succeed, the verdict is NO-GO and the project closes.

## Independent Variables

N/A — this is a pure analysis experiment with no model training. The "independent variable" is the analytical lens: the synthesis resolves tensions across 22+ experiments by examining each finding through the updated framework (breakeven WR as binding constraint, not oracle ceiling).

## Controls

| Control | Value | Rationale |
|---------|-------|-----------|
| Input data | All `.kit/results/*/metrics.json` and `analysis.md` files | No new computation — analysis of existing results only |
| Methodology | Evidence matrix + explicit tension resolution | Structured analysis prevents cherry-picking |
| Prior synthesis | R6 synthesis (2026-02-17) + SYNTHESIS.md (2026-02-24) | Baseline for what has changed |
| Decision framework | GO/CONDITIONAL GO/NO-GO with explicit criteria | Pre-committed to prevent motivated reasoning |

## Metrics (ALL must be reported)

### Primary

1. **go_nogo_verdict**: {GO, CONDITIONAL_GO, NO_GO} — updated project status with full evidence chain
2. **highest_priority_next_experiment**: Specific experiment with hypothesis, success criterion, compute cost, and prior probability of success

### Secondary

| Metric | Description |
|--------|-------------|
| experiments_analyzed | Total experiments included in synthesis |
| tensions_resolved | Count of T1–T4 tensions resolved with clear verdicts |
| closed_lines_count | Number of permanently closed lines of investigation |
| open_questions_ranked | Ordered list of remaining open questions with priors |
| accuracy_gap_current | Best model accuracy (0.4504) minus best breakeven WR available (29.6% at 20:3) |
| expectancy_gap_current | Best model expectancy (-$0.001) vs. $0.00 breakeven |
| architecture_recommendation | Updated canonical architecture specification |
| cpcv_vs_walkforward_gap | Expectancy divergence between CPCV and walk-forward evaluation |
| label_geometry_viability_prior | Estimated probability that Phase 1 label training yields positive expectancy |

### Sanity Checks

| Check | Expected | Failure means |
|-------|----------|---------------|
| SC-S1: All 22+ experiments accounted for | Every experiment in RESEARCH_LOG.md has an entry in the evidence matrix | Analysis is incomplete |
| SC-S2: No contradictory verdicts | Each tension resolves to a single consistent verdict | Logical error in synthesis |
| SC-S3: Closed lines have complete evidence chains | Every "closed" line cites >=2 supporting experiments | Premature closure |
| SC-S4: Open questions have falsifiable hypotheses | Each ranked question has a specific direction and magnitude | Vague research agenda |

## Baselines

| Baseline | Source | Value | Notes |
|----------|--------|-------|-------|
| Prior synthesis verdict | R6 (2026-02-17) | CONDITIONAL GO | CNN+GBT Hybrid recommended |
| Updated synthesis verdict | SYNTHESIS.md (2026-02-24) | GO (implicit) | XGB tuning + label design as next steps |
| Best model accuracy | XGB tuning (2026-02-25) | 0.4504 (CPCV) | On 20 features, tuned params |
| Best model expectancy | XGB tuning (2026-02-25) | -$0.001 (CPCV) / -$0.140 (WF) | At knife-edge of breakeven |
| Oracle edge | R7 (2026-02-17) | $4.00/trade, WR 64.3% | At 10:5 geometry |
| Label geometry oracle data | Label sensitivity (2026-02-26) | 123 geometries mapped, peak $4.13 | $5.00 gate failed but breakeven WR data is key finding |

## Success Criteria (immutable once RUN begins)

- [ ] **SC-1**: Evidence matrix covers all 22+ experiments × 5 strategic questions (Q1–Q5)
- [ ] **SC-2**: Go/no-go verdict rendered with explicit decision rule and full evidence chain
- [ ] **SC-3**: Label geometry strategic pivot documented — why breakeven WR (not oracle ceiling) is the correct metric, with PnL projections at 4 candidate geometries
- [ ] **SC-4**: Closed lines of investigation compiled — each with >=2 supporting experiments and an evidence chain
- [ ] **SC-5**: Open questions ranked by priority — each with falsifiable hypothesis, success criterion, compute estimate, and prior probability
- [ ] **SC-6**: Architecture status updated — CNN closed (evidence chain), XGB plateau (evidence chain), GBT canonical (justification)
- [ ] **SC-7**: Statistical limitations inventory updated — abort miscalibration, WF-CPCV divergence, long/short asymmetry, single-year regime dependence
- [ ] **SC-8**: Single highest-priority next experiment recommended with full specification
- [ ] **SC-9**: Synthesis document produced at `.kit/results/synthesis-v2/analysis.md`
- [ ] **SC-10**: `.kit/SYNTHESIS.md` updated with latest findings
- [ ] No sanity check failures

## Minimum Viable Experiment

Before the full synthesis, verify data availability and cross-reference the 3 most recent experiments:

1. **Data access gate.** Confirm all result files exist for: label-design-sensitivity, xgb-hyperparam-tuning, e2e-cnn-classification. Load `metrics.json` for each. Assert all 3 load without error.

2. **Quick tension check.** Cross-reference the 3 most recent findings:
   - XGB tuning: accuracy plateau at 0.4504 (0.33pp range)
   - Label sensitivity: breakeven WR at 15:3 = 33.3%, at 20:3 = 29.6%
   - E2E CNN: GBT beats CNN by 5.9pp

   Assert: These 3 findings are internally consistent (XGB accuracy plateau + low breakeven WR = geometry is the lever, not tuning; CNN closure is independent). If contradictory, investigate before full synthesis.

3. **Prior synthesis delta.** Load SYNTHESIS.md (2026-02-24). Identify which of its 4 recommendations have been acted on:
   - [x] XGBoost hyperparameter tuning → DONE (Outcome C)
   - [ ] Label design sensitivity → PARTIALLY DONE (Phase 0 only, abort at oracle gate)
   - [ ] Walk-forward as primary metric → NOTED but not formalized
   - [ ] Regime-conditional trading → NOT STARTED

   Assert: At least 1 recommendation has been completed. If 0, the synthesis has no new input.

Pass all 3 gates → proceed to full protocol.

## Full Protocol

### Step 1: Load ALL Results

Read the following files (no new computation):

**Foundation (R1–R4):**
- `.kit/results/subordination-test/metrics.json`
- `.kit/results/info-decomposition/metrics.json`
- `.kit/results/info-decomposition/analysis.md`
- `.kit/results/book-encoder-bias/metrics.json`
- `.kit/results/temporal-predictability/metrics.json`
- `.kit/results/temporal-predictability/analysis.md`

**Temporal Chain (R4b–R4d):**
- `.kit/results/temporal-predictability-event-bars/analysis.md`
- `.kit/results/temporal-predictability-completion/analysis.md`
- `.kit/results/temporal-predictability-dollar-tick-actionable/analysis.md`

**CNN Pipeline (9B–9E, R3b):**
- `.kit/results/hybrid-model-training/metrics.json`
- `.kit/results/cnn-reproduction-diagnostic/metrics.json`
- `.kit/results/r3-reproduction-pipeline-comparison/metrics.json`
- `.kit/results/hybrid-model-corrected/metrics.json`
- `.kit/results/e2e-cnn-classification/metrics.json`
- `.kit/results/r3b-genuine-tick-bars/metrics.json`

**Oracle & Optimization:**
- `.kit/results/oracle-expectancy/metrics.json`
- `.kit/results/xgb-hyperparam-tuning/metrics.json`
- `.kit/results/xgb-hyperparam-tuning/analysis.md`
- `.kit/results/label-design-sensitivity/metrics.json`
- `.kit/results/label-design-sensitivity/analysis.md`
- `.kit/results/label-design-sensitivity/oracle_heatmap_data.json`

**Other:**
- `.kit/results/synthesis/metrics.json` (prior R6 synthesis)
- `.kit/results/full-year-export/metrics.json`
- `.kit/SYNTHESIS.md` (prior synthesis document, 2026-02-24)
- `.kit/RESEARCH_LOG.md` (cumulative findings)

### Step 2: Compile Evidence Matrix

Build a complete experiment × question matrix showing how each experiment contributes to each strategic question. Include ALL 22+ experiments.

**Rows:** Every experiment (R1, R2, R3, R4, R4b, R4c, R4d, R6, R7, 9B, 9C, 9D, 9E, R3b-original, R3b-genuine, full-year-export, e2e-cnn-classification, xgb-hyperparam-tuning, bidir-reexport, label-design-sensitivity, infrastructure phases, TB-Fix).

**Columns:**
- Q1: Go/No-Go Status
- Q2: Highest-Priority Next Experiment
- Q3: Architecture and Feature Set
- Q4: What Has Been Eliminated
- Q5: Statistical Limitations

Each cell: verdict + key metric + confidence level.

### Step 3: Resolve Tensions T1–T4

For each tension, cite specific metrics from loaded results and produce a clear resolution with confidence level.

**T1: What Is the Go/No-Go Status After All Evidence?**

Decision rule:
```
GO: At least one unexplored high-leverage intervention exists with
    >50% prior probability of closing the viability gap.
CONDITIONAL GO: Interventions exist but <50% prior, or require
    significant additional data/compute.
NO-GO: All plausible interventions have been exhausted.
```

Evidence to weigh:
- Oracle edge: $4.00/trade (R7) — exploitable edge exists
- Best model: -$0.001/trade CPCV (XGB tuning) — $0.001 from breakeven
- Walk-forward: -$0.140/trade — deployment-realistic estimate is worse
- Label geometry: breakeven WR at 15:3 = 33.3% — 12pp below model's 45%
- Accuracy transfer uncertainty: unknown (Phase 1 never ran)
- XGB plateau: 0.33pp accuracy range — tuning exhausted
- CNN closure: 5.9pp deficit — no viable CNN path

**T2: Was Optimizing Oracle Net Exp the Wrong Variable?**

Compare oracle net exp ranking vs. breakeven WR ranking for the 123 mapped geometries. Show that the top-10 by oracle net exp (moderate ratios 1.6:1–2.7:1) are NOT the top-10 by model viability (high ratios 5:1+). Quantify the inverse correlation in the high-ratio region.

**T3: Can the Model's ~45% Accuracy Transfer to High-Ratio Geometries?**

This is the central unresolved question. Compile all available evidence:
- Label distribution shifts with geometry (from Phase 0 data: monotonic trade count reduction with wider targets)
- Long recall already 0.149 at 10:5 — what happens at 15:3 where hold fraction increases?
- XGB accuracy plateau (0.33pp range) suggests the feature set determines accuracy, not the label distribution — potential positive signal for transfer
- Counter-evidence: more imbalanced labels may cause accuracy to drop via majority-class prediction

Assign a probability estimate (calibrated) with explicit reasoning.

**T4: Is the CNN Line Truly Closed?**

Evidence chain:
- E2E CNN classification: 5.9pp worse than GBT (3 reasons: spatial signal encodes variance not boundaries, long recall 0.21, hold prediction dominance)
- 9E hybrid: expectancy -$0.37/trade
- 9B: CNN R²=-0.002 (pipeline broken, not architecture — but normalization fix in 9E still shows non-viable expectancy)
- R3b-genuine: tick_100 R²=0.124 but p=0.21, single-fold driven
- CNN regression R²=0.084 is genuine but does not convert to classification

Verdict: CLOSED for classification. Regression signal acknowledged but non-actionable.

### Step 4: Produce Recommendations

Rank-order next experiments by expected information value. For each:
- State the falsifiable hypothesis with direction and magnitude
- Define the binary success criterion
- Estimate compute cost (wall-clock, GPU hours, dollar cost)
- Assign prior probability of success with explicit reasoning

Candidates to evaluate:
- (A) Phase 1 label geometry training — XGBoost at 4 geometries selected by breakeven WR
- (B) Regime-conditional trading — Q1-Q2 only strategy
- (C) 2-class short/not-short reformulation
- (D) Cost reduction investigation (limit orders, maker rebates)
- (E) Class-weighted XGBoost with PnL-aligned loss
- (F) Accept no-edge and close the project

### Step 5: Update Closed Lines

Compile the definitive list of what has been eliminated with the evidence chain for each. Each closed line must cite >=2 supporting experiments.

Known closed lines:
1. Temporal encoder (R4, R4b, R4c, R4d — 0/168+ passes)
2. Message encoder (R2 — 0/40 passes)
3. Event-driven bars over time bars (R1 — 0/3 passes)
4. CNN for classification (e2e-cnn, 9E, 9B)
5. CNN+GBT hybrid for trading (9E, e2e-cnn)
6. XGBoost hyperparameter tuning as accuracy lever (xgb-tuning — 0.33pp plateau)
7. Oracle net expectancy as viability metric (label-design-sensitivity — inverse correlation with model viability)

### Step 6: Produce Decision Document

Write the updated synthesis to `.kit/results/synthesis-v2/analysis.md` with this structure:

```
1. Executive Summary
   - Updated go/no-go verdict
   - Single highest-priority next experiment
   - Architecture recommendation
   - Key numbers: accuracy gap, breakeven WR gap, expectancy gap

2. Complete Evidence Matrix (22+ experiments × 5 questions)

3. Strategic Pivot: Breakeven WR vs Oracle Ceiling
   - Why the project was optimizing the wrong variable
   - Inverse correlation between oracle net exp and model viability
   - Payoff structure analysis with PnL projections at 4 geometries

4. Tension Resolutions (T1–T4)
   - Each with verdict, confidence, evidence chain

5. Architecture Status (Updated)
   - CNN line: CLOSED
   - GBT on 20 features: canonical
   - XGBoost tuning: exhausted
   - Feature set: binding constraint

6. Closed Lines of Investigation (7+ lines with evidence chains)

7. Open Questions (Ranked by Priority)
   - Each with hypothesis, success criterion, compute, prior

8. Statistical Limitations (Updated)
   - Abort criterion miscalibration
   - Walk-forward divergence (-$0.140 vs -$0.001)
   - Long/short recall asymmetry (0.149 vs 0.634)
   - Single-year regime dependence
   - volatility_50 feature monopoly (49.7% gain share)

9. Recommendation
   - Explicit next action with full justification
```

Also update `.kit/SYNTHESIS.md` with the new findings.

## Resource Budget

**Tier:** Quick

- Max GPU-hours: 0
- Max wall-clock time: 30 minutes
- Max training runs: 0 (analysis only)
- Max seeds per configuration: N/A

### Compute Profile
```yaml
compute_type: cpu
estimated_rows: 0
model_type: other
sequential_fits: 0
parallelizable: false
memory_gb: 1
gpu_type: none
estimated_wall_hours: 0.5
```

### Wall-Time Estimation

| Phase | Work | Estimate |
|-------|------|----------|
| MVE gates | Load 3 recent results, verify consistency | ~2 min |
| Step 1: Load all results | Read 22+ metrics/analysis files | ~5 min |
| Step 2: Evidence matrix | Cross-reference experiments × questions | ~5 min |
| Step 3: Tension resolution | Analyze T1–T4 with specific metrics | ~5 min |
| Step 4: Recommendations | Rank-order next experiments | ~3 min |
| Step 5: Closed lines | Compile with evidence chains | ~3 min |
| Step 6: Document production | Write synthesis-v2/analysis.md + SYNTHESIS.md | ~7 min |
| **Total** | | **~30 min** |

## Abort Criteria

- **Result files missing:** If >3 of the listed result files are missing or unreadable → STOP. The synthesis cannot proceed without the input data.
- **Internal contradiction in recent findings:** If xgb-tuning accuracy and label-sensitivity breakeven WR data are contradictory (e.g., reported accuracy at 10:5 differs by >5pp between experiments) → STOP. Investigate data integrity before synthesizing.
- **Wall-clock > 45 minutes:** Abort and deliver partial synthesis with completed sections.

## Confounds to Watch For

1. **Confirmation bias toward GO.** The synthesis follows a successful (if aborted) label-sensitivity experiment that revealed a promising lever. Guard against treating the breakeven WR argument as a proven result — it remains a hypothesis that requires Phase 1 training to test. The prior probability assignment must be calibrated, not optimistic.

2. **Walk-forward vs. CPCV expectancy divergence.** CPCV expectancy (-$0.001) paints a near-breakeven picture. Walk-forward (-$0.140) is dramatically worse. The synthesis must explicitly address which estimate to use for the go/no-go decision. Using CPCV alone would be misleading.

3. **Single-year data.** All 22+ experiments use 2022 MES data only. GBT's Q1-Q2 profitability and Q3-Q4 losses may be year-specific regime effects. The synthesis should flag this as the single largest threat to external validity.

4. **volatility_50 feature monopoly (49.7% gain share).** If the model's performance is almost entirely determined by one feature's relationship to the label, the model is fragile to volatility regime shifts. The synthesis should assess whether this risk changes the go/no-go calculus.

5. **Accuracy-transfer uncertainty is large.** The model's 45% accuracy at 10:5 labels may not transfer to 15:3 or 20:3 labels. The label distribution shifts substantially (more holds, fewer directional labels). The synthesis must NOT assume accuracy transfers — it must quantify the uncertainty and present both scenarios (transfers vs. doesn't).

6. **Anchoring on the $0.001 gap.** The tuned model is $0.001/trade from CPCV breakeven. This creates a psychological anchor suggesting the model is "almost there." But walk-forward says it's $0.14/trade away. The synthesis should present both numbers and let the evidence determine which is more relevant.

## Deliverables

```
.kit/results/synthesis-v2/
  analysis.md              # Full synthesis document (8 sections)
  evidence_matrix.csv      # 22+ experiments × 5 questions
  architecture_decision.json  # Updated architecture recommendation
  metrics.json             # Summary metrics for synthesis

.kit/SYNTHESIS.md          # Updated top-level synthesis (replaces 2026-02-24 version)
```

## Exit Criteria

- [ ] All 22+ experiment result files loaded and cross-referenced
- [ ] Complete evidence matrix produced (experiments × questions)
- [ ] Go/no-go decision updated with full evidence chain
- [ ] Label geometry as strategic lever documented with PnL projections
- [ ] Closed lines of investigation compiled with evidence chains
- [ ] Architecture status updated (CNN closed, XGB plateau, GBT canonical)
- [ ] Open questions ranked by priority with success priors
- [ ] Statistical limitations updated (abort miscalibration, WF divergence, feature monopoly)
- [ ] Single highest-priority next experiment recommended
- [ ] Synthesis document produced at `.kit/results/synthesis-v2/analysis.md`
- [ ] `.kit/SYNTHESIS.md` updated with latest findings
- [ ] Summary entry ready for `.kit/RESEARCH_LOG.md`
