# Phase 6: Synthesis [Research]

**Spec**: TRAJECTORY.md §11 Phase 6
**Depends on**: ALL prior phases — 1, 2, 3, R1, 4, 5, R2, R3, R4.
**Unlocks**: Next spec (model architecture build).

---

## Objective

Combine findings from all engineering and research phases into a single decision document that determines whether and how to proceed with model training.

---

## Questions to Answer

### Go/No-Go Decisions

1. **Does the oracle have positive expectancy?** (Phase 3 results)
   - If no → stop. The oracle's signals are not worth replicating after costs.
   - If yes → proceed to supervised training.

2. **Which labeling method is preferred?** (Phase 3 comparison)
   - First-to-hit vs. triple barrier: expectancy, label-return correlation, stability.

### Architecture Decisions

3. **Which bar type is optimal?** (R1 + R4 + Phase 5 bar comparison)
   - Subordination test results (R1): normality, heteroskedasticity.
   - Temporal predictability (R4): which bar type maximizes exploitable AR structure.
   - Feature signal quality (Phase 5): aggregate MI across Track A.
   - Synthesize into a single bar type recommendation for all downstream work.

4. **Which encoder stages are needed?** (R2 results → §7.2 simplification cascade)
   - Spatial encoder: justified if $R^2_{\text{book}} - R^2_{\text{bar}}$ > relative ε + significant.
   - Message encoder: justified if Tier 2 $R^2_{\text{book+msg\_learned}} - R^2_{\text{book}}$ > relative ε + significant.
   - Temporal encoder: justified if $R^2_{\text{full}} - R^2_{\text{book+msg}}$ > relative ε + significant.
   - Output: one of the 4 architecture options from §7.2.

5. **What is the spatial encoder architecture?** (R3 results)
   - CNN vs. attention vs. MLP on raw book snapshots.

### Feature Selection

6. **Which features are most predictive at which horizons?** (Phase 5)
   - Top-20 stability-selected features per bar type.
   - Input selection for GBT interpretable baseline.
   - Decay characteristics (short-horizon signals vs. regime indicators).

### Robustness

7. **Does the subordination model hold for MES?** (R1)
   - If yes → event-driven bars are well-motivated theoretically and empirically.
   - If weakly → bars work pragmatically but theoretical grounding is thin.

8. **Is the book state sufficient or do messages add value?** (R2 Tier 2)
   - Definitive message encoder go/no-go.

9. **Is the oracle robust across regimes?** (Phase 3 regime stratification)
   - Cross-regime stability score.
   - If fragile → need regime filter or different approach.

10. **What are the statistical limitations?** (Phase 5 + R1-R4)
    - Power analysis: what effect sizes can we detect?
    - Comparisons that failed correction: suggestive but inconclusive.
    - Single-year (2022) limitation: bear market + hiking cycle, may not generalize.

---

## Document Structure

```
results/synthesis.md

1. Executive Summary
   - Go/no-go: proceed with model training? YES/NO + justification
   - Recommended architecture (from §7.2 cascade)
   - Recommended bar type + parameters
   - Recommended labeling method

2. Oracle Expectancy (Phase 3)
   - Summary table: best config → metrics
   - Regime stability assessment
   - Oracle comparison: first-to-hit vs. triple barrier

3. Bar Type Selection (R1 + R4 + Phase 5)
   - Subordination test results
   - Temporal predictability results
   - Signal quality comparison
   - Final recommendation with justification

4. Architecture Decision (R2 + R3)
   - Information decomposition results (Tier 1 + Tier 2)
   - Book encoder comparison
   - Simplification cascade outcome
   - Final architecture recommendation

5. Feature Ranking (Phase 5)
   - Top features per horizon
   - Decay characteristics
   - GBT baseline input selection

6. Limitations and Caveats
   - Power analysis summary
   - Failed corrections
   - Single-year data caveat
   - Regime fragility (if applicable)

7. Next Steps
   - Model build spec inputs (architecture, bar type, features, labels)
   - Open questions for future research
```

---

## Exit Criteria

- [ ] All 10 questions above answered with supporting data
- [ ] Go/no-go decision documented
- [ ] Architecture recommendation from §7.2 simplification cascade
- [ ] Bar type recommendation with cross-validation from R1, R4, Phase 5
- [ ] Feature selection for GBT baseline
- [ ] Statistical limitations documented
- [ ] Synthesis document produced at `results/synthesis.md`
