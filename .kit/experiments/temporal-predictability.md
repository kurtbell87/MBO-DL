# Phase R4: Entropy Rate and Temporal Predictability [Research]

**Spec**: TRAJECTORY.md §2.4 (entropy rate theory)
**Depends on**: Phase 1 (bar-construction) — needs bar builders. Phase R1 — uses recommended bar type parameters.
**Unlocks**: Phase 6 (synthesis) — provides final bar type recommendation for temporal encoder.

---

## Hypothesis

The optimal bar type for the temporal encoder maximizes exploitable temporal structure — i.e., maximizes autoregressive R² for predicting returns at horizons h > 1 (separating genuine temporal structure from trivial 1-lag autocorrelation).

---

## Method

### Bar Configurations

Use the recommended parameters from R1 plus a few alternatives for comparison. At minimum:
- Volume bars: V from R1 recommendation
- Tick bars: K from R1 recommendation
- Time bars: 1s, 60s (control)
- Dollar bars: D from R1 recommendation (if tested in R1)

All bar types matched to comparable daily bar counts where possible.

### Autoregressive R² Computation

For each bar type, compute:
- **Input**: Previous 10 returns (return_{-1}, ..., return_{-10})
- **Target**: return_h for h ∈ {1, 5, 10, 20}
- **Models**:
  - Linear AR (captures linear temporal dependence)
  - GBT AR (captures nonlinear temporal structure)
- **CV**: 5-fold expanding-window time-series CV.

### Statistical Framework

- Report per-fold AR R² for each bar_type × horizon × model combination.
- Paired t-test across folds for bar type comparisons within each horizon × model.
- Holm-Bonferroni correction for bar type comparisons.

---

## Key Questions

1. Which bar type produces the most temporally predictable returns?
2. Is the temporal structure linear (AR sufficient) or nonlinear (GBT >> AR)?
3. At which horizons is temporal predictability strongest?

**Subtlety**: Highly autocorrelated returns might reflect spurious serial dependence (e.g., time bars during lunch → identical bars). The multi-horizon analysis (h > 1) controls for this — genuine temporal structure persists at h > 1, trivial 1-lag effects do not.

---

## Implementation

```
research/R4_entropy_rate.py

Dependencies: scikit-learn (LinearRegression, CV), xgboost, numpy, polars, scipy
Input: Bar return series from C++ bar builders (via export)
```

---

## Deliverable

```
Table: bar_type × horizon × model_type → AR_R² (with corrected p-values)

Finding: Which bar type produces the most temporally predictable returns?
         Linear vs. nonlinear temporal structure?

Decision: Final bar type recommendation for the temporal encoder.
```

---

## Exit Criteria

- [ ] AR R² computed for all bar_type × horizon × model combinations
- [ ] Linear AR vs. GBT AR compared (linear vs. nonlinear temporal structure)
- [ ] Holm-Bonferroni correction applied for bar type comparisons
- [ ] Multi-horizon analysis confirms genuine temporal structure (not trivial 1-lag)
- [ ] Final bar type recommendation for temporal encoder
- [ ] Results written to `.kit/results/R4_temporal/`
