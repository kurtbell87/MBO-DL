# Phase R2: Information Decomposition [Research]

**Spec**: TRAJECTORY.md §2.2 (theory), §7.2 (simplification cascade), §8.7 (corrections)
**Depends on**: Phase 4 (feature-computation) — needs exported features + raw representations.
**Unlocks**: Phase 6 (synthesis) — provides architecture go/no-go decisions.

---

## Hypothesis

The predictive information about future returns decomposes across three sources:
- **Spatial** (book state): $I(\mathbf{B}_t; r_{t+h})$
- **Message** (intra-bar events): $I(\mathbf{M}_t; r_{t+h} | \mathbf{B}_t)$
- **Temporal** (history): $I(\mathbf{H}_t; r_{t+h} | \mathbf{B}_t, \mathbf{M}_t)$

Each term maps to an encoder stage. If a term ≈ 0, the corresponding encoder adds no value.

---

## Method

### Tier 1: Hand-Crafted Proxies (Cheap)

Train linear model + shallow MLP (2 hidden layers, 64 units) to predict `return_5` from:

| Config | Input | Output Metric |
|--------|-------|---------------|
| (a) | Track A hand-crafted features only | $R^2_{\text{bar}}$ |
| (b) | Track B raw book snapshot (flattened 40 features) | $R^2_{\text{book}}$ |
| (c) | Track B raw book + Category 6 message summaries | $R^2_{\text{book+msg\_summary}}$ |
| (d) | (c) + lookback window of 20 previous bars' book snapshots | $R^2_{\text{full\_summary}}$ |

**Cross-validation**: 5-fold expanding-window time-series CV. Report mean and std of R² across folds.

### Tier 2: Learned Message Encoder (Expensive, Definitive)

| Config | Input | Output Metric |
|--------|-------|---------------|
| (e) | Flattened book + LSTM (1 layer, 32 hidden) on raw MBO event sequence | $R^2_{\text{book+msg\_learned}}$ |
| (f) | Flattened book + 1-layer transformer (2 heads, d_model=32) on events | $R^2_{\text{book+msg\_attn}}$ |

- Truncate message sequences to max 500 events per bar (keep most recent if exceeded).
- Pad shorter sequences.
- Log truncation rate per bar type/parameter.
- Raw events retrieved from `DayEventBuffer` export.

### Analysis

Compute gaps:
- $R^2_{\text{book}} - R^2_{\text{bar}}$ → information lost by hand-crafting
- $R^2_{\text{book+msg\_summary}} - R^2_{\text{book}}$ → Tier 1: do summaries add value?
- $R^2_{\text{book+msg\_learned}} - R^2_{\text{book}}$ → Tier 2: do raw messages add value?
- $R^2_{\text{full\_summary}} - R^2_{\text{book+msg\_summary}}$ → does temporal history add value?

### Threshold Policy (Revised from §2.2)

Both conditions must hold for an encoder stage to be justified:
1. **Relative threshold**: R² gap > 20% of baseline R².
2. **Statistical threshold**: Paired t-test (or Wilcoxon if non-normal) on per-fold R² values, p < 0.05 after Holm-Bonferroni correction across comparisons.

### Multi-Horizon

Repeat for `return_1`, `return_5`, `return_20`, `return_100` to check if gaps vary by horizon.

---

## Decision Matrix (§7.2)

```
If Tier 2 gap > Tier 1 gap by significant margin:
  → Category 6 summaries are insufficient. Message encoder justified.
  → Hand-crafted features miss sequential patterns.

If Tier 2 gap ≈ Tier 1 gap:
  → Summaries capture message information. Message encoder adds complexity without benefit.

If both gaps ≈ 0:
  → Messages carry no incremental info beyond book state. Drop message encoder.

Architecture recommendation:
  If R²_book - R²_bar > relative ε AND significant:         Include spatial encoder (CNN)
  If R²_book+msg_learned - R²_book > relative ε AND sig:    Include message encoder
  If R²_full - R²_book+msg > relative ε AND significant:    Include temporal encoder (SSM)
```

---

## Implementation

```
research/R2_information_decomposition.py

Dependencies: torch (LSTM, transformer), scikit-learn (CV, metrics), polars, scipy
Input: Feature export from Phase 4 + raw event export from DayEventBuffer
```

---

## Deliverable

```
Table: return_horizon × representation × tier → (mean_R², std_R², per-fold R² values)
Table: Information gaps with relative magnitude, paired test p-values, corrected p-values

Critical comparison: Tier 1 vs Tier 2 gap analysis

Output: Recommended architecture from simplification cascade (§7.2):
  - Full three-level (message + spatial + temporal)
  - Two-level (spatial + temporal, no message encoder)
  - CNN + history (spatial + temporal, hand-crafted messages)
  - GBT baseline (hand-crafted features only)
```

---

## Exit Criteria

- [ ] Tier 1 R² gaps computed for all representation tracks × return horizons
- [ ] Tier 2 learned message encoder (LSTM or transformer) tested on raw event sequences
- [ ] Threshold policy applied: relative gap > 20% AND statistically significant
- [ ] Architecture decision from §7.2 simplification cascade filled in with Tier 2 results
- [ ] Truncation rate logged for message sequences
- [ ] All comparisons report raw p-values, corrected p-values, and power analysis
- [ ] Results written to `.kit/results/R2_decomposition/`
