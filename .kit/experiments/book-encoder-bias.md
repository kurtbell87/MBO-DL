# Phase R3: Book Encoder Inductive Bias [Research]

**Spec**: TRAJECTORY.md §2.3 (sufficient statistics / CNN embedding)
**Depends on**: Phase 4 (feature-computation) — needs raw book snapshot export.
**Unlocks**: Phase 6 (synthesis) — provides spatial encoder architecture recommendation.

---

## Hypothesis

The CNN's spatial prior (price ladder ordering → Conv1d) is well-suited for the order book if predictive patterns are spatially local (adjacent price levels). If long-range spatial correlations exist (e.g., large hidden order 8 levels deep), an attention-based encoder may outperform. If no spatial structure exists, a simple MLP on flattened features suffices.

---

## Method

Train three architectures on raw book snapshots → `return_5`:

| Model | Architecture | Input |
|-------|-------------|-------|
| CNN | v0.6 architecture scaled down (Conv1d on 20-level price ladder) | (20, 2) |
| Attention | Each level attends to all 20 levels, 2 heads, 1 layer | (20, 2) |
| MLP | Fully connected on flattened 40-feature book vector | 40 |

All models matched in parameter count to ensure fair comparison.

### Cross-Validation

- 5-fold expanding-window time-series CV.
- Record per-fold R² for each model.

### Statistical Comparison

- Paired t-test on per-fold R² values for all 3 pairwise comparisons.
- Holm-Bonferroni correction (3 comparisons).
- Report whether differences are significant at corrected p < 0.05.

---

## Decision Rules

```
If R²_CNN ≈ R²_attn >> R²_MLP:
  → Spatial structure matters, local prior is fine → use CNN

If R²_attn >> R²_CNN >> R²_MLP:
  → Spatial structure matters, but long-range → consider attention encoder

If R²_MLP ≈ R²_CNN ≈ R²_attn:
  → No spatial structure to exploit → MLP on flattened book is sufficient
```

### Sufficiency Test (§2.3)

Compare R² of predicting `return_5` from the CNN embedding (d-dimensional) vs. the raw 40-dim book vector. If CNN achieves comparable R² with d << 40, it compresses without losing predictive information.

---

## Implementation

```
research/R3_book_sufficiency.py

Dependencies: torch (CNN, MultiheadAttention, MLP), scikit-learn (CV), polars
Input: Raw book snapshot export from Phase 4 (flattened 40-dim per bar)
```

---

## Deliverable

```
Table: model × fold → R² (with mean, std)
Table: Pairwise comparisons with test statistics, raw p, corrected p

Finding: Which inductive bias fits MES book data?
Decision: Spatial encoder architecture recommendation (CNN / attention / MLP)
```

---

## Exit Criteria

- [ ] CNN, attention, and MLP trained on identical data splits
- [ ] Pairwise significance tests with Holm-Bonferroni correction
- [ ] CNN embedding sufficiency test (R² at reduced dimensionality)
- [ ] Spatial encoder architecture recommended
- [ ] Results written to `.kit/results/R3_book_encoder/`
