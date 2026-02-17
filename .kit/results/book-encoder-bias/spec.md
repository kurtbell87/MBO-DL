# Phase R3: Book Encoder Inductive Bias [Research]

**Spec**: TRAJECTORY.md §2.3 (sufficient statistics / CNN embedding)
**Depends on**: Phase 4 (feature-computation) — needs raw book snapshot export (B.1 format).
**Unlocks**: Phase 6 (synthesis) — provides spatial encoder architecture recommendation.
**GPU budget**: ≤ 2 hours (of the 4-hour project cap). **Max runs**: 6.

---

## Hypothesis

The CNN spatial encoder (Conv1d on the 20-level price ladder) embeds an
assumption that predictive patterns are **spatially local** — adjacent price
levels share structure. This experiment tests whether that assumption holds
for MES order book data by comparing three inductive biases:

1. **Local spatial** (CNN): Learns from adjacent-level patterns via Conv1d kernels.
2. **Global spatial** (Attention): Every level attends to every other level — captures long-range correlations (e.g., hidden liquidity 8 levels deep).
3. **No spatial** (MLP): Treats the 40-element flattened book as an unstructured vector.

If the CNN matches or exceeds attention, the local prior is justified and the
v0.6 architecture is validated. If attention dominates, the spatial encoder
needs redesign. If the MLP matches both, spatial structure is irrelevant and
the simplest representation wins.

---

## Data

### Source

Raw book snapshots from Phase 4 Track B.1 export: shape `(20, 2)` per bar.
- Rows 0–9: Bids (deepest first → best bid at row 9).
- Rows 10–19: Asks (best ask at row 10 → deepest last).
- Column 0: `price_delta = (price - mid_price) / TICK_SIZE` (ticks from mid).
- Column 1: `log1p(size)`, z-scored across the day.

### Sampling

- Use **19 non-rollover trading days** spread across 2022 (≈1 per month plus extras in volatile months). Select days with representative volume profiles — exclude half-days and holidays.
- Target: ~50k–100k bars total (depending on bar type chosen; use volume-100 bars if R1 is not yet complete, or the R1-recommended bar type if available).
- Flatten each `(20, 2)` snapshot to a 40-element vector for the MLP; keep `(20, 2)` for CNN and Attention.

### Target

`return_5` (5-bar forward return), consistent with R2's primary horizon.

### Train/Test Splits

5-fold **expanding-window** time-series CV:

| Fold | Train days | Test days |
|------|-----------|-----------|
| 1 | days 1–4 | days 5–7 |
| 2 | days 1–7 | days 8–10 |
| 3 | days 1–10 | days 11–13 |
| 4 | days 1–13 | days 14–16 |
| 5 | days 1–16 | days 17–19 |

No shuffling. No leakage across days.

---

## Models

All three models are matched to **~12k parameters** (±10%) for fair comparison.
All use the same optimizer, learning rate schedule, and training budget.

### Model A: CNN (Local Spatial Prior)

```
Input: (B, 2, 20)          # channels-first: (price_delta, size) × 20 levels
Conv1d(2 → 32, kernel=3, padding=1) + BatchNorm1d(32) + ReLU
Conv1d(32 → 32, kernel=3, padding=1) + BatchNorm1d(32) + ReLU
AdaptiveAvgPool1d(1)        # → (B, 32)
Linear(32 → 16) + ReLU
Linear(16 → 1)              # scalar return prediction
```
**Param count**: ~2×(2×32×3 + 32) + 2×(32×32×3 + 32) + (32×16 + 16) + (16×1 + 1) ≈ 7.3k
(Adjust channel width to hit ~12k target if needed.)

### Model B: Attention (Global Spatial Prior)

```
Input: (B, 20, 2)           # 20 tokens (levels), each with 2 features
Linear(2 → 16)              # project to d_model=16
+ Learned positional embedding (20 × 16)
MultiheadAttention(d_model=16, nhead=2, 1 layer)
  → Q, K, V projections (16 → 16 each)
  → feedforward: Linear(16 → 32) + ReLU + Linear(32 → 16)
  → LayerNorm + residual
Mean-pool over 20 tokens     # → (B, 16)
Linear(16 → 16) + ReLU
Linear(16 → 1)
```
**Param count**: project(2×16+16) + pos(20×16) + QKV(3×16×16+3×16) + out(16×16+16) + FF(16×32+32+32×16+16) + LN(2×16×2) + head(16×16+16+16×1+1) ≈ 3.3k
(Increase d_model or add a second head layer to approach ~12k.)

### Model C: MLP (No Spatial Prior)

```
Input: (B, 40)              # flattened book vector
Linear(40 → 64) + BatchNorm1d(64) + ReLU + Dropout(0.1)
Linear(64 → 32) + BatchNorm1d(32) + ReLU + Dropout(0.1)
Linear(32 → 1)
```
**Param count**: (40×64+64) + 64×2 + (64×32+32) + 32×2 + (32×1+1) ≈ 5.0k
(Adjust hidden widths to hit ~12k target.)

### Parameter Matching Protocol

Before training, print each model's `sum(p.numel() for p in model.parameters())`.
All three must be within ±10% of each other. Adjust hidden dimensions until matched.
Log final counts in the results table.

---

## Training Protocol

| Setting | Value |
|---------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-3 (cosine decay to 1e-5) |
| Weight decay | 1e-4 |
| Batch size | 512 |
| Epochs | 50 (early stopping patience = 10, monitor val loss) |
| Loss | MSE on `return_5` |
| Metric | Out-of-sample R² per fold |

Each fold trains from scratch (no warm-starting across folds).

### Input Normalization

- Price deltas: already in ticks (from B.1 export). No further normalization.
- Sizes: already `log1p` + z-scored per day (from B.1 export). No further normalization.
- If any NaN/inf in inputs, replace with 0 and log count.

---

## Analysis

### Primary Comparison

For each of the 5 folds, record out-of-sample R² for all three models.
Report:

```
Table 1: Model × Fold → R²
         + row for mean R² and std R²
```

### Pairwise Statistical Tests

Three pairwise comparisons on the 5-fold R² vectors:
1. CNN vs. Attention
2. CNN vs. MLP
3. Attention vs. MLP

- **Test**: Paired t-test (if Shapiro-Wilk on differences p > 0.05) or Wilcoxon signed-rank (otherwise).
- **Correction**: Holm-Bonferroni across 3 comparisons.
- **Threshold**: Corrected p < 0.05.
- Report raw p, corrected p, effect size (Cohen's d), and 95% CI on R² difference.

### Sufficiency Test (§2.3)

Using the CNN model's penultimate layer (the 16-dim vector before the final linear), extract embeddings for all test-set bars. Then:

1. Train a **linear probe** (Ridge regression, α chosen by nested CV) from the 16-dim CNN embedding → `return_5`.
2. Train the same linear probe from the **raw 40-dim** flattened book → `return_5`.
3. Compare R² values.

```
Table 2: Representation × Fold → R² (linear probe)
         CNN-16d vs. Raw-40d
         + paired t-test (1 comparison, no correction needed)
```

**Interpretation**: If `R²(CNN-16d) ≥ 0.9 × R²(Raw-40d)`, the CNN compresses without meaningful information loss → the 16-dim embedding is a sufficient statistic for the book.

---

## Decision Rules

```
RULE 1 — Spatial structure test:
  If (CNN ≈ Attn) AND (CNN >> MLP) at corrected p < 0.05:
    → Spatial structure matters. Local prior is sufficient.
    → DECISION: Use Conv1d encoder (v0.6 architecture validated).

  If (Attn >> CNN >> MLP) at corrected p < 0.05:
    → Spatial structure matters, but long-range.
    → DECISION: Replace Conv1d with attention-based spatial encoder.

  If all three models ≈ equal (no significant pairwise differences):
    → No exploitable spatial structure in book.
    → DECISION: Use MLP on flattened book (simplest).

  If (MLP >> CNN) or (MLP >> Attn):
    → Spatial priors actively hurt. Possible overfitting to structure.
    → DECISION: Use MLP. Investigate why spatial models underperform.

RULE 2 — Sufficiency test:
  If R²(CNN-16d) ≥ 0.9 × R²(Raw-40d):
    → CNN embedding is a sufficient statistic.
    → DECISION: Use 16-dim embedding as spatial encoder output.

  If R²(CNN-16d) < 0.9 × R²(Raw-40d):
    → CNN loses information.
    → DECISION: Increase embedding dimension or revisit architecture.

"≈" means: pairwise corrected p ≥ 0.05 (no significant difference).
">>" means: pairwise corrected p < 0.05 AND Cohen's d > 0.5.
```

---

## Implementation

```
File: research/R3_book_encoder_bias.py

Dependencies:
  - torch (Conv1d, MultiheadAttention, Linear, AdamW, CosineAnnealingLR)
  - scikit-learn (Ridge, cross_val_score for nested CV in sufficiency test)
  - scipy (ttest_rel, wilcoxon, shapiro)
  - polars (data loading)
  - numpy

Input:
  - Book snapshots: Phase 4 Track B.1 export (parquet or CSV, 20×2 per bar)
  - Returns: `return_5` from Phase 4 Track A export

Output directory: .kit/results/R3_book_encoder/
```

---

## Deliverables

```
.kit/results/R3_book_encoder/
├── model_comparison.csv        # Table 1: model × fold → R²
├── pairwise_tests.csv          # Pairwise stats: pair, raw_p, corrected_p, cohens_d, ci_lo, ci_hi
├── sufficiency_test.csv        # Table 2: representation × fold → R² (linear probe)
├── param_counts.json           # {cnn: N, attention: N, mlp: N}
├── training_curves.csv         # epoch × model × fold → train_loss, val_loss
├── analysis.md                 # Human-readable summary with decision
└── models/                     # Saved best checkpoints per fold × model (optional)
```

### Required Outputs in `analysis.md`

1. Table 1 (model comparison) with mean ± std R².
2. Table 2 (pairwise tests) with all statistics.
3. Table 3 (sufficiency test) with compression ratio and R² retention.
4. Explicit statement of which Decision Rule was triggered.
5. Architecture recommendation for Phase 6 synthesis.

---

## Exit Criteria

- [ ] All three models (CNN, Attention, MLP) trained on identical data splits
- [ ] Parameter counts within ±10% of each other, logged
- [ ] 5-fold expanding-window CV completed for all models
- [ ] Pairwise significance tests with Holm-Bonferroni correction reported
- [ ] Effect sizes (Cohen's d) and 95% CIs reported for all comparisons
- [ ] Shapiro-Wilk normality check on R² differences (to choose t-test vs. Wilcoxon)
- [ ] CNN embedding sufficiency test completed (linear probe at 16d vs. 40d)
- [ ] Spatial encoder architecture recommended with justification
- [ ] All results written to `.kit/results/R3_book_encoder/`
- [ ] `analysis.md` contains explicit decision rule outcome
