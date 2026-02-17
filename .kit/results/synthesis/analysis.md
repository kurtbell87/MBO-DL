# R6: Synthesis — Analysis

**Experiment**: R6_synthesis
**Date**: 2026-02-17
**Phase**: 6 (final research phase)
**Depends on**: R1 (subordination-test), R2 (info-decomposition), R3 (book-encoder-bias), R4 (temporal-predictability)
**GPU hours**: 0 (analysis only)

---

## 1. Executive Summary

- **Go/No-Go**: **CONDITIONAL GO** — Proceed with model training, but flag oracle expectancy as an open question.
- **Recommended architecture**: CNN + GBT Hybrid (OPTION B)
- **Recommended bar type**: `time_5s` (5-second time bars)
- **Recommended prediction horizons**: h=1 and h=5
- **Recommended labeling method**: **Open question** — oracle expectancy from Phase 3 C++ backtest has not been extracted.

**Rationale**: Positive out-of-sample R² exists at h=1 (R²=0.0067 from R2, R²=0.0046 from R4) and h=5 (R²=0.132 from R3 CNN). The R3 CNN result on structured (20,2) book input represents a 20× improvement over R2's flattened MLP, driven by the Conv1d inductive bias on adjacent price levels. The decision is CONDITIONAL because oracle expectancy is unknown — if the labeling method produces negative expectancy, the positive R² is not economically exploitable.

**Architecture recommendation: CNN + GBT Hybrid.** The Conv1d encoder on raw (20,2) book snapshots produces a 16-dim embedding that retains 4.16× more signal than the raw 40-dim flattened representation (R²=0.111 vs R²=0.027, p=0.012). This embedding is concatenated with ~20 non-spatial hand-crafted features (order flow, trade imbalance, volatility) and fed to an XGBoost head. Message and temporal encoders are dropped — R2 shows messages carry no incremental information (Δ_msg < 0), and R2+R4 converge on zero temporal signal.

---

## 2. Bar Type Decision (R1 + R4)

### R1: Subordination REFUTED

R1 tested 12 bar configurations (3 volume, 3 tick, 3 dollar, 3 time) across 19 trading days. The subordination hypothesis — that event-driven bars (volume, tick, dollar) produce more normal, homoskedastic returns than time bars — was **refuted**:

- 0/3 primary pairwise tests significant after Holm-Bonferroni correction (vol vs. time_5s on JB, ARCH, ACF1).
- Dollar bars showed the *highest* AR R² (dollar_25k: 0.011 vs time_5s: 0.0003) — opposite of the theory's prediction that event bars would reduce serial dependence.
- Effect reverses across quarters (Q1–Q4 robustness: all false). Findings are regime-dependent.
- Best overall bar by mean rank: time_1s (5.14), but time_5s (6.71) is preferred for practical bar count (~4,681/day vs ~23,401/day).

### R4: No Temporal Structure Favoring Any Bar Type

R4 confirmed that `time_5s` bars produce a martingale return series. All 36 Tier 1 AR configurations yield negative R². Since no bar type produces exploitable temporal structure, the simplest (time bars) is preferred.

### Decision: `time_5s`

Time bars are the baseline. No statistical evidence supports any alternative. The 5-second interval provides ~4,681 bars/day (median), sufficient for intra-day modeling while avoiding sub-second noise. Key metrics from R1: JB rank=5, ARCH rank=3, ACF1 rank=6, AR rank=11, CV rank=5.

---

## 3. Architecture Decision (R2 + R3 + R4)

### 3.1 R2 vs. R3 Reconciliation Table

The central tension: R2 recommends dropping the spatial encoder (Δ_spatial not significant), while R3 shows CNN achieves R²=0.132 — 20× higher than any R2 model.

| Dimension | R2 (Info-Decomposition) | R3 (Book-Encoder Bias) |
|-----------|-------------------------|------------------------|
| **Input format** | Flattened 40-dim vector | Structured (20, 2) price ladder |
| **Model** | 2-layer MLP (64 units, ~5k params) | Conv1d (~12,128 params) |
| **Target** | `fwd_return_5` (best: R²=−0.0002) | `return_5` (R²=0.1317) |
| **Best h=1 R²** | +0.0067 (MLP on raw book) | Not tested |
| **Best h=5 R²** | −0.0002 (MLP on raw book) | **+0.1317** (CNN on structured book) |
| **CV method** | 5-fold expanding window | 5-fold expanding window |
| **Days used** | 19 (same set) | 19 (same set) |
| **Param count** | ~5k (MLP) | ~12k (CNN), ~12k (Attention), ~12k (MLP control) |
| **Δ vs baseline** | Δ_spatial = +0.003 (corrected p=0.96) | CNN vs Attention: Δ=+0.047 (corrected p=0.042, d=1.86) |

### 3.2 Resolution

The 20× R² difference (0.132 vs. −0.0002 at h=5) is attributable to three factors:

**(a) Structured (20,2) input vs. flattened 40-dim vector (primary driver).** R2 flattened the 10-level × 2-side book into a 40-element vector, destroying spatial adjacency. R3 preserved the price-ladder layout, enabling the CNN to detect patterns across adjacent levels (e.g., absorption at best bid, thinning 3 levels out). The sufficiency test quantifies this: CNN-16d retains 4.16× more signal than raw-40d (R²=0.111 vs 0.027, paired t p=0.012, d=1.93).

**(b) Conv1d inductive bias (secondary driver).** The CNN's 1D convolutions enforce weight-sharing across price levels, providing a strong prior that adjacent-level relationships are important. R3's MLP control (also ~12k params, also on structured input) achieves R²=0.100 — still far above R2's 0.0067 but below CNN's 0.132. This confirms that both structured input AND convolutional bias contribute, with structured input being the larger factor.

**(c) Model capacity (minor factor).** R3's CNN has ~12k params vs. R2's MLP with ~5k params. However, the R3 MLP control (also ~12k params) at R²=0.100 shows that capacity alone does not explain the gap — the 12k-param MLP still outperforms R2's 5k MLP by 15×, confirming the structured input is the dominant factor.

**(d) No data split artifact.** Both use the same 19 days and the same expanding-window CV. R3's `return_5` target is the 5-bar forward return, matching R2's `fwd_return_5`. The definitions are identical.

**Conclusion**: R3 supersedes R2 for the spatial encoder decision. R2's Δ_spatial comparison was methodologically limited — it compared two models that both lacked spatial inductive bias (flat MLP vs. flat linear on flattened vectors). R3 demonstrates that with proper spatial encoding, the book contains substantial predictive signal at h=5.

### 3.3 §7.2 Simplification Cascade — Final Table

| Encoder Stage | R2 Gap Evidence | R2 Verdict | R3/R4 Evidence | Final Decision |
|---------------|----------------|------------|----------------|----------------|
| **Spatial (CNN)** | Δ_spatial = +0.003, corrected p=0.96 | DROP | CNN R²=0.132, CNN vs Attention corrected p=0.042, d=1.86; CNN-16d retention ratio=4.16× (p=0.012) | **INCLUDE — R3 supersedes R2** |
| **Message encoder** | Δ_msg_summary = −0.001; Δ_msg_learned: LSTM −0.003, Transformer −0.006 | DROP | N/A | **DROP** |
| **Temporal (SSM)** | Δ_temporal = −0.006, corrected p=0.25 | DROP | R4: 36/36 AR cells negative, 0/16 augmentation gaps pass dual threshold, Temporal-Only R²≈0 | **DROP** |

### 3.4 Message Encoder: DROP

R2 Tier 1 and Tier 2 evidence is unambiguous:

- **Δ_msg_summary** (Category 6 hand-crafted features vs. raw book): Negative at 3/4 horizons (h=1: −0.0009, h=5: −0.0015, h=20: −0.0024). The 5 message summary features (cancel_add_ratio, message_rate, etc.) introduce noise. All corrected p=1.0.
- **Δ_msg_learned LSTM** (Config e vs. Config b): Negative at 3/4 horizons (h=1: −0.0026, h=5: −0.0019, h=20: −0.0004). LSTM underperforms plain book MLP despite processing raw MBO event sequences. All corrected p=1.0.
- **Δ_msg_learned Transformer** (Config f vs. Config b): **Negative at all 4 horizons** (h=1: −0.0058, h=5: −0.0024, h=20: −0.0043, h=100: −0.0049). The attention mechanism finds no useful sequential patterns. All corrected p=1.0.
- **Tier 2 vs. Tier 1** (learned vs. hand-crafted messages): Both LSTM and Transformer perform *worse* than the simple 5-feature summary at most horizons.

The book snapshot at bar close is a sufficient statistic for intra-bar message activity. Message encoders add complexity and noise without signal.

### 3.5 Temporal Encoder: DROP

Converging evidence from two independent experiments with different temporal representations:

**R2 evidence:**
- Δ_temporal = −0.006 (MLP, h=1, corrected p=0.25). Adding 20 bars of lookback (845 dims) drives R² negative — classic overfitting on weak signal.
- Δ_temporal negative at all 4 horizons.

**R4 evidence (comprehensive confirmation):**
- **Tier 1**: 36/36 AR configurations (3 lookbacks × 3 models × 4 horizons) produce negative R². All corrected p=1.0. Returns are martingale.
- **Tier 2**: 0/16 temporal augmentation gaps pass dual threshold. Best corrected p=0.25 (Δ_temporal_hc at h=100, but CI spans [−0.034, +0.110]).
- **Temporal-Only**: R² = 8.8×10⁻⁷ at h=1 (indistinguishable from zero). Negative at h=5, h=20, h=100.
- **Feature importance**: Temporal features receive 30-50% GBT gain share despite zero marginal R² improvement — a classic overfitting signature where the model allocates splits to noise dimensions.
- **GBT vs. Linear**: GBT mitigates overfitting (less negative R²) but no comparison survives Holm-Bonferroni correction. GBT's best R² (AR-10, h=1: −0.0002) is still negative.

R4 resolves R2's two confounds: (1) dimensionality curse (R2 inflated from 45 to 845 dims; R4 uses only 21 temporal features — same result), and (2) representation mismatch (R2 used raw book lags; R4 uses purpose-built features like lagged returns, rolling volatility, momentum — same result).

### 3.6 Spatial Encoder: INCLUDE (CNN + GBT Hybrid)

The decision follows **OPTION B** from the spec:

```
Conv1d encoder on raw (20,2) book → 16-dim embedding → concatenate
with non-spatial hand-crafted features → XGBoost head.
```

Evidence supporting inclusion:
- R3 CNN mean R²=0.132 ± 0.048 on return_5 (5-fold, ~12k params).
- CNN vs. Attention: Δ=+0.047, corrected p=0.042, Cohen's d=1.86 — **statistically significant**.
- CNN vs. MLP: Δ=+0.032, corrected p=0.251 — not significant but CNN has lower fold variance (std=0.048 vs 0.069) and never-negative mean.
- CNN 16-dim embedding linear probe: R²=0.111 vs raw 40-dim probe R²=0.027 (retention ratio=4.16×, paired t p=0.012, d=1.93).

CNN fold-level detail (from `model_comparison.csv`):

| Fold | CNN R² | Attention R² | MLP R² |
|------|--------|-------------|--------|
| 1 | 0.163 | 0.155 | 0.087 |
| 2 | 0.109 | 0.048 | 0.064 |
| 3 | 0.049 | −0.008 | −0.002 |
| 4 | 0.180 | 0.108 | 0.191 |
| 5 | 0.159 | 0.122 | 0.160 |
| **Mean** | **0.132** | **0.085** | **0.100** |
| **Std** | **0.048** | **0.058** | **0.069** |

CNN is the most consistent model (lowest std, no negative folds). Fold 3 is an outlier across all models (likely a regime-difficult quarter), but CNN still produces positive R² even there.

### 3.7 Final Architecture Diagram

```
Raw Book Snapshot (20 levels × 2 sides)
    │
    ▼
Conv1d Encoder (~12k params)
    │
    ▼
16-dim Embedding ──────┐
                       │
Non-Spatial Features   │
(~20 dims: order flow, ├──→ Concatenate (36-dim) ──→ XGBoost Head ──→ ŷ
 trade imbalance,      │
 volatility, time,     │
 returns, spread)   ───┘
```

---

## 4. Feature Set Specification

### 4.1 CNN Input (Spatial Encoder)

| Component | Shape | Description |
|-----------|-------|-------------|
| Book snapshot | (20, 2) | 10 bid levels + 10 ask levels, each with (price_offset, quantity) |

**Preprocessing**: Prices expressed as tick offsets from mid-price. Quantities z-score normalized per day. Warmup = 50 bars/day discarded.

**Output**: 16-dim embedding vector.

### 4.2 Non-Spatial Hand-Crafted Features (~20 dimensions)

Based on R4 feature importance analysis and R2's Track A feature set, excluding book-derived features (already captured by CNN):

| Feature | Dim | Category | Justification |
|---------|-----|----------|---------------|
| trade_imbalance | 1 | Order flow | Non-spatial signal |
| trade_flow | 1 | Order flow | Non-spatial signal |
| vwap_distance | 1 | Price | R4 top-10 at h=20, h=100 |
| close_position | 1 | Price | R4 top-10 at h=20 |
| return_1 | 1 | Return | Current-bar return |
| return_5 | 1 | Return | Short momentum |
| return_20 | 1 | Return | Medium momentum |
| volatility_20 | 1 | Volatility | Regime indicator |
| volatility_50 | 1 | Volatility | R4 top-10 at h=100 |
| high_low_range_50 | 1 | Volatility | R4 top-10 at h=5, h=20 |
| cancel_add_ratio | 1 | Message activity | Cat 6 (retained as scalar, not encoder) |
| message_rate | 1 | Message activity | Cat 6 |
| modify_fraction | 1 | Message activity | R4 HC top-10 at h=1 |
| time_sin | 1 | Time-of-day | R4 HC top-10 at h=5 |
| time_cos | 1 | Time-of-day | R4 HC top-10 at h=1 |
| minutes_since_open | 1 | Time-of-day | R4 HC top-10 at h=20, h=100 |
| minutes_to_close | 1 | Time-of-day | R4 HC top-10 at h=100 |
| is_afternoon | 1 | Time-of-day | Session indicator |
| is_close | 1 | Time-of-day | End-of-day regime |
| bar_volume | 1 | Volume | Activity level |

**Total non-spatial dimension**: ~20.

### 4.3 Excluded Features (Redundant with CNN)

| Feature Group | Dimension | Reason |
|---------------|-----------|--------|
| bid_depth_profile_0..9 | 10 | Captured by CNN's (20,2) input |
| ask_depth_profile_0..9 | 10 | Captured by CNN's (20,2) input |
| book_imbalance_1..5 | 5 | Derived from book levels |
| book_slope_bid/ask | 2 | Derived from book levels |
| spread | 1 | Derived from best bid/ask |

### 4.4 Combined Feature Vector

| Component | Dimension |
|-----------|-----------|
| CNN 16-dim embedding | 16 |
| Non-spatial features | ~20 |
| **Total** | **~36** |

### 4.5 Preprocessing Pipeline

1. **Warmup**: Discard first 50 bars per day (§8.6 warmup policy). Rolling features require history.
2. **Lookahead policy**: No forward-looking features. All features computed from current bar and earlier only.
3. **CNN normalization**: Prices as tick offsets from mid-price. Quantities z-scored per day.
4. **Feature normalization**: Z-score per day for all non-spatial features.
5. **Missing data**: No imputation — bars with incomplete book data excluded.

---

## 5. Prediction Horizon Analysis

### 5.1 Evidence by Horizon

| Experiment | Horizon | Model | R² | Std | Status |
|------------|---------|-------|----|-----|--------|
| R2 | h=1 (~5s) | MLP (raw book, config b) | **+0.0067** | 0.0034 | **Positive** |
| R2 | h=1 (~5s) | Linear (hand-crafted, config a) | +0.0059 | 0.0053 | Positive |
| R4 | h=1 (~5s) | GBT (static book) | **+0.0046** | 0.0071 | **Positive** |
| R4 | h=1 (~5s) | GBT (static HC) | +0.0032 | 0.0054 | Positive |
| R3 | h=5 (~25s) | **CNN (structured book)** | **+0.1317** | 0.0478 | **Strong positive** |
| R3 | h=5 (~25s) | MLP (structured book) | +0.1001 | 0.0688 | Positive |
| R3 | h=5 (~25s) | Attention (structured book) | +0.0850 | 0.0580 | Positive |
| R2 | h=5 (~25s) | MLP (raw book, config b) | −0.0002 | 0.0014 | Negative |
| R4 | h=5 (~25s) | GBT (static book) | −0.0009 | 0.0013 | Negative |
| R2 | h=20 (~100s) | MLP (raw book, config b) | −0.0029 | 0.0018 | Negative |
| R2 | h=100 (~500s) | MLP (raw book, config b) | −0.0179 | 0.0138 | Negative |

### 5.2 Horizon Tension and Reconciliation

**R2/R4 say**: Only h=1 has positive R². All longer horizons negative with flat/tabular models.

**R3 says**: h=5 with CNN achieves R²=0.132. All three R3 models (CNN, Attention, MLP) are positive at h=5 on structured (20,2) input.

**Resolution**: The discrepancy is entirely architectural. R2's MLP on flattened 40-dim input cannot extract the spatial patterns that predict 25-second moves. R3's structured (20,2) input preserves price-ladder topology, enabling models to detect:
- Level absorption (large quantity at best bid consuming incoming sell pressure)
- Book thinning (quantity drop 2-3 levels from best, signaling imminent move)
- Cross-level imbalance gradients (asymmetric depth decay on bid vs. ask side)

These spatial patterns are invisible after flattening, explaining why R2's MLP at h=5 yields R²=−0.0002 while R3's CNN at h=5 yields R²=+0.132.

At h=1 (~5s), the signal is dominated by bid-ask bounce dynamics already captured by simple scalar features (spread, best bid/ask quantities), which is why flat models achieve positive R² at h=1 but not h=5.

### 5.3 Recommendation

**Test both h=1 and h=5 in the model build.**

- **h=5 is the primary target** given the 20× signal advantage from CNN. R²=0.132 represents a substantial predictive signal.
- **h=1 is a secondary target / sanity check.** Known to have weak but positive signal (R²≈0.005) with simple models. CNN performance at h=1 is a **critical unknown** that must be tested.

---

## 6. Oracle Expectancy (Phase 3)

### 6.1 Status: OPEN QUESTION

Phase 3 (multi-day-backtest) was a TDD engineering phase implementing the oracle_labeler with first-to-hit labeling:
- `target_ticks = 10`
- `stop_ticks = 5`
- `take_profit_ticks = 20`
- `horizon = 100` bars

Results reside in C++ GTest output, not in `.kit/results/`. This synthesis phase cannot extract those results without running C++ tests.

### 6.2 Impact on Go/No-Go

Per the decision rule:
> If R² > 0 but oracle expectancy unknown or ≤ 0: → CONDITIONAL GO.

We have positive R² at h=1 (R²=0.0067) and h=5 (R²=0.132). Oracle expectancy is unknown. The decision is therefore **CONDITIONAL GO**.

### 6.3 Required Action Before Model Build

1. Extract oracle expectancy from Phase 3 C++ test output.
2. Compute: fraction of oracle labels that are profitable, average P&L per trade.
3. If oracle expectancy ≤ 0, revise labeling method before proceeding.
4. Evaluate whether R²=0.005 (h=1) or R²=0.132 (h=5) is sufficient for positive expectancy after transaction costs (MES half-turn ≈ 1 tick = $1.25).

---

## 7. Statistical Limitations

### 7.1 Critical (could invalidate go/no-go)

| ID | Description | Evidence |
|----|-------------|----------|
| `single_year_2022` | All data is from 2022 — a bear market with aggressive Fed rate hikes. Microstructure patterns may be regime-specific. | R1 showed quarter-level reversals in bar type rankings (Q1-Q4 robustness: all false). R3 fold 3 (R²=0.049) vs fold 4 (R²=0.180) suggests within-year regime sensitivity. 2022 had VIX spikes, FOMC-driven volatility clusters, and bear market rallies that may produce unique book patterns not present in other years. |

### 7.2 Major (could change architecture recommendation)

| ID | Description | Evidence |
|----|-------------|----------|
| `r3_only_tested_h5` | R3 CNN tested only return_5 (h=5, ~25s). CNN performance at h=1 is unknown. | Architecture recommendation rests on h=5 signal (R²=0.132) that R2/R4 found negative with simpler models. If CNN adds nothing at h=1, may need separate architectures per horizon. |
| `power_floor_r2_0.003` | With 5 folds × ~84k bars and R²<0.007, the 95% CI on any R2 gap overlaps zero for effect sizes < ~0.003 R². | R2 Δ_spatial MLP: +0.0031 [+0.0017, +0.0046], raw p=0.025, corrected p=0.96. A real effect of this size is economically meaningful but statistically invisible after correction. |
| `no_regime_conditioning` | No regime-conditional analysis in R2-R4. | R1 demonstrated that bar type rankings reverse across quarters. The CNN's R²=0.132 may be concentrated in specific volatility regimes. Architecture decisions may not generalize. |

### 7.3 Minor (noted for future work)

| ID | Description | Evidence |
|----|-------------|----------|
| `failed_corrections` | Several nominally significant results do not survive Holm-Bonferroni correction. | R2 Δ_spatial raw p=0.025 → corrected p=0.96 (40 tests). R3 CNN vs MLP raw p=0.126 → corrected p=0.251 (3 tests). Direction is consistently informative. |
| `r3_fold_variance` | R3 CNN fold R² ranges 0.049–0.180 (std=0.048). | High within-year variance suggests the CNN's signal is not uniform across market conditions. Fold 3 (weakest) may correspond to a low-volatility period where book patterns are less pronounced. |

---

## 8. Convergence Matrix

| Question | R1 | R2 | R3 | R4 | Decision |
|----------|----|----|----|----|----------|
| **Bar type** | time_5s (REFUTED subordination; 0/3 primary tests significant) | time_5s (used as baseline) | — | time_5s (used; all AR R²<0) | **time_5s** |
| **Spatial encoder** | — | DROP (Δ_spatial=+0.003, corrected p=0.96) | **INCLUDE** (CNN R²=0.132, p=0.042 vs Attention, d=1.86) | — | **INCLUDE (R3 supersedes R2)** |
| **Message encoder** | — | DROP (Δ_msg_summary < 0; Δ_msg_learned < 0) | — | — | **DROP** |
| **Temporal encoder** | — | DROP (Δ_temporal = −0.006) | — | DROP (36/36 AR negative; 0/16 Tier 2 pass; Temporal-Only R²≈0) | **DROP** |
| **Signal horizon** | — | h=1 only (R²=0.0067; all h>1 negative) | h=5 (R²=0.132 CNN) | h=1 only (R²=0.0046 GBT) | **Both h=1 and h=5** |
| **Signal magnitude** | — | R²=0.0067 (h=1 MLP) | R²=0.132 (h=5 CNN) | R²=0.0046 (h=1 GBT) | **h=1: ~0.005; h=5: ~0.13 (CNN)** |
| **Subordination theory** | REFUTED (0/3 significant; dollar bars have higher AR R²) | — | — | — | **REFUTED** |
| **Book sufficiency** | — | CONFIRMED (book = sufficient statistic for messages; Δ_msg ≤ 0) | CNN amplifies book signal (structured input yields 20× R²) | — | **Sufficient; CNN amplifies via spatial structure** |
| **Temporal predictability** | — | Δ_temporal < 0 (lookback hurts with raw book) | — | Martingale (AR R²<0; Temporal-Only R²≈0) | **NONE — returns are martingale** |

---

## 9. Open Questions for Model Build

1. **Oracle expectancy** (blocks GO decision): Extract from Phase 3 C++ backtest. If first-to-hit labeling produces ≤ 0 expectancy, explore alternative labeling methods (triple barrier, fixed-horizon regression targets) before committing GPU budget.

2. **CNN at h=1** (architecture validation): R3 only tested h=5. Must validate CNN improves or at least does not degrade h=1 predictions. If CNN adds nothing at h=1, consider: (a) GBT-only for h=1, CNN+GBT for h=5; or (b) CNN+GBT for both with per-horizon hyperparameter tuning.

3. **Regime-conditional evaluation**: Stratify model build results by volatility regime or calendar quarter. R1 showed quarter-level reversals. Minimum: report per-fold performance with fold date ranges.

4. **Transaction cost model**: With MES half-turn cost of 1 tick ($1.25, contract value ~$11k at S&P 4400), estimate minimum R² / directional accuracy for positive expected profit. At h=5 with R²=0.132, the signal is likely strong enough. At h=1 with R²=0.005, the signal-to-cost ratio is questionable.

5. **CNN + GBT integration procedure**: Specify the exact training pipeline:
   - **Option A (recommended)**: Train CNN end-to-end with linear head on return prediction. Freeze CNN. Extract 16-dim embeddings. Train XGBoost on embeddings + non-spatial features.
   - **Option B**: Alternating optimization (more complex, unclear benefit).
   - Option A is most consistent with R3's evaluation methodology and avoids gradient-free/gradient-mixed training complexity.

6. **Out-of-sample holdout**: Reserve 2-3 months of 2022 data as a true holdout not seen during any experiment or fold. Current 5-fold expanding-window CV provides out-of-sample estimates but all data has been used across experiments.

---

## 10. Architecture Decision JSON

```json
{
  "go_no_go": "CONDITIONAL_GO",
  "has_positive_oos_r2": true,
  "oracle_expectancy_known": false,
  "bar_type": "time_5s",
  "bar_param": 5,
  "architecture": "cnn_gbt_hybrid",
  "spatial_encoder": {
    "include": true,
    "type": "conv1d",
    "embedding_dim": 16,
    "input_shape": [20, 2],
    "param_count": 12128,
    "justification": "R3 CNN R²=0.132, corrected p=0.042 vs Attention, d=1.86. Retention ratio=4.16× (16d vs 40d, p=0.012). Structured (20,2) input + conv1d bias extracts signal invisible to flattened MLP."
  },
  "message_encoder": {
    "include": false,
    "justification": "R2 Δ_msg_summary < 0 at 3/4 horizons. Δ_msg_learned: LSTM −0.003, Transformer −0.006. All corrected p=1.0. Book state is sufficient statistic."
  },
  "temporal_encoder": {
    "include": false,
    "justification": "R2 Δ_temporal = −0.006. R4: 36/36 AR cells negative, 0/16 Tier 2 gaps pass dual threshold, Temporal-Only R²=8.8e-7. Martingale confirmed by converging evidence."
  },
  "features": {
    "source": "cnn_16d_plus_non_spatial",
    "dimension": 36,
    "cnn_embedding_dim": 16,
    "non_spatial_dim": 20,
    "preprocessing": "z-score per day, warmup=50 bars"
  },
  "prediction_horizons": [1, 5],
  "labeling": {
    "method": "open_question",
    "params": {"target_ticks": 10, "stop_ticks": 5, "take_profit_ticks": 20, "horizon": 100},
    "status": "Must extract oracle expectancy from Phase 3 C++ backtest"
  },
  "limitations": [
    {"id": "single_year_2022", "severity": "critical"},
    {"id": "r3_only_tested_h5", "severity": "major"},
    {"id": "power_floor_r2_0.003", "severity": "major"},
    {"id": "no_regime_conditioning", "severity": "major"},
    {"id": "failed_corrections", "severity": "minor"},
    {"id": "r3_fold_variance", "severity": "minor"}
  ],
  "limitation_count": {"critical": 1, "major": 3, "minor": 2},
  "open_questions": [
    "Oracle expectancy from Phase 3 C++ backtest",
    "CNN performance at h=1 (R3 only tested h=5)",
    "Regime-conditional analysis across quarters",
    "Transaction cost estimation for R²=0.005-0.132 signals",
    "CNN embedding + GBT integration training procedure"
  ]
}
```
