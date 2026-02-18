# R4b: Temporal Predictability on Event-Driven Bars — Cross-Bar Comparison

**Date:** 2026-02-18
**Spec:** `.kit/experiments/temporal-predictability-event-bars.md`
**Depends on:** R4 (time_5s), R1 (subordination test)

---

## Summary

**VERDICT: NO TEMPORAL ENCODER NEEDED.** Dollar bars contain weak, statistically marginal autoregressive structure at sub-second horizons (h=1: R²=+0.0006), but this signal is entirely redundant with static features — temporal augmentation fails the dual threshold for all bar types. The R4 "NO TEMPORAL SIGNAL" conclusion generalizes across bar types with a minor caveat: dollar bars have detectable (but non-exploitable) temporal structure. Proceed with CNN+GBT, no temporal encoder.

---

## 1. Cross-Bar-Type Tier 1 Comparison

### Best Tier 1 R² per bar type × horizon

| Bar Type | R1 AR R² | h=1 | h=5 | h=20 | h=100 |
|----------|----------|-----|-----|------|-------|
| time_5s | 0.00034 | −0.0002 | −0.0005 | −0.0025 | −0.0092 |
| volume_100 | 0.00047 | −0.000135 | −0.000540 | −0.001775 | −0.010558 |
| **dollar_25k** | **0.01131** | **+0.000633** | **+0.000364** | +0.000080 | −0.000286 |

Best model shown per cell (GBT for time_5s/volume_100, linear for dollar_25k).

**Key findings:**
- Dollar_25k is the **only bar type with positive Tier 1 R²**, confirming R1's observation that dollar bars have 33× higher AR R² than time bars.
- The positive R² is concentrated at h=1 and h=5 (sub-second horizons). By h=20 (~2.8s) the signal is negligible; by h=100 (~14s) it's negative.
- Volume_100 results are essentially identical to time_5s — both show fully negative R² across all configs. This matches R1's prior (volume_100 AR R² = 0.00047 ≈ time_5s 0.00034).

### Tier 1 R² ordering matches R1 AR R² ordering

| R1 AR R² ranking | R4b Tier 1 best h=1 R² |
|-------------------|------------------------|
| dollar_25k: 0.01131 | +0.000633 |
| volume_100: 0.00047 | −0.000135 |
| time_5s: 0.00034 | −0.0002 |

The Tier 1 results are consistent with R1's subordination test data, but with a critical difference: R1's AR R² was fit on a single train set without CV. The R4b out-of-sample R² is ~18× smaller (0.000633 vs 0.01131), revealing that most of R1's measured AR was in-sample overfit.

---

## 2. Cross-Bar-Type Tier 2 Comparison

### Best Tier 2 R² (Static-HC GBT) per bar type × horizon

| Bar Type | h=1 | h=5 | h=20 | h=100 |
|----------|-----|-----|------|-------|
| time_5s | +0.0032 | −0.0189 | −0.0039 | −0.0462 |
| volume_100 | +0.0077 | −0.0001 | −0.0046 | −0.0162 |
| dollar_25k | **+0.0803** | **+0.0435** | **+0.0123** | +0.0013 |

Dollar_25k dominates all horizons on static features alone. This is the most significant finding: **dollar bars have dramatically higher static feature predictability** — 10-25× higher R² than time_5s at h=1.

### Temporal Augmentation Gaps (Δ_temporal_book)

| Bar Type | h=1 | h=5 | h=20 | h=100 |
|----------|-----|-----|------|-------|
| time_5s | −0.0021 (p=0.733) | +0.0004 (p=0.733) | +0.0003 (p=1.000) | +0.0004 (p=1.000) |
| volume_100 | +0.0000 (p=1.000) | −0.0006 (p=1.000) | −0.0044 (p=1.000) | +0.0006 (p=1.000) |
| dollar_25k | +0.0132 (p=0.250) | −0.0029 (p=0.647) | +0.0012 (p=0.647) | −0.0016 (p=0.647) |

**No bar type passes the dual threshold on temporal augmentation.** Dollar_25k at h=1 shows the largest Δ (+0.013) but with corrected p=0.250, well above the 0.05 threshold. The wide 95% CI [−0.008, +0.035] confirms this is noise.

---

## 3. Decision Rule Evaluation

### Per bar type

| Rule | time_5s | volume_100 | dollar_25k |
|------|---------|------------|------------|
| **Rule 1**: AR GBT R² > 0, corrected p < 0.05 | FAIL | FAIL | FAIL |
| **Rule 2**: Δ_temporal_book passes dual threshold | FAIL | FAIL | FAIL |
| **Rule 3**: Temporal-Only R² > 0, corrected p < 0.05 | FAIL | FAIL | **PASS** (h=1, h=5) |
| **Rule 4**: Reconciliation with R4 | N/A | N/A | See below |

### Rule 3 detail — Dollar_25k Temporal-Only

Dollar_25k is the only bar type where Temporal-Only features have standalone predictive power:

| Horizon | Temporal-Only R² | Corrected p |
|---------|------------------|-------------|
| h=1 | +0.0118 | 0.0005 |
| h=5 | +0.0043 | 0.0029 |
| h=20 | +0.0004 | 0.3249 |
| h=100 | −0.0009 | 1.0000 |

Temporal-only at h=1 (R²=0.012, p=0.0005) is statistically significant and replicable. However, this does **not** translate to improved prediction when combined with static features (Rule 2 fails). The temporal signal is a subset of what static features already capture.

### Rule 4 — Reconciliation

R4 (time_5s) concluded "NO TEMPORAL SIGNAL — drop SSM." R4b shows:
- **Volume_100**: Fully confirms R4. All rules fail. Returns are martingale.
- **Dollar_25k**: Partially nuances R4. Temporal-only signal exists (Rule 3 passes), but temporal augmentation adds nothing (Rule 2 fails). The practical conclusion is the same: **no temporal encoder needed**.

---

## 4. Horizon Mismatch Analysis

Bar horizons are in bars, not clock time. Average bar durations differ drastically:

| Bar Type | Avg Duration | h=1 | h=5 | h=20 | h=100 |
|----------|-------------|-----|-----|------|-------|
| time_5s | 5s | 5s | 25s | 100s | 500s |
| volume_100 | ~3.9s | ~4s | ~20s | ~78s | ~390s |
| dollar_25k | ~0.14s | ~0.14s | ~0.7s | ~2.8s | ~14s |

Dollar_25k's positive R² at h=1 and h=5 corresponds to 0.14s and 0.7s in clock time. This is genuine microstructure territory — quote revisions, inventory management, HFT patterns. Dollar_25k h=100 (~14s) is roughly equivalent to time_5s h=3.

**Apples-to-apples clock-time comparison:** Dollar_25k h≈35 (~5s) ≈ time_5s h=1. At this dollar-bar horizon, the R4b Tier 1 R² is near zero (interpolating between h=20: +0.00008 and h=100: −0.00029). So at equivalent clock time, dollar bars show no advantage over time bars for temporal structure.

---

## 5. Linear vs. Nonlinear Assessment

| Bar Type | Best Linear h=1 R² | Best GBT h=1 R² | Linear ≥ GBT? |
|----------|---------------------|-------------------|---------------|
| time_5s | −0.0010 | −0.0002 | No (but both negative) |
| volume_100 | −0.0009 | −0.0003 | No (but both negative) |
| dollar_25k | **+0.000633** | +0.000471 | **Yes** |

Dollar_25k's temporal signal is **linear** — Ridge and Linear consistently match or outperform GBT at all horizons. This means:
- No nonlinear temporal encoder (SSM, LSTM, Transformer) is warranted
- If temporal signal were to be exploited, simple linear features would suffice
- The GBT's regularization (depth=4, subsampling) provides no advantage over linear models for this signal

---

## 6. Feature Importance Comparison

### Temporal feature share in Book+Temporal GBT

| Bar Type | h=1 | h=5 | h=20 | h=100 |
|----------|-----|-----|------|-------|
| time_5s | 47.5% | 50.8% | 42.6% | 34.9% |
| volume_100 | 49.9% | 53.8% | 44.6% | 32.6% |
| dollar_25k | 22.5% | 21.6% | 27.0% | 24.8% |

Paradoxically, **dollar_25k has the lowest temporal feature share** despite being the only bar type with actual temporal signal. In time_5s and volume_100, GBT allocates ~50% of importance to temporal features that provide zero marginal R² — classic overfitting to noise dimensions. In dollar_25k, the static features (book_snap) are so dominant (R²=0.067-0.080 at h=1) that temporal features get proportionally less attention.

Top static feature across all bar types: `book_snap_19` and `book_snap_21` (best bid/ask at level 10) — consistent with R3's finding that the CNN book encoder works.

---

## 7. Executability Assessment

Dollar_25k's temporal signal operates at ~140ms bar intervals. This is:
- **Below retail execution latency**: MES market orders via CME Globex have round-trip times of 5-50ms from co-located infrastructure, 50-200ms from retail. A 140ms bar means the signal may have decayed by the time an order reaches the exchange.
- **Below retail data latency**: Retail MBO feeds have 10-100ms delays. The signal may not be observable in real time.
- **Within HFT territory**: This timescale is competitive with dedicated market-making infrastructure. Retail cannot compete.

Even if the temporal signal were exploitable (it isn't — Rule 2 fails), the timescale makes it irrelevant for the target use case (MES retail microstructure model at 5s bars).

---

## 8. Cross-Bar-Type Decision

Per the experiment spec § Decision Framework:

**Neither the "both fail" nor the "dollar passes Rule 2" path applies exactly.** The actual outcome is:

- Dollar_25k passes Rule 3 (temporal-only has standalone power) but fails Rule 2 (temporal doesn't improve static features)
- Volume_100 and time_5s fail all rules

This is the **MARGINAL SIGNAL** outcome: temporal structure exists in dollar-bar event time but is redundant with the static feature set. The spec's Path B (cross-scale temporal feature aggregation) is not justified because temporal features don't improve prediction even *within* dollar bars — aggregating them to time_5s bars would add noise, not signal.

**Recommendation**: Proceed with CNN+GBT on time_5s bars, no temporal encoder, with high confidence. The gap in the experimental chain (R4 only tested time_5s) is now closed. All three bar types converge on the same practical conclusion.

---

## 9. Summary Table

| Metric | time_5s (R4) | volume_100 (R4b) | dollar_25k (R4b) |
|--------|-------------|-----------------|-----------------|
| Bars (19 days) | 87,970 | 115,661 | 3,124,720 |
| R1 AR R² | 0.00034 | 0.00047 | 0.01131 |
| Best Tier 1 R² (h=1) | −0.0002 | −0.000135 | **+0.000633** |
| Best Tier 2 R² (h=1) | +0.0046 | +0.0077 | **+0.0803** |
| Δ_temporal_book (h=1) | −0.0021 | +0.0000 | +0.0132 |
| Dual threshold passes | 0/16 | 0/16 | 0/16 |
| Rule 1 (AR structure) | FAIL | FAIL | FAIL |
| Rule 2 (temporal augmentation) | FAIL | FAIL | FAIL |
| Rule 3 (temporal-only) | FAIL | FAIL | PASS (h=1,5) |
| Signal character | — | — | Linear |
| Avg bar duration | 5s | ~3.9s | ~0.14s |
| Verdict | NO SIGNAL | NO SIGNAL | MARGINAL (redundant) |

---

## 10. Downstream Impact

- **Architecture**: CNN+GBT Hybrid confirmed. No temporal encoder. No SSM.
- **Bar type**: time_5s confirmed as primary (oracle expectancy already validated).
- **Dollar bars**: Not pursued further. Signal is sub-second microstructure, not tradeable at retail timescales, and redundant with static features.
- **Confidence**: HIGH — three bar types × three temporal tests × three model families all converge.
