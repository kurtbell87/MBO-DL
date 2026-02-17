# R2: Information Decomposition — Analysis

**Experiment**: R2_information_decomposition
**Date**: 2026-02-17
**Spec**: `.kit/experiments/info-decomposition.md`
**Finding**: **FEATURES SUFFICIENT** — No encoder stage (spatial CNN, message encoder, temporal SSM) passes the dual threshold. Recommendation: GBT baseline.

---

## 1. Data Summary

| Parameter | Value |
|-----------|-------|
| Bar type | `time_5s` (5-second time bars) |
| Days | 19 (same as R1) |
| Total bars | 87,970 |
| Warmup | 50 bars/day discarded |
| Device | CPU |

---

## 2. Truncation Statistics (Tier 2)

| Metric | Value |
|--------|-------|
| Bars with > 500 events | 75,118 |
| Truncation rate (%) | 85.39 |
| Median events per bar | 1,319 |
| P95 events per bar | 3,701 |
| Max events per bar | 23,238 |

The 500-event cap truncated 85% of bars. The median bar contains 1,319 MBO events — 2.6x the cap. This severe truncation means Tier 2 models (LSTM, Transformer) operated on heavily clipped sequences, discarding the majority of intra-bar message information. However, this does not rescue the message encoder hypothesis: if the most recent 500 events contained exploitable sequential patterns, the learned models should still have outperformed the book snapshot. They did not.

---

## 3. R² Matrix — Table 1 (MLP/Tier2, Primary)

All values: mean ± std across 5 expanding-window folds.

| Horizon | (a) R²\_bar | (b) R²\_book | (c) R²\_book+msg\_sum | (d) R²\_full\_summary | (e) R²\_LSTM | (f) R²\_Transformer |
|---------|------------|-------------|----------------------|----------------------|-------------|---------------------|
| return\_1 | 0.0036 ± 0.0033 | **0.0067 ± 0.0034** | 0.0058 ± 0.0046 | −0.0004 ± 0.0043 | 0.0041 ± 0.0038 | 0.0009 ± 0.0080 |
| return\_5 | −0.0057 ± 0.0049 | −0.0002 ± 0.0014 | −0.0016 ± 0.0011 | −0.0026 ± 0.0018 | −0.0020 ± 0.0010 | −0.0026 ± 0.0019 |
| return\_20 | −0.0144 ± 0.0171 | −0.0029 ± 0.0018 | −0.0053 ± 0.0028 | −0.0079 ± 0.0032 | −0.0034 ± 0.0028 | −0.0072 ± 0.0029 |
| return\_100 | −0.0212 ± 0.0181 | −0.0179 ± 0.0138 | −0.0138 ± 0.0086 | −0.0198 ± 0.0061 | −0.0139 ± 0.0124 | −0.0228 ± 0.0123 |

### Key observations

1. **Only the 1-bar horizon shows positive R².** Config (b) raw book MLP achieves R²=0.0067 — the experiment's best result. Even this is extremely weak: the model explains 0.67% of return variance at the ~5-second horizon.

2. **All longer horizons are negative.** R² < 0 means the models perform worse than predicting the mean. The 5-bar (~25s), 20-bar (~100s), and 100-bar (~500s) horizons are unpredictable from any feature set tested.

3. **Hand-crafted features (a) underperform raw book (b) on return\_1.** The 62-dim hand-crafted set (R²=0.0036) is worse than the 40-dim raw book snapshot (R²=0.0067). The feature engineering pipeline adds dimensionality without adding signal.

4. **Adding message summaries (c) hurts.** Config (c) R²=0.0058 < Config (b) R²=0.0067. The 5 Category 6 features (cancel\_add\_ratio, message\_rate, etc.) introduce noise.

5. **Temporal lookback (d) hurts severely.** Config (d) R²=−0.0004 — adding 20 bars of lookback drives the model to negative R². This is classic overfitting: 845 input dimensions with weak signal and ~85k samples.

6. **Tier 2 learned encoders fail.** LSTM (e) R²=0.0041 and Transformer (f) R²=0.0009 both underperform the simple book MLP (b) R²=0.0067 at return\_1. The Transformer is especially poor, suggesting the attention mechanism finds no useful sequential patterns in truncated MBO event sequences.

---

## 4. Information Gaps — Table 2

All gaps computed from MLP R² values (Tier 1) or Tier 2 R² values. Statistical tests: paired t-test on 5-fold R² differences (Wilcoxon when Shapiro-Wilk rejects normality). Holm-Bonferroni correction applied within each gap family.

### 4.1 Δ\_spatial (R²\_book − R²\_bar): Does raw book beat hand-crafted features?

| Horizon | MLP Δ\_R² | 95% CI | Raw p | Corrected p | Passes? |
|---------|----------|--------|-------|-------------|---------|
| return\_1 | +0.0031 | [+0.0017, +0.0046] | 0.025 | 0.96 | No |
| return\_5 | +0.0055 | [+0.0012, +0.0100] | 0.113 | 1.0 | No |
| return\_20 | +0.0115 | [+0.0002, +0.0279] | 0.249 | 1.0 | No |
| return\_100 | +0.0033 | [−0.0151, +0.0229] | 0.786 | 1.0 | No |

The raw book consistently outperforms hand-crafted features (positive gaps at return\_1 through return\_20), but none survive Holm-Bonferroni correction. The raw return\_1 gap of +0.0031 has a nominally low p=0.025 but corrects to p=0.96. The direction is informative — raw book representation is at least as good as engineered features — but the magnitude is negligible.

### 4.2 Δ\_msg\_summary (R²\_book+msg\_sum − R²\_book): Do Category 6 summaries add value?

| Horizon | MLP Δ\_R² | 95% CI | Raw p | Corrected p | Passes? |
|---------|----------|--------|-------|-------------|---------|
| return\_1 | −0.0009 | [−0.0039, +0.0015] | 0.594 | 1.0 | No |
| return\_5 | −0.0015 | [−0.0028, +0.0004] | 0.198 | 1.0 | No |
| return\_20 | −0.0024 | [−0.0061, +0.0006] | 0.260 | 1.0 | No |
| return\_100 | +0.0041 | [−0.0018, +0.0100] | 0.310 | 1.0 | No |

Message summaries add no value at any horizon. The gap is negative at 3 of 4 horizons — the summaries actively degrade performance. This supports the null hypothesis: intra-bar message activity (cancel/add ratios, message rates, toxicity) is already captured by the resulting book state.

### 4.3 Δ\_msg\_learned (R²\_LSTM/Transformer − R²\_book): Do raw message sequences add value? (Tier 2)

**LSTM (Config e):**

| Horizon | Δ\_R² | 95% CI | Raw p | Corrected p | Passes? |
|---------|-------|--------|-------|-------------|---------|
| return\_1 | −0.0026 | [−0.0047, −0.0009] | 0.074 | 1.0 | No |
| return\_5 | −0.0019 | [−0.0029, −0.0009] | 0.034 | 1.0 | No |
| return\_20 | −0.0004 | [−0.0036, +0.0028] | 0.836 | 1.0 | No |
| return\_100 | +0.0040 | [+0.0010, +0.0064] | 0.059 | 1.0 | No |

**Transformer (Config f):**

| Horizon | Δ\_R² | 95% CI | Raw p | Corrected p | Passes? |
|---------|-------|--------|-------|-------------|---------|
| return\_1 | −0.0058 | [−0.0119, −0.0018] | 0.063 | 1.0 | No |
| return\_5 | −0.0024 | [−0.0041, −0.0014] | 0.044 | 1.0 | No |
| return\_20 | −0.0043 | [−0.0065, −0.0024] | 0.030 | 1.0 | No |
| return\_100 | −0.0049 | [−0.0094, +0.0006] | 0.172 | 1.0 | No |

Learned message encoders perform *worse* than the plain book MLP at every horizon except LSTM at return\_100 (+0.004, not significant). The Transformer consistently degrades performance. This is the strongest evidence against the message encoder: even a neural sequence model with attention cannot extract exploitable sequential patterns from MBO events beyond what the book snapshot already captures.

### 4.4 Δ\_temporal (R²\_full\_summary − R²\_book+msg\_sum): Does lookback history add value?

| Horizon | MLP Δ\_R² | 95% CI | Raw p | Corrected p | Passes? |
|---------|----------|--------|-------|-------------|---------|
| return\_1 | −0.0062 | [−0.0084, −0.0043] | 0.006 | 0.25 | No |
| return\_5 | −0.0010 | [−0.0028, +0.0009] | 0.436 | 1.0 | No |
| return\_20 | −0.0025 | [−0.0068, +0.0007] | 0.298 | 1.0 | No |
| return\_100 | −0.0060 | [−0.0124, +0.0004] | 0.179 | 1.0 | No |

Temporal lookback consistently *hurts* — the gap is negative at all four horizons. At return\_1 the degradation is nominally significant (raw p=0.006) but does not survive correction. With 845 input dimensions (20 × 40 book snapshots + 5 message summaries), the MLP overfits the weak signal. A temporal encoder (SSM) operating on this feature space would face the same curse of dimensionality.

### 4.5 Δ\_tier2\_vs\_tier1 (Δ\_msg\_learned − Δ\_msg\_summary): Does learning beat hand-crafting for messages?

**LSTM vs. summaries:**

| Horizon | Δ\_R² | 95% CI | Raw p | Corrected p | Passes? |
|---------|-------|--------|-------|-------------|---------|
| return\_1 | −0.0017 | [−0.0028, −0.0007] | 0.050 | 1.0 | No |
| return\_5 | −0.0004 | [−0.0012, +0.0003] | 0.420 | 1.0 | No |
| return\_20 | +0.0020 | [−0.0011, +0.0051] | 0.353 | 1.0 | No |
| return\_100 | −0.0001 | [−0.0046, +0.0045] | 0.981 | 1.0 | No |

**Transformer vs. summaries:**

| Horizon | Δ\_R² | 95% CI | Raw p | Corrected p | Passes? |
|---------|-------|--------|-------|-------------|---------|
| return\_1 | −0.0049 | [−0.0085, −0.0023] | 0.053 | 1.0 | No |
| return\_5 | −0.0010 | [−0.0024, +0.0005] | 0.304 | 1.0 | No |
| return\_20 | −0.0019 | [−0.0043, +0.0004] | 0.239 | 1.0 | No |
| return\_100 | −0.0090 | [−0.0165, −0.0041] | 0.063 | 1.0 | No |

Learned encoders do not beat hand-crafted summaries. Both LSTM and Transformer gaps are negative or near-zero at every horizon. The neural sequence models add nothing that the simple 5-feature summary doesn't — and in most cases they actively degrade performance due to additional model complexity on insufficient signal.

---

## 5. Threshold Policy Results

**Dual threshold**: (1) R² gap > 20% of baseline R², AND (2) Holm-Bonferroni corrected p < 0.05.

**Result: 0 of 40 tests pass the dual threshold.**

While some gaps pass the relative threshold (the baseline R² is so small or negative that even tiny differences exceed 20% of it), none survive the statistical threshold after multiple comparison correction. The minimum corrected p-value across all 40 tests is 0.25 (Δ\_temporal at return\_1, MLP). The experiment has zero statistically significant findings.

---

## 6. Architecture Decision (§7.2 Simplification Cascade)

| Encoder Stage | Gap | Best Δ\_R² (MLP, return\_1) | Passes Threshold? | Include? |
|---------------|-----|----------------------------|-------------------|----------|
| Spatial (CNN) | Δ\_spatial | +0.0031 | No (corrected p=0.96) | **No** |
| Message encoder | Δ\_msg\_learned | −0.0026 (LSTM) / −0.0058 (Transformer) | No (corrected p=1.0) | **No** |
| Temporal (SSM) | Δ\_temporal | −0.0062 | No (corrected p=0.25) | **No** |

### Decision Matrix Logic

- **Δ\_msg\_learned < 0 AND Δ\_msg\_summary < 0**: Messages carry no incremental info beyond book state. **Drop message encoder.**
- **Δ\_spatial** does not pass threshold (corrected p=0.96): Raw book does not reliably beat hand-crafted features. **Drop CNN — hand-crafted features are sufficient.**
- **Δ\_temporal < 0**: Temporal lookback actively hurts. **Drop SSM.**

### Resulting Architecture

**GBT BASELINE** — Hand-crafted features fed to a gradient-boosted tree model. No deep-learning encoder stages (CNN, message encoder, or SSM) are justified by the data.

---

## 7. Interpretation

### 7.1 Why is signal so weak?

The best R² of 0.0067 at the 1-bar (~5s) horizon is consistent with market microstructure theory for a liquid, electronically-traded instrument like MES. At this timescale:

- **Efficient market pricing** — MES is a derivative of ES, which is among the most traded futures globally. Microstructure alpha is competed away within ticks.
- **Bid-ask bounce dominates returns** — At 5-second intervals, most "returns" are bid-ask bounce noise, not directional moves. The book snapshot captures the current bid-ask spread, explaining why it has the best R².
- **Horizon decay** — Signal vanishes rapidly beyond 1 bar. At 5+ bars (~25s+), the R² is negative, consistent with a no-alpha regime.

### 7.2 Why does temporal lookback hurt?

Config (d) adds 20 × 40 = 800 dimensions of lagged book snapshots. With R² < 0.01 at the 1-bar horizon, the signal-to-noise ratio is far too low to support 845-dimensional regression. The MLP memorizes noise in the training folds and generalizes poorly. This is a textbook overfitting scenario, not evidence that temporal patterns are absent — but an SSM trained on this data would face the same problem.

### 7.3 Why do Tier 2 models underperform?

Two factors compound:

1. **Truncation**: 85% of bars were truncated to 500 events from a median of 1,319. The learned models operated on incomplete message sequences.
2. **Insufficient signal**: Even with complete sequences, there is no extractable signal for the model to learn. The LSTM and Transformer both converge to solutions worse than the book-only MLP, indicating that processing raw event sequences introduces noise rather than reducing it.

### 7.4 Null hypothesis status

The null hypothesis — that the book snapshot is a sufficient statistic for intra-bar messages — **cannot be rejected**. All message-related gaps (summary and learned) are negative or statistically insignificant. The book state at bar close fully summarizes whatever information the message sequence contained.

---

## 8. Caveats and Limitations

1. **Truncation severity**: The 500-event cap was too aggressive for 5-second MES bars (median 1,319 events). A higher cap or downsampling strategy might change Tier 2 results — though the consistent direction of the gaps (negative) suggests this is unlikely.

2. **Model capacity**: The MLP (2×64, ReLU) and LSTM/Transformer (d=32, 1 layer) are deliberately shallow. Deeper models might extract nonlinear patterns, but given R² < 0.007, there is little signal to extract.

3. **Bar timescale**: The 5-second bar aggregation may be too coarse. Sub-second or tick-level analysis might reveal message patterns invisible at this resolution. However, R1 showed no advantage for tick-based bars.

4. **Single instrument**: Results are specific to MES (Micro E-mini S&P 500) in 2022. Other instruments with lower liquidity or different microstructure (e.g., individual equities, commodity futures) might show stronger information decomposition.

5. **CPU training**: All models ran on CPU. This limited hyperparameter search and model scale but is unlikely to change the qualitative finding given the signal ceiling.

---

## 9. Recommendations

1. **Proceed with GBT baseline architecture.** Use the 62 hand-crafted features (or the 40-dim raw book snapshot, which performs comparably) as input to XGBoost/LightGBM. No encoder stages are justified.

2. **R3 (book-encoder-bias)** should still test CNN on raw book images. The Δ\_spatial gap was positive but insignificant — a CNN operating on the 10×4 book structure (not flattened) might capture spatial patterns the MLP misses. This is a cheap test.

3. **R4 (temporal-predictability)** should investigate whether temporal patterns emerge at longer sequence scales or with different features. The current result shows lookback hurts with raw book snapshots, but SSMs with more sophisticated state representations might fare differently.

4. **If GBT R² on the 1-bar horizon remains < 0.01**, consider abandoning return prediction entirely and pivoting to classification (up/down) or spread prediction tasks, which may have better signal properties for MES.
