# Research Log

Cumulative findings from all experiments. Each entry is a concise summary — full details are in the linked analysis documents.

Read this file FIRST when starting any new research task. It is the institutional memory of this project.

---

<!-- New entries go at the top. Format:

## [exp-NNN-name] — [CONFIRMED/REFUTED/INCONCLUSIVE]
**Date:** YYYY-MM-DD
**Hypothesis:** [one line]
**Key result:** [one line with the critical number]
**Lesson:** [one line — what we learned]
**Next:** [one line — what to do about it]
**Details:** results/exp-NNN/analysis.md

-->

## R2-info-decomposition — FEATURES SUFFICIENT
**Date:** 2026-02-17
**Hypothesis:** Predictive information about future returns decomposes across three sources (spatial book state, intra-bar message sequence, temporal history) — each maps to an encoder stage (CNN, message encoder, SSM).
**Key result:** No information gap passes the dual threshold (relative >20% of baseline R² AND corrected p<0.05). Best R² is Config (b) MLP on fwd_return_1: 0.0067±0.0034. All longer horizons (5/20/100-bar) show negative R². Δ_spatial=+0.003 (p=0.96), Δ_msg_summary=−0.001 (p=1.0), Δ_msg_learned=−0.003 (p=1.0), Δ_temporal=−0.006 (p=0.25). 0/40 tests pass Holm-Bonferroni correction.
**Lesson:** At the 5-second bar scale on MES, the raw 40-dim book snapshot captures all available linear/MLP-extractable predictive signal. Hand-crafted features (62-dim) add nothing over raw book. Message summaries and learned message encoders add nothing over book state — the null hypothesis (book is sufficient statistic for messages) cannot be rejected. Temporal lookback hurts (overfitting). Signal is extremely weak (R²<0.007) and confined to the 1-bar (~5s) horizon.
**Next:** Architecture recommendation is GBT_BASELINE — hand-crafted features are sufficient, no CNN/message-encoder/SSM stages justified. R3 (book-encoder-bias) should test if CNNs on raw book images outperform flattened features. R4 (temporal-predictability) should investigate if SSM adds value at longer sequence scales.
**Details:** `.kit/results/info-decomposition/metrics.json`, `.kit/results/info-decomposition/analysis.md`

## R1-subordination-test — REFUTED
**Date:** 2026-02-17
**Hypothesis:** Event-driven bars (volume, tick, dollar) produce more IID Gaussian returns than time bars for MES, per the Clark (1973) / Ané & Geman (2000) subordination model.
**Key result:** Best event bar (dollar_25k, mean_rank=5.0) did not beat best time bar (time_1s, mean_rank=5.14) on the 3 primary metrics (JB normality, ARCH heteroskedasticity, ACF volatility clustering). 0/3 primary pairwise tests significant after Holm-Bonferroni correction.
**Lesson:** Subordination is a poor fit for MES microstructure. Dollar bars show significantly higher AR R² (temporal predictability, p<0.001) — the opposite of what the theory predicts. Volume and tick bars are no better than matched-frequency time bars on normality or heteroskedasticity. Quarter robustness fails: all 4 quarters show reversed rankings, indicating regime dependence.
**Next:** Proceed to R2 (info-decomposition) and R3 (book-encoder-bias). For R4 (temporal-predictability), use time bars as baseline — no justification for event-driven bar preference.
**Details:** `.kit/results/subordination-test/metrics.json`
