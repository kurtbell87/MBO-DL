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

## R6-synthesis — CONDITIONAL GO
**Date:** 2026-02-17
**Hypothesis:** Collate R1–R4 findings to determine go/no-go, architecture, bar type, feature set, and labeling method.
**Key result:** CONDITIONAL GO. CNN + GBT Hybrid architecture. R3 CNN R²=0.132 on structured (20,2) book input resolves R2-R3 tension — spatial encoder adds massive value when input preserves adjacency. Message encoder and temporal encoder dropped (R2+R4). Bar type: time_5s. Horizons: h=1 and h=5. Oracle expectancy flagged as open question.
**Lesson:** R2's recommendation to drop the spatial encoder was based on flattened input that destroyed spatial structure. R3's Conv1d on structured input achieves 20× higher R², with statistical significance (CNN vs Attention corrected p=0.042, d=1.86) and a 4.16× retention ratio on the 16-dim embedding. Methodological differences between experiments must be reconciled before architecture decisions — the resolution is not a contradiction but a scope expansion.
**Next:** Resolve open questions: (1) Extract oracle expectancy from Phase 3 C++ test output, (2) Test CNN at h=1, (3) Design CNN+GBT integration pipeline, (4) Estimate transaction costs. Proceed to model architecture build spec.
**Details:** `.kit/results/synthesis/metrics.json`, `.kit/results/synthesis/analysis.md`

## R4-temporal-predictability — NO TEMPORAL SIGNAL
**Date:** 2026-02-17
**Hypothesis:** MES 5-second bar returns contain exploitable autoregressive structure beyond the current-bar feature set (lagged returns, rolling volatility, momentum features).
**Key result:** All 36 Tier 1 AR configs produce negative R² (best: AR-10 GBT h1 = −0.0002). All Tier 2 temporal augmentation gaps fail dual threshold — Δ_temporal_book ranges from −0.002 to +0.0004, none significant after Holm-Bonferroni. Temporal-Only R² ≈ 0 at h=1, negative at all longer horizons. 0/52 tests pass correction.
**Lesson:** MES returns at 5-second bars are a martingale difference sequence. No temporal representation — lagged returns, rolling volatility, momentum, mean reversion — adds predictive value beyond current-bar features. Converges with R2 Δ_temporal = −0.006 finding: the temporal information gap is zero regardless of whether raw book snapshots (R2) or low-dimensional derived features (R4) are used. GBT mitigates overfitting versus linear AR but still finds no signal.
**Next:** Drop SSM / temporal encoder from architecture. Combined with R2 (no spatial/message gap) and R3 results, finalize synthesis (Phase 6). Static current-bar features with GBT is the recommended architecture.
**Details:** `.kit/results/temporal-predictability/metrics.json`, `.kit/results/temporal-predictability/analysis.md`

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
