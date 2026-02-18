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

## R4d-temporal-predictability-dollar-tick-actionable — CONFIRMED
**Date:** 2026-02-18
**Hypothesis:** Temporal features fail the dual threshold on dollar bars at empirically calibrated actionable timescales (≥5s) and tick bars at 5-minute timescales.
**Key result:** 0/38 dual threshold passes across 5 operating points (dollar $5M/7s, $10M/14s, $50M/69s; tick 500/50s, 3000/300s). Dollar $5M (47,865 bars): AR R²=−0.00035, Δ_temporal_book=−0.0018 (p=1.0). All Tier 1 AR R² negative except dollar_50M h=1 (+0.0025, noise: std=6×mean, p=1.0). Empirical calibration table for 10 thresholds produced — dollar bars ARE achievable at actionable timescales but contain no temporal signal. Cumulative R4 chain: 0/168+ dual threshold passes across 7 bar types, 0.14s–300s.
**Lesson:** R4b's sub-second temporal signal (Temporal-Only R²=+0.012 at $25k/0.14s bars) decays to noise by $5M/7s bars (R²=−0.0005). Volume-math overestimates bar duration by consistent 4× factor (empirical ≈ 0.25× estimate). Calibration table is the lasting deliverable. R4 line closed: 7 bar types, 0.14s–300s, zero signal.
**Next:** R4 line permanently closed. Proceed to CNN at h=1 (P1) and transaction cost sensitivity (P2) — the remaining open questions before model build spec.
**Details:** `.kit/results/temporal-predictability-dollar-tick-actionable/analysis.md`

## R4c-temporal-predictability-completion — CONFIRMED
**Date:** 2026-02-18
**Hypothesis:** Temporal features fail the dual threshold across all remaining gaps: tick bars (tick_50/100/250), extended horizons (h=200/500/1000 on time_5s, ~17-83min), and event bars at actionable timescales (≥5s).
**Key result:** 0/54+ dual threshold passes. All Tier 1 AR R² negative across 4 bar types × 7 horizons. Extended horizons show accelerating degradation (h1000 R²=−0.152). Dollar bars entirely sub-actionable (max ~0.9s/bar at $1M). Tick bars at 10s and 25s timescales match time_5s null result.
**Lesson:** MES returns are martingale across all bar types (time, tick, dollar, volume) and timescales (5s to 83min). The R4 chain (R4→R4b→R4c) tested 6 bar types, 7+ horizons, 3 model classes, 5 feature configs, 200+ statistical tests — zero pass Rule 2. Temporal encoder dropped permanently.
**Next:** Proceed to model architecture build spec: CNN+GBT Hybrid, static features, time_5s bars, triple barrier labels. No temporal encoder. The R4 line is closed.
**Details:** `.kit/results/temporal-predictability-completion/analysis.md`

## R4b-temporal-predictability-event-bars — NO TEMPORAL SIGNAL (robust)
**Date:** 2026-02-18
**Hypothesis:** Dollar bars (and possibly volume bars) contain exploitable autoregressive structure absent from time bars. Temporal feature augmentation improves prediction on event-driven bars.
**Key result:** Volume_100: all 36 Tier 1 AR configs negative R² (matches time_5s). Dollar_25k: marginal positive AR R² at h=1 only (R²=0.0006, corrected p=0.014), but temporal augmentation fails dual threshold at every horizon. Temporal-Only has standalone power (h=1 R²=0.012, p<0.001) but is redundant with static features (weighted_imbalance captures the same signal). Linear >= GBT — signal is linear, not nonlinear.
**Lesson:** R4's "NO TEMPORAL SIGNAL" conclusion generalizes across all three bar types. Dollar_25k's marginal AR signal operates at ~140ms (genuine microstructure, not retail-tradeable) and is already captured by hand-crafted features. The R1 prior (dollar_25k AR R²=0.01131, 33x higher) was inflated by in-sample bias — rigorous 5-fold CV shows 18x less (R²=0.0006). No temporal encoder or SSM is justified regardless of bar type.
**Next:** R4b closes the experimental gap. Proceed to model architecture build spec with full confidence: CNN+GBT, no temporal encoder, time_5s bars, static features.
**Details:** `.kit/results/temporal-predictability-event-bars/analysis.md`

## R4b-temporal-predictability-event-bars — MARGINAL SIGNAL (REDUNDANT)
**Date:** 2026-02-18
**Hypothesis:** Dollar bars (and possibly volume bars) contain exploitable autoregressive structure absent from time bars, justifying a temporal encoder on event-driven bar sequences.
**Key result:** Dollar_25k has weak positive AR R² at sub-second horizons (h=1: +0.000633, h=5: +0.000364) — the only bar type with positive Tier 1 R². Temporal-Only features have standalone power (h=1: R²=0.012, p=0.0005). However, temporal augmentation fails the dual threshold for ALL bar types (0/48 gaps pass). Volume_100 shows no signal (identical to time_5s). Dollar_25k static features achieve R²=0.080 at h=1 — 10-25× higher than time_5s — but temporal features add nothing on top.
**Lesson:** Dollar bars' AR structure (33× higher than time_5s per R1) is real but operates at ~140ms timescales — genuine HFT microstructure, not tradeable at retail. The signal is LINEAR (Ridge ≥ GBT), low-dimensional, and entirely redundant with static book features. The high static-feature R² on dollar bars is the more interesting finding: book state is far more predictive at sub-second timescales, but this doesn't transfer to the 5s execution horizon. At equivalent clock time (~5s), dollar bars show no temporal advantage over time bars.
**Next:** R4 "NO TEMPORAL SIGNAL" conclusion is robust across bar types. Proceed with CNN+GBT on time_5s, no temporal encoder, with high confidence. The experimental chain gap (R4 only tested time_5s) is closed.
**Details:** `.kit/results/temporal-predictability-event-bars/analysis.md`, `.kit/results/temporal-predictability-event-bars/dollar_25k/metrics.json`, `.kit/results/temporal-predictability-event-bars/volume_100/metrics.json`

## R7-oracle-expectancy — GO
**Date:** 2026-02-17
**Hypothesis:** The oracle produces positive expectancy after realistic MES execution costs (commission $0.62/side, fixed spread 1 tick) across out-of-sample days.
**Key result:** GO. Triple barrier: $4.00/trade expectancy, PF=3.30, WR=64.3%, Sharpe=0.362, net PnL=$19,479 over 19 days (4,873 trades). All 6 success criteria pass. First-to-hit: $1.56/trade, PF=2.11, WR=53.2%, 5/6 pass (DD exceeds 50× expectancy). Per-quarter TB expectancy: Q1=$5.39, Q2=$3.16, Q3=$3.41, Q4=$3.39 — stable across all quarters.
**Lesson:** Triple barrier labeling is strictly superior to first-to-hit for MES at these oracle parameters (target=10, stop=5, TP=20 ticks, vol_horizon=500). Higher win rate (64% vs 53%) and dramatically lower drawdown ($152 vs $400) drive the difference. The oracle's edge is real and robust — it doesn't concentrate in a single quarter or regime. At 256 trades/day on time_5s bars, there is ample sample size.
**Next:** CONDITIONAL GO upgraded to full GO. Proceed to model architecture build spec: CNN + GBT Hybrid with triple barrier labels. Remaining open questions: CNN at h=1, transaction cost sensitivity, CNN+GBT integration pipeline.
**Details:** `.kit/results/oracle-expectancy/metrics.json`, `.kit/results/oracle-expectancy/summary.json`

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
