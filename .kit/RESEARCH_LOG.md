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

## r3b-genuine-tick-bars — CONFIRMED (low confidence)
**Date:** 2026-02-19
**Hypothesis:** At least one genuine tick-bar threshold in [50, 10000] produces CNN spatial R² ≥ 0.107 (20% above time_5s baseline of 0.089).
**Key result:** tick_100 (100 trades/bar, ~5.7s median, ~4,171 bars/day) achieves mean R² = 0.124 (+39% vs baseline). BUT: paired t-test p=0.21, depends on fold 5 outlier (R²=0.259). Excluding fold 5, tick_100 R²=0.091 (COMPARABLE). Inverted-U curve: tick_25 R²=0.064 (WORSE), tick_100 R²=0.124 (BETTER), tick_500 R²=0.050 (WORSE, 3/5 folds only).
**Lesson:** Genuine tick bars at ~100 trades/bar may offer modest CNN improvement, but the evidence is not statistically robust. tick_25 (sub-second) degrades signal — overfitting at fine granularity. tick_500 data-starves the CNN. The tick-bar fix is validated: all 8 thresholds cv=0.189-0.467, p10≠p90. NOTE: run agent fixed bar_feature_export.cpp in-run (StreamingBookBuilder.emit_snapshot trade_count) — needs formal TDD cycle.
**Next:** (1) Tick_100 replication with multi-seed if pursuing event bars, (2) End-to-end CNN classification on tb_label remains HIGHEST PRIORITY, (3) XGBoost hyperparameter tuning.
**Details:** `.kit/results/r3b-genuine-tick-bars/analysis.md`

## r3b-event-bar-cnn — INCONCLUSIVE
**Date:** 2026-02-19
**Hypothesis:** CNN spatial R² on tick-bar book snapshots exceeds time_5s baseline (0.084) by ≥20% at some threshold, indicating event bars improve spatial prediction.
**Key result:** All 4 thresholds WORSE than baseline (peak R²=0.057 at tick_100, Δ=-0.027). BUT: bar construction defect — "tick" bars are actually time bars at different frequencies (bars_per_day std=0.0, duration variance=0 at all thresholds). Hypothesis was never testable.
**Lesson:** The C++ bar_feature_export tool's tick bar construction counts fixed-rate book snapshots (10/s), not trade events. tick_100 = time_10s, tick_500 = time_50s, etc. CNN spatial signal degrades at slower frequencies (time_5s > time_10s > time_50s). Data starvation at tick_1000/tick_1500 (420–732 train samples for 12,128-param CNN) makes those comparison points invalid. Event-bar hypothesis survives untested.
**Next:** Handoff: fix bar_feature_export tick bar construction to count trades. Proceed with time_5s CNN+GBT pipeline (main direction unchanged). Low-priority rerun of R3b with genuine tick bars if bar construction is fixed.
**Details:** `.kit/results/R3b-event-bar-cnn/analysis.md`

## r3-reproduction-pipeline-comparison — REFUTED (Step 2) / CONFIRMED (Step 1)
**Date:** 2026-02-18
**Hypothesis:** R3's CNN R²=0.132 reproduces on R3-format data (Step 1), and the C++ export differs structurally from R3's Python export, causing the 9B/9C R²=0.002 collapse (Step 2).
**Key result:** Step 1 PASS: mean R²=0.1317 (Δ=-0.0003 from R3, per-fold corr=0.9997). Step 2 FAIL: data is byte-identical (identity rate=1.0, max diff=0.0). Root cause: missing TICK_SIZE normalization + per-day z-scoring + test-as-validation leakage. Proper validation R²=0.084 (36% lower than R3's leaked 0.132).
**Lesson:** There was never a "Python vs C++ pipeline" — R3 loaded from the same C++ export as 9B/9C. The 9B/9C failure was caused by omitting TICK_SIZE division on prices and per-day z-scoring on sizes. R3's R²=0.132 includes ~36% inflation from test-as-validation leakage; true CNN R²≈0.084 with proper validation. CNN spatial signal is real but weaker than previously believed.
**Next:** Apply TICK_SIZE normalization + per-day z-scoring in production training pipeline. Re-attempt CNN+GBT integration with corrected normalization and proper validation. Multi-seed robustness study to confirm R²≈0.084.
**Details:** `.kit/results/r3-reproduction-pipeline-comparison/analysis.md`

## cnn-reproduction-diagnostic — REFUTED
**Date:** 2026-02-18
**Hypothesis:** Fixing 3 protocol deviations (z-score normalization, architecture 2→32→64, no cosine LR) would restore CNN mean OOS R² ≥ 0.10 on Phase 9A data (time_5s.csv).
**Key result:** MVE gate FAIL. Fold 5 train R² = 0.002 (threshold: 0.05). Architecture matched R3 exactly (12,128 params, Conv1d 2→59→59, 0% deviation). 5 normalization variants tested — ALL produce R² < 0.002. Z-scored vs raw indistinguishable (0.0015 vs 0.0008). Phase B's post-mortem was wrong. 0/2 evaluated success criteria pass; 3/5 not evaluated (abort cascade).
**Lesson:** The 3 protocol deviations were NOT the root cause — they are inconsequential. The data pipeline difference (Phase 9A C++ export vs R3's Phase 4 Python export) is the primary suspect. The predictive spatial structure is absent from `time_5s.csv`, not hidden behind configuration errors. R6's CNN+GBT recommendation is currently ungrounded — its enabling evidence has not been independently reproduced. Also: the diagnostic spec's architecture description was materially wrong (said 2→32→32 / 4k params; R3 actually used 2→59→59 / 12k params).
**Next:** Data pipeline comparison — reproduce R3's Python-based export from raw .dbn.zst files for the same 19 days. Direct column-by-column comparison, then train CNN on R3-format data. If R3-format data works → handoff to fix C++ export. If it also fails → R3's result may be artifactual.
**Details:** `.kit/results/cnn-reproduction-diagnostic/analysis.md`

## hybrid-model-training — REFUTED
**Date:** 2026-02-18
**Hypothesis:** CNN+GBT Hybrid (Conv1d spatial encoder + 20 hand-crafted features → XGBoost on triple barrier labels) reproduces R3's CNN R²>=0.08 and achieves positive expectancy under base costs.
**Key result:** CNN R²(h=5) = -0.002 across all 5 folds (R3 baseline: 0.132). Train R² = 0.001 — CNN cannot fit training data. XGBoost accuracy 0.41 (>0.33 random) but expectancy -$0.44/trade (base costs). GBT-only outperforms hybrid. 4/10 success criteria pass, 6/10 fail.
**Lesson:** CNN signal is real (R3 proved it) but fragile — does not transfer to a new pipeline with different normalization (z-score both channels vs raw prices) and architecture (Conv1d 2→32→64 vs 2→32→32). The pipeline is broken (train R² ≈ 0), not the hypothesis. XGBoost learns regime identification (volatility, time-of-day) but the edge is thinner than 1 tick/trade.
**Next:** Reproduce R3's exact CNN protocol in Python (raw price normalization, cosine LR, R3 architecture). Fix pipeline before re-attempting integration.
**Details:** `.kit/results/hybrid-model-training/analysis.md`

## Research-Audit — COMPLETE
**Date:** 2026-02-18
**Hypothesis:** N/A (documentation audit, not experiment)
**Key result:** All R1–R6, R4a–R4d findings reconciled. 0 blocking gaps. Ready for model build.
**Lesson:** 10+ experiments reconciled into single source of truth. Temporal vs spatial predictability distinction is the core insight. time_5s + CNN + GBT Hybrid architecture validated.
**Next:** Proceed to Phase B — Python CNN+GBT pipeline via Research kit.
**Details:** `RESEARCH_AUDIT.md`

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
