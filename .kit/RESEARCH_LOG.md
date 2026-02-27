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

## volume-flow-conditioned-entry — REFUTED (Outcome B, diagnostic evaporates in simulation)
**Date:** 2026-02-27
**Hypothesis:** Volume/activity-based entry gating reduces timeout fraction below 41% and improves per-trade expectancy toward $3.50.
**Key result:** 20pp bar-level diagnostic signal (all 5 features "strong" tier) evaporates to **1.76pp max** in sequential simulation. Best config exp=$3.10/trade (stacked, vs $3.50 target), min_acct=$33K (vs $30K target), timeout=41.2% (vs 36.3% target). SC-2/3/5 FAIL.
**Lesson:** Sequential execution's 66.1% hold-skip rate already self-selects for high-activity bars — gating on activity features provides negligible incremental selection. Timeout fraction (~41.3%) is a structural constant of the volume horizon mechanism at 19:7/50K, invariant to BOTH time-of-day (PR #40) and volume/activity gating. Entry-time filtering is exhausted as an intervention class.
**Next:** (1) Barrier geometry re-parameterization (reduce volume horizon 50K→10-25K, or reduce time horizon 3600s→600-1200s). (2) Accept cutoff=270 ($3.02/trade, $34K, Calmar 2.49) and proceed to paper trading. (3) Regime-conditional trading (Q1-Q2 only).
**Details:** `.kit/results/volume-flow-conditioned-entry/analysis.md`

## timeout-filtered-sequential — REFUTED (Outcome B, mechanism falsified)
**Date:** 2026-02-27
**Hypothesis:** Time-of-day entry cutoff (minutes_since_open <= cutoff) achieves >= $3.50/trade expectancy and <= $30K min account by avoiding timeout-prone late-day entries.
**Key result:** Timeout fraction is **invariant at ~41.3%** across all 7 cutoff levels (range 0.21pp). Time-of-day does NOT determine timeouts. Best cutoff=270: exp $3.02/trade (< $3.50 target), min_acct $34K (> $30K target). Calmar +15%, DD -29%, but annual PnL -18% ($104K→$85K). ROC improves 216%→249%.
**Lesson:** Timeouts are driven by the volume horizon (50,000 contracts), not clock time — even entries 4.5h into RTH have ample time for the typical 2.3-minute barrier race. The cutoff=270 improvement comes from hold-skip restructuring (66%→34%), not timeout avoidance. Late-day trades in the 5–5.75h window are actually BETTER than average (SC-S4 U-shape).
**Next:** (1) Volume-flow conditioned entry — target the actual timeout mechanism. (2) Volatility-conditional entry (volatility_50 at entry). (3) Accept $34K/$25.5K account sizing and proceed to paper trading.
**Details:** `.kit/results/timeout-filtered-sequential/analysis.md`

## trade-level-risk-metrics — REFUTED (SC-2, SC-4, SC-8 fail; productive)
**Date:** 2026-02-27
**Hypothesis:** Sequential 1-contract execution at 19:7 produces per-trade expectancy >= $0.50 with min account <= $5,000.
**Key result:** seq_exp = **$2.50/trade** (5x threshold), annual $103,605/1-MES, BUT min_account = **$48,000** (9.6x the $5K target). 162 trades/day (vs expected 40-80), avg_bars_held = 28 (vs expected 75), hold_skip_rate = 66.1% (vs expected 43%). Win rate 49.93% — edge is pure payoff asymmetry, zero directional skill. Timeout trades dilute $5.00 barrier-hit expectancy to $2.50.
**Lesson:** Sequential entry creates a non-random selection pattern: entries cluster during volatile moments (shorter barrier races, higher per-trade exp) but exit into calm periods (higher hold-skip at entry attempts). The edge is real and substantial, but risk sizing requires medium accounts ($48K all-paths, $26.6K 95%-paths), not the retail micro-accounts the hypothesis targeted. Three spec assumptions were wrong: avg hold (28 vs 75), hold-skip rate (66% vs 43%), and trades/day (162 vs 50-80). All trace to volatility-timing selection bias.
**Next:** (1) Timeout-filtered sequential execution — avoid entries where barrier unlikely to resolve before day end (+$1-2/trade est.). (2) Multi-contract scaling analysis (N=2,5,10,20,36). (3) Long-perspective labels at 19:7 (existing P0).
**Details:** `.kit/results/trade-level-risk-metrics/analysis.md`

## cpcv-corrected-costs — CONFIRMED (Outcome A)
**Date:** 2026-02-27
**Hypothesis:** Two-stage XGBoost at 19:7 (w=1.0, T=0.50) achieves CPCV mean realized expectancy > $0.00 under corrected-base costs ($2.49 RT) with PBO < 0.50 across 45 splits.
**Key result:** CPCV mean exp = **$1.81/trade** (95% CI [$1.46, $2.16]), PBO = 0.067 (3/45 negative), t = 10.29, p < 1e-13. Holdout: $1.46/trade. All 6 SC pass, all 5 sanity checks pass. Break-even RT = $4.30. Edge is structural (payoff asymmetry, not directional skill — pooled dir acc = 50.16%). All 10 temporal groups positive. Q3-Q4 stronger ($2.18-$2.93) than Q1-Q2 ($1.39-$1.49) due to volatility-dependent barrier reachability.
**Lesson:** The first statistically validated positive-expectancy configuration in the project. The edge comes from the 19:7 payoff ratio (breakeven acc = 34.6%) combined with Stage 1 volatility-based reachability filtering — not from direction prediction. Cost correction ($3.74 → $2.49) was the key unlock. Hold-bar variance (-$9.39 to +$3.48 per split) is the dominant risk factor. Pessimistic costs ($4.99) are not viable (mean -$0.69).
**Next:** Paper trading infrastructure (R|API+ integration, 1 /MES). Hold-bar exit optimization. Multi-year validation.
**Details:** `.kit/results/cpcv-corrected-costs/analysis.md`

## class-weighted-stage1 — REFUTED (Outcome C)
**Date:** 2026-02-27
**Hypothesis:** There exists a (scale_pos_weight, threshold) pair such that the two-stage pipeline at 19:7 achieves realized WF expectancy > $1.50/trade with trade rate > 15%, by spreading the Stage 1 probability distribution.
**Key result:** IQR DECREASES monotonically with lower weight (4.0pp -> 3.8 -> 3.2 -> 2.0 -> 0.8pp at weights 1.0/0.5/0.33/0.25/0.20). Class weighting COMPRESSES rather than spreads the probability surface. Recall collapses 465x (93% -> 0.2%) at first step. Precision also decreases (0.56 -> 0.43). Weights 0.25/0.20 produce zero trades. Baseline (w=1.0, T=0.50, $0.90/trade) remains optimal.
**Lesson:** `scale_pos_weight < 1` in XGBoost binary:logistic is a logit-shift, not a spread mechanism — it shifts the sigmoid operating point uniformly, compressing ALL probabilities downward. The probability compression on this reachability task is structural and untuneable. Three experiments (threshold-sweep, class-weighting, geometry variation) have now exhausted all parameter-level interventions on the two-stage pipeline.
**Next:** Long-perspective labels at 19:7 (P0 open question — changes labeling scheme, not model parameters). Probability recalibration (Platt/isotonic) as low-cost test. Regime-conditional trading (Q1-Q2 only).
**Details:** `.kit/results/class-weighted-stage1/analysis.md`

## threshold-sweep — REFUTED (Outcome C)
**Date:** 2026-02-27
**Hypothesis:** There exists a Stage 1 probability threshold T* > 0.5 such that the two-stage pipeline at 19:7 achieves realized WF expectancy > $1.50/trade with trade rate > 15%, by reducing hold-bar fraction below 25%.
**Key result:** No threshold achieves exp > $1.50 at >15% trade rate. Optimal = baseline (T=0.50, $0.90/trade, 85.2%). Root cause: 80.6% of Stage 1 P(directional) predictions cluster in [0.50, 0.60] — trade rate cliff from 64.5% (T=0.55) to 4.5% (T=0.60). Dir-bar PnL improves at moderate thresholds ($3.77→$4.49→$6.07 at T=0.50/0.60/0.65), confirming signal exists, but model cannot generate enough high-confidence predictions for viable trading. SC-1/2/3 FAIL, SC-4 PASS ($3.77 > $3.00).
**Lesson:** XGBoost binary:logistic on this reachability task produces near-degenerate probabilities (IQR [0.536, 0.576] — 4pp wide). Threshold optimization requires a smooth confidence gradient; this model provides a step function. The $0.90/trade at T=0.50 is the ceiling for this pipeline + geometry combination without changing the model or labeling scheme.
**Next:** Long-perspective labels at varied geometries (P0 open question) — changes the labeling scheme, not the threshold. Probability recalibration (Platt/isotonic) unlikely to help but quick to test. Class-weighted loss could reshape probability distribution at source.
**Details:** `.kit/results/threshold-sweep/analysis.md`

## pnl-realized-return — REFUTED (Outcome C — SC-2 fails, but SC-1 passes)
**Date:** 2026-02-27
**Hypothesis:** Under realized-return PnL model (actual forward returns on hold-bar trades instead of full barrier payoffs), two-stage pipeline at 19:7 achieves WF per-trade expectancy > $0.00 with hold-bar directional accuracy > 52%.
**Key result:** Realized WF expectancy $0.90/trade (SC-1 PASS), but hold-bar dir accuracy 51.04% < 52% (SC-2 FAIL). Hold-bar gross PnL +$1.06 (SC-3 PASS), 3/3 folds positive (SC-4 PASS). Result driven by Fold 2 outlier ($2.54 vs $0.01/$0.16). Hold-bar fwd returns unbounded (p10/p90 = -63/+63 ticks, not ±19 as spec assumed — volume horizon ends barrier race early). PnL decomposition: 0.556 × $3.77 (dir) + 0.444 × (-$2.68) (hold) = $0.90. 10:5 uniformly negative (-$1.65). Break-even RT: $4.64. Strategy marginally viable but fragile.
**Lesson:** Hold-bar directional signal exists (51.04% > 50%, gross PnL +$1.06) but is sub-threshold and unstable across folds (48-54%). Hold bars destroy 57% of directional-bar edge. The volume horizon (50,000) causes barrier race truncation, making hold-bar trades unbounded risk exposure rather than bounded range-bets. Spec's first-principles analysis was wrong about return bounds. Realized ($0.90) exceeds spec's conservative estimate ($0.44) — hold-bar model adds marginal value vs pure cost drag.
**Next:** Stage 1 threshold optimization (sweep 0.5-0.9) to reduce hold-bar exposure. At threshold 0.80 (est. 5% hold), PnL decomposition predicts ~$3.45/trade. Then CPCV validation at optimal threshold.
**Details:** `.kit/results/pnl-realized-return/analysis.md`

## 2class-directional — CONFIRMED (with critical PnL model caveat)
**Date:** 2026-02-26
**Hypothesis:** Two-stage XGBoost (Stage 1: directional-vs-hold filter, Stage 2: long-vs-short) at 19:7 geometry achieves positive WF expectancy with >10% trade rate.
**Key result:** All 4 SCs pass. Trade rate 85.2% (301x increase from 3-class's 0.28%). Dir accuracy 50.05% (above 38.4% BEV). Reported expectancy $3.78/trade BUT likely inflated ~8x by PnL model assigning full barrier payoffs to hold-bar trades (44.4% of trades). Corrected estimate ~$0.44/trade ± $0.16 — positive but marginal. 10:5 control matches 3-class baseline (-$0.51 vs -$0.50). Stage 2 at 19:7 learns NO directional features (top-10 all volatility/activity). Direction prediction is essentially a coin flip; economics driven entirely by 2.71:1 payoff ratio.
**Lesson:** Two-stage decomposition successfully liberates trade volume at wide barriers by removing cross-entropy wrong-direction penalty from Stage 1. Reachability is learnable (58.6% Stage 1 acc). Direction at 19:7 is not (~50%). The favorable payoff ratio, not prediction skill, is the value driver. Hold-bar PnL treatment is the dominant factor in economic results.
**Next:** (1) Corrected PnL model validation — highest priority, resolves the 8x uncertainty. (2) CPCV with corrected PnL for proper confidence intervals. (3) Stage 1 threshold optimization to reduce label0_hit_rate. (4) Intermediate geometry exploration (14:6, 15:5) for better direction/payoff trade-off.
**Details:** `.kit/results/2class-directional/analysis.md`

## label-geometry-1h — INCONCLUSIVE
**Date:** 2026-02-26
**Hypothesis:** XGBoost at high-ratio geometries (15:3, 19:7, 20:3) with 1-hour time horizon achieves CPCV directional accuracy > breakeven WR + 2pp, producing positive expectancy.
**Key result:** Time horizon fix SUCCESS (hold rate 90.7%→32.6% at 10:5). But model refuses to trade at high-ratio geometries (0.003-0.28% directional prediction rate at 15:3/19:7/20:3). At 10:5 (the only geometry with real trade volume), directional accuracy 50.67% is 2.6pp BELOW breakeven (53.28%), expectancy -$0.49/trade. 19:7 CPCV dir_acc 55.9% technically passes SC-3 but on 0.28% of bars — holdout collapses to 50.0% on 52 trades (coin flip, 96.8% label0_hit_rate). SC-S1 abort criterion triggered (baseline accuracy 0.384 < 0.40).
**Lesson:** The model's directional accuracy is ~50-51% regardless of label type, geometry, or hyperparameters — a hard ceiling of the 20-feature set. High-ratio geometries lower breakeven but the model cannot be induced to trade (defaults to hold-prediction even at 47% hold rate). The geometry hypothesis is theoretically sound (oracle profitable at all geometries) but unexploitable with the current model/features. Balanced bidirectional labels are 6.5pp harder than long-perspective labels (0.384 vs 0.449 3-class accuracy).
**Next:** (1) 2-class formulation (directional-vs-hold) to decouple barrier-reachability from direction prediction. (2) Class-weighted XGBoost to force directional predictions at 19:7. (3) Long-perspective labels at varied geometries (original P0 still open).
**Details:** `.kit/results/label-geometry-1h/analysis.md`

## label-geometry-phase1 — REFUTED
**Date:** 2026-02-26
**Hypothesis:** XGBoost at high-ratio geometries (15:3, 19:7, 20:3) achieves directional accuracy > breakeven WR + 2pp on bidirectional labels, producing positive expectancy.
**Key result:** 3/4 geometries SKIPPED (>95% hold: 15:3=97.1%, 19:7=98.6%, 20:3=98.9%). Control (10:5) has 90.7% hold — model degenerates to hold-predictor (directional recall <2%, directional accuracy 53.65% = +0.37pp above breakeven). CPCV exp -$0.610. Holdout +$0.069 ($1.69/day on 0.53% trading rate).
**Lesson:** Bidirectional triple barrier labels on time_5s bars produce fundamentally degenerate class distributions at ALL geometries. The ~45% accuracy baseline was on long-perspective labels with balanced classes. Bidirectional labels + 5s bars = ~91-99% hold. The geometry hypothesis was untestable under these label conditions.
**Next:** Re-run geometry sweep on long-perspective labels (--legacy-labels) which produce balanced distributions. Verify class balance at varied geometries first. The geometry hypothesis survives in principle.
**Details:** `.kit/results/label-geometry-phase1/analysis.md`

## synthesis-v2 — CONFIRMED
**Date:** 2026-02-26
**Hypothesis:** Comprehensive synthesis of 23 experiments establishes GO status — label geometry training at breakeven-favorable ratios has >50% prior of positive expectancy.
**Key result:** GO verdict (55-60% prior). Strategic pivot: breakeven WR, not oracle ceiling, is the correct viability metric. At 15:3 geometry, BEV WR = 33.3% — 12pp below model's 45% accuracy. 8 closed lines (CNN, temporal, message, XGB tuning, oracle metric, event bars, pipeline hypothesis, CNN+GBT hybrid). GBT-only on 20 features is canonical architecture.
**Lesson:** The project was optimizing the wrong variable for 6+ experiments. Oracle net exp ranking inversely correlates with model viability at high ratios. The model doesn't need to improve — the payoff structure needs to change.
**Next:** Phase 1 label geometry training at 4 geometries (10:5, 15:3, 19:7, 20:3). Resolves the central uncertainty: does accuracy transfer to high-ratio geometries?
**Details:** `.kit/results/synthesis/analysis.md`, `.kit/results/synthesis-v2/analysis.md`

## label-design-sensitivity — REFUTED (Outcome C — abort criterion miscalibrated)
**Date:** 2026-02-26
**Hypothesis:** Widening target/stop ratio from 2:1 (10:5) to >=3:1 lowers breakeven WR below model's ~45% accuracy, enabling positive expectancy. SC-2 gate: oracle net exp > $5.00/trade.
**Key result:** 0/123 geometries pass $5.00 oracle net exp at base costs ($3.74 RT). Peak $4.126 at (16:10). BUT the $5.00 threshold was the wrong metric — breakeven WR at (15:3) is 33.3%, providing 12pp margin below model's 45% accuracy. Model profitable at (15:3) down to ~35% directional accuracy. Abort prevented Phase 1 training.
**Lesson:** Oracle net expectancy and model viability are different metrics. High-ratio geometries (5:1+) have LOW oracle net exp but the most favorable breakeven WRs. The project was optimizing the wrong variable — the binding constraint is breakeven WR vs model accuracy, not oracle ceiling. Oracle margin (~10-12pp) is geometry-invariant. Cost sensitivity is extreme ($1.25 RT difference = 0 vs 8 geometries above $5.00).
**Next:** (1) Rerun Phase 1 directly: train XGBoost at 4 geometries (10:5, 15:3, 19:7, 20:3) selected by breakeven WR diversity instead of oracle net exp. Skip oracle gate. (2) Key measurement: does model accuracy transfer to high-ratio geometries?
**Details:** `.kit/results/label-design-sensitivity/analysis.md`

## xgb-hyperparam-tuning — REFUTED (Outcome C — MARGINAL)
**Date:** 2026-02-25
**Hypothesis:** Systematic XGBoost hyperparameter tuning on full-year 1.16M-bar dataset improves 3-class accuracy by ≥2.0pp (0.449→0.469) and pushes CPCV expectancy to breakeven ($0.00) under base costs.
**Key result:** CPCV accuracy 0.4504 (delta +0.15pp — landscape is a plateau, 64 configs span only 0.33pp, std=0.0006). CPCV expectancy -$0.001 (delta +$0.065 from -$0.066 — 98.4% reduction via class rebalancing, not accuracy). Breakeven RT cost = $3.74 exactly. Walk-forward expectancy -$0.140 (much more pessimistic than CPCV). SC-1/2/3 FAIL, SC-4/5 PASS. 50/64 configs beat default. Holdout +2.0pp (0.403→0.423). Long recall worsened (0.201→0.149). volatility_50 gain share = 49.7%.
**Lesson:** The 20-feature set, not hyperparameters, is the binding constraint on XGBoost accuracy — the entire search landscape is a 0.33pp plateau. Expectancy is more sensitive to class prediction distribution than raw accuracy (0.15pp acc → $0.065 exp via suppressed longs). Label design and class weighting are higher-leverage interventions than tuning.
**Next:** (1) Label design sensitivity — wider target (15:3 ratio → breakeven ~42.5% vs current 45%) is highest priority. (2) 2-class short/no-short formulation — long recall 0.149 means the model can't predict longs. (3) Class-weighted XGBoost with PnL-aligned loss.
**Details:** `.kit/results/xgb-hyperparam-tuning/analysis.md`

## e2e-cnn-classification — REFUTED (Outcome D)
**Date:** 2026-02-22
**Hypothesis:** End-to-end Conv1d CNN trained on 3-class triple barrier labels (cross-entropy, no intermediate regression) on full-year data (251 days, 1.16M bars) achieves accuracy >= 0.42 and breakeven expectancy.
**Key result:** E2E-CNN CPCV mean accuracy = 0.390 (BELOW 9E's 0.419). GBT-only accuracy = 0.449. CNN underperforms GBT by -5.9pp accuracy and -$0.069 expectancy. Outcome D: hand-crafted features beat CNN for classification. CPCV 45 splits, 10-group, PBO=0.222 (not overfit). Holdout accuracy 0.421, expectancy -$0.204 (GBT). GBT marginally positive in Q1 (+$0.003) and Q2 (+$0.029) under base costs, negative in Q3-Q4. Deflated Sharpe ~0. CNN+Features augmented config skipped (wall-clock). CNN trained on CPU (~180-320s/split), not GPU.
**Lesson:** The CNN spatial signal (R²=0.089 for regression) does not encode class-discriminative book patterns. End-to-end cross-entropy training is 5.9pp WORSE than XGBoost on 20 hand-crafted features. The 16-dim penultimate layer learns return-variance features, not class boundaries. The regression-to-classification bottleneck was not the problem — CNN simply doesn't add value for classification. GBT's Q1-Q2 positive expectancy (with default hyperparams) is the most actionable signal: the edge exists in those regimes but is consumed by Q3-Q4. Long (+1) recall is only 0.21 vs short (-1) 0.45 — model is asymmetrically confident on shorts.
**Next:** (1) XGBoost hyperparameter tuning on full-year data (default params, never optimized — most promising given Q1-Q2 profitability). (2) Label design sensitivity (wider target / narrower stop to lower breakeven). (3) Regime-conditional trading (Q1-Q2 only if robust). CNN line closed for classification.
**Details:** `.kit/results/e2e-cnn-classification/metrics.json`

## full-year-export — CONFIRMED
**Date:** 2026-02-20
**Hypothesis:** bar_feature_export (Parquet) scales to all ~250 RTH trading days in 2022 without systematic failures or data corruption. Full-year dataset production-ready.
**Key result:** 251/251 days exported, 1,160,150 total rows, 0 failures, 0 duplicate timestamps, 0 day gaps. 19/19 reference days validate within float32 precision (max rel_err=4.99e-6; forward returns exact at 0.0). Quarter balance: Q1=62, Q2=62, Q3=64, Q4=63. Wall-clock: 77s (11-way parallel). Schema: 149 columns, zstd compression, 255.7 MB total.
**Lesson:** Export infrastructure is production-ready and fast (~2.3s/day sequential, 77s total parallel). Parquet is canonical format (3.1x compression, proper fwd_return_N naming). Contract rollover required adding MESH3 for post-Dec days. SC-4 tolerance of 1e-10 was unrealistic for float32 data — actual 4.99e-6 is representational, not corruption.
**Next:** Full-year dataset unblocks: (1) end-to-end CNN classification on full year (13x more data), (2) full-year CNN+GBT replication, (3) label design sensitivity studies.
**Details:** `.kit/results/full-year-export/analysis.md`

## r3b-genuine-tick-bars — CONFIRMED (low confidence)
**Date:** 2026-02-19
**Hypothesis:** At least one genuine tick-bar threshold produces CNN spatial R² >= 0.107 (20%+ above time_5s baseline of 0.089).
**Key result:** tick_100 mean OOS R² = 0.124 (delta +0.035 vs baseline), but paired t-test p = 0.21 (not significant) — verdict depends entirely on fold 5's anomalous R² = 0.259 (excluding fold 5: mean = 0.091, COMPARABLE). tick_25 WORSE (0.064), tick_500 WORSE (0.050, incomplete 3/5 folds). Inverted-U curve. Data volume confound ruled out (r = -0.149). Tick-bar fix validated: all 8 thresholds have cv 0.188-0.467, p10 != p90.
**Lesson:** Genuine trade-event tick bars at ~100 trades/bar (~5.7s, ~4,171 bars/day — near-matched to time_5s) show a promising but statistically fragile improvement. The 39% relative R² gain is driven by a single fold and a single threshold. Not actionable without replication. Sub-second tick bars (tick_25) degrade signal — there is an optimal granularity, not a monotonic benefit from event bars.
**Next:** (1) End-to-end CNN classification on tb_label (P1, unchanged — the regression-to-classification bottleneck is the main obstacle regardless of bar type). (2) If pursuing event bars: tick_100 replication with 3 seeds/fold and adjacent thresholds (tick_50, tick_200) to test whether the peak is broad or noise.
**Details:** `.kit/results/r3b-genuine-tick-bars/analysis.md`

## hybrid-model-corrected — REFUTED (Outcome B)
**Date:** 2026-02-19
**Hypothesis:** CNN+GBT Hybrid with corrected normalization (TICK_SIZE division + per-day z-scoring) and proper validation achieves R²>=0.05, XGBoost accuracy>=0.38, and expectancy>=$0.50/trade under base costs.
**Key result:** CNN R²=0.089 (reproduces 9D's 0.084, 5/5 folds within +/-0.015). XGBoost accuracy 0.419. But expectancy=-$0.37/trade, PF=0.924. Gross edge $3.37/trade consumed by base costs $3.74/trade — 0.30 ticks short of breakeven. Profitable only under optimistic costs (+$0.88/trade at $2.49 RT). Hybrid beats GBT-nobook (+0.4pp acc, +$0.075 exp) and GBT-book (+0.9pp acc). Raw book features hurt XGBoost; CNN acts as denoiser.
**Lesson:** CNN spatial signal is real and reproducible (3rd independent reproduction). The regression-to-classification gap is the bottleneck — R²=0.089 on returns does not efficiently convert to viable 3-class classification through frozen embeddings. volatility_50 dominates XGBoost feature importance (19.9 gain). Model needs +2.0pp win rate (51.3% to 53.3%) for breakeven.
**Next:** (1) End-to-end CNN classification on tb_label (eliminate regression-to-classification gap), (2) XGBoost hyperparameter tuning, (3) Label design sensitivity (wider target to lower breakeven win rate), (4) CNN at h=1 with corrected normalization.
**Details:** `.kit/results/hybrid-model-corrected/analysis.md`

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
