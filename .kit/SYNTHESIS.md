# Research Synthesis

**Generated:** 2026-02-24
**Trigger:** manual
**Experiments analyzed:** 17
**Questions addressed:** 12 answered / 16 total (4 not started)

## Executive Summary

The MBO-DL research program has completed 27+ phases (10 engineering, 17 research) evaluating architecture components for MES microstructure prediction. The core finding is that **MES 5-second bar returns are a martingale difference sequence** -- no temporal encoder, message encoder, or event-driven bar construction provides exploitable signal. CNN spatial encoding on structured (20,2) book input produces genuine predictive signal for return regression (R2=0.084 with proper validation), but this signal **does not transfer to viable 3-class classification**: end-to-end CNN accuracy is 5.9pp worse than GBT on hand-crafted features. The CNN line is permanently closed for classification. GBT-only on full-year CPCV (1.16M bars, 251 days) achieves accuracy 0.449 and expectancy -$0.064/trade under base costs -- only $0.06/trade from breakeven -- with **default, never-optimized hyperparameters**. The two highest-priority next steps are XGBoost hyperparameter tuning and label design sensitivity, both of which could independently close the viability gap.

## Key Findings

### Finding 1: MES Returns Are Martingale at All Actionable Timescales
**Confidence:** Very High
**Evidence:** R4, R4b, R4c, R4d (0/168+ dual threshold passes across 7 bar types, 0.14s-300s)

MES 5-second bar returns contain zero exploitable temporal structure. The R4 experimental chain is the most exhaustive test in the program: 36 Tier 1 AR configurations (all negative R2), 16+ Tier 2 augmentation gaps (all fail dual threshold), and Temporal-Only R2 indistinguishable from zero. This result generalizes across time bars, volume bars, dollar bars, and tick bars at timescales from 5 seconds to 83 minutes. The only temporal signal detected was on dollar_25k bars at ~140ms (sub-second HFT microstructure, non-actionable for retail). All temporal features -- lagged returns, rolling volatility, momentum, mean reversion -- receive 30-50% of GBT importance in-sample but produce zero marginal R2 out-of-sample, a textbook overfitting signature. No SSM, temporal encoder, or autoregressive model is justified.

### Finding 2: CNN Spatial Signal Is Real but Does Not Transfer to Classification
**Confidence:** High
**Evidence:** R3, 9D, 9E, R3b-genuine, e2e-cnn-classification

Conv1d on structured (20,2) book input captures genuine spatial structure: R2=0.084 (proper validation, 3 independent reproductions). The CNN 16-dim embedding retains 4.16x more signal than raw 40-dim flattened book (linear probe R2=0.111 vs 0.027, p=0.012). CNN significantly outperforms Attention (corrected p=0.042, d=1.86). However, this regression signal does not encode class-discriminative book patterns. End-to-end CNN classification on 3-class triple barrier labels (full-year CPCV, 45 splits) achieves accuracy 0.390 -- **5.9pp worse than GBT-only** (0.449). CNN expectancy -$0.146/trade vs GBT -$0.064. The CNN's 16-dim penultimate layer learns return-variance features, not class boundaries. Long recall is only 0.21, vs short recall 0.45 -- the model is asymmetrically confident on shorts. The CNN line is permanently closed for classification.

### Finding 3: Book Snapshot Is a Sufficient Statistic
**Confidence:** High
**Evidence:** R2 (0/40 dual threshold passes)

The order book snapshot at bar close fully summarizes all information available from intra-bar message sequences, temporal history, and hand-crafted feature engineering. R2 tested 6 configurations across 4 horizons: raw book MLP (best R2=0.0067 at h=1), hand-crafted features, message summaries, LSTM/Transformer on raw MBO events. No encoder stage passes the dual threshold. Message summaries hurt (delta=-0.001 to -0.0024). LSTM and Transformer learned encoders perform worse than the plain book MLP. Temporal lookback actively degrades performance (delta=-0.006 at h=1). The message encoder and temporal encoder are dropped with highest confidence.

### Finding 4: GBT on Hand-Crafted Features Is 0.30 Ticks from Breakeven
**Confidence:** High
**Evidence:** e2e-cnn-classification (full-year CPCV), hybrid-model-corrected (9E)

GBT-only on 20 hand-crafted features achieves full-year CPCV accuracy 0.449, PF 0.986, expectancy -$0.064/trade. Breakeven requires 53.3% win rate; the model achieves ~51.3%. The gap is 2.0pp or approximately $0.06/trade -- less than 0.05 ticks. This result uses **default hyperparameters inherited from 9B (the broken pipeline era), never optimized**. GBT is marginally profitable in Q1 (+$0.003) and Q2 (+$0.029) under base costs. The edge exists seasonally but is consumed by Q3-Q4 losses. Under optimistic costs ($2.49 RT), the model is profitable across all configurations (+$1.19/trade, PF 1.29). volatility_50 dominates feature importance (19.9 gain, 2x the next feature). Walk-forward validation agrees with CPCV on accuracy (0.456 vs 0.449) but shows worse expectancy (-$0.267), suggesting CPCV mildly overestimates expectancy.

### Finding 5: Oracle Confirms Exploitable Edge Exists
**Confidence:** High
**Evidence:** R7 oracle-expectancy

The oracle (perfect foresight) on triple barrier labels produces $4.00/trade expectancy, PF=3.30, WR=64.3%, Sharpe=0.362 across 19 days (4,873 trades). All 6 success criteria pass. Per-quarter stability: Q1=$5.39, Q2=$3.16, Q3=$3.41, Q4=$3.39 -- the edge does not concentrate in a single regime. Triple barrier labeling is strictly superior to first-to-hit (FTH: $1.56/trade, PF=2.11, WR=53.2%). The oracle gap -- from $4.00/trade (oracle) to -$0.064/trade (GBT) -- defines the opportunity. Models need to capture only ~1.6% of the oracle edge to break even.

### Finding 6: Subordination Theory Is a Poor Fit for MES
**Confidence:** High
**Evidence:** R1 subordination-test

The Clark/Ane-Geman subordination model (event bars produce more Gaussian returns) is refuted for MES. 0/3 primary pairwise tests significant after correction. Dollar bars have catastrophically non-Gaussian returns (JB=109M, 28x worse than time_1s) despite lowest volatility clustering ACF. Dollar bars' high AR R2 (0.011, 5.2x time_1s) is opposite the theory's prediction and reflects sub-tick microstructure aliasing, not signal. Quarter robustness fails across all four quarters. time_5s is the recommended bar type: simplest, no threshold tuning, consistent performance.

### Finding 7: CNN Reproduction Required TICK_SIZE Normalization
**Confidence:** Very High
**Evidence:** 9C (cnn-reproduction-diagnostic), 9D (r3-reproduction-pipeline-comparison), 9E (hybrid-model-corrected)

The 9B/9C CNN pipeline failures (R2=-0.002 vs R3's 0.132) were caused by missing TICK_SIZE division on book prices (divide by 0.25 to convert index points to tick offsets) and per-fold z-scoring instead of per-day z-scoring on sizes. The "Python vs C++ pipeline" hypothesis was wrong -- R3 loaded from the same C++ export (byte-identical data, identity rate=1.0). R3's reported R2=0.132 includes ~36% inflation from test-as-validation leakage; proper-validation R2=0.084. This proper value was independently confirmed 3 times (9D, 9E, R3b-genuine). The lesson: always specify normalization as concrete operations in experiment specs.

### Finding 8: Genuine Tick Bars Show Promising but Fragile Signal
**Confidence:** Low
**Evidence:** R3b-genuine-tick-bars

Genuine trade-event tick bars at tick_100 (~5.7s median, ~4,171 bars/day) produce CNN R2=0.124 -- a 39% relative improvement over time_5s (0.089). However: paired t-test p=0.21 (not significant), the result depends entirely on fold 5's anomalous R2=0.259 (excluding fold 5, mean=0.091, comparable to baseline), only 1 of 3 tested thresholds crosses the criterion, and tick_25 and tick_500 are both worse (inverted-U curve). Data volume confound ruled out (r=-0.149). Not actionable without multi-seed replication.

## Negative Results

| Hypothesis | Verdict | Key Insight | Experiment |
|-----------|---------|-------------|------------|
| Event bars produce more Gaussian returns (subordination) | REFUTED | Dollar bars have 28x worse normality than time_1s; theory direction reversed | R1 |
| Temporal features improve prediction at any horizon | REFUTED | 0/168+ dual threshold passes across 7 bar types, 5s-300s | R4, R4b, R4c, R4d |
| Message sequences add value beyond book snapshot | REFUTED | LSTM and Transformer both worse than plain book MLP; all gaps negative | R2 |
| CNN end-to-end classification beats GBT | REFUTED (Outcome D) | CNN 5.9pp worse on accuracy, -$0.069 on expectancy; spatial signal does not encode class boundaries | e2e-cnn-classification |
| CNN+GBT hybrid economically viable under base costs | REFUTED (Outcome B) | Expectancy -$0.37/trade (9E, 19 days); improved to -$0.064 on full year but still negative | hybrid-model-corrected, e2e-cnn-classification |
| R3's R2=0.132 reproduces in new pipeline | REFUTED (initially) | Root cause: normalization, not data pipeline; proper R2=0.084 after leakage correction | 9B, 9C, 9D |
| Three CNN protocol deviations caused 9B failure | REFUTED | Z-scoring was not "FATAL"; data is byte-identical; normalization was root cause | 9C (cnn-reproduction-diagnostic) |
| Tick bars from C++ export are genuine event bars | REFUTED (bar defect) | bar_feature_export counted fixed-rate snapshots, not trades; all tick bar results void | R3b-event-bar-cnn |

## Open Questions

1. **Can XGBoost hyperparameter tuning improve accuracy by 2+pp?** (P1, Not started) -- Default params from 9B era, never optimized. GBT already shows Q1-Q2 positive expectancy. Grid search over max_depth, learning_rate, n_estimators, subsample, colsample, min_child_weight could close the $0.06/trade gap. Spec exists: `.kit/experiments/xgb-hyperparam-tuning.md`. Local compute, ~2-3 hours.

2. **Can label design (wider target/narrower stop) lower breakeven win rate below current ~45%?** (P1, Not started) -- At 15:3 ratio, breakeven drops from 53.3% to ~42.5%, well below current 51.3%. Orthogonal to model architecture. Requires re-export from C++ oracle_expectancy tool. Spec exists: `.kit/experiments/label-design-sensitivity.md`.

3. **How sensitive is oracle expectancy to transaction cost assumptions?** (P2, Not started) -- Determines edge robustness under spread={1,2,3} ticks and commission variation. The model flips between profitable and unprofitable with just 1 tick of cost difference ($2.49 vs $3.74 RT).

4. **Does regime-conditional trading (Q1-Q2 only) produce positive expectancy?** (P3, Not started) -- GBT profitable in H1 2022, negative in H2. Limited by single year of data -- cannot validate regime prediction without 2023+ data. No spec yet.

## Infrastructure Needs

1. **No blocking infrastructure needs.** The full-year Parquet dataset (1.16M bars, 251 days, 255.7 MB) is production-ready. The C++ export pipeline runs in 77 seconds (11-way parallel). Parallel batch dispatch is shipped. Docker/ECR/EBS cloud pipeline is verified E2E.

2. **Tick bar construction is fixed** (TB-Fix TDD cycle, 2026-02-19). bar_feature_export --bar-type tick now counts action='T' trade events. All prior "tick bar" experiments (R1, R4c, R4d, original R3b) were actually time bars at different frequencies -- those results are void for tick bars specifically. Dollar and volume bars were always genuine.

3. **GPU utilization gap.** The e2e-cnn-classification experiment ran CNN on CPU (~180-320s/split on g5.xlarge) despite having an A10G GPU available. The bootstrap script installed PyTorch CPU variant. Future CNN experiments should use CUDA-enabled PyTorch for 5-10x speedup. This is non-blocking since the CNN line is closed.

## Recommendations

1. **XGBoost hyperparameter tuning on full-year CPCV** -- The single highest-priority action. Default params were never optimized. The 2pp win rate gap (51.3% to 53.3%) is small enough that tuning over max_depth, learning_rate, n_estimators, subsample, colsample_bytree, min_child_weight could close it. Expected compute: local CPU, ~2-3 hours on Apple Silicon. Use the full-year CPCV framework (45 splits, 10-group, k=2) for robust evaluation. If tuning improves accuracy by 2pp, expectancy flips positive.

2. **Label design sensitivity in parallel with #1** -- Test alternative triple barrier parameters: {target: 15, stop: 3}, {target: 12, stop: 4}, {target: 8, stop: 8}. At 15:3 ratio, breakeven win rate drops to ~42.5% (vs current ~45% model performance). This is orthogonal to model tuning and can run concurrently. Requires re-export with modified oracle parameters. Expected compute: local CPU, ~1-2 hours per configuration.

3. **Walk-forward as primary evaluation metric** -- CPCV expectancy (-$0.064) is mildly optimistic vs walk-forward (-$0.267). For deployment-realistic estimates, report walk-forward expectancy as the primary metric. CPCV accuracy (0.449) and walk-forward accuracy (0.456) agree closely, so CPCV remains valid for model selection -- just not for expectancy estimation.

4. **Regime-conditional trading exploration (after #1 and #2)** -- If tuning and label redesign each provide partial improvement, a regime-conditional strategy (trade only in favorable regimes) could combine them. GBT is profitable in Q1 (+$0.003) and Q2 (+$0.029) with default params. Caution: single year of data severely limits regime analysis. Cannot validate without 2023+ data.

5. **Do NOT revisit CNN for classification.** The evidence is conclusive across three independent approaches: (a) 9E regression-to-embedding-to-XGBoost, (b) end-to-end cross-entropy, (c) 9B's broken pipeline all show CNN adding zero or negative value for trading decisions. The spatial signal is real for regression but irrelevant for the classification task.

## Methodology Notes

### What Worked Well

- **Dual threshold policy** (relative >20% of baseline AND Holm-Bonferroni corrected p<0.05) prevented false positives across 200+ statistical tests. Zero spurious discoveries in the temporal chain.
- **Expanding-window time-series CV** with day boundaries prevented temporal leakage. The 5-fold design was consistently applied across all experiments.
- **CPCV with PBO and DSR** (Combinatorially Purged Cross-Validation) on full-year data provided much tighter confidence bounds than 5-fold on 19 days. PBO=0.222 confirmed model selection was not overfit.
- **MVE gates** (Minimum Viable Experiment) saved substantial compute by aborting early when prerequisites failed (9C diagnostic saved ~70% of budget).
- **Ablation design** in 9E (Hybrid vs GBT-book vs GBT-nobook) cleanly isolated CNN contribution.
- **Research Log as institutional memory** -- each experiment builds on predecessors with clear provenance.

### What Should Be Done Differently

- **Normalization must be specified as concrete operations**, not prose descriptions. "Normalize to ticks" caused three failed reproduction attempts (9B, 9C, 9C diagnostic). The fix ("divide channel 0 by TICK_SIZE=0.25; apply log1p then per-day z-scoring to channel 1") should have been in the original spec.
- **Always verify bar construction semantics before running experiments.** The tick bar defect (counting snapshots, not trades) wasted one full experimental cycle (original R3b) and rendered R1/R4c/R4d tick bar results void. A simple diagnostic (check bars_per_day_std > 0) would have caught this immediately.
- **Test-as-validation leakage inflated R3's reported R2 by 36%.** All future CNN experiments must use proper 80/20 train/validation splits for early stopping. R3's R2=0.132 was widely cited in architecture decisions; the true value is 0.084.
- **Single-year data limits regime analysis.** R1 showed quarter-level reversals across all 4 quarters. GBT's Q1-Q2 profitability may not generalize. Multi-year data is essential before any deployment decision.
- **Cloud GPU utilization should be verified.** The e2e-cnn experiment ran on CPU despite having a GPU available. A preflight check for torch.cuda.is_available() would prevent this waste.

## Appendix: Experiment Summary Table

| Experiment | Question | Verdict | Key Metric | Value |
|-----------|----------|---------|------------|-------|
| R1 (subordination-test) | Are event bars superior to time bars? | REFUTED | Primary pairwise tests significant | 0/3 |
| R2 (info-decomposition) | Does information decompose across spatial/message/temporal? | FEATURES SUFFICIENT | Dual threshold passes | 0/40 |
| R3 (book-encoder-bias) | Does CNN on structured book outperform flattened? | CONFIRMED | CNN mean OOS R2 (h=5) | 0.132 (leaked) / 0.084 (proper) |
| R4 (temporal-predictability) | Do MES 5s returns have temporal structure? | REFUTED | AR configs with positive R2 | 0/36 |
| R4b (event-bar temporal) | Do event bars have temporal structure? | MARGINAL (redundant) | Dual threshold passes | 0/48 |
| R4c (temporal completion) | Temporal signal at tick bars or extended horizons? | CONFIRMED (null) | Dual threshold passes | 0/54+ |
| R4d (actionable dollar/tick) | Temporal signal at actionable timescales? | CONFIRMED (null) | Dual threshold passes | 0/38 |
| R6 (synthesis) | Go/no-go and architecture recommendation? | CONDITIONAL GO | Architecture | CNN+GBT Hybrid |
| R7 (oracle-expectancy) | Oracle profitable after costs? | GO | TB expectancy/trade | $4.00 |
| 9B (hybrid-model-training) | CNN+GBT hybrid reproduces R3? | REFUTED | CNN R2 (h=5) | -0.002 |
| 9C (cnn-reproduction-diagnostic) | Fixing 3 deviations restores CNN? | REFUTED | MVE fold 5 train R2 | 0.002 |
| 9D (r3-reproduction-pipeline) | Data pipeline causes CNN failure? | Step 1 CONFIRMED / Step 2 REFUTED | Pipeline identity rate | 1.0 (byte-identical) |
| R3b-original (event-bar-cnn) | CNN on tick bars improves R2? | INCONCLUSIVE | Bar construction defect | tick bars = time bars |
| 9E (hybrid-model-corrected) | Corrected CNN+GBT viable? | REFUTED (Outcome B) | Expectancy (base) | -$0.37/trade |
| R3b-genuine (genuine tick bars) | CNN on real tick bars? | CONFIRMED (low confidence) | tick_100 mean R2 | 0.124 (p=0.21) |
| Full-year export | 251-day dataset production-ready? | CONFIRMED | Days exported | 251/251, 1.16M bars |
| E2E CNN classification | CNN classification beats GBT? | REFUTED (Outcome D) | GBT accuracy - CNN accuracy | +5.9pp |

### Cumulative Statistics

| Statistic | Value |
|-----------|-------|
| Total dual threshold tests (temporal chain) | 168+ |
| Dual threshold passes (temporal chain) | 0 |
| Bar types tested (temporal) | 7 (time_5s, volume_100, dollar_25k, tick_50, tick_100, tick_250, dollar_5M, dollar_10M, dollar_50M, tick_500, tick_3000) |
| Timescale range tested | 0.14s - 300s |
| CNN reproductions (proper validation) | 3 (9D, 9E, R3b-genuine) |
| CNN proper-validation R2 | 0.084 +/- 0.07 |
| Full-year GBT accuracy (CPCV) | 0.449 |
| Full-year GBT expectancy (base) | -$0.064/trade |
| GBT breakeven cost (estimated) | ~$3.68 RT |
| Oracle expectancy (TB) | $4.00/trade |
| Oracle win rate (TB) | 64.3% |
| GPU hours consumed | 0.0 |
| Total engineering phases | 10 |
| Total research phases | 17 |
| Total unit tests | 1003/1004 pass (1 disabled, 1 skipped) |
