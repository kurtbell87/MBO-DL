# Research Synthesis

**Generated:** 2026-02-26
**Trigger:** synthesis-v2 experiment
**Experiments analyzed:** 23
**Questions addressed:** 16 answered / 21 total (5 ranked open)

## Executive Summary

The MBO-DL research program has completed 23 phases (10 engineering, 13+ research) evaluating architecture components and optimization levers for MES microstructure prediction. **Updated verdict: GO** — with label geometry training as the primary justification.

The central finding is a **strategic pivot**: the project was optimizing the wrong variable. Oracle net expectancy ($/trade) was used as the viability gate, but **breakeven win rate vs. model accuracy** is the actual binding constraint. At 10:5 geometry, breakeven WR is 53.3% — 8.3pp above the model's 45% accuracy, guaranteeing losses. At 15:3, breakeven drops to 33.3% — 11.7pp BELOW model accuracy, with positive PnL projections down to 35% directional accuracy. This lever has never been tested with actual model training.

XGBoost hyperparameter tuning confirmed the feature set is the binding constraint (0.33pp accuracy range across 64 configs). Tuning brought CPCV expectancy to -$0.001/trade (knife-edge), but walk-forward shows -$0.140. The CNN line is permanently closed for classification (5.9pp worse than GBT across 5 experiments). GBT-only on 20 features is the canonical architecture.

**Single highest-priority next experiment:** Phase 1 label geometry training at 4 geometries (10:5, 15:3, 19:7, 20:3), 55-60% prior probability of success.

## Key Findings

### Finding 1: MES Returns Are Martingale at All Actionable Timescales
**Confidence:** Very High
**Evidence:** R4, R4b, R4c, R4d (0/168+ dual threshold passes across 7 bar types, 0.14s-300s)

MES 5-second bar returns contain zero exploitable temporal structure. No SSM, temporal encoder, or autoregressive model is justified.

### Finding 2: CNN Spatial Signal Does Not Transfer to Classification
**Confidence:** High
**Evidence:** R3, 9D, 9E, R3b-genuine, e2e-cnn-classification

CNN proper-validation R²=0.084 (3 independent reproductions). But end-to-end CNN classification is 5.9pp worse than GBT. Spatial signal encodes return variance, not class boundaries. CNN line permanently closed.

### Finding 3: Book Snapshot Is a Sufficient Statistic
**Confidence:** High
**Evidence:** R2 (0/40 dual threshold passes)

Order book snapshot at bar close fully summarizes all available information. Message encoder and temporal lookback dropped.

### Finding 4: XGBoost Hyperparameter Surface Is a Plateau
**Confidence:** High
**Evidence:** xgb-tuning (0.33pp range across 64 configs, std=0.0006)

Feature set is the binding constraint on accuracy, not hyperparameters. Default params near-optimal. Tuning brought expectancy from -$0.066 to -$0.001 (CPCV) via class-distribution shift (suppressed longs), not accuracy improvement.

### Finding 5: Breakeven WR, Not Oracle Ceiling, Is the Correct Viability Metric
**Confidence:** High
**Evidence:** label-design-sensitivity (123 geometries mapped)

Oracle net exp ranking inversely correlates with model viability in high-ratio region. Top-10 by oracle net exp have BEV WR 42-50% (model loses). Geometries ranked 60th-80th by oracle net exp have BEV WR 29-33% (model wins at 45% accuracy). The $5.00 oracle abort criterion was fundamentally miscalibrated.

### Finding 6: Oracle Confirms Exploitable Edge Exists
**Confidence:** High
**Evidence:** R7 oracle-expectancy

Oracle $4.00/trade, PF=3.30, WR=64.3% (19 days, 4,873 trades). Per-quarter stable: Q1=$5.39, Q2=$3.16, Q3=$3.41, Q4=$3.39. Models need to capture only ~1.6% of oracle edge to break even.

### Finding 7: Walk-Forward Divergence Is a Critical Limitation
**Confidence:** High
**Evidence:** xgb-tuning

CPCV expectancy -$0.001 vs walk-forward -$0.140 ($0.139 gap). CPCV's combinatorial splits allow training on future regime patterns. Walk-forward is deployment-realistic. The near-breakeven CPCV is an optimistic bound.

### Finding 8: Long/Short Recall Asymmetry Worsens Under Tuning
**Confidence:** High
**Evidence:** xgb-tuning, e2e-cnn

Long recall 0.201→0.149 under tuning. Short recall 0.586→0.634. Model increasingly short-biased. The "near-breakeven" CPCV is achieved partly by avoiding long trades.

## Negative Results

| Hypothesis | Verdict | Key Insight | Experiment |
|-----------|---------|-------------|------------|
| Event bars produce more Gaussian returns | REFUTED | Dollar bars 28x worse normality than time_1s | R1 |
| Temporal features improve prediction | REFUTED | 0/168+ passes, 7 bar types, 5s-300s | R4 chain |
| Message sequences add value | REFUTED | LSTM/Transformer worse than book MLP | R2 |
| CNN classification beats GBT | REFUTED | 5.9pp worse accuracy, -$0.069 expectancy | e2e-cnn |
| CNN+GBT hybrid viable | REFUTED | Exp -$0.37/trade (9E); GBT-only superior | 9E, e2e-cnn |
| XGB tuning closes accuracy gap | REFUTED | 0.33pp plateau; features are binding constraint | xgb-tuning |
| Oracle net exp gates viability | REFUTED | Inversely correlates with model viability at high ratios | label-sensitivity |
| Python vs C++ pipeline caused CNN failure | REFUTED | Byte-identical data; normalization was root cause | 9D, 9C |

## Closed Lines of Investigation (8)

1. **Temporal encoder** — R4, R4b, R4c, R4d: 0/168+ passes, 7 bar types, 0.14s-300s
2. **Message encoder** — R2: 0/40 passes, LSTM/Transformer worse
3. **Event-driven bars** — R1, R4b: 0/3 pairwise, volume_100 = time_5s
4. **CNN for classification** — e2e-cnn, 9E, 9B, R3b-genuine: 5.9pp worse, p=0.21
5. **CNN+GBT hybrid** — 9E, e2e-cnn: exp -$0.37, GBT superior
6. **XGB hyperparameter tuning** — xgb-tuning: 0.33pp plateau
7. **Oracle net exp as viability metric** — label-sensitivity: inverse correlation
8. **Python vs C++ pipeline hypothesis** — 9D, 9C: byte-identical, normalization root cause

## Open Questions (Ranked)

| Rank | Question | Prior | Compute |
|------|----------|-------|---------|
| 1 | Phase 1 label geometry training (accuracy transfer to high-ratio geometries?) | 55-60% | 2-4h local |
| 2 | Regime-conditional trading (Q1-Q2 only) | 40% | 1-2h local |
| 3 | 2-class short/not-short reformulation | 45% | 1-2h local |
| 4 | Cost reduction (limit orders, maker rebates) | 60% | Minimal |
| 5 | Class-weighted XGBoost (PnL-aligned loss) | 35% | 1-2h local |

## Architecture Recommendation

**GBT-only on 20 hand-crafted features** from book snapshot at bar close, time_5s bars. XGBoost with tuned params (LR=0.013, L2=6.6, depth=6). No temporal, message, or CNN components. CPCV for model selection, walk-forward for deployment estimates.

**Next lever:** Label geometry (breakeven WR reduction via target:stop ratio). This is the remaining high-leverage intervention before the project reaches a terminal decision.

## Statistical Limitations

1. **Walk-forward divergence**: CPCV -$0.001 vs WF -$0.140. WF is deployment-realistic.
2. **Abort criterion miscalibration**: $5.00 oracle gate measured wrong variable.
3. **Long/short asymmetry**: Long recall 0.149 may worsen at high-ratio geometries.
4. **Single-year data**: All results from 2022 MES. Regime effects may not generalize.
5. **volatility_50 monopoly**: 49.7% gain share. Model fragile to volatility regime shifts.
6. **Accuracy-transfer uncertainty**: Central unresolved question. Phase 1 is the test.

## Infrastructure Status

All prerequisites for Phase 1 label geometry training are COMPLETE:
- bar_feature_export with --target/--stop CLI flags
- oracle_expectancy with --target/--stop/--take-profit CLI flags
- Bidirectional triple barrier labels (152-column Parquet)
- Full-year dataset (251 days, 1.16M bars)
- Docker/ECR/EBS cloud pipeline (optional for GPU workloads)
- Parallel batch dispatch

## Recommendations

1. **Phase 1 label geometry training** — Highest priority. Train XGBoost at 4 geometries (10:5, 15:3, 19:7, 20:3). CPCV + walk-forward evaluation. 2-4 hours local. Resolves central uncertainty.
2. **Regime-conditional trading** — After Phase 1. Trade Q1-Q2 only or volatility-gated. Single-year limitation severe.
3. **2-class short/not-short** — If Phase 1 shows long recall collapse at high ratios.
4. **Walk-forward as primary metric** — For all future experiments. CPCV for model selection only.
5. **Do NOT revisit CNN** — Line permanently closed (5 experiments, high confidence).

## Appendix: Experiment Summary Table

| Experiment | Verdict | Key Metric | Value |
|-----------|---------|------------|-------|
| R1 subordination-test | REFUTED | Pairwise tests significant | 0/3 |
| R2 info-decomposition | FEATURES SUFFICIENT | Dual threshold passes | 0/40 |
| R3 book-encoder-bias | CONFIRMED | CNN R² (proper) | 0.084 |
| R4 temporal-predictability | REFUTED | AR configs positive R² | 0/36 |
| R4b temporal-event-bars | MARGINAL | Dual threshold passes | 0/48 |
| R4c temporal-completion | CONFIRMED (null) | Dual threshold passes | 0/54+ |
| R4d temporal-actionable | CONFIRMED (null) | Dual threshold passes | 0/38 |
| R6 synthesis-v1 | CONDITIONAL GO | Architecture | CNN+GBT Hybrid (revised) |
| R7 oracle-expectancy | GO | TB expectancy/trade | $4.00 |
| 9B hybrid-training | REFUTED | CNN R² | -0.002 |
| 9C cnn-diagnostic | REFUTED | MVE fold 5 train R² | 0.002 |
| 9D pipeline-comparison | Step1/Step2 | Pipeline identity rate | 1.0 |
| 9E hybrid-corrected | REFUTED (B) | Expectancy (base) | -$0.37 |
| R3b-original | INCONCLUSIVE | Bar defect | tick=time |
| R3b-genuine | CONFIRMED (low) | tick_100 R² | 0.124 (p=0.21) |
| Full-year export | CONFIRMED | Days/bars | 251/1.16M |
| E2E CNN classification | REFUTED (D) | GBT-CNN accuracy gap | +5.9pp |
| XGB hyperparameter tuning | REFUTED (C) | Accuracy range | 0.33pp |
| Label design sensitivity | REFUTED (C) | Oracle viable (>$5) | 0/123 |
| Bidir reexport | CONFIRMED | Files exported | 312/312 |
| TB-Fix | DONE | Tick bars fixed | counts trades |
| Oracle params CLI | DONE | CLI flags | --target/--stop |
| Geometry CLI | DONE | CLI flags | --target/--stop |

### Cumulative Statistics

| Statistic | Value |
|-----------|-------|
| Total experiments | 23 |
| Dual threshold passes (temporal) | 0/168+ |
| Bar types tested (temporal) | 7+ |
| Timescale range | 0.14s-300s |
| CNN reproductions (proper) | 3 |
| CNN proper R² | 0.084 |
| GBT accuracy (CPCV, tuned) | 0.4504 |
| GBT expectancy (CPCV, tuned) | -$0.001 |
| GBT expectancy (WF, tuned) | -$0.140 |
| Breakeven WR at 15:3 | 33.3% |
| Breakeven WR at 20:3 | 29.6% |
| Oracle expectancy (TB) | $4.00/trade |
| Oracle win rate (TB) | 64.3% |
| Closed lines | 8 |
| Open questions | 5 (ranked) |
| Total unit tests | 1144+ |
| GPU hours consumed | 0.0 |
