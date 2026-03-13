# Research Synthesis

**Generated:** 2026-02-27
**Trigger:** max_cycles (16 program cycles completed)
**Experiments analyzed:** 30 (27 research experiments + 2 synthesis reviews + 1 infrastructure validation)
**Questions addressed:** 26 answered / 31 total

## Executive Summary

The MBO-DL research program investigated optimal architecture and trading strategy for MES (Micro E-mini S&P 500) microstructure prediction using 312 days of MBO data from 2022. After 30 experimental phases spanning architecture search, feature engineering, label design, and execution optimization, the program converged on a **two-stage XGBoost pipeline at 19:7 triple barrier geometry** as the first and only statistically validated positive-expectancy configuration ($1.81/trade CPCV, p<1e-13, PBO=0.067). The edge is structural — driven entirely by payoff asymmetry (2.71:1 reward-to-risk ratio at 34.6% breakeven win rate) rather than directional prediction skill (accuracy ~50%). Eight lines of investigation were permanently closed: temporal encoding, message encoding, event-driven bars, CNN classification, CNN+GBT hybrid, XGBoost hyperparameter tuning, oracle net expectancy as viability metric, and the "Python vs. C++ pipeline" hypothesis. The central paradigm shift: the project evolved from "build a better predictor" to "find the right payoff structure for the predictor we have."

## Key Findings

### Finding 1: MES 5-second returns are a martingale — no exploitable temporal signal exists
**Confidence:** Very High
**Evidence:** R4, R4b, R4c, R4d (4 experiments, 0/168+ dual threshold passes)

MES returns at 5-second bars contain zero exploitable autoregressive structure at any timescale (0.14s–300s) or bar type (time, tick, dollar, volume). The R4 chain tested 7 bar types, 7+ horizons, 3 model classes (linear AR, Ridge, GBT), 5 feature configurations, and 200+ statistical tests. Zero passed the dual threshold (relative >20% of baseline R² AND corrected p<0.05). Dollar bars show marginal positive AR R² at sub-second horizons (~140ms), but this is non-actionable HFT microstructure, linear, and entirely redundant with static book features. Extended horizons (17–83 minutes) show accelerating degradation. The temporal encoder (SSM) is permanently dropped.

### Finding 2: Book snapshot is a sufficient statistic — additional information sources add nothing
**Confidence:** High
**Evidence:** R2 (0/40 dual threshold passes across 4 horizons × 5 information gaps × 2 model tiers)

The 40-dimensional order book snapshot at bar close captures all extractable predictive information. Hand-crafted message summaries (cancel/add ratios, message rate, toxicity) add nothing (gap negative at 3/4 horizons). Learned message encoders (LSTM, Transformer on truncated MBO sequences) perform *worse* than a plain book MLP. Temporal lookback (20-bar history) actively degrades performance due to overfitting (845-dim input with R²<0.007 signal). The null hypothesis — book is a sufficient statistic for intra-bar messages — cannot be rejected.

### Finding 3: CNN captures genuine spatial structure but cannot classify profitably
**Confidence:** High
**Evidence:** R3, 9B, 9C, 9D, 9E, R3b-genuine, e2e-cnn (7 experiments)

The Conv1d CNN on structured (20-level, 2-channel) book input achieves R²≈0.084 for return regression (proper validation; R3's original 0.132 included ~36% inflation from test-as-validation leakage). This spatial signal is real: three independent reproductions confirm it, the CNN is the only model that never produces negative R² across 5 folds, and the 16-dim embedding outperforms the 40-dim raw book on linear probe by 4.2x. However, this regression signal does NOT transfer to classification: end-to-end CNN achieves accuracy 0.390 vs GBT's 0.449 (5.9pp worse). The CNN penultimate layer learns return-variance features, not class boundaries. Long (+1) recall is only 0.21. The CNN+GBT hybrid produces expectancy of -$0.37/trade — the CNN acts as a denoiser but the improvement over GBT-only is too small to flip the sign. CNN classification line is permanently closed.

**Root cause of the R3→9B reproduction failure:** The data was byte-identical between R3 and the 9A C++ export (9D experiment confirmed identity rate=1.0). The failure was caused by missing TICK_SIZE normalization on prices (÷0.25) and per-fold z-scoring instead of per-day z-scoring on sizes. This consumed 4 experiments (9B, 9C, 9D, 9E) to diagnose and resolve.

### Finding 4: Event-driven bars offer no systematic advantage over time bars for MES
**Confidence:** High
**Evidence:** R1, R4b, R3b-original, R3b-genuine (4 experiments)

The Clark (1973) / Ané & Geman (2000) subordination model is refuted for MES microstructure. No event-bar configuration (volume, tick, dollar) significantly outperforms time bars on normality, homoskedasticity, or volatility clustering after correction for multiple comparisons. Dollar bars are catastrophically non-Gaussian (JB=109 million, 28x worse than time_1s) and their elevated AR R² is a sub-tick microstructure artifact. Volume bars show 9–10% daily count CV (genuine event bars) but no predictive advantage. Tick bars required a TDD fix (original bar_feature_export counted book snapshots, not trade events). After the fix, genuine tick bars at tick_100 showed R²=0.124 (vs 0.089 baseline), but p=0.21 — driven by a single anomalous fold. time_5s is the canonical bar type.

### Finding 5: XGBoost directional accuracy is a hard ceiling at ~45–50% — the binding constraint
**Confidence:** Very High
**Evidence:** xgb-tuning, label-geometry-1h, 2class-directional, cpcv-corrected-costs, threshold-sweep, class-weighted-stage1 (6+ experiments)

The 20-feature set, not the model or hyperparameters, determines XGBoost's directional accuracy. The hyperparameter landscape is a 0.33pp plateau across 64 configurations (std=0.0006). Accuracy is ~45% for 3-class long-perspective labels, ~50% for binary directional classification at 19:7 bidirectional labels — both invariant to geometry, label type, class weighting, or threshold optimization. volatility_50 dominates feature importance (49.7% gain share in the tuned model). The model's Stage 2 (direction prediction) at 19:7 shows no directional features in the top-10 — weighted_imbalance is absent. Direction prediction at wide barriers is fundamentally unpredictable with book-snapshot features.

### Finding 6: Payoff asymmetry, not prediction skill, enables profitability
**Confidence:** High
**Evidence:** label-design-sensitivity, synthesis-v2, 2class-directional, cpcv-corrected-costs (4 experiments)

The project's central paradigm shift: breakeven win rate (not oracle ceiling or accuracy improvement) is the correct viability metric. At 10:5 geometry, breakeven WR is 53.3% — above the model's ~45-50% accuracy, guaranteeing losses. At 19:7 geometry, breakeven WR drops to 34.6% — 15pp below observed accuracy, enabling positive expectancy despite coin-flip direction prediction. Oracle net expectancy ranking is inversely correlated with model viability in the high-ratio region: geometries where the oracle earns the most are where the model loses. The oracle margin (WR - breakeven WR) is remarkably stable at 10–12pp across all 123 tested geometries. The strategic pivot from "build a better predictor" to "find the right payoff structure" was the most important intellectual contribution of the research program.

### Finding 7: Two-stage XGBoost at 19:7 is the first validated positive-expectancy configuration
**Confidence:** High (for specific cost model)
**Evidence:** 2class-directional, pnl-realized-return, cpcv-corrected-costs (3 experiments forming a validation chain)

The two-stage formulation (Stage 1: barrier reachability filter; Stage 2: long/short direction) at 19:7 geometry achieves CPCV mean expectancy of $1.81/trade (95% CI [$1.46, $2.16]), PBO=0.067 (3/45 splits negative), t=10.29, p<1e-13 under corrected-base costs ($2.49 RT). Holdout expectancy: $1.46/trade. Break-even RT: $4.30. All 10 temporal groups are positive in expectation — no quarter is systematically unprofitable. The edge is structural: Stage 1 identifies barrier-reachable regimes using activity/volatility features (message_rate dominates at 717.7 gain), while Stage 2's direction prediction is a coin flip (50.16% pooled accuracy). Q3-Q4 is stronger ($2.18–$2.93) than Q1-Q2 ($1.39–$1.49) due to volatility-dependent barrier reachability.

**Critical dependencies:** Viable at base costs ($2.49 RT) and optimistic costs ($1.24 RT, $3.06/trade). NOT viable at pessimistic costs ($4.99 RT, -$0.69/trade, CI entirely below zero). Cost correction from $3.74 to $2.49 (AMP volume-tiered commissions) was the key unlock — the gross edge ($4.30/trade) was always present.

### Finding 8: Sequential execution is viable at medium accounts but not retail micro-accounts
**Confidence:** High
**Evidence:** trade-level-risk-metrics (1 experiment, 45 CPCV splits)

Sequential 1-contract execution produces $2.50/trade on 162 trades/day = $412.77/day = $103,605/year on 1 MES. Per-trade expectancy exceeds bar-level ($2.50 vs $1.81) because hold bars are skipped. However, minimum account is $48,000 (all-paths survival) or $26,600 (95%-paths) — 9.6x the $5K retail target. Three spec assumptions were invalidated by volatility-timing selection bias: avg hold 28 bars (not 75), hold-skip rate 66.1% (not 43%), trades/day 162 (not 50–80). Sequential entry clusters during volatile moments (shorter barrier races, higher per-trade expectancy) but exits into calm periods (higher hold-skip). Win rate 49.93% — the edge is pure 2.71:1 payoff asymmetry. Timeout trades dilute $5.00 barrier-hit theoretical to $2.50 observed (50% dilution).

### Finding 9: Timeout fraction (~41.3%) is a structural constant of the volume horizon
**Confidence:** High
**Evidence:** timeout-filtered-sequential, volume-flow-conditioned-entry (2 experiments)

Time-of-day entry filtering has zero effect on timeout fraction (invariant at 41.3% across 7 cutoff levels, total variation 0.21pp). Volume/activity-based gating shows massive 20pp bar-level diagnostic signal but collapses to 1.76pp maximum reduction in sequential simulation — 91% signal evaporation. The reason: sequential execution's 66.1% hold-skip rate already self-selects for high-activity bars, rendering explicit activity gating redundant. The cross-table reveals the diagnostic signal is concentrated in one corner of feature space (low volatility + low activity → 71.5% timeout); outside that corner, timeout fraction is structurally pinned. Entry-time filtering is exhausted as an intervention class for timeout reduction.

### Finding 10: Two-stage decomposition successfully decouples reachability from direction
**Confidence:** High
**Evidence:** 2class-directional, label-geometry-phase1, label-geometry-1h (3 experiments)

The 3-class XGBoost at 19:7 produced 0.28% trade rate — cross-entropy loss penalizes wrong-direction predictions on directional bars, causing the model to default to hold prediction (98.6% hold-hit rate). The two-stage formulation eliminates this trap: Stage 1 binary classification (directional-vs-hold) restores 85.2% trade rate (301x increase). Stage 1 accuracy (58.6%) exceeds majority baseline by 6pp, confirming reachability is learnable. The decomposition's value is structural (removing wrong-direction penalty) rather than representational (both stages learn the same volatility/activity features). At 10:5, Stage 2 does learn directional features (weighted_imbalance rank 6); at 19:7, it finds none — confirming direction-prediction difficulty scales with barrier width.

## Negative Results

These are findings about what does NOT work — each prevents future researchers from repeating failed approaches.

| Hypothesis | Verdict | Key Insight | Experiment(s) |
|-----------|---------|-------------|---------------|
| Event-driven bars superior to time bars (subordination theory) | REFUTED | Dollar bars catastrophically non-Gaussian (JB=109M). MES is a derivative — information arrives via ES arbitrage, not MES order flow | R1 |
| Message sequences add predictive value beyond book snapshot | REFUTED | Book snapshot is sufficient statistic. LSTM/Transformer worse than plain MLP. 0/40 gaps pass correction | R2 |
| Temporal structure in MES 5s returns (any bar type, any horizon) | REFUTED | Martingale. 0/168+ dual threshold passes across 7 bar types, 0.14s–300s timescale | R4, R4b, R4c, R4d |
| CNN spatial signal transfers to classification | REFUTED | CNN acc 0.390 vs GBT 0.449 (5.9pp worse). Regression R²=0.084 encodes variance, not class boundaries | e2e-cnn, 9E |
| CNN+GBT hybrid produces viable trading signals | REFUTED | Expectancy -$0.37/trade (base). CNN denoising adds too little to flip sign | 9E, e2e-cnn |
| XGBoost hyperparameter tuning improves accuracy by 2+pp | REFUTED | 0.33pp plateau across 64 configs (std=0.0006). Features are the binding constraint | xgb-tuning |
| Oracle net expectancy is the correct viability metric | REFUTED | Inversely correlated with model viability at high ratios. Breakeven WR is the correct metric | label-design-sensitivity |
| Bidirectional labels at wide geometries enable geometry exploitation | REFUTED | 91–99% hold rate at all wide geometries. Model degenerates to hold-predictor | label-geometry-phase1 |
| 1-hour time horizon fixes degenerate hold rate at all geometries | INCONCLUSIVE | Fixes 10:5 (90.7%→32.6%) but model refuses to trade at high ratios (0.003–0.28% dir pred rate) | label-geometry-1h |
| Stage 1 threshold optimization (0.5→0.9) improves 19:7 expectancy | REFUTED | XGBoost P(directional) compressed: 80.6% in [0.50, 0.60], IQR 4pp. Trade rate cliff at T=0.55→0.60 | threshold-sweep |
| Class-weighted XGBoost spreads probability distribution | REFUTED | scale_pos_weight < 1 is a logit-shift, not a spread. Recall collapses 465x, precision also drops | class-weighted-stage1 |
| Time-of-day filtering reduces timeouts | REFUTED | Timeout fraction invariant at 41.3% (range 0.21pp across 7 cutoffs). Driven by volume horizon, not clock time | timeout-filtered-sequential |
| Volume/activity gating reduces timeouts | REFUTED | 20pp diagnostic signal evaporates to 1.76pp in simulation. Hold-skip already self-selects for high-activity bars | volume-flow-conditioned-entry |
| Sequential execution viable at $5K retail accounts | REFUTED | Min account $48K (9.6x target). Volatility-timing selection bias → 162 trades/day, avg hold 28 bars | trade-level-risk-metrics |
| "Python vs C++ pipeline" explains CNN reproduction failure | REFUTED | Data is byte-identical. Root cause: TICK_SIZE normalization + per-day z-scoring omission | 9D, 9C |
| Protocol deviations (z-score, architecture, LR) explain CNN failure | REFUTED | Architecture matched exactly (12,128 params). All 5 normalization variants produce R²<0.002 | 9C |

## Open Questions

Prioritized by importance and feasibility. 5 questions remain unanswered.

1. **Long-perspective labels at 19:7** — Does the favorable payoff structure produce positive expectancy when using long-perspective (non-bidirectional) labels that avoid hold-dominance? Bidirectional labels create degenerate hold rates at wide geometries; long-perspective labels produce balanced classes. The accuracy-transfer question at 19:7 with balanced classes has never been tested. *Feasible: 2–4 hours local CPU. Highest remaining information value.*

2. **Multi-contract scaling** — Does N-contract staggered sequential execution produce N-proportional daily PnL with sqrt(N)-proportional drawdown? Sequential 1-contract captures only 5.7% of bar-level PnL. Concurrent mean = 35.9 contracts (~7 ES) is the theoretical ceiling. *Feasible: Analytical/simulation, local CPU. Important for deployment sizing.*

3. **Regime-conditional trading** — Does restricting to favorable regimes (Q1–Q2, or volatility-gated) produce positive expectancy on a per-regime basis? The 19:7 pipeline is positive in all quarters but with 2:1 dispersion (Q4 $2.93 vs Q2 $1.39). *Limitation: Single year of data — risk of overfitting regime definition to 2022.*

4. **Transaction cost sensitivity** — Can execution costs be reduced below $2.49 RT through limit orders or maker rebates? The break-even RT is $4.30, providing $1.81 margin. Limit-order execution ($1.24 RT) would nearly double the edge to $3.06/trade. *Feasible: Analytical, using existing MBO data for fill-rate estimation.*

5. **Message-sequence value in high-volatility regimes** — Does message information help specifically during high-volatility periods, even though the aggregate R2 gap was null? If positive, reopens message encoder line for regime-conditional use. *Lower priority: R2 tested 40 configurations comprehensively with no signal.*

## Infrastructure Needs

All engineering infrastructure prerequisites are complete. No outstanding infrastructure blocks further research.

| Infrastructure | Status | Notes |
|---------------|--------|-------|
| Full-year Parquet export (251 days, 1.16M bars) | COMPLETE | 149 columns, 255.7 MB zstd, production-ready |
| Bidirectional triple barrier labels | COMPLETE | 152 columns, `--legacy-labels` flag for backward compatibility |
| CLI geometry control (`--target`, `--stop`) | COMPLETE | bar_feature_export and oracle_expectancy |
| Time horizon control (`--max-time-horizon`, `--volume-horizon`) | COMPLETE | Defaults 3600s / 50000 contracts |
| Docker/ECR/EBS cloud pipeline | COMPLETE | E2E verified, c5.2xlarge ~10 min for full-year export |
| Tick bar fix (count trades, not snapshots) | COMPLETE | `bar_feature_export --bar-type tick` now genuine |
| S3 artifact store | COMPLETE | Content-addressed deduplication for large result files |
| 1,144+ unit tests, 22 integration tests | COMPLETE | All TDD phases exit 0 |

**For deployment (not yet built):**
- R|API+ (Rithmic) integration for paper/live trading — installed but not wired into code
- Real-time bar construction and feature computation pipeline
- Position management and risk monitoring system

## Recommendations

1. **Test long-perspective labels at 19:7 geometry** — This is the single highest remaining information-value experiment. Bidirectional labels create degenerate hold distributions at wide geometries; long-perspective labels produce balanced classes. If directional accuracy at 19:7 is ≥38.4% (breakeven WR) on long-perspective labels, the payoff asymmetry guarantees positive expectancy. Estimated compute: 2–4 hours local CPU. All infrastructure is in place (`--legacy-labels --target 19 --stop 7 --max-time-horizon 3600`).

2. **Accept cutoff=270 as the production configuration and proceed to paper trading** — The two-stage 19:7 pipeline with cutoff=270 produces $3.02/trade, $337/day, $34K minimum account, Calmar 2.59. Further entry-time optimization has been exhausted (timeout fraction is structural). Paper trading with Rithmic R|API+ on 1 /MES validates real-world fill rates, slippage, and latency that the simulation cannot capture.

3. **Reduce barrier horizons to address timeout dilution** — The 41.3% timeout fraction is the largest single drag ($5.00 barrier-hit theoretical → $2.50 observed). The horizon parameters (3600s time, 50000 volume) were set conservatively. Reducing to 600–1200s time or 10000–25000 volume would reduce timeout fraction at the cost of more hold-labeled bars. This is a geometry re-parameterization, distinct from entry-time filtering (which is exhausted).

4. **Run multi-year validation before live deployment** — All 30 experiments use 2022 MES data exclusively. The Q3-Q4 volatility regime (Fed rate hikes, equity drawdown) may inflate barrier reachability metrics. The edge may not persist in 2023+ if volatility regimes differ. This is the single largest threat to external validity. Required data: 2023 and/or 2024 MES MBO.

5. **Investigate hold-bar exit management** — Hold-bar PnL swings from -$9.39 to +$3.48 per CPCV split, driving the majority of per-split variance. Hold bars have unbounded forward returns (±63 ticks at p10/p90) due to the volume horizon. Active hold-bar management (e.g., tighter time-based exits) could reduce variance while preserving mean expectancy.

## Methodology Notes

**What worked well:**

- **Dual threshold policy** (relative >20% AND corrected p<0.05) prevented premature commitment to weak signals across 200+ statistical tests.
- **CPCV with 45 splits** provided robust confidence intervals and PBO estimates. Walk-forward (3-fold) was useful for initial screening but insufficient for final validation.
- **Pre-committed abort criteria** saved compute by halting unpromising experiments early (9C MVE gate, label-design-sensitivity Phase 0 abort). However, the label-design-sensitivity abort was miscalibrated — it used oracle net expectancy instead of breakeven WR, preventing Phase 1 training that would have been informative.
- **Sequential experimental chain** (2class-directional → pnl-realized-return → threshold-sweep → class-weighted-stage1 → cpcv-corrected-costs → trade-level-risk-metrics → timeout-filtered-sequential → volume-flow-conditioned-entry) efficiently explored the two-stage pipeline parameter space in 8 cycles.
- **Institutional memory** (RESEARCH_LOG.md, QUESTIONS.md) prevented repeating failed approaches and maintained intellectual coherence across 30+ experiments.

**What should be done differently:**

- **Cost model should have been corrected earlier.** The $3.74 base cost assumption masked viability for multiple experiments (9E, e2e-cnn, xgb-tuning). The $2.49 corrected cost (AMP volume-tiered) was the key unlock — had it been used from the start, the path to the 19:7 configuration would have been shorter.
- **Walk-forward vs CPCV divergence should be tracked from the start.** The $0.139 gap discovered in xgb-tuning is a critical limitation on all deployment estimates. Every experiment should report both metrics.
- **CNN debugging consumed 4 experiments (9B→9C→9D→9E).** The root cause (normalization) could have been found faster with systematic ablation of data loading steps rather than end-to-end reproduction attempts.
- **The label-design-sensitivity $5.00 abort criterion was the wrong metric.** It measured oracle ceiling instead of model viability. This delayed the geometry exploration by one full cycle.
- **Hold-bar PnL modeling should be specified precisely in every experiment spec.** The 2class-directional experiment's ~8x PnL inflation (discovered in pnl-realized-return) could have been avoided with explicit hold-bar treatment in the experiment spec.

## Appendix: Experiment Summary Table

| Experiment | Question | Verdict | Key Metric | Value |
|-----------|----------|---------|------------|-------|
| R1 subordination-test | Event bars better than time bars? | REFUTED | Pairwise significance (primary metrics) | 0/9 |
| R2 info-decomposition | Info decomposes across spatial/message/temporal? | REFUTED (null) | Dual threshold passes | 0/40 |
| R3 book-encoder-bias | CNN outperforms flattened features? | CONFIRMED | CNN mean OOS R² | 0.132 (leaked), 0.084 (proper) |
| R4 temporal-predictability | Temporal structure in MES 5s returns? | REFUTED | AR R² (best config) | -0.0002 |
| R4b temporal-event-bars | Temporal structure in event bars? | REFUTED | Dual threshold passes | 0/48 |
| R4c temporal-completion | Temporal at extended horizons/tick bars? | REFUTED | Dual threshold passes | 0/54+ |
| R4d temporal-actionable | Temporal at actionable timescales (7s-300s)? | REFUTED | Dual threshold passes | 0/38 |
| R6 synthesis | Architecture recommendation? | CONFIRMED | Recommended architecture | CNN+GBT Hybrid (later revised) |
| R7 oracle-expectancy | Oracle positive after costs? | CONFIRMED (GO) | Oracle net exp/trade | $4.00 |
| 9B hybrid-model-training | CNN+GBT hybrid viable? | REFUTED | CNN R² (h=5) | -0.002 |
| 9C cnn-reproduction | Protocol deviations explain CNN failure? | REFUTED | Train R² (fold 5) | 0.002 |
| 9D pipeline-comparison | C++ export differs from R3's data? | Step 1 CONFIRMED, Step 2 REFUTED | Data identity rate | 1.0 (byte-identical) |
| 9E hybrid-corrected | Corrected CNN+GBT viable? | REFUTED (Outcome B) | Expectancy/trade (base) | -$0.37 |
| R3b-original event-bar-cnn | CNN improves on tick bars? | INCONCLUSIVE (voided) | Tick bar CV | 0.000 (broken bars) |
| R3b-genuine tick-bars | CNN on genuine tick bars? | CONFIRMED (low conf.) | tick_100 mean OOS R² | 0.124 (p=0.21) |
| full-year-export | Full-year export production-ready? | CONFIRMED | Days exported / days total | 251/251 |
| e2e-cnn-classification | E2E CNN classification viable? | REFUTED (Outcome D) | CNN acc vs GBT acc | 0.390 vs 0.449 |
| xgb-hyperparam-tuning | Tuning improves accuracy by 2+pp? | REFUTED (Outcome C) | Accuracy range across 64 configs | 0.33pp |
| label-design-sensitivity | Oracle > $5.00 at any geometry? | REFUTED (miscalibrated criterion) | Peak oracle net exp (base) | $4.126 |
| synthesis-v2 | Comprehensive synthesis — GO/NO-GO? | CONFIRMED (GO) | Prior probability for geometry | 55-60% |
| label-geometry-phase1 | Bidirectional labels viable at wide geometry? | REFUTED | Hold rate at 15:3 / 19:7 / 20:3 | 97.1% / 98.6% / 98.9% |
| label-geometry-1h | 1-hour horizon fixes hold degeneration? | INCONCLUSIVE | Dir pred rate at 19:7 | 0.28% |
| 2class-directional | 2-stage decouples reachability and direction? | CONFIRMED (PnL caveat) | Trade rate at 19:7 | 85.2% (301x increase) |
| pnl-realized-return | Corrected PnL still positive at 19:7? | CONFIRMED (SC-2 fails) | Realized exp/trade | $0.90 |
| threshold-sweep | Threshold optimization improves 19:7 exp? | REFUTED | P(dir) IQR | 4pp (degenerate) |
| class-weighted-stage1 | Class weighting spreads probabilities? | REFUTED | IQR at w=0.50 | 3.2pp (compressed) |
| cpcv-corrected-costs | CPCV validates 19:7 at corrected costs? | CONFIRMED (Outcome A) | CPCV mean exp/trade | $1.81, p<1e-13 |
| trade-level-risk-metrics | Sequential viable at $5K accounts? | REFUTED (productive) | Min account / seq exp | $48K / $2.50/trade |
| timeout-filtered-sequential | Time-of-day reduces timeouts? | REFUTED | Timeout fraction variation | 0.21pp (invariant) |
| volume-flow-conditioned-entry | Volume gating reduces timeouts? | REFUTED | Max timeout reduction | 1.76pp (91% evaporation) |

### Cumulative Statistics

| Statistic | Value |
|-----------|-------|
| Total experiments | 30 |
| CONFIRMED | 8 (R3, R6, R7, full-year-export, synthesis-v2, 2class, pnl-realized, cpcv-corrected) |
| REFUTED | 17 |
| INCONCLUSIVE | 3 (R3b-original, label-geometry-1h, label-geometry-phase1 implicitly) |
| CONFIRMED low confidence | 1 (R3b-genuine) |
| Outcome A (strong positive) | 1 (cpcv-corrected-costs) |
| Lines permanently closed | 8 |
| Questions answered | 26 / 31 |
| Questions remaining | 5 |
| Research cycles (program) | 16 |
| GPU hours consumed | 0.0 |
| Engineering TDD phases | 14 (all exit 0) |
| Unit tests | 1,144+ |
| Integration tests | 22 |
| Data: days processed | 251 RTH trading days (2022 MES MBO) |
| Data: total bars | 1,160,150 |
| Data: Parquet columns | 152 (bidirectional) |
| Best validated expectancy (CPCV) | $1.81/trade (95% CI [$1.46, $2.16]) |
| Best validated expectancy (sequential) | $2.50/trade |
| Best validated annual PnL | $103,605 (1 MES sequential) |
| Minimum viable account | $48,000 (all-paths) / $26,600 (95%-paths) |
| Break-even RT cost | $4.30 |
| Directional accuracy (pooled) | 50.16% (coin flip) |
| Edge source | Payoff asymmetry (2.71:1 at 19:7), not prediction skill |
