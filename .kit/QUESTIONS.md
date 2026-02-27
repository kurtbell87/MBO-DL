# Research Questions

The research agenda for this project. Questions are organized by priority and status.

---

## 1. Goal

Determine the optimal architecture for MES microstructure prediction: bar type, feature set, encoder stages (spatial/temporal/message), and labeling method.

**Success looks like:** A final architecture decision backed by rigorous experimental evidence across all candidate components. Each encoder stage either justified by dual-threshold evidence or permanently dropped.

---

## 2. Constraints

| Constraint | Decision |
|------------|----------|
| Framework  | C++20 (pipeline + models), Python (research analysis + CNN/XGBoost training) |
| Compute    | Local for CPU-only experiments (<1GB data). RunPod for GPU. EC2 spot only for large data (>10GB). |
| Timeline   | R1–R4d complete → 9A–9E complete → E2E CNN REFUTED → XGBoost tuning + label design next |
| Baselines  | GBT-only on full-year data (accuracy 0.449, expectancy -$0.064 base, Q1-Q2 marginally positive) |

---

## 3. Non-Goals (This Phase)

- Live/paper trading integration (R|API+ installed but not integrated)
- Multi-instrument generalization (MES only)
- Multi-year validation (2022 data only; regime-conditional conclusions limited to 1 year)

---

## 4. Open Questions

Status: `Not started` | `In progress` | `Answered` | `Blocked` | `Deferred`

| Priority | Question | Status | Parent | Blocker | Decision Gate | Experiment(s) |
|----------|----------|--------|--------|---------|---------------|---------------|
| P1 | Can XGBoost hyperparameter tuning improve classification accuracy by 2+pp? Default params never optimized. GBT shows Q1-Q2 positive expectancy (+$0.003, +$0.029 base). | Answered | — | — | Determines if the 2pp gap to breakeven is a tuning issue vs architectural limit | xgb-hyperparam-tuning |
| P1 | Can label design (wider target 15t / narrower stop 3t) lower breakeven win rate below current ~45%? | Answered | — | — | Breakeven drops from 53.3% to ~42.5% at 15:3 ratio | label-design-sensitivity |
| P0 | Does XGBoost achieve directional accuracy > breakeven WR + 2pp at high-ratio geometries (15:3, 19:7, 20:3)? | Answered | — | — | Determines if favorable payoff structure enables positive expectancy. Oracle ceiling mapped (peak $4.13); breakeven WRs at 29-38% provide 7-16pp theoretical margin vs 45% accuracy. | label-geometry-phase1 |
| P0 | Does XGBoost at high-ratio geometries (15:3, 19:7, 20:3) achieve positive expectancy using long-perspective labels? | Not started | — | — | Bidirectional labels produce balanced classes at 3600s but model defaults to hold-predictor at high ratios. Long-perspective labels avoid hold-dominance by design — test whether directional accuracy transfers. | — |
| P1 | Does a 2-class formulation (directional-vs-hold) at 19:7 enable high-ratio geometry exploitation? | Answered | — | — | Model's strength is barrier-reachability detection (volatility features), weakness is direction. 2-class decoupling may convert hold-prediction tendency into a useful filter. | 2class-directional |
| P0 | Is the two-stage expectancy at 19:7 genuinely positive under corrected hold-bar PnL ($0 for hold-bar trades)? | Answered | 2-class formulation | — | Resolves 8x uncertainty in reported $3.78/trade. Corrected estimate ~$0.44 ± $0.16 — the go/no-go for 19:7 geometry exploitation. | pnl-realized-return |
| P0 | Does Stage 1 threshold optimization (0.5→0.9) improve realized expectancy at 19:7 by reducing hold-bar exposure? | Answered | corrected PnL | — | PnL decomposition predicts each 10pp hold-frac reduction = +$0.27/trade. At 5% hold (threshold~0.80), est. $3.45/trade. Must validate with actual threshold sweep. | threshold-sweep |
| P2 | Does class-weighted XGBoost at 19:7 force directional predictions while maintaining >38.4% directional accuracy? | Answered | — | — | If forced directional predictions maintain accuracy above 19:7 breakeven WR, the favorable payoff converts signal to profit. | class-weighted-stage1 |
| P2 | Does message sequence information help specifically in high-volatility regimes, even though aggregate R2 was null? | Not started | — | — | If Δ_msg > 0 in high-vol tercile, reopens message encoder line for regime-conditional use | regime-stratified-info-decomposition |
| P2 | How sensitive is oracle expectancy to transaction cost assumptions? | Not started | — | — | Determines edge robustness under realistic execution | — |
| P3 | Does regime-conditional trading (Q1-Q2 only) produce positive expectancy? | Not started | — | — | GBT profitable in H1 2022, negative in H2. Limited by single year of data. | — |
| P3 | Does CNN spatial R² improve on genuine trade-triggered tick bars vs time_5s? | Answered | — | — | Determines whether event bars should replace time_5s | R3b-genuine-tick-bars (CONFIRMED low confidence) |
| P1 | Does timeout-filtered sequential execution (skip entries where barrier unlikely to resolve before day end) improve per-trade expectancy toward $5.00 barrier-hit maximum? | Not started | — | — | Observed $2.50/trade vs $5.00 barrier-hit theoretical — 50% timeout dilution is the largest single drag. Filtering is zero-model-change improvement path. Determines if sequential expectancy can reach $4+/trade, reducing account minimum. | — |
| P2 | Does multi-contract sequential execution (N=2,5,10,20,36 MES) produce N-proportional daily PnL with sqrt(N)-proportional drawdown? | Not started | — | — | Sequential 1-contract captures 5.7% of bar-level PnL. Concurrent_mean=35.9 → ~36 contracts is ceiling. Determines if institutional-scale deployment is viable. | — |

---

## 5. Answered Questions

| Question | Answer Type | Answer | Evidence |
|----------|-------------|--------|----------|
| Are event-driven bars (volume, tick, dollar) superior to time bars per subordination theory? | REFUTED | No — time bars are the baseline. Dollar bars show higher AR but at non-actionable timescales. | `.kit/results/subordination-test/metrics.json` (R1) |
| Does predictive information decompose across spatial, message, and temporal sources? | CONFIRMED (null) | No information gap passes dual threshold. Book snapshot is sufficient statistic. | `.kit/results/info-decomposition/analysis.md` (R2) |
| Does CNN spatial encoding on structured book input outperform flattened features? | CONFIRMED | CNN R²=0.132 on (20,2) book — 20× higher than flattened MLP. **Caveat (r3-reproduction):** R²=0.132 includes ~36% inflation from test-as-validation leakage; proper-validation R²≈0.084. Still 12× higher than flattened MLP R²=0.007. | `.kit/results/book-encoder-bias/` (R3), `.kit/results/r3-reproduction-pipeline-comparison/analysis.md` |
| Do MES 5-second bar returns contain exploitable temporal structure? | REFUTED | All 36 AR configs negative R². 0/52 tests pass. Martingale. | `.kit/results/temporal-predictability/analysis.md` (R4) |
| Do event-driven bars (volume, dollar) contain temporal structure absent from time bars? | REFUTED | Volume: no signal. Dollar: marginal sub-second AR only, redundant with static features. 0/48 dual passes. | `.kit/results/temporal-predictability-event-bars/analysis.md` (R4b) |
| Does temporal signal emerge on tick bars, at actionable timescales (≥5s), or at extended horizons (17-83min)? | REFUTED | 0/54+ dual passes (R4c) + 0/38 more (R4d, 5 operating points) = 0/168+ cumulative across R4 chain. Dollar bars at 7s, 14s, 69s and tick bars at 50s, 300s all null. MES is martingale across 7 bar types and timescales 0.14s-300s. | `.kit/results/temporal-predictability-completion/analysis.md` (R4c), `.kit/results/temporal-predictability-dollar-tick-actionable/analysis.md` (R4d) |
| What is the recommended architecture? | CONFIRMED | CNN+GBT Hybrid, static features, time_5s bars, triple barrier labels. No temporal/message encoder. | `.kit/results/synthesis/analysis.md` (R6) |
| Does the oracle produce positive expectancy after costs? | CONFIRMED | GO. $4.00/trade, PF=3.30, WR=64.3%, Sharpe=0.362. All 6 criteria pass. | `.kit/results/oracle-expectancy/metrics.json` (R7) |
| Does the Phase 9A C++ bar export produce book data equivalent to R3's data? | CONFIRMED | YES — byte-identical. features.csv (R3) and time_5s.csv (9C) have identity rate=1.0, max diff=0.0. R3 loaded from the same C++ export, not a separate Python pipeline. The "Python vs C++" distinction was a false premise. | `.kit/results/r3-reproduction-pipeline-comparison/analysis.md` |
| Why does the CNN fail to reproduce R3's R²=0.132 in Phases 9B/9C? | CONFIRMED | Root cause: missing TICK_SIZE normalization on prices (÷0.25 for tick offsets) + per-fold z-scoring instead of per-day z-scoring on sizes. Data is identical; difference is post-loading normalization. R3's R²=0.132 also includes ~36% inflation from test-as-validation leakage (proper-validation R²≈0.084). | `.kit/results/r3-reproduction-pipeline-comparison/analysis.md` |
| Does CNN spatial encoding work at h=1 (5s horizon)? | INCONCLUSIVE | CNN R²(h=1)=0.0017, but pipeline broken (train R² at h=5 = 0.001 vs R3's 0.132). No valid horizon comparison possible. **Root cause now known (normalization)** — retest with TICK_SIZE + per-day z-score normalization required. | `.kit/results/hybrid-model-training/analysis.md` |
| Does corrected CNN+GBT pipeline produce economically viable trading signals? | REFUTED (Outcome B) | CNN R²=0.089 CONFIRMED (matches 9D's 0.084). But expectancy=-$0.37/trade under base costs, PF=0.924. Gross edge $3.37 consumed by $3.74 RT cost. Profitable only under optimistic costs (+$0.88 at $2.49 RT). Win rate 51.3% vs 53.3% breakeven. Regression-to-classification gap is bottleneck. | `.kit/results/hybrid-model-corrected/analysis.md` |
| Does CNN spatial R² improve on genuine trade-triggered tick bars vs time_5s? | CONFIRMED (low confidence) | tick_100 R²=0.124 vs time_5s 0.089 (+39% relative), but paired t-test p=0.21 (not significant). Verdict depends on fold 5 (R²=0.259); excluding fold 5, mean=0.091 (COMPARABLE). Single-threshold peak among 3 tested. Data volume confound ruled out (r=-0.149). tick_25 WORSE, tick_500 WORSE. Needs replication. | `.kit/results/r3b-genuine-tick-bars/analysis.md` |
| Can end-to-end CNN classification on tb_label close the 2pp win rate gap? | REFUTED (Outcome D) | CNN acc=0.390, GBT acc=0.449 — CNN 5.9pp WORSE. CNN expectancy -$0.146, GBT -$0.064. Long recall only 0.21. CNN spatial signal (R²=0.089 regression) does NOT encode class-discriminative boundaries. CNN line permanently closed for classification. | `.kit/results/e2e-cnn-classification/metrics.json` |
| Can XGBoost hyperparameter tuning improve classification accuracy by 2+pp? | REFUTED (Outcome C) | No — accuracy landscape is a 0.33pp plateau (64 configs, std=0.0006). Best tuned CPCV accuracy 0.4504 vs default 0.4489 (+0.15pp). Expectancy improved -$0.066→-$0.001 via class rebalancing (suppressed longs), not accuracy. Breakeven RT=$3.74 exactly. Feature set is the binding constraint, not hyperparameters. | `.kit/results/xgb-hyperparam-tuning/analysis.md` |
| Can label design lower breakeven WR below ~45% and enable positive expectancy? | REFUTED (Outcome C — methodological caveat) | PARTIALLY: Breakeven WR IS lowered mechanically (15:3 = 33.3%, 20:3 = 29.6%). Oracle is profitable at ALL 123 geometries. BUT SC-2 ($5.00 oracle net exp) failed — peak $4.126 at base costs. Abort prevented Phase 1 training. The $5.00 threshold was miscalibrated: it filtered on oracle per-trade profit (penalizes high-ratio geometries) instead of breakeven WR vs model accuracy (favors high-ratio geometries). Model viability at favorable geometries remains untested. | `.kit/results/label-design-sensitivity/analysis.md` |
| Does XGBoost achieve directional accuracy > breakeven WR + 2pp at high-ratio geometries with bidirectional labels? | REFUTED | No — bidirectional labels on time_5s bars produce 91-99% hold at all geometries (10:5=90.7%, 15:3=97.1%, 19:7=98.6%, 20:3=98.9%). Model degenerates to hold-predictor. CPCV exp=-$0.610. Directional accuracy 53.65% at holdout — only +0.37pp above breakeven (noise). The ~45% accuracy baseline was on long-perspective labels with balanced classes; it does not transfer to bidirectional labels. Geometry hypothesis untestable under these conditions. | `.kit/results/label-geometry-phase1/analysis.md` |
| Does XGBoost at high-ratio geometries with 1-hour time horizon achieve positive expectancy on bidirectional labels? | INCONCLUSIVE | Time horizon fix succeeded (hold 90.7%→32.6% at 10:5). Labels now balanced. But model refuses to trade at high-ratio geometries (0.003-0.28% dir pred rate). At 10:5 (only reliable geometry), dir acc 50.67% is 2.6pp below breakeven, exp -$0.49/trade. 19:7 technically passes SC-3 on 0.28% of bars but holdout = 50% on 52 trades (coin flip). Practical reality = Outcome B: directional signal is ~50% regardless of geometry, favorable payoff structure unexploitable. | `.kit/results/label-geometry-1h/analysis.md` |
| Is the two-stage expectancy at 19:7 genuinely positive under corrected hold-bar PnL? | CONFIRMED (nuanced) | YES — realized-return model gives $0.90/trade (positive). Conservative model ($0 gross, costs charged) gives ~$0.44/trade (also positive). BUT hold-bar directional accuracy 51.04% < 52% threshold (SC-2 fails). Result driven by Fold 2 outlier; 3-fold variance too high for statistical confidence (std $1.16 on mean $0.90). Hold-bar forward returns unbounded (±63 ticks at p10/p90, not ±19 as assumed). Threshold optimization is the clear next step. | `.kit/results/pnl-realized-return/analysis.md` |
| Does Stage 1 threshold optimization (0.5→0.9) improve realized expectancy at 19:7 by reducing hold-bar exposure? | REFUTED | No — XGBoost P(directional) is catastrophically compressed (80.6% in [0.50, 0.60], IQR 4pp). Trade rate cliff: 64.5% at T=0.55 to 4.5% at T=0.60. No threshold achieves exp > $1.50 at >15% trade rate. Optimal = baseline (T=0.50, $0.90). Dir-bar PnL improves at moderate thresholds (signal exists) but model cannot produce enough high-confidence predictions. Threshold optimization is a dead end for this model. | `.kit/results/threshold-sweep/analysis.md` |
| Does class-weighted XGBoost at 19:7 force directional predictions while maintaining >38.4% directional accuracy? | REFUTED | No — `scale_pos_weight < 1` does not spread the probability distribution; it compresses it. IQR decreases monotonically (4.0pp -> 3.8 -> 3.2 -> 2.0 -> 0.8pp). Recall collapses 465x at weight=0.50 (93%->0.2%). Precision ALSO decreases (0.556->0.426). Weights 0.25/0.20 produce zero positive predictions. The mechanism is a logit-shift, not a spread — structural limitation of XGBoost binary:logistic on this task. Baseline (w=1.0, T=0.50, $0.90/trade) is the ceiling for this pipeline. | `.kit/results/class-weighted-stage1/analysis.md` |
| Does a 2-class formulation (directional-vs-hold) at 19:7 enable high-ratio geometry exploitation? | CONFIRMED (caveat) | YES for trade volume: 301x increase (0.28%→85.2% trade rate). YES for mechanism: Stage 1 reachability learnable (58.6% acc), decomposition removes cross-entropy hold-prediction trap. CAVEAT for economics: reported $3.78/trade likely inflated ~8x by PnL model assigning full barrier payoffs to hold-bar trades (44.4% of trades). Corrected estimate ~$0.44/trade ± $0.16 — positive but marginal. Dir accuracy 50.05% ≈ coin flip; economics driven by 2.71:1 payoff ratio, not prediction skill. Stage 2 learns NO directional features at 19:7 (top-10 all volatility/activity). | `.kit/results/2class-directional/analysis.md` |
| Is the two-stage 19:7 pipeline statistically validated as positive-expectancy under corrected costs ($2.49 RT) with CPCV? | CONFIRMED (Outcome A) | YES — CPCV mean $1.81/trade, 95% CI [$1.46, $2.16], PBO=0.067 (3/45 negative), t=10.29, p<1e-13. Holdout $1.46/trade. All 6 SC pass. Break-even RT $4.30. Edge is structural (payoff asymmetry at 19:7, breakeven acc=34.6%, not directional skill at 50.16%). All 10 groups positive. Pessimistic costs ($4.99) not viable (-$0.69). Hold-bar variance is dominant risk. | `.kit/results/cpcv-corrected-costs/analysis.md` |
| Is sequential 1-contract execution viable at small account sizes ($5K) with >= $0.50/trade expectancy? | REFUTED (productive) | Edge exceeds threshold: $2.50/trade (5x), $412.77/day, $103,605/year on 1 MES. BUT min_account = $48,000 (9.6x threshold). 162 trades/day (not 40-80), avg_bars_held = 28 (not 75), hold_skip = 66.1% (not 43%). Sequential entry creates volatility-timing selection bias: entries during volatile moments (shorter races), exits into calm periods (higher hold-skip). Win rate 49.93% = coin flip; edge is pure 2.71:1 payoff asymmetry. Timeout trades dilute $5 barrier-hit expectancy to $2.50. Strategy viable at medium accounts ($48K all-paths, $26.6K 95%-paths), not retail micro-accounts. | `.kit/results/trade-level-risk-metrics/analysis.md` |

Answer types: `CONFIRMED` | `REFUTED` | `Deferred` | `Superseded`

---

## 6. Working Hypotheses

- MES 5-second returns are a martingale difference sequence. No temporal encoder is justified.
- CNN on structured (20,2) book input captures spatial structure that flattened features destroy (proper-validation R²≈0.084 vs. 0.007), but this regression signal does NOT transfer to classification — CNN accuracy 0.388 vs GBT 0.449 on full-year CPCV.
- Static book features are a sufficient statistic for prediction — message sequence and temporal history add nothing.
- **GBT-only is the most promising path.** CNN line closed for classification (Outcome D from E2E experiment).
- **XGBoost directional accuracy is ~50-51% regardless of label type, geometry, or hyperparameters.** This is a hard ceiling of the 20-feature set on MES time_5s bars. xgb-tuning showed 0.33pp accuracy plateau; label-geometry-1h shows same ~50% directional accuracy at 10:5 on balanced bidirectional labels.
- **High-ratio geometry hypothesis is theoretically valid but practically unexploitable.** Oracle confirms profitable at all geometries. But the model defaults to hold-prediction at wider barriers (0.003-0.28% dir pred rate even at 47% hold). The favorable payoff structure cannot be accessed.
- **Next priority: formulation changes**, not feature/model changes. 2-class (directional-vs-hold) to decouple reachability from direction. Class-weighted loss to force directional predictions. Long-perspective labels at varied geometries.
