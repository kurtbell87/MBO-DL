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
| P1 | Can XGBoost hyperparameter tuning improve classification accuracy by 2+pp? Default params never optimized. GBT shows Q1-Q2 positive expectancy (+$0.003, +$0.029 base). | Not started | — | — | Determines if the 2pp gap to breakeven is a tuning issue vs architectural limit | xgb-hyperparam-tuning |
| P1 | Can label design (wider target 15t / narrower stop 3t) lower breakeven win rate below current ~45%? | Not started | — | — | Breakeven drops from 53.3% to ~42.5% at 15:3 ratio | label-design-sensitivity |
| P2 | How sensitive is oracle expectancy to transaction cost assumptions? | Not started | — | — | Determines edge robustness under realistic execution | — |
| P3 | Does regime-conditional trading (Q1-Q2 only) produce positive expectancy? | Not started | — | — | GBT profitable in H1 2022, negative in H2. Limited by single year of data. | — |
| P3 | Does CNN spatial R² improve on genuine trade-triggered tick bars vs time_5s? | Answered | — | — | Determines whether event bars should replace time_5s | R3b-genuine-tick-bars (CONFIRMED low confidence) |

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

Answer types: `CONFIRMED` | `REFUTED` | `Deferred` | `Superseded`

---

## 6. Working Hypotheses

- MES 5-second returns are a martingale difference sequence. No temporal encoder is justified.
- CNN on structured (20,2) book input captures spatial structure that flattened features destroy (proper-validation R²≈0.084 vs. 0.007), but this regression signal does NOT transfer to classification — CNN accuracy 0.388 vs GBT 0.449 on full-year CPCV.
- Static book features are a sufficient statistic for prediction — message sequence and temporal history add nothing.
- **GBT-only is the most promising path.** CNN line closed for classification (Outcome D from E2E experiment). GBT shows marginal profitability in Q1 (+$0.003) and Q2 (+$0.029) under base costs with default hyperparameters — never tuned.
- XGBoost hyperparameter tuning is the highest-priority next step — default params from 9B, never optimized on full-year data.
- Label design sensitivity (wider target / narrower stop) may lower breakeven win rate enough to flip aggregate expectancy positive.
