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
| Framework  | C++20 (pipeline + models), Python (research analysis) |
| Compute    | CPU only (no GPU required for current experiments) |
| Timeline   | Sequential R1–R4c research chain |
| Baselines  | R4 time_5s GBT, dual threshold (Δ>20% + corrected p<0.05) |

---

## 3. Non-Goals (This Phase)

- Live/paper trading integration (R|API+ installed but not integrated)
- Multi-instrument generalization (MES only)
- Deep learning training infrastructure (CNN training deferred to build phase)

---

## 4. Open Questions

Status: `Not started` | `In progress` | `Answered` | `Blocked` | `Deferred`

| Priority | Question | Status | Parent | Blocker | Decision Gate | Experiment(s) |
|----------|----------|--------|--------|---------|---------------|---------------|
| P2 | How sensitive is oracle expectancy to transaction cost assumptions? | Not started | — | — | Determines edge robustness under realistic execution | — |

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

Answer types: `CONFIRMED` | `REFUTED` | `Deferred` | `Superseded`

---

## 6. Working Hypotheses

- MES 5-second returns are a martingale difference sequence. No temporal encoder is justified.
- CNN on structured (20,2) book input captures spatial structure that flattened features destroy (proper-validation R²≈0.084 vs. 0.007; R3's reported R²=0.132 was inflated ~36% by test-as-validation leakage).
- Static book features are a sufficient statistic for prediction — message sequence and temporal history add nothing.
- Triple barrier labeling is strictly superior to first-to-hit for MES at oracle parameters.
- The CNN+GBT Hybrid architecture is the correct next step for the model build phase.
