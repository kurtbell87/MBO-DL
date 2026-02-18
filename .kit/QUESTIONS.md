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
| P1 | Does CNN spatial encoding work at h=1 (5s horizon)? | Not started | — | — | Confirms CNN architecture at primary operating timescale | — |
| P2 | How sensitive is oracle expectancy to transaction cost assumptions? | Not started | — | — | Determines edge robustness under realistic execution | — |

---

## 5. Answered Questions

| Question | Answer Type | Answer | Evidence |
|----------|-------------|--------|----------|
| Are event-driven bars (volume, tick, dollar) superior to time bars per subordination theory? | REFUTED | No — time bars are the baseline. Dollar bars show higher AR but at non-actionable timescales. | `.kit/results/subordination-test/metrics.json` (R1) |
| Does predictive information decompose across spatial, message, and temporal sources? | CONFIRMED (null) | No information gap passes dual threshold. Book snapshot is sufficient statistic. | `.kit/results/info-decomposition/analysis.md` (R2) |
| Does CNN spatial encoding on structured book input outperform flattened features? | CONFIRMED | CNN R²=0.132 on (20,2) book — 20× higher than flattened MLP. | `.kit/results/book-encoder-bias/` (R3) |
| Do MES 5-second bar returns contain exploitable temporal structure? | REFUTED | All 36 AR configs negative R². 0/52 tests pass. Martingale. | `.kit/results/temporal-predictability/analysis.md` (R4) |
| Do event-driven bars (volume, dollar) contain temporal structure absent from time bars? | REFUTED | Volume: no signal. Dollar: marginal sub-second AR only, redundant with static features. 0/48 dual passes. | `.kit/results/temporal-predictability-event-bars/analysis.md` (R4b) |
| Does temporal signal emerge on tick bars, at actionable timescales (≥5s), or at extended horizons (17-83min)? | REFUTED | 0/54+ dual passes (R4c) + 0/14 more (R4d) = 0/168+ cumulative across R4 chain. Dollar bars at 7s and tick bars at 300s both null. MES is martingale across 7 bar types and timescales 0.14s-83min. | `.kit/results/temporal-predictability-completion/analysis.md` (R4c), `.kit/results/temporal-predictability-dollar-tick-actionable/analysis.md` (R4d) |
| What is the recommended architecture? | CONFIRMED | CNN+GBT Hybrid, static features, time_5s bars, triple barrier labels. No temporal/message encoder. | `.kit/results/synthesis/analysis.md` (R6) |
| Does the oracle produce positive expectancy after costs? | CONFIRMED | GO. $4.00/trade, PF=3.30, WR=64.3%, Sharpe=0.362. All 6 criteria pass. | `.kit/results/oracle-expectancy/metrics.json` (R7) |

Answer types: `CONFIRMED` | `REFUTED` | `Deferred` | `Superseded`

---

## 6. Working Hypotheses

- MES 5-second returns are a martingale difference sequence. No temporal encoder is justified.
- CNN on structured (20,2) book input captures spatial structure that flattened features destroy (R²=0.132 vs. 0.007).
- Static book features are a sufficient statistic for prediction — message sequence and temporal history add nothing.
- Triple barrier labeling is strictly superior to first-to-hit for MES at oracle parameters.
- The CNN+GBT Hybrid architecture is the correct next step for the model build phase.
