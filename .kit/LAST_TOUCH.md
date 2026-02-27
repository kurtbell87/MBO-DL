# Last Touch — Cold-Start Briefing

**Read this file first. Then read `CLAUDE.md` for the full protocol. Do not read source files.**

---

## CRITICAL: Orchestrator Protocol

**You are the ORCHESTRATOR. You NEVER write code. You NEVER read source files. You NEVER run verification commands.**

- ALL code is written by kit sub-agents (`.kit/tdd.sh`, `.kit/experiment.sh`, `.kit/math.sh`)
- You ONLY: read/write state `.md` files, launch kit phases (MCP tools or bash), check exit codes via dashboard
- If you need code written -> delegate to a kit phase
- If you need something verified -> the kit phase already verified it (trust exit 0)
- See `CLAUDE.md` ABSOLUTE RULES for the full list of things you must never do

---

## TL;DR — Where We Are and What To Do

### Just Completed (2026-02-27)

1. **CPCV Validation at Corrected Costs — CONFIRMED (Outcome A), PR #38** — 45-split CPCV (N=10, k=2) on the two-stage pipeline at 19:7 (w=1.0, T=0.50) under corrected AMP costs. **First statistically validated positive-expectancy configuration in the project.** All 6 SC pass. All 5 sanity checks pass.

   | Metric | Value |
   |--------|-------|
   | CPCV mean expectancy (base $2.49) | **$1.81/trade** |
   | 95% CI | [$1.46, $2.16] — entirely above zero |
   | PBO | 6.7% (3/45 splits negative) |
   | Fraction positive (base) | 93.3% (42/45) |
   | t-stat / p-value | 10.29 / 1.35e-13 |
   | Holdout expectancy (base) | $1.46/trade |
   | Break-even RT | $4.30 |
   | Pooled dir accuracy | 50.16% (coin flip — edge is payoff asymmetry) |
   | Wall-clock | 2.6 min (98 XGB fits) |

   **Critical insight:** Edge comes from 19:7 payoff asymmetry (breakeven at 34.6% accuracy), NOT directional prediction skill (50.16%). Strategy is "bet on asymmetric payoffs when volatility makes barriers reachable."

   **Regime dependence:** All 10 groups positive. Late-year (groups 6-9, Jul-Oct) $2.37/trade vs early-year (groups 0-5, Jan-Jun) $1.44. 2022's elevated volatility helps barrier reachability.

   **Cost sensitivity:** Viable at base ($1.81), strong at optimistic ($3.06), fails at pessimistic (-$0.69). Break-even RT $4.30 provides $1.81 margin above base.

2. **Prior experiments building to this:** PnL Realized Return (PR #35), Threshold Sweep (PR #36, REFUTED), Class-Weighted Stage 1 (PR #37, REFUTED). All parameter-level interventions exhausted — the baseline pipeline at $0.90/trade (old costs) was the ceiling. Cost correction ($3.74 → $2.49) unlocked the positive result.

### Next: Paper Trading Infrastructure (Rithmic R|API+)

Outcome A triggers the deployment path per the decision rules. Begin with 1 /MES contract.

**Other high-value follow-ups:**
- **Hold-bar exit optimization** — 43% hold fraction with unbounded returns is the largest variance driver. Test stop-loss on hold bars.
- **Multi-year validation** — 2022 is regime-specific (rising rates, elevated vol). Test on 2023/2024 data if available.
- **Regime-conditional position sizing** — Groups 6-9 are 1.65x more profitable. Position size by message_rate/volatility.

### Background: Key Verdicts

- **CPCV CONFIRMED (Outcome A)** — $1.81/trade, PBO 6.7%, p < 1e-13. First validated pipeline.
- **CNN line CLOSED** for classification (GBT beats CNN by 5.9pp accuracy).
- **XGBoost tuning EXHAUSTED** — 0.33pp plateau. Feature set is the constraint.
- **Oracle edge EXISTS** — $3.22-$9.44/trade at 3600s across all 4 geometries.
- **Directional accuracy ceiling** — ~50-51% regardless of geometry/labels/hyperparameters.
- **Cost correction was the key unlock** — $3.74 → $2.49 base RT recovered $1.25/trade.
- **Edge is structural, not predictive** — 19:7 payoff asymmetry × coin-flip accuracy > 34.6% BEV.

---

## Normalization Protocol (CRITICAL — institutional memory)

Three prior experiments (9B, 9C, R3b) failed because of normalization errors. The correct protocol, verified in 9D and 9E:

1. **Book prices (channel 0)**: Divide by TICK_SIZE=0.25 -> integer tick offsets. Do NOT z-score.
2. **Book sizes (channel 1)**: log1p() -> z-score PER DAY (not per-fold, not globally).
3. **Non-spatial features**: z-score using train-fold stats only.

If you see the sub-agent z-scoring channel 0 or using per-fold z-scoring on sizes, that's the bug that cost us 3 experiment cycles.

---

## Project Status

**30+ phases complete (15 engineering + 26 research). Branch: `experiment/cpcv-corrected-costs`. 1144+ unit tests registered. COMPUTE_TARGET=local.**

### What's Built
- **C++20 data pipeline**: Bar construction, order book replay, multi-day backtest, feature computation/export, oracle expectancy, Parquet export, bidirectional TB labels. 1144+ unit tests, 22 integration tests.
- **Parquet schema**: 152 columns (149 original + `tb_both_triggered`, `tb_long_triggered`, `tb_short_triggered`). `--legacy-labels` flag for 149-column backward compat.
- **CLI flags**: `--target`, `--stop`, `--max-time-horizon`, `--volume-horizon`, `--legacy-labels` on both tools.
- **Full-year dataset**: 251 Parquet files (149-col, time_5s bars, 1,160,150 bars, zstd compression). S3 artifact store.
- **Cloud pipeline**: Docker image in ECR, EBS snapshot with 49GB MBO data, IAM profile. Verified E2E.
- **Parallel batch dispatch**: `cloud-run batch run` launches N experiments in parallel.

### Key Research Results

| Experiment | Finding | Key Number | Implication |
|-----------|---------|------------|-------------|
| R1 | Subordination refuted | 0/3 significant | Time bars are the baseline |
| R2 | Features sufficient | R2=0.0067 | Book snapshot is sufficient statistic |
| R3 | CNN best encoder | R2=0.132 (leaked) / 0.084 (proper) | Spatial structure matters |
| R4/R4b/R4c/R4d | No temporal signal | 0/168+ passes | Drop SSM/temporal encoder permanently |
| R6 | Synthesis | CONDITIONAL GO -> GO | CNN + GBT Hybrid architecture |
| R7 | Oracle expectancy | $4.00/trade, PF=3.30 | Edge exists at oracle level |
| 9B-9D | Normalization root cause | R2: -0.002 -> 0.089 | TICK_SIZE + per-day z-score required |
| 9E | Pipeline bottleneck | exp=-$0.37/trade | Regression->classification gap |
| 10 | E2E CNN classification | GBT wins by 5.9pp | CNN line closed |
| XGB Tune | Accuracy plateau | 0.33pp span, 64 configs | Feature set is binding constraint |
| Label-Sens P0 | Oracle heatmap | 123 geometries, peak $4.13 | Phase 1 training needed |
| **CPCV Corrected** | **CONFIRMED (Outcome A)** | **$1.81/trade, PBO 6.7%, p<1e-13** | **First validated pipeline. Deploy.** |

---

## State Files (read order)

1. **This file** — you're here
2. **`CLAUDE.md`** — full protocol, absolute rules, current state, institutional memory
3. **`.kit/RESEARCH_LOG.md`** — cumulative findings from all experiments
4. **`.kit/QUESTIONS.md`** — open and answered research questions
5. **`.kit/experiments/cpcv-corrected-costs.md`** — CONFIRMED experiment spec
6. **`.kit/results/cpcv-corrected-costs/analysis.md`** — full CPCV analysis with confound discussion

---

Updated: 2026-02-27. CPCV validation CONFIRMED (Outcome A): $1.81/trade, PBO 6.7%, p<1e-13. PR #38. Next: paper trading (Rithmic R|API+).
