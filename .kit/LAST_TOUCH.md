# Last Touch — Cold-Start Briefing

**Read this file first. Then read `CLAUDE.md` for the full protocol. Do not read source files.**

---

## CRITICAL: Orchestrator Protocol

**You are the ORCHESTRATOR. You NEVER write code. You NEVER read source files. You NEVER run verification commands.**

- ALL code is written by kit sub-agents (`.kit/tdd.sh`, `.kit/experiment.sh`, `.kit/math.sh`)
- You ONLY: read/write state `.md` files, launch kit phases (MCP tools or bash), check exit codes via dashboard
- If you need code written → delegate to a kit phase
- If you need something verified → the kit phase already verified it (trust exit 0)
- See `CLAUDE.md` §ABSOLUTE RULES for the full list of things you must never do

---

## TL;DR — Where We Are and What To Do

We've built a complete MES microstructure research platform over 26 phases. The CNN line is **closed for classification** (Outcome D — GBT beats CNN by 5.9pp). GBT-only is the path forward. Cloud-run infrastructure is now **production-grade** with heartbeat monitoring, log streaming, pre-flight validation, and stale run cleanup.

**Your job: launch the next research priority — XGBoost hyperparameter tuning.**

### Last Completed: Cloud-Run Reliability Overhaul (2026-02-23)

Spec: `.kit/docs/cloud-run-reliability.md`. Branch: `chore/cloud-run-reliability`.

**What changed (8 files modified, 2 new):**
- `ec2-bootstrap-gpu.sh` / `ec2-bootstrap.sh` — sync daemon (heartbeat 60s, log 60s, results every 5min), EXIT trap syncs results before exit_code
- `s3.py` — `check_heartbeat()`, `tail_log()` with follow mode
- `remote.py` — `poll_status()` checks EC2 instance state + heartbeat, `gc_stale_runs()` for dead instances
- `cloud-run` CLI — `logs` subcommand (`--lines`, `--follow`), `--validate`/`--skip-smoke` on `run`, enhanced `status`/`ls` (elapsed, cost, heartbeat, log lines)
- `validate.py` (new) — syntax check, import check, smoke test, `validate_all()`
- `state.py` — `gc_stale()` cleans orphaned local state entries
- `experiment.sh` — compute directive template includes `--validate`
- `tests/test_cloud_run_reliability.py` (new) — all tests pass

### Next Actions (Priority Order)

1. **XGBoost hyperparameter tuning on full-year data** — default params from 9B never optimized. GBT already shows Q1-Q2 positive expectancy (+$0.003, +$0.029) with default hyperparams. Most promising path given Outcome D.
2. **Label design sensitivity** — test wider target (15 ticks) / narrower stop (3 ticks).
3. **Regime-conditional trading** — Q1-Q2 only strategy.
4. **2-class formulation** — directional only (merge tb_label=0 into abstain).

### Cloud-Run Usage (Updated)

Cloud-run now supports:
```bash
# Tail logs in real-time
orchestration-kit/tools/cloud-run logs <run-id> --follow

# Validate before launch (auto in experiment.sh)
orchestration-kit/tools/cloud-run run --validate <script> "python <script>" ...

# Check status with heartbeat + cost
orchestration-kit/tools/cloud-run status <run-id>

# Clean up stale runs
orchestration-kit/tools/cloud-run gc
```

---

## Normalization Protocol (CRITICAL — institutional memory)

Three prior experiments (9B, 9C, R3b) failed because of normalization errors. The correct protocol, verified in 9D and 9E:

1. **Book prices (channel 0)**: Divide by TICK_SIZE=0.25 → integer tick offsets. Do NOT z-score.
2. **Book sizes (channel 1)**: log1p() → z-score PER DAY (not per-fold, not globally).
3. **Non-spatial features**: z-score using train-fold stats only.

---

## Project Status

**26 phases complete (10 engineering + 12 research + 1 data export + 2 infra + 1 kit modification). Branch: `chore/cloud-run-reliability`. Tests: all pass.**

### What's Built
- **C++20 data pipeline**: Bar construction, order book replay, multi-day backtest, feature computation/export, oracle expectancy, Parquet export. 1003+ unit tests, 22 integration tests.
- **Full-year dataset**: 251 Parquet files (time_5s bars, 1,160,150 bars, 149 columns, zstd compression). Stored in S3 artifact store.
- **Cloud pipeline**: Docker image in ECR, EBS snapshot with 49GB MBO data, IAM profile. Verified E2E. Now with heartbeat, log streaming, pre-flight validation, and stale GC.
- **EC2 mandatory execution**: `experiment.sh` mandates cloud-run for RUN phases when `COMPUTE_TARGET=ec2`. Compute directive includes `--validate`.

### Key Research Results

| Experiment | Finding | Key Number | Implication |
|-----------|---------|------------|-------------|
| R1 | Subordination refuted | 0/3 significant | Time bars are the baseline |
| R2 | Features sufficient | R²=0.0067 | Book snapshot is sufficient statistic |
| R3 | CNN best encoder | R²=0.132 (leaked) / 0.084 (proper) | Spatial structure matters |
| R4/R4b/R4c/R4d | No temporal signal | 0/168+ passes | Drop SSM/temporal encoder permanently |
| 9E | Pipeline bottleneck | exp=-$0.37/trade | Regression→classification gap is the limit |
| **10** | **CNN classification refuted** | **GBT +5.9pp** | **CNN line closed; GBT-only path forward** |

---

## State Files (read order)

1. **This file** — you're here
2. **`CLAUDE.md`** — full protocol, absolute rules, current state, institutional memory
3. **`.kit/RESEARCH_LOG.md`** — cumulative findings from all 12+ experiments
4. **`.kit/QUESTIONS.md`** — open and answered research questions

---

Updated: 2026-02-23. Next action: XGBoost hyperparameter tuning or merge cloud-run-reliability branch to main.
