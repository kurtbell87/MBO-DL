# AGENTS.md — MBO-DL Agent Coordination

## Current State (updated 2026-02-23, Cloud-Run Reliability Overhaul — COMPLETE)

- **Build:** Green.
- **Tests:** All pass. 1003/1004 unit tests (1 disabled, 1 skipped) + cloud-run reliability tests. TDD phases exited 0.
- **Integration tests:** 22 tests, excluded from default ctest (`--label-exclude integration`).
- **Cloud-run reliability (cloud-run-reliability):** Complete. Bootstrap scripts have sync daemon (heartbeat 60s, log 60s, results 5min). `poll_status()` checks EC2 instance state + heartbeat. `cloud-run logs` subcommand with `--follow`. Pre-flight `--validate` flag (syntax + imports + smoke test). `gc_stale()` cleans orphaned local state. Enhanced `status`/`ls` with elapsed time + cost estimates.
- **Cloud execution (research-cloud-execution):** Complete. `experiment.sh` mandates EC2 via `cloud-run` when `COMPUTE_TARGET=ec2`. Compute directive template now includes `--validate`.
- **26+ phases complete** (10 engineering + 12 research + 1 data export + 2 infra + 1 kit modification). Full-year dataset + cloud GPU pipeline ready.

## Completed TDD Phases (Orchestrator Spec — predecessor)

| Phase | Module | Red | Green | Refactor | Ship |
|-------|--------|-----|-------|----------|------|
| 1 | book_builder | done | done | done | done |
| 2 | feature_encoder | done | done | done | done |
| 3 | oracle_labeler + trajectory_builder | done | done | done | done |
| 4 | MLP model | done | done | done | done |
| 5 | GBT model | done | done | done | done |
| 6 | CNN model | done | done | done | done |
| 7 | integration-overfit (N=32) | done | done | done | done |
| 8 | SSM model | skipped | skipped | skipped | skipped |
| 9 | serialization | done | done | done | done |
| 10 | N=128 overfit validation | done | done | done | done |

## Completed TDD Phases (TRAJECTORY.md — current)

| Phase | Spec | Red | Green | Refactor | Ship |
|-------|------|-----|-------|----------|------|
| 1 | bar-construction | done | done | done | done |
| 2 | oracle-replay | done | done | done | done |
| 3 | multi-day-backtest | done | done | done | done |
| 4 | feature-computation | done | done | done | done |
| 5 | feature-analysis | done | done | done | done |
| 7 | oracle-expectancy | done | done | done | done |
| 8 | bar-feature-export | done | done | done | done |
| 9A | hybrid-model | done | done | done | done |
| TB-Fix | tick-bar-fix | done | done | done | done |
| 7-params | oracle-expectancy-params | done | done | done | done |
| Bidir-TB | bidirectional-label-export | done | done | done | done |
| Bidir-Wire | bidirectional-export-wiring | done | done | done | done |
| Geom-CLI | bar-feature-export-geometry | done | done | done | done |
| TH-CLI | time-horizon-cli | done | done | done | done |

## Next Action

1. **Geometry sweep on long-perspective labels** (P0, FULLY UNBLOCKED): Re-run geometry hypothesis with `--legacy-labels --max-time-horizon 3600` to fix degenerate hold rate from 5-minute cap. Spec needed (adapt from label-geometry-phase1.md).
2. **Regime-conditional trading** (P2): Q1-Q2 only strategy. Spec not yet created.
3. **Tick_100 multi-seed replication** (P3): Confirm tick_100 R²=0.124 with multi-seed. Spec not yet created.

## Agent Roles

| Agent | Scope | Entry point |
|-------|-------|-------------|
| Orchestrator | Sequences phases, reads state files only, never writes code | `CLAUDE.md` → `.kit/LAST_TOUCH.md` |
| TDD sub-agent | Executes red/green/refactor phases (C++ engineering) | `.kit/tdd.sh <phase> <spec>` |
| Research sub-agent | Executes survey/frame/run/read/log phases (experiments) | `.kit/experiment.sh <phase> <spec>` |
| Research Synthesist | Reads all results, produces `.kit/SYNTHESIS.md` with cross-experiment findings | `/synthesize` command |
| Breadcrumb steward | Updates navigation docs before ship | This file, `CLAUDE.md`, `.kit/LAST_TOUCH.md` |

## Constraints

- Sub-agents own source/test files. Orchestrator never reads them.
- Trust exit codes: exit 0 = success, exit 1 = read capsule.
- Integration tests are labeled — never run in default ctest.
- Kit state files live in `.kit/`, not project root.
