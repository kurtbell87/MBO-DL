# AGENTS.md — MBO-DL Agent Coordination

## Current State (updated 2026-02-23, Parallel Batch Dispatch — In Progress)

- **Build:** Green.
- **Unit tests:** 1003/1004 pass (1 disabled, 1 skipped). TDD phases exited 0.
- **Integration tests:** 22 tests, excluded from default ctest (`--label-exclude integration`).
- **Parallel batch dispatch (feat/parallel-batch-dispatch):** In progress. Adds N-way parallel experiment dispatch to cloud-run. New `batch.py` module, CLI `batch {run,status,pull,ls}` subcommands, MCP `kit.research_batch` tool, `batch_id` tracking in state/remote, `parallelizable` surfacing in preflight, `experiment.sh batch` shell command. Spec: `.kit/docs/parallel-batch-dispatch.md`.
- **Cloud execution (research-cloud-execution):** Complete. `experiment.sh` now mandates EC2 via `cloud-run` when `COMPUTE_TARGET=ec2`. `sync_results()` auto-pulls results between RUN and READ. Block commands (`cycle`, `full`, `program`) work with EC2 automatically.
- **25+ phases complete** (10 engineering + 12 research + 1 data export + 1 infra + 1 kit modification). Full-year dataset + cloud GPU pipeline ready.

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

## Next Action

1. **Complete parallel batch dispatch TDD cycle** on `feat/parallel-batch-dispatch`. Verify all exit criteria in `.kit/docs/parallel-batch-dispatch.md`, commit, and merge to main.
2. **XGBoost hyperparameter tuning:** Grid search to close the 2pp win rate gap (use batch dispatch for parallel hyperparameter sweeps).
3. **Label design sensitivity:** Test wider target (15 ticks) / narrower stop (3 ticks).

## Agent Roles

| Agent | Scope | Entry point |
|-------|-------|-------------|
| Orchestrator | Sequences TDD phases, reads state files only | `CLAUDE.md` → `.kit/LAST_TOUCH.md` |
| TDD sub-agent | Executes red/green/refactor phases | `.kit/tdd.sh <phase> <spec>` |
| Breadcrumb steward | Updates navigation docs before ship | This file, `CLAUDE.md`, `.kit/LAST_TOUCH.md` |

## Constraints

- Sub-agents own source/test files. Orchestrator never reads them.
- Trust exit codes: exit 0 = success, exit 1 = read capsule.
- Integration tests are labeled — never run in default ctest.
- Kit state files live in `.kit/`, not project root.
