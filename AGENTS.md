# AGENTS.md — MBO-DL Agent Coordination

## Current State (updated 2026-02-23, Parallel Batch Dispatch — Complete)

- **Build:** Green.
- **Unit tests:** 1003/1004 pass (1 disabled, 1 skipped). TDD phases exited 0.
- **Integration tests:** 22 tests, excluded from default ctest (`--label-exclude integration`).
- **Parallel batch dispatch (`tdd/parallel-batch-dispatch`):** Complete. All components delivered: `batch.py` module, CLI `batch {run,status,pull,ls}` subcommands, MCP `kit.research_batch` tool, `batch_id` tracking, `parallelizable` in preflight, `experiment.sh batch` command. Specs: `.kit/docs/parallel-batch-dispatch.md`, `.kit/docs/experiment-batch-command.md`.
- **Cloud execution (research-cloud-execution):** Complete. `experiment.sh` now mandates EC2 via `cloud-run` when `COMPUTE_TARGET=ec2`. `sync_results()` auto-pulls results between RUN and READ. Block commands (`cycle`, `full`, `program`, `batch`) work with EC2 automatically.
- **26+ phases complete** (10 engineering + 12 research + 1 data export + 1 infra + 2 kit modifications). Full-year dataset + cloud GPU pipeline ready.

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

1. **Commit and merge parallel batch dispatch** from `tdd/parallel-batch-dispatch` to main. All exit criteria met for both specs.
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
