# AGENTS.md — MBO-DL Agent Coordination

## Current State (updated 2026-02-20, Nested Run Visibility — spec + submodule setup)

- **Branch:** `feat/nested-run-visibility`
- **Build:** Green.
- **Unit tests:** 1003/1004 pass (1 disabled, 1 skipped) + tick_bar_fix tests. TDD phases exited 0.
- **Integration tests:** 22 tests, excluded from default ctest (`--label-exclude integration`).
- **Nested Run Visibility (NRV):** Spec written at `.kit/docs/nested-run-visibility.md`. Orchestration-kit submodule updated with MCP bug fixes (`--run-id` REMAINDER, DEVNULL error blindness, `--reasoning` REMAINDER, missing `KIT_STATE_DIR`). New MCP tools `kit.run_tree` and `kit.run_events` defined in spec. TDD full cycle pending — 0/10 exit criteria completed.
- **21 phases complete** (9 engineering + 12 research). NRV is infrastructure TDD (not model research).

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
| NRV | nested-run-visibility | pending | pending | pending | pending |

## Next Action

1. **Run NRV TDD cycle:** `kit.tdd` with `spec_path=".kit/docs/nested-run-visibility.md"`. Implements `child_run_spawned` event, dashboard parent-child filter, `kit.run_tree` + `kit.run_events` MCP tools. 10 exit criteria, 16 test cases.
2. **After NRV:** End-to-end CNN classification (highest model priority), XGBoost tuning, or label sensitivity.

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
