# AGENTS.md — MBO-DL Agent Coordination

## Current State (updated 2026-02-19, Tick Bar Fix TDD — COMPLETE)

- **Build:** Green.
- **Unit tests:** 1003/1004 pass (1 disabled, 1 skipped) + new tick_bar_fix tests. TDD phases exited 0.
- **Integration tests:** 22 tests, excluded from default ctest (`--label-exclude integration`).
- **Tick bar fix (TB-Fix):** Complete. `book_builder.hpp` emits `trade_count` per snapshot. `tick_bar_builder.hpp` accumulates trade counts, closes bars at threshold. Regression: time/dollar/volume bars unchanged.
- **19 phases complete** (9 engineering + 10 research). Tick bars now genuine event bars. R3b rerun unblocked.

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

1. **CNN Pipeline Fix (HIGHEST PRIORITY):** Apply TICK_SIZE normalization (÷0.25) + per-day z-scoring in Python training pipeline. Re-attempt CNN+GBT with proper validation. Expected R²≈0.084.
2. **R3b Rerun (event-bar research):** Rerun R3b with genuine tick bars now that construction is fixed.

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
