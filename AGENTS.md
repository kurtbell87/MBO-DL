# AGENTS.md — MBO-DL Agent Coordination

## Current State (updated 2026-02-25)

- **Build:** Green.
- **Unit tests:** 1144+ registered (label-exclude integration). New bidirectional TB tests added. TDD phases exited 0.
- **Integration tests:** 22 tests, excluded from default ctest (`--label-exclude integration`).
- **29+ phases complete** (12 engineering + 17 research). CNN line closed (Outcome D). GBT-only path forward.
- **Active TDD:** Bidirectional triple barrier labels — independent long/short race evaluation. Spec: `.kit/docs/bidirectional-label-export.md`. Changed: `triple_barrier.hpp`, `CMakeLists.txt`, new `bidirectional_tb_test.cpp`.
- **Prior TDD:** Oracle expectancy CLI parameterized (`--target/--stop/--take-profit/--output/--help`). Spec: `.kit/docs/oracle-expectancy-params.md`.
- **Compute:** Local preferred for CPU-only experiments (<1GB). RunPod for GPU. EC2 spot for large data only.
- **Full-year dataset:** 1.16M bars, 251 days, 255MB Parquet, S3-backed.

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
| Bidir-TB | bidirectional-label-export | in progress | in progress | — | — |

## Next Action

1. **Complete bidirectional-label-export TDD cycle** (P0): Finish exit criteria, integrate into `bar_feature_export`, verify T1-T10. Blocks label-design-sensitivity. Spec: `.kit/docs/bidirectional-label-export.md`.
2. **Label design sensitivity** (P1): Test wider target / narrower stop. Requires bidirectional labels. Spec: `.kit/experiments/label-design-sensitivity.md`. Local compute.
3. **XGBoost hyperparameter tuning** (P1): Grid search to close the 2pp win rate gap. Spec: `.kit/experiments/xgb-hyperparam-tuning.md`. Local compute.
4. **Regime-conditional trading** (P3): Q1-Q2 only strategy. Spec not yet created.

## Agent Roles

| Agent | Scope | Entry point |
|-------|-------|-------------|
| Orchestrator | Sequences phases, reads state files only, never writes code | `CLAUDE.md` → `.kit/LAST_TOUCH.md` |
| TDD sub-agent | Executes red/green/refactor phases (C++ engineering) | `.kit/tdd.sh <phase> <spec>` |
| Research sub-agent | Executes survey/frame/run/read/log phases (experiments) | `.kit/experiment.sh <phase> <spec>` |
| Research Synthesist | Reads all results, produces `.kit/SYNTHESIS.md` with cross-experiment findings | `/synthesize` command or `.claude/prompts/synthesize.md` |
| Breadcrumb steward | Updates navigation docs before ship | This file, `CLAUDE.md`, `.kit/LAST_TOUCH.md` |

## Constraints

- Sub-agents own source/test files. Orchestrator never reads them.
- Trust exit codes: exit 0 = success, exit 1 = read capsule.
- Integration tests are labeled — never run in default ctest.
- Kit state files live in `.kit/`, not project root.
