# AGENTS.md — MBO-DL Agent Coordination

## Current State (updated 2026-02-25)

- **Build:** Green.
- **Unit tests:** 1144+ registered (label-exclude integration). Bidirectional TB + export wiring tests added. TDD phases exited 0.
- **Integration tests:** 22 tests, excluded from default ctest (`--label-exclude integration`).
- **30+ phases complete** (13 engineering + 17 research). CNN line closed (Outcome D). GBT-only path forward.
- **Last TDD (COMPLETE):** Bidirectional export wiring — `bar_feature_export` defaults to bidirectional labels, 152-column Parquet schema, `--legacy-labels` flag. Spec: `.kit/docs/bidirectional-export-wiring.md`. Changed: `CMakeLists.txt`, `tests/parquet_export_test.cpp`, `tools/bar_feature_export.cpp`. New: `tests/bidirectional_export_test.cpp`.
- **Prior TDD (COMPLETE):** Bidirectional TB labels (`compute_bidirectional_tb_label()`) + Oracle expectancy CLI params.
- **Compute:** Local preferred for CPU-only experiments (<1GB). RunPod for GPU. EC2 spot for large data only.
- **Full-year dataset:** 1.16M bars, 251 days, 255MB Parquet, S3-backed. Needs re-export with 152-column schema.

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

## Next Action

1. **Re-export full-year data with bidirectional labels** (P0): Run `bar_feature_export` on 251 days with 152-column schema (EC2). Produces updated Parquet for label-design-sensitivity.
2. **Label design sensitivity** (P1): Test wider target / narrower stop. Requires bidirectional full-year export. Spec: `.kit/experiments/label-design-sensitivity.md`. Local compute.
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
