# AGENTS.md — MBO-DL Agent Coordination

## Current State (updated 2026-02-25)

- **Build:** Green.
- **Unit tests:** 1144+ registered (label-exclude integration). Geometry CLI tests in `bar_feature_export_geometry_test.cpp`. TDD phases exited 0.
- **Integration tests:** 22 tests, excluded from default ctest (`--label-exclude integration`).
- **30+ phases complete** (14 engineering + 17 research). CNN line closed (Outcome D). XGBoost tuning done (REFUTED, Outcome C). GBT-only path forward.
- **Last completed:** bar-feature-export-geometry TDD (`--target`/`--stop` CLI flags). All label-design-sensitivity prerequisites now DONE.
- **Prior TDD (COMPLETE):** Bidirectional export wiring (PR #27), bidirectional TB labels (PR #26), oracle expectancy CLI params.
- **Compute:** Local preferred for CPU-only experiments (<1GB). RunPod for GPU. EC2 spot for large data only.
- **Bidirectional dataset:** 312 files, 152-col Parquet, S3: `s3://kenoma-labs-research/results/bidirectional-reexport/`.

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

## Next Action

1. **Label design sensitivity** (P1, FULLY UNBLOCKED): Oracle heatmap sweep (144 geometries) + GBT training on best geometries using bidirectional data. All prerequisites DONE (--target/--stop CLI, bidirectional re-export, oracle CLI params). Spec: `.kit/experiments/label-design-sensitivity.md`.
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
