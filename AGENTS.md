# AGENTS.md — MBO-DL Agent Coordination

## Current State (updated 2026-02-17, Phase 4 feature-computation complete)

- **Build:** Green.
- **Unit tests:** 726/727 pass, 0 failures, 1 disabled (`BookBuilderIntegrationTest.ProcessSingleDayFile`).
- **Integration tests:** 22 tests, excluded from default ctest (`--label-exclude integration`).
- **Phase 4 (feature-computation):** Complete. Track A hand-crafted features (~45), Track B raw representations, forward returns, warm-up enforcement, CSV export. 173 new unit tests.
- **`ORCHESTRATOR_SPEC.md`** archived to `completed_specs/`.

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
| 4 | feature-computation | done | done | done | pending |

## Next Action

Ship Phase 4 (commit), then start Phase 5 (feature-analysis):
```bash
source .orchestration-kit.env && ./.kit/tdd.sh red .kit/docs/feature-analysis.md
```

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
