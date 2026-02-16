# AGENTS.md — MBO-DL Agent Coordination

## Current State (updated 2026-02-16, pre-ship)

- **Build:** Green.
- **Unit tests:** 204/205 pass, 0 failures, 1 disabled (`BookBuilderIntegrationTest.ProcessSingleDayFile`).
- **Integration tests:** 14 tests, excluded from default ctest (`--label-exclude integration`).
- **Phase 7 (integration-overfit):** Red + green exit 0. Refactor pending.
- **Phase 8 (SSM):** Skipped — requires CUDA + Python, no GPU available.

## Completed TDD Phases

| Phase | Module | Red | Green | Refactor |
|-------|--------|-----|-------|----------|
| 1 | book_builder | done | done | done |
| 2 | feature_encoder | done | done | done |
| 3 | oracle_labeler + trajectory_builder | done | done | done |
| 4 | MLP model | done | done | done |
| 5 | GBT model | done | done | done |
| 6 | CNN model | done | done | done |
| 7 | integration-overfit | done | done | **pending** |
| 8 | SSM model | skipped | skipped | skipped |

## Next Action

Run refactor for integration-overfit, then ship:
```bash
source .master-kit.env && ./.kit/tdd.sh refactor .kit/docs/integration-overfit.md
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
