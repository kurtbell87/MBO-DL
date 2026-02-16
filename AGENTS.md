# AGENTS.md — MBO-DL Agent Coordination

## Current State (updated 2026-02-16, ship-ready)

- **Build:** Green.
- **Unit tests:** 219/219 pass, 0 failures, 1 disabled (`BookBuilderIntegrationTest.ProcessSingleDayFile`).
- **Integration tests:** 14 tests, excluded from default ctest (`--label-exclude integration`).
- **Phase 7 (integration-overfit):** Complete (red + green + refactor exit 0).
- **Phase 8 (SSM):** Skipped — requires CUDA + Python, no GPU available.
- **Phase 9 (serialization):** TDD cycle complete (red + green + refactor exit 0). 15 serialization tests passing.

## Completed TDD Phases

| Phase | Module | Red | Green | Refactor |
|-------|--------|-----|-------|----------|
| 1 | book_builder | done | done | done |
| 2 | feature_encoder | done | done | done |
| 3 | oracle_labeler + trajectory_builder | done | done | done |
| 4 | MLP model | done | done | done |
| 5 | GBT model | done | done | done |
| 6 | CNN model | done | done | done |
| 7 | integration-overfit | done | done | done |
| 8 | SSM model | skipped | skipped | skipped |
| 9 | serialization | done | done | done |

## Next Action

Ship serialization:
```bash
source .master-kit.env && ./.kit/tdd.sh ship .kit/docs/serialization.md
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
