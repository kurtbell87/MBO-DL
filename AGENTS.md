# AGENTS.md — MBO-DL Agent Coordination

## Status

**ORCHESTRATOR_SPEC.md is COMPLETE.** All build phases shipped. N=128 overfit validation added.

## Completed TDD Phases

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

**Tests:** 220 unit + 22 integration. All pass.

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
