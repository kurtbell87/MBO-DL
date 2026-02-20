# Agents — MBO-DL Session State

## Current State (updated 2026-02-20, Nested Run Visibility — spec + submodule setup)

**Build:** Green.
**Tests:** 1003/1004 unit tests pass (1 disabled, 1 skipped), 22 integration tests (labeled, excluded).
**Branch:** `feat/nested-run-visibility`

### Completed This Cycle

- **Spec created:** `.kit/docs/nested-run-visibility.md` (10 exit criteria, 16 test cases)
- **Submodule updated:** `orchestration-kit` with MCP server bug fixes (REMAINDER, DEVNULL, KIT_STATE_DIR)
- **Env updated:** `.orchestration-kit.env` for new MCP tool support

### Phase Sequence

| # | Spec | Status |
|---|------|--------|
| 1 | bar-construction | **Done** |
| 2 | oracle-replay | **Done** |
| 3 | multi-day-backtest | **Done** |
| R1 | subordination-test | **Done** (REFUTED) |
| 4 | feature-computation | **Done** |
| 5 | feature-analysis | **Done** |
| R2 | info-decomposition | **Done** (FEATURES SUFFICIENT) |
| R3 | book-encoder-bias | **Done** (CNN WINS) |
| R4 | temporal-predictability | **Done** (NO SIGNAL) |
| 6 | synthesis | **Done** (CONDITIONAL GO) |
| 7 | oracle-expectancy | **Done** |
| 7b | oracle_expectancy tool | **Done** (GO) |
| 8 | bar-feature-export | **Done** |
| R4b | temporal-predictability-event-bars | **Done** (NO SIGNAL — robust) |
| R4c | temporal-predictability-completion | **Done** (CONFIRMED — all nulls) |
| R4d | temporal-predictability-dollar-tick-actionable | **Done** (CONFIRMED) |
| 9A | hybrid-model | **Done** (C++ TB labels) |
| 9B-9E | hybrid pipeline research | **Done** (CNN R²=0.089, pipeline not viable) |
| R3b | event-bar CNN | **Done** (tick_100 R²=0.124, p=0.21) |
| TB-Fix | tick-bar-fix | **Done** |
| **NRV** | **nested-run-visibility** | **In progress** — TDD cycle pending |

### Next Actions

1. Run NRV TDD cycle: `kit.tdd` with `spec_path=".kit/docs/nested-run-visibility.md"`
2. After NRV: end-to-end CNN classification, XGBoost tuning, or label sensitivity
