# Last Touch — Cold-Start Briefing

## CRITICAL: Orchestrator Protocol

**You are the ORCHESTRATOR. You NEVER write code. You NEVER read source files. You NEVER run verification commands.**

- ALL code is written by kit sub-agents (`.kit/tdd.sh`, `.kit/experiment.sh`, `.kit/math.sh`)
- You ONLY: read/write state `.md` files, run kit phases, check exit codes
- If you need code written → delegate to a kit phase
- If you need something verified → the kit phase already verified it (trust exit 0)
- See `CLAUDE.md` §ABSOLUTE RULES for the full list

## Project Status

**21 phases complete (9 engineering + 12 research). Branch: `feat/nested-run-visibility`. Infrastructure TDD cycle for orchestrator observability — spec written, submodule updated, TDD full cycle pending.**

## What was completed this cycle

- **Nested Run Visibility spec created** — `.kit/docs/nested-run-visibility.md`
  - Defines `child_run_spawned` event (R1), dashboard parent-child filter (R2), `kit.run_tree` MCP tool (R3), `kit.run_events` MCP tool (R4)
  - 10 exit criteria, 16 test cases (T1–T16)
  - Scope: `orchestration-kit/tools/kit`, `orchestration-kit/mcp/server.py`, `orchestration-kit/tools/dashboard`
- **Orchestration-kit submodule updated** (`d16d0cc`) — MCP server bug fixes:
  - `--run-id` swallowed by argparse REMAINDER → fixed (placed before positionals)
  - DEVNULL error blindness → fixed (capture to `runs/mcp-launches/{run_id}.log`)
  - `--reasoning` swallowed by REMAINDER → fixed
  - Missing `KIT_STATE_DIR` → fixed (auto-detect + forward to subprocesses)
- **`.orchestration-kit.env` updated** — environment config changes for new MCP tools

## What exists

A C++20 MES microstructure model suite with:
- **Infrastructure**: Bar construction, oracle replay, multi-day backtest, feature computation/export, feature analysis, oracle expectancy report, bar feature export, tick bar fix (9 TDD phases)
- **Research results**: 12 complete research phases. CNN spatial signal confirmed (proper-validation R²≈0.089). End-to-end Hybrid pipeline not viable under base costs.
- **Architecture decision**: CNN + GBT Hybrid — signal is REAL but pipeline design is the bottleneck. The regression-to-classification gap prevents viable trading.
- **Labeling decision**: Triple barrier (preferred over first-to-hit)
- **Temporal verdict**: NO TEMPORAL SIGNAL — confirmed across 7 bar types, 0.14s–300s

## Phase Sequence

| # | Spec | Kit | Status |
|---|------|-----|--------|
| 1 | `.kit/docs/bar-construction.md` | TDD | **Done** |
| 2 | `.kit/docs/oracle-replay.md` | TDD | **Done** |
| 3 | `.kit/docs/multi-day-backtest.md` | TDD | **Done** |
| R1 | `.kit/experiments/subordination-test.md` | Research | **Done** (REFUTED) |
| 4 | `.kit/docs/feature-computation.md` | TDD | **Done** |
| 5 | `.kit/docs/feature-analysis.md` | TDD | **Done** |
| R2 | `.kit/experiments/info-decomposition.md` | Research | **Done** (FEATURES SUFFICIENT) |
| R3 | `.kit/experiments/book-encoder-bias.md` | Research | **Done** (CNN WINS) |
| R4 | `.kit/experiments/temporal-predictability.md` | Research | **Done** (NO SIGNAL) |
| 6 | `.kit/experiments/synthesis.md` | Research | **Done** (CONDITIONAL GO) |
| 7 | `.kit/docs/oracle-expectancy.md` | TDD | **Done** |
| 7b | `tools/oracle_expectancy.cpp` | Research | **Done** (GO) |
| 8 | `.kit/docs/bar-feature-export.md` | TDD | **Done** |
| R4b | `.kit/experiments/temporal-predictability-event-bars.md` | Research | **Done** (NO SIGNAL — robust) |
| R4c | `.kit/experiments/temporal-predictability-completion.md` | Research | **Done** (CONFIRMED — all nulls) |
| R4d | `.kit/experiments/temporal-predictability-dollar-tick-actionable.md` | Research | **Done** (CONFIRMED) |
| 9A | `.kit/docs/hybrid-model.md` | TDD | **Done** (C++ TB label export) |
| 9B | `.kit/experiments/hybrid-model-training.md` | Research | **Done (REFUTED)** — normalization wrong |
| 9C | `.kit/experiments/cnn-reproduction-diagnostic.md` | Research | **Done (REFUTED)** — deviations not root cause |
| 9D | `.kit/experiments/r3-reproduction-pipeline-comparison.md` | Research | **Done (CONFIRMED/REFUTED)** — root cause resolved |
| R3b | `.kit/experiments/r3b-event-bar-cnn.md` | Research | **Done (INCONCLUSIVE)** — bar defect |
| TB-Fix | `.kit/docs/tick-bar-fix.md` | TDD | **Done** — tick bars fixed |
| **9E** | **`.kit/experiments/hybrid-model-corrected.md`** | **Research** | **Done (REFUTED — Outcome B)** — CNN R²=0.089, exp=-$0.37 |
| **NRV** | **`.kit/docs/nested-run-visibility.md`** | **TDD** | **In progress** — spec written, submodule updated, TDD cycle pending |

## Test summary

- **1003 unit tests** pass, 1 disabled, 1 skipped, 1004 total + tick_bar_fix tests
- **22 integration tests** (14 N=32 + 8 N=128) — labeled `integration`, excluded from default ctest
- Unit test time: ~14 min. Integration: ~20 min.

## What to do next

### 1. Run NRV TDD Cycle (IMMEDIATE)
Run `kit.tdd` with `spec_path=".kit/docs/nested-run-visibility.md"`. This implements:
- `child_run_spawned` event in parent's events.jsonl (R1)
- Dashboard `/api/runs?parent_run_id=X` filter (R2)
- `kit.run_tree` MCP tool — full execution tree with status (R3)
- `kit.run_events` MCP tool — tail last N events (R4)
- MCP bug fixes already in submodule (R5/EC-5)

Files to be modified by TDD sub-agent:
- `orchestration-kit/tools/kit` (child_run_spawned event emission)
- `orchestration-kit/mcp/server.py` (new MCP tools + bug fixes)
- `orchestration-kit/tools/dashboard` (parent-child query)
- New test file(s)

### 2. After NRV: Model Pipeline Work
- End-to-end CNN classification (highest priority)
- XGBoost hyperparameter tuning
- Label design sensitivity
- CNN at h=1
- Tick_100 multi-seed replication

## Key research results

| Experiment | Finding | Key Number |
|-----------|---------|------------|
| R1 | Subordination refuted for MES | 0/3 primary tests significant |
| R2 | Hand-crafted features sufficient | Best R²=0.0067 (1-bar MLP) |
| R3 | CNN best book encoder | R²=0.132 (leaked), proper R²≈0.084 |
| R4 | No temporal signal (time_5s) | All 36 AR configs negative R² |
| R4b | No temporal signal (event bars) | All AR configs negative R² |
| R4c | Tick + extended horizons null | 0/54+ passes across 4 bar types |
| R4d | Dollar + tick actionable null | 0/38 passes, 7s–300s coverage |
| Synthesis | CNN + GBT Hybrid | CONDITIONAL GO → GO |
| Oracle | TB passes all 6 criteria | $4.00/trade, PF=3.30 |
| 9B | CNN normalization wrong | R²=-0.002 |
| 9C | Deviations not root cause | R²=0.002 |
| 9D | R3 reproduced, root cause resolved | R²=0.1317 (leaked) / 0.084 (proper) |
| R3b | Tick bars are time bars | Peak R²=0.057, all < baseline |
| **9E** | **CNN viable, pipeline not** | **R²=0.089, exp=-$0.37/trade** |

## Build commands

```bash
cmake --build build -j12                                                  # build
cd build && ctest --output-on-failure --label-exclude integration         # unit tests (~14 min)
cd build && ctest --output-on-failure --label-regex integration           # integration tests (~20 min)
./build/oracle_expectancy                                                  # oracle expectancy extraction
./build/bar_feature_export --bar-type <type> --bar-param <param> --output <csv>  # feature export
```

---

Updated: 2026-02-20 (Nested Run Visibility — spec + submodule setup. Branch `feat/nested-run-visibility`. TDD cycle pending: 10 exit criteria, 16 tests.)
