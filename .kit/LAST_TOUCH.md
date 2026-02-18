# Last Touch — Cold-Start Briefing

## Project Status

**13 phases complete (8 engineering + 5 research), R4b in progress.** Oracle expectancy extracted on 19 real MES days. **VERDICT: GO.** Triple barrier passes all 6 success criteria. R4b testing temporal predictability on event-driven bars.

## What was completed this cycle

- **R4b Phase A**: TDD sub-cycle for `bar_feature_export` tool — PR #14 shipped, 1003 tests pass
- **Bug fix**: StreamingBookBuilder was not populating `snap.trades[]` — volume/dollar bar builders got `has_trade=false` for every snapshot. Added `fill_trades()` to `emit_snapshot()` in `tools/bar_feature_export.cpp`
- **R4b Phase B**: Volume_100 feature export — 115,661 bars (matches R1 ~116K), CSV schema validated
- **R4b Phase C**: Dollar_25k feature export — 3,124,720 bars (matches R1 ~3.1M), CSV schema validated
- **R4b Phase D**: R4 Python script parameterized (`--input-csv`, `--output-dir`, `--bar-label`). Volume_100 R4 complete: **NO TEMPORAL SIGNAL** (all 36 Tier 1 AR configs negative R²)
- **R4b Phase E (in progress)**: Dollar_25k R4 analysis running in background (bash task `be954cb`). Early Tier 1 results show **positive R²** at h=1 and h=5

## Dollar_25k early Tier 1 results (partial — still running)

| Config | R² | p>0 |
|--------|-----|-----|
| AR-10_linear_h1 | +0.000633 | 0.0012 |
| AR-10_gbt_h1 | +0.000373 | 0.0131 |
| AR-10_linear_h5 | +0.000364 | 0.0210 |
| AR-50_linear_h1 | +0.000584 | 0.0030 |
| AR-50_gbt_h1 | +0.000429 | 0.0067 |
| AR-10_h20 | ~0 | ns |
| AR-10_h100 | negative | ns |

Signal is at h=1 and h=5 only (short horizons). Linear ≈ GBT (signal appears linear, not nonlinear).

## What exists

A C++20 MES microstructure model suite with:
- **Infrastructure**: Bar construction, oracle replay, multi-day backtest, feature computation/export, feature analysis, oracle expectancy report, bar feature export (8 TDD phases)
- **Research results**: Subordination test, info decomposition, book encoder bias, temporal predictability, synthesis, oracle expectancy, R4b event bars (7 research phases, R4b partial)
- **Architecture decision**: CNN + GBT Hybrid — Conv1d on raw (20,2) book -> 16-dim embedding -> concat with ~20 non-spatial features -> XGBoost
- **Labeling decision**: Triple barrier (preferred over first-to-hit)

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
| R4b | `.kit/experiments/temporal-predictability-event-bars.md` | Research | **In Progress** |

## Test summary

- **1003 unit tests** pass, 1 disabled, 1 skipped, 1004 total
- **22 integration tests** (14 N=32 + 8 N=128) — labeled `integration`, excluded from default ctest
- Unit test time: ~14 min. Integration: ~20 min.

## What to do next

### Immediate: Complete R4b

1. **Check if dollar_25k R4 finished**: Look for `.kit/results/temporal-predictability-event-bars/dollar_25k/metrics.json`. If not, rerun:
   ```bash
   python3 research/R4_temporal_predictability.py \
     --input-csv .kit/results/temporal-predictability-event-bars/dollar_25k/features.csv \
     --output-dir .kit/results/temporal-predictability-event-bars/dollar_25k \
     --bar-label dollar_25k
   ```

2. **Phase F**: Write comparison analysis — see spec § "Step 6" and § "Decision Framework"

3. **Update state files**: Check off exit criteria, update RESEARCH_LOG.md, CLAUDE.md

### After R4b: Model architecture build spec

## Resume protocol for R4b

See `.kit/experiments/temporal-predictability-event-bars.md` § "Resume Protocol".

| Check | File | If exists |
|-------|------|-----------|
| Volume_100 R4 done? | `.kit/results/temporal-predictability-event-bars/volume_100/metrics.json` | Done |
| Dollar_25k R4 done? | `.kit/results/temporal-predictability-event-bars/dollar_25k/metrics.json` | Skip to Phase F |
| Comparison done? | `.kit/results/temporal-predictability-event-bars/analysis.md` | Review for completeness |

## Key research results

| Experiment | Finding | Key Number |
|-----------|---------|------------|
| R1 | Subordination refuted for MES | 0/3 primary tests significant |
| R2 | Hand-crafted features sufficient | Best R²=0.0067 (1-bar MLP) |
| R3 | CNN best book encoder | R²=0.132, CNN>Attention p=0.042 |
| R4 | No temporal signal (time_5s) | All 36 AR configs negative R² |
| R4b vol100 | No temporal signal (volume bars) | All 36 AR configs negative R² |
| R4b dollar25k | **Positive AR R² at h=1, h=5** | AR-10 linear h1: R²=+0.000633 (partial) |
| Synthesis | CNN + GBT Hybrid | CONDITIONAL GO -> **GO** |
| Oracle Expectancy | TB passes all 6 criteria | $4.00/trade, PF=3.30 |

## Build commands

```bash
cmake --build build -j12                                                  # build
cd build && ctest --output-on-failure --label-exclude integration         # unit tests (~14 min)
cd build && ctest --output-on-failure --label-regex integration           # integration tests (~20 min)
./build/oracle_expectancy                                                  # oracle expectancy extraction
./build/bar_feature_export --bar-type <type> --bar-param <param> --output <csv>  # feature export
```

---

Updated: 2026-02-17 (R4b phases A-D complete, E in progress)
