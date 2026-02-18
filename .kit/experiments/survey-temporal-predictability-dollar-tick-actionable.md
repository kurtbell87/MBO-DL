# Survey: R4d — Temporal Predictability at Actionable Dollar & Tick Timescales

## Prior Internal Experiments

### R4 chain summary (R4 → R4b → R4c)

The temporal encoder investigation has been the most extensively tested question in this project. Across the chain:

| Experiment | Bar types | Timescales | Horizons | Dual passes | Verdict |
|------------|-----------|------------|----------|-------------|---------|
| R4 | time_5s | 5s | h=1,5,20,100 (5s-500s) | 0/52 | NO SIGNAL |
| R4b | volume_100, dollar_25k | ~3.9s, ~0.14s | h=1,5,20,100 | 0/48 | NO SIGNAL (robust) |
| R4c Arm 1 | tick_50 | ~5s | h=1,5,20,100 | 0/16 | NO SIGNAL |
| R4c Arm 2 | tick_100, tick_250 | ~10s, ~25s | h=1,5,20,100 | 0/32 | NO SIGNAL |
| R4c Arm 3 | time_5s | 5s | h=200,500,1000 (17-83min) | 0/6 | NO SIGNAL |

**Total: 0/154+ dual threshold passes across 6 bar types, 7 horizons, 200+ individual tests.**

### What R4c's Arm 2 actually tested (and what it missed)

R4c's calibration for dollar bars was **extrapolated from R1 bar counts**, not empirically measured. The calibration table (`calibration_table.json`) shows all dollar entries marked `"status": "extrapolated_from_R1"`. The maximum dollar threshold tested was $1M, which extrapolated to ~0.9s median duration. R4c concluded "dollar bars entirely sub-actionable" and skipped the full temporal analysis on dollar bars.

**The gap R4d addresses:** R4c never actually constructed dollar bars at thresholds above $1M. The spec's volume math suggests $5M-$10M thresholds should produce bars at 3-10s — the actionable floor. Dollar bars at $250M-$3B would target 5-30min timescales. None of these have been tested empirically or through the R4 protocol.

For tick bars, R4c tested tick_50 (~5s), tick_100 (~10s), and tick_250 (~25s), but never went beyond ~25s. Tick bars at 3,000-25,000 thresholds targeting 5-30min timescales are untested.

### R1 data on dollar bar AR degradation with threshold

From R1 `metrics.json`, in-sample AR R² decreases with increasing dollar threshold:

| Dollar threshold | Total bars (19 days) | Median daily bars | AR R² (in-sample) |
|-----------------|---------------------|-------------------|-------------------|
| $25k | 3,125,290 | 165,101 | 0.01131 |
| $50k | 2,358,449 | 124,356 | 0.00967 |
| $100k | 1,572,471 | 81,859 | 0.00713 |

R4b showed the $25k in-sample AR R² was 18x inflated: CV yields 0.0006 vs. in-sample 0.011. The declining trend with threshold strongly predicts that higher-threshold dollar bars will show even less AR structure. However, this is extrapolation — the experiment demands empirical verification.

### R4c extended horizon results (time_5s)

R4c Arm 3 tested h=200/500/1000 on time_5s and found **accelerating** degradation:

| Horizon | AR R² | Δ_temporal_book |
|---------|-------|-----------------|
| h=100 (R4) | −0.0092 | +0.0004 |
| h=200 | −0.0217 | −0.0021 |
| h=500 | −0.0713 | −0.0094 |
| h=1000 | −0.1522 | −0.118 |

At h=1000 (~83min), temporal features *cost* 0.118 R² units. The degradation is super-linear. This establishes a strong prior that extended horizons on event-driven bars will also show negative results.

---

## Current Infrastructure

### Bar Construction (Ready — no changes needed)

- `src/bars/dollar_bar_builder.hpp` — Dollar bars with configurable threshold (double) and multiplier (default 5.0 for MES). Accumulates `price × size × multiplier`, emits when ≥ threshold. Supports arbitrarily large thresholds (double precision).
- `src/bars/tick_bar_builder.hpp` — Tick bars with uint32_t threshold. Counts trades, emits at threshold. Max ~4.3B ticks (far beyond needs).
- `src/bars/bar_factory.hpp` — Factory: `BarFactory::create("dollar", threshold)` and `BarFactory::create("tick", threshold)`. Dollar uses double directly; tick casts to uint32_t.

### Feature Export (Ready — no changes needed)

- `tools/bar_feature_export.cpp` — CLI: `./bar_feature_export --bar-type dollar --bar-param 5000000 --output path.csv`
- Exports 146 columns: 6 metadata + 62 hand-crafted features + 40 book snap + 33 message summary + 4 forward returns (`return_1, return_5, return_20, return_100`) + 1 MBO count
- Forward returns in ticks (not log returns). NaN forward returns → row skipped during export.
- Processes all 19 SELECTED_DAYS with contract rollover handling.
- 50 warmup bars/day excluded via `is_warmup` flag.

**Critical limitation for sparse bars:** At very large thresholds (e.g., $3B dollar bars → ~10 bars/day), most bars will have NaN `fwd_return_100` (since i+100 > total bars). These rows are dropped, yielding near-empty CSVs for h=100. The spec correctly identifies this and restricts horizons to h={1,5,20} for operating points with <50 bars/session.

### Analysis Script (Ready — minor adaptation needed)

- `research/R4_temporal_predictability.py` — ~800 lines, parameterized with `--input-csv`, `--output-dir`, `--bar-label`.
- HORIZONS = [1, 5, 20, 100] **hardcoded** — not a CLI parameter.
- Forward returns consumed as `fwd_return_1`, `fwd_return_5`, `fwd_return_20`, `fwd_return_100` from CSV columns named `return_1`, etc.
- 5-fold expanding-window CV, Holm-Bonferroni, GBT + Ridge + Linear models.
- 21 temporal features computed per-day (no cross-day lookback).

**No changes needed** for the R4 protocol at standard horizons. The spec does not call for extended horizons (h=200/500/1000) on event-driven bars — those were R4c Arm 3 on time_5s only. R4d tests event bars at h={1,5,20,100} only.

### Forward Returns in C++ (h={1,5,20,100} only)

- `src/features/bar_features.hpp` defines `fwd_return_1/5/20/100` as float fields.
- Computed as `(close_mid[i+h] - close_mid[i]) / tick_size` — forward-looking, in ticks.
- Returns NaN when `i+h >= bars.size()`.

No C++ changes needed for R4d. Extended horizons are not part of this spec (they were already tested in R4c Arm 3 on time_5s).

### Build & Tests

- CMake build green. 1003/1004 unit tests pass.
- `bar_feature_export` target compiles and has been used successfully in R4b (dollar_25k, volume_100) and R4c (tick_50, tick_100, tick_250).

---

## Known Failure Modes

### 1. Extrapolated calibration (R4c's mistake — the reason R4d exists)

R4c's calibration table was extrapolated from R1 bar counts, not built empirically. This led to the premature conclusion that dollar bars are "entirely sub-actionable." R4d's Phase 1 **must** construct bars at each threshold empirically across all 19 days and measure actual durations. The spec is explicit: "The calibration must be empirical."

### 2. In-sample AR R² inflation (18x observed)

R1's dollar_25k AR R²=0.01131 collapsed to 0.0006 under rigorous 5-fold CV in R4b. Any single-split or in-sample evaluation is unreliable. All R4d analysis must use the same 5-fold expanding-window protocol.

### 3. Sparse bars at long timescales

At 15-30min timescales, bars per RTH session drop to ~13-26. With 19 days, total bars are ~250-500. Issues:
- h=100 lookback consumes most or all of a 13-bar session → restrict to h={1,5,20}
- 5-fold CV with <50 bars/session gives very few test-fold observations per fold
- Minimum viable: ≥100 total bars across all 19 days per threshold (spec's abort criterion)

### 4. Forward return NaN truncation at sparse thresholds

`bar_feature_export.cpp` skips rows with NaN `fwd_return_1`. At very large thresholds, the last bar of each session has NaN for all forward returns (no future bar exists). With only 10-13 bars/session, losing 1 bar is 8-10% data loss per session — acceptable but must be documented.

### 5. Feature importance ≠ signal (the "splits vs. signal" trap)

R4c confirmed: GBT allocates 22-53% of importance to temporal features that provide zero OOS R² improvement. Naive feature selection would incorrectly retain temporal features. Only rigorous dual-threshold evaluation catches this.

### 6. Dollar bar multiplier

`DollarBarBuilder` uses `multiplier=5.0` by default. The spec's thresholds ($5M, $10M, etc.) are in raw notional dollars. With the multiplier, the effective notional per trade is `price × size × 5`. The calibration sweep must account for this: if the spec says $5M, the `--bar-param` argument should be 5000000, and the builder accumulates `price × size × 5` until reaching that threshold. This is consistent with how R4b's $25k threshold was parameterized.

---

## Key Codebase Entry Points

| File | Purpose | Relevance to R4d |
|------|---------|------------------|
| `tools/bar_feature_export.cpp` | CLI data generation | Phase 1 calibration + Phase 3 feature export |
| `research/R4_temporal_predictability.py` | Full R4 analysis protocol | Phase 3 temporal analysis |
| `src/bars/dollar_bar_builder.hpp` | Dollar bar threshold + multiplier logic | Understanding calibration math |
| `src/bars/tick_bar_builder.hpp` | Tick bar threshold logic | Understanding tick calibration |
| `src/bars/bar_factory.hpp` | Factory for bar type + threshold | Interface for export tool |
| `src/features/bar_features.hpp` | 62 features + forward returns | Feature definitions |
| `.kit/results/temporal-predictability-completion/arm2_calibration/calibration_table.json` | R4c's extrapolated calibration | Reference (what R4d replaces) |
| `.kit/results/subordination-test/metrics.json` | R1 bar counts at various thresholds | Baseline for calibration validation |
| `.kit/results/temporal-predictability-event-bars/analysis.md` | R4b cross-bar analysis | Reference for dollar_25k/volume_100 results |
| `.kit/results/temporal-predictability-completion/analysis.md` | R4c full analysis | Immediate predecessor — R4d extends this |

---

## Architectural Priors

This experiment does not test a new architecture — it tests whether the existing "no temporal encoder" conclusion holds under conditions R4c did not cover. The architectural priors are unchanged:

**MES returns are martingale at all tested timescales (5s-83min).** R4c tested this on time_5s (h=1 through h=1000) and tick bars (tick_50 through tick_250). R4d extends to dollar bars at actionable timescales and tick bars at 5-30min timescales.

**Dollar bar AR structure decays with threshold.** R1 in-sample data shows AR R² drops monotonically: $25k=0.011, $50k=0.0097, $100k=0.0071. Under CV, dollar_25k drops to 0.0006. The extrapolation to actionable-timescale dollar bars ($5M+) is that AR R² will be negligible or negative.

**Tick bars are indistinguishable from time bars.** R4c showed tick_50 AR R² = −0.000187 (CV) vs. time_5s = −0.0002. The R1 tick_50 in-sample value (0.00034) was inflated. Extending tick thresholds to 3,000-25,000 will produce even sparser bars with less statistical power.

**No domain theory predicts temporal signal at these scales.** Madhavan (2000), Biais et al. (2005) on microstructure efficiency; Moskowitz et al. (2012) find momentum at monthly not intraday scales. The experiment is designed for efficient null confirmation.

---

## External Context

The R4d spec positions this as a "patch" to R4c that addresses two specific calibration gaps:

1. **Dollar bars were never tested at actionable timescales.** R4c's $1M cap produced ~0.9s bars. The volume math shows $5-10M should yield ~3-10s bars. This is a legitimate gap — while the prior strongly favors the null, "we never tested it" is different from "we tested it and it's null."

2. **Neither bar type was tested at 5-30min timescales.** R4c Arm 3 tested 17-83min on time_5s only. The spec asks whether event-driven bar construction at those timescales reveals structure time bars miss. This is the most speculative arm — R4c showed time_5s h=200/500/1000 are deeply negative, and there's no theoretical reason event bars would differ.

**Academic context:** Clark (1973) / Ané & Geman (2000) subordination theory motivates event-driven bars, but R1 already refuted the theory for MES. Dollar bars show higher AR R² at sub-second timescales (genuine microstructure) but this is non-actionable. No published research demonstrates event-driven bars producing exploitable temporal signal at minute+ timescales on liquid equity index futures.

---

## Constraints and Considerations

### Compute

The spec estimates ~17 hours total:
- Phase 1 calibration: ~4 hr (17 thresholds × 19 days, bar construction only)
- Phase 3 dollar analysis (5 operating points): ~8 hr
- Phase 3 tick analysis (3 new operating points): ~4 hr
- Phase 4 cross-timescale analysis: ~1 hr

The MVE (minimum viable experiment) prioritizes: (1) empirical calibration sweep, (2) dollar bars at 3 key timescales (5s, 5min, 30min), (3) tick bars at 5min and 15min.

### Data volume at large thresholds

| Target timescale | Dollar threshold (est.) | Bars/day (est.) | Total bars (19 days) | h=100 feasible? |
|-----------------|------------------------|-----------------|---------------------|-----------------|
| ~5s | $5-10M | ~2,000-4,000 | ~40-80k | Yes |
| ~30s | $25-50M | ~400-800 | ~8-15k | Yes |
| ~5min | $250M | ~50-80 | ~1-1.5k | Marginal (≈50 bars/session) |
| ~15min | $1B | ~20-30 | ~400-600 | No (restrict to h≤20) |
| ~30min | $3B | ~10-15 | ~200-300 | No (restrict to h≤20) |

At 30min timescales, the sample size drops to ~200-300 bars total. This is thin but viable for Tier 1 AR and restricted Tier 2 analysis. The spec's abort criterion (skip if <100 total bars) provides a safety valve.

### Reuse of R4c results

R4c already has valid results for:
- tick_100 (~10s) — 0/8 dual passes
- tick_250 (~25s) — 0/8 dual passes

These should be included in R4d's cross-timescale analysis plots but **not re-run**. The spec is explicit about this.

### Phase 1 calibration approach

The spec calls for empirical bar construction across all 19 days. Two options:
1. **Full feature export** at each threshold — expensive (~2-3hr per threshold × 17 = prohibitive)
2. **Lightweight calibration** — run `bar_feature_export` and parse only the timestamp/bar_index columns for duration stats, or build a dedicated calibration script

The R4c survey recommended option 2 (parse CSV metadata columns). Alternatively, a simple Python script could read the first few rows of each export to compute duration stats, but this still requires the full C++ bar construction. The most efficient approach is to run `bar_feature_export` at each threshold on all 19 days but only compute stats from the metadata columns (timestamp, bar_index), discarding the feature data. The export is dominated by MBO file reading + bar construction, not feature computation, so there's limited speedup from skipping features. Running all 17 thresholds sequentially at ~15-20min each is ~4-6 hours.

---

## Recommendation

### Should R4d be run?

**The case for:** R4c's dollar bar conclusion was based on extrapolation, not empirical testing. "We extrapolated and concluded sub-actionable" is methodologically weaker than "we tested empirically and confirmed the null." The calibration gap is real. If someone challenged the R4 chain's completeness, the dollar bar gap at actionable timescales is the single most defensible criticism.

**The case against:** The prior is overwhelmingly negative. R4/R4b/R4c tested 154+ configurations with zero passes. The trend in dollar bar AR R² with threshold size is monotonically declining. R4c's extended horizons on time_5s showed accelerating degradation. There is no theoretical or empirical basis for expecting a different result. Running R4d is spending ~17 hours to confirm a near-certain null.

**Recommendation:** If the goal is methodological completeness (defensible against any critic), run R4d starting with the calibration sweep (Phase 1) — this is independently valuable as documentation. If the goal is practical progress toward the CNN+GBT build phase, skip R4d and proceed to the model architecture spec. The R4 chain already provides the highest-confidence null result in this project.

### If R4d is run, the FRAME agent should focus on:

1. **Phase 1 calibration as the primary deliverable.** The empirical threshold-to-timescale mapping is useful infrastructure regardless of temporal analysis results. It documents how to configure dollar/tick bars for any future experiment at any target timescale.

2. **MVE-first execution.** Calibration sweep → select 3 dollar operating points (5s, 5min, 30min) → run R4 protocol. If all 3 return negative, the remaining operating points are redundant.

3. **Short-circuit logic.** If calibration reveals that the maximum achievable dollar bar timescale is still <5s (contradicting the volume math), then dollar bars are confirmed sub-actionable by construction and Arm 2c is skipped entirely. Document and close.

4. **Horizon restriction at sparse operating points.** For thresholds with <50 bars/session, restrict to h={1,5,20}. For <26 bars/session, restrict to h={1,5}. Document why.

5. **Integration with R4c results.** The cross-timescale analysis (Phase 4) should unify R4c tick results (tick_100, tick_250) with R4d's new tick thresholds into a single timescale response curve. This produces the definitive "AR R² vs. bar duration" plot spanning 5s to 30min for both bar types.
