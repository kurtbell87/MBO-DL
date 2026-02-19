# Survey: R4c — Temporal Predictability Completion

## Prior Internal Experiments

### R4 (time_5s) — NO TEMPORAL SIGNAL
- **36 Tier 1 AR configs**, all negative R². Best: AR-10 GBT h=1: -0.0002.
- **16 Tier 2 augmentation gaps**, 0/16 pass dual threshold. Best corrected p=0.733.
- **Temporal-Only R** at h=1: effectively zero (8.8e-7).
- MES 5s returns confirmed as martingale difference sequence at horizons 5s-500s.

### R4b (volume_100, dollar_25k) — NO SIGNAL (robust)
- **Volume_100**: All 36 Tier 1 configs negative. All Tier 2 gaps fail. Identical to time_5s pattern.
- **Dollar_25k**: Only bar type with positive Tier 1 R² (h=1: +0.000633, linear). Temporal-Only has standalone power (h=1: R²=0.012, p=0.0005). BUT temporal augmentation fails dual threshold (0/16 pass). Signal is linear (Ridge >= GBT), operates at ~140ms, redundant with static features (weighted_imbalance captures it).
- R1's dollar_25k AR R²=0.01131 was 18x inflated by in-sample bias; rigorous 5-fold CV yields 0.0006.

### R1 (subordination test) — REFUTED
- Tick bar data exists from R1. Key numbers for R4c context:
  - **tick_25**: AR R²=0.00078, mean_rank=5.29, 177,441 bars (19 days)
  - **tick_50**: AR R²=0.00034, mean_rank=6.00, 88,521 bars (19 days)
  - **tick_100**: AR R²=0.00094, mean_rank=5.43, 44,061 bars (19 days)
- Tick_50 AR R² (0.00034) is identical to time_5s (0.00034). Very strong prior: tick bars will show no temporal signal (same as time_5s/volume_100 pattern).
- Dollar bar AR R² across thresholds from R1: $25k=0.01131, $50k=0.00967, $100k=0.00713. AR R² drops as threshold increases (fewer bars, longer duration). At actionable timescales (≥5s), dollar bars will have even lower AR R² than $25k's already-negligible cross-validated 0.0006.

### R2 (info-decomposition) — FEATURES SUFFICIENT
- Temporal lookback delta on raw book: -0.006 (hurts). No encoder stage passes dual threshold.
- Book snapshot is sufficient statistic for messages at 5s scale.

### R3 (book-encoder-bias) — CNN BEST
- CNN R²=0.132 on structured (20,2) book input — spatial encoder adds value when adjacency preserved.

### R6 (synthesis) — CONDITIONAL GO
- CNN+GBT Hybrid. Temporal encoder dropped. Time_5s bars. R4c addresses remaining MI/decay gap.

### R7 (oracle-expectancy) — GO
- Triple barrier: $4.00/trade expectancy, PF=3.30, 4,873 trades over 19 days. All success criteria pass.

---

## Current Infrastructure

### Bar Construction (C++, fully implemented)
- `src/bars/tick_bar_builder.hpp` — tick bars, threshold-based closure at `tick_count >= threshold`
- `src/bars/dollar_bar_builder.hpp` — dollar bars with multiplier, closure on cumulative dollar volume
- `src/bars/volume_bar_builder.hpp` — volume bars
- `src/bars/time_bar_builder.hpp` — time bars
- `src/bars/bar_factory.hpp` — factory: `BarFactory::create("tick"|"dollar"|"volume"|"time", threshold)`
- All 4 bar types tested and production-ready.

### Feature Export (C++, production-ready)
- `tools/bar_feature_export.cpp` — CLI: `./bar_feature_export --bar-type <type> --bar-param <threshold> --output <csv>`
- Pipeline: StreamingBookBuilder -> BarFactory -> BarFeatureComputer -> FeatureExporter CSV
- Handles all 4 contract rollovers (MESH2/MESM2/MESU2/MESZ2), same 19 SELECTED_DAYS
- Output: 5 metadata + 62 Track A + 40 book_snap + 33 msg_summary + 4 forward returns
- 50 warmup bars per day excluded via `is_warmup` flag

### Feature Pipeline (C++, `src/features/`)
- `bar_features.hpp` — 62 hand-crafted features (6 categories: book shape, order flow, price dynamics, cross-scale, time context, message microstructure)
- `feature_export.hpp` — CSV header generation and per-row formatting
- `raw_representations.hpp` — Track B (book snapshots, message summaries, lookback sequences)

### Analysis Script (Python, adaptable)
- `research/R4_temporal_predictability.py` — 800+ lines, already parameterized with `--input-csv`, `--output-dir`, `--bar-label` from R4b work
- 21 temporal features: `lag_return_{1..10}`, `rolling_vol_{5,20,100}`, `vol_ratio`, `momentum_{5,20,100}`, `mean_reversion_20`, `abs_return_lag1`, `signed_vol`
- 5-fold expanding-window time-series CV, Holm-Bonferroni correction, paired permutation tests
- XGBoost, Ridge, Linear models. HORIZONS currently [1, 5, 20, 100].

### Statistical Infrastructure (C++)
- `src/analysis/statistical_tests.hpp` — Jarque-Bera, ARCH LM, ACF, Ljung-Box, AR R² via OLS
- `src/analysis/bar_comparison.hpp` — cross-bar-type comparison framework

### Build & Tests
- CMake build green. 1003/1004 unit tests pass. `bar_feature_export` target exists and compiles.
- `bar_feature_export` was built and tested as part of R4b TDD sub-cycle (spec: `.kit/docs/bar-feature-export.md`).

---

## Known Failure Modes

1. **In-sample AR R² inflation.** R1's dollar_25k AR R²=0.01131 was 18x higher than R4b's cross-validated 0.0006. Any AR R² from non-CV single-split evaluation is unreliable. R4c must use the same 5-fold CV protocol.

2. **Temporal feature importance ≠ temporal value.** R4 showed GBT allocates 30-50% importance to temporal features that provide zero marginal R². The model splits on temporal noise in-sample but this doesn't generalize. Feature importance alone is not evidence of signal.

3. **Horizon mismatch across bar types.** h=1 on dollar_25k is ~0.14s, on tick_50 is ~variable, on time_5s is 5s. Direct bar-horizon comparisons across bar types are not apples-to-apples. R4c's Arm 2 explicitly addresses this by calibrating to clock-time equivalence first.

4. **Dollar bar pathology at high frequency.** Dollar_25k produces ~165k bars/day (~3.1M total). Bars are sub-tick, capturing bid-ask bounce. Extremely non-Gaussian (JB=109M). Any temporal signal at these frequencies is microstructure noise, not tradeable.

5. **Forward return computation at long horizons.** Extended horizons (h=200/500/1000 on time_5s) require bars that extend 17-83 minutes into the future. Near end-of-session bars will have missing forward returns. The analysis script must handle NaN forward returns (truncation, not imputation). The existing script already handles h=100 (~8min), but h=1000 (~83min) will lose the last ~1000 bars per day (~200 of ~4600 RTH bars, ~4% data loss). Not fatal but should be documented.

6. **Cross-day lookback prohibition.** Temporal features use no cross-day lookback. At h=1000, the effective warm-up is 1000 bars (~83 minutes into RTH). Only ~3600 bars/day remain for analysis. This is still adequate (19 days x 3600 = ~68k training rows) but represents a meaningful reduction from R4's ~84k.

7. **HC+Temporal catastrophic degradation.** R4 showed HC+Temporal at h=20 hitting R²=-0.034 driven by a single fold at R²=-0.138. Severe overfitting. The GBT model fits 62 HC + 21 temporal = 83 features, causing curse of dimensionality at longer horizons where signal is weakest.

---

## Key Codebase Entry Points

| File | Purpose |
|------|---------|
| `tools/bar_feature_export.cpp` | Primary data generation tool for all arms |
| `research/R4_temporal_predictability.py` | Analysis script (needs horizon extension for Arm 3) |
| `src/bars/tick_bar_builder.hpp` | Tick bar construction (Arm 1) |
| `src/bars/dollar_bar_builder.hpp` | Dollar bar construction (Arm 2) |
| `src/bars/bar_factory.hpp` | Factory for selecting bar type + threshold |
| `src/features/bar_features.hpp` | 62-feature computation pipeline |
| `src/features/feature_export.hpp` | CSV export formatting |
| `.kit/results/temporal-predictability-event-bars/analysis.md` | R4b cross-bar analysis (reference) |
| `.kit/results/temporal-predictability/analysis.md` | R4 time_5s analysis (reference) |
| `.kit/results/subordination-test/metrics.json` | R1 tick bar statistics (baseline) |

---

## Architectural Priors

**This problem is: "Do temporal features add predictive value for MES microstructure?"**

The answer is already "no" for 3 of 4 bar types (time_5s, volume_100, dollar_25k) at horizons 5s-500s. R4c tests the remaining cells:

1. **Tick bars (Arm 1):** R1 measured tick_50 AR R²=0.00034 — identical to time_5s. The strong prior is NO SIGNAL. Tick bars aggregate fixed trade counts regardless of size, which is essentially a noisy version of time sampling for MES (where trade size variation is modest). No domain theory predicts tick bars would surface temporal structure that time/volume/dollar bars miss.

2. **Actionable-timescale event bars (Arm 2):** Dollar_25k's positive AR R² operates at ~140ms. Increasing the dollar threshold to produce ≥5s bars necessarily reduces the bar count and AR R². By R1's own data: $25k AR R²=0.01131, $100k AR R²=0.00713. Extrapolating, thresholds producing 5-30s bars will have AR R² near time_5s levels. The calibration step (2a) is cheap and independently valuable — it documents the threshold-to-timescale mapping. But the temporal analysis (2c) is very likely to confirm the null.

3. **Extended horizons (Arm 3):** Testing h=200/500/1000 on time_5s (17-83 minutes). This is the most interesting arm. At these horizons, we're testing regime-scale structure: momentum, mean-reversion at the session level. However, 5-fold CV with 19 days provides very few independent observations at these scales. A 1000-bar horizon spans ~83 minutes; there are only ~5 independent non-overlapping windows per session. With 19 days, that's ~95 independent observations for the longest horizon. Statistical power will be low. The prior from R4: at h=100 (~8min), all R² values are deeply negative (time_5s: -0.013, dollar_25k: -0.0003). The trend is monotonically worsening with horizon. h=200+ is extremely unlikely to reverse this trend.

**No domain theory or prior evidence supports a temporal encoder for MES at retail timescales.** The DOMAIN_PRIORS.md lists SSM (Mamba) as "best for temporal dynamics — sweep-then-reload, momentum sequences" but this is a generic capability statement, not an MES-specific prediction. The empirical evidence (R4, R4b, R2) unanimously rejects temporal structure.

---

## External Context

The finding that microstructure returns are approximately martingale at short horizons is well-established in the market microstructure literature:

- **Efficient markets at short horizons.** Madhavan (2000), Biais et al. (2005) show that for liquid instruments, return autocorrelation decays rapidly. MES (a highly liquid derivative tracking ES) is expected to be among the most efficient at sub-minute scales.
- **Event-bar temporal structure is microstructure, not alpha.** The ~140ms AR R² in dollar_25k aligns with known bid-ask bounce autocorrelation (Roll 1984). This is not exploitable signal — it's the mechanical consequence of trades alternating between bid and ask prices.
- **Temporal features at medium horizons (15-60 min).** Academic evidence for momentum at these scales on equity index futures is weak. Moskowitz et al. (2012) find time-series momentum at monthly scales, not intraday. Intraday mean-reversion is more commonly documented (Hasbrouck & Seppi 2001) but requires position sizing and transaction cost management that exceeds the scope of the current model.

The prior from external literature strongly supports the null hypothesis for all three R4c arms.

---

## Constraints and Considerations

### Compute
- **Arm 1 (tick_50):** ~3 hours. Feature export (~2.5hr) + analysis (~40min). Low cost.
- **Arm 2a (calibration):** ~3 hours. Bar construction at 13 thresholds, duration statistics only. No feature export.
- **Arm 2b-c (temporal analysis at 6 thresholds):** ~10 hours. Feature export for 3 dollar + 3 tick thresholds, then analysis.
- **Arm 3 (extended horizons):** ~1.5 hours. Reuses existing time_5s features. Only script modification (extend HORIZONS list).
- **Total: ~17.5 hours.** Spec recommends Arm 3 first (highest info value, lowest cost).

### Data
- Same 19 RTH days from R1/R2/R4/R4b. No new data needed.
- Feature export via `bar_feature_export` is proven for all 4 bar types.

### Infrastructure Gaps
- **Arm 2a calibration sweep**: No existing tool computes bar duration statistics across thresholds. This requires either: (a) a new small C++ utility that constructs bars at each threshold and reports timing stats, or (b) running `bar_feature_export` at each threshold and computing stats from the CSV metadata columns. Option (b) is simpler but slower (~3hr per threshold x 13 thresholds = prohibitive). Option (a) is the correct approach — a lightweight sweep that only builds bars and measures durations without computing features.
- **Extended HORIZONS in Python**: `R4_temporal_predictability.py` currently hardcodes `HORIZONS = [1, 5, 20, 100]`. Extending to [200, 500, 1000] requires: (1) adding these to the HORIZONS list, (2) ensuring forward return columns `fwd_return_200`, `fwd_return_500`, `fwd_return_1000` exist in the CSV, (3) handling NaN forward returns at end-of-session. The C++ `feature_export.hpp` currently computes `fwd_return_{1,5,20,100}` — it needs extension to compute forward returns at h=200/500/1000. **This is a code change to `bar_features.hpp` and `feature_export.hpp`.**

### Key Infrastructure Decision: Forward Returns for Extended Horizons

The existing C++ pipeline computes `fwd_return_1/5/20/100` in `bar_features.hpp`. For Arm 3, we need `fwd_return_200/500/1000`. Two options:

1. **Modify C++ pipeline** (TDD sub-cycle): Add fwd_return_{200,500,1000} to `BarFeatureComputer`. Requires re-export of time_5s features.
2. **Compute in Python**: Load existing time_5s CSV, compute extended forward returns from the `close` price column in Python. Avoids C++ changes, reuses existing CSV.

Option 2 is strongly preferred: no C++ change, no re-export, faster iteration. The analysis script already has access to bar-level close prices via the CSV.

---

## Recommendation

### Priority Order (matches spec's MVE section)

1. **Arm 3 first** — highest information value, lowest cost (~1.5hr). Extend HORIZONS in the Python script. Compute fwd_return_{200,500,1000} from close prices in Python. If all extended horizons show negative R² (extremely likely given the monotonically worsening trend from h=1 to h=100), the temporal encoder question is closed at all timescales up to ~83 minutes.

2. **Arm 2a second** — calibration sweep is cheap and independently valuable. Build a lightweight bar-duration-stats tool (or script) that maps threshold -> timescale. This produces the threshold-to-timescale mapping needed for Arm 2c, and is useful for documenting the operating characteristics of each bar type regardless of temporal analysis results.

3. **Arm 2c third** — if calibration reveals thresholds producing ≥5s bars, run temporal analysis at those thresholds. Expect negative results consistent with R4/R4b.

4. **Arm 1 (tick_50) may be redundant with Arm 2.** If Arm 2's tick bar calibration already selects tick_50 (or a nearby threshold) as an actionable-timescale operating point, Arm 1 adds nothing. The spec itself notes this: "Skip if Arm 2c covers tick bars adequately." Recommendation: defer Arm 1 until Arm 2a calibration is complete.

### What the FRAME agent should focus on
- Design the compute plan for the priority order above
- Specify how to compute fwd_return_{200,500,1000} in Python without C++ changes
- Specify the Arm 2a calibration approach (lightweight sweep vs. full feature export)
- Define abort criteria: if Arm 3 returns fully negative R² at all extended horizons AND Arm 2a shows no threshold produces actionable-timescale bars, should the experiment short-circuit?
- R4c is expected to confirm the null. The FRAME agent should design for efficient null confirmation, not for discovering signal.
