# Survey: R3b — CNN Spatial Predictability on Event Bars

## Prior Internal Experiments

### Directly Relevant

1. **R3 (book-encoder-bias):** CNN R²=0.132 on time_5s (20,2) book snapshots. Conv1d significantly outperforms Attention (corrected p=0.042, d=1.86) and MLP (d=0.86, not significant after correction). 16-dim embedding is information-amplifying (4.16× R² retention vs raw-40d). **Caveat:** R²=0.132 includes ~36% inflation from test-as-validation leakage. Proper-validation R²=0.084 (9D reproduction).

2. **9D (r3-reproduction-pipeline-comparison):** Confirmed R3 reproduces perfectly (mean R²=0.1317, Δ=-0.0003). Proper validation drops mean to R²=0.084. Fold 3 (October 2022) collapses to R²=-0.047 under proper validation. Root cause of 9B/9C failure fully identified: missing TICK_SIZE normalization + per-day z-scoring.

3. **R4b (temporal-predictability-event-bars):** Dollar_25k Static-HC R²=0.080 at h=1, Static-Book R²=0.067 at h=1 — **10-25× higher than time_5s** using flat GBT features. This is the strongest prior evidence that event-bar book snapshots are more predictive than time-bar snapshots. However, these were flat features through XGBoost, NOT CNN on structured (20,2) input. The CNN's spatial advantage (local book gradients) has never been tested on event bars.

4. **R4d (temporal-predictability-dollar-tick-actionable):** Empirical calibration table for MES 2022 bar durations. Key data points:
   - tick_500: 50.0s median, 7,923 bars (19d), 417 bars/session
   - tick_3000: 300.0s median, 513 bars (19d), 27 bars/session (marginal feasibility)
   - dollar_$5M: 7.0s median, 47,865 bars
   - dollar_$10M: 13.9s median, 23,648 bars
   - Volume-math overestimates duration by systematic 4× factor.

5. **R4c (temporal-predictability-completion):** Tested tick_50, tick_100, tick_250 bars. Tick_100 ≈ 10s, tick_250 ≈ 25s. All tested for temporal signal only; spatial/CNN signal was never evaluated.

### Not Directly Relevant but Informative

6. **R1 (subordination-test):** Event bars do NOT produce more Gaussian returns than time bars. However, R1 tested return distribution properties, not CNN learnability — a fundamentally different question.

7. **R2 (info-decomposition):** Book snapshot is sufficient statistic at 5s scale. Flat MLP R²=0.007 on (40d) raw book. The 12× gap between CNN (0.084) and flat MLP (0.007) demonstrates the value of spatial encoding.

8. **9B (hybrid-model-training):** CNN R²=-0.002 with broken normalization. Now understood as pipeline bug, not signal absence.

### Key Gap This Experiment Fills

**CNN on structured (20,2) book input has ONLY been tested on time_5s bars.** R4b showed flat-feature R²=0.080 on dollar_25k bars — dramatically higher than time_5s flat R²=0.003-0.008. If the CNN spatial advantage (12× over flat) transfers to event bars, CNN R² on event bars could be substantially higher than 0.084. But this has never been measured.

## Current Infrastructure

### C++ Data Pipeline
- **`build/bar_feature_export`** — Compiled C++ binary that constructs bars from raw `.dbn.zst` MBO data, computes features, and exports CSV. Already used for time_5s, dollar_25k, volume_100, tick_50, tick_100, tick_250, tick_500, tick_3000, dollar_5M, dollar_10M, dollar_50M exports across R4b/R4c/R4d experiments.
- **Source:** `tools/bar_feature_export.cpp`
- **Bar types supported:** time, tick, volume, dollar (via BarFactory)
- **Output format:** Flat CSV with 40 book columns (20 price offsets + 20 sizes), non-spatial features, and forward returns.
- **19 selected days** hardcoded in export tool (same across all experiments).
- **Data source:** `DATA/GLBX-20260207-L953CAPU5B/glbx-mdp3-YYYYMMDD.mbo.dbn.zst` (312 daily files, ~49 GB).

### Python CNN Training
- **`scripts/hybrid_model/cnn_encoder_r3.py`** — R3-exact CNN encoder (Conv1d 2→59→59, 12,128 params). New file (untracked in git).
- **`scripts/hybrid_model/run_r3_reproduction.py`** — R3 reproduction script used by 9D. Implements proper validation (80/20 train/val split).
- **`scripts/hybrid_model/run_corrected_experiment.py`** — Corrected hybrid pipeline. New file (untracked).
- **`scripts/hybrid_model/cnn_encoder.py`** — Original (broken) CNN encoder from 9B.
- **`scripts/hybrid_model/run_experiment.py`** — Original 9B experiment runner.
- **`scripts/hybrid_model/run_cnn_diagnostic.py`** — 9C diagnostic runner.

### Existing Data Files
- **`.kit/results/hybrid-model/time_5s.csv`** — 87,970 bars × 19 days. The time_5s baseline data. Byte-identical to R3's features.csv.
- **No tick-bar CSVs exist for the thresholds needed by R3b.** Prior R4 experiments exported event-bar CSVs to their own result directories, but these contained temporal features, not the (20,2) book structure needed for CNN training. Need to verify whether the existing export format matches what the CNN expects.

### Corrected Normalization Protocol (Verified)
From `.kit/results/hybrid-model-corrected/step1_cnn/normalization_verification.txt`:
- Channel 0: raw values / TICK_SIZE (0.25) → tick offsets in [-22.5, 22.5]. Note: only 7.2% are integer-valued (this surprised me but was confirmed as correct in 9D — the offsets are fractional because book levels don't always align to tick boundaries relative to mid).
- Channel 1: log1p(size), z-scored per day → mean=0.0, std=1.0 per day.
- Architecture: 12,128 params exactly. Conv1d(2→59→59) + BN + ReLU × 2 → AdaptiveAvgPool → Linear(59→16→1).

### Expanding-Window CV Structure
| Fold | Train Days | Test Days |
|------|-----------|-----------|
| 1 | Days 1–4 | Days 5–7 |
| 2 | Days 1–7 | Days 8–10 |
| 3 | Days 1–10 | Days 11–13 |
| 4 | Days 1–13 | Days 14–16 |
| 5 | Days 1–16 | Days 17–19 |

Validation split: 80/20 of train days for early stopping. Test set never seen during training.

### Hybrid-Model-Corrected (In Progress)
The corrected hybrid experiment (`.kit/experiments/hybrid-model-corrected.md`) has started — normalization and architecture are verified, but no CNN training results exist yet. Only `step1_cnn/normalization_verification.txt` and `step1_cnn/architecture_verification.txt` are present. This experiment is a prerequisite/parallel track to R3b — it confirms the corrected pipeline produces R²≈0.084 on time_5s before switching bar types.

## Known Failure Modes

1. **TICK_SIZE normalization omission (FATAL).** Three prior experiments failed because prices were not divided by 0.25. This was the root cause of the 0.132→0.002 collapse in 9B/9C. The corrected protocol is now well-documented but must be re-verified for each new data export.

2. **Per-fold vs per-day z-scoring.** R3 used per-day z-scoring on sizes. 9B used per-fold z-scoring. The difference matters because intraday volume patterns vary dramatically across days.

3. **Test-as-validation leakage.** R3's original R²=0.132 included ~36% inflation from using test data for early stopping. Proper validation (80/20 train/val split) produces R²=0.084. All new experiments MUST use proper validation.

4. **Fold 3 regime vulnerability.** October 2022 (fold 3 test days 11-13) consistently produces weak or negative CNN R² across experiments. Under proper validation, fold 3 R²=-0.047 on time_5s. This is likely a market regime effect. Event bars may or may not attenuate this — it's one of R3b's hypotheses (H4).

5. **Bar count variation across thresholds.** Event bars produce dramatically different bar counts per day depending on time of day. Morning/close sessions may have 10× more bars than midday. Very large tick thresholds (tick_3000: 27 bars/session) produce too few bars for reliable 5-fold CV.

6. **Forward return definition changes with bar type.** fwd_return_5 on tick_500 bars spans ~250s; on tick_2000 bars it spans ~10min. The prediction task's meaning changes with bar size. The CNN's spatial signal may be scale-dependent — it's unclear whether the book snapshot at a 50s-bar close has more or less predictive structure than at a 5s-bar close.

7. **Export schema compatibility.** The spec assumes the same CSV schema for all bar types. Need to verify that `bar_feature_export` produces identical column layouts for tick bars as for time bars. Any mismatch silently breaks the training pipeline.

## Key Codebase Entry Points

| File | Description |
|------|-------------|
| `tools/bar_feature_export.cpp` | C++ tool that constructs bars + features → CSV. Supports time, tick, volume, dollar bar types. |
| `build/bar_feature_export` | Compiled binary. Ready to use. |
| `scripts/hybrid_model/cnn_encoder_r3.py` | R3-exact CNN architecture (12,128 params). Used by 9D reproduction. |
| `scripts/hybrid_model/run_r3_reproduction.py` | Full 5-fold CNN training with proper validation. The template for R3b's training loop. |
| `scripts/hybrid_model/run_corrected_experiment.py` | Corrected hybrid pipeline (CNN + XGBoost). Newer than run_r3_reproduction.py. |
| `.kit/results/hybrid-model/time_5s.csv` | Baseline time_5s data (87,970 bars, 19 days). |
| `.kit/results/hybrid-model-corrected/step1_cnn/` | Normalization and architecture verification for corrected pipeline. |

## Architectural Priors

### Why CNN on Structured Book Input
The MES order book has spatial structure: adjacent price levels share predictive patterns (bid/ask pressure gradients, queue imbalance, near-touch liquidity clustering). Conv1d with kernel_size=3 captures local gradients across the 20-level depth. R3 proved this: CNN R²=0.084 vs MLP R²=0.007 on the same data — a 12× advantage from preserving spatial adjacency.

### Why Event Bars Might Help
Time bars sample the book at fixed clock intervals regardless of market activity. A 5-second bar at 9:31 AM (50+ trades) and at 12:15 PM (2 trades) look the same to the CNN, but their information content differs by an order of magnitude. Event bars (tick, volume, dollar) normalize by activity — each bar represents the same amount of market participation. This should produce:
- **More homogeneous prediction tasks** — the CNN learns one "type" of book state rather than mixing active and quiet regimes.
- **Potentially stronger spatial patterns** — if book structure after 500 ticks of activity has more consistent spatial gradients than book structure after an arbitrary 5 seconds.

### Why Event Bars Might NOT Help (or Hurt)
- The CNN may not care about regime heterogeneity. The spatial patterns (queue imbalance, spread dynamics) may be equally strong regardless of activity level.
- Larger event bars (tick_2000+) mean the book has time to "settle" — the spatial gradients visible at bar close may be weaker (more mean-reverting) than at a fast 5s snapshot.
- Fewer bars per day at larger thresholds reduces training data, potentially increasing variance without improving signal.

### Consistency with DOMAIN_PRIORS.md
DOMAIN_PRIORS.md identifies CNN as "good at local spatial patterns in the book" and lists tick size as 0.25. The experiment is aligned with these priors — it tests whether the CNN's spatial advantage varies with bar type while keeping all other variables constant.

## External Context

The question "do event bars improve deep learning on order book data" is lightly studied in the academic literature (de Prado's "Advances in Financial Machine Learning" advocates event bars; empirical results are mixed). Key practitioners' findings:

- **Event bars for ML features:** Generally acknowledged to produce more homogeneous feature distributions. Dollar bars in particular normalize for time-of-day volume patterns.
- **CNN on order book:** Recent work (Zhang et al., DeepLOB, etc.) typically uses time-sampled book snapshots. Event-sampled book snapshots for CNN are less explored.
- **Scale sensitivity:** The optimal timescale for spatial book prediction is an open question. There's no consensus on whether fast or slow snapshots are more informative for spatial models.

The R3b experiment is a clean empirical test of a question that practitioners debate but rarely test systematically. The 4-threshold sweep is appropriate for characterizing the R² vs. bar-size curve.

## Constraints and Considerations

1. **Compute:** CPU only. CNN with 12k params trains in ~10-15s/fold on ~50k-74k rows. 4 thresholds × 5 folds = 20 CNN fits ≈ 5 minutes training. Bar construction/feature export is the bottleneck (~30-60 min per threshold × 4 thresholds).

2. **Data:** 19 trading days only. Event bars at larger thresholds may produce very few bars — tick_3000 had only 513 bars (27/session) in R4d, which was insufficient for stable 5-fold CV. The spec's abort criterion of <100 bars/day is appropriate.

3. **Calibration gap:** R4d's calibration table covers tick_500 and tick_3000, but the R3b spec needs thresholds mapping to ~5-10s, ~30-60s, ~2-5min, ~10-15min. Need an initial calibration sweep for tick thresholds between 100 and 10,000 to select exact values.

4. **The hybrid-model-corrected experiment is not yet complete.** It would provide an independent confirmation of R²≈0.084 on time_5s with the corrected pipeline before comparing against event bars. R3b's baseline of 0.084 comes from 9D, which used seed=42+fold_idx. The corrected experiment uses seed=42. If results differ, seed sensitivity is a confound.

5. **Forward return semantics change.** fwd_return_5 means "5 bars ahead" — on tick_500 bars (~50s each), that's ~250s of order flow. On time_5s bars, it's ~25s. The CNN's prediction task is qualitatively different at each bar size. This is by design (the spec explicitly chooses event-bar-denominated returns), but comparisons of absolute R² across bar sizes must account for this.

6. **Bar count uniformity.** time_5s produces exactly 4,630 bars/day. Event bars produce variable bars/day (more during active periods). The CNN training set composition will differ — more morning/afternoon samples, fewer midday. This changes the effective sample distribution.

## Recommendation

The FRAME agent should focus on:

1. **Prioritize the calibration step.** The exact tick thresholds are not yet chosen. R4d's calibration table partially covers the range, but R3b needs thresholds in the ~100-10,000 range mapped to the four target durations. Run bar construction (no features, no CNN) for a sweep of thresholds first.

2. **Run hybrid-model-corrected to completion first (or in parallel).** This confirms the corrected pipeline produces R²≈0.084 on time_5s before changing bar types. If the corrected pipeline fails on time_5s, the R3b comparison is meaningless.

3. **Use the MVE approach from the spec.** Medium threshold (fold 5 only) first. This gives a quick signal/no-signal check before committing to the full 4-threshold × 5-fold sweep.

4. **Watch for the dollar_25k anomaly.** R4b's Static-HC R²=0.080 on dollar_25k with flat features was anomalously high compared to time_5s. If the R3b spec had included a dollar bar point at matched timescale, it would test whether this anomaly extends to CNN encoding. The spec allows adding a 5th dollar-bar configuration if one lands cleanly at a tick-bar timescale — this is worth doing given the R4b evidence.

5. **The most informative outcome is the curve shape.** Four points characterizing CNN R² vs. bar size is the primary deliverable. Whether the curve is peaked (optimal scale exists), monotonic (larger/smaller is always better), or flat (bar type doesn't matter) leads to qualitatively different conclusions and different actions.
