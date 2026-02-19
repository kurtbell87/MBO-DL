# Structural Differences: R3 vs C++ Pipeline

## Finding: PIPELINES ARE IDENTICAL

The raw book_snap columns in features.csv and time_5s.csv
are byte-identical. Identity rate = 1.0.
Max absolute difference = 0.0.
All per-level correlations = 1.0.

## Root Cause of R3 vs 9C Gap

The R²=0.132 → 0.002 gap is NOT caused by different data.
R3 and 9C used the SAME data. The differences are:

1. **Test-as-validation leakage (PRIMARY):**
   R3 used the test set for early stopping model selection.
   9C used proper 80/20 train/val split.

2. **Price normalization:**
   R3: divide by TICK_SIZE (0.25) → integer ticks
   9C: raw price offsets (no division)

3. **Size z-score granularity:**
   R3: z-score per DAY (each day independently)
   9C: z-score per FOLD (global across all train days)

4. **Seed:**
   R3: SEED + fold_idx (varies per fold)
   9C: fixed SEED (42, same for all folds)
