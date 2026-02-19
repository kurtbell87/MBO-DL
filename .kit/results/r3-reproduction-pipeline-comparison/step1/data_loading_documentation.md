# R3 Data Loading Documentation

## Source

R3's code: research/R3_book_encoder_bias.py
Data path: .kit/results/info-decomposition/features.csv
Library: polars (NOT Python databento)

## CRITICAL FINDING
R3 does NOT load from raw .dbn.zst files via Python databento.
R3 loads from features.csv, which is the SAME C++ bar_feature_export
used by Phases 9B and 9C (time_5s.csv).
The two files are BYTE-IDENTICAL across all 87,970 rows x 40 book columns.

## Data Loading Path

1. C++ bar_feature_export writes features.csv / time_5s.csv (identical)
2. R3 loads with polars: pl.read_csv(features.csv)
3. Extracts book_snap_0..book_snap_39 (40 columns)
4. Interleaved layout: (price_0, size_0, price_1, size_1, ..., price_19, size_19)
5. Even indices = price offsets from mid (in index points)
6. Odd indices = raw sizes (integer lot counts)

## Normalization

- Price (even indices): divide by TICK_SIZE (0.25) → integer ticks from mid
- Size (odd indices): log1p(abs(x)) → z-score PER DAY (not per fold)
- R3 uses per-day z-scoring; 9C used per-fold z-scoring

## Mid-Price Formula

- Mid-price is computed in the C++ bar_feature_export
- Price offsets = (level_price - mid_price) stored in CSV
- Both R3 and 9C use the SAME pre-computed offsets

## Level Ordering

- Rows 0-9: bids (level 0 = best bid, or deepest — depends on C++ export)
- Rows 10-19: asks (level 10 = best ask, or deepest)
- Ordering is identical between R3 and 9C (same CSV)

## Reshape

- R3: (N, 40) → reshape(-1, 20, 2) → each row has (price, size)
- Model permutes internally: (B, 20, 2) → (B, 2, 20) for Conv1d

## Validation Methodology (CRITICAL DIFFERENCE)

- R3: Uses TEST SET as validation for early stopping (data leakage!)
  train_model(model, train_X, train_y, TEST_X, TEST_Y, ...)
  Then evaluates R² on the SAME test set used for model selection.
- 9C: Uses last 20% of TRAIN data as validation (proper)
  This means R3's R²=0.132 has upward selection bias.
