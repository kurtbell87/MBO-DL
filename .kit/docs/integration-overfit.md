# integration_overfit — TDD Spec

## Summary

End-to-end integration test that runs the full pipeline on real MBO data and verifies all 3 C++ models can overfit N=32 samples to >=99% accuracy. This is the final validation that the entire pipeline is correct.

## Pipeline

```
glbx-mdp3-20220103.mbo.dbn.zst (MESM2, instrument_id=13615)
  → book_builder (RTH 09:30-10:00 ET, ~18000 snapshots)
  → feature_encoder (194-dim per snapshot)
  → trajectory_builder + oracle_labeler (position-dependent labels)
  → subsample N=32 (evenly spaced, k = floor(trajectory_length / 32))
  → label distribution check (>=3 of 4 classes, no class >80%, no class 4)
  → overfit MLP (<=500 epochs, target >=99%)
  → overfit CNN (<=500 epochs, target >=99%)
  → overfit GBT (<=1000 rounds, target >=99%)
```

## Data

- **File**: `DATA/GLBX-20260207-L953CAPU5B/glbx-mdp3-20220103.mbo.dbn.zst`
- **Instrument**: MESM2, `instrument_id = 13615`
- **Session**: RTH 09:30:00.000 – 10:00:00.000 ET (30-minute window)
- **Expected**: ~18,000 snapshots at 100ms intervals

## Configuration

### Book Builder
- `instrument_id`: 13615
- Session: 09:30-10:00 ET
- Warm-up from 09:29:00

### Oracle
- `horizon`: 100 (10 seconds)
- `target_ticks`: 10 (2.50 points)
- `stop_ticks`: 5 (1.25 points)
- `take_profit_ticks`: 20 (5.00 points)
- `tick_size`: 0.25

### Sampling
- `N_overfit`: 32
- Method: every k-th window, `k = floor(trajectory_length / 32)`
- If label distribution fails, shift offset by `k/2` and retry

### Training (Neural)
- Optimizer: Adam, lr=1e-3, weight_decay=0
- Batch size: 32
- Loss: CrossEntropyLoss
- Gradient clipping: max_norm=1.0
- Dropout: 0.0
- Seed: 42
- Max epochs: 500

### Training (GBT)
- XGBoost multi:softmax, num_class=5
- max_depth=10, learning_rate=0.1, n_estimators=1000
- subsample=1.0, colsample_bytree=1.0
- seed=42

## Success Criteria

| Model | Target | Max Iterations |
|-------|--------|---------------|
| MLP | >=99% accuracy | 500 epochs |
| CNN | >=99% accuracy | 500 epochs |
| GBT | >=99% accuracy | 1000 rounds |

Additionally:
- No NaN/Inf in any loss or gradient
- Loss monotonically decreases (on average over 10-epoch windows) for first 100 epochs (neural)
- No class collapse (accuracy stuck at majority class frequency for >50 epochs)
- All snapshots within RTH window
- mid_price > 0, spread >= 0 for all snapshots
- No crossed book (best_bid < best_ask for all snapshots)

## Implementation

A single `overfit_harness` executable that:

1. Reads the `.dbn.zst` file via book_builder
2. Validates snapshot quality (non-empty, within RTH, no crossed book)
3. Encodes features via feature_encoder
4. Builds trajectory via trajectory_builder
5. Subsamples N=32 with label distribution validation
6. Trains and evaluates MLP, CNN, GBT
7. Reports per-model: final accuracy, final loss, epochs/rounds to target, pass/fail
8. Exit 0 if all models pass, exit 1 if any fail

## File Layout

```
src/
  overfit_harness.cpp    # Main executable
tests/
  integration_overfit_test.cpp  # GTest integration test (can also be a standalone binary)
```

## Test Cases

### Data Validation
1. **File loads** — book_builder successfully reads the .dbn.zst file without error.
2. **Snapshot count** — At least 17,000 snapshots in 30-minute RTH window (expect ~18,000).
3. **Snapshot quality** — All snapshots: mid_price > 0, spread >= 0, no crossed book.
4. **Timestamps in range** — All snapshot timestamps within 09:30-10:00 ET.

### Feature Encoding
5. **Feature shape** — All encoded features have exactly 194 dimensions.
6. **No NaN in features** — All feature values are finite.

### Trajectory
7. **Trajectory length** — `trajectory_length >= 32` (sufficient for subsampling).
8. **Labels valid** — All labels in {0, 1, 2, 3}. No label == 4.
9. **Label distribution** — >=3 of 4 classes present, no class >80%.

### Overfit
10. **MLP overfit** — Reaches >=99% accuracy within 500 epochs on N=32.
11. **CNN overfit** — Reaches >=99% accuracy within 500 epochs on N=32.
12. **GBT overfit** — Reaches >=99% accuracy within 1000 rounds on N=32.
13. **No NaN during training** — No NaN/Inf in any loss value during any model's training.
14. **Deterministic** — Two runs with seed=42 produce identical final accuracy for all models.
