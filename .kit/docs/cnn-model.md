# cnn_model — TDD Spec

## Summary

C++ libtorch CNN model that treats the order book as a structured spatial signal per timestep, then convolves temporally. Uses the price ladder construction from §5.3 of the spec.

## Architecture

```
Input: (B, 600, 194)

Step 1 — Split features per timestep using index constants:
  book_spatial: (B, 600, 20, 2) — price ladder (price_delta, size_norm)
  trade_features: (B, 600, 150) — features[40:190]
  scalar_features: (B, 600, 4) — features[190:194]

Step 2 — Spatial convolution (per timestep, shared weights):
  Reshape book_spatial → (B*600, 2, 20) — 2 channels over 20 levels
  → Conv1d(in=2, out=32, kernel=3, padding=1) → ReLU
  → Conv1d(in=32, out=64, kernel=3, padding=1) → ReLU
  → AdaptiveAvgPool1d(1) → (B*600, 64)
  → Reshape → (B, 600, 64)

Step 3 — Concatenate:
  cat(spatial_out, trade_features, scalar_features) → (B, 600, 218)

Step 4 — Temporal convolution:
  Permute → (B, 218, 600)
  → Conv1d(in=218, out=128, kernel=5, padding=2) → ReLU
  → Conv1d(in=128, out=256, kernel=5, padding=2) → ReLU
  → AdaptiveAvgPool1d(1) → (B, 256)

Step 5 — Classification:
  → Linear(256, 5) → logits
```

## Price Ladder Construction

Per timestep, construct a 20-level price ladder:
```
[bid[9], bid[8], ..., bid[1], bid[0], ask[0], ask[1], ..., ask[8], ask[9]]
```
Each level has 2 channels: (price_delta, size_norm).
- Bids reversed: deepest bid at index 0, best bid at index 9
- Asks normal: best ask at index 10, deepest ask at index 19

Uses feature index constants from `feature_encoder.hpp`:
- Bid prices: [BID_PRICE_BEGIN:BID_PRICE_END] = [0:10]
- Bid sizes: [BID_SIZE_BEGIN:BID_SIZE_END] = [10:20]
- Ask prices: [ASK_PRICE_BEGIN:ASK_PRICE_END] = [20:30]
- Ask sizes: [ASK_SIZE_BEGIN:ASK_SIZE_END] = [30:40]
- Trades: [40:190]
- Scalars: [190:194]

## Interface

```cpp
class CNNModel : public torch::nn::Module {
public:
    CNNModel(int num_classes = 5);
    torch::Tensor forward(torch::Tensor x);  // x: (B, 600, 194) → (B, 5)
};
```

## Training

Same training loop as MLP (reuse `training_loop.hpp`):
- Adam, lr=1e-3, weight_decay=0, batch_size=32
- CrossEntropyLoss, gradient clipping max_norm=1.0
- Dropout=0.0 for overfit test
- Max 500 epochs, target >=99% accuracy on N=32

## Dependencies

- libtorch (already available from MLP phase)
- `feature_encoder.hpp` (index constants)
- `training_loop.hpp` (overfit infrastructure)
- GTest

## File Layout

```
src/
  cnn_model.hpp    # CNNModel class declaration
  cnn_model.cpp    # Implementation
tests/
  cnn_model_test.cpp  # GTest unit tests
```

## Test Cases

1. **Output shape** — Input (B=4, 600, 194), output (4, 5).
2. **Forward pass no crash** — Random input, forward completes without error.
3. **Price ladder construction** — Known feature vector, verify spatial tensor has correct layout (bids reversed, asks normal).
4. **Spatial path output** — After spatial conv + pool, verify shape is (B, 600, 64).
5. **Concatenation** — Verify cat produces (B, 600, 218) = 64 + 150 + 4.
6. **Temporal path output** — After temporal conv + pool, verify shape is (B, 256).
7. **Gradient flow** — Forward + backward, all parameters have non-zero gradients.
8. **No NaN in output** — Random input, verify finite outputs.
9. **Loss decreases** — 10 epochs on synthetic data, loss decreases.
10. **Deterministic** — Two runs with seed=42, identical loss at epoch 10.
11. **Overfit synthetic** — 32 synthetic samples, train, verify >= 95% accuracy within 500 epochs.
12. **Checkpoint save/load** — Save, load, verify identical predictions.
