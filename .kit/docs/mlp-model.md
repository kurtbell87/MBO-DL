# mlp_model — TDD Spec

## Summary

C++ libtorch MLP model for the overfit correctness harness. Flattens the observation window and passes through dense layers. Includes a training loop that overfits N=32 samples to >=99% accuracy.

## Architecture

```
Input: (B, 600, 194) → Flatten → (B, 116400)
→ Linear(116400, 512) → ReLU → Dropout(0.1)
→ Linear(512, 256) → ReLU → Dropout(0.1)
→ Linear(256, 128) → ReLU
→ Linear(128, 5) → logits
```

**Parameter count**: ~59.7M (dominated by first linear layer: 116400 × 512)

**Dropout override**: During overfit test, set dropout to 0.0 (call `model->eval()` is wrong — we want gradients. Instead, construct with dropout=0.0 for overfit, or use a `set_dropout` method).

## Interface

```cpp
class MLPModel : public torch::nn::Module {
public:
    MLPModel(int input_dim = 116400, int num_classes = 5, float dropout = 0.1);
    torch::Tensor forward(torch::Tensor x);  // x: (B, 600, 194) → (B, 5)
};
```

## Training Configuration (Fixed for Overfit)

- Optimizer: Adam
- Learning rate: 1e-3
- Weight decay: 0
- Batch size: 32
- Loss: CrossEntropyLoss (no class weights)
- Gradient clipping: max_norm = 1.0
- Max epochs: 500
- Seed: 42
- Dropout: 0.0

## Training Loop

```cpp
struct OverfitResult {
    float final_accuracy;
    float final_loss;
    int epochs_to_target;  // -1 if target not reached
    bool success;           // accuracy >= target
};

OverfitResult overfit_mlp(
    const std::vector<TrainingSample>& samples,  // N=32 or N=128
    int max_epochs = 500,
    float target_accuracy = 0.99f,
    float lr = 1e-3f
);
```

Loop per epoch:
1. Forward pass on full batch (N=32) or mini-batches of 32 (N=128)
2. Compute CrossEntropyLoss
3. Backward pass
4. Clip gradients (max_norm=1.0)
5. Optimizer step
6. Compute accuracy over ALL N samples
7. If accuracy >= target → stop and return success
8. Check for NaN/Inf in loss → abort

## Success Criteria

| N | Target Accuracy | Max Epochs |
|---|----------------|------------|
| 32 | >= 99% | 500 |
| 128 | >= 95% | 500 |

## Dependencies

- libtorch >= 2.1 (via CMake FetchContent or find_package)
- `feature_encoder.hpp` (FEATURE_DIM, W constants)
- `trajectory_builder.hpp` (TrainingSample struct)
- GTest for tests

## File Layout

```
src/
  mlp_model.hpp          # MLPModel class declaration
  mlp_model.cpp          # Implementation
  training_loop.hpp      # OverfitResult + overfit functions
  training_loop.cpp      # Training loop implementation
tests/
  mlp_model_test.cpp     # GTest unit tests
```

## Test Cases

### Model Architecture Tests

1. **Output shape** — Input (B=4, 600, 194), output shape is (4, 5).
2. **Forward pass no crash** — Random input tensor, forward completes without error.
3. **Parameter count** — Total parameters approximately 59.7M (within 1% tolerance).
4. **Gradient flow** — After forward + backward, all parameters have non-zero gradients.
5. **No NaN in output** — Random input produces finite output values.

### Training Loop Tests

6. **Loss decreases** — After 10 epochs on synthetic data, loss is lower than initial loss.
7. **No NaN during training** — 50 epochs on synthetic data, no NaN/Inf in loss.
8. **Gradient clipping active** — Verify gradient norms are <= 1.0 after clipping.
9. **Deterministic with seed** — Two runs with seed=42 produce identical loss at epoch 10.
10. **Accuracy computation** — Synthetic data with known labels, verify accuracy is computed correctly.

### Overfit Tests (synthetic data)

11. **Overfit tiny synthetic** — Generate 32 random samples with known labels (e.g., label = hash(sample_idx) % 5). Train MLP. Verify it reaches >= 95% accuracy within 500 epochs. (Note: random features may not reach 99% easily, so use 95% threshold for synthetic test. Real pipeline data should reach 99%.)
12. **Checkpoint save/load** — Save model, reload, verify identical predictions on same input.
