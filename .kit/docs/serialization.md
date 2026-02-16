# Phase 5: Model Serialization

## Goal

Verify that all trained models can be saved and reloaded with identical outputs. This is the final validation gate before the spec is considered complete.

## Requirements

### 1. libtorch Checkpoint Save/Load (MLP and CNN)

- After training a model to overfit on N=32 synthetic samples, save the model state using `torch::save()`.
- Reload into a fresh model instance using `torch::load()`.
- Run a forward pass on the same N=32 batch with both the original and reloaded model.
- Assert: outputs are **bitwise identical** (not just close — exact match via `torch::equal()`).

### 2. ONNX Export (MLP and CNN)

- Export each trained libtorch model to ONNX format using `torch::onnx::export()` or equivalent.
- Load the ONNX model using ONNX Runtime C++ API (`Ort::Session`).
- Run inference on the same N=32 batch.
- Assert: ONNX output matches libtorch output within tolerance (`atol=1e-4`, `rtol=1e-4`).
- Note: `atol=1e-5` is too tight for float32 after multiple layers; 1e-4 is standard.

### 3. XGBoost Save/Load (GBT)

- After training GBT to overfit on N=32 samples, save using `XGBoosterSaveModel()`.
- Reload into a fresh booster using `XGBoosterLoadModel()`.
- Run predictions on the same N=32 batch.
- Assert: predictions are **identical** (exact match, integer class labels).

### 4. SSM (Skipped)

- SSM requires CUDA + Python. No GPU available. Skip entirely.
- Do NOT attempt ONNX export for SSM even if it were available (custom CUDA kernels).

## Test Structure

Write tests in `tests/serialization_test.cpp`:

- `SerializationTest.MLPCheckpointRoundTrip` — save/load MLP, verify identical outputs
- `SerializationTest.CNNCheckpointRoundTrip` — save/load CNN, verify identical outputs
- `SerializationTest.MLPOnnxExport` — export MLP to ONNX, verify within tolerance
- `SerializationTest.CNNOnnxExport` — export CNN to ONNX, verify within tolerance
- `SerializationTest.GBTSaveLoadRoundTrip` — save/load GBT, verify identical predictions

Each test should:
1. Create a model with deterministic seed (SEED=42)
2. Train to overfit on N=32 synthetic samples (use `test_helpers.hpp` utilities)
3. Save the model to a temp file
4. Load into a fresh instance
5. Compare outputs

## Implementation Notes

- Use `std::filesystem::temp_directory_path()` for temp files, clean up after each test.
- For ONNX: if ONNX Runtime is not available as a dependency, the ONNX tests should be conditionally compiled (CMake option). If adding onnxruntime via FetchContent is too heavy, an acceptable alternative is to export the ONNX file and verify it loads (schema validation) without full inference comparison. Document the choice.
- The GBT round-trip uses `XGBoosterSaveModel` / `XGBoosterLoadModel` (XGBoost C API) — these are already available from the existing xgboost dependency.
- Models should be trained just enough to have non-trivial weights (a few epochs suffice — we're testing serialization, not convergence).

## Acceptance Criteria

- [ ] MLP checkpoint save/load produces bitwise identical forward pass
- [ ] CNN checkpoint save/load produces bitwise identical forward pass
- [ ] MLP ONNX export matches libtorch within atol=1e-4 (or ONNX file validates if no runtime)
- [ ] CNN ONNX export matches libtorch within atol=1e-4 (or ONNX file validates if no runtime)
- [ ] GBT save/load produces identical predictions
- [ ] All tests pass in ctest
- [ ] No new test regressions (existing 204 tests still pass)

## Validation Gate (from ORCHESTRATOR_SPEC.md)

```
Assert: all save/load round-trips produce identical outputs
Assert: ONNX models (MLP, CNN) match libtorch within tolerance (atol=1e-4, rtol=1e-4)
Assert: GBT predictions match after reload
```
