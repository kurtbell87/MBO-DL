# tests/ — MBO-DL Test Suite (GTest)

## Unit tests (220 pass, 1 disabled)

| File | Tests for |
|------|-----------|
| `book_builder_test.cpp` | BookBuilder + BookSnapshot |
| `feature_encoder_test.cpp` | 194-dim feature encoding |
| `oracle_labeler_test.cpp` | Oracle labeling logic |
| `trajectory_builder_test.cpp` | Trajectory construction + sampling |
| `mlp_model_test.cpp` | MLP architecture + overfit |
| `cnn_model_test.cpp` | CNN architecture + overfit |
| `gbt_features_test.cpp` | GBT hand-crafted features |
| `gbt_model_test.cpp` | XGBoost wrapper + overfit |
| `serialization_test.cpp` | Checkpoint save/load + ONNX export |
| `test_helpers.hpp` | Shared test utilities |

## Integration tests (22, excluded from default ctest)

| File | Tests for |
|------|-----------|
| `integration_overfit_test.cpp` | Full pipeline on real MBO data, N=32 overfit for MLP/CNN/GBT (14 tests) |
| `n128_overfit_test.cpp` | N=128 overfit validation for MLP/CNN/GBT ≥95% accuracy (8 tests) |

Run integration tests explicitly: `cd build && ctest --output-on-failure --label-regex integration`

## Running

```bash
cmake --build build -j12
cd build && ctest --output-on-failure --label-exclude integration   # unit tests only (~5 min)
cd build && ctest --output-on-failure --label-regex integration     # integration tests only (~20 min)
```
