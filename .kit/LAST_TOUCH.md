# Last Touch — Cold-Start Briefing

## Project Status

**ORCHESTRATOR_SPEC.md is COMPLETE.** All build phases shipped. N=128 overfit validation added.

## What was built

A C++20 MES microstructure model suite that reads raw Databento MBO (L3) order data from `.dbn.zst` files and trains 3 model architectures (MLP, CNN, GBT) to intentionally overfit on N=32 and N=128 samples — proving end-to-end correctness of the entire pipeline.

```
Raw MBO (.dbn.zst) → book_builder → BookSnapshot[W=600]
  → feature_encoder → (B, 600, 194)
  → trajectory_builder + oracle_labeler → (window, label) pairs
  → MLP model  → overfit N=32 ≥99%, N=128 ≥95%  ✓
  → CNN model  → overfit N=32 ≥99%, N=128 ≥95%  ✓
  → GBT model  → overfit N=32 ≥99%, N=128 ≥95%  ✓
  → Serialization: checkpoint save/load + ONNX     ✓
  → SSM model (Python/CUDA)                        SKIPPED (no GPU)
```

## Completed phases

| Phase | Module | Status |
|-------|--------|--------|
| 1 | book_builder | Shipped |
| 2 | feature_encoder | Shipped |
| 3 | oracle_labeler + trajectory_builder | Shipped |
| 4 | MLP model | Shipped |
| 5 | GBT model | Shipped |
| 6 | CNN model | Shipped |
| 7 | Integration overfit (N=32, real data) | Shipped |
| 8 | SSM model | Skipped (CUDA) |
| 9 | Serialization | Shipped |
| 10 | N=128 overfit validation | Shipped |

## Test summary

- **220 unit tests** pass (1 disabled: `BookBuilderIntegrationTest.ProcessSingleDayFile`)
- **22 integration tests** (14 N=32 + 8 N=128) — labeled `integration`, excluded from default ctest
- Unit test time: ~5 min. Integration: ~20 min.

## What changed this cycle

- Added `tests/n128_overfit_test.cpp` — 8 integration tests for N=128 overfit (MLP, CNN, GBT)
- Updated `src/gbt_model.hpp` — changes to support N=128 overfit
- Updated `tests/test_helpers.hpp` — shared helpers for N=128 sampling
- Updated `tests/gbt_model_test.cpp`, `tests/cnn_model_test.cpp`, `tests/serialization_test.cpp` — test adjustments
- Spec: `.kit/docs/n128-overfit.md`

## Key files

| File | Role |
|---|---|
| `ORCHESTRATOR_SPEC.md` | Master spec (completed) |
| `.kit/docs/n128-overfit.md` | N=128 overfit spec |
| `CMakeLists.txt` | Top-level CMake build |
| `src/book_builder.hpp` | MBO → L2 book snapshots |
| `src/feature_encoder.hpp` | 194-dim feature encoding |
| `src/oracle_labeler.hpp` | Stateless oracle with lookahead |
| `src/trajectory_builder.hpp` | Position state + training sample generation |
| `src/mlp_model.hpp` | MLP architecture (libtorch) |
| `src/cnn_model.hpp` | CNN with spatial + temporal conv (libtorch) |
| `src/gbt_features.hpp` | 16 hand-crafted features |
| `src/gbt_model.hpp` | XGBoost C API wrapper |
| `src/training_loop.hpp` | Neural network overfit loop |
| `src/serialization.hpp` | Save/load + ONNX export |
| `tests/integration_overfit_test.cpp` | Full-pipeline integration test (N=32) |
| `tests/n128_overfit_test.cpp` | N=128 overfit integration test |
| `tests/serialization_test.cpp` | Serialization round-trip tests |

## Next actions

- Defined by user. Overfit harness complete at both N=32 and N=128.

## Build commands

```bash
cmake --build build -j12                                                  # build
cd build && ctest --output-on-failure --label-exclude integration         # unit tests (~5 min)
cd build && ctest --output-on-failure --label-regex integration           # integration tests (~20 min)
```

---

Updated: 2026-02-16
