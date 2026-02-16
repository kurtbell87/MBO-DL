# Last Touch — Cold-Start Briefing

## What to do next

**Serialization (Phase 9): TDD cycle complete, ship now.**

1. **Ship serialization** — `source .master-kit.env && ./.kit/tdd.sh ship .kit/docs/serialization.md`
2. **SSM model (optional)** — Requires CUDA + Python. Skip if no GPU available.
3. **To run integration tests explicitly** — `cd build && ctest --output-on-failure --label-regex integration`

## What was just completed (this cycle)

- Added `src/serialization.hpp` — save/load utilities for MLP, CNN (libtorch checkpoint + ONNX export), and GBT (XGBoost C API).
- Added `tests/serialization_test.cpp` — 15 tests covering checkpoint round-trips, ONNX export, edge cases.
- Updated `CMakeLists.txt` to build `serialization_test`.
- Updated `tests/gbt_model_test.cpp` and `tests/integration_overfit_test.cpp` (minor adjustments).
- Breadcrumbs updated: `CLAUDE.md`, `AGENTS.md`, `.kit/LAST_TOUCH.md`.

## All completed TDD cycles (exit 0)

| Phase | Module | Run IDs |
|-------|--------|---------|
| 1 | book_builder | red=`20260215T213556Z`, green=`20260215T214238Z`, refactor=`20260215T214714Z` |
| 2 | feature_encoder | red=`20260215T215213Z`, green=`20260215T215551Z`, refactor=`20260215T215723Z` |
| 3 | oracle_labeler + trajectory_builder | red=`20260215T220102Z`, green=`20260215T220637Z`, refactor=`20260215T221351Z` |
| 4 | MLP model | red=`20260215T221807Z`, green=`20260215T222111Z`, refactor=`20260215T224214Z` |
| 5 | GBT model | red=`20260215T225251Z`, green=`20260215T225958Z`, refactor=`20260215T231449Z` |
| 6 | CNN model | red=`20260215T232238Z`, green=`20260215T232602Z`, refactor=`20260216T000805Z` |
| 7 | integration-overfit | red=`20260216T004130Z`, green=`20260216T004723Z`, refactor=done |
| 9 | serialization | TDD cycle complete — ship pending |

## Test status (verified 2026-02-16)

- **219/219 unit tests pass**, 0 failures (1 disabled: `BookBuilderIntegrationTest.ProcessSingleDayFile`).
- **14 integration tests** excluded from default ctest via `--label-exclude integration`.
- **15 serialization tests** passing (MLP/CNN checkpoint round-trips, ONNX export, GBT save/load, edge cases).
- Total unit test time: ~294s (~5 min).

## Infrastructure fixes applied

1. **CLAUDECODE env var guard removed from kit scripts.** All three kit shell scripts now `unset CLAUDECODE` at startup.
2. **Integration tests excluded from default ctest.** `CMakeLists.txt` labels integration_overfit_test with `LABELS "integration"`. `.master-kit.env` sets `TEST_CMD` with `--label-exclude integration`.

## Architecture overview

```
Raw MBO (.dbn.zst) → book_builder → BookSnapshot[W=600]              ← DONE
  → feature_encoder → (B, 600, 194)                                   ← DONE
  → trajectory_builder + oracle_labeler → (window, label) pairs        ← DONE
  → MLP model → (B, 5) logits → overfit N=32                          ← DONE
  → CNN model → (B, 5) logits → overfit N=32                          ← DONE
  → GBT model (16 hand-crafted features) → overfit N=32               ← DONE
  → Integration overfit test (real data, all 3 models)                 ← DONE
  → Serialization (checkpoint + ONNX round-trips)                     ← TDD DONE, SHIP PENDING
  → SSM model (Python/CUDA) → overfit N=32                            ← SKIPPED (needs GPU)
```

## Key files

| File | Role |
|---|---|
| `CMakeLists.txt` | Top-level build (FetchContent: databento-cpp, libtorch, xgboost, GTest) |
| `src/book_builder.hpp` | BookSnapshot struct + BookBuilder class |
| `src/feature_encoder.hpp` | 194-dim feature encoding + constants |
| `src/oracle_labeler.hpp` | Stateless oracle with lookahead |
| `src/trajectory_builder.hpp` | TrainingSample + position state management |
| `src/mlp_model.hpp` | MLP architecture |
| `src/cnn_model.hpp` | CNN with spatial + temporal convolutions |
| `src/gbt_features.hpp` | 16 hand-crafted features |
| `src/gbt_model.hpp` | XGBoost C API wrapper |
| `src/training_loop.hpp` | Neural network overfit loop |
| `src/serialization.hpp` | Save/load for MLP, CNN (checkpoint + ONNX), GBT |
| `tests/serialization_test.cpp` | 15 serialization round-trip tests |
| `tests/integration_overfit_test.cpp` | Full-pipeline integration test (real data, N=32 overfit) |
| `tests/test_helpers.hpp` | Shared test utilities |
| `.kit/docs/serialization.md` | Serialization spec |

---

Updated: 2026-02-16 (breadcrumbs phase, serialization TDD complete)
