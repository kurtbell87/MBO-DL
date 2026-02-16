# N=128 Overfit Validation

## Goal

Close the two remaining exit criteria gaps from ORCHESTRATOR_SPEC.md §10:

1. All available models pass the N=128 overfit test (≥95% train accuracy)
2. N=128 accuracy is evaluated on all 128 samples per epoch

## Requirements

### Integration tests for N=128 overfit

Add tests to `tests/integration_overfit_test.cpp` (or a new test file) that:

1. Build a trajectory from real MBO data (same as existing integration tests — `glbx-mdp3-20220103.mbo.dbn.zst`, instrument_id `13615`)
2. Sample N=128 evenly-spaced windows from the trajectory
3. Train each available model (MLP, CNN, GBT) on the 128 samples
4. Assert final accuracy ≥95% on all 128 samples

### Accuracy evaluation requirement

The N=128 tests must verify that accuracy is computed over **all 128 samples per epoch**, not per-batch. The existing training loop already does this (evaluates on the full dataset after each epoch), but the test should assert:

- The accuracy reported is computed on all 128 samples (not a subset)
- This can be verified by checking that the training function receives all 128 samples and the accuracy denominator is 128

### Model-specific notes

- **MLP**: May need more epochs than N=32 (up to 1000). The spec notes MLP has slow gradient propagation through 116K-dim input.
- **CNN**: Should converge faster than MLP. 500 epochs max should suffice.
- **GBT**: XGBoost with overfit-friendly params. 1000 rounds max.
- **SSM**: Skip (no CUDA available).

### Test names

- `N128OverfitTest.MLPOverfit128` — MLP reaches ≥95% on 128 samples
- `N128OverfitTest.CNNOverfit128` — CNN reaches ≥95% on 128 samples
- `N128OverfitTest.GBTOverfit128` — GBT reaches ≥95% on 128 samples

### Labels

These tests train models and may take several minutes each. Label them as `integration` in CMake so they're excluded from the default `ctest` run (same as existing integration tests).

## Acceptance Criteria

- [ ] MLP achieves ≥95% accuracy on N=128 training set
- [ ] CNN achieves ≥95% accuracy on N=128 training set
- [ ] GBT achieves ≥95% accuracy on N=128 training set
- [ ] Accuracy is evaluated on all 128 samples (not per-batch)
- [ ] Tests are labeled `integration` in CMake
- [ ] Existing 219 unit tests still pass
- [ ] No regressions in existing integration tests
