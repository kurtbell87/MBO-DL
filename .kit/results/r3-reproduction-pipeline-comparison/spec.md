# Experiment: R3 Reproduction & Data Pipeline Comparison

## Hypothesis

**Step 1 (R3 Reproduction):** Training R3's exact CNN architecture (Conv1d 2→59→59, 12,128 params) on book tensors constructed via R3's Python data-loading path (raw `.dbn.zst` → Python databento → (N, 2, 20) with tick-normalized price offsets and day-level z-scored log1p sizes) will achieve mean OOS R² **≥ 0.10** at h=5 across 5 expanding-window folds on the same 19 trading days, reproducing ≥76% of R3's reported R²=0.132.

**Step 2 (Pipeline Comparison, conditional on Step 1 PASS):** The C++ `bar_feature_export` (`time_5s.csv`) produces book tensors that are **structurally non-equivalent** to R3's Python-exported tensors for the same trading days — differing in at least one of: mid-price reference, price offset scale, level ordering, level selection, or size normalization — and this structural difference is the root cause of the CNN R²=0.132→0.002 collapse observed in Phases 9B and 9C.

## Independent Variables

### Step 1: R3 Reproduction (single configuration — no manipulation)

Faithful re-execution of R3's pipeline on R3's data format. The independent variable is implicit: R3's Python data pipeline vs. the C++ pipeline used in Phases 9B/9C. This step uses only R3's path; comparison happens in Step 2.

- **Data loading:** Python databento library reading raw `.dbn.zst` files
- **Book format:** (N, 2, 20) — Channel 0: `(price - mid) / TICK_SIZE` (integer ticks from mid); Channel 1: `log1p(size)`, z-scored per day
- **Level layout:** Rows 0–9 = bids (deepest→best bid at row 9); Rows 10–19 = asks (best ask at row 10→deepest)
- **Architecture:** Conv1d(2→59→59) + BN + ReLU × 2 → AdaptiveAvgPool1d(1) → Linear(32→16) + ReLU → Linear(16→1). 12,128 params.
- **Training:** AdamW(lr=1e-3, wd=1e-4), CosineAnnealingLR(T_max=50, eta_min=1e-5), batch=512, patience=10, MSE on `fwd_return_5`
- **CV:** 5-fold expanding-window on 19 non-rollover days (identical to R3's fold structure)

### Step 2: Pipeline Comparison (conditional on Step 1 PASS, 2 levels)

| Level | Description | Source |
|-------|-------------|--------|
| **R3 Python** | Book tensors from Step 1's Python data loading | Raw `.dbn.zst` → Python databento → (N, 2, 20) |
| **C++ Export** | Book columns from `time_5s.csv`, reshaped to (N, 2, 20) | Raw `.dbn.zst` → C++ `bar_feature_export` → CSV → reshape |

## Controls

| Control | Value | Rationale |
|---------|-------|-----------|
| Trading days | Same 19 non-rollover days used in R3, Phases 9B, 9C | Eliminates calendar selection as variable |
| Raw data | `DATA/GLBX-20260207-L953CAPU5B/*.dbn.zst` | Same underlying MBO events for both pipelines |
| CNN architecture | Conv1d(2→59→59), 12,128 params | Matched to R3 exactly (confirmed by 9C: 0% deviation) |
| Optimizer | AdamW(lr=1e-3, wd=1e-4) | R3's exact config |
| LR schedule | CosineAnnealingLR(T_max=50, eta_min=1e-5) | R3's exact config |
| Batch size | 512 | R3's exact config |
| Early stopping | Patience=10 on val loss, max 50 epochs | R3's exact config |
| Loss | MSE on `fwd_return_5` | R3's exact config |
| Seed | 42 (torch, numpy, random) | Reproducibility |
| Hardware | CPU only | CNN ~12k params; no GPU needed |
| Bar type | time_5s | Locked by R1 + R6 synthesis |

## Metrics (ALL must be reported)

### Primary

1. **mean_cnn_r2_h5** (Step 1): Mean out-of-sample R² of CNN regression at h=5 across 5 folds. Directly tests whether R3's signal reproduces on R3-format data.
2. **pipeline_structural_equivalence** (Step 2): Boolean — are R3-Python and C++ book tensors structurally equivalent (identity rate > 0.99 and per-level channel correlations > 0.99)?

### Secondary

| Metric | Description |
|--------|-------------|
| per_fold_cnn_r2_h5 | Per-fold test R² (compare with R3: [0.163, 0.109, 0.049, 0.180, 0.159]) |
| per_fold_cnn_train_r2_h5 | Per-fold train R² (must be > 0.05 — the 9B/9C smoking gun was train R² ≈ 0) |
| epochs_trained_per_fold | Epochs before early stopping (R3 context: 50 max, patience 10) |
| tensor_identity_rate | Fraction of matched bars where tensors are element-wise identical (ε=1e-4) |
| channel_0_per_level_corr | Pearson correlation of price offsets per level between the two pipelines |
| channel_1_per_level_corr | Pearson correlation of sizes per level between the two pipelines |
| bar_count_discrepancy | Absolute difference in bar count per day between the two pipelines |
| value_range_comparison | Min/max/mean/std of each channel, each pipeline (summary table) |
| structural_differences_list | Categorical list of all identified differences |
| transfer_r2 | R3-trained CNN (Step 1 fold 5) evaluated on C++ data (tests feature transfer) |
| retrained_cpp_r2 | CNN retrained from scratch on C++ data with R3 protocol (tests if C++ data has any structure) |

### Sanity Checks

| Check | Expected | Failure means |
|-------|----------|---------------|
| CNN param count | 12,128 ± 5% | Architecture mismatch — results invalid |
| Channel 0 sample values (R3 data) | Integer-valued tick offsets (e.g., -10, -5, -1, 0, 1, 5, 10) | Data loading is wrong — not R3's format |
| Channel 1 sample values (R3 data) | Z-scored log1p sizes (mean ≈ 0, std ≈ 1 per day) | Size normalization error |
| LR at epochs 1 / 25 / 50 | Decays from ~1e-3 toward ~1e-5 | CosineAnnealingLR not applied |
| Train R² per fold > 0.05 | All folds > 0.05 | Pipeline broken — same failure as 9B/9C |
| No NaN in model outputs | 0 NaN | Normalization or forward pass bug |
| Fold day boundaries non-overlapping | No day in both train and test | Temporal leakage |
| R3 day count matches | 19 unique trading days loaded | Data loading selected wrong days |
| Bar count per day | ~4,000–5,000 bars (time_5s, RTH) | Bar construction logic error |

## Baselines

### 1. R3 Original Result (reproduction target)
- **Source:** R3 book-encoder-bias experiment (`.kit/results/book-encoder-bias/analysis.md`)
- **Per-fold R²:** [0.163, 0.109, 0.049, 0.180, 0.159]
- **Mean:** 0.132 ± 0.048
- **Protocol:** Conv1d 2→59→59, 12,128 params, raw tick offsets, CosineAnnealingLR, AdamW(lr=1e-3, wd=1e-4), batch=512, patience=10, MSE loss
- **Data:** Phase 4 Track B.1 Python export from `.dbn.zst` files
- **Role:** Reproduction target. Threshold: ≥ 0.10 (76% of 0.132).

### 2. Phase 9C CNN Result (broken-pipeline reference)
- **Source:** cnn-reproduction-diagnostic (`.kit/results/cnn-reproduction-diagnostic/analysis.md`)
- **Value:** Fold 5 train R² = 0.002, test R² = 0.0001
- **Data:** Phase 9A C++ export (`time_5s.csv`), same 19 days
- **Protocol:** R3-exact architecture (12,128 params, 0% deviation), 5 normalization variants tested — ALL R² < 0.002
- **Role:** The broken state. This experiment tests whether the data pipeline (not model configuration) explains the 0.132→0.002 gap.

### 3. Phase 9B CNN Result (initial failure)
- **Source:** hybrid-model-training (`.kit/results/hybrid-model-training/analysis.md`)
- **Value:** Mean test R² = -0.002, train R² = 0.001
- **Data:** Same `time_5s.csv`
- **Role:** Earlier negative reference. Originally attributed to 3 protocol deviations; 9C disproved that explanation.

## Success Criteria (immutable once RUN begins)

- [ ] **SC-1**: mean_cnn_r2_h5 ≥ 0.10 on R3-format data (R3's signal reproduces)
- [ ] **SC-2**: No fold train R² < 0.05 in Step 1 (CNN can fit R3-format training data)
- [ ] **SC-3**: Per-fold R² correlation with R3's per-fold values > 0.5 (same temporal pattern, not just mean match)
- [ ] **SC-4** (conditional on SC-1 PASS): pipeline_structural_equivalence == False (pipelines produce different tensors, explaining the gap)
- [ ] **SC-5** (conditional on SC-1 PASS): At least one specific structural difference identified and documented with quantitative evidence
- [ ] **SC-6**: No sanity check failures

## Minimum Viable Experiment

Before the full 5-fold run, execute a **single-fold gate on fold 5** (days 1–16 train, days 17–19 test — maximum training data, best chance of reproducing R3).

**Phase 0: Data loading verification (MANDATORY before any training):**
1. Locate R3's code: search for `research/R3_book_encoder_bias.py` or equivalent R3 data-loading implementation.
2. If R3's code exists: inspect its data loading path. Document exactly how it reads `.dbn.zst` files, constructs bar boundaries, builds the (20, 2) book tensor, calculates mid-price, computes price offsets, and normalizes sizes.
3. If R3's code does NOT exist: recreate the data loading from R3's spec (`.kit/experiments/book-encoder-bias.md` Data section). Use the Python `databento` library to read `.dbn.zst` files. Construct (N, 2, 20) tensors per the documented format:
   - Channel 0: `(price - mid_price) / TICK_SIZE` — integer ticks from mid
   - Channel 1: `log1p(size)`, z-scored per day (train stats for train, but day-level z-score per R3)
   - Rows 0–9: bids (deepest first → best bid at row 9)
   - Rows 10–19: asks (best ask at row 10 → deepest last)
4. Load data for the 19 days. Verify: 19 unique days, ~4,000–5,000 bars per day, ~80,000–95,000 total bars.
5. Print sample values: channel 0 must show integer-valued tick offsets. Channel 1 must show z-scored log1p sizes.
6. **ABORT if channel 0 values are not integer-like (tolerance ±0.01) or if bar count is < 50,000 or > 120,000.**

**Phase 1: MVE training (fold 5 only):**
1. Build CNN model, verify param count = 12,128 ± 5%.
2. Train on fold 5 with R3-exact protocol.
3. Record train R² and test R².
4. **Gate A:** train R² < 0.05 → R3's Python pipeline ALSO fails. **STOP.** R3's original R²=0.132 is suspect — it was NOT reproducible even on its own data format. Report to analysis.md.
5. **Gate B:** test R² < 0.05 → signal may be period- or seed-specific. Proceed to full 5-fold cautiously.
6. **Gate C:** test R² ≥ 0.10 → strong reproduction evidence. Proceed to full 5-fold.

## Full Protocol

### Step 0: Environment Setup
- Set seed=42 globally (torch, numpy, random).
- Log Python version, PyTorch version, databento library version.
- Verify dependencies: torch, databento, numpy, polars/pandas, scikit-learn.

### Step 1: R3 Data Loading and Validation
1. Execute MVE Phase 0 (data loading verification) as described above.
2. Save all 19 days of R3-format book tensors to `.npy` files for Step 2 comparison.
3. Construct `fwd_return_5` target: 5-bar forward return computed from bar close prices.
4. Log: total bars, bars per day, unique days, channel value distributions.

### Step 2: Define Expanding-Window Splits

| Fold | Train Days | Test Days |
|------|-----------|-----------|
| 1 | Days 1–4 | Days 5–7 |
| 2 | Days 1–7 | Days 8–10 |
| 3 | Days 1–10 | Days 11–13 |
| 4 | Days 1–13 | Days 14–16 |
| 5 | Days 1–16 | Days 17–19 |

### Step 3: Run MVE (fold 5 only)
Execute MVE Phase 1. Abort if Gate A triggers.

### Step 4: Full 5-Fold CNN Training (Step 1 of the hypothesis)

For each fold k in [1..5]:
1. Split data by day boundaries per Step 2.
2. Channel 0 (price offsets): NO further normalization — already in ticks from mid.
3. Channel 1 (sizes): Already z-scored per day from data loading. If R3 applied additional per-fold normalization, match that. Document the exact normalization applied.
4. Reserve last 20% of train days as validation set (for early stopping only).
5. Train CNN with R3-exact protocol:
   - AdamW(lr=1e-3, weight_decay=1e-4)
   - CosineAnnealingLR(T_max=50, eta_min=1e-5)
   - Batch size: 512
   - Max epochs: 50, early stopping patience=10 on val loss
   - MSE loss on `fwd_return_5`
6. Save best checkpoint (lowest val loss).
7. Record: train R², test R², epochs trained, final LR.

### Step 5: Step 1 Gate Evaluation

Compute mean test R² across 5 folds. Compare to R3 per-fold values.

| Outcome | Action |
|---------|--------|
| Mean R² ≥ 0.10 | **PASS** — R3's signal is REAL and data-pipeline-dependent. Proceed to Step 2 (pipeline comparison). |
| 0.05 ≤ Mean R² < 0.10 | **MARGINAL** — Signal exists but weaker than reported. Proceed to Step 2. Flag that R3's R²=0.132 may have been inflated. |
| Mean R² < 0.05 | **FAIL** — R3's signal does NOT reproduce even on R3-format data. Do NOT proceed to Step 2. Report: R3's R²=0.132 is likely artifactual. Architecture decision must be revisited. |

### Step 6: Pipeline Comparison (conditional on Step 5 PASS or MARGINAL)

**6a. Load C++ Export Data**
1. Load `.kit/results/hybrid-model/time_5s.csv`.
2. Identify 40 book columns. Reshape to (N, 2, 20) using the column mapping from 9B/9C.
3. Identify the same 19 trading days.

**6b. Align Bars by Timestamp**
1. For each of the 19 days, match bars between R3-Python and C++ by timestamp.
2. Report: bars per day in each pipeline, timestamp alignment rate, unmatched bars count.
3. If bar counts differ by > 5% for any day, flag and investigate (different bar boundary logic).

**6c. Element-by-Element Tensor Comparison**
For each matched bar pair:
1. Compare channel 0 (price offsets): compute absolute difference, relative difference, and Pearson correlation per level.
2. Compare channel 1 (sizes): same metrics.
3. Classify differences:
   - **Scale difference:** Constant multiplicative factor (e.g., index points vs ticks → 4× factor)
   - **Reference shift:** Constant additive offset (different mid-price calculation)
   - **Ordering difference:** Levels in different sequence (bid/ask layout swapped)
   - **Content difference:** Fundamentally different values (different levels selected, different aggregation)
   - **Missing data handling:** Different fill values for thin books

**6d. Diagnostic Summary Table**

| Dimension | R3 Python | C++ Export | Match? |
|-----------|-----------|------------|--------|
| Bar count per day | N | N | Yes/No (± %) |
| Channel 0 units | ticks (integers) | index points (floats) | No (expected) |
| Channel 0 value range | [min, max] | [min, max] | — |
| Channel 1 normalization | log1p + day z-score | log1p + ? | Yes/No |
| Channel 1 value range | [min, max] | [min, max] | — |
| Level ordering (bids) | deepest→best | ? | Yes/No |
| Level ordering (asks) | best→deepest | ? | Yes/No |
| Mid-price formula | ? (document) | ? (document) | Yes/No |
| Missing level encoding | ? | ? | Yes/No |

**6e. Cross-Pipeline CNN Evaluation**
1. **Transfer test:** Take Step 1's best CNN model (fold 5 checkpoint). Forward pass C++ data through it. Record test R². If R² drops >50% vs Step 1's fold 5 R², the data difference is material for the learned features.
2. **Retrain test:** Train a fresh CNN on C++ data (fold 5, same R3 protocol). Record train and test R². Compare with Step 1 fold 5 and Phase 9C fold 5 (train R²=0.002).

### Step 7: Write Analysis

Produce `analysis.md` with all required sections:

1. **Step 1 verdict:** PASS / MARGINAL / FAIL with mean R² and per-fold comparison table vs R3.
2. **Data loading documentation:** Exactly how R3's data was loaded (library, mid-price formula, normalization, level ordering).
3. **R3 comparison table:** This experiment vs R3 original, per-fold and mean ± std.
4. **If Step 1 PASS:** Pipeline comparison summary (6d table), structural differences list, cross-pipeline R² results.
5. **Root cause verdict:** Is the data pipeline confirmed as root cause? What specifically differs?
6. **Recommendation:** Fix C++ export (handoff to TDD) / Abandon CNN path / Further investigation needed.
7. **Explicit pass/fail for SC-1 through SC-6.**

## Resource Budget

**Tier:** Standard

- Max GPU-hours: 0 (CPU only)
- Max wall-clock time: 120 minutes
- Max training runs: 8 (1 MVE + 5 full folds + 1 transfer eval + 1 retrain on C++ data)
- Max seeds per configuration: 1 (seed=42; variance assessed across 5 temporal folds)

### Compute Profile
```yaml
compute_type: cpu
estimated_rows: 90000
model_type: pytorch
sequential_fits: 8
parallelizable: false
memory_gb: 8
gpu_type: none
estimated_wall_hours: 1.0
```

**Estimate breakdown:**
- Data loading from 19 `.dbn.zst` files via Python databento: ~15-30 minutes (dominant cost — MBO data is large; each file has millions of events that must be aggregated into time_5s bars and book snapshots).
- CNN training (~12k params on ~75k rows): ~15s/fold × 6 = ~90s.
- C++ CSV loading + reshape + comparison: ~2 minutes.
- Cross-pipeline CNN eval (1 forward pass + 1 retrain): ~30s.
- Analysis and file I/O: ~5 minutes.
- **Total estimated: ~30-45 minutes.** Budget of 120 minutes provides 3-4× headroom for slow .dbn.zst I/O.

## Abort Criteria

- **R3 code not found AND databento Python library not available:** Cannot load R3-format data. ABORT immediately. Report missing infrastructure.
- **Data loading failure:** Cannot read `.dbn.zst` files or construct book tensors. ABORT with specific error. Likely missing databento Python library.
- **Bar count wildly wrong:** < 50,000 or > 120,000 total bars from 19 days → data loading logic error. ABORT.
- **Channel 0 not integer-like:** If R3-format price offsets are not approximately integer-valued (tolerance ±0.01), the data loading does not match R3's spec. ABORT and diagnose.
- **NaN loss:** Any fold produces NaN loss within 5 epochs → normalization or overflow bug. ABORT.
- **MVE Gate A:** Fold 5 train R² < 0.05 → R3's pipeline fails too. STOP (do not proceed to full 5-fold). This is not an error — it is a valid experimental outcome (R3's result was not reproducible).
- **Per-run time:** Any single CNN fit exceeds 5 minutes (20× expected) → investigate.
- **Wall-clock:** Total exceeds 120 minutes → ABORT remaining work, report what completed.

## Confounds to Watch For

1. **Python environment drift.** R3 was run on a specific PyTorch/databento version. If library APIs or numerical behavior changed, minor R² differences (±0.02) are expected. A 0.132→<0.05 gap cannot be explained by version drift. The RUN agent must log all library versions. If the databento Python API has breaking changes, this is a showstopper — document and abort.

2. **Day selection ambiguity.** R3 specified "19 non-rollover trading days spread across 2022." The exact day list may be embedded in R3's code or may require reconstruction. If the 19 days used here differ from R3's original 19 days, results are not directly comparable. Mitigation: log the exact day list; cross-reference with Phase 9A's `time_5s.csv` (which also used 19 days — they should be the same days). If days differ, this is a secondary confound, not the primary one (9C showed R²=0.002 on any 19-day subset of this data).

3. **Book construction logic (if recreating R3's loading).** If R3's original code is unavailable and the RUN agent recreates the loading, subtle differences in bar boundary determination, mid-price calculation, or level selection could produce a book tensor that looks correct (integer ticks, right shape) but has different content than R3 used. This is the #1 risk if R3's code is missing. Mitigation: verify loaded data against R3's documented statistics if available; print extensive diagnostics.

4. **`.dbn.zst` file integrity.** If raw data files were modified since R3's original run (unlikely for archived market data), the comparison is invalid. Mitigation: check file sizes and modification dates vs expectations.

5. **fwd_return_5 computation.** R3 computed forward returns from bar close prices. If the Python and C++ pipelines use slightly different close price calculations (last trade price vs mid at bar close vs VWAP), the target variable itself may differ, confounding R² comparison. Mitigation: compute target from R3-loaded data (not from `time_5s.csv`). In Step 2, also compare the two pipelines' forward returns.

6. **R3's result was genuinely artifactual.** Two independent attempts (9B, 9C) failed on C++ data. If this experiment also fails on R3-format data (Gate A), the most parsimonious explanation is that R3's R²=0.132 was inflated by a bug (temporal leakage, train/test confusion, etc.). This is the adversarial hypothesis that SC-1 FAIL would support. It must be stated upfront, not treated as an afterthought.

7. **Single seed limitation.** Seed=42 is deterministic but represents one initialization trajectory. If R3's original result was seed-sensitive (one lucky initialization on a noisy objective), this experiment could reproduce or fail to reproduce depending on whether seed=42 matches R3's original seed. R3's spec does not specify its original seed. Mitigation: if mean R² falls in the 0.05–0.12 marginal zone, recommend follow-up with 3 additional seeds.

## Decision Rules

```
OUTCOME A — R3 Reproduces + Pipelines Differ (EXPECTED):
  SC-1 PASS + SC-4 PASS + SC-5 PASS
  → R3's CNN signal is REAL but pipeline-specific.
  → C++ export is the root cause of the 0.132→0.002 gap.
  → HANDOFF to TDD: fix C++ bar_feature_export to match R3's data contract.
  → CNN+GBT architecture path is VIABLE pending C++ fix.

OUTCOME B — R3 Does NOT Reproduce (ADVERSARIAL):
  SC-1 FAIL (mean R² < 0.05)
  → R3's R²=0.132 was NOT reproducible even on R3-format data.
  → R3's result is likely artifactual (temporal leakage, measurement error, or seed-specific).
  → ABANDON CNN spatial encoder path.
  → Pivot to GBT-only architecture with refined feature engineering.
  → R6 synthesis "CNN+GBT" recommendation is INVALIDATED.

OUTCOME C — R3 Reproduces + Pipelines Are Equivalent (UNEXPECTED):
  SC-1 PASS + SC-4 FAIL (identity rate > 0.99)
  → Both pipelines produce the same data, yet CNN works on one and not the other.
  → Deep investigation needed: reshape logic, bar boundary alignment, or numerical precision.
  → This would be the most puzzling outcome — suggests a subtle code bug in 9B/9C experiments.

OUTCOME D — Marginal Reproduction (AMBIGUOUS):
  0.05 ≤ mean R² < 0.10
  → Signal exists but weaker than R3 reported.
  → R3's R²=0.132 was inflated (environment, seed, or measurement).
  → Proceed to Step 2 cautiously.
  → CNN path is POSSIBLE but the expected edge is smaller than R6 assumed.
  → Recommend follow-up with 3 additional seeds before committing.
```

## Deliverables

```
.kit/results/r3-reproduction-pipeline-comparison/
  step1/
    data_loading_documentation.md   # How R3 data was loaded, libraries, formulas
    data_verification.txt           # Sample values, bar counts, day list
    mve_diagnostics.txt             # Fold 5 gate: train R², test R², param count
    fold_results.json               # Per-fold: {train_r2, test_r2, epochs_trained, final_lr}
    r3_comparison_table.csv         # This run vs R3 original, per-fold
    book_tensors/                   # Saved R3-format tensors for Step 2 (19 × .npy)
  step2/                            # Only populated if Step 1 PASS or MARGINAL
    pipeline_comparison.json        # Element-by-element summary: identity rate, correlations
    diagnostic_table.csv            # 6d comparison table
    structural_differences.md       # Human-readable list of ALL differences found
    cross_pipeline_r2.json          # Transfer R² + retrained R² on C++ data
    value_distributions.csv         # Per-channel, per-pipeline: min, max, mean, std per level
  aggregate_metrics.json            # All primary + secondary metrics
  analysis.md                       # Human-readable with all required sections + SC pass/fail
```

## Exit Criteria

- [ ] R3's data loading path located and inspected (or recreated from spec with documentation)
- [ ] R3's data loading documented: library, bar construction, book tensor format, mid-price formula, level ordering, size normalization
- [ ] Data verification complete: 19 days, channel 0 = integer ticks, channel 1 = z-scored log1p, bar count reasonable
- [ ] MVE gate executed (fold 5 train and test R² reported)
- [ ] Full 5-fold Step 1 completed (or aborted at MVE gate with documented reason)
- [ ] Per-fold R² comparison table: this run vs R3 original
- [ ] Step 1 verdict: PASS (≥0.10) / MARGINAL (0.05–0.10) / FAIL (<0.05)
- [ ] If Step 1 PASS/MARGINAL: R3 book tensors saved as .npy for Step 2
- [ ] If Step 1 PASS/MARGINAL: Pipeline comparison completed (6c element-by-element + 6d diagnostic table)
- [ ] If Step 1 PASS/MARGINAL: Structural differences documented with quantitative evidence
- [ ] If Step 1 PASS/MARGINAL: Cross-pipeline CNN R² reported (transfer + retrained)
- [ ] analysis.md written with all required sections
- [ ] All SC-1 through SC-6 evaluated explicitly
- [ ] Decision outcome (A/B/C/D) stated with next-step recommendation
