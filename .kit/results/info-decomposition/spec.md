# Phase R2: Information Decomposition [Research]

**Spec**: TRAJECTORY.md §2.2 (theory), §7.2 (simplification cascade), §8.7 (corrections)
**Depends on**: Phase 4 (feature-computation), Phase 5 (feature-analysis) — needs exported features + raw representations.
**R1 finding**: Subordination REFUTED — time bars are the baseline. Use `time_5s` (4,681 bars/day median) as primary bar type.
**Unlocks**: Phase 6 (synthesis) — provides architecture go/no-go decisions.

---

## Hypothesis

The predictive information about future returns decomposes across three sources (chain rule for mutual information):

$$I(\mathbf{B}_t, \mathbf{M}_t, \mathbf{H}_t\,;\, r_{t+h}) = \underbrace{I(\mathbf{B}_t\,;\, r_{t+h})}_{\text{spatial (CNN)}} + \underbrace{I(\mathbf{M}_t\,;\, r_{t+h} \mid \mathbf{B}_t)}_{\text{message encoder}} + \underbrace{I(\mathbf{H}_t\,;\, r_{t+h} \mid \mathbf{B}_t, \mathbf{M}_t)}_{\text{temporal (SSM)}}$$

Where:
- **B_t** = order book state at bar boundary (10 bid + 10 ask levels, 40 features flattened)
- **M_t** = sequence of MBO messages within bar t (Adds, Cancels, Modifies, Trades)
- **H_t** = history of previous bars (B_{t-1}, M_{t-1}, ...)

Each term maps to an encoder stage. If a term ≈ 0, the corresponding encoder adds no value and should be dropped per the §7.2 simplification cascade.

**Null hypothesis**: The book snapshot at bar close is a sufficient statistic for the intra-bar message sequence. Messages carry no incremental predictive information beyond what the resulting book state captures.

---

## Bar Type

**Primary**: `time_5s` (5-second time bars).

R1 showed no justification for event-driven bars on MES. `time_5s` produces ~4,681 bars/day (RTH), giving ~93,620 bars across 20 days — adequate power for all comparisons.

---

## Data Selection

**Source**: `DATA/GLBX-20260207-L953CAPU5B/` — 312 daily `.dbn.zst` files, MES MBO 2022.

**Day selection**: Same 19 days used in R1 (reuse `days_used` from subordination-test metrics.json). This ensures cross-experiment consistency and avoids selection bias.

```
Days: 20220103, 20220121, 20220211, 20220304, 20220331, 20220401, 20220422,
      20220513, 20220603, 20220630, 20220701, 20220722, 20220812, 20220902,
      20220930, 20221003, 20221024, 20221114, 20221205
```

**Session**: RTH only (09:30–16:00 ET). Exclude rollover windows via `RolloverCalendar::is_excluded()`.

**Warmup**: Discard first 50 bars per day (EWMA/rolling window warm-up per §8.6 policy).

**Expected sample size**: ~88,540 bars total (from R1 `time_5s` count), ~84,540 after warmup exclusion.

---

## Protocol

### Step 0: Data Export (C++)

Generate two CSV exports using existing infrastructure:

1. **Feature CSV** — via `FeatureExporter`:
   - Track A: 62 hand-crafted features (all 6 categories)
   - Track B: 40-dim flattened book snapshot (`book_snap_0..39`)
   - Track B: 33-dim message summary (`msg_summary_0..32`)
   - Forward returns: `fwd_return_1`, `fwd_return_5`, `fwd_return_20`, `fwd_return_100`
   - Metadata: `timestamp`, `bar_type`, `bar_param`, `day`, `is_warmup`

2. **Raw event export** — via `DayEventBuffer::get_events()`:
   - For each bar: extract MBO events indexed by `mbo_event_begin` / `mbo_event_end`
   - Per event: `[action, price, size, side, ts_event]` (5 floats)
   - Actions: 0=Add, 1=Cancel, 2=Modify, 3=Trade
   - Serialize to binary or parquet (one file per day)

### Step 1: Tier 1 — Hand-Crafted Proxies (Cheap)

Train two model types (linear regression + shallow MLP) to predict each return horizon from increasingly rich feature sets:

| Config | Input | Dim | Model | Output Metric |
|--------|-------|-----|-------|---------------|
| (a) | Track A hand-crafted features (all 62) | 62 | Linear + MLP | R²_bar |
| (b) | Track B raw book snapshot (flattened) | 40 | Linear + MLP | R²_book |
| (c) | Track B raw book + Category 6 message summaries | 40 + 5 = 45 | Linear + MLP | R²_book+msg_summary |
| (d) | Config (c) + lookback window of 20 previous bars' book snapshots | 45 + (20 × 40) = 845 | Linear + MLP | R²_full_summary |

**Category 6 features** (5 hand-crafted message summaries from `BarFeatureRow`):
- `cancel_add_ratio`, `message_rate`, `modify_fraction`, `order_flow_toxicity`, `cancel_concentration`

**MLP architecture**: 2 hidden layers, 64 units each, ReLU activation, dropout=0.1, BatchNorm. Train with AdamW, lr=1e-3, weight_decay=1e-4, 50 epochs, early stopping on validation loss (patience=10).

**Lookback construction** (Config d): For bar t, concatenate book snapshots from bars t-20 through t-1. Bars with insufficient lookback (first 20 bars per day after warmup) are excluded.

**Cross-validation**: 5-fold expanding-window time-series CV:
- Fold 1: Train on days 1–4, test on days 5–8
- Fold 2: Train on days 1–8, test on days 9–11
- Fold 3: Train on days 1–11, test on days 12–14
- Fold 4: Train on days 1–14, test on days 15–17
- Fold 5: Train on days 1–17, test on days 18–19

Day boundaries ensure no look-ahead leakage within folds.

**Standardization**: Z-score normalize all inputs using training fold statistics only. Apply same transform to test fold.

### Step 2: Tier 2 — Learned Message Encoder (Expensive, Definitive)

| Config | Input | Architecture | Output Metric |
|--------|-------|-------------|---------------|
| (e) | Flattened book (40) + LSTM on raw MBO events | Book MLP (40→32) ⊕ LSTM(5, 32, 1 layer) → concat(64) → Linear(1) | R²_book+msg_learned |
| (f) | Flattened book (40) + Transformer on raw MBO events | Book MLP (40→32) ⊕ TransformerEncoder(d=32, 2 heads, 1 layer) → concat(64) → Linear(1) | R²_book+msg_attn |

**Event sequence preprocessing**:
- Truncate to max 500 events per bar (keep most recent if exceeded)
- Pad shorter sequences with zero-vectors
- Per event: `[action_onehot(4), price_delta_from_mid, log1p(size), side]` → 7 features per event
- Log truncation rate: report fraction of bars where sequence was truncated

**Training**: Same CV folds and training hyperparameters as Tier 1 MLP. Batch size 256, gradient clipping at 1.0.

**Seed control**: Fix 3 random seeds (42, 123, 456) per config per fold. Report mean across seeds to reduce initialization variance. Total training runs: 2 configs × 5 folds × 3 seeds = 30 Tier 2 runs.

### Step 3: Multi-Horizon Sweep

Repeat Steps 1 and 2 for all four return horizons:
- `fwd_return_1` (1-bar, ~5s ahead)
- `fwd_return_5` (5-bar, ~25s ahead)
- `fwd_return_20` (20-bar, ~100s ahead)
- `fwd_return_100` (100-bar, ~500s ahead)

### Step 4: Analysis

**Compute information gaps** (for each horizon, using MLP R² values — linear included as sanity check):

| Gap | Formula | Question |
|-----|---------|----------|
| Δ_spatial | R²_book − R²_bar | Does raw book beat hand-crafted features? |
| Δ_msg_summary | R²_book+msg_summary − R²_book | Do Category 6 summaries add value? |
| Δ_msg_learned | R²_book+msg_learned − R²_book | Do raw message sequences add value? (Tier 2) |
| Δ_temporal | R²_full_summary − R²_book+msg_summary | Does lookback history add value? |
| Δ_tier2_vs_tier1 | Δ_msg_learned − Δ_msg_summary | Does learning beat hand-crafting for messages? |

**Statistical tests** (per gap, per horizon):
- Paired t-test on per-fold R² differences (5 paired values)
- If Shapiro-Wilk p < 0.05 on paired differences, use Wilcoxon signed-rank instead
- Report both raw and Holm-Bonferroni corrected p-values

**Multiple comparison correction**:
- 5 gaps × 4 horizons × 2 model types (linear, MLP) = 40 tests
- Apply Holm-Bonferroni within each gap family (8 tests per gap: 4 horizons × 2 model types)
- Report: point estimate, 95% CI (bootstrap, 1000 iterations), raw p, corrected p

### Step 5: Threshold Policy (Revised from §2.2)

Both conditions must hold for an encoder stage to be justified:

1. **Relative threshold**: R² gap > 20% of baseline R². Example: if R²_book = 0.005, message encoder justified only if Δ_msg > 0.001.
2. **Statistical threshold**: Corrected p < 0.05.

---

## Decision Matrix (§7.2 Simplification Cascade)

The message encoder decision is gated on **Tier 2** (learned sequence model), not Tier 1 (hand-crafted summaries):

```
If Δ_msg_learned > Δ_msg_summary AND Δ_msg_learned passes threshold:
  → Category 6 summaries are insufficient. Message encoder justified.
  → Hand-crafted features miss sequential patterns (sweeps, quote stuffing, etc.).

If Δ_msg_learned ≈ Δ_msg_summary AND both pass threshold:
  → Summaries capture message information. Use Category 6 features instead
    of message encoder (simpler, cheaper).

If Δ_msg_learned ≈ 0 AND Δ_msg_summary ≈ 0:
  → Messages carry no incremental info beyond book state. Drop message encoder.

Architecture recommendation (fill each row with empirical results):

  Encoder Stage     | Gap        | Passes Threshold? | Include?
  ------------------|------------|-------------------|----------
  Spatial (CNN)     | Δ_spatial  |                   |
  Message encoder   | Δ_msg_*    |                   |
  Temporal (SSM)    | Δ_temporal |                   |

Resulting architecture (one of):
  - Full three-level: CNN + message encoder + SSM
  - Two-level: CNN + SSM (drop message encoder)
  - CNN + GBT features (drop temporal encoder if Δ_temporal ≈ 0)
  - GBT baseline only (if neither spatial nor temporal gaps pass threshold)
```

---

## Implementation

```
Language: Python (PyTorch + scikit-learn)
Entry point: research/R2_information_decomposition.py

Dependencies:
  - torch >= 2.0 (MLP, LSTM, TransformerEncoder)
  - scikit-learn (TimeSeriesSplit adaptation, LinearRegression, r2_score)
  - polars (CSV loading, feature selection)
  - scipy (shapiro, wilcoxon, ttest_rel)
  - numpy

Input:
  - Feature CSV from FeatureExporter (Track A + Track B + returns)
  - Raw event binary/parquet from DayEventBuffer export

Output:
  - .kit/results/info-decomposition/metrics.json
  - .kit/results/info-decomposition/analysis.md
```

---

## Compute Budget

| Item | Estimate |
|------|----------|
| Data export (C++) | ~15 min (19 days × time_5s bar construction + feature computation) |
| Tier 1: Linear (4 configs × 4 horizons × 5 folds) | ~5 min (CPU, trivial) |
| Tier 1: MLP (4 configs × 4 horizons × 5 folds = 80 runs) | ~30 min (CPU/GPU, 50 epochs each) |
| Tier 2: LSTM + Transformer (2 configs × 4 horizons × 5 folds × 3 seeds = 120 runs) | ~2–3 hours (GPU, 500-event sequences) |
| Analysis + statistical tests | ~5 min |
| **Total wall-clock** | **~3–4 hours** |
| **GPU hours** | **~2–3** (Tier 2 only) |
| **Runs** | **~200** (80 Tier 1 + 120 Tier 2) |

**Within budget**: < 4 GPU-hours, < 10 logical experiment runs (200 training runs are sub-runs within a single experiment).

---

## Deliverables

### Table 1: R² Matrix (MLP model, primary)

```
Horizon | Config (a)  | Config (b) | Config (c)       | Config (d)       | Config (e)        | Config (f)
        | R²_bar      | R²_book    | R²_book+msg_sum  | R²_full_summary  | R²_book+msg_lstm  | R²_book+msg_attn
--------|-------------|------------|------------------|------------------|-------------------|------------------
return_1|             |            |                  |                  |                   |
return_5|             |            |                  |                  |                   |
return_20|            |            |                  |                  |                   |
return_100|           |            |                  |                  |                   |

Each cell: mean_R² ± std_R² (5-fold)
```

### Table 2: Information Gaps

```
Gap              | Horizon | MLP Δ_R² | 95% CI    | Raw p    | Corrected p | Passes Threshold?
-----------------|---------|----------|-----------|----------|-------------|------------------
Δ_spatial        | 1       |          |           |          |             |
Δ_spatial        | 5       |          |           |          |             |
...
Δ_msg_learned    | 1       |          |           |          |             |
...
Δ_tier2_vs_tier1 | 5       |          |           |          |             |
```

### Table 3: Truncation Statistics

```
Metric                    | Value
--------------------------|------
Bars with > 500 events    |
Truncation rate (%)       |
Median events per bar     |
P95 events per bar        |
Max events per bar        |
```

### Summary Finding

One of:
- **FULL ARCHITECTURE**: All three encoder stages pass threshold. Build CNN + message encoder + SSM.
- **DROP MESSAGE**: Δ_msg ≈ 0. Book state is sufficient. Build CNN + SSM.
- **DROP TEMPORAL**: Δ_temporal ≈ 0. Current state is sufficient. Build CNN (+ message if justified).
- **FEATURES SUFFICIENT**: Δ_spatial ≈ 0. Hand-crafted features match raw book. Use GBT baseline.

---

## Exit Criteria

- [ ] C++ data export completed (feature CSV + raw event data for 19 days × time_5s)
- [ ] Tier 1 R² values computed for all 4 configs × 4 horizons × 2 model types
- [ ] Tier 2 R² values computed for LSTM + transformer × 4 horizons (3 seeds each)
- [ ] All 5 information gaps computed with paired statistical tests
- [ ] Holm-Bonferroni correction applied across comparisons
- [ ] Threshold policy applied: relative gap > 20% of baseline R² AND corrected p < 0.05
- [ ] Truncation rate logged for Tier 2 message sequences
- [ ] Bootstrap 95% CIs reported for all gaps
- [ ] Architecture decision from §7.2 simplification cascade filled in with empirical results
- [ ] Results written to `.kit/results/info-decomposition/`
- [ ] Summary entry appended to `.kit/RESEARCH_LOG.md`
