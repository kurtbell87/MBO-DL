# Survey: Can end-to-end CNN classification on tb_label close the viability gap?

## Prior Internal Experiments

The project has an extensive experimental chain directly relevant to this question:

**R3 (book-encoder-bias)**: Established CNN spatial signal on structured (20,2) book input. Reported R²=0.132, but 9D proved this was inflated ~36% by test-as-validation leakage. True proper-validation R²=0.084. Conv1d(2→59→59), 12,128 params.

**9D (r3-reproduction-pipeline-comparison)**: Root-caused the 9B/9C failures. The data was byte-identical — the problem was missing TICK_SIZE normalization (÷0.25 on prices) and per-fold z-scoring instead of per-day z-scoring on sizes. Proper-validation R²=0.084 is the ground truth CNN regression performance.

**9E (hybrid-model-corrected)**: The direct predecessor and motivation for this experiment. Results:
- CNN regression R²=0.089 (3rd independent reproduction, matches 9D's 0.084)
- XGBoost classification accuracy: 0.419
- Expectancy: **-$0.37/trade** (base $3.74 RT), PF=0.924
- Gross edge: $3.37/trade, consumed by $3.74 costs
- Win rate: 51.3%, breakeven: 53.3% (+2.0pp needed)
- Profitable only under optimistic costs ($2.49 RT): +$0.88/trade
- volatility_50 dominated feature importance (19.9 gain, 2× next feature)
- CNN embedding features ranked ~10th in XGBoost importance

**FYE (full-year-export)**: Produced the full-year dataset. 251 trading days, 1,160,150 bars, 149 columns, Parquet with zstd compression, 255.7 MB. Stored in S3 artifact store (needs `artifact-store hydrate` before use). All 10 success criteria passed. Schema includes `book_snap_0..39` (40 columns for CNN), `tb_label`, `tb_exit_type`, `tb_bars_held`, and 62 Track A features. Forward returns renamed to `fwd_return_N` in Parquet (not `return_N`).

**R7 (oracle-expectancy)**: The oracle ceiling. $4.00/trade, PF=3.30, WR=64.3%, Sharpe=0.362. Triple barrier (target=10, stop=5, vol_horizon=500).

**9B (hybrid-model-training)**: The broken predecessor. CNN R²=-0.002 due to missing normalization. Demonstrated that XGBoost can learn regime identification (acc=0.41) even without CNN signal.

## Current Infrastructure

### Data Pipeline
- **Full-year Parquet dataset**: `.kit/results/full-year-export/*.parquet` (251 files, S3-stored). Column names: `book_snap_0..39`, `tb_label`, `tb_exit_type`, `tb_bars_held`, `fwd_return_1/5/20/100`, plus 62 Track A features. Note: book snapshot columns are `book_snap_N` (not `bid_price_offset_N`/`ask_price_offset_N` as in the old CSV format).
- **19-day CSV reference**: `.kit/results/hybrid-model/time_5s.csv` (87,970 bars). All prior CNN experiments used this.
- **bar_feature_export**: C++ binary that generates Parquet from raw .dbn.zst files. 77s for full year (11-way parallel). Fast enough for regeneration.

### Training Infrastructure
- **Python scripts from 9E**: `.kit/results/hybrid-model-corrected/run_experiment.py` and `assemble_metrics.py` — implement CNN regression + frozen embedding + XGBoost pipeline with corrected normalization.
- **Dependencies**: `requirements.txt` has numpy, polars, scipy, scikit-learn, xgboost. **PyTorch is NOT in requirements.txt** — prior experiments installed it ad-hoc. The new experiment will need `torch` explicitly.
- **Docker/ECR/EBS pipeline**: Verified E2E (2026-02-21). Dockerfile + ec2-bootstrap.sh can launch GPU instances. However, 12K-param CNN on CPU is estimated at 30-60s per CPCV split (135 total = 65-110 min), so GPU may not be necessary.

### Normalization Protocol (Verified Across 3 Experiments)
1. Channel 0 (book prices): Divide by TICK_SIZE=0.25 → integer tick offsets. Do NOT z-score.
2. Channel 1 (book sizes): log1p → per-day z-score (day mean/std). NOT per-fold.
3. Non-spatial features: Per-fold z-score using train-fold stats. NaN → 0.0.

### CNN Architecture (Locked)
```
Conv1d(2→59, k=3, pad=1) + BN(59) + ReLU
Conv1d(59→59, k=3, pad=1) + BN(59) + ReLU
AdaptiveAvgPool1d(1) → (B, 59)
Linear(59, 16) + ReLU
Linear(16, 3)           # classification head (replaces Linear(16,1) regression)
Total: ~12,162 params
```

### Validation Infrastructure
- Prior experiments used 5-fold expanding-window CV on 19 days (high variance, fold 3 always weakest).
- The spec proposes CPCV (N=10, k=2) on 201 development days + 50-day holdout. This is a major upgrade: 45 splits, 9 backtest paths, PBO, DSR.
- Purge (500 bars) + embargo (4,600 bars ≈ 1 day) to prevent label leakage.

## Known Failure Modes

1. **TICK_SIZE normalization omission**: Caused 9B and 9C failures. Without ÷0.25 on price offsets, CNN train R² collapses to ~0.001. This is the #1 pipeline killer and must be verified first.

2. **Per-fold vs per-day z-scoring**: Using per-fold z-scoring on sizes destroyed CNN signal. Per-day granularity is required.

3. **Test-as-validation leakage**: R3 used test set for early stopping, inflating R² by 36%. All CPCV splits must use internal 80/20 train/val split for early stopping.

4. **CrossEntropyLoss optimization landscape**: This is NEW territory. The CNN was always trained on MSE regression. Switching to CrossEntropyLoss on a 3-class target fundamentally changes what the penultimate layer learns. No prior experiment tests this. If the model fails to learn, it may need architecture changes (not just hyperparameter tuning).

5. **Label distribution skew**: The asymmetric 10:5 target/stop ratio may produce imbalanced classes. 9E showed fold 3 (Oct 2022) had 45.5% neutral class. Inverse-frequency class weights are specified as a treatment variable.

6. **Regime non-stationarity**: 2022 contains diverse regimes (Jan peak, Mar-Jun drawdown, Jun-Oct range, Nov-Dec consolidation). The 19-day prior experiments sampled one regime. 251 days span all of them, which may help or hurt.

7. **Book snapshot column mapping**: The Parquet schema uses `book_snap_0..39`, which must be correctly reshaped to (N, 2, 20). The mapping from `book_snap_N` to (channel, level) must match the original `bid_price_offset_N`/`bid_size_N`/`ask_price_offset_N`/`ask_size_N` convention.

8. **S3 artifact hydration**: The Parquet files are S3 symlinks. Must run `artifact-store hydrate` before data loading.

## Key Codebase Entry Points

| File/Path | Description |
|-----------|-------------|
| `.kit/results/full-year-export/*.parquet` | Full-year dataset (S3, needs hydration) |
| `.kit/results/full-year-export/.s3-manifest.json` | S3 artifact manifest for 251 Parquet files |
| `.kit/results/full-year-export/metrics.json` | Full-year export metrics (schema, column names) |
| `.kit/results/hybrid-model-corrected/run_experiment.py` | 9E training script (CNN regression + XGBoost) |
| `.kit/results/hybrid-model-corrected/analysis.md` | 9E results (the baseline to beat) |
| `.kit/experiments/e2e-cnn-classification.md` | **Already-written experiment spec** (detailed, 577 lines) |
| `.kit/results/hybrid-model/time_5s.csv` | 19-day CSV reference data |
| `requirements.txt` | Python deps (missing torch) |

## Architectural Priors

The CNN spatial encoding approach is well-motivated for this problem:

- **Input structure**: The limit order book is a spatial array — 10 bid levels + 10 ask levels, each with (price_offset, size). This is inherently 2D spatial structure where adjacency matters (level 3 is between levels 2 and 4).
- **Conv1d on (2, 20)**: Treats the 20-level book as a 1D spatial sequence with 2 channels. Kernel size 3 captures local level interactions. This preserved spatial structure that flattened MLPs destroyed (R²=0.084 vs 0.007).
- **Classification vs regression**: The CNN currently learns return-variance features (MSE loss). End-to-end classification should learn class-boundary features (where in book-state space do the three outcomes concentrate?). This is a different optimization objective that could be more or less effective.

**Why MLP is NOT appropriate here**: R2 proved that flattened (40-dim) MLP achieves R²=0.007 — the spatial adjacency structure in the book is essential. CNN is the correct inductive bias for this data.

**Why end-to-end is architecturally promising**: The 9E pipeline has a regression→classification bottleneck. The CNN optimizes for MSE on returns (continuous), then the 16-dim embedding is frozen and fed to XGBoost on tb_label (discrete 3-class). This means:
1. The CNN never sees the classification objective during training.
2. The 16-dim features capture return-variance directions, not class boundaries.
3. XGBoost must learn to classify from an embedding that wasn't designed for classification.

End-to-end CrossEntropyLoss training eliminates all three issues. The 16-dim penultimate layer should learn spatial patterns that discriminate between {target hit, stop hit, expired} outcomes, not return prediction patterns.

## External Context

End-to-end classification over multi-stage regression→classification pipelines is standard in deep learning practice. The information loss from freezing embeddings trained on a proxy objective is well-documented. Key applicable findings:

- **Task-specific fine-tuning > frozen features**: Across vision, NLP, and tabular domains, end-to-end training on the target loss consistently outperforms frozen-embedding approaches (the "feature extraction" paradigm).
- **Small model + large data**: 12K params on 1.16M samples is a very favorable parameter-to-sample ratio (~96 samples per parameter). Overfitting risk is minimal; underfitting is the concern.
- **CPCV for financial data**: Combinatorial Purged Cross-Validation (de Prado, AFML) is the appropriate validation framework for this problem class. Purge + embargo + uniqueness-weighted samples address the specific challenges of overlapping financial labels.
- **Class-weighted CrossEntropyLoss**: Standard approach for imbalanced classification. The inverse-frequency weighting specified in the experiment is the most common and well-understood approach.

This problem class (low-signal financial classification with order book spatial structure) is niche. The CNN-on-book approach has been explored in quantitative finance research but typically with much larger architectures. The 12K-param CNN is appropriate for the low-signal regime (R²=0.089 on regression ≈ very weak signal for classification).

## Constraints and Considerations

### Compute
- 12K-param CNN, ~925K training bars per CPCV split, 45 splits × 3 configs × 2 weight schemes = up to 270 training runs.
- Estimated 30-60s per CNN fit on CPU → 2.25-4.5 hours for all CNN runs.
- The spec budgets 4 hours wall-clock max. This is tight. If CNN training exceeds 2 min/split, cloud GPU is recommended.
- Docker/ECR/EBS pipeline is verified — cloud escalation is feasible.

### Data
- Full-year Parquet files are in S3. Must hydrate before use (~256 MB download).
- Parquet column names differ from CSV (book_snap_N vs bid/ask_price_offset_N). Column mapping must be carefully verified.
- 50-day holdout (days 202-251, ~Nov-Dec 2022) is in a specific regime (year-end, lower vol). Not ideal but provides true OOS estimate.

### Statistical
- CPCV produces 9 backtest paths from 45 splits. PBO measures overfitting probability. This is vastly more robust than the 5-fold approach used in all prior experiments.
- The 2pp win rate gap (51.3% → 53.3%) is the target. This is a small but meaningful improvement — a 4% relative increase in win rate.
- The experiment tests 6 configurations (3 models × 2 weightings), producing a Deflated Sharpe Ratio correction.

### PyTorch Dependency
- `torch` is not in `requirements.txt`. The RUN agent must install it. Prior experiments handled this ad-hoc. Should be formalized.

## Recommendation

The FRAME agent should focus on **executing the existing spec** (`.kit/experiments/e2e-cnn-classification.md`), which is already comprehensive and well-designed. The spec:

1. **Already exists** and is detailed (577 lines, 12 success criteria, 5 outcome branches, MVE gates, CPCV protocol, holdout design, cost sensitivity, confusion matrix).
2. **Addresses the exact bottleneck**: regression→classification handoff eliminated by end-to-end CrossEntropyLoss.
3. **Leverages the full-year dataset**: 13× more data than prior experiments.
4. **Uses rigorous validation**: CPCV with purge/embargo + 50-day holdout.

**Key risks to flag:**
- CrossEntropyLoss on the CNN has never been tested. If the model fails to learn (Outcome C), it may need architecture modifications (dropout, different capacity) — which would require a follow-up experiment.
- The spec's compute estimate (65-110 min) may be optimistic. 45 splits × multiple configs × 100 max epochs could push past 4 hours. Cloud GPU escalation path should be ready.
- S3 hydration must happen before the RUN phase.
- `torch` installation must be handled.

**What NOT to change**: The CNN architecture (Conv1d 2→59→59) is locked — it has 3 independent reproductions at R²≈0.084-0.089. The normalization protocol is verified. The triple barrier parameters are locked by R7. The spec already accounts for all of these.

**Most productive focus**: Get the experiment running on the full-year data with correct infrastructure (hydration, torch install, Parquet column mapping). The spec itself is sound.
