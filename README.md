# MES Microstructure Model Suite

End-to-end quantitative research platform for discovering and validating intraday trading signals from CME Micro E-mini S&P 500 (MES) market-by-order (L3) data. Built from raw tick data through feature engineering, model training, and statistically rigorous backtesting with realistic transaction costs.

## Key Results

- **30+ systematic experiments** across bar construction, information decomposition, encoder architecture search, label design, and execution simulation
- **Statistically validated positive expectancy:** $1.81/trade (95% CI [$1.46, $2.16]), PBO = 6.7%, p < 1e-13 via combinatorially purged cross-validation (45 splits, 251 trading days, 1.16M bars)
- **Sequential execution simulation:** $2.50/trade, 162 trades/day, annualized Sharpe 2.27, Calmar 2.16
- **Edge source identified:** payoff asymmetry (19:7 tick target/stop ratio, breakeven accuracy 34.6%) combined with volatility-conditioned barrier reachability filtering — not directional prediction (win rate ~50%)

## Data Pipeline (C++20)

High-performance pipeline processing 49 GB of Databento MBO (L3) market-by-order data — 312 daily `.dbn.zst` files covering full-year 2022 MES order flow.

**Pipeline stages:**
1. **Order book reconstruction** — Full L3 book rebuild from MBO messages (add/modify/cancel/trade) using `databento::DbnFileStore` API. Maintains price-level depth with nanosecond precision timestamps
2. **Bar construction** — Configurable bar types: time bars (1s–60s), tick bars (25–3000 trades), dollar bars ($25K–$50M), volume bars (50–500 contracts). Validated genuine event-bar semantics (non-zero daily count variance)
3. **Feature computation** — 20 hand-crafted microstructure features per bar from 10-level book snapshots (see Feature Engineering below)
4. **Triple barrier labeling** — Bidirectional (independent long/short race evaluation) and legacy modes. Configurable target/stop geometry and time/volume horizons. Oracle expectancy validation
5. **Parquet export** — 149–152 column schema with zstd compression. Full-year export: 251 RTH days, 1,160,150 bars, 255.7 MB. Parallelized (77s wall-clock, 11-way)

**C++ tooling:**
- `bar_feature_export` — Configurable bar construction + feature extraction + triple barrier labeling. CLI flags for bar type, geometry (`--target`, `--stop`), time horizon (`--max-time-horizon`), volume horizon, legacy/bidirectional labels
- `oracle_expectancy` — Oracle strategy P&L with parameterized geometry and cost models
- 1,144+ unit tests, 22 integration tests

## Feature Engineering

20 features engineered from 10-level order book microstructure, computed per bar:

| Category | Features | Description |
|----------|----------|-------------|
| **Book pressure** | `weighted_imbalance`, `level_{1..5}_imbalance` | Bid/ask size imbalance at each price level, volume-weighted aggregate |
| **Depth profile** | `total_bid_depth`, `total_ask_depth`, `depth_ratio` | Aggregate liquidity on each side of the book |
| **Spread & pricing** | `spread`, `midprice`, `microprice` | Inside spread, mid, volume-weighted mid |
| **Volatility** | `volatility_50` (50-bar rolling) | Return volatility — dominant feature (49.7% XGBoost gain share) |
| **Activity** | `trade_count`, `volume` | Per-bar trade flow metrics |
| **Forward returns** | `fwd_return_{1,5,10,20,50,100}` | Multi-horizon forward mid returns for label construction |

Features are computed entirely in C++ from raw MBO data. Python never touches raw order flow — it operates only on pre-computed Parquet output.

## Modeling

### Architecture: Two-Stage XGBoost

Arrived at through systematic elimination of 8 alternative architectures across 30+ experiments:

- **Stage 1 (Reachability filter):** Binary classifier predicting whether a triple barrier's target or stop will be hit before timeout. Filters out hold/timeout bars (reduces from 100% to ~85% trade rate)
- **Stage 2 (Direction):** Binary classifier predicting long vs. short among Stage 1 filtered bars

### Architecture Search Summary

| Architecture | Experiments | Verdict | Key Finding |
|-------------|------------|---------|-------------|
| Event-driven bars (tick, dollar, volume) | R1, R3b, R4b | Refuted | Time bars equivalent or better; subordination theory poor fit for MES |
| Spatial CNN encoder | R2, R3, 9B–9E | Closed | CNN R²=0.089 on returns, but 5.9pp worse than XGBoost for classification |
| Temporal encoder (SSM/AR) | R4, R4b, R4c, R4d | Closed | 0/168+ dual-threshold passes across 7 bar types, 0.14s–300s. MES is martingale |
| Message encoder | R2 | Closed | Book state is sufficient statistic for message sequence |
| CNN+GBT hybrid | 9B, 9E, 10 | Closed | Hand-crafted features beat CNN embeddings for classification |
| 3-class direct | CPCV, tuning | Superseded | Two-stage decomposition unlocks 300x trade volume |
| High-ratio geometry | Phase 1, 1h | Confirmed | 19:7 target/stop lowers breakeven WR to 34.6% |
| Payoff asymmetry | 2-class, CPCV-corrected | **Confirmed** | Edge is structural: payoff ratio, not prediction accuracy |

### CNN Spatial Signal Investigation

Dedicated 7-experiment diagnostic chain (9B → 9C → 9D → 9E → R3b → R3b-genuine → 10) to resolve a CNN reproduction failure. Root cause: missing tick-size normalization on prices + per-day z-scoring on sizes + test-as-validation leakage inflating R² by 36%. Proper-validation CNN R²=0.089 — real but insufficient for classification. Institutional lesson: always specify normalization as concrete operations in experiment specs.

## Validation Framework

### Combinatorially Purged Cross-Validation (CPCV)

- **45 test paths** (10 groups, k=2) per de Prado (2018)
- **Purging + embargo:** eliminates train/test leakage from overlapping barrier windows
- **Probability of Backtest Overfitting (PBO):** 6.7% (3/45 paths negative)
- **Walk-forward confirmation:** 3-fold temporal, directional accuracy 45.6% (consistent with CPCV)
- **Holdout:** 50 out-of-sample days, expectancy $1.46/trade

### Transaction Cost Modeling

Three cost tiers benchmarked against CME MES fee schedule:
- **Base:** $2.49 RT (commission $1.24 + 1-tick spread) — validated profitable
- **Conservative:** $3.74 RT — breakeven
- **Pessimistic:** $4.99 RT — not viable

Break-even round-trip cost: $4.30 (1.9x base costs — structural margin).

### Statistical Rigor

- All experiments use pre-registered hypotheses with explicit success criteria
- Holm-Bonferroni correction for multiple comparisons (200+ tests across R4 chain)
- Effect sizes reported alongside p-values
- Dual threshold gates: both statistical significance AND economic materiality required
- Abort criteria prevent pursuing unviable hypotheses

## Infrastructure

| Component | Technology |
|-----------|-----------|
| Data pipeline | C++20, CMake, Databento C++ SDK, Apache Arrow/Parquet |
| Modeling | Python, XGBoost, PyTorch (CNN experiments), scikit-learn |
| Validation | Custom CPCV implementation, walk-forward, PBO |
| Cloud compute | AWS EC2 (spot), Docker/ECR, EBS snapshots, S3 artifact store |
| Testing | Google Test (1,144+ unit tests), pytest |
| Data | Databento MBO L3 (49 GB, 312 daily files, full-year 2022 MES) |

## Project Structure

```
src/                    # C++20 data pipeline, feature computation, labeling
tools/                  # CLI binaries (bar_feature_export, oracle_expectancy)
tests/                  # 1,144+ unit tests + 22 integration tests
scripts/                # Python training, validation, analysis
.kit/experiments/       # 37 experiment specifications (hypothesis, SC, protocol)
.kit/results/           # Experiment outputs (metrics, analysis, Parquet)
DATA/                   # 49 GB Databento MBO L3 (.dbn.zst, not tracked)
```

## Research Log

Full experiment history with hypotheses, quantitative results, lessons learned, and decision rationale maintained in `.kit/RESEARCH_LOG.md`. Each experiment follows a structured protocol: Survey (literature + codebase review) → Frame (hypothesis + success criteria) → Run (execution) → Read (analysis + verdict).

## Tech Stack

- **Languages:** C++20 (data pipeline, 10K+ LOC), Python (modeling + analysis)
- **ML:** XGBoost, PyTorch, scikit-learn, pandas, numpy
- **Data:** Apache Arrow/Parquet, Databento SDK, zstd compression
- **Infrastructure:** CMake, Docker, AWS (EC2/ECR/EBS/S3), GitHub Actions
- **Testing:** Google Test, pytest, CPCV, walk-forward validation
- **Statistics:** Holm-Bonferroni correction, paired t-tests, PBO, deflated Sharpe ratio
