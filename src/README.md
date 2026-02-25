# src/ — MBO-DL Source (Header-Only C++20)

All modules are header-only `.hpp` files.

| File | Module | Description |
|------|--------|-------------|
| `book_builder.hpp` | book_builder | BookSnapshot struct (incl. `trade_count`) + BookBuilder (MBO → L2 book reconstruction) |
| `feature_encoder.hpp` | feature_encoder | 194-dim feature encoding from BookSnapshot windows |
| `oracle_labeler.hpp` | oracle_labeler | Stateless oracle with lookahead for position-dependent labels |
| `trajectory_builder.hpp` | trajectory_builder | TrainingSample struct + position state management |
| `mlp_model.hpp` | MLP | Multi-layer perceptron (libtorch) |
| `cnn_model.hpp` | CNN | Spatial + temporal convolution model (libtorch) |
| `gbt_features.hpp` | GBT features | 16 hand-crafted features for XGBoost |
| `gbt_model.hpp` | GBT model | XGBoost C API wrapper |
| `training_loop.hpp` | training_loop | Neural network overfit training loop |
| `serialization.hpp` | serialization | Checkpoint save/load (libtorch + XGBoost) + ONNX export |

### `backtest/` — Backtesting & Oracle Infrastructure

| File | Description |
|------|-------------|
| `oracle_replay.hpp` | Oracle replay engine for multi-day backtesting |
| `oracle_expectancy_report.hpp` | Expectancy report generation + JSON output |
| `oracle_comparison.hpp` | Oracle parameter comparison utilities |
| `triple_barrier.hpp` | Triple barrier labeling (target/stop/take-profit) |
| `trade_record.hpp` | Trade record struct for backtest output |
| `backtest_result_io.hpp` | Backtest result I/O (serialization) |
| `multi_day_runner.hpp` | Multi-day backtest orchestration |
| `success_criteria.hpp` | Statistical success criteria for backtest validation |
| `regime_stratification.hpp` | Regime-based stratification for analysis |
| `rollover.hpp` | Futures rollover calendar |
| `execution_costs.hpp` | Transaction cost model (commission, spread, slippage) |

### `bars/` — Bar Construction

| File | Description |
|------|-------------|
| `bar_builder_base.hpp` | Base class for bar builders (time, tick, dollar, volume) |
| `tick_bar_builder.hpp` | Tick bar builder — accumulates `trade_count` from snapshots, closes at threshold |

### `features/` — Feature Computation

| File | Description |
|------|-------------|
| `bar_features.hpp` | Bar-level feature computation for export |

## Pipeline order

```
book_builder → feature_encoder → trajectory_builder + oracle_labeler → {MLP, CNN, GBT} → serialization
```
