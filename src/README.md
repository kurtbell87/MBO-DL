# src/ — MBO-DL Source (Header-Only C++20)

All modules are header-only `.hpp` files.

| File | Module | Description |
|------|--------|-------------|
| `book_builder.hpp` | book_builder | BookSnapshot struct + BookBuilder (MBO → L2 book reconstruction) |
| `feature_encoder.hpp` | feature_encoder | 194-dim feature encoding from BookSnapshot windows |
| `oracle_labeler.hpp` | oracle_labeler | Stateless oracle with lookahead for position-dependent labels |
| `trajectory_builder.hpp` | trajectory_builder | TrainingSample struct + position state management |
| `mlp_model.hpp` | MLP | Multi-layer perceptron (libtorch) |
| `cnn_model.hpp` | CNN | Spatial + temporal convolution model (libtorch) |
| `gbt_features.hpp` | GBT features | 16 hand-crafted features for XGBoost |
| `gbt_model.hpp` | GBT model | XGBoost C API wrapper |
| `training_loop.hpp` | training_loop | Neural network overfit training loop |
| `serialization.hpp` | serialization | Checkpoint save/load (libtorch + XGBoost) + ONNX export |

## Pipeline order

```
book_builder → feature_encoder → trajectory_builder + oracle_labeler → {MLP, CNN, GBT} → serialization
```
