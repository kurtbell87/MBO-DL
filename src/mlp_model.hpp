#pragma once

#include <torch/torch.h>

#include "feature_encoder.hpp"  // W, FEATURE_DIM

// ---------------------------------------------------------------------------
// Compat shim: torch C++ clip_grad_norm_ returns double, but some code
// expects a Tensor-like .item<T>() API. This wrapper adapts the return type.
// ---------------------------------------------------------------------------
namespace torch { namespace nn { namespace utils {

struct ClipGradNormResult {
    double value;
    template <typename T>
    T item() const { return static_cast<T>(value); }
};

// Custom rvalue overload: computes norms in FP64 for precision, then clips in-place.
// Returns a ClipGradNormResult (with .item<T>() support).
inline ClipGradNormResult clip_grad_norm_(
    std::vector<Tensor>&& parameters,
    double max_norm,
    double norm_type = 2.0,
    bool error_if_nonfinite = false) {
    // Collect params that have gradients
    std::vector<Tensor> params_with_grad;
    for (const auto& p : parameters) {
        if (p.grad().defined()) {
            params_with_grad.push_back(p);
        }
    }
    if (params_with_grad.empty()) return {0.0};

    // Compute per-parameter norms in FP64 for precision
    double total_norm_sq = 0.0;
    for (const auto& p : params_with_grad) {
        auto grad_d = p.grad().data().to(torch::kFloat64);
        double pnorm = grad_d.norm(norm_type).item<double>();
        total_norm_sq += std::pow(pnorm, norm_type);
    }
    double total_norm = std::pow(total_norm_sq, 1.0 / norm_type);

    // Clip
    double clip_coef = max_norm / (total_norm + 1e-6);
    if (clip_coef < 1.0) {
        for (auto& p : params_with_grad) {
            p.grad().data().mul_(clip_coef);
        }
    }

    return {total_norm};
}

}}} // namespace torch::nn::utils

// ---------------------------------------------------------------------------
// MLPModelImpl — 4-layer MLP for action classification
//
// Architecture (from spec):
//   Flatten: (B, 600, 194) → (B, 116400)
//   Linear 116400 → 512, ReLU, Dropout
//   Linear 512 → 256, ReLU, Dropout
//   Linear 256 → 128, ReLU, Dropout
//   Linear 128 → 5
// ---------------------------------------------------------------------------
struct MLPModelImpl : torch::nn::Module {
    MLPModelImpl(int input_dim = W * FEATURE_DIM,
                 int num_classes = 5,
                 float dropout_rate = 0.1f)
        : fc1(register_module("fc1", torch::nn::Linear(input_dim, 512))),
          bn1(register_module("bn1", torch::nn::BatchNorm1d(512))),
          fc2(register_module("fc2", torch::nn::Linear(512, 256))),
          bn2(register_module("bn2", torch::nn::BatchNorm1d(256))),
          fc3(register_module("fc3", torch::nn::Linear(256, 128))),
          bn3(register_module("bn3", torch::nn::BatchNorm1d(128))),
          fc4(register_module("fc4", torch::nn::Linear(128, num_classes))),
          drop1(register_module("drop1", torch::nn::Dropout(dropout_rate))),
          drop2(register_module("drop2", torch::nn::Dropout(dropout_rate))),
          drop3(register_module("drop3", torch::nn::Dropout(dropout_rate)))
    {}

    torch::Tensor forward(torch::Tensor x) {
        // x: (B, W, FEATURE_DIM) → flatten to (B, W * FEATURE_DIM)
        x = x.flatten(1);
        x = drop1(torch::relu(bn1(fc1(x))));
        x = drop2(torch::relu(bn2(fc2(x))));
        x = drop3(torch::relu(bn3(fc3(x))));
        x = fc4(x);
        return x;
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr};
    torch::nn::BatchNorm1d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
    torch::nn::Dropout drop1{nullptr}, drop2{nullptr}, drop3{nullptr};
};

TORCH_MODULE(MLPModel);
