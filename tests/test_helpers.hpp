#pragma once

// test_helpers.hpp — shared test utilities for MLP and CNN model tests

#include "feature_encoder.hpp"     // FEATURE_DIM, W
#include "trajectory_builder.hpp"  // TrainingSample

#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace test_helpers {

constexpr int NUM_CLASSES = 5;

// Create a random input tensor of shape (B, W, FEATURE_DIM).
inline torch::Tensor make_input(int batch_size, int seed = 42) {
    torch::manual_seed(seed);
    return torch::randn({batch_size, W, FEATURE_DIM});
}

// Generate N synthetic TrainingSamples with deterministic labels.
// Label is (sample_index % num_classes) so the distribution is balanced.
inline std::vector<TrainingSample> make_synthetic_samples(int n, int seed = 42) {
    torch::manual_seed(seed);
    std::vector<TrainingSample> samples;
    samples.reserve(n);

    for (int i = 0; i < n; ++i) {
        TrainingSample s;
        s.label = i % NUM_CLASSES;
        s.window.resize(W);

        // Fill with deterministic pseudo-random features.
        // Use a simple hash-like scheme so each sample has distinct features.
        for (int t = 0; t < W; ++t) {
            for (int f = 0; f < FEATURE_DIM; ++f) {
                float val = std::sin(static_cast<float>(i * 1000 + t * 10 + f) * 0.01f);
                s.window[t][f] = val;
            }
        }

        samples.push_back(std::move(s));
    }
    return samples;
}

// Count total parameters in a module.
inline int64_t count_parameters(const torch::nn::Module& module) {
    int64_t total = 0;
    for (const auto& p : module.parameters()) {
        total += p.numel();
    }
    return total;
}

}  // namespace test_helpers
