#pragma once

#include "mlp_model.hpp"
#include "cnn_model.hpp"
#include "trajectory_builder.hpp"  // TrainingSample
#include "feature_encoder.hpp"     // W, FEATURE_DIM

#include <torch/torch.h>

#include <cstring>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// OverfitResult — output of overfit_mlp()
// ---------------------------------------------------------------------------
struct OverfitResult {
    float final_accuracy = 0.0f;
    float final_loss = 0.0f;
    int epochs_to_target = -1;
    bool success = false;
};

// ---------------------------------------------------------------------------
// samples_to_tensors — convert TrainingSamples to (input, label) tensor pair
// ---------------------------------------------------------------------------
inline std::pair<torch::Tensor, torch::Tensor> samples_to_tensors(
    const std::vector<TrainingSample>& samples) {
    int n = static_cast<int>(samples.size());
    auto input_tensor = torch::zeros({n, W, FEATURE_DIM});
    auto label_tensor = torch::zeros({n}, torch::kLong);
    auto input_acc = input_tensor.accessor<float, 3>();
    for (int i = 0; i < n; ++i) {
        label_tensor[i] = samples[i].label;
        for (int t = 0; t < W; ++t) {
            std::memcpy(&input_acc[i][t][0], samples[i].window[t].data(),
                        FEATURE_DIM * sizeof(float));
        }
    }
    return {input_tensor, label_tensor};
}

// ---------------------------------------------------------------------------
// train_model — shared training loop for overfit_mlp/overfit_cnn
//
// Template on ModelType (MLPModel or CNNModel).
// clip_gradients: if true, applies clip_grad_norm_(max_norm=1.0) before step.
// ---------------------------------------------------------------------------
template <typename ModelType>
OverfitResult train_model(
    ModelType& model,
    const torch::Tensor& input_tensor,
    const torch::Tensor& label_tensor,
    int max_epochs,
    float target_accuracy,
    float lr,
    bool clip_gradients)
{
    int n = static_cast<int>(input_tensor.size(0));

    torch::optim::Adam optimizer(model->parameters(),
                                  torch::optim::AdamOptions(lr));

    OverfitResult result;

    for (int epoch = 0; epoch < max_epochs; ++epoch) {
        optimizer.zero_grad();

        auto output = model->forward(input_tensor);
        auto loss = torch::nn::functional::cross_entropy(output, label_tensor);

        loss.backward();
        if (clip_gradients) {
            torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
        }
        optimizer.step();

        auto preds = output.argmax(1);
        float accuracy = preds.eq(label_tensor).sum().template item<float>() / static_cast<float>(n);
        float loss_val = loss.template item<float>();

        result.final_loss = loss_val;
        result.final_accuracy = accuracy;

        if (accuracy >= target_accuracy && result.epochs_to_target < 0) {
            result.epochs_to_target = epoch + 1;
            result.success = true;
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// overfit_mlp — train an MLP to overfit a small set of TrainingSamples
//
// Deterministic: sets torch::manual_seed(42) at the start.
// ---------------------------------------------------------------------------
inline OverfitResult overfit_mlp(
    const std::vector<TrainingSample>& samples,
    int max_epochs = 500,
    float target_accuracy = 0.95f,
    float lr = 1e-3f)
{
    torch::manual_seed(42);

    constexpr int num_classes = 5;
    constexpr int input_dim = W * FEATURE_DIM;

    auto [input_tensor, label_tensor] = samples_to_tensors(samples);

    MLPModel model(input_dim, num_classes, 0.0f);
    model->train();

    return train_model(model, input_tensor, label_tensor,
                       max_epochs, target_accuracy, lr, /*clip_gradients=*/false);
}

// ---------------------------------------------------------------------------
// overfit_cnn — train a CNN to overfit a small set of TrainingSamples
//
// Deterministic: sets torch::manual_seed(42) at the start.
// Uses gradient clipping (max_norm=1.0).
// ---------------------------------------------------------------------------
inline OverfitResult overfit_cnn(
    const std::vector<TrainingSample>& samples,
    int max_epochs = 500,
    float target_accuracy = 0.95f,
    float lr = 1e-3f)
{
    torch::manual_seed(42);

    constexpr int num_classes = 5;

    auto [input_tensor, label_tensor] = samples_to_tensors(samples);

    CNNModel model(num_classes);
    model->train();

    return train_model(model, input_tensor, label_tensor,
                       max_epochs, target_accuracy, lr, /*clip_gradients=*/true);
}
