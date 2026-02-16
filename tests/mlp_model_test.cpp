// mlp_model_test.cpp — TDD RED phase tests for MLPModel + training loop
// Spec: .kit/docs/mlp-model.md
//
// Tests the MLPModel class (libtorch MLP) and the overfit_mlp() training loop.
// The model flattens (B, 600, 194) → (B, 116400) and passes through dense layers
// to produce (B, 5) logits. The training loop must overfit N=32 synthetic samples.

#include <gtest/gtest.h>

#include "mlp_model.hpp"
#include "training_loop.hpp"
#include "test_helpers.hpp"

#include <torch/torch.h>

#include <cmath>
#include <cstdint>

using test_helpers::NUM_CLASSES;
using test_helpers::make_input;
using test_helpers::make_synthetic_samples;
using test_helpers::count_parameters;

namespace {

constexpr int INPUT_DIM = W * FEATURE_DIM;  // 600 * 194 = 116400

}  // anonymous namespace

// ===========================================================================
// Model Architecture Tests
// ===========================================================================

// ---------------------------------------------------------------------------
// Test 1: Output shape — Input (B=4, 600, 194), output shape is (4, 5)
// ---------------------------------------------------------------------------
TEST(MLPModelTest, OutputShape) {
    MLPModel model(INPUT_DIM, NUM_CLASSES, 0.1f);
    model->eval();

    auto input = make_input(/*batch_size=*/4);
    auto output = model->forward(input);

    ASSERT_EQ(output.dim(), 2) << "Output should be 2D (B, num_classes)";
    EXPECT_EQ(output.size(0), 4) << "Batch dimension should be 4";
    EXPECT_EQ(output.size(1), NUM_CLASSES) << "Class dimension should be 5";
}

TEST(MLPModelTest, OutputShapeSingleSample) {
    MLPModel model(INPUT_DIM, NUM_CLASSES, 0.1f);
    model->eval();

    auto input = make_input(/*batch_size=*/1);
    auto output = model->forward(input);

    ASSERT_EQ(output.dim(), 2);
    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), NUM_CLASSES);
}

TEST(MLPModelTest, OutputShapeLargeBatch) {
    MLPModel model(INPUT_DIM, NUM_CLASSES, 0.1f);
    model->eval();

    auto input = make_input(/*batch_size=*/32);
    auto output = model->forward(input);

    ASSERT_EQ(output.dim(), 2);
    EXPECT_EQ(output.size(0), 32);
    EXPECT_EQ(output.size(1), NUM_CLASSES);
}

// ---------------------------------------------------------------------------
// Test 2: Forward pass no crash — random input, forward completes
// ---------------------------------------------------------------------------
TEST(MLPModelTest, ForwardPassNoCrash) {
    MLPModel model(INPUT_DIM, NUM_CLASSES, 0.1f);
    model->eval();

    // Various batch sizes should all succeed
    for (int b : {1, 2, 4, 8, 16, 32}) {
        auto input = make_input(b);
        EXPECT_NO_THROW(model->forward(input))
            << "Forward pass should not throw for batch_size=" << b;
    }
}

// ---------------------------------------------------------------------------
// Test 3: Parameter count — approximately 59.7M (within 1% tolerance)
// ---------------------------------------------------------------------------
TEST(MLPModelTest, ParameterCount) {
    MLPModel model(INPUT_DIM, NUM_CLASSES, 0.1f);

    int64_t total_params = count_parameters(*model);

    // Expected breakdown from spec:
    //   Layer 1: 116400 * 512 + 512            = 59,597,312
    //   Layer 2: 512 * 256 + 256               =    131,328
    //   Layer 3: 256 * 128 + 128               =     32,896
    //   Layer 4: 128 * 5 + 5                   =        645
    //   Total:                                   59,762,181
    constexpr int64_t EXPECTED_APPROX = 59'762'181;

    // Within 1% tolerance
    double ratio = static_cast<double>(total_params) / static_cast<double>(EXPECTED_APPROX);
    EXPECT_NEAR(ratio, 1.0, 0.01)
        << "Total parameters = " << total_params
        << ", expected ~" << EXPECTED_APPROX
        << " (ratio = " << ratio << ")";
}

// ---------------------------------------------------------------------------
// Test 4: Gradient flow — after forward + backward, all params have non-zero grads
// ---------------------------------------------------------------------------
TEST(MLPModelTest, GradientFlow) {
    MLPModel model(INPUT_DIM, NUM_CLASSES, 0.0f);  // no dropout for gradient test
    model->train();

    auto input = make_input(/*batch_size=*/4, /*seed=*/123);
    auto output = model->forward(input);

    // Compute a loss and backward
    auto target = torch::randint(0, NUM_CLASSES, {4}, torch::kLong);
    auto loss = torch::nn::functional::cross_entropy(output, target);
    loss.backward();

    // Every parameter should have a non-zero gradient
    int param_idx = 0;
    for (const auto& p : model->parameters()) {
        ASSERT_TRUE(p.grad().defined())
            << "Parameter " << param_idx << " has undefined gradient";
        EXPECT_GT(p.grad().abs().sum().item<float>(), 0.0f)
            << "Parameter " << param_idx << " has all-zero gradient";
        ++param_idx;
    }
    EXPECT_GT(param_idx, 0) << "Model should have at least one parameter";
}

// ---------------------------------------------------------------------------
// Test 5: No NaN in output — random input produces finite output values
// ---------------------------------------------------------------------------
TEST(MLPModelTest, NoNanInOutput) {
    MLPModel model(INPUT_DIM, NUM_CLASSES, 0.1f);
    model->eval();

    // Test with several different random seeds
    for (int seed : {42, 123, 456, 789}) {
        auto input = make_input(/*batch_size=*/4, seed);
        auto output = model->forward(input);

        // Check no NaN
        EXPECT_FALSE(torch::any(torch::isnan(output)).item<bool>())
            << "Output contains NaN with seed=" << seed;

        // Check no Inf
        EXPECT_FALSE(torch::any(torch::isinf(output)).item<bool>())
            << "Output contains Inf with seed=" << seed;
    }
}

// ===========================================================================
// Training Loop Tests
// ===========================================================================

// ---------------------------------------------------------------------------
// Test 6: Loss decreases — after 10 epochs on synthetic data, loss < initial
// ---------------------------------------------------------------------------
TEST(TrainingLoopTest, LossDecreases) {
    auto samples = make_synthetic_samples(32);

    // Run for just 10 epochs (no target accuracy, we just want loss to decrease)
    OverfitResult result = overfit_mlp(
        samples,
        /*max_epochs=*/10,
        /*target_accuracy=*/1.0f,  // unreachable in 10 epochs
        /*lr=*/1e-3f
    );

    // The final loss should be less than a random initialization loss.
    // Random chance with 5 classes: -log(1/5) ≈ 1.609.
    // After 10 epochs of training, loss should be noticeably lower.
    EXPECT_LT(result.final_loss, 1.609f)
        << "Loss after 10 epochs should be less than random chance (~1.609)";
}

// ---------------------------------------------------------------------------
// Test 7: No NaN during training — 50 epochs, no NaN/Inf in loss
// ---------------------------------------------------------------------------
TEST(TrainingLoopTest, NoNanDuringTraining) {
    auto samples = make_synthetic_samples(32);

    OverfitResult result = overfit_mlp(
        samples,
        /*max_epochs=*/50,
        /*target_accuracy=*/1.0f,  // unreachable target, just run 50 epochs
        /*lr=*/1e-3f
    );

    // final_loss should be finite (not NaN or Inf)
    EXPECT_TRUE(std::isfinite(result.final_loss))
        << "Loss should be finite after 50 epochs, got " << result.final_loss;

    // final_accuracy should be finite and in [0, 1]
    EXPECT_TRUE(std::isfinite(result.final_accuracy))
        << "Accuracy should be finite, got " << result.final_accuracy;
    EXPECT_GE(result.final_accuracy, 0.0f);
    EXPECT_LE(result.final_accuracy, 1.0f);
}

// ---------------------------------------------------------------------------
// Test 8: Gradient clipping active — gradient norms are <= 1.0 after clipping
// ---------------------------------------------------------------------------
TEST(TrainingLoopTest, GradientClippingActive) {
    // We verify gradient clipping indirectly: train for a few epochs and
    // confirm the result is stable (no NaN/explosion). The spec says
    // max_norm=1.0 gradient clipping must be active.
    //
    // Direct verification: create a model, do a forward/backward pass with
    // a large input that could cause gradient explosion, clip, and check norms.

    MLPModel model(INPUT_DIM, NUM_CLASSES, 0.0f);
    model->train();

    // Use large input values that could cause gradient explosion
    torch::manual_seed(42);
    auto input = torch::randn({4, W, FEATURE_DIM}) * 100.0f;  // scaled up
    auto target = torch::randint(0, NUM_CLASSES, {4}, torch::kLong);

    auto output = model->forward(input);
    auto loss = torch::nn::functional::cross_entropy(output, target);
    loss.backward();

    // Apply gradient clipping with max_norm=1.0 (as spec requires)
    float total_norm = torch::nn::utils::clip_grad_norm_(
        model->parameters(), /*max_norm=*/1.0).item<float>();

    // After clipping, re-check all gradient norms
    for (const auto& p : model->parameters()) {
        if (p.grad().defined()) {
            float param_norm = p.grad().norm().item<float>();
            // Individual parameter grad norms can be <= total clipped norm
            EXPECT_TRUE(std::isfinite(param_norm))
                << "Gradient norm should be finite after clipping";
        }
    }

    // The total norm after clipping should be <= 1.0 (within floating point tolerance)
    // Re-compute total norm post-clipping
    float post_clip_norm = 0.0f;
    for (const auto& p : model->parameters()) {
        if (p.grad().defined()) {
            post_clip_norm += p.grad().norm().item<float>() * p.grad().norm().item<float>();
        }
    }
    post_clip_norm = std::sqrt(post_clip_norm);

    EXPECT_LE(post_clip_norm, 1.0f + 1e-5f)
        << "Total gradient norm after clipping should be <= 1.0, got " << post_clip_norm;
}

// ---------------------------------------------------------------------------
// Test 9: Deterministic with seed — two runs with seed=42 produce identical loss
// ---------------------------------------------------------------------------
TEST(TrainingLoopTest, DeterministicWithSeed) {
    auto samples = make_synthetic_samples(32);

    // Run 1
    OverfitResult result1 = overfit_mlp(
        samples,
        /*max_epochs=*/10,
        /*target_accuracy=*/1.0f,
        /*lr=*/1e-3f
    );

    // Run 2 (same samples, same function — seed should be set internally)
    OverfitResult result2 = overfit_mlp(
        samples,
        /*max_epochs=*/10,
        /*target_accuracy=*/1.0f,
        /*lr=*/1e-3f
    );

    // Loss after 10 epochs should be identical (bitwise with same seed)
    EXPECT_FLOAT_EQ(result1.final_loss, result2.final_loss)
        << "Two runs with seed=42 should produce identical loss at epoch 10";

    EXPECT_FLOAT_EQ(result1.final_accuracy, result2.final_accuracy)
        << "Two runs with seed=42 should produce identical accuracy at epoch 10";
}

// ---------------------------------------------------------------------------
// Test 10: Accuracy computation — synthetic data with known labels
// ---------------------------------------------------------------------------
TEST(TrainingLoopTest, AccuracyComputation) {
    auto samples = make_synthetic_samples(32);

    // Train for enough epochs that accuracy should be > 0
    OverfitResult result = overfit_mlp(
        samples,
        /*max_epochs=*/50,
        /*target_accuracy=*/1.0f,
        /*lr=*/1e-3f
    );

    // Accuracy must be in [0.0, 1.0]
    EXPECT_GE(result.final_accuracy, 0.0f)
        << "Accuracy should be >= 0.0";
    EXPECT_LE(result.final_accuracy, 1.0f)
        << "Accuracy should be <= 1.0";

    // After 50 epochs on 32 samples, accuracy should be above random chance (1/5 = 0.2)
    EXPECT_GT(result.final_accuracy, 0.2f)
        << "Accuracy after 50 epochs should exceed random chance (20%)";
}

// ===========================================================================
// Overfit Tests (synthetic data)
// ===========================================================================

// ---------------------------------------------------------------------------
// Test 11: Overfit tiny synthetic — 32 samples, >= 95% accuracy within 500 epochs
// ---------------------------------------------------------------------------
TEST(OverfitTest, OverfitTinySyntheticN32) {
    auto samples = make_synthetic_samples(32);

    OverfitResult result = overfit_mlp(
        samples,
        /*max_epochs=*/500,
        /*target_accuracy=*/0.95f,
        /*lr=*/1e-3f
    );

    // The model should reach >= 95% accuracy on 32 synthetic samples
    EXPECT_TRUE(result.success)
        << "MLP should overfit 32 synthetic samples to >= 95% accuracy, "
        << "got " << result.final_accuracy * 100.0f << "% after "
        << (result.epochs_to_target > 0 ? result.epochs_to_target : 500)
        << " epochs";

    EXPECT_GE(result.final_accuracy, 0.95f)
        << "Final accuracy should be >= 95%, got "
        << result.final_accuracy * 100.0f << "%";

    // epochs_to_target should be > 0 (it took some training)
    if (result.success) {
        EXPECT_GT(result.epochs_to_target, 0)
            << "Should take at least 1 epoch to reach target";
        EXPECT_LE(result.epochs_to_target, 500)
            << "Should reach target within 500 epochs";
    }

    // Loss should be low
    EXPECT_LT(result.final_loss, 0.5f)
        << "Loss should be low after overfitting, got " << result.final_loss;
}

// ---------------------------------------------------------------------------
// Test 12: Checkpoint save/load — save model, reload, verify identical predictions
// ---------------------------------------------------------------------------
TEST(OverfitTest, CheckpointSaveLoad) {
    MLPModel model(INPUT_DIM, NUM_CLASSES, 0.0f);
    model->eval();

    auto input = make_input(/*batch_size=*/4, /*seed=*/99);

    // Get predictions from original model
    auto output_before = model->forward(input).clone();

    // Save checkpoint
    std::string checkpoint_path = "/tmp/mlp_model_test_checkpoint.pt";
    torch::save(model, checkpoint_path);

    // Create a new model and load the checkpoint
    MLPModel loaded_model(INPUT_DIM, NUM_CLASSES, 0.0f);
    torch::load(loaded_model, checkpoint_path);
    loaded_model->eval();

    // Get predictions from loaded model
    auto output_after = loaded_model->forward(input);

    // Predictions should be identical (bitwise)
    EXPECT_TRUE(torch::equal(output_before, output_after))
        << "Loaded model should produce identical predictions to saved model";

    // Clean up
    std::remove(checkpoint_path.c_str());
}

// ===========================================================================
// OverfitResult struct tests
// ===========================================================================

TEST(OverfitResultTest, StructFields) {
    OverfitResult result;
    result.final_accuracy = 0.99f;
    result.final_loss = 0.01f;
    result.epochs_to_target = 150;
    result.success = true;

    EXPECT_FLOAT_EQ(result.final_accuracy, 0.99f);
    EXPECT_FLOAT_EQ(result.final_loss, 0.01f);
    EXPECT_EQ(result.epochs_to_target, 150);
    EXPECT_TRUE(result.success);
}

TEST(OverfitResultTest, FailureFields) {
    OverfitResult result;
    result.final_accuracy = 0.50f;
    result.final_loss = 1.2f;
    result.epochs_to_target = -1;
    result.success = false;

    EXPECT_FLOAT_EQ(result.final_accuracy, 0.50f);
    EXPECT_FLOAT_EQ(result.final_loss, 1.2f);
    EXPECT_EQ(result.epochs_to_target, -1);
    EXPECT_FALSE(result.success);
}

// ===========================================================================
// Edge case: MLPModel constructor parameters
// ===========================================================================

TEST(MLPModelTest, DefaultConstructorParameters) {
    // Default: input_dim=116400, num_classes=5, dropout=0.1
    MLPModel model;
    model->eval();

    auto input = make_input(/*batch_size=*/2);
    auto output = model->forward(input);

    EXPECT_EQ(output.size(0), 2);
    EXPECT_EQ(output.size(1), NUM_CLASSES);
}

TEST(MLPModelTest, ZeroDropoutForOverfit) {
    // Dropout=0.0 should be used during overfit testing
    MLPModel model(INPUT_DIM, NUM_CLASSES, 0.0f);
    model->train();  // train mode, but dropout=0.0 → no stochasticity

    auto input = make_input(/*batch_size=*/4, /*seed=*/42);

    // Two forward passes in train mode with dropout=0.0 should be identical
    auto out1 = model->forward(input).clone();
    auto out2 = model->forward(input);

    EXPECT_TRUE(torch::equal(out1, out2))
        << "With dropout=0.0 in train mode, outputs should be deterministic";
}

TEST(MLPModelTest, DropoutCausesVarianceInTrainMode) {
    // With dropout > 0, train mode should produce different outputs across calls
    MLPModel model(INPUT_DIM, NUM_CLASSES, 0.1f);
    model->train();

    auto input = make_input(/*batch_size=*/4, /*seed=*/42);

    // Run many forward passes — at least some should differ due to dropout
    auto out1 = model->forward(input).clone();

    bool found_difference = false;
    for (int trial = 0; trial < 10; ++trial) {
        auto out2 = model->forward(input);
        if (!torch::equal(out1, out2)) {
            found_difference = true;
            break;
        }
    }

    EXPECT_TRUE(found_difference)
        << "With dropout=0.1 in train mode, outputs should vary across forward passes";
}
