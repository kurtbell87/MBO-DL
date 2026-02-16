// serialization_test.cpp — tests for Model Serialization (Phase 5)
// Spec: .kit/docs/serialization.md
//
// Verifies that all trained models can be saved and reloaded with identical
// outputs. This is the final validation gate before the spec is complete.
//
// Tests cover:
//   - MLP checkpoint save/load (bitwise identical via torch::equal)
//   - CNN checkpoint save/load (bitwise identical via torch::equal)
//   - MLP ONNX export (file exists + validates, or full inference if runtime available)
//   - CNN ONNX export (file exists + validates, or full inference if runtime available)
//   - GBT save/load (identical integer class predictions)
//   - Edge cases: temp file cleanup, save untrained model, etc.

#include <gtest/gtest.h>

#include "mlp_model.hpp"
#include "cnn_model.hpp"
#include "gbt_model.hpp"
#include "gbt_features.hpp"
#include "serialization.hpp"
#include "training_loop.hpp"
#include "test_helpers.hpp"

#include <torch/torch.h>

#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using test_helpers::NUM_CLASSES;
using test_helpers::make_input;
using test_helpers::make_synthetic_samples;

namespace {

constexpr int N = 32;        // Spec: N=32 synthetic samples
constexpr int SEED = 42;     // Spec: deterministic seed
constexpr int INPUT_DIM = W * FEATURE_DIM;

// ---------------------------------------------------------------------------
// Helpers — synthetic GBT data (mirrors gbt_model_test.cpp pattern)
// ---------------------------------------------------------------------------
struct SyntheticGBTData {
    std::vector<std::array<float, GBT_FEATURE_DIM>> features;
    std::vector<int> labels;
};

SyntheticGBTData make_gbt_data(int n) {
    SyntheticGBTData data;
    data.features.reserve(n);
    data.labels.reserve(n);

    for (int i = 0; i < n; ++i) {
        std::array<float, GBT_FEATURE_DIM> f{};
        int label = i % 5;
        f[0] = static_cast<float>(label) * 10.0f;
        f[1] = static_cast<float>(i) * 0.1f;
        for (int j = 2; j < GBT_FEATURE_DIM; ++j) {
            f[j] = static_cast<float>((i * 7 + j * 13) % 100) / 100.0f;
        }
        data.features.push_back(f);
        data.labels.push_back(label);
    }
    return data;
}

// Temp file path helper — unique per test via suffix
std::string temp_path(const std::string& suffix) {
    return (std::filesystem::temp_directory_path() / ("serialization_test_" + suffix)).string();
}

}  // anonymous namespace

// ===========================================================================
// Test Fixture — cleans up temp files after each test
// ===========================================================================
class SerializationTest : public ::testing::Test {
protected:
    void TearDown() override {
        for (const auto& p : temp_files_) {
            std::filesystem::remove(p);
        }
    }

    // Register a temp file for cleanup
    std::string temp(const std::string& suffix) {
        auto p = temp_path(suffix);
        temp_files_.push_back(p);
        return p;
    }

private:
    std::vector<std::string> temp_files_;
};

// ===========================================================================
// 1. MLP Checkpoint Round-Trip
//    Spec: "save model state using torch::save(), reload into fresh instance,
//           assert outputs are bitwise identical via torch::equal()"
// ===========================================================================
TEST_F(SerializationTest, MLPCheckpointRoundTrip) {
    torch::manual_seed(SEED);

    // Create and train MLP to overfit on N=32 synthetic samples
    auto samples = make_synthetic_samples(N, SEED);
    auto [input_tensor, label_tensor] = samples_to_tensors(samples);

    MLPModel original(INPUT_DIM, NUM_CLASSES, 0.0f);
    original->train();
    train_model(original, input_tensor, label_tensor,
                /*max_epochs=*/100, /*target_accuracy=*/0.95f,
                /*lr=*/1e-3f, /*clip_gradients=*/false);

    // Switch to eval mode for deterministic forward pass
    original->eval();
    auto output_original = original->forward(input_tensor).clone();

    // Save checkpoint
    auto checkpoint_path = temp("mlp_checkpoint.pt");
    torch::save(original, checkpoint_path);

    // Verify file was created and is non-empty
    ASSERT_TRUE(std::filesystem::exists(checkpoint_path))
        << "Checkpoint file should exist after save";
    ASSERT_GT(std::filesystem::file_size(checkpoint_path), 0u)
        << "Checkpoint file should be non-empty";

    // Load into a fresh model instance
    MLPModel reloaded(INPUT_DIM, NUM_CLASSES, 0.0f);
    torch::load(reloaded, checkpoint_path);
    reloaded->eval();

    // Forward pass on same input
    auto output_reloaded = reloaded->forward(input_tensor);

    // Assert bitwise identical (spec: exact match via torch::equal)
    EXPECT_TRUE(torch::equal(output_original, output_reloaded))
        << "MLP checkpoint round-trip must produce bitwise identical outputs.\n"
        << "Max absolute diff: "
        << (output_original - output_reloaded).abs().max().item<float>();
}

// ===========================================================================
// 2. CNN Checkpoint Round-Trip
//    Spec: same contract as MLP — bitwise identical via torch::equal()
// ===========================================================================
TEST_F(SerializationTest, CNNCheckpointRoundTrip) {
    torch::manual_seed(SEED);

    auto samples = make_synthetic_samples(N, SEED);
    auto [input_tensor, label_tensor] = samples_to_tensors(samples);

    CNNModel original(NUM_CLASSES);
    original->train();
    train_model(original, input_tensor, label_tensor,
                /*max_epochs=*/100, /*target_accuracy=*/0.95f,
                /*lr=*/1e-3f, /*clip_gradients=*/true);

    original->eval();
    auto output_original = original->forward(input_tensor).clone();

    auto checkpoint_path = temp("cnn_checkpoint.pt");
    torch::save(original, checkpoint_path);

    ASSERT_TRUE(std::filesystem::exists(checkpoint_path))
        << "Checkpoint file should exist after save";
    ASSERT_GT(std::filesystem::file_size(checkpoint_path), 0u)
        << "Checkpoint file should be non-empty";

    CNNModel reloaded(NUM_CLASSES);
    torch::load(reloaded, checkpoint_path);
    reloaded->eval();

    auto output_reloaded = reloaded->forward(input_tensor);

    EXPECT_TRUE(torch::equal(output_original, output_reloaded))
        << "CNN checkpoint round-trip must produce bitwise identical outputs.\n"
        << "Max absolute diff: "
        << (output_original - output_reloaded).abs().max().item<float>();
}

// ===========================================================================
// 3. MLP ONNX Export
//    Spec: "Export to ONNX format. If onnxruntime not available, export the
//           ONNX file and verify it loads (schema validation)."
//    We test: (a) export produces a non-empty .onnx file
//             (b) if onnxruntime is available, verify inference within tolerance
// ===========================================================================
TEST_F(SerializationTest, MLPOnnxExport) {
    torch::manual_seed(SEED);

    auto samples = make_synthetic_samples(N, SEED);
    auto [input_tensor, label_tensor] = samples_to_tensors(samples);

    MLPModel model(INPUT_DIM, NUM_CLASSES, 0.0f);
    model->train();
    train_model(model, input_tensor, label_tensor,
                /*max_epochs=*/50, /*target_accuracy=*/0.95f,
                /*lr=*/1e-3f, /*clip_gradients=*/false);
    model->eval();

    auto onnx_path = temp("mlp_model.onnx");

    // Export to ONNX — implementation should provide this capability
    // The export function should write a valid ONNX file
    export_to_onnx(model, input_tensor, onnx_path);

    ASSERT_TRUE(std::filesystem::exists(onnx_path))
        << "ONNX file should exist after export";
    ASSERT_GT(std::filesystem::file_size(onnx_path), 0u)
        << "ONNX file should be non-empty";

    // Get libtorch reference output for comparison
    auto libtorch_output = model->forward(input_tensor);

    // Verify ONNX inference matches libtorch within tolerance
    // Implementation must provide load_and_run_onnx() or equivalent
    auto onnx_output = load_and_run_onnx(onnx_path, input_tensor);

    ASSERT_EQ(onnx_output.sizes(), libtorch_output.sizes())
        << "ONNX output shape must match libtorch output shape";

    EXPECT_TRUE(torch::allclose(onnx_output, libtorch_output,
                                /*rtol=*/1e-4, /*atol=*/1e-4))
        << "ONNX output must match libtorch within atol=1e-4, rtol=1e-4.\n"
        << "Max absolute diff: "
        << (onnx_output - libtorch_output).abs().max().item<float>();
}

// ===========================================================================
// 4. CNN ONNX Export
//    Same contract as MLP ONNX export
// ===========================================================================
TEST_F(SerializationTest, CNNOnnxExport) {
    torch::manual_seed(SEED);

    auto samples = make_synthetic_samples(N, SEED);
    auto [input_tensor, label_tensor] = samples_to_tensors(samples);

    CNNModel model(NUM_CLASSES);
    model->train();
    train_model(model, input_tensor, label_tensor,
                /*max_epochs=*/50, /*target_accuracy=*/0.95f,
                /*lr=*/1e-3f, /*clip_gradients=*/true);
    model->eval();

    auto onnx_path = temp("cnn_model.onnx");

    export_to_onnx(model, input_tensor, onnx_path);

    ASSERT_TRUE(std::filesystem::exists(onnx_path))
        << "ONNX file should exist after export";
    ASSERT_GT(std::filesystem::file_size(onnx_path), 0u)
        << "ONNX file should be non-empty";

    auto libtorch_output = model->forward(input_tensor);
    auto onnx_output = load_and_run_onnx(onnx_path, input_tensor);

    ASSERT_EQ(onnx_output.sizes(), libtorch_output.sizes())
        << "ONNX output shape must match libtorch output shape";

    EXPECT_TRUE(torch::allclose(onnx_output, libtorch_output,
                                /*rtol=*/1e-4, /*atol=*/1e-4))
        << "ONNX output must match libtorch within atol=1e-4, rtol=1e-4.\n"
        << "Max absolute diff: "
        << (onnx_output - libtorch_output).abs().max().item<float>();
}

// ===========================================================================
// 5. GBT Save/Load Round-Trip
//    Spec: "save using XGBoosterSaveModel(), reload into fresh booster,
//           predictions are identical (exact match, integer class labels)"
// ===========================================================================
TEST_F(SerializationTest, GBTSaveLoadRoundTrip) {
    auto data = make_gbt_data(N);
    auto model_path = temp("gbt_model.xgb");

    // Train GBT to overfit
    GBTModel original;
    original.train(data.features, data.labels, /*num_rounds=*/500);

    // Get predictions from original
    auto pred_original = original.predict(data.features);

    // Save
    original.save(model_path);

    ASSERT_TRUE(std::filesystem::exists(model_path))
        << "GBT model file should exist after save";
    ASSERT_GT(std::filesystem::file_size(model_path), 0u)
        << "GBT model file should be non-empty";

    // Load into a fresh instance
    GBTModel reloaded;
    reloaded.load(model_path);

    // Predict with reloaded model
    auto pred_reloaded = reloaded.predict(data.features);

    // Assert identical predictions (exact match — integer class labels)
    ASSERT_EQ(pred_original.size(), pred_reloaded.size())
        << "Prediction vector sizes must match";

    for (size_t i = 0; i < pred_original.size(); ++i) {
        EXPECT_EQ(pred_original[i], pred_reloaded[i])
            << "GBT prediction mismatch at index " << i
            << ": original=" << pred_original[i]
            << " reloaded=" << pred_reloaded[i];
    }
}

// ===========================================================================
// Edge Cases and Error Handling
// ===========================================================================

// ---------------------------------------------------------------------------
// MLP: Verify trained model produces non-trivial weights before checkpoint
// (guards against testing serialization of a randomly-initialized model)
// ---------------------------------------------------------------------------
TEST_F(SerializationTest, MLPCheckpointAfterTraining_NonTrivialWeights) {
    torch::manual_seed(SEED);

    auto samples = make_synthetic_samples(N, SEED);
    auto [input_tensor, label_tensor] = samples_to_tensors(samples);

    // Untrained model
    MLPModel untrained(INPUT_DIM, NUM_CLASSES, 0.0f);
    untrained->eval();
    auto output_untrained = untrained->forward(input_tensor).clone();

    // Trained model
    torch::manual_seed(SEED);
    MLPModel trained(INPUT_DIM, NUM_CLASSES, 0.0f);
    trained->train();
    auto result = train_model(trained, input_tensor, label_tensor,
                              /*max_epochs=*/100, /*target_accuracy=*/0.95f,
                              /*lr=*/1e-3f, /*clip_gradients=*/false);
    trained->eval();
    auto output_trained = trained->forward(input_tensor);

    // Trained model must produce different outputs than untrained
    EXPECT_FALSE(torch::equal(output_untrained, output_trained))
        << "Trained model outputs should differ from untrained model";
}

// ---------------------------------------------------------------------------
// CNN: Same non-trivial weights guard
// ---------------------------------------------------------------------------
TEST_F(SerializationTest, CNNCheckpointAfterTraining_NonTrivialWeights) {
    torch::manual_seed(SEED);

    auto samples = make_synthetic_samples(N, SEED);
    auto [input_tensor, label_tensor] = samples_to_tensors(samples);

    CNNModel untrained(NUM_CLASSES);
    untrained->eval();
    auto output_untrained = untrained->forward(input_tensor).clone();

    torch::manual_seed(SEED);
    CNNModel trained(NUM_CLASSES);
    trained->train();
    train_model(trained, input_tensor, label_tensor,
                /*max_epochs=*/100, /*target_accuracy=*/0.95f,
                /*lr=*/1e-3f, /*clip_gradients=*/true);
    trained->eval();
    auto output_trained = trained->forward(input_tensor);

    EXPECT_FALSE(torch::equal(output_untrained, output_trained))
        << "Trained CNN outputs should differ from untrained model";
}

// ---------------------------------------------------------------------------
// GBT: Predictions after overfit should achieve high accuracy
// (guards against testing serialization of a poorly-trained model)
// ---------------------------------------------------------------------------
TEST_F(SerializationTest, GBTOverfitBeforeSave_HighAccuracy) {
    auto data = make_gbt_data(N);

    GBTModel model;
    model.train(data.features, data.labels, /*num_rounds=*/500);
    auto predictions = model.predict(data.features);

    int correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == data.labels[i]) correct++;
    }
    float accuracy = static_cast<float>(correct) / static_cast<float>(N);

    EXPECT_GE(accuracy, 0.95f)
        << "GBT should overfit N=32 to >= 95% before testing serialization, got "
        << accuracy * 100.0f << "%";
}

// ---------------------------------------------------------------------------
// MLP: Multiple save/load cycles produce consistent results
// ---------------------------------------------------------------------------
TEST_F(SerializationTest, MLPMultipleSaveLoadCycles) {
    torch::manual_seed(SEED);

    auto samples = make_synthetic_samples(N, SEED);
    auto [input_tensor, label_tensor] = samples_to_tensors(samples);

    MLPModel model(INPUT_DIM, NUM_CLASSES, 0.0f);
    model->train();
    train_model(model, input_tensor, label_tensor,
                /*max_epochs=*/50, /*target_accuracy=*/0.95f,
                /*lr=*/1e-3f, /*clip_gradients=*/false);
    model->eval();

    auto path1 = temp("mlp_cycle1.pt");
    auto path2 = temp("mlp_cycle2.pt");

    // Save → load → save → load should produce identical results
    torch::save(model, path1);

    MLPModel loaded1(INPUT_DIM, NUM_CLASSES, 0.0f);
    torch::load(loaded1, path1);
    loaded1->eval();

    torch::save(loaded1, path2);

    MLPModel loaded2(INPUT_DIM, NUM_CLASSES, 0.0f);
    torch::load(loaded2, path2);
    loaded2->eval();

    auto output1 = loaded1->forward(input_tensor);
    auto output2 = loaded2->forward(input_tensor);

    EXPECT_TRUE(torch::equal(output1, output2))
        << "Multiple save/load cycles should preserve bitwise identical outputs";
}

// ---------------------------------------------------------------------------
// GBT: Multiple save/load cycles produce consistent results
// ---------------------------------------------------------------------------
TEST_F(SerializationTest, GBTMultipleSaveLoadCycles) {
    auto data = make_gbt_data(N);

    GBTModel model;
    model.train(data.features, data.labels, /*num_rounds=*/200);

    auto path1 = temp("gbt_cycle1.xgb");
    auto path2 = temp("gbt_cycle2.xgb");

    model.save(path1);

    GBTModel loaded1;
    loaded1.load(path1);
    loaded1.save(path2);

    GBTModel loaded2;
    loaded2.load(path2);

    auto pred1 = loaded1.predict(data.features);
    auto pred2 = loaded2.predict(data.features);

    ASSERT_EQ(pred1.size(), pred2.size());
    for (size_t i = 0; i < pred1.size(); ++i) {
        EXPECT_EQ(pred1[i], pred2[i])
            << "GBT predictions diverged after multiple save/load cycles at index " << i;
    }
}

// ---------------------------------------------------------------------------
// MLP: Checkpoint file cleaned up properly (verify temp_directory_path usage)
// ---------------------------------------------------------------------------
TEST_F(SerializationTest, MLPCheckpointUseTempDirectory) {
    torch::manual_seed(SEED);

    MLPModel model(INPUT_DIM, NUM_CLASSES, 0.0f);
    model->eval();

    auto checkpoint_path = temp("mlp_temp_dir_test.pt");

    // Verify path is under temp directory
    auto temp_dir = std::filesystem::temp_directory_path();
    EXPECT_TRUE(checkpoint_path.find(temp_dir.string()) != std::string::npos)
        << "Checkpoint should use temp directory path";

    torch::save(model, checkpoint_path);
    ASSERT_TRUE(std::filesystem::exists(checkpoint_path));

    // After TearDown, file should be removed (tested implicitly)
}

// ---------------------------------------------------------------------------
// GBT: Save untrained model should throw
// ---------------------------------------------------------------------------
TEST_F(SerializationTest, GBTSaveUntrainedThrows) {
    GBTModel model;
    auto path = temp("gbt_untrained.xgb");

    EXPECT_THROW(model.save(path), std::runtime_error)
        << "Saving an untrained GBT model should throw";
}

// ---------------------------------------------------------------------------
// MLP: Output shape preserved after round-trip
// ---------------------------------------------------------------------------
TEST_F(SerializationTest, MLPOutputShapePreservedAfterRoundTrip) {
    torch::manual_seed(SEED);

    MLPModel model(INPUT_DIM, NUM_CLASSES, 0.0f);
    model->eval();

    auto input = make_input(/*batch_size=*/N, SEED);
    auto output_before = model->forward(input);

    auto path = temp("mlp_shape_test.pt");
    torch::save(model, path);

    MLPModel loaded(INPUT_DIM, NUM_CLASSES, 0.0f);
    torch::load(loaded, path);
    loaded->eval();

    auto output_after = loaded->forward(input);

    EXPECT_EQ(output_before.sizes(), output_after.sizes())
        << "Output tensor shape must be preserved after checkpoint round-trip";
    EXPECT_EQ(output_after.size(0), N);
    EXPECT_EQ(output_after.size(1), NUM_CLASSES);
}

// ---------------------------------------------------------------------------
// CNN: Output shape preserved after round-trip
// ---------------------------------------------------------------------------
TEST_F(SerializationTest, CNNOutputShapePreservedAfterRoundTrip) {
    torch::manual_seed(SEED);

    CNNModel model(NUM_CLASSES);
    model->eval();

    auto input = make_input(/*batch_size=*/N, SEED);
    auto output_before = model->forward(input);

    auto path = temp("cnn_shape_test.pt");
    torch::save(model, path);

    CNNModel loaded(NUM_CLASSES);
    torch::load(loaded, path);
    loaded->eval();

    auto output_after = loaded->forward(input);

    EXPECT_EQ(output_before.sizes(), output_after.sizes())
        << "Output tensor shape must be preserved after checkpoint round-trip";
    EXPECT_EQ(output_after.size(0), N);
    EXPECT_EQ(output_after.size(1), NUM_CLASSES);
}

// ---------------------------------------------------------------------------
// GBT: Prediction count preserved after round-trip
// ---------------------------------------------------------------------------
TEST_F(SerializationTest, GBTPredictionCountPreservedAfterRoundTrip) {
    auto data = make_gbt_data(N);

    GBTModel model;
    model.train(data.features, data.labels, /*num_rounds=*/100);

    auto pred_before = model.predict(data.features);

    auto path = temp("gbt_count_test.xgb");
    model.save(path);

    GBTModel loaded;
    loaded.load(path);
    auto pred_after = loaded.predict(data.features);

    EXPECT_EQ(pred_before.size(), static_cast<size_t>(N))
        << "Original predictions should have N=" << N << " entries";
    EXPECT_EQ(pred_after.size(), static_cast<size_t>(N))
        << "Reloaded predictions should have N=" << N << " entries";
}
