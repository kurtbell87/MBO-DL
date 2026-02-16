// gbt_model_test.cpp — TDD RED phase tests for GBT Model Wrapper
// Spec: .kit/docs/gbt-model.md + ORCHESTRATOR_SPEC.md §5.4
//
// Tests the GBTModel class (XGBoost C API wrapper) and the overfit_gbt()
// utility function. Covers training, prediction, save/load, determinism,
// and prediction speed.

#include <gtest/gtest.h>
#include "gbt_model.hpp"
#include "gbt_features.hpp"
#include "book_builder.hpp"
#include "trajectory_builder.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <numeric>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Helpers — synthetic training data
// ---------------------------------------------------------------------------

struct SyntheticData {
    std::vector<std::array<float, GBT_FEATURE_DIM>> features;
    std::vector<int> labels;
};

SyntheticData make_synthetic_data(int n) {
    SyntheticData data;
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

// Generate synthetic TrainingSample objects for overfit_gbt testing
std::vector<TrainingSample> make_training_samples(int n) {
    std::vector<TrainingSample> samples;
    samples.reserve(n);

    for (int i = 0; i < n; ++i) {
        TrainingSample s;
        // Minimal window — overfit_gbt extracts GBT features from this
        s.window.resize(W);
        for (int t = 0; t < W; ++t) {
            std::array<float, FEATURE_DIM> row{};
            // Encode label signal into the window
            row[0] = static_cast<float>(i % 5) * 10.0f;
            row[1] = static_cast<float>(i) * 0.1f;
            s.window[t] = row;
        }
        s.label = i % 5;
        samples.push_back(std::move(s));
    }

    return samples;
}

// Temporary file path for save/load tests
std::string temp_model_path() {
    return (std::filesystem::temp_directory_path() / "gbt_test_model.xgb").string();
}

}  // anonymous namespace

// ===========================================================================
// Test Fixture
// ===========================================================================
class GBTModelTest : public ::testing::Test {
protected:
    void TearDown() override {
        // Clean up any temp model files
        std::filesystem::remove(temp_model_path());
    }
};

// ===========================================================================
// 14. Train and predict — predictions are valid class indices
// ===========================================================================
TEST_F(GBTModelTest, TrainAndPredict_ValidClassIndices) {
    auto data = make_synthetic_data(32);

    GBTModel model;
    model.train(data.features, data.labels, 100);

    auto predictions = model.predict(data.features);

    ASSERT_EQ(predictions.size(), 32u);

    for (size_t i = 0; i < predictions.size(); ++i) {
        EXPECT_GE(predictions[i], 0) << "Prediction " << i << " below valid range";
        EXPECT_LE(predictions[i], 4) << "Prediction " << i << " above valid range";
    }
}

TEST_F(GBTModelTest, TrainAndPredict_AllClassesRepresented) {
    // With 32 samples (approximately 6-7 per class) and 100 rounds of
    // boosting, the model should predict at least some of each class.
    auto data = make_synthetic_data(32);

    GBTModel model;
    model.train(data.features, data.labels, 200);

    auto predictions = model.predict(data.features);

    // Count predictions per class
    std::vector<int> counts(5, 0);
    for (int p : predictions) {
        counts[p]++;
    }

    // With the deterministic pattern, the model should predict all 5 classes
    for (int c = 0; c < 5; ++c) {
        EXPECT_GT(counts[c], 0) << "Class " << c << " never predicted";
    }
}

// ===========================================================================
// 15. Overfit synthetic — train to high accuracy
// ===========================================================================
TEST_F(GBTModelTest, OverfitSynthetic_HighAccuracy) {
    // 32 samples with deterministic labels, 1000 rounds.
    // Should reach >= 95% train accuracy.
    auto data = make_synthetic_data(32);

    GBTModel model;
    model.train(data.features, data.labels, 1000);

    auto predictions = model.predict(data.features);

    int correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == data.labels[i]) correct++;
    }

    float accuracy = static_cast<float>(correct) / static_cast<float>(predictions.size());
    EXPECT_GE(accuracy, 0.95f) << "Overfit accuracy " << accuracy << " below 95% threshold";
}

TEST_F(GBTModelTest, OverfitGBTFunction_ReturnsSuccess) {
    // Test the overfit_gbt convenience function
    auto samples = make_training_samples(32);

    GBTOverfitResult result = overfit_gbt(samples, 1000, 0.99f);

    EXPECT_TRUE(result.success) << "overfit_gbt failed to reach target accuracy";
    EXPECT_GE(result.accuracy, 0.95f)
        << "overfit_gbt accuracy " << result.accuracy << " below 95%";
    EXPECT_GT(result.rounds, 0) << "overfit_gbt should report positive round count";
    EXPECT_LE(result.rounds, 1000) << "overfit_gbt rounds exceed max";
}

// ===========================================================================
// 16. Save and load — identical predictions
// ===========================================================================
TEST_F(GBTModelTest, SaveAndLoad_IdenticalPredictions) {
    auto data = make_synthetic_data(32);
    std::string path = temp_model_path();

    // Train and save
    GBTModel model1;
    model1.train(data.features, data.labels, 200);
    auto pred_before = model1.predict(data.features);
    model1.save(path);

    // Load into a new model and predict
    GBTModel model2;
    model2.load(path);
    auto pred_after = model2.predict(data.features);

    ASSERT_EQ(pred_before.size(), pred_after.size());
    for (size_t i = 0; i < pred_before.size(); ++i) {
        EXPECT_EQ(pred_before[i], pred_after[i])
            << "Prediction mismatch at index " << i << " after save/load";
    }
}

TEST_F(GBTModelTest, SaveToInvalidPath_Throws) {
    auto data = make_synthetic_data(8);

    GBTModel model;
    model.train(data.features, data.labels, 10);

    // Save to a non-existent directory should fail
    EXPECT_THROW(model.save("/nonexistent/dir/model.xgb"), std::runtime_error);
}

TEST_F(GBTModelTest, LoadFromNonexistent_Throws) {
    GBTModel model;
    EXPECT_THROW(model.load("/nonexistent/model.xgb"), std::runtime_error);
}

// ===========================================================================
// 17. Deterministic with seed — reproducible results
// ===========================================================================
TEST_F(GBTModelTest, DeterministicWithSeed) {
    auto data = make_synthetic_data(32);

    // Run 1: train with seed=42
    GBTModel model1;
    model1.train(data.features, data.labels, 200);
    auto pred1 = model1.predict(data.features);

    // Run 2: train with same data, same params, same seed
    GBTModel model2;
    model2.train(data.features, data.labels, 200);
    auto pred2 = model2.predict(data.features);

    ASSERT_EQ(pred1.size(), pred2.size());
    for (size_t i = 0; i < pred1.size(); ++i) {
        EXPECT_EQ(pred1[i], pred2[i])
            << "Prediction mismatch at index " << i << " between identical runs";
    }
}

// ===========================================================================
// 18. Prediction speed — 128 samples in < 10ms
// ===========================================================================
TEST_F(GBTModelTest, PredictionSpeed_128Samples) {
    auto data = make_synthetic_data(128);

    GBTModel model;
    model.train(data.features, data.labels, 100);

    // Time the prediction
    auto start = std::chrono::high_resolution_clock::now();
    auto predictions = model.predict(data.features);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    ASSERT_EQ(predictions.size(), 128u);
    EXPECT_LT(duration_ms, 10) << "Prediction took " << duration_ms << "ms, exceeds 10ms limit";
}

// ===========================================================================
// 19. Constructor/destructor — no leaks (smoke test)
// ===========================================================================
TEST(GBTModelLifecycleTest, ConstructAndDestruct) {
    // Verify GBTModel can be constructed and destroyed without crashing.
    { GBTModel model; }
    SUCCEED();
}

TEST(GBTModelLifecycleTest, PredictBeforeTrainThrows) {
    // Predicting on an untrained model should throw or return empty.
    GBTModel model;
    auto data = make_synthetic_data(8);

    EXPECT_THROW(model.predict(data.features), std::runtime_error);
}

// ===========================================================================
// 20. XGBoost config matches spec
// ===========================================================================
TEST_F(GBTModelTest, ConfigMatchesSpec) {
    // The spec requires these XGBoost params:
    //   objective: multi:softmax
    //   num_class: 5
    //   max_depth: 10
    //   learning_rate: 0.1
    //   subsample: 1.0
    //   colsample_bytree: 1.0
    //   min_child_weight: 1
    //   seed: 42
    //
    // We verify indirectly: train with 5-class data, get predictions in {0..4}
    auto data = make_synthetic_data(50);

    GBTModel model;
    model.train(data.features, data.labels, 500);

    auto preds = model.predict(data.features);

    // multi:softmax → integer class predictions
    for (int p : preds) {
        EXPECT_GE(p, 0);
        EXPECT_LE(p, 4);
    }

    // With 500 rounds, max_depth=10 on 50 samples, should overfit
    int correct = 0;
    for (size_t i = 0; i < preds.size(); ++i) {
        if (preds[i] == data.labels[i]) correct++;
    }
    float accuracy = static_cast<float>(correct) / static_cast<float>(preds.size());
    EXPECT_GE(accuracy, 0.90f) << "XGBoost config may not match spec: accuracy=" << accuracy;
}

// ===========================================================================
// 21. Empty input handling
// ===========================================================================
TEST_F(GBTModelTest, TrainWithEmptyData_Throws) {
    GBTModel model;
    std::vector<std::array<float, GBT_FEATURE_DIM>> empty_features;
    std::vector<int> empty_labels;

    EXPECT_THROW(model.train(empty_features, empty_labels), std::invalid_argument);
}

TEST_F(GBTModelTest, TrainWithMismatchedSizes_Throws) {
    GBTModel model;
    auto data = make_synthetic_data(32);
    data.labels.pop_back();  // Make sizes mismatch

    EXPECT_THROW(model.train(data.features, data.labels), std::invalid_argument);
}

// ===========================================================================
// 22. GBTOverfitResult struct fields
// ===========================================================================
TEST(GBTOverfitResultTest, StructFieldsExist) {
    GBTOverfitResult result{};
    result.accuracy = 0.99f;
    result.rounds = 500;
    result.success = true;

    EXPECT_FLOAT_EQ(result.accuracy, 0.99f);
    EXPECT_EQ(result.rounds, 500);
    EXPECT_TRUE(result.success);
}
