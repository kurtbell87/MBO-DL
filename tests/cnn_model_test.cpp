// cnn_model_test.cpp — TDD RED phase tests for CNNModel
// Spec: .kit/docs/cnn-model.md
//
// Tests the CNNModel class (libtorch CNN) that treats the order book as a
// structured spatial signal per timestep, then convolves temporally.
// Architecture: (B, 600, 194) → price ladder split → spatial conv → temporal conv → (B, 5)
//
// The CNN uses feature index constants from feature_encoder.hpp to split the
// 194-dim feature vector into book_spatial (20, 2), trade_features (150),
// and scalar_features (4) per timestep.

#include <gtest/gtest.h>

#include "cnn_model.hpp"
#include "feature_encoder.hpp"     // FEATURE_DIM, W, index constants
#include "training_loop.hpp"       // samples_to_tensors
#include "test_helpers.hpp"

#include <torch/torch.h>

#include <string>

using test_helpers::NUM_CLASSES;
using test_helpers::make_input;
using test_helpers::make_synthetic_samples;

namespace {

// Create a deterministic input tensor where specific feature indices have
// known values, for verifying the price ladder construction.
// Sets bid prices, bid sizes, ask prices, ask sizes to recognizable values.
torch::Tensor make_known_input(int batch_size) {
    auto x = torch::zeros({batch_size, W, FEATURE_DIM});

    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < W; ++t) {
            // Bid prices: [0:10] — set to level index (0..9)
            for (int i = 0; i < 10; ++i) {
                x[b][t][BID_PRICE_BEGIN + i] = static_cast<float>(i);
            }
            // Bid sizes: [10:20] — set to (level + 10)
            for (int i = 0; i < 10; ++i) {
                x[b][t][BID_SIZE_BEGIN + i] = static_cast<float>(i + 10);
            }
            // Ask prices: [20:30] — set to (level + 100)
            for (int i = 0; i < 10; ++i) {
                x[b][t][ASK_PRICE_BEGIN + i] = static_cast<float>(i + 100);
            }
            // Ask sizes: [30:40] — set to (level + 200)
            for (int i = 0; i < 10; ++i) {
                x[b][t][ASK_SIZE_BEGIN + i] = static_cast<float>(i + 200);
            }
            // Trade features [40:190] — set to 0.5
            for (int i = TRADE_PRICE_BEGIN; i < TRADE_AGGRESSOR_END; ++i) {
                x[b][t][i] = 0.5f;
            }
            // Scalar features [190:194] — set to 1.0
            for (int i = SPREAD_TICKS_IDX; i <= POSITION_STATE_IDX; ++i) {
                x[b][t][i] = 1.0f;
            }
        }
    }
    return x;
}

}  // anonymous namespace

// ===========================================================================
// Test 1: Output shape — Input (B=4, 600, 194), output shape is (4, 5)
// ===========================================================================
TEST(CNNModelTest, OutputShape) {
    CNNModel model(NUM_CLASSES);
    model->eval();

    auto input = make_input(/*batch_size=*/4);
    auto output = model->forward(input);

    ASSERT_EQ(output.dim(), 2) << "Output should be 2D (B, num_classes)";
    EXPECT_EQ(output.size(0), 4) << "Batch dimension should be 4";
    EXPECT_EQ(output.size(1), NUM_CLASSES) << "Class dimension should be 5";
}

TEST(CNNModelTest, OutputShapeSingleSample) {
    CNNModel model(NUM_CLASSES);
    model->eval();

    auto input = make_input(/*batch_size=*/1);
    auto output = model->forward(input);

    ASSERT_EQ(output.dim(), 2);
    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), NUM_CLASSES);
}

TEST(CNNModelTest, OutputShapeLargeBatch) {
    CNNModel model(NUM_CLASSES);
    model->eval();

    auto input = make_input(/*batch_size=*/32);
    auto output = model->forward(input);

    ASSERT_EQ(output.dim(), 2);
    EXPECT_EQ(output.size(0), 32);
    EXPECT_EQ(output.size(1), NUM_CLASSES);
}

// ===========================================================================
// Test 2: Forward pass no crash — random input, forward completes
// ===========================================================================
TEST(CNNModelTest, ForwardPassNoCrash) {
    CNNModel model(NUM_CLASSES);
    model->eval();

    for (int b : {1, 2, 4, 8, 16, 32}) {
        auto input = make_input(b);
        EXPECT_NO_THROW(model->forward(input))
            << "Forward pass should not throw for batch_size=" << b;
    }
}

// ===========================================================================
// Test 3: Price ladder construction — verify spatial tensor layout
//
// Per the spec, the price ladder is constructed as:
//   [bid[9], bid[8], ..., bid[1], bid[0], ask[0], ask[1], ..., ask[8], ask[9]]
// Each level has 2 channels: (price_delta, size_norm).
//
// We can't directly observe internal tensors, so we construct a known input
// and verify the model's build_price_ladder method (or test via the spatial
// path output). We test the contract: given known feature indices, the
// spatial tensor at each level should match the expected reindexing.
//
// If the model exposes build_price_ladder publicly, we test it directly.
// Otherwise, we verify the full forward pass produces correct shapes and
// that the model uses the correct feature indices by checking that swapping
// bid/ask features changes the output.
// ===========================================================================
TEST(CNNModelTest, PriceLadderInputSensitivity) {
    // Verify the model is sensitive to the book features at the correct indices.
    // If the price ladder construction is wrong, perturbing specific feature
    // indices should NOT change the output (because those features aren't wired
    // correctly). We verify that perturbing book features DOES change the output.

    CNNModel model(NUM_CLASSES);
    model->eval();

    torch::manual_seed(42);
    auto baseline = torch::randn({2, W, FEATURE_DIM});
    auto output_baseline = model->forward(baseline).clone();

    // Perturb only bid prices [0:10] with large values
    auto perturbed_bid = baseline.clone();
    perturbed_bid.index({torch::indexing::Slice(), torch::indexing::Slice(),
                         torch::indexing::Slice(BID_PRICE_BEGIN, BID_PRICE_END)}) += 100.0f;
    auto output_bid = model->forward(perturbed_bid);

    EXPECT_FALSE(torch::allclose(output_baseline, output_bid, /*rtol=*/1e-3, /*atol=*/1e-3))
        << "Output should change when bid prices [0:10] are perturbed — "
        << "model must read from BID_PRICE_BEGIN:BID_PRICE_END";

    // Perturb only ask sizes [30:40]
    auto perturbed_ask_size = baseline.clone();
    perturbed_ask_size.index({torch::indexing::Slice(), torch::indexing::Slice(),
                              torch::indexing::Slice(ASK_SIZE_BEGIN, ASK_SIZE_END)}) += 100.0f;
    auto output_ask_size = model->forward(perturbed_ask_size);

    EXPECT_FALSE(torch::allclose(output_baseline, output_ask_size, /*rtol=*/1e-3, /*atol=*/1e-3))
        << "Output should change when ask sizes [30:40] are perturbed — "
        << "model must read from ASK_SIZE_BEGIN:ASK_SIZE_END";
}

TEST(CNNModelTest, PriceLadderBidReversal) {
    // Verify the model treats bids in reversed order.
    // Construct two inputs that differ only in the ordering of bid levels.
    // If the model correctly reverses bids, swapping bid[0] and bid[9] should
    // produce different outputs than swapping within the already-reversed layout.

    CNNModel model(NUM_CLASSES);
    model->eval();

    auto input_a = make_known_input(2);
    auto output_a = model->forward(input_a).clone();

    // Swap bid price level 0 and level 9 (price_delta and size_norm)
    auto input_b = input_a.clone();
    for (int t = 0; t < W; ++t) {
        // Swap bid_price[0] <-> bid_price[9]
        auto tmp_price = input_b.index({torch::indexing::Slice(), t, BID_PRICE_BEGIN + 0}).clone();
        input_b.index_put_({torch::indexing::Slice(), t, BID_PRICE_BEGIN + 0},
                           input_b.index({torch::indexing::Slice(), t, BID_PRICE_BEGIN + 9}));
        input_b.index_put_({torch::indexing::Slice(), t, BID_PRICE_BEGIN + 9}, tmp_price);

        // Swap bid_size[0] <-> bid_size[9]
        auto tmp_size = input_b.index({torch::indexing::Slice(), t, BID_SIZE_BEGIN + 0}).clone();
        input_b.index_put_({torch::indexing::Slice(), t, BID_SIZE_BEGIN + 0},
                           input_b.index({torch::indexing::Slice(), t, BID_SIZE_BEGIN + 9}));
        input_b.index_put_({torch::indexing::Slice(), t, BID_SIZE_BEGIN + 9}, tmp_size);
    }

    auto output_b = model->forward(input_b);

    // After swapping bid[0] and bid[9], the output should differ because
    // the price ladder places them at different spatial positions
    EXPECT_FALSE(torch::equal(output_a, output_b))
        << "Swapping bid levels 0 and 9 should change the output, "
        << "confirming the model uses bid reversal in the price ladder";
}

// ===========================================================================
// Test 4: Spatial path output shape — (B, 600, 64) after spatial conv + pool
//
// We test this indirectly: the concatenation step requires spatial_out to
// have 64 features. If spatial is wrong, the temporal conv input channels
// will be wrong and the forward pass will crash or produce wrong output dims.
// The output shape test (Test 1) already validates the full pipeline.
// Here we test that the model has the expected spatial conv layer structure.
// ===========================================================================
TEST(CNNModelTest, SpatialConvLayerStructure) {
    CNNModel model(NUM_CLASSES);

    // Verify the model contains the expected spatial conv layers by checking
    // that named parameters include spatial conv weights with correct shapes.
    // Spatial conv1: Conv1d(in=2, out=32, kernel=3)
    // Spatial conv2: Conv1d(in=32, out=64, kernel=3)
    bool found_spatial_conv1 = false;
    bool found_spatial_conv2 = false;

    for (const auto& pair : model->named_parameters()) {
        const auto& name = pair.key();
        const auto& param = pair.value();

        // Look for a conv layer with shape (32, 2, 3) — spatial_conv1
        if (param.dim() == 3 && param.size(0) == 32 && param.size(1) == 2 && param.size(2) == 3) {
            found_spatial_conv1 = true;
        }
        // Look for a conv layer with shape (64, 32, 3) — spatial_conv2
        if (param.dim() == 3 && param.size(0) == 64 && param.size(1) == 32 && param.size(2) == 3) {
            found_spatial_conv2 = true;
        }
    }

    EXPECT_TRUE(found_spatial_conv1)
        << "Should have a spatial Conv1d(in=2, out=32, kernel=3) — "
        << "weight shape (32, 2, 3)";
    EXPECT_TRUE(found_spatial_conv2)
        << "Should have a spatial Conv1d(in=32, out=64, kernel=3) — "
        << "weight shape (64, 32, 3)";
}

// ===========================================================================
// Test 5: Concatenation — verify the model produces (B, 600, 218) internally
//
// 218 = 64 (spatial) + 150 (trade) + 4 (scalar)
// We verify indirectly via temporal conv input channels = 218.
// ===========================================================================
TEST(CNNModelTest, TemporalConvInputChannels) {
    CNNModel model(NUM_CLASSES);

    // The temporal conv1 should have in_channels=218 (from concatenation).
    // Look for a Conv1d weight with shape (128, 218, 5).
    bool found_temporal_conv1 = false;

    for (const auto& pair : model->named_parameters()) {
        const auto& param = pair.value();

        // Temporal conv1: Conv1d(in=218, out=128, kernel=5)
        if (param.dim() == 3 && param.size(0) == 128 && param.size(1) == 218 && param.size(2) == 5) {
            found_temporal_conv1 = true;
        }
    }

    EXPECT_TRUE(found_temporal_conv1)
        << "Should have a temporal Conv1d(in=218, out=128, kernel=5) — "
        << "weight shape (128, 218, 5). This confirms concatenation produces "
        << "218 = 64 (spatial) + 150 (trade) + 4 (scalar) channels";
}

// ===========================================================================
// Test 6: Temporal path output — verify temporal conv layers
//
// Temporal conv2: Conv1d(in=128, out=256, kernel=5, padding=2)
// After AdaptiveAvgPool1d(1) → (B, 256)
// Then Linear(256, 5) → logits
// ===========================================================================
TEST(CNNModelTest, TemporalConvLayerStructure) {
    CNNModel model(NUM_CLASSES);

    // Temporal conv2: Conv1d(in=128, out=256, kernel=5)
    bool found_temporal_conv2 = false;
    // Classification head: Linear(256, 5)
    bool found_classifier = false;

    for (const auto& pair : model->named_parameters()) {
        const auto& param = pair.value();

        if (param.dim() == 3 && param.size(0) == 256 && param.size(1) == 128 && param.size(2) == 5) {
            found_temporal_conv2 = true;
        }
        // Linear(256, 5) weight shape: (5, 256)
        if (param.dim() == 2 && param.size(0) == 5 && param.size(1) == 256) {
            found_classifier = true;
        }
    }

    EXPECT_TRUE(found_temporal_conv2)
        << "Should have a temporal Conv1d(in=128, out=256, kernel=5) — "
        << "weight shape (256, 128, 5)";
    EXPECT_TRUE(found_classifier)
        << "Should have a classification Linear(256, 5) — weight shape (5, 256)";
}

// ===========================================================================
// Test 7: Gradient flow — forward + backward, all parameters have non-zero grads
// ===========================================================================
TEST(CNNModelTest, GradientFlow) {
    CNNModel model(NUM_CLASSES);
    model->train();

    auto input = make_input(/*batch_size=*/4, /*seed=*/123);
    auto output = model->forward(input);

    auto target = torch::randint(0, NUM_CLASSES, {4}, torch::kLong);
    auto loss = torch::nn::functional::cross_entropy(output, target);
    loss.backward();

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

// ===========================================================================
// Test 8: No NaN in output — random input produces finite output values
// ===========================================================================
TEST(CNNModelTest, NoNanInOutput) {
    CNNModel model(NUM_CLASSES);
    model->eval();

    for (int seed : {42, 123, 456, 789}) {
        auto input = make_input(/*batch_size=*/4, seed);
        auto output = model->forward(input);

        EXPECT_FALSE(torch::any(torch::isnan(output)).item<bool>())
            << "Output contains NaN with seed=" << seed;

        EXPECT_FALSE(torch::any(torch::isinf(output)).item<bool>())
            << "Output contains Inf with seed=" << seed;
    }
}

// ===========================================================================
// Test 9: Loss decreases — 10 epochs on synthetic data, loss decreases
// ===========================================================================
TEST(CNNModelTest, LossDecreases) {
    torch::manual_seed(42);

    CNNModel model(NUM_CLASSES);
    model->train();

    auto samples = make_synthetic_samples(32);
    auto [input_tensor, label_tensor] = samples_to_tensors(samples);

    torch::optim::Adam optimizer(model->parameters(),
                                  torch::optim::AdamOptions(1e-3));

    // Record initial loss
    float initial_loss;
    {
        auto output = model->forward(input_tensor);
        auto loss = torch::nn::functional::cross_entropy(output, label_tensor);
        initial_loss = loss.item<float>();
    }

    // Train for 10 epochs
    float final_loss = initial_loss;
    for (int epoch = 0; epoch < 10; ++epoch) {
        optimizer.zero_grad();
        auto output = model->forward(input_tensor);
        auto loss = torch::nn::functional::cross_entropy(output, label_tensor);
        loss.backward();
        torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
        optimizer.step();
        final_loss = loss.item<float>();
    }

    EXPECT_LT(final_loss, initial_loss)
        << "Loss should decrease after 10 epochs of training. "
        << "Initial: " << initial_loss << ", Final: " << final_loss;
}

// ===========================================================================
// Test 10: Deterministic — two runs with seed=42, identical loss at epoch 10
// ===========================================================================
TEST(CNNModelTest, DeterministicWithSeed) {
    auto samples = make_synthetic_samples(32);
    auto [input_tensor, label_tensor] = samples_to_tensors(samples);

    auto run_training = [&]() -> float {
        torch::manual_seed(42);
        CNNModel model(NUM_CLASSES);
        model->train();

        torch::optim::Adam optimizer(model->parameters(),
                                      torch::optim::AdamOptions(1e-3));

        float loss_val = 0.0f;
        for (int epoch = 0; epoch < 10; ++epoch) {
            optimizer.zero_grad();
            auto output = model->forward(input_tensor);
            auto loss = torch::nn::functional::cross_entropy(output, label_tensor);
            loss.backward();
            torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
            optimizer.step();
            loss_val = loss.item<float>();
        }
        return loss_val;
    };

    float loss_run1 = run_training();
    float loss_run2 = run_training();

    EXPECT_FLOAT_EQ(loss_run1, loss_run2)
        << "Two runs with seed=42 should produce identical loss at epoch 10. "
        << "Run 1: " << loss_run1 << ", Run 2: " << loss_run2;
}

// ===========================================================================
// Test 11: Overfit synthetic — 32 samples, train, verify >= 95% accuracy
//          within 500 epochs (spec says >= 99% target, >= 95% exit criterion)
// ===========================================================================
TEST(CNNModelTest, OverfitSyntheticN32) {
    torch::manual_seed(42);

    auto samples = make_synthetic_samples(32);
    auto [input_tensor, label_tensor] = samples_to_tensors(samples);
    int n = static_cast<int>(samples.size());

    CNNModel model(NUM_CLASSES);
    model->train();

    torch::optim::Adam optimizer(model->parameters(),
                                  torch::optim::AdamOptions(1e-3));

    constexpr int MAX_EPOCHS = 500;
    constexpr float TARGET_ACCURACY = 0.95f;

    float final_accuracy = 0.0f;
    float final_loss = 0.0f;
    int epochs_to_target = -1;

    for (int epoch = 0; epoch < MAX_EPOCHS; ++epoch) {
        optimizer.zero_grad();
        auto output = model->forward(input_tensor);
        auto loss = torch::nn::functional::cross_entropy(output, label_tensor);
        loss.backward();
        torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
        optimizer.step();

        auto preds = output.argmax(1);
        float accuracy = preds.eq(label_tensor).sum().item<float>() / static_cast<float>(n);
        final_accuracy = accuracy;
        final_loss = loss.item<float>();

        if (accuracy >= TARGET_ACCURACY && epochs_to_target < 0) {
            epochs_to_target = epoch + 1;
            // Don't break — continue to get the final accuracy
        }
    }

    EXPECT_GE(final_accuracy, TARGET_ACCURACY)
        << "CNN should overfit 32 synthetic samples to >= 95% accuracy within "
        << MAX_EPOCHS << " epochs, got " << final_accuracy * 100.0f << "%";

    EXPECT_GT(epochs_to_target, 0)
        << "Should reach target accuracy within " << MAX_EPOCHS << " epochs";

    EXPECT_LT(final_loss, 0.5f)
        << "Loss should be low after overfitting, got " << final_loss;
}

// ===========================================================================
// Test 12: Checkpoint save/load — save, load, verify identical predictions
// ===========================================================================
TEST(CNNModelTest, CheckpointSaveLoad) {
    CNNModel model(NUM_CLASSES);
    model->eval();

    auto input = make_input(/*batch_size=*/4, /*seed=*/99);

    // Get predictions from original model
    auto output_before = model->forward(input).clone();

    // Save checkpoint
    std::string checkpoint_path = "/tmp/cnn_model_test_checkpoint.pt";
    torch::save(model, checkpoint_path);

    // Create a new model and load the checkpoint
    CNNModel loaded_model(NUM_CLASSES);
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
// Additional architectural validation tests
// ===========================================================================

// Verify the model uses the correct feature index constants
TEST(CNNModelTest, FeatureIndexConstantsConsistency) {
    // Verify the index constants from feature_encoder.hpp match expectations
    // for the CNN's feature split: book [0:40], trades [40:190], scalars [190:194]

    // Book spatial: 4 * L = 40 features → (20 levels × 2 channels)
    EXPECT_EQ(BID_PRICE_BEGIN, 0);
    EXPECT_EQ(BID_PRICE_END, 10);
    EXPECT_EQ(BID_SIZE_BEGIN, 10);
    EXPECT_EQ(BID_SIZE_END, 20);
    EXPECT_EQ(ASK_PRICE_BEGIN, 20);
    EXPECT_EQ(ASK_PRICE_END, 30);
    EXPECT_EQ(ASK_SIZE_BEGIN, 30);
    EXPECT_EQ(ASK_SIZE_END, 40);

    // Trade features: 150 values
    constexpr int TRADE_FEATURES_DIM = TRADE_AGGRESSOR_END - TRADE_PRICE_BEGIN;
    EXPECT_EQ(TRADE_FEATURES_DIM, 150)
        << "Trade features should span 150 dimensions [40:190]";

    // Scalar features: 4 values [190:194]
    constexpr int SCALAR_FEATURES_DIM = FEATURE_DIM - SPREAD_TICKS_IDX;
    EXPECT_EQ(SCALAR_FEATURES_DIM, 4)
        << "Scalar features should span 4 dimensions [190:194]";

    // Total: 40 + 150 + 4 = 194
    EXPECT_EQ(40 + TRADE_FEATURES_DIM + SCALAR_FEATURES_DIM, FEATURE_DIM)
        << "Book (40) + Trade (150) + Scalar (4) should equal FEATURE_DIM (194)";
}

// Verify trade features sensitivity (separate from spatial path)
TEST(CNNModelTest, TradeFeatureSensitivity) {
    CNNModel model(NUM_CLASSES);
    model->eval();

    torch::manual_seed(42);
    auto baseline = torch::randn({2, W, FEATURE_DIM});
    auto output_baseline = model->forward(baseline).clone();

    // Perturb only trade features [40:190]
    auto perturbed = baseline.clone();
    perturbed.index({torch::indexing::Slice(), torch::indexing::Slice(),
                     torch::indexing::Slice(TRADE_PRICE_BEGIN, TRADE_AGGRESSOR_END)}) += 100.0f;
    auto output_perturbed = model->forward(perturbed);

    EXPECT_FALSE(torch::allclose(output_baseline, output_perturbed, /*rtol=*/1e-3, /*atol=*/1e-3))
        << "Output should change when trade features [40:190] are perturbed — "
        << "model must incorporate trade features in the concatenation path";
}

// Verify scalar features sensitivity (separate from spatial and trade)
TEST(CNNModelTest, ScalarFeatureSensitivity) {
    CNNModel model(NUM_CLASSES);
    model->eval();

    torch::manual_seed(42);
    auto baseline = torch::randn({2, W, FEATURE_DIM});
    auto output_baseline = model->forward(baseline).clone();

    // Perturb only scalar features [190:194]
    auto perturbed = baseline.clone();
    perturbed.index({torch::indexing::Slice(), torch::indexing::Slice(),
                     torch::indexing::Slice(SPREAD_TICKS_IDX, FEATURE_DIM)}) += 100.0f;
    auto output_perturbed = model->forward(perturbed);

    EXPECT_FALSE(torch::allclose(output_baseline, output_perturbed, /*rtol=*/1e-3, /*atol=*/1e-3))
        << "Output should change when scalar features [190:194] are perturbed — "
        << "model must incorporate scalar features in the concatenation path";
}

// Verify the default constructor uses num_classes=5
TEST(CNNModelTest, DefaultConstructorNumClasses) {
    CNNModel model;
    model->eval();

    auto input = make_input(/*batch_size=*/2);
    auto output = model->forward(input);

    EXPECT_EQ(output.size(1), 5)
        << "Default constructor should produce 5-class output";
}
