#pragma once

#include <torch/torch.h>

#include "feature_encoder.hpp"  // W, FEATURE_DIM, index constants

// ---------------------------------------------------------------------------
// CNNModelImpl — CNN that treats order book as structured spatial signal
//
// Architecture:
//   Input: (B, 600, 194)
//
//   Price ladder: bid[9..0] + ask[0..9] = 20 levels × 2 channels
//   Spatial: Conv1d(2,32,3) → BN → ReLU → Conv1d(32,64,3) → BN → ReLU → pool → 64
//   Concat: 64 (spatial) + 150 (trade) + 4 (scalar) = 218
//   Temporal: Conv1d(218,128,5) → BN → ReLU → Conv1d(128,256,5) → BN → ReLU
//             → Conv1d(256,256,3) → BN → ReLU
//   Pool: concat(avg_pool(4), max_pool(4)) → flatten → 2048 → BN → Linear(2048,256) → ReLU
//   Classifier: Linear(256, 5)
// ---------------------------------------------------------------------------
struct CNNModelImpl : torch::nn::Module {
    CNNModelImpl(int num_classes = 5)
        : spatial_conv1(register_module("spatial_conv1",
              torch::nn::Conv1d(torch::nn::Conv1dOptions(2, 32, 3).padding(1)))),
          spatial_bn1(register_module("spatial_bn1",
              torch::nn::BatchNorm1d(32))),
          spatial_conv2(register_module("spatial_conv2",
              torch::nn::Conv1d(torch::nn::Conv1dOptions(32, 64, 3).padding(1)))),
          spatial_bn2(register_module("spatial_bn2",
              torch::nn::BatchNorm1d(64))),
          spatial_pool(register_module("spatial_pool",
              torch::nn::AdaptiveAvgPool1d(1))),
          temporal_conv1(register_module("temporal_conv1",
              torch::nn::Conv1d(torch::nn::Conv1dOptions(218, 128, 5).padding(2)))),
          temporal_bn1(register_module("temporal_bn1",
              torch::nn::BatchNorm1d(128))),
          temporal_conv2(register_module("temporal_conv2",
              torch::nn::Conv1d(torch::nn::Conv1dOptions(128, 256, 5).padding(2)))),
          temporal_bn2(register_module("temporal_bn2",
              torch::nn::BatchNorm1d(256))),
          temporal_conv3(register_module("temporal_conv3",
              torch::nn::Conv1d(torch::nn::Conv1dOptions(256, 256, 3).padding(1)))),
          temporal_bn3(register_module("temporal_bn3",
              torch::nn::BatchNorm1d(256))),
          temporal_pool(register_module("temporal_pool",
              torch::nn::AdaptiveAvgPool1d(4))),
          temporal_maxpool(register_module("temporal_maxpool",
              torch::nn::AdaptiveMaxPool1d(4))),
          pool_bn(register_module("pool_bn",
              torch::nn::BatchNorm1d(2048))),
          pool_proj(register_module("pool_proj",
              torch::nn::Linear(2048, 256))),
          classifier(register_module("classifier",
              torch::nn::Linear(256, num_classes)))
    {}

    torch::Tensor forward(torch::Tensor x) {
        // x: (B, W, FEATURE_DIM) = (B, 600, 194)
        int64_t B = x.size(0);
        int64_t T_len = x.size(1);  // W = 600

        // --- Price ladder construction ---
        auto bid_prices = x.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                    torch::indexing::Slice(BID_PRICE_BEGIN, BID_PRICE_END)});
        auto bid_sizes = x.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                   torch::indexing::Slice(BID_SIZE_BEGIN, BID_SIZE_END)});
        auto ask_prices = x.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                    torch::indexing::Slice(ASK_PRICE_BEGIN, ASK_PRICE_END)});
        auto ask_sizes = x.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                   torch::indexing::Slice(ASK_SIZE_BEGIN, ASK_SIZE_END)});

        // Reverse bids: [9, 8, ..., 1, 0]
        auto bid_prices_rev = bid_prices.flip(2);
        auto bid_sizes_rev = bid_sizes.flip(2);

        // Price ladder: 20 levels × 2 channels
        auto ladder_prices = torch::cat({bid_prices_rev, ask_prices}, 2);
        auto ladder_sizes = torch::cat({bid_sizes_rev, ask_sizes}, 2);
        auto ladder = torch::stack({ladder_prices, ladder_sizes}, 3);  // (B, T, 20, 2)

        // Reshape for spatial Conv1d: (B*T, 2, 20)
        auto spatial_in = ladder.reshape({B * T_len, 20, 2}).permute({0, 2, 1});

        // Spatial convolution with BatchNorm
        auto spatial_out = torch::relu(spatial_bn1(spatial_conv1(spatial_in)));   // (B*T, 32, 20)
        spatial_out = torch::relu(spatial_bn2(spatial_conv2(spatial_out)));       // (B*T, 64, 20)
        spatial_out = spatial_pool(spatial_out);                                  // (B*T, 64, 1)
        spatial_out = spatial_out.squeeze(2);                                     // (B*T, 64)
        spatial_out = spatial_out.reshape({B, T_len, 64});                       // (B, T, 64)

        // --- Trade features ---
        auto trade_features = x.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                        torch::indexing::Slice(TRADE_PRICE_BEGIN, TRADE_AGGRESSOR_END)});

        // --- Scalar features ---
        auto scalar_features = x.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                         torch::indexing::Slice(SPREAD_TICKS_IDX, FEATURE_DIM)});

        // --- Concatenation: (B, T, 218) ---
        auto combined = torch::cat({spatial_out, trade_features, scalar_features}, 2);

        // --- Temporal path ---
        auto temporal_in = combined.permute({0, 2, 1});                              // (B, 218, T)
        auto temporal_out = torch::relu(temporal_bn1(temporal_conv1(temporal_in)));   // (B, 128, T)
        temporal_out = torch::relu(temporal_bn2(temporal_conv2(temporal_out)));       // (B, 256, T)
        temporal_out = torch::relu(temporal_bn3(temporal_conv3(temporal_out)));       // (B, 256, T)

        // --- Dual multi-bin pooling: avg(4) + max(4) → flatten → 2048 → BN → 256 ---
        auto avg_pooled = temporal_pool(temporal_out).flatten(1);                    // (B, 256*4=1024)
        auto max_pooled = temporal_maxpool(temporal_out).flatten(1);                 // (B, 256*4=1024)
        auto pooled = torch::cat({avg_pooled, max_pooled}, 1);                      // (B, 2048)
        auto projected = torch::relu(pool_proj(pool_bn(pooled)));                    // (B, 256)

        // --- Classification head ---
        return classifier(projected);
    }

    torch::nn::Conv1d spatial_conv1{nullptr}, spatial_conv2{nullptr};
    torch::nn::BatchNorm1d spatial_bn1{nullptr}, spatial_bn2{nullptr};
    torch::nn::AdaptiveAvgPool1d spatial_pool{nullptr};
    torch::nn::Conv1d temporal_conv1{nullptr}, temporal_conv2{nullptr}, temporal_conv3{nullptr};
    torch::nn::BatchNorm1d temporal_bn1{nullptr}, temporal_bn2{nullptr}, temporal_bn3{nullptr};
    torch::nn::AdaptiveAvgPool1d temporal_pool{nullptr};
    torch::nn::AdaptiveMaxPool1d temporal_maxpool{nullptr};
    torch::nn::BatchNorm1d pool_bn{nullptr};
    torch::nn::Linear pool_proj{nullptr};
    torch::nn::Linear classifier{nullptr};
};

TORCH_MODULE(CNNModel);
