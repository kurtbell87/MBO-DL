#pragma once

#include "book_builder.hpp"
#include "feature_encoder.hpp"
#include "oracle_labeler.hpp"

#include <array>
#include <limits>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// TrainingSample — (window, label) pair for supervised learning
// ---------------------------------------------------------------------------
struct TrainingSample {
    std::vector<std::array<float, FEATURE_DIM>> window;
    int label = 0;
};

// ---------------------------------------------------------------------------
// build_trajectory — produce (window, label) pairs from a sequence of snapshots
// ---------------------------------------------------------------------------
inline std::vector<TrainingSample> build_trajectory(
    const std::vector<BookSnapshot>& snapshots,
    int horizon = 100)
{
    int n = static_cast<int>(snapshots.size());
    if (n < W + horizon) {
        throw std::invalid_argument("build_trajectory requires at least W + horizon = " +
                                    std::to_string(W + horizon) + " snapshots, got " +
                                    std::to_string(n));
    }

    int num_samples = n - horizon - W + 1;
    std::vector<TrainingSample> samples;
    samples.reserve(num_samples);

    // State machine
    int position_state = 0;
    float entry_price = std::numeric_limits<float>::quiet_NaN();

    for (int s = 0; s < num_samples; ++s) {
        // t is the index of the last snapshot in the window (current time)
        int t = W - 1 + s;

        // Extract window: snapshots[t - W + 1 .. t] inclusive
        std::vector<BookSnapshot> win(snapshots.begin() + (t - W + 1),
                                      snapshots.begin() + t + 1);

        // Encode window with current position state
        auto encoded = encode_window(win, static_cast<float>(position_state));

        // Get oracle label
        int label = oracle_label(snapshots, t, position_state, entry_price);

        // Record sample
        TrainingSample sample;
        sample.window = std::move(encoded);
        sample.label = label;
        samples.push_back(std::move(sample));

        // Update position state machine
        if (label == 1) {
            // ENTER_LONG
            position_state = 1;
            entry_price = snapshots[t].mid_price;
        } else if (label == 2) {
            // ENTER_SHORT
            position_state = -1;
            entry_price = snapshots[t].mid_price;
        } else if (label == 3) {
            // EXIT
            position_state = 0;
            entry_price = std::numeric_limits<float>::quiet_NaN();
        }
        // label == 0 (HOLD): no state change
    }

    return samples;
}
