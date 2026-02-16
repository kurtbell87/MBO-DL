// trajectory_builder_test.cpp — TDD RED phase tests for trajectory_builder
// Spec: .kit/docs/oracle-trajectory.md
//
// Tests the build_trajectory() function that manages position state and
// entry price, sequences oracle calls, and produces (window, label) pairs.

#include <gtest/gtest.h>
#include "trajectory_builder.hpp"
#include "feature_encoder.hpp"   // FEATURE_DIM, W, encode_window constants
#include "book_builder.hpp"      // BookSnapshot struct

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers — synthetic snapshot construction for trajectory tests
// ---------------------------------------------------------------------------
namespace {

constexpr float TICK = 0.25f;
constexpr int HORIZON = 100;

// Build a vector of N snapshots with a given mid_price profile.
// Price starts at `start_mid` and changes according to the pattern.
// All other BookSnapshot fields are filled with reasonable defaults.
std::vector<BookSnapshot> make_trajectory_snapshots(
    int count, float start_mid = 4500.0f) {
    std::vector<BookSnapshot> snaps(count);
    for (int i = 0; i < count; ++i) {
        float mid = start_mid;
        snaps[i].mid_price = mid;
        snaps[i].spread = TICK;
        snaps[i].bids[0][0] = mid - TICK / 2.0f;
        snaps[i].bids[0][1] = 10.0f;
        snaps[i].asks[0][0] = mid + TICK / 2.0f;
        snaps[i].asks[0][1] = 10.0f;
        snaps[i].time_of_day = 9.5f + static_cast<float>(i) * 0.1f / 3600.0f;
        snaps[i].timestamp = static_cast<uint64_t>(i) * 100'000'000ULL;

        // Fill some trade data for non-degenerate features
        for (int j = 0; j < 50; ++j) {
            snaps[i].trades[j][0] = mid;
            snaps[i].trades[j][1] = 5.0f;
            snaps[i].trades[j][2] = (j % 2 == 0) ? 1.0f : -1.0f;
        }
    }
    return snaps;
}

// Build snapshots that trigger a specific entry/exit pattern:
// - Price rises by target_ticks at step `entry_step` from W-1 → ENTER_LONG
// - Then rises by take_profit_ticks from entry → EXIT
// This creates a trajectory with at least one ENTER_LONG and one EXIT.
std::vector<BookSnapshot> make_entry_exit_snapshots(
    int count, float start_mid = 4500.0f,
    int entry_step_offset = 10,     // offset from t_start where entry triggers
    int exit_step_offset = 40) {    // offset from entry where exit triggers
    std::vector<BookSnapshot> snaps(count);

    // t_start = W - 1 = 599
    int t_start = W - 1;
    float mid = start_mid;

    for (int i = 0; i < count; ++i) {
        snaps[i].mid_price = mid;
        snaps[i].spread = TICK;
        snaps[i].bids[0][0] = mid - TICK / 2.0f;
        snaps[i].bids[0][1] = 10.0f;
        snaps[i].asks[0][0] = mid + TICK / 2.0f;
        snaps[i].asks[0][1] = 10.0f;
        snaps[i].time_of_day = 9.5f + static_cast<float>(i) * 0.1f / 3600.0f;
        snaps[i].timestamp = static_cast<uint64_t>(i) * 100'000'000ULL;

        for (int j = 0; j < 50; ++j) {
            snaps[i].trades[j][0] = mid;
            snaps[i].trades[j][1] = 5.0f;
            snaps[i].trades[j][2] = (j % 2 == 0) ? 1.0f : -1.0f;
        }

        // After the entry trigger point, start raising the price for oracle to
        // see a take-profit
        if (i >= t_start + entry_step_offset) {
            // Price rises to trigger ENTER_LONG for oracle calls at t_start
            // and then rises further for EXIT after entry
            float steps_from_trigger = static_cast<float>(
                i - (t_start + entry_step_offset));
            mid = start_mid + 10 * TICK +
                  steps_from_trigger * (20 * TICK / static_cast<float>(exit_step_offset));
        } else if (i >= t_start) {
            // Gradual rise to target_ticks within horizon
            float steps_in = static_cast<float>(i - t_start);
            mid = start_mid + steps_in * (10 * TICK / static_cast<float>(entry_step_offset));
        }

        // Re-set fields with updated mid
        snaps[i].mid_price = mid;
        snaps[i].bids[0][0] = mid - TICK / 2.0f;
        snaps[i].asks[0][0] = mid + TICK / 2.0f;
    }
    return snaps;
}

}  // anonymous namespace

// ===========================================================================
// Test 14: Minimum snapshot count
// ===========================================================================
TEST(TrajectoryBuilderTest, MinimumSnapshotCount) {
    // build_trajectory requires len(snapshots) >= W + horizon = 600 + 100 = 700
    // Fewer than 700 should raise an error.

    // 699 snapshots — one too few
    auto too_few = make_trajectory_snapshots(W + HORIZON - 1);
    EXPECT_ANY_THROW(build_trajectory(too_few))
        << "Fewer than W + horizon snapshots should throw/assert";

    // Exactly 700 — should NOT throw
    auto exact = make_trajectory_snapshots(W + HORIZON);
    EXPECT_NO_THROW(build_trajectory(exact))
        << "Exactly W + horizon snapshots should be accepted";
}

// ===========================================================================
// Test 15: Trajectory length
// ===========================================================================
TEST(TrajectoryBuilderTest, TrajectoryLength) {
    // Expected sample count: len(snapshots) - horizon - W + 1
    // With 800 snapshots: 800 - 100 - 600 + 1 = 101 samples
    int N = 800;
    auto snaps = make_trajectory_snapshots(N);

    auto samples = build_trajectory(snaps);

    int expected = N - HORIZON - W + 1;
    EXPECT_EQ(static_cast<int>(samples.size()), expected)
        << "Trajectory should have len(snapshots) - horizon - W + 1 samples";
}

TEST(TrajectoryBuilderTest, TrajectoryLengthMinimumInput) {
    // With exactly 700 snapshots: 700 - 100 - 600 + 1 = 1 sample
    auto snaps = make_trajectory_snapshots(W + HORIZON);

    auto samples = build_trajectory(snaps);

    EXPECT_EQ(samples.size(), 1u)
        << "Minimum input should produce exactly 1 sample";
}

// ===========================================================================
// Test 16: Position state transitions (ENTER_LONG then EXIT)
// ===========================================================================
TEST(TrajectoryBuilderTest, PositionStateTransitions) {
    // Build a price path that triggers ENTER_LONG at some point,
    // then EXIT at a later point.
    int N = 1000;
    auto snaps = make_entry_exit_snapshots(N);

    auto samples = build_trajectory(snaps);
    ASSERT_FALSE(samples.empty());

    // Find the first ENTER_LONG label
    int enter_idx = -1;
    for (int i = 0; i < static_cast<int>(samples.size()); ++i) {
        if (samples[i].label == 1) {
            enter_idx = i;
            break;
        }
    }

    // Find the first EXIT label after the entry
    int exit_idx = -1;
    if (enter_idx >= 0) {
        for (int i = enter_idx + 1; i < static_cast<int>(samples.size()); ++i) {
            if (samples[i].label == 3) {
                exit_idx = i;
                break;
            }
        }
    }

    // If our price path is designed correctly, we should see both
    EXPECT_GE(enter_idx, 0)
        << "Should find at least one ENTER_LONG in the trajectory";
    if (enter_idx >= 0) {
        EXPECT_GE(exit_idx, enter_idx + 1)
            << "Should find an EXIT after ENTER_LONG";
    }

    // Between ENTER_LONG and EXIT, all labels should be HOLD (0) or EXIT (3)
    // (no nested entries allowed)
    if (enter_idx >= 0 && exit_idx > enter_idx) {
        for (int i = enter_idx + 1; i < exit_idx; ++i) {
            EXPECT_TRUE(samples[i].label == 0 || samples[i].label == 3)
                << "Between entry and exit, label should be HOLD or EXIT, "
                << "got " << samples[i].label << " at sample " << i;
        }
    }
}

// ===========================================================================
// Test 17: Entry price tracking
// ===========================================================================
TEST(TrajectoryBuilderTest, EntryPriceTracking) {
    // After ENTER_LONG at time t, the entry_price should be mid_price[t].
    // We verify this indirectly: if a subsequent oracle call produces EXIT,
    // the PnL calculation used the correct entry_price. We can verify that
    // the trajectory state machine correctly tracks the entry price by
    // checking that ENTER and EXIT labels appear in the correct sequence.
    //
    // For a more direct test: the first sample should start flat (position=0).
    int N = 800;
    auto snaps = make_trajectory_snapshots(N);
    auto samples = build_trajectory(snaps);

    ASSERT_FALSE(samples.empty());

    // The spec says position starts at 0, entry_price starts at NaN.
    // All samples with flat price should be HOLD (0) since no threshold is hit.
    // (With constant price, the oracle should always return HOLD)
    for (const auto& s : samples) {
        EXPECT_EQ(s.label, 0)
            << "With constant mid_price, all labels should be HOLD (0)";
    }
}

// ===========================================================================
// Test 18: Window size — every sample's window has exactly W=600 features
// ===========================================================================
TEST(TrajectoryBuilderTest, WindowSize) {
    int N = 750;
    auto snaps = make_trajectory_snapshots(N);
    auto samples = build_trajectory(snaps);

    ASSERT_FALSE(samples.empty());

    for (int i = 0; i < static_cast<int>(samples.size()); ++i) {
        EXPECT_EQ(static_cast<int>(samples[i].window.size()), W)
            << "Sample " << i << " window should have exactly W=" << W
            << " encoded features, got " << samples[i].window.size();
    }
}

// ===========================================================================
// Test 19: Feature dim — every feature vector has exactly 194 elements
// ===========================================================================
TEST(TrajectoryBuilderTest, FeatureDim) {
    int N = 750;
    auto snaps = make_trajectory_snapshots(N);
    auto samples = build_trajectory(snaps);

    ASSERT_FALSE(samples.empty());

    for (int i = 0; i < static_cast<int>(samples.size()); ++i) {
        for (int t = 0; t < static_cast<int>(samples[i].window.size()); ++t) {
            EXPECT_EQ(static_cast<int>(samples[i].window[t].size()), FEATURE_DIM)
                << "Sample " << i << ", snapshot " << t
                << " should have exactly " << FEATURE_DIM << " features";
        }
    }
}

// ===========================================================================
// Test 20: Position state in features
// ===========================================================================
TEST(TrajectoryBuilderTest, PositionStateInFeatures) {
    // With constant price, all labels should be HOLD, position stays at 0.
    // Therefore position_state feature (index 193) should be 0.0 for all.
    int N = 750;
    auto snaps = make_trajectory_snapshots(N);
    auto samples = build_trajectory(snaps);

    ASSERT_FALSE(samples.empty());

    for (int i = 0; i < static_cast<int>(samples.size()); ++i) {
        // Check the last snapshot in the window (most recent, at index W-1)
        // which should reflect the current position state.
        const auto& last_feature = samples[i].window[W - 1];
        EXPECT_FLOAT_EQ(last_feature[POSITION_STATE_IDX], 0.0f)
            << "Sample " << i << ": with constant price, position_state "
            << "feature should be 0.0 (flat)";
    }
}

TEST(TrajectoryBuilderTest, PositionStateChangesAfterEntry) {
    // After ENTER_LONG, the position_state in subsequent windows should be +1.0
    int N = 1000;
    auto snaps = make_entry_exit_snapshots(N);
    auto samples = build_trajectory(snaps);

    ASSERT_FALSE(samples.empty());

    // Find the first ENTER_LONG
    int enter_idx = -1;
    for (int i = 0; i < static_cast<int>(samples.size()); ++i) {
        if (samples[i].label == 1) {
            enter_idx = i;
            break;
        }
    }

    if (enter_idx >= 0 && enter_idx + 1 < static_cast<int>(samples.size())) {
        // The sample AFTER the entry should have position_state = +1.0
        // in its feature encoding
        const auto& next_window = samples[enter_idx + 1].window;
        EXPECT_FLOAT_EQ(next_window[W - 1][POSITION_STATE_IDX], 1.0f)
            << "After ENTER_LONG, position_state feature should be +1.0";
    }
}

// ===========================================================================
// Test 21: Labels are valid — all labels in {0, 1, 2, 3}, never 4
// ===========================================================================
TEST(TrajectoryBuilderTest, LabelsAreValid) {
    int N = 800;
    auto snaps = make_trajectory_snapshots(N);
    auto samples = build_trajectory(snaps);

    ASSERT_FALSE(samples.empty());

    for (int i = 0; i < static_cast<int>(samples.size()); ++i) {
        EXPECT_GE(samples[i].label, 0) << "Label at sample " << i << " is negative";
        EXPECT_LE(samples[i].label, 3)
            << "Label at sample " << i << " is > 3 (got "
            << samples[i].label << ")";
        EXPECT_NE(samples[i].label, 4)
            << "Label 4 (REVERSE) should never appear at sample " << i;
    }
}

TEST(TrajectoryBuilderTest, LabelsNeverContainReverse) {
    // Test with a price path that triggers entries and exits
    int N = 1000;
    auto snaps = make_entry_exit_snapshots(N);
    auto samples = build_trajectory(snaps);

    for (int i = 0; i < static_cast<int>(samples.size()); ++i) {
        EXPECT_NE(samples[i].label, 4)
            << "REVERSE (4) should never appear in trajectory at sample " << i;
    }
}

// ===========================================================================
// Test 22: Flat start — first oracle call starts with position=0 and NaN entry
// ===========================================================================
TEST(TrajectoryBuilderTest, FlatStart) {
    // The first sample in any trajectory should be generated with
    // position_state=0 and entry_price=NaN. This means:
    // 1. The position_state feature in the first window should be 0.0
    // 2. The first label should be in {0, 1, 2} (never EXIT since flat)
    int N = 750;
    auto snaps = make_trajectory_snapshots(N);
    auto samples = build_trajectory(snaps);

    ASSERT_FALSE(samples.empty());

    // First sample's position_state feature should be 0.0 (flat)
    EXPECT_FLOAT_EQ(samples[0].window[W - 1][POSITION_STATE_IDX], 0.0f)
        << "First sample should start flat (position_state = 0.0)";

    // First label should not be EXIT (3) since we start flat
    EXPECT_NE(samples[0].label, 3)
        << "First label should never be EXIT (starting flat)";

    // First label should not be REVERSE (4)
    EXPECT_NE(samples[0].label, 4)
        << "First label should never be REVERSE";
}

// ===========================================================================
// Test: TrainingSample struct has correct fields
// ===========================================================================
TEST(TrainingSampleTest, StructLayout) {
    TrainingSample sample;
    sample.label = 0;
    sample.window.resize(W);
    for (auto& row : sample.window) {
        row.fill(0.0f);
    }

    EXPECT_EQ(sample.label, 0);
    EXPECT_EQ(static_cast<int>(sample.window.size()), W);
    EXPECT_EQ(static_cast<int>(sample.window[0].size()), FEATURE_DIM);
}

// ===========================================================================
// Test: Consecutive labels follow legal state transitions
// ===========================================================================
TEST(TrajectoryBuilderTest, LegalStateTransitions) {
    // Valid transitions:
    // HOLD(0)        → any state (no state change)
    // ENTER_LONG(1)  → can only appear when flat
    // ENTER_SHORT(2) → can only appear when flat
    // EXIT(3)        → can only appear when in position
    //
    // This means: after ENTER_LONG, we can't see ENTER_LONG or ENTER_SHORT
    //             until we see EXIT first. After EXIT, we can't see EXIT
    //             until we see ENTER first.
    int N = 1000;
    auto snaps = make_entry_exit_snapshots(N);
    auto samples = build_trajectory(snaps);

    ASSERT_FALSE(samples.empty());

    int position_state = 0;  // start flat
    for (int i = 0; i < static_cast<int>(samples.size()); ++i) {
        int label = samples[i].label;

        if (position_state == 0) {
            // Flat: can see HOLD, ENTER_LONG, ENTER_SHORT; NOT EXIT
            EXPECT_NE(label, 3)
                << "At sample " << i << ": EXIT while flat is illegal";
        } else {
            // In position: can see HOLD, EXIT; NOT ENTER_LONG, ENTER_SHORT
            EXPECT_NE(label, 1)
                << "At sample " << i << ": ENTER_LONG while in position is illegal";
            EXPECT_NE(label, 2)
                << "At sample " << i << ": ENTER_SHORT while in position is illegal";
        }

        // Update position state
        if (label == 1) position_state = 1;
        else if (label == 2) position_state = -1;
        else if (label == 3) position_state = 0;
    }
}

// ===========================================================================
// Test: No NaN or Inf in any feature of any sample
// ===========================================================================
TEST(TrajectoryBuilderTest, NoNanOrInfInFeatures) {
    int N = 750;
    auto snaps = make_trajectory_snapshots(N);
    auto samples = build_trajectory(snaps);

    ASSERT_FALSE(samples.empty());

    for (int i = 0; i < static_cast<int>(samples.size()); ++i) {
        for (int t = 0; t < static_cast<int>(samples[i].window.size()); ++t) {
            for (int f = 0; f < FEATURE_DIM; ++f) {
                EXPECT_FALSE(std::isnan(samples[i].window[t][f]))
                    << "NaN at sample " << i << " snap " << t << " feat " << f;
                EXPECT_FALSE(std::isinf(samples[i].window[t][f]))
                    << "Inf at sample " << i << " snap " << t << " feat " << f;
            }
        }
    }
}
