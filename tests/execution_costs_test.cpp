// execution_costs_test.cpp — TDD RED phase tests for ExecutionCosts
// Spec: .kit/docs/oracle-replay.md §ExecutionCosts
//
// Tests the ExecutionCosts struct: default values, per-side cost computation,
// and round-trip cost computation with FIXED and EMPIRICAL spread models.
//
// No implementation files exist yet — all tests must fail to compile.

#include <gtest/gtest.h>

// Header that the implementation must provide (spec §Project Structure):
#include "backtest/execution_costs.hpp"

#include <cmath>

// ===========================================================================
// Fixture
// ===========================================================================
class ExecutionCostsTest : public ::testing::Test {
protected:
    ExecutionCosts costs;  // default-constructed
};

// ===========================================================================
// 1. Default values
// ===========================================================================

TEST_F(ExecutionCostsTest, DefaultCommissionPerSide) {
    EXPECT_FLOAT_EQ(costs.commission_per_side, 0.62f);
}

TEST_F(ExecutionCostsTest, DefaultSpreadModelIsFixed) {
    EXPECT_EQ(costs.spread_model, ExecutionCosts::SpreadModel::FIXED);
}

TEST_F(ExecutionCostsTest, DefaultFixedSpreadTicks) {
    EXPECT_EQ(costs.fixed_spread_ticks, 1);
}

TEST_F(ExecutionCostsTest, DefaultSlippageTicks) {
    EXPECT_EQ(costs.slippage_ticks, 0);
}

TEST_F(ExecutionCostsTest, DefaultContractMultiplier) {
    EXPECT_FLOAT_EQ(costs.contract_multiplier, 5.0f);
}

TEST_F(ExecutionCostsTest, DefaultTickSize) {
    EXPECT_FLOAT_EQ(costs.tick_size, 0.25f);
}

TEST_F(ExecutionCostsTest, DefaultTickValue) {
    EXPECT_FLOAT_EQ(costs.tick_value, 1.25f);
}

// ===========================================================================
// 2. per_side_cost — FIXED spread model
// ===========================================================================

TEST_F(ExecutionCostsTest, PerSideCostFixedSpreadDefault) {
    // Per-side = commission + (fixed_spread_ticks / 2) * tick_value + slippage * tick_value
    // = 0.62 + (1 / 2.0) * 1.25 + 0 = 0.62 + 0.625 = 1.245
    float expected = 0.62f + 0.5f * 1.25f;
    EXPECT_FLOAT_EQ(costs.per_side_cost(), expected);
}

TEST_F(ExecutionCostsTest, PerSideCostFixedIgnoresActualSpread) {
    // In FIXED mode, the actual_spread_ticks argument should be ignored.
    float cost_default = costs.per_side_cost(1.0f);
    float cost_wide = costs.per_side_cost(3.0f);
    EXPECT_FLOAT_EQ(cost_default, cost_wide);
}

TEST_F(ExecutionCostsTest, PerSideCostWithSlippage) {
    costs.slippage_ticks = 2;
    // commission + half_spread_cost + slippage_cost
    // = 0.62 + 0.625 + 2 * 1.25 = 0.62 + 0.625 + 2.50 = 3.745
    float expected = 0.62f + 0.5f * 1.25f + 2.0f * 1.25f;
    EXPECT_FLOAT_EQ(costs.per_side_cost(), expected);
}

TEST_F(ExecutionCostsTest, PerSideCostWiderFixedSpread) {
    costs.fixed_spread_ticks = 2;
    // = 0.62 + (2 / 2.0) * 1.25 + 0 = 0.62 + 1.25 = 1.87
    float expected = 0.62f + 1.0f * 1.25f;
    EXPECT_FLOAT_EQ(costs.per_side_cost(), expected);
}

// ===========================================================================
// 3. per_side_cost — EMPIRICAL spread model
// ===========================================================================

TEST_F(ExecutionCostsTest, PerSideCostEmpiricalUsesActualSpread) {
    costs.spread_model = ExecutionCosts::SpreadModel::EMPIRICAL;
    float actual_spread_ticks = 2.0f;
    // = 0.62 + (2.0 / 2.0) * 1.25 + 0 = 0.62 + 1.25 = 1.87
    float expected = 0.62f + (actual_spread_ticks / 2.0f) * 1.25f;
    EXPECT_FLOAT_EQ(costs.per_side_cost(actual_spread_ticks), expected);
}

TEST_F(ExecutionCostsTest, PerSideCostEmpiricalNarrowSpread) {
    costs.spread_model = ExecutionCosts::SpreadModel::EMPIRICAL;
    float actual_spread_ticks = 0.5f;
    float expected = 0.62f + (0.5f / 2.0f) * 1.25f;
    EXPECT_FLOAT_EQ(costs.per_side_cost(actual_spread_ticks), expected);
}

TEST_F(ExecutionCostsTest, PerSideCostEmpiricalWithSlippage) {
    costs.spread_model = ExecutionCosts::SpreadModel::EMPIRICAL;
    costs.slippage_ticks = 1;
    float actual_spread_ticks = 3.0f;
    // = 0.62 + (3.0 / 2.0) * 1.25 + 1 * 1.25 = 0.62 + 1.875 + 1.25 = 3.745
    float expected = 0.62f + (3.0f / 2.0f) * 1.25f + 1.0f * 1.25f;
    EXPECT_FLOAT_EQ(costs.per_side_cost(actual_spread_ticks), expected);
}

// ===========================================================================
// 4. round_trip_cost
// ===========================================================================

TEST_F(ExecutionCostsTest, RoundTripCostMinimumDefault) {
    // Two per-side costs: entry + exit
    // = 2 * (0.62 + 0.625) = 2 * 1.245 = 2.49
    float per_side = costs.per_side_cost();
    float expected = 2.0f * per_side;
    EXPECT_FLOAT_EQ(costs.round_trip_cost(1.0f, 1.0f), expected);
}

TEST_F(ExecutionCostsTest, RoundTripCostEmpiricalDifferentSpreads) {
    costs.spread_model = ExecutionCosts::SpreadModel::EMPIRICAL;
    float entry_spread = 1.0f;
    float exit_spread = 3.0f;
    float entry_cost = costs.per_side_cost(entry_spread);
    float exit_cost = costs.per_side_cost(exit_spread);
    EXPECT_FLOAT_EQ(costs.round_trip_cost(entry_spread, exit_spread),
                    entry_cost + exit_cost);
}

TEST_F(ExecutionCostsTest, RoundTripCostCommissionAppliedBothSides) {
    // Verify commission appears in both entry and exit
    // Minimum round-trip commission = 2 * 0.62 = $1.24
    float rt = costs.round_trip_cost(1.0f, 1.0f);
    // Should be at least $1.24 (commission alone)
    EXPECT_GE(rt, 2.0f * 0.62f);
}

TEST_F(ExecutionCostsTest, RoundTripCostSpecMinimum) {
    // Spec states: "Per round-trip: $1.24 commission + $1.25 spread (1 tick) + slippage = ~$2.49 minimum"
    float rt = costs.round_trip_cost(1.0f, 1.0f);
    EXPECT_NEAR(rt, 2.49f, 0.01f);
}

// ===========================================================================
// 5. Edge cases
// ===========================================================================

TEST_F(ExecutionCostsTest, ZeroCommission) {
    costs.commission_per_side = 0.0f;
    float expected = 0.5f * 1.25f;  // half-spread only
    EXPECT_FLOAT_EQ(costs.per_side_cost(), expected);
}

TEST_F(ExecutionCostsTest, ZeroSpreadTicks) {
    costs.fixed_spread_ticks = 0;
    // = 0.62 + 0 + 0 = 0.62
    EXPECT_FLOAT_EQ(costs.per_side_cost(), 0.62f);
}

TEST_F(ExecutionCostsTest, TickValueConsistencyCheck) {
    // tick_value should equal contract_multiplier * tick_size
    EXPECT_FLOAT_EQ(costs.tick_value, costs.contract_multiplier * costs.tick_size);
}
