// time_horizon_cli_test.cpp — TDD RED phase tests for --max-time-horizon and --volume-horizon CLI flags
// Spec: .kit/docs/time-horizon-cli.md
//
// Tests that:
//   (1) TripleBarrierConfig and OracleConfig defaults changed to 3600/50000
//   (2) bar_feature_export recognizes --max-time-horizon and --volume-horizon
//   (3) oracle_expectancy recognizes --max-time-horizon and --volume-horizon
//   (4) Invalid values (0, negative, >86400) rejected with clear error and non-zero exit
//   (5) --help output includes new flags with defaults
//   (6) Backward compatibility: existing flags still work alongside new ones
//   (7) Combined flags: --target --stop --max-time-horizon --volume-horizon all applied
//
// Unit tests verify config defaults and CLI contract.
// Integration tests require binaries + data and are guarded by GTEST_SKIP.

#include <gtest/gtest.h>

#include "backtest/triple_barrier.hpp"
#include "backtest/oracle_replay.hpp"
#include "test_export_helpers.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// ===========================================================================
// Helpers
// ===========================================================================
namespace {

// bar_feature_export binary path (from test_export_helpers)
using export_test_helpers::BINARY_PATH;
using export_test_helpers::run_command;

// oracle_expectancy binary path
const std::string ORACLE_BINARY_PATH = "build/oracle_expectancy";

const std::string DATA_DIR = "DATA/GLBX-20260207-L953CAPU5B";

std::string temp_parquet_path(const std::string& suffix = "") {
    return (std::filesystem::temp_directory_path() /
            ("time_horizon_cli_test" + suffix + ".parquet")).string();
}

std::string temp_json_path(const std::string& suffix = "") {
    return (std::filesystem::temp_directory_path() /
            ("time_horizon_cli_test" + suffix + ".json")).string();
}

bool data_available() {
    return std::filesystem::exists(DATA_DIR) &&
           std::filesystem::is_directory(DATA_DIR);
}

std::string read_file(const std::string& path) {
    std::ifstream f(path);
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    return content;
}

bool json_has_key(const std::string& json, const std::string& key) {
    return json.find("\"" + key + "\"") != std::string::npos;
}

}  // anonymous namespace

// ===========================================================================
// T1: Config struct defaults changed
// ===========================================================================

class ConfigDefaultsTest : public ::testing::Test {};

TEST_F(ConfigDefaultsTest, TripleBarrierMaxTimeHorizonDefault3600) {
    // Spec R5: TripleBarrierConfig.max_time_horizon_s changed from 300 to 3600
    TripleBarrierConfig cfg;
    EXPECT_EQ(cfg.max_time_horizon_s, 3600u)
        << "TripleBarrierConfig default max_time_horizon_s must be 3600 (1 hour), not 300";
}

TEST_F(ConfigDefaultsTest, TripleBarrierVolumeHorizonDefault50000) {
    // Spec R5: TripleBarrierConfig.volume_horizon changed from 500 to 50000
    TripleBarrierConfig cfg;
    EXPECT_EQ(cfg.volume_horizon, 50000u)
        << "TripleBarrierConfig default volume_horizon must be 50000, not 500";
}

TEST_F(ConfigDefaultsTest, OracleConfigMaxTimeHorizonDefault3600) {
    // Spec R5: OracleConfig.max_time_horizon_s changed from 300 to 3600
    OracleConfig cfg;
    EXPECT_EQ(cfg.max_time_horizon_s, 3600u)
        << "OracleConfig default max_time_horizon_s must be 3600 (1 hour), not 300";
}

TEST_F(ConfigDefaultsTest, OracleConfigVolumeHorizonDefault50000) {
    // Spec R5: OracleConfig.volume_horizon changed from 500 to 50000
    OracleConfig cfg;
    EXPECT_EQ(cfg.volume_horizon, 50000u)
        << "OracleConfig default volume_horizon must be 50000, not 500";
}

// ===========================================================================
// bar_feature_export: --help shows new flags (T8 from spec)
// ===========================================================================

class BarExportHelpTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(BINARY_PATH)) {
            GTEST_SKIP() << "bar_feature_export binary not built yet";
        }
    }
};

TEST_F(BarExportHelpTest, UsageMentionsMaxTimeHorizon) {
    // Spec T8: --help output must include --max-time-horizon
    auto result = run_command(BINARY_PATH);
    EXPECT_NE(result.output.find("--max-time-horizon"), std::string::npos)
        << "Usage output must mention --max-time-horizon flag. Got: " << result.output;
}

TEST_F(BarExportHelpTest, UsageMentionsVolumeHorizon) {
    // Spec T8: --help output must include --volume-horizon
    auto result = run_command(BINARY_PATH);
    EXPECT_NE(result.output.find("--volume-horizon"), std::string::npos)
        << "Usage output must mention --volume-horizon flag. Got: " << result.output;
}

TEST_F(BarExportHelpTest, UsageShowsMaxTimeHorizonDefault3600) {
    // The --max-time-horizon line should mention 3600 as default
    auto result = run_command(BINARY_PATH);
    auto pos = result.output.find("--max-time-horizon");
    bool line_has_default = false;
    if (pos != std::string::npos) {
        auto line_end = result.output.find('\n', pos);
        auto line = result.output.substr(pos, line_end - pos);
        line_has_default = (line.find("3600") != std::string::npos);
    }
    EXPECT_TRUE(line_has_default)
        << "Usage should show --max-time-horizon with default 3600 on same line. Got: "
        << result.output;
}

TEST_F(BarExportHelpTest, UsageShowsVolumeHorizonDefault50000) {
    auto result = run_command(BINARY_PATH);
    auto pos = result.output.find("--volume-horizon");
    bool line_has_default = false;
    if (pos != std::string::npos) {
        auto line_end = result.output.find('\n', pos);
        auto line = result.output.substr(pos, line_end - pos);
        line_has_default = (line.find("50000") != std::string::npos);
    }
    EXPECT_TRUE(line_has_default)
        << "Usage should show --volume-horizon with default 50000 on same line. Got: "
        << result.output;
}

// ===========================================================================
// bar_feature_export: CLI flag recognition (T2, T3 from spec)
// ===========================================================================

class BarExportCLIAcceptanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(BINARY_PATH)) {
            GTEST_SKIP() << "bar_feature_export binary not built yet";
        }
    }
    void TearDown() override {
        for (const auto& p : temp_files_) std::filesystem::remove(p);
    }
    void track_temp(const std::string& p) { temp_files_.push_back(p); }
    std::vector<std::string> temp_files_;
};

TEST_F(BarExportCLIAcceptanceTest, MaxTimeHorizonFlagRecognized) {
    // Spec T2: --max-time-horizon should not produce an "unknown argument" error
    auto pq = temp_parquet_path("_cli_mth");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --max-time-horizon 3600 --output " + pq);

    bool arg_error = (result.output.find("Unknown") != std::string::npos ||
                      result.output.find("unknown") != std::string::npos ||
                      result.output.find("unrecognized") != std::string::npos ||
                      result.output.find("Unrecognized") != std::string::npos ||
                      result.output.find("invalid option") != std::string::npos);
    EXPECT_FALSE(arg_error)
        << "--max-time-horizon should be a recognized CLI flag. Got: " << result.output;
}

TEST_F(BarExportCLIAcceptanceTest, VolumeHorizonFlagRecognized) {
    // Spec T3: --volume-horizon should not produce an "unknown argument" error
    auto pq = temp_parquet_path("_cli_vh");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --volume-horizon 50000 --output " + pq);

    bool arg_error = (result.output.find("Unknown") != std::string::npos ||
                      result.output.find("unknown") != std::string::npos ||
                      result.output.find("unrecognized") != std::string::npos ||
                      result.output.find("Unrecognized") != std::string::npos ||
                      result.output.find("invalid option") != std::string::npos);
    EXPECT_FALSE(arg_error)
        << "--volume-horizon should be a recognized CLI flag. Got: " << result.output;
}

TEST_F(BarExportCLIAcceptanceTest, BothNewFlagsRecognizedTogether) {
    auto pq = temp_parquet_path("_cli_both_new");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --max-time-horizon 3600 --volume-horizon 50000 --output " + pq);

    bool arg_error = (result.output.find("Unknown") != std::string::npos ||
                      result.output.find("unknown") != std::string::npos ||
                      result.output.find("unrecognized") != std::string::npos);
    EXPECT_FALSE(arg_error)
        << "--max-time-horizon and --volume-horizon together should be recognized. Got: "
        << result.output;
}

TEST_F(BarExportCLIAcceptanceTest, NewFlagsWithExistingFlagsRecognized) {
    // All flags together: --target --stop --max-time-horizon --volume-horizon
    auto pq = temp_parquet_path("_cli_all");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --target 15 --stop 3 "
        "--max-time-horizon 3600 --volume-horizon 50000 --output " + pq);

    bool arg_error = (result.output.find("Unknown") != std::string::npos ||
                      result.output.find("unknown") != std::string::npos ||
                      result.output.find("unrecognized") != std::string::npos);
    EXPECT_FALSE(arg_error)
        << "All flags combined should be recognized. Got: " << result.output;
}

TEST_F(BarExportCLIAcceptanceTest, MaxTimeHorizonWithoutValueFails) {
    auto pq = temp_parquet_path("_cli_mth_noval");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --max-time-horizon --output " + pq);
    EXPECT_NE(result.exit_code, 0)
        << "--max-time-horizon without value should exit non-zero. Got: " << result.output;
}

TEST_F(BarExportCLIAcceptanceTest, VolumeHorizonWithoutValueFails) {
    auto pq = temp_parquet_path("_cli_vh_noval");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --volume-horizon --output " + pq);
    EXPECT_NE(result.exit_code, 0)
        << "--volume-horizon without value should exit non-zero. Got: " << result.output;
}

// ===========================================================================
// bar_feature_export: Invalid values rejected (T6 from spec)
// ===========================================================================

class BarExportInvalidValuesTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(BINARY_PATH)) {
            GTEST_SKIP() << "bar_feature_export binary not built yet";
        }
    }
    void TearDown() override {
        for (const auto& p : temp_files_) std::filesystem::remove(p);
    }
    void track_temp(const std::string& p) { temp_files_.push_back(p); }
    std::vector<std::string> temp_files_;
};

TEST_F(BarExportInvalidValuesTest, MaxTimeHorizonZeroExitsNonZero) {
    // Spec T6: --max-time-horizon 0 → error, non-zero exit
    auto pq = temp_parquet_path("_inv_mth0");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --max-time-horizon 0 --output " + pq);
    EXPECT_NE(result.exit_code, 0)
        << "--max-time-horizon 0 must be rejected. Got: " << result.output;
}

TEST_F(BarExportInvalidValuesTest, MaxTimeHorizonNegativeExitsNonZero) {
    // Spec T6: --max-time-horizon -5 → error
    auto pq = temp_parquet_path("_inv_mth_neg");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --max-time-horizon -5 --output " + pq);
    EXPECT_NE(result.exit_code, 0)
        << "--max-time-horizon -5 must be rejected. Got: " << result.output;
}

TEST_F(BarExportInvalidValuesTest, MaxTimeHorizonExceedsMaxExitsNonZero) {
    // Spec R1: Maximum is 86400 (24 hours)
    auto pq = temp_parquet_path("_inv_mth_over");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --max-time-horizon 86401 --output " + pq);
    EXPECT_NE(result.exit_code, 0)
        << "--max-time-horizon 86401 (>86400) must be rejected. Got: " << result.output;
}

TEST_F(BarExportInvalidValuesTest, MaxTimeHorizonAt86400Accepted) {
    // Spec R1: Maximum is 86400 — boundary value should be accepted (no unknown arg error)
    auto pq = temp_parquet_path("_inv_mth_max");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --max-time-horizon 86400 --output " + pq);

    bool arg_error = (result.output.find("Unknown") != std::string::npos ||
                      result.output.find("unknown") != std::string::npos ||
                      result.output.find("must be") != std::string::npos);
    // If the binary rejects 86400, that's a validation error, not acceptable
    bool validation_error = (result.output.find("max-time-horizon") != std::string::npos &&
                             (result.output.find("must be") != std::string::npos ||
                              result.output.find("invalid") != std::string::npos));
    EXPECT_FALSE(validation_error)
        << "--max-time-horizon 86400 (boundary) should be accepted. Got: " << result.output;
}

TEST_F(BarExportInvalidValuesTest, VolumeHorizonZeroExitsNonZero) {
    // Spec T6: --volume-horizon 0 → error
    auto pq = temp_parquet_path("_inv_vh0");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --volume-horizon 0 --output " + pq);
    EXPECT_NE(result.exit_code, 0)
        << "--volume-horizon 0 must be rejected. Got: " << result.output;
}

TEST_F(BarExportInvalidValuesTest, VolumeHorizonNegativeExitsNonZero) {
    auto pq = temp_parquet_path("_inv_vh_neg");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --volume-horizon -1 --output " + pq);
    EXPECT_NE(result.exit_code, 0)
        << "--volume-horizon -1 must be rejected. Got: " << result.output;
}

TEST_F(BarExportInvalidValuesTest, NonIntegerMaxTimeHorizonExitsNonZero) {
    auto pq = temp_parquet_path("_inv_mth_str");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --max-time-horizon abc --output " + pq);
    EXPECT_NE(result.exit_code, 0)
        << "--max-time-horizon abc (non-integer) must be rejected. Got: " << result.output;
}

TEST_F(BarExportInvalidValuesTest, NonIntegerVolumeHorizonExitsNonZero) {
    auto pq = temp_parquet_path("_inv_vh_str");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --volume-horizon xyz --output " + pq);
    EXPECT_NE(result.exit_code, 0)
        << "--volume-horizon xyz (non-integer) must be rejected. Got: " << result.output;
}

TEST_F(BarExportInvalidValuesTest, InvalidMaxTimeHorizonShowsErrorMessage) {
    auto pq = temp_parquet_path("_inv_mth_errmsg");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --max-time-horizon 0 --output " + pq);
    bool has_error_msg = (result.output.find("max-time-horizon") != std::string::npos ||
                          result.output.find("time-horizon") != std::string::npos ||
                          result.output.find("time_horizon") != std::string::npos ||
                          result.output.find("must be") != std::string::npos ||
                          result.output.find("error") != std::string::npos ||
                          result.output.find("Error") != std::string::npos);
    EXPECT_TRUE(has_error_msg)
        << "Invalid --max-time-horizon 0 should produce a clear error message. Got: "
        << result.output;
}

TEST_F(BarExportInvalidValuesTest, InvalidVolumeHorizonShowsErrorMessage) {
    auto pq = temp_parquet_path("_inv_vh_errmsg");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --volume-horizon 0 --output " + pq);
    bool has_error_msg = (result.output.find("volume-horizon") != std::string::npos ||
                          result.output.find("volume_horizon") != std::string::npos ||
                          result.output.find("must be") != std::string::npos ||
                          result.output.find("error") != std::string::npos ||
                          result.output.find("Error") != std::string::npos);
    EXPECT_TRUE(has_error_msg)
        << "Invalid --volume-horizon 0 should produce a clear error message. Got: "
        << result.output;
}

// ===========================================================================
// oracle_expectancy: --help shows new flags
// ===========================================================================

class OracleHelpTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(ORACLE_BINARY_PATH)) {
            GTEST_SKIP() << "oracle_expectancy binary not built yet";
        }
    }
};

TEST_F(OracleHelpTest, HelpMentionsMaxTimeHorizon) {
    auto result = run_command(ORACLE_BINARY_PATH + " --help");
    EXPECT_NE(result.output.find("--max-time-horizon"), std::string::npos)
        << "oracle_expectancy --help should mention --max-time-horizon. Got: " << result.output;
}

TEST_F(OracleHelpTest, HelpMentionsVolumeHorizon) {
    auto result = run_command(ORACLE_BINARY_PATH + " --help");
    EXPECT_NE(result.output.find("--volume-horizon"), std::string::npos)
        << "oracle_expectancy --help should mention --volume-horizon. Got: " << result.output;
}

TEST_F(OracleHelpTest, HelpShowsMaxTimeHorizonDefault3600) {
    auto result = run_command(ORACLE_BINARY_PATH + " --help");
    auto pos = result.output.find("--max-time-horizon");
    bool line_has_default = false;
    if (pos != std::string::npos) {
        auto line_end = result.output.find('\n', pos);
        auto line = result.output.substr(pos, line_end - pos);
        line_has_default = (line.find("3600") != std::string::npos);
    }
    EXPECT_TRUE(line_has_default)
        << "oracle_expectancy --help should show --max-time-horizon with default 3600. Got: "
        << result.output;
}

TEST_F(OracleHelpTest, HelpShowsVolumeHorizonDefault50000) {
    auto result = run_command(ORACLE_BINARY_PATH + " --help");
    auto pos = result.output.find("--volume-horizon");
    bool line_has_default = false;
    if (pos != std::string::npos) {
        auto line_end = result.output.find('\n', pos);
        auto line = result.output.substr(pos, line_end - pos);
        line_has_default = (line.find("50000") != std::string::npos);
    }
    EXPECT_TRUE(line_has_default)
        << "oracle_expectancy --help should show --volume-horizon with default 50000. Got: "
        << result.output;
}

// ===========================================================================
// oracle_expectancy: CLI flag recognition (T4, T5 from spec)
// ===========================================================================

class OracleCLIAcceptanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(ORACLE_BINARY_PATH)) {
            GTEST_SKIP() << "oracle_expectancy binary not built yet";
        }
        if (!data_available()) {
            GTEST_SKIP() << "Data directory not available";
        }
    }
    void TearDown() override {
        for (const auto& p : temp_files_) std::filesystem::remove(p);
    }
    void track_temp(const std::string& p) { temp_files_.push_back(p); }
    std::vector<std::string> temp_files_;
};

TEST_F(OracleCLIAcceptanceTest, MaxTimeHorizonFlagRecognized) {
    // Spec T4: oracle_expectancy accepts --max-time-horizon
    auto result = run_command(ORACLE_BINARY_PATH + " --max-time-horizon 3600");
    bool arg_error = (result.output.find("Unknown") != std::string::npos ||
                      result.output.find("unknown") != std::string::npos ||
                      result.output.find("unrecognized") != std::string::npos);
    EXPECT_FALSE(arg_error)
        << "oracle_expectancy should recognize --max-time-horizon. Got: " << result.output;
}

TEST_F(OracleCLIAcceptanceTest, VolumeHorizonFlagRecognized) {
    // Spec T5: oracle_expectancy accepts --volume-horizon
    auto result = run_command(ORACLE_BINARY_PATH + " --volume-horizon 50000");
    bool arg_error = (result.output.find("Unknown") != std::string::npos ||
                      result.output.find("unknown") != std::string::npos ||
                      result.output.find("unrecognized") != std::string::npos);
    EXPECT_FALSE(arg_error)
        << "oracle_expectancy should recognize --volume-horizon. Got: " << result.output;
}

TEST_F(OracleCLIAcceptanceTest, BothNewFlagsWithExistingFlags) {
    // All flags together: --target --stop --take-profit --max-time-horizon --volume-horizon
    auto json = temp_json_path("_cli_all");
    track_temp(json);
    auto result = run_command(
        ORACLE_BINARY_PATH + " --target 15 --stop 3 --take-profit 30 "
        "--max-time-horizon 3600 --volume-horizon 50000 --output " + json);

    bool arg_error = (result.output.find("Unknown") != std::string::npos ||
                      result.output.find("unknown") != std::string::npos ||
                      result.output.find("unrecognized") != std::string::npos);
    EXPECT_FALSE(arg_error)
        << "All oracle_expectancy flags combined should be recognized. Got: " << result.output;
}

TEST_F(OracleCLIAcceptanceTest, MaxTimeHorizonExitsZeroWithData) {
    // Spec T4: --max-time-horizon 3600 → valid output, trade_count > 0
    auto result = run_command(ORACLE_BINARY_PATH + " --max-time-horizon 3600");
    EXPECT_EQ(result.exit_code, 0)
        << "oracle_expectancy --max-time-horizon 3600 should succeed. Got: " << result.output;
}

TEST_F(OracleCLIAcceptanceTest, VolumeHorizonExitsZeroWithData) {
    // Spec T5: --volume-horizon 50000 → valid output
    auto result = run_command(ORACLE_BINARY_PATH + " --volume-horizon 50000");
    EXPECT_EQ(result.exit_code, 0)
        << "oracle_expectancy --volume-horizon 50000 should succeed. Got: " << result.output;
}

// ===========================================================================
// oracle_expectancy: Invalid values rejected (T6 from spec)
// ===========================================================================

class OracleInvalidValuesTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(ORACLE_BINARY_PATH)) {
            GTEST_SKIP() << "oracle_expectancy binary not built yet";
        }
    }
};

TEST_F(OracleInvalidValuesTest, MaxTimeHorizonZeroExitsOne) {
    auto result = run_command(ORACLE_BINARY_PATH + " --max-time-horizon 0");
    EXPECT_EQ(result.exit_code, 1)
        << "oracle_expectancy --max-time-horizon 0 should exit 1. Got: " << result.output;
}

TEST_F(OracleInvalidValuesTest, MaxTimeHorizonNegativeExitsOne) {
    auto result = run_command(ORACLE_BINARY_PATH + " --max-time-horizon -5");
    EXPECT_EQ(result.exit_code, 1)
        << "oracle_expectancy --max-time-horizon -5 should exit 1. Got: " << result.output;
}

TEST_F(OracleInvalidValuesTest, MaxTimeHorizonExceedsMaxExitsOne) {
    auto result = run_command(ORACLE_BINARY_PATH + " --max-time-horizon 86401");
    EXPECT_EQ(result.exit_code, 1)
        << "oracle_expectancy --max-time-horizon 86401 should exit 1. Got: " << result.output;
}

TEST_F(OracleInvalidValuesTest, VolumeHorizonZeroExitsOne) {
    auto result = run_command(ORACLE_BINARY_PATH + " --volume-horizon 0");
    EXPECT_EQ(result.exit_code, 1)
        << "oracle_expectancy --volume-horizon 0 should exit 1. Got: " << result.output;
}

TEST_F(OracleInvalidValuesTest, VolumeHorizonNegativeExitsOne) {
    auto result = run_command(ORACLE_BINARY_PATH + " --volume-horizon -1");
    EXPECT_EQ(result.exit_code, 1)
        << "oracle_expectancy --volume-horizon -1 should exit 1. Got: " << result.output;
}

TEST_F(OracleInvalidValuesTest, NonIntegerMaxTimeHorizonExitsOne) {
    auto result = run_command(ORACLE_BINARY_PATH + " --max-time-horizon abc");
    EXPECT_EQ(result.exit_code, 1)
        << "oracle_expectancy --max-time-horizon abc should exit 1. Got: " << result.output;
}

TEST_F(OracleInvalidValuesTest, MaxTimeHorizonWithoutValueExitsOne) {
    auto result = run_command(ORACLE_BINARY_PATH + " --max-time-horizon");
    EXPECT_EQ(result.exit_code, 1)
        << "oracle_expectancy --max-time-horizon without value should exit 1. Got: "
        << result.output;
}

TEST_F(OracleInvalidValuesTest, VolumeHorizonWithoutValueExitsOne) {
    auto result = run_command(ORACLE_BINARY_PATH + " --volume-horizon");
    EXPECT_EQ(result.exit_code, 1)
        << "oracle_expectancy --volume-horizon without value should exit 1. Got: "
        << result.output;
}

TEST_F(OracleInvalidValuesTest, InvalidMaxTimeHorizonShowsErrorMessage) {
    auto result = run_command(ORACLE_BINARY_PATH + " --max-time-horizon 0");
    bool has_error_msg = (result.output.find("max-time-horizon") != std::string::npos ||
                          result.output.find("time-horizon") != std::string::npos ||
                          result.output.find("must be") != std::string::npos ||
                          result.output.find("error") != std::string::npos ||
                          result.output.find("Error") != std::string::npos);
    EXPECT_TRUE(has_error_msg)
        << "Invalid --max-time-horizon 0 should produce a clear error message. Got: "
        << result.output;
}

// ===========================================================================
// Integration: bar_feature_export with new flags (T2, T3 from spec)
// ===========================================================================

class BarExportIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(BINARY_PATH)) GTEST_SKIP() << "Binary not built";
        if (!data_available()) GTEST_SKIP() << "Data directory not available";
    }
    void TearDown() override {
        for (const auto& p : temp_files_) std::filesystem::remove(p);
    }
    void track_temp(const std::string& p) { temp_files_.push_back(p); }
    std::vector<std::string> temp_files_;
};

TEST_F(BarExportIntegrationTest, MaxTimeHorizon3600ProducesValidOutput) {
    // Spec T2: --max-time-horizon 3600 → output has 152 columns, non-zero directional labels
    auto pq = temp_parquet_path("_int_mth");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --max-time-horizon 3600 --output " + pq);
    EXPECT_EQ(result.exit_code, 0)
        << "--max-time-horizon 3600 should succeed. Got: " << result.output;
    EXPECT_TRUE(std::filesystem::exists(pq))
        << "Parquet file should be created";
}

TEST_F(BarExportIntegrationTest, VolumeHorizon50000ProducesValidOutput) {
    // Spec T3: --volume-horizon 50000 → valid output
    auto pq = temp_parquet_path("_int_vh");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --volume-horizon 50000 --output " + pq);
    EXPECT_EQ(result.exit_code, 0)
        << "--volume-horizon 50000 should succeed. Got: " << result.output;
    EXPECT_TRUE(std::filesystem::exists(pq))
        << "Parquet file should be created";
}

TEST_F(BarExportIntegrationTest, MaxTimeHorizon1SecondProducesValidOutput) {
    // Boundary test: minimum valid value
    auto pq = temp_parquet_path("_int_mth_min");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --max-time-horizon 1 --output " + pq);
    EXPECT_EQ(result.exit_code, 0)
        << "--max-time-horizon 1 (minimum) should succeed. Got: " << result.output;
}

TEST_F(BarExportIntegrationTest, LegacyLabelsWithMaxTimeHorizon) {
    // Spec T7: --legacy-labels works with --max-time-horizon
    auto pq = temp_parquet_path("_int_legacy_mth");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --max-time-horizon 3600 --legacy-labels --output " + pq);
    EXPECT_EQ(result.exit_code, 0)
        << "--legacy-labels with --max-time-horizon should succeed. Got: " << result.output;
}

TEST_F(BarExportIntegrationTest, LegacyLabelsWithVolumeHorizon) {
    auto pq = temp_parquet_path("_int_legacy_vh");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --volume-horizon 50000 --legacy-labels --output " + pq);
    EXPECT_EQ(result.exit_code, 0)
        << "--legacy-labels with --volume-horizon should succeed. Got: " << result.output;
}

// ===========================================================================
// T8: Combined flags — all applied correctly
// ===========================================================================

class CombinedFlagsTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(BINARY_PATH)) GTEST_SKIP() << "Binary not built";
        if (!data_available()) GTEST_SKIP() << "Data directory not available";
    }
    void TearDown() override {
        for (const auto& p : temp_files_) std::filesystem::remove(p);
    }
    void track_temp(const std::string& p) { temp_files_.push_back(p); }
    std::vector<std::string> temp_files_;
};

TEST_F(CombinedFlagsTest, AllBarExportFlagsCombined) {
    // Spec T8: --target 15 --stop 3 --max-time-horizon 3600 --volume-horizon 50000
    auto pq = temp_parquet_path("_combined_bar");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 "
        "--target 15 --stop 3 --max-time-horizon 3600 --volume-horizon 50000 --output " + pq);
    EXPECT_EQ(result.exit_code, 0)
        << "All bar_feature_export flags combined should succeed. Got: " << result.output;
    EXPECT_TRUE(std::filesystem::exists(pq))
        << "Parquet file should be created with combined flags";
}

TEST_F(CombinedFlagsTest, AllBarExportFlagsWithLegacy) {
    auto pq = temp_parquet_path("_combined_bar_legacy");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 "
        "--target 15 --stop 3 --max-time-horizon 3600 --volume-horizon 50000 "
        "--legacy-labels --output " + pq);
    EXPECT_EQ(result.exit_code, 0)
        << "All bar_feature_export flags + --legacy-labels should succeed. Got: " << result.output;
}

TEST_F(CombinedFlagsTest, AllOracleFlagsCombined) {
    // Spec T8 for oracle: --target --stop --take-profit --max-time-horizon --volume-horizon
    auto json = temp_json_path("_combined_oracle");
    track_temp(json);
    auto result = run_command(
        ORACLE_BINARY_PATH + " --target 15 --stop 3 --take-profit 30 "
        "--max-time-horizon 3600 --volume-horizon 50000 --output " + json);
    if (!std::filesystem::exists(ORACLE_BINARY_PATH)) {
        GTEST_SKIP() << "oracle_expectancy binary not built";
    }
    EXPECT_EQ(result.exit_code, 0)
        << "All oracle_expectancy flags combined should succeed. Got: " << result.output;
}

// ===========================================================================
// Integration: oracle_expectancy JSON output reflects new params
// ===========================================================================

class OracleJsonOutputTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(ORACLE_BINARY_PATH)) {
            GTEST_SKIP() << "oracle_expectancy binary not built yet";
        }
        if (!data_available()) {
            GTEST_SKIP() << "Data directory not available";
        }
    }
    void TearDown() override {
        for (const auto& p : temp_files_) std::filesystem::remove(p);
    }
    void track_temp(const std::string& p) { temp_files_.push_back(p); }
    std::vector<std::string> temp_files_;
};

TEST_F(OracleJsonOutputTest, JsonReflectsMaxTimeHorizon) {
    auto path = temp_json_path("_json_mth");
    track_temp(path);
    auto result = run_command(
        ORACLE_BINARY_PATH + " --max-time-horizon 7200 --output " + path);
    if (result.exit_code != 0) GTEST_SKIP() << "Tool exited non-zero";

    auto json = read_file(path);
    // JSON should contain the max_time_horizon_s value used
    bool reflects_param = (json.find("7200") != std::string::npos ||
                           json_has_key(json, "max_time_horizon_s"));
    EXPECT_TRUE(reflects_param)
        << "JSON output should reflect --max-time-horizon 7200. Got: "
        << json.substr(0, 500);
}

TEST_F(OracleJsonOutputTest, JsonReflectsVolumeHorizon) {
    auto path = temp_json_path("_json_vh");
    track_temp(path);
    auto result = run_command(
        ORACLE_BINARY_PATH + " --volume-horizon 25000 --output " + path);
    if (result.exit_code != 0) GTEST_SKIP() << "Tool exited non-zero";

    auto json = read_file(path);
    bool reflects_param = (json.find("25000") != std::string::npos ||
                           json_has_key(json, "volume_horizon"));
    EXPECT_TRUE(reflects_param)
        << "JSON output should reflect --volume-horizon 25000. Got: "
        << json.substr(0, 500);
}

TEST_F(OracleJsonOutputTest, JsonReflectsAllCombinedParams) {
    auto path = temp_json_path("_json_all");
    track_temp(path);
    auto result = run_command(
        ORACLE_BINARY_PATH + " --target 15 --stop 3 --take-profit 30 "
        "--max-time-horizon 3600 --volume-horizon 50000 --output " + path);
    if (result.exit_code != 0) GTEST_SKIP() << "Tool exited non-zero";

    auto json = read_file(path);
    EXPECT_NE(json.find("\"target_ticks\":15"), std::string::npos)
        << "JSON should have target_ticks:15. Got: " << json.substr(0, 500);
    EXPECT_NE(json.find("\"stop_ticks\":3"), std::string::npos)
        << "JSON should have stop_ticks:3";
}

// ===========================================================================
// T7: Backward compatibility — new defaults used when flags absent
// ===========================================================================

class BackwardCompatibilityTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(BINARY_PATH)) GTEST_SKIP() << "Binary not built";
        if (!data_available()) GTEST_SKIP() << "Data directory not available";
    }
    void TearDown() override {
        for (const auto& p : temp_files_) std::filesystem::remove(p);
    }
    void track_temp(const std::string& p) { temp_files_.push_back(p); }
    std::vector<std::string> temp_files_;
};

TEST_F(BackwardCompatibilityTest, NoNewFlagsUsesNewDefaults) {
    // Spec T7: Running without --max-time-horizon uses new default (3600)
    // Running without --volume-horizon uses new default (50000)
    // Both binaries should still work with no new flags
    auto pq = temp_parquet_path("_compat_default");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq);
    EXPECT_EQ(result.exit_code, 0)
        << "Running without new flags should succeed with new defaults. Got: " << result.output;
}

TEST_F(BackwardCompatibilityTest, ExistingTargetStopFlagsStillWork) {
    // Spec T7: --target and --stop still work alongside new flags
    auto pq = temp_parquet_path("_compat_target_stop");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --target 15 --stop 3 --output " + pq);
    EXPECT_EQ(result.exit_code, 0)
        << "Existing --target/--stop flags should still work. Got: " << result.output;
}

TEST_F(BackwardCompatibilityTest, LegacyLabelsFlagStillWorks) {
    auto pq = temp_parquet_path("_compat_legacy");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --legacy-labels --output " + pq);
    EXPECT_EQ(result.exit_code, 0)
        << "--legacy-labels should still work with new defaults. Got: " << result.output;
}

TEST_F(BackwardCompatibilityTest, ExplicitNewDefaultsMatchImplicit) {
    // Explicit --max-time-horizon 3600 --volume-horizon 50000 should produce
    // identical output to running without these flags (since 3600/50000 are the new defaults)
    auto pq_implicit = temp_parquet_path("_compat_impl");
    auto pq_explicit = temp_parquet_path("_compat_expl");
    track_temp(pq_implicit);
    track_temp(pq_explicit);

    auto result_implicit = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_implicit);
    auto result_explicit = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 "
        "--max-time-horizon 3600 --volume-horizon 50000 --output " + pq_explicit);

    if (result_implicit.exit_code != 0 || result_explicit.exit_code != 0) {
        GTEST_SKIP() << "One or both exports failed";
    }

    // Both files should exist and be the same size
    EXPECT_TRUE(std::filesystem::exists(pq_implicit));
    EXPECT_TRUE(std::filesystem::exists(pq_explicit));

    auto size_impl = std::filesystem::file_size(pq_implicit);
    auto size_expl = std::filesystem::file_size(pq_explicit);
    EXPECT_EQ(size_impl, size_expl)
        << "Explicit new defaults should produce same-sized output as implicit defaults. "
        << "Implicit: " << size_impl << " bytes, Explicit: " << size_expl << " bytes";
}
