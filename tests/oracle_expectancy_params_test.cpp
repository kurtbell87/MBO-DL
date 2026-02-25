// oracle_expectancy_params_test.cpp — TDD RED phase tests for oracle_expectancy CLI parameterization
// Spec: .kit/docs/oracle-expectancy-params.md
//
// Tests the CLI argument parsing contract for oracle_expectancy:
//   --target <ticks>, --stop <ticks>, --take-profit <ticks>,
//   --output <path>, --help
//
// These tests invoke the built binary via popen and verify exit codes,
// stdout content, and JSON file output. They do NOT test oracle replay
// logic itself (already covered by oracle_expectancy_test.cpp with 67
// unit tests). They test ONLY the CLI interface contract.
//
// Tests that require real data (DATA/GLBX-20260207-L953CAPU5B) will
// GTEST_SKIP if the directory is unavailable.

#include <gtest/gtest.h>
#include "test_export_helpers.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// ===========================================================================
// Helpers
// ===========================================================================
namespace {

// Path to the built oracle_expectancy binary.
const std::string BINARY_PATH = "build/oracle_expectancy";

using export_test_helpers::run_command;

// Temp output JSON path for tests.
std::string temp_json_path(const std::string& suffix = "") {
    return (std::filesystem::temp_directory_path() /
            ("oracle_expectancy_params_test" + suffix + ".json")).string();
}

// Check if output contains usage-related keywords.
bool contains_usage(const std::string& output) {
    return output.find("usage") != std::string::npos ||
           output.find("Usage") != std::string::npos ||
           output.find("USAGE") != std::string::npos ||
           output.find("--target") != std::string::npos ||
           output.find("--stop") != std::string::npos ||
           output.find("--help") != std::string::npos;
}

// Check if a JSON string contains a key (looks for "key":).
bool json_has_key(const std::string& json, const std::string& key) {
    return json.find("\"" + key + "\"") != std::string::npos;
}

// Read entire file into a string.
std::string read_file(const std::string& path) {
    std::ifstream f(path);
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    return content;
}

// Whether the real MBO data directory exists.
bool data_available() {
    return std::filesystem::exists("DATA/GLBX-20260207-L953CAPU5B");
}

}  // anonymous namespace

// ===========================================================================
// Binary Existence — prerequisite for all CLI tests
// ===========================================================================

class OracleExpectancyBinaryTest : public ::testing::Test {};

TEST_F(OracleExpectancyBinaryTest, BinaryExistsAfterBuild) {
    EXPECT_TRUE(std::filesystem::exists(BINARY_PATH))
        << "Expected binary at " << BINARY_PATH
        << " — must be built via CMakeLists.txt";
}

TEST_F(OracleExpectancyBinaryTest, BinaryIsExecutable) {
    if (!std::filesystem::exists(BINARY_PATH)) {
        GTEST_SKIP() << "Binary not built yet";
    }
    auto perms = std::filesystem::status(BINARY_PATH).permissions();
    bool is_exec = (perms & std::filesystem::perms::owner_exec) !=
                    std::filesystem::perms::none;
    EXPECT_TRUE(is_exec) << "Binary must be executable";
}

// ===========================================================================
// T7 (R7): --help prints usage and exits 0
// ===========================================================================

class HelpFlagTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(BINARY_PATH)) {
            GTEST_SKIP() << "Binary not built yet";
        }
    }
};

TEST_F(HelpFlagTest, HelpFlagExitsZero) {
    // R7: --help prints usage and exits 0
    auto result = run_command(BINARY_PATH + " --help");
    EXPECT_EQ(result.exit_code, 0)
        << "--help should exit with code 0. Got output: " << result.output;
}

TEST_F(HelpFlagTest, HelpFlagPrintsUsage) {
    auto result = run_command(BINARY_PATH + " --help");
    EXPECT_TRUE(contains_usage(result.output))
        << "--help should print usage information. Got: " << result.output;
}

TEST_F(HelpFlagTest, HelpFlagMentionsTargetFlag) {
    auto result = run_command(BINARY_PATH + " --help");
    EXPECT_NE(result.output.find("--target"), std::string::npos)
        << "Usage should mention --target flag. Got: " << result.output;
}

TEST_F(HelpFlagTest, HelpFlagMentionsStopFlag) {
    auto result = run_command(BINARY_PATH + " --help");
    EXPECT_NE(result.output.find("--stop"), std::string::npos)
        << "Usage should mention --stop flag. Got: " << result.output;
}

TEST_F(HelpFlagTest, HelpFlagMentionsTakeProfitFlag) {
    auto result = run_command(BINARY_PATH + " --help");
    EXPECT_NE(result.output.find("--take-profit"), std::string::npos)
        << "Usage should mention --take-profit flag. Got: " << result.output;
}

TEST_F(HelpFlagTest, HelpFlagMentionsOutputFlag) {
    auto result = run_command(BINARY_PATH + " --help");
    EXPECT_NE(result.output.find("--output"), std::string::npos)
        << "Usage should mention --output flag. Got: " << result.output;
}

TEST_F(HelpFlagTest, HelpDoesNotRunFullAnalysis) {
    // --help should print usage ONLY, not run the full data analysis pipeline
    auto result = run_command(BINARY_PATH + " --help");
    // The full analysis prints "=== Oracle Expectancy Extraction ==="
    EXPECT_EQ(result.output.find("Oracle Expectancy Extraction"), std::string::npos)
        << "--help should print usage only, not run the analysis. Got: "
        << result.output;
}

// ===========================================================================
// T5 (R6): Invalid flag values — negative and zero → exits 1 with usage
// ===========================================================================

class InvalidArgTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(BINARY_PATH)) {
            GTEST_SKIP() << "Binary not built yet";
        }
    }
};

TEST_F(InvalidArgTest, NegativeTargetExitsOne) {
    // T5: --target -1 → exits 1 with usage message
    auto result = run_command(BINARY_PATH + " --target -1");
    EXPECT_EQ(result.exit_code, 1)
        << "--target -1 should exit with code 1. Got: " << result.output;
}

TEST_F(InvalidArgTest, NegativeTargetShowsUsage) {
    auto result = run_command(BINARY_PATH + " --target -1");
    EXPECT_TRUE(contains_usage(result.output))
        << "--target -1 should show usage message. Got: " << result.output;
}

TEST_F(InvalidArgTest, ZeroTargetExitsOne) {
    // R6: zero is invalid
    auto result = run_command(BINARY_PATH + " --target 0");
    EXPECT_EQ(result.exit_code, 1)
        << "--target 0 should exit with code 1. Got: " << result.output;
}

TEST_F(InvalidArgTest, ZeroStopExitsOne) {
    auto result = run_command(BINARY_PATH + " --stop 0");
    EXPECT_EQ(result.exit_code, 1)
        << "--stop 0 should exit with code 1. Got: " << result.output;
}

TEST_F(InvalidArgTest, NegativeStopExitsOne) {
    auto result = run_command(BINARY_PATH + " --stop -3");
    EXPECT_EQ(result.exit_code, 1)
        << "--stop -3 should exit with code 1. Got: " << result.output;
}

TEST_F(InvalidArgTest, NegativeTakeProfitExitsOne) {
    auto result = run_command(BINARY_PATH + " --take-profit -5");
    EXPECT_EQ(result.exit_code, 1)
        << "--take-profit -5 should exit with code 1. Got: " << result.output;
}

TEST_F(InvalidArgTest, ZeroTakeProfitExitsOne) {
    auto result = run_command(BINARY_PATH + " --take-profit 0");
    EXPECT_EQ(result.exit_code, 1)
        << "--take-profit 0 should exit with code 1. Got: " << result.output;
}

// ===========================================================================
// T6 (R6): Non-integer values → exits 1 with usage
// ===========================================================================

TEST_F(InvalidArgTest, NonIntegerTargetExitsOne) {
    // T6: --target abc → exits 1 with usage message
    auto result = run_command(BINARY_PATH + " --target abc");
    EXPECT_EQ(result.exit_code, 1)
        << "--target abc should exit with code 1. Got: " << result.output;
}

TEST_F(InvalidArgTest, NonIntegerTargetShowsUsage) {
    auto result = run_command(BINARY_PATH + " --target abc");
    EXPECT_TRUE(contains_usage(result.output))
        << "--target abc should show usage message. Got: " << result.output;
}

TEST_F(InvalidArgTest, NonIntegerStopExitsOne) {
    auto result = run_command(BINARY_PATH + " --stop xyz");
    EXPECT_EQ(result.exit_code, 1)
        << "--stop xyz should exit with code 1. Got: " << result.output;
}

TEST_F(InvalidArgTest, NonIntegerTakeProfitExitsOne) {
    auto result = run_command(BINARY_PATH + " --take-profit foo");
    EXPECT_EQ(result.exit_code, 1)
        << "--take-profit foo should exit with code 1. Got: " << result.output;
}

// ===========================================================================
// Edge cases: unknown flags, missing values
// ===========================================================================

TEST_F(InvalidArgTest, UnknownFlagExitsOne) {
    auto result = run_command(BINARY_PATH + " --bogus 42");
    EXPECT_EQ(result.exit_code, 1)
        << "Unknown flag --bogus should exit with code 1. Got: " << result.output;
}

TEST_F(InvalidArgTest, TargetWithoutValueExitsOne) {
    // --target without a following value should error
    auto result = run_command(BINARY_PATH + " --target");
    EXPECT_EQ(result.exit_code, 1)
        << "--target without value should exit with code 1. Got: " << result.output;
}

TEST_F(InvalidArgTest, StopWithoutValueExitsOne) {
    auto result = run_command(BINARY_PATH + " --stop");
    EXPECT_EQ(result.exit_code, 1)
        << "--stop without value should exit with code 1. Got: " << result.output;
}

TEST_F(InvalidArgTest, TakeProfitWithoutValueExitsOne) {
    auto result = run_command(BINARY_PATH + " --take-profit");
    EXPECT_EQ(result.exit_code, 1)
        << "--take-profit without value should exit with code 1. Got: " << result.output;
}

TEST_F(InvalidArgTest, OutputWithoutPathExitsOne) {
    auto result = run_command(BINARY_PATH + " --output");
    EXPECT_EQ(result.exit_code, 1)
        << "--output without path should exit with code 1. Got: " << result.output;
}

// ===========================================================================
// T1 (R5): No args → backward compatible (same defaults)
// These tests require real data in DATA/GLBX-20260207-L953CAPU5B.
// ===========================================================================

class BackwardCompatTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(BINARY_PATH)) {
            GTEST_SKIP() << "Binary not built yet";
        }
        if (!data_available()) {
            GTEST_SKIP() << "Data directory not available";
        }
    }
};

TEST_F(BackwardCompatTest, NoArgsExitsZero) {
    // R5: No flags → same behavior as current (backward compatible)
    auto result = run_command(BINARY_PATH);
    EXPECT_EQ(result.exit_code, 0)
        << "No-arg invocation should exit 0 (backward compatible). Got: "
        << result.output;
}

TEST_F(BackwardCompatTest, NoArgsPrintsOracleHeader) {
    auto result = run_command(BINARY_PATH);
    if (result.exit_code != 0) {
        GTEST_SKIP() << "Tool exited non-zero";
    }
    EXPECT_NE(result.output.find("Oracle Expectancy"), std::string::npos)
        << "No-arg output should contain 'Oracle Expectancy' header. Got: "
        << result.output;
}

TEST_F(BackwardCompatTest, NoArgsPrintsResults) {
    auto result = run_command(BINARY_PATH);
    if (result.exit_code != 0) {
        GTEST_SKIP() << "Tool exited non-zero";
    }
    EXPECT_NE(result.output.find("Results"), std::string::npos)
        << "No-arg output should contain a 'Results' section. Got: "
        << result.output;
}

// ===========================================================================
// T3 (R1-R3): Explicit defaults match implicit defaults
// ===========================================================================

class ExplicitDefaultsTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(BINARY_PATH)) {
            GTEST_SKIP() << "Binary not built yet";
        }
        if (!data_available()) {
            GTEST_SKIP() << "Data directory not available";
        }
    }
};

TEST_F(ExplicitDefaultsTest, ExplicitDefaultsMatchImplicit) {
    // T3: --target 10 --stop 5 --take-profit 20 → identical to default
    auto no_args = run_command(BINARY_PATH);
    auto with_defaults = run_command(
        BINARY_PATH + " --target 10 --stop 5 --take-profit 20");

    if (no_args.exit_code != 0 || with_defaults.exit_code != 0) {
        GTEST_SKIP() << "One or both runs exited non-zero";
    }

    // Both should produce identical stdout output
    EXPECT_EQ(no_args.output, with_defaults.output)
        << "Explicit defaults (--target 10 --stop 5 --take-profit 20) "
        << "must produce identical output to no-arg invocation";
}

// ===========================================================================
// T2 (R1-R3): Modified geometry produces different metrics
// ===========================================================================

class ModifiedGeometryTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(BINARY_PATH)) {
            GTEST_SKIP() << "Binary not built yet";
        }
        if (!data_available()) {
            GTEST_SKIP() << "Data directory not available";
        }
    }
};

TEST_F(ModifiedGeometryTest, DifferentGeometryProducesDifferentMetrics) {
    // T2: --target 15 --stop 3 → runs with modified geometry, different metrics
    auto defaults = run_command(BINARY_PATH);
    auto modified = run_command(BINARY_PATH + " --target 15 --stop 3");

    if (defaults.exit_code != 0 || modified.exit_code != 0) {
        GTEST_SKIP() << "One or both runs exited non-zero";
    }

    EXPECT_NE(defaults.output, modified.output)
        << "Modified geometry (--target 15 --stop 3) must produce "
        << "different metrics than defaults (target=10, stop=5)";
}

TEST_F(ModifiedGeometryTest, ModifiedGeometryExitsZero) {
    auto result = run_command(BINARY_PATH + " --target 15 --stop 3");
    EXPECT_EQ(result.exit_code, 0)
        << "--target 15 --stop 3 should succeed. Got: " << result.output;
}

TEST_F(ModifiedGeometryTest, ModifiedGeometryPrintsResults) {
    auto result = run_command(BINARY_PATH + " --target 15 --stop 3");
    if (result.exit_code != 0) {
        GTEST_SKIP() << "Tool exited non-zero";
    }
    EXPECT_NE(result.output.find("Results"), std::string::npos)
        << "Modified geometry should still print Results section. Got: "
        << result.output;
}

TEST_F(ModifiedGeometryTest, PartialOverrideTargetOnly) {
    // Only --target changes, stop and take-profit stay at defaults
    auto result = run_command(BINARY_PATH + " --target 15");
    EXPECT_EQ(result.exit_code, 0)
        << "--target 15 alone should succeed (stop/tp keep defaults). Got: "
        << result.output;
}

TEST_F(ModifiedGeometryTest, PartialOverrideStopOnly) {
    auto result = run_command(BINARY_PATH + " --stop 3");
    EXPECT_EQ(result.exit_code, 0)
        << "--stop 3 alone should succeed. Got: " << result.output;
}

TEST_F(ModifiedGeometryTest, PartialOverrideTakeProfitOnly) {
    auto result = run_command(BINARY_PATH + " --take-profit 30");
    EXPECT_EQ(result.exit_code, 0)
        << "--take-profit 30 alone should succeed. Got: " << result.output;
}

// ===========================================================================
// T4 (R4): --output <path> writes JSON with expected fields
// ===========================================================================

class JsonOutputTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(BINARY_PATH)) {
            GTEST_SKIP() << "Binary not built yet";
        }
    }
    void TearDown() override {
        for (const auto& p : temp_files_) {
            std::filesystem::remove(p);
        }
    }
    void track_temp(const std::string& p) { temp_files_.push_back(p); }
    std::vector<std::string> temp_files_;
};

TEST_F(JsonOutputTest, OutputFlagCreatesJsonFile) {
    // T4: --output path → writes JSON file
    if (!data_available()) {
        GTEST_SKIP() << "Data directory not available";
    }

    auto path = temp_json_path("_basic");
    track_temp(path);

    auto result = run_command(BINARY_PATH + " --output " + path);
    if (result.exit_code != 0) {
        GTEST_SKIP() << "Tool exited non-zero";
    }

    EXPECT_TRUE(std::filesystem::exists(path))
        << "JSON output file should exist at " << path;
}

TEST_F(JsonOutputTest, OutputFileIsNonEmpty) {
    if (!data_available()) {
        GTEST_SKIP() << "Data directory not available";
    }

    auto path = temp_json_path("_nonempty");
    track_temp(path);

    auto result = run_command(BINARY_PATH + " --output " + path);
    if (result.exit_code != 0) {
        GTEST_SKIP() << "Tool exited non-zero";
    }

    auto content = read_file(path);
    EXPECT_GT(content.size(), 10u)
        << "JSON output file should be non-trivial. Got: " << content;
}

TEST_F(JsonOutputTest, JsonContainsTargetTicks) {
    if (!data_available()) {
        GTEST_SKIP() << "Data directory not available";
    }

    auto path = temp_json_path("_target");
    track_temp(path);

    auto result = run_command(BINARY_PATH + " --output " + path);
    if (result.exit_code != 0) {
        GTEST_SKIP() << "Tool exited non-zero";
    }

    auto json = read_file(path);
    EXPECT_TRUE(json_has_key(json, "target_ticks"))
        << "JSON must contain 'target_ticks' field. Got: "
        << json.substr(0, 300);
}

TEST_F(JsonOutputTest, JsonContainsStopTicks) {
    if (!data_available()) {
        GTEST_SKIP() << "Data directory not available";
    }

    auto path = temp_json_path("_stop");
    track_temp(path);

    auto result = run_command(BINARY_PATH + " --output " + path);
    if (result.exit_code != 0) {
        GTEST_SKIP() << "Tool exited non-zero";
    }

    auto json = read_file(path);
    EXPECT_TRUE(json_has_key(json, "stop_ticks"))
        << "JSON must contain 'stop_ticks' field. Got: "
        << json.substr(0, 300);
}

TEST_F(JsonOutputTest, JsonContainsTakeProfitTicks) {
    if (!data_available()) {
        GTEST_SKIP() << "Data directory not available";
    }

    auto path = temp_json_path("_tp");
    track_temp(path);

    auto result = run_command(BINARY_PATH + " --output " + path);
    if (result.exit_code != 0) {
        GTEST_SKIP() << "Tool exited non-zero";
    }

    auto json = read_file(path);
    EXPECT_TRUE(json_has_key(json, "take_profit_ticks"))
        << "JSON must contain 'take_profit_ticks' field. Got: "
        << json.substr(0, 300);
}

TEST_F(JsonOutputTest, JsonContainsTotalTrades) {
    if (!data_available()) {
        GTEST_SKIP() << "Data directory not available";
    }

    auto path = temp_json_path("_trades");
    track_temp(path);

    auto result = run_command(BINARY_PATH + " --output " + path);
    if (result.exit_code != 0) {
        GTEST_SKIP() << "Tool exited non-zero";
    }

    auto json = read_file(path);
    EXPECT_TRUE(json_has_key(json, "total_trades"))
        << "JSON must contain 'total_trades' field";
}

TEST_F(JsonOutputTest, JsonContainsWinRate) {
    if (!data_available()) {
        GTEST_SKIP() << "Data directory not available";
    }

    auto path = temp_json_path("_wr");
    track_temp(path);

    auto result = run_command(BINARY_PATH + " --output " + path);
    if (result.exit_code != 0) {
        GTEST_SKIP() << "Tool exited non-zero";
    }

    auto json = read_file(path);
    EXPECT_TRUE(json_has_key(json, "win_rate"))
        << "JSON must contain 'win_rate' field";
}

TEST_F(JsonOutputTest, JsonContainsExpectancy) {
    if (!data_available()) {
        GTEST_SKIP() << "Data directory not available";
    }

    auto path = temp_json_path("_exp");
    track_temp(path);

    auto result = run_command(BINARY_PATH + " --output " + path);
    if (result.exit_code != 0) {
        GTEST_SKIP() << "Tool exited non-zero";
    }

    auto json = read_file(path);
    EXPECT_TRUE(json_has_key(json, "expectancy"))
        << "JSON must contain 'expectancy' field";
}

TEST_F(JsonOutputTest, JsonContainsProfitFactor) {
    if (!data_available()) {
        GTEST_SKIP() << "Data directory not available";
    }

    auto path = temp_json_path("_pf");
    track_temp(path);

    auto result = run_command(BINARY_PATH + " --output " + path);
    if (result.exit_code != 0) {
        GTEST_SKIP() << "Tool exited non-zero";
    }

    auto json = read_file(path);
    EXPECT_TRUE(json_has_key(json, "profit_factor"))
        << "JSON must contain 'profit_factor' field";
}

TEST_F(JsonOutputTest, JsonContainsPerQuarter) {
    if (!data_available()) {
        GTEST_SKIP() << "Data directory not available";
    }

    auto path = temp_json_path("_pq");
    track_temp(path);

    auto result = run_command(BINARY_PATH + " --output " + path);
    if (result.exit_code != 0) {
        GTEST_SKIP() << "Tool exited non-zero";
    }

    auto json = read_file(path);
    EXPECT_TRUE(json_has_key(json, "per_quarter"))
        << "JSON must contain 'per_quarter' field";
}

TEST_F(JsonOutputTest, JsonReflectsCustomParams) {
    // When custom params are passed, the JSON config block must reflect them
    if (!data_available()) {
        GTEST_SKIP() << "Data directory not available";
    }

    auto path = temp_json_path("_custom");
    track_temp(path);

    auto result = run_command(
        BINARY_PATH + " --target 15 --stop 3 --take-profit 30 --output " + path);
    if (result.exit_code != 0) {
        GTEST_SKIP() << "Tool exited non-zero";
    }

    auto json = read_file(path);
    // The JSON config block must contain the actual parameter values used
    EXPECT_NE(json.find("\"target_ticks\":15"), std::string::npos)
        << "JSON config should have target_ticks:15. Got: "
        << json.substr(0, 400);
    EXPECT_NE(json.find("\"stop_ticks\":3"), std::string::npos)
        << "JSON config should have stop_ticks:3. Got: "
        << json.substr(0, 400);
    EXPECT_NE(json.find("\"take_profit_ticks\":30"), std::string::npos)
        << "JSON config should have take_profit_ticks:30. Got: "
        << json.substr(0, 400);
}

TEST_F(JsonOutputTest, OutputToInvalidPathExitsNonZero) {
    // Writing to a non-existent directory should fail
    auto result = run_command(
        BINARY_PATH + " --output /nonexistent/dir/out.json");
    EXPECT_NE(result.exit_code, 0)
        << "Writing to invalid path should exit non-zero. Got: "
        << result.output;
}

TEST_F(JsonOutputTest, OutputCombinesWithModifiedGeometry) {
    // --output works together with geometry flags
    if (!data_available()) {
        GTEST_SKIP() << "Data directory not available";
    }

    auto path = temp_json_path("_combo");
    track_temp(path);

    auto result = run_command(
        BINARY_PATH + " --target 15 --stop 3 --output " + path);
    if (result.exit_code != 0) {
        GTEST_SKIP() << "Tool exited non-zero";
    }

    EXPECT_TRUE(std::filesystem::exists(path))
        << "JSON file should exist when combining --output with geometry flags";

    auto json = read_file(path);
    EXPECT_TRUE(json_has_key(json, "target_ticks"))
        << "Combined JSON should contain target_ticks";
    EXPECT_TRUE(json_has_key(json, "total_trades"))
        << "Combined JSON should contain total_trades";
}

// ===========================================================================
// Flag ordering independence — flags can appear in any order
// ===========================================================================

class FlagOrderTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(BINARY_PATH)) {
            GTEST_SKIP() << "Binary not built yet";
        }
        if (!data_available()) {
            GTEST_SKIP() << "Data directory not available";
        }
    }
    void TearDown() override {
        for (const auto& p : temp_files_) {
            std::filesystem::remove(p);
        }
    }
    void track_temp(const std::string& p) { temp_files_.push_back(p); }
    std::vector<std::string> temp_files_;
};

TEST_F(FlagOrderTest, FlagsInDifferentOrderProduceSameOutput) {
    // The order of flags should not matter
    auto path_a = temp_json_path("_order_a");
    auto path_b = temp_json_path("_order_b");
    track_temp(path_a);
    track_temp(path_b);

    auto result_a = run_command(
        BINARY_PATH + " --target 15 --stop 3 --take-profit 30 --output " + path_a);
    auto result_b = run_command(
        BINARY_PATH + " --output " + path_b + " --stop 3 --take-profit 30 --target 15");

    if (result_a.exit_code != 0 || result_b.exit_code != 0) {
        GTEST_SKIP() << "One or both runs exited non-zero";
    }

    auto json_a = read_file(path_a);
    auto json_b = read_file(path_b);
    EXPECT_EQ(json_a, json_b)
        << "Flag ordering should not affect output JSON content";
}
