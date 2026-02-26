// bar_feature_export_geometry_test.cpp — TDD RED phase tests for --target/--stop CLI flags
// Spec: .kit/docs/bar-feature-export-geometry.md
//
// Tests that bar_feature_export:
//   (1) recognizes --target and --stop CLI flags
//   (2) uses default values (10, 5) when flags are absent → identical output to current binary
//   (3) passes custom target/stop values to triple barrier computation
//   (4) rejects invalid values (<=0, target<=stop) with clear error and non-zero exit
//   (5) works with --legacy-labels using custom target/stop
//   (6) shows --target and --stop in --help / usage output
//
// Unit tests verify CLI contract. Integration tests require the binary + data.
// Integration tests are guarded by GTEST_SKIP when prerequisites are missing.

#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>
#include <parquet/metadata.h>

#include "features/bar_features.hpp"
#include "features/raw_representations.hpp"
#include "test_export_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

// ===========================================================================
// Helpers
// ===========================================================================
namespace {

using export_test_helpers::BINARY_PATH;
using export_test_helpers::run_command;

const std::string DATA_DIR = "DATA/GLBX-20260207-L953CAPU5B";

std::string temp_parquet_path(const std::string& suffix = "") {
    return (std::filesystem::temp_directory_path() /
            ("bar_feature_export_geometry_test" + suffix + ".parquet")).string();
}

// Read a Parquet file into an Arrow Table.
std::shared_ptr<arrow::Table> read_parquet_table(const std::string& path) {
    auto open_result = arrow::io::ReadableFile::Open(path);
    if (!open_result.ok()) return nullptr;

    auto file_reader_result = parquet::arrow::OpenFile(
        open_result.ValueOrDie(), arrow::default_memory_pool());
    if (!file_reader_result.ok()) return nullptr;
    auto reader = file_reader_result.MoveValueUnsafe();

    std::shared_ptr<arrow::Table> table;
    auto status = reader->ReadTable(&table);
    if (!status.ok()) return nullptr;

    return table;
}

// Extract double values from an Arrow column into a vector.
std::vector<double> extract_doubles(const std::shared_ptr<arrow::ChunkedArray>& col) {
    std::vector<double> values;
    for (int chunk = 0; chunk < col->num_chunks(); ++chunk) {
        auto arr = std::dynamic_pointer_cast<arrow::DoubleArray>(col->chunk(chunk));
        if (!arr) continue;
        for (int64_t i = 0; i < arr->length(); ++i) {
            values.push_back(arr->Value(i));
        }
    }
    return values;
}

// Count occurrences of each label value in a double vector.
std::map<int, int> count_labels(const std::vector<double>& labels) {
    std::map<int, int> counts;
    for (double v : labels) {
        counts[static_cast<int>(v)]++;
    }
    return counts;
}

bool data_available() {
    return std::filesystem::exists(DATA_DIR) &&
           std::filesystem::is_directory(DATA_DIR);
}

}  // anonymous namespace

// ===========================================================================
// T3: --help shows --target and --stop flags with defaults
// ===========================================================================

class GeometryHelpTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(BINARY_PATH)) {
            GTEST_SKIP() << "Binary not built yet";
        }
    }
};

TEST_F(GeometryHelpTest, NoArgsUsageMentionsTargetFlag) {
    // Spec T3: usage output must include --target
    auto result = run_command(BINARY_PATH);
    EXPECT_NE(result.output.find("--target"), std::string::npos)
        << "Usage output must mention --target flag. Got: " << result.output;
}

TEST_F(GeometryHelpTest, NoArgsUsageMentionsStopFlag) {
    // Spec T3: usage output must include --stop
    auto result = run_command(BINARY_PATH);
    EXPECT_NE(result.output.find("--stop"), std::string::npos)
        << "Usage output must mention --stop flag. Got: " << result.output;
}

TEST_F(GeometryHelpTest, UsageShowsTargetDefault10) {
    // Spec: --target default is 10. Usage should indicate this.
    auto result = run_command(BINARY_PATH);
    // Look for "default: 10" or "default 10" or "(10)" near --target context
    bool mentions_default = (result.output.find("10") != std::string::npos);
    EXPECT_TRUE(mentions_default)
        << "Usage should mention the default value 10 for --target. Got: "
        << result.output;
}

TEST_F(GeometryHelpTest, UsageShowsStopDefault5) {
    // Spec: --stop default is 5. Usage should show this in the --stop description line.
    auto result = run_command(BINARY_PATH);
    // Must have "--stop" in the usage (checked by NoArgsUsageMentionsStopFlag),
    // AND the stop description line should mention the default "5".
    // Look for a line containing both "--stop" and "5" (the default).
    auto pos = result.output.find("--stop");
    bool stop_line_has_default = false;
    if (pos != std::string::npos) {
        // Check the rest of the line from --stop onward for "5"
        auto line_end = result.output.find('\n', pos);
        auto stop_line = result.output.substr(pos, line_end - pos);
        stop_line_has_default = (stop_line.find("5") != std::string::npos);
    }
    EXPECT_TRUE(stop_line_has_default)
        << "Usage should mention --stop with default value 5 on the same line. Got: "
        << result.output;
}

TEST_F(GeometryHelpTest, UsageDescribesTargetAsTicks) {
    // Spec: --target is "Target (take-profit) barrier in ticks"
    auto result = run_command(BINARY_PATH);
    // The --target line in the usage should mention "tick" (not the bar-type "tick")
    auto pos = result.output.find("--target");
    bool target_line_mentions_ticks = false;
    if (pos != std::string::npos) {
        auto line_end = result.output.find('\n', pos);
        auto target_line = result.output.substr(pos, line_end - pos);
        target_line_mentions_ticks = (target_line.find("tick") != std::string::npos ||
                                      target_line.find("Tick") != std::string::npos);
    }
    EXPECT_TRUE(target_line_mentions_ticks)
        << "Usage --target line should describe the value in ticks. Got: "
        << result.output;
}

// ===========================================================================
// CLI Flag Acceptance — --target and --stop are recognized (no data needed)
// ===========================================================================

class GeometryCLIAcceptanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(BINARY_PATH)) {
            GTEST_SKIP() << "Binary not built yet";
        }
    }
    void TearDown() override {
        for (const auto& p : temp_files_) std::filesystem::remove(p);
    }
    void track_temp(const std::string& p) { temp_files_.push_back(p); }
    std::vector<std::string> temp_files_;
};

TEST_F(GeometryCLIAcceptanceTest, TargetFlagRecognized) {
    // --target should not produce an "unknown argument" error
    auto pq = temp_parquet_path("_cli_target");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --target 15 --output " + pq);

    bool arg_error = (result.output.find("Unknown") != std::string::npos ||
                      result.output.find("unknown") != std::string::npos ||
                      result.output.find("unrecognized") != std::string::npos ||
                      result.output.find("Unrecognized") != std::string::npos ||
                      result.output.find("invalid option") != std::string::npos);
    EXPECT_FALSE(arg_error)
        << "--target should be a recognized CLI flag. Got: " << result.output;
}

TEST_F(GeometryCLIAcceptanceTest, StopFlagRecognized) {
    auto pq = temp_parquet_path("_cli_stop");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --stop 3 --output " + pq);

    bool arg_error = (result.output.find("Unknown") != std::string::npos ||
                      result.output.find("unknown") != std::string::npos ||
                      result.output.find("unrecognized") != std::string::npos ||
                      result.output.find("Unrecognized") != std::string::npos ||
                      result.output.find("invalid option") != std::string::npos);
    EXPECT_FALSE(arg_error)
        << "--stop should be a recognized CLI flag. Got: " << result.output;
}

TEST_F(GeometryCLIAcceptanceTest, BothFlagsRecognizedTogether) {
    auto pq = temp_parquet_path("_cli_both");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --target 15 --stop 3 --output " + pq);

    bool arg_error = (result.output.find("Unknown") != std::string::npos ||
                      result.output.find("unknown") != std::string::npos ||
                      result.output.find("unrecognized") != std::string::npos ||
                      result.output.find("Unrecognized") != std::string::npos);
    EXPECT_FALSE(arg_error)
        << "--target and --stop together should be recognized. Got: " << result.output;
}

TEST_F(GeometryCLIAcceptanceTest, FlagOrderDoesNotMatter) {
    // --stop before --target should also be accepted
    auto pq = temp_parquet_path("_cli_order");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --stop 3 --bar-type time --target 15 --bar-param 5 --output " + pq);

    bool arg_error = (result.output.find("Unknown") != std::string::npos ||
                      result.output.find("unknown") != std::string::npos ||
                      result.output.find("unrecognized") != std::string::npos);
    EXPECT_FALSE(arg_error)
        << "Flag ordering should not matter. Got: " << result.output;
}

TEST_F(GeometryCLIAcceptanceTest, TargetWithoutValueFails) {
    // --target without a following integer should error
    auto pq = temp_parquet_path("_cli_target_noval");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --target --output " + pq);
    EXPECT_NE(result.exit_code, 0)
        << "--target without value should exit non-zero. Got: " << result.output;
}

TEST_F(GeometryCLIAcceptanceTest, StopWithoutValueFails) {
    auto pq = temp_parquet_path("_cli_stop_noval");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --stop --output " + pq);
    EXPECT_NE(result.exit_code, 0)
        << "--stop without value should exit non-zero. Got: " << result.output;
}

// ===========================================================================
// T4: Invalid values rejected — target <= 0, stop <= 0, target <= stop
// ===========================================================================

class GeometryInvalidValuesTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(BINARY_PATH)) {
            GTEST_SKIP() << "Binary not built yet";
        }
    }
    void TearDown() override {
        for (const auto& p : temp_files_) std::filesystem::remove(p);
    }
    void track_temp(const std::string& p) { temp_files_.push_back(p); }
    std::vector<std::string> temp_files_;
};

TEST_F(GeometryInvalidValuesTest, TargetZeroExitsNonZero) {
    // Spec T4: --target 0 → error, non-zero exit
    auto pq = temp_parquet_path("_inv_target0");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --target 0 --output " + pq);
    EXPECT_NE(result.exit_code, 0)
        << "--target 0 must be rejected. Got: " << result.output;
}

TEST_F(GeometryInvalidValuesTest, TargetNegativeExitsNonZero) {
    // Spec T4: negative target → error
    auto pq = temp_parquet_path("_inv_target_neg");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --target -5 --output " + pq);
    EXPECT_NE(result.exit_code, 0)
        << "--target -5 must be rejected. Got: " << result.output;
}

TEST_F(GeometryInvalidValuesTest, StopZeroExitsNonZero) {
    // Spec T4: --stop 0 → error
    auto pq = temp_parquet_path("_inv_stop0");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --stop 0 --output " + pq);
    EXPECT_NE(result.exit_code, 0)
        << "--stop 0 must be rejected. Got: " << result.output;
}

TEST_F(GeometryInvalidValuesTest, StopNegativeExitsNonZero) {
    // Spec T4: --stop -1 → error, non-zero exit
    auto pq = temp_parquet_path("_inv_stop_neg");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --stop -1 --output " + pq);
    EXPECT_NE(result.exit_code, 0)
        << "--stop -1 must be rejected. Got: " << result.output;
}

TEST_F(GeometryInvalidValuesTest, TargetLessThanOrEqualStopExitsNonZero) {
    // Spec T4: --target 3 --stop 5 (target <= stop) → error, non-zero exit
    auto pq = temp_parquet_path("_inv_target_le_stop");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --target 3 --stop 5 --output " + pq);
    EXPECT_NE(result.exit_code, 0)
        << "--target 3 --stop 5 (target <= stop) must be rejected. Got: " << result.output;
}

TEST_F(GeometryInvalidValuesTest, TargetEqualsStopExitsNonZero) {
    // target == stop should also be rejected (edge of target <= stop)
    auto pq = temp_parquet_path("_inv_target_eq_stop");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --target 5 --stop 5 --output " + pq);
    EXPECT_NE(result.exit_code, 0)
        << "--target 5 --stop 5 (target == stop) must be rejected. Got: " << result.output;
}

TEST_F(GeometryInvalidValuesTest, InvalidTargetShowsErrorMessage) {
    // The error output should contain some indication of what went wrong
    auto pq = temp_parquet_path("_inv_errmsg");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --target 0 --output " + pq);
    // Should mention "target" or "error" or "invalid"
    bool has_error_msg = (result.output.find("target") != std::string::npos ||
                          result.output.find("Target") != std::string::npos ||
                          result.output.find("error") != std::string::npos ||
                          result.output.find("Error") != std::string::npos ||
                          result.output.find("invalid") != std::string::npos ||
                          result.output.find("Invalid") != std::string::npos);
    EXPECT_TRUE(has_error_msg)
        << "Invalid --target 0 should produce a clear error message. Got: " << result.output;
}

TEST_F(GeometryInvalidValuesTest, TargetLessStopShowsErrorMessage) {
    auto pq = temp_parquet_path("_inv_ts_errmsg");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --target 3 --stop 5 --output " + pq);
    bool has_error_msg = (result.output.find("target") != std::string::npos ||
                          result.output.find("Target") != std::string::npos ||
                          result.output.find("stop") != std::string::npos ||
                          result.output.find("Stop") != std::string::npos ||
                          result.output.find("error") != std::string::npos ||
                          result.output.find("Error") != std::string::npos ||
                          result.output.find("must be greater") != std::string::npos);
    EXPECT_TRUE(has_error_msg)
        << "target <= stop should produce a clear error message. Got: " << result.output;
}

TEST_F(GeometryInvalidValuesTest, NonIntegerTargetExitsNonZero) {
    auto pq = temp_parquet_path("_inv_target_str");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --target abc --output " + pq);
    EXPECT_NE(result.exit_code, 0)
        << "--target abc (non-integer) must be rejected. Got: " << result.output;
}

TEST_F(GeometryInvalidValuesTest, NonIntegerStopExitsNonZero) {
    auto pq = temp_parquet_path("_inv_stop_str");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --stop xyz --output " + pq);
    EXPECT_NE(result.exit_code, 0)
        << "--stop xyz (non-integer) must be rejected. Got: " << result.output;
}

TEST_F(GeometryInvalidValuesTest, FloatTargetExitsNonZero) {
    // --target 10.5 should be rejected (must be integer)
    auto pq = temp_parquet_path("_inv_target_float");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --target 10.5 --output " + pq);
    EXPECT_NE(result.exit_code, 0)
        << "--target 10.5 (float) must be rejected — integer required. Got: " << result.output;
}

// ===========================================================================
// Integration test fixture — requires binary + data
// ===========================================================================

class GeometryIntegrationTestBase : public ::testing::Test {
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

// ===========================================================================
// T1: Default geometry matches current output
// ===========================================================================

class T1_DefaultGeometryTest : public GeometryIntegrationTestBase {};

TEST_F(T1_DefaultGeometryTest, DefaultExportSucceeds) {
    // Running without --target/--stop should succeed (backward compatible)
    auto pq = temp_parquet_path("_t1_default");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq);
    EXPECT_EQ(result.exit_code, 0)
        << "Default export (no --target/--stop) should succeed. Got: " << result.output;
}

TEST_F(T1_DefaultGeometryTest, DefaultExportProducesParquet) {
    auto pq = temp_parquet_path("_t1_exists");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq);
    if (result.exit_code != 0) GTEST_SKIP() << "Export failed";

    EXPECT_TRUE(std::filesystem::exists(pq))
        << "Default export should create Parquet file";
    EXPECT_GT(std::filesystem::file_size(pq), 1000u)
        << "Parquet file should be non-trivial";
}

TEST_F(T1_DefaultGeometryTest, ExplicitDefaultsMatchImplicit) {
    // Spec T1: Default values (10, 5) produce byte-identical output
    auto pq_default = temp_parquet_path("_t1_implicit");
    auto pq_explicit = temp_parquet_path("_t1_explicit");
    track_temp(pq_default);
    track_temp(pq_explicit);

    auto result_default = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_default);
    auto result_explicit = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --target 10 --stop 5 --output " + pq_explicit);

    if (result_default.exit_code != 0 || result_explicit.exit_code != 0) {
        GTEST_SKIP() << "One or both exports failed";
    }

    auto table_default = read_parquet_table(pq_default);
    auto table_explicit = read_parquet_table(pq_explicit);
    ASSERT_NE(table_default, nullptr);
    ASSERT_NE(table_explicit, nullptr);

    // Row count must match
    EXPECT_EQ(table_default->num_rows(), table_explicit->num_rows())
        << "Explicit defaults (--target 10 --stop 5) must produce same row count as implicit defaults";

    // Column count must match
    EXPECT_EQ(table_default->num_columns(), table_explicit->num_columns())
        << "Explicit defaults must produce same column count";
}

TEST_F(T1_DefaultGeometryTest, ExplicitDefaultsProduceSameLabelDistribution) {
    // Label distribution must be identical when using explicit defaults
    auto pq_default = temp_parquet_path("_t1_dist_impl");
    auto pq_explicit = temp_parquet_path("_t1_dist_expl");
    track_temp(pq_default);
    track_temp(pq_explicit);

    auto result_default = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_default);
    auto result_explicit = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --target 10 --stop 5 --output " + pq_explicit);

    if (result_default.exit_code != 0 || result_explicit.exit_code != 0) {
        GTEST_SKIP() << "One or both exports failed";
    }

    auto table_default = read_parquet_table(pq_default);
    auto table_explicit = read_parquet_table(pq_explicit);
    ASSERT_NE(table_default, nullptr);
    ASSERT_NE(table_explicit, nullptr);

    int label_idx_d = table_default->schema()->GetFieldIndex("tb_label");
    int label_idx_e = table_explicit->schema()->GetFieldIndex("tb_label");
    ASSERT_NE(label_idx_d, -1);
    ASSERT_NE(label_idx_e, -1);

    auto labels_d = extract_doubles(table_default->column(label_idx_d));
    auto labels_e = extract_doubles(table_explicit->column(label_idx_e));

    auto dist_d = count_labels(labels_d);
    auto dist_e = count_labels(labels_e);

    EXPECT_EQ(dist_d, dist_e)
        << "Explicit defaults (--target 10 --stop 5) must produce identical "
        << "label distribution as implicit defaults";
}

// ===========================================================================
// T2: Non-default geometry produces different labels
// ===========================================================================

class T2_NonDefaultGeometryTest : public GeometryIntegrationTestBase {};

TEST_F(T2_NonDefaultGeometryTest, CustomGeometryExitsZero) {
    // --target 15 --stop 3 should succeed
    auto pq = temp_parquet_path("_t2_custom");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --target 15 --stop 3 --output " + pq);
    EXPECT_EQ(result.exit_code, 0)
        << "--target 15 --stop 3 should succeed. Got: " << result.output;
}

TEST_F(T2_NonDefaultGeometryTest, CustomGeometryProducesDifferentLabelDistribution) {
    // Spec T2: label distribution must differ from default (10, 5)
    auto pq_default = temp_parquet_path("_t2_def");
    auto pq_custom = temp_parquet_path("_t2_cust");
    track_temp(pq_default);
    track_temp(pq_custom);

    auto result_default = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_default);
    auto result_custom = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --target 15 --stop 3 --output " + pq_custom);

    if (result_default.exit_code != 0 || result_custom.exit_code != 0) {
        GTEST_SKIP() << "One or both exports failed";
    }

    auto table_default = read_parquet_table(pq_default);
    auto table_custom = read_parquet_table(pq_custom);
    ASSERT_NE(table_default, nullptr);
    ASSERT_NE(table_custom, nullptr);

    int label_idx_d = table_default->schema()->GetFieldIndex("tb_label");
    int label_idx_c = table_custom->schema()->GetFieldIndex("tb_label");
    ASSERT_NE(label_idx_d, -1);
    ASSERT_NE(label_idx_c, -1);

    auto labels_d = extract_doubles(table_default->column(label_idx_d));
    auto labels_c = extract_doubles(table_custom->column(label_idx_c));

    auto dist_d = count_labels(labels_d);
    auto dist_c = count_labels(labels_c);

    EXPECT_NE(dist_d, dist_c)
        << "Custom geometry (--target 15 --stop 3) MUST produce a different "
        << "label distribution than default (10, 5). "
        << "Default: [-1]=" << dist_d[-1] << " [0]=" << dist_d[0] << " [1]=" << dist_d[1]
        << " | Custom: [-1]=" << dist_c[-1] << " [0]=" << dist_c[0] << " [1]=" << dist_c[1];
}

TEST_F(T2_NonDefaultGeometryTest, CustomGeometryPreservesRowCount) {
    // Different geometry should produce the same number of rows (same bars,
    // just different labels). Row count is determined by bar construction,
    // not by label geometry.
    auto pq_default = temp_parquet_path("_t2_rc_def");
    auto pq_custom = temp_parquet_path("_t2_rc_cust");
    track_temp(pq_default);
    track_temp(pq_custom);

    auto result_default = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_default);
    auto result_custom = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --target 15 --stop 3 --output " + pq_custom);

    if (result_default.exit_code != 0 || result_custom.exit_code != 0) {
        GTEST_SKIP() << "One or both exports failed";
    }

    auto table_default = read_parquet_table(pq_default);
    auto table_custom = read_parquet_table(pq_custom);
    ASSERT_NE(table_default, nullptr);
    ASSERT_NE(table_custom, nullptr);

    EXPECT_EQ(table_default->num_rows(), table_custom->num_rows())
        << "Different geometry should not change the number of output bars";
}

TEST_F(T2_NonDefaultGeometryTest, CustomGeometryPreservesColumnCount) {
    // Column count should be the same 152 (bidirectional default)
    auto pq = temp_parquet_path("_t2_cols");
    track_temp(pq);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --target 15 --stop 3 --output " + pq);
    if (result.exit_code != 0) GTEST_SKIP() << "Export failed";

    auto table = read_parquet_table(pq);
    ASSERT_NE(table, nullptr);

    EXPECT_EQ(table->num_columns(), 152)
        << "Custom geometry should still produce 152-column bidirectional output. "
        << "Got " << table->num_columns();
}

TEST_F(T2_NonDefaultGeometryTest, CustomGeometryPreservesFeatureValues) {
    // Feature columns (book_imbalance_1 etc.) should be identical between
    // different geometries since bar construction doesn't depend on label params
    auto pq_default = temp_parquet_path("_t2_feat_def");
    auto pq_custom = temp_parquet_path("_t2_feat_cust");
    track_temp(pq_default);
    track_temp(pq_custom);

    auto result_default = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_default);
    auto result_custom = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --target 15 --stop 3 --output " + pq_custom);

    if (result_default.exit_code != 0 || result_custom.exit_code != 0) {
        GTEST_SKIP() << "One or both exports failed";
    }

    auto table_default = read_parquet_table(pq_default);
    auto table_custom = read_parquet_table(pq_custom);
    ASSERT_NE(table_default, nullptr);
    ASSERT_NE(table_custom, nullptr);

    // Check that a representative feature column is identical
    int feat_idx_d = table_default->schema()->GetFieldIndex("book_imbalance_1");
    int feat_idx_c = table_custom->schema()->GetFieldIndex("book_imbalance_1");
    ASSERT_NE(feat_idx_d, -1);
    ASSERT_NE(feat_idx_c, -1);

    auto feats_d = extract_doubles(table_default->column(feat_idx_d));
    auto feats_c = extract_doubles(table_custom->column(feat_idx_c));

    ASSERT_EQ(feats_d.size(), feats_c.size());

    // Check first N rows for equality
    size_t check_count = std::min(feats_d.size(), size_t(100));
    for (size_t i = 0; i < check_count; ++i) {
        EXPECT_DOUBLE_EQ(feats_d[i], feats_c[i])
            << "book_imbalance_1 at row " << i << " should be identical across "
            << "geometry changes (features don't depend on label params)";
    }
}

TEST_F(T2_NonDefaultGeometryTest, CustomGeometryLabelsInValidRange) {
    // tb_label should still be in {-1, 0, +1} with custom geometry
    auto pq = temp_parquet_path("_t2_valid");
    track_temp(pq);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --target 15 --stop 3 --output " + pq);
    if (result.exit_code != 0) GTEST_SKIP() << "Export failed";

    auto table = read_parquet_table(pq);
    ASSERT_NE(table, nullptr);
    ASSERT_GT(table->num_rows(), 0);

    int label_idx = table->schema()->GetFieldIndex("tb_label");
    ASSERT_NE(label_idx, -1);

    auto labels = extract_doubles(table->column(label_idx));
    for (size_t i = 0; i < labels.size(); ++i) {
        EXPECT_TRUE(labels[i] == -1.0 || labels[i] == 0.0 || labels[i] == 1.0)
            << "tb_label at row " << i << " is " << labels[i]
            << " — must be -1.0, 0.0, or 1.0 with custom geometry";
    }
}

TEST_F(T2_NonDefaultGeometryTest, WiderTargetIncreasesHoldCount) {
    // Spec T2: "Higher target → fewer directional labels (more HOLD)"
    auto pq_default = temp_parquet_path("_t2_hold_def");
    auto pq_wide = temp_parquet_path("_t2_hold_wide");
    track_temp(pq_default);
    track_temp(pq_wide);

    auto result_default = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_default);
    // Use a much wider target to make the effect more pronounced
    auto result_wide = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --target 20 --stop 5 --output " + pq_wide);

    if (result_default.exit_code != 0 || result_wide.exit_code != 0) {
        GTEST_SKIP() << "One or both exports failed";
    }

    auto table_default = read_parquet_table(pq_default);
    auto table_wide = read_parquet_table(pq_wide);
    ASSERT_NE(table_default, nullptr);
    ASSERT_NE(table_wide, nullptr);

    int label_idx_d = table_default->schema()->GetFieldIndex("tb_label");
    int label_idx_w = table_wide->schema()->GetFieldIndex("tb_label");
    ASSERT_NE(label_idx_d, -1);
    ASSERT_NE(label_idx_w, -1);

    auto labels_d = extract_doubles(table_default->column(label_idx_d));
    auto labels_w = extract_doubles(table_wide->column(label_idx_w));

    auto dist_d = count_labels(labels_d);
    auto dist_w = count_labels(labels_w);

    // With target=20 (wider) vs target=10, we expect more HOLD labels (label=0)
    // because more bars won't reach the wider take-profit barrier before timeout
    EXPECT_GT(dist_w[0], dist_d[0])
        << "Wider target (20 vs 10) should produce more HOLD labels. "
        << "Default HOLD=" << dist_d[0] << " Wide HOLD=" << dist_w[0];
}

// ===========================================================================
// T5: --legacy-labels works with custom target/stop
// ===========================================================================

class T5_LegacyModeWithGeometryTest : public GeometryIntegrationTestBase {};

TEST_F(T5_LegacyModeWithGeometryTest, LegacyWithCustomGeometryExitsZero) {
    // Spec T5: --target 15 --stop 3 --legacy-labels should succeed
    auto pq = temp_parquet_path("_t5_legacy");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --target 15 --stop 3 --legacy-labels --output " + pq);
    EXPECT_EQ(result.exit_code, 0)
        << "--target 15 --stop 3 --legacy-labels should succeed. Got: " << result.output;
}

TEST_F(T5_LegacyModeWithGeometryTest, LegacyWithCustomGeometryProduces149Columns) {
    // Spec T5: produces 149-column output (not 152)
    auto pq = temp_parquet_path("_t5_cols");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --target 15 --stop 3 --legacy-labels --output " + pq);
    if (result.exit_code != 0) GTEST_SKIP() << "Export failed";

    auto table = read_parquet_table(pq);
    ASSERT_NE(table, nullptr);

    EXPECT_EQ(table->num_columns(), 149)
        << "--legacy-labels with custom geometry must produce 149 columns (not 152). "
        << "Got " << table->num_columns();
}

TEST_F(T5_LegacyModeWithGeometryTest, LegacyWithCustomGeometryRespectTarget) {
    // The target/stop values should be respected in legacy mode too —
    // label distribution should differ from legacy with default geometry
    auto pq_default = temp_parquet_path("_t5_leg_def");
    auto pq_custom = temp_parquet_path("_t5_leg_cust");
    track_temp(pq_default);
    track_temp(pq_custom);

    auto result_default = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --legacy-labels --output " + pq_default);
    auto result_custom = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --target 15 --stop 3 --legacy-labels --output " + pq_custom);

    if (result_default.exit_code != 0 || result_custom.exit_code != 0) {
        GTEST_SKIP() << "One or both exports failed";
    }

    auto table_default = read_parquet_table(pq_default);
    auto table_custom = read_parquet_table(pq_custom);
    ASSERT_NE(table_default, nullptr);
    ASSERT_NE(table_custom, nullptr);

    int label_idx_d = table_default->schema()->GetFieldIndex("tb_label");
    int label_idx_c = table_custom->schema()->GetFieldIndex("tb_label");
    ASSERT_NE(label_idx_d, -1);
    ASSERT_NE(label_idx_c, -1);

    auto labels_d = extract_doubles(table_default->column(label_idx_d));
    auto labels_c = extract_doubles(table_custom->column(label_idx_c));

    auto dist_d = count_labels(labels_d);
    auto dist_c = count_labels(labels_c);

    EXPECT_NE(dist_d, dist_c)
        << "Custom geometry with --legacy-labels must produce different label distribution "
        << "than default geometry with --legacy-labels. Target/stop must be respected "
        << "in the legacy computation path.";
}

TEST_F(T5_LegacyModeWithGeometryTest, LegacyCustomGeometryHasNoNewColumns) {
    // Legacy mode should not have bidirectional diagnostic columns
    auto pq = temp_parquet_path("_t5_no_bidir");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --target 15 --stop 3 --legacy-labels --output " + pq);
    if (result.exit_code != 0) GTEST_SKIP() << "Export failed";

    auto table = read_parquet_table(pq);
    ASSERT_NE(table, nullptr);

    auto schema = table->schema();
    EXPECT_EQ(schema->GetFieldIndex("tb_both_triggered"), -1)
        << "tb_both_triggered must NOT exist in legacy mode";
    EXPECT_EQ(schema->GetFieldIndex("tb_long_triggered"), -1)
        << "tb_long_triggered must NOT exist in legacy mode";
    EXPECT_EQ(schema->GetFieldIndex("tb_short_triggered"), -1)
        << "tb_short_triggered must NOT exist in legacy mode";
}

// ===========================================================================
// Edge cases: boundary values, partial overrides, extreme geometry
// ===========================================================================

class GeometryEdgeCasesTest : public GeometryIntegrationTestBase {};

TEST_F(GeometryEdgeCasesTest, TargetOneStopOneInvalid) {
    // target=1, stop=1 → target == stop → rejected
    auto pq = temp_parquet_path("_edge_11");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --target 1 --stop 1 --output " + pq);
    EXPECT_NE(result.exit_code, 0)
        << "--target 1 --stop 1 must be rejected (target <= stop). Got: " << result.output;
}

TEST_F(GeometryEdgeCasesTest, MinimumValidGeometry) {
    // Smallest valid geometry: target=2, stop=1
    auto pq = temp_parquet_path("_edge_min");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --target 2 --stop 1 --output " + pq);
    EXPECT_EQ(result.exit_code, 0)
        << "--target 2 --stop 1 (minimum valid geometry) should succeed. Got: " << result.output;
}

TEST_F(GeometryEdgeCasesTest, LargeTargetValue) {
    // Very large target value should work
    auto pq = temp_parquet_path("_edge_large");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --target 100 --stop 5 --output " + pq);
    EXPECT_EQ(result.exit_code, 0)
        << "--target 100 --stop 5 should succeed. Got: " << result.output;
}

TEST_F(GeometryEdgeCasesTest, TargetOnlyOverride) {
    // Only --target provided, --stop uses default (5)
    auto pq = temp_parquet_path("_edge_target_only");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --target 15 --output " + pq);
    EXPECT_EQ(result.exit_code, 0)
        << "--target 15 alone (stop defaults to 5) should succeed. Got: " << result.output;
}

TEST_F(GeometryEdgeCasesTest, StopOnlyOverride) {
    // Only --stop provided, --target uses default (10)
    auto pq = temp_parquet_path("_edge_stop_only");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --stop 3 --output " + pq);
    EXPECT_EQ(result.exit_code, 0)
        << "--stop 3 alone (target defaults to 10) should succeed. Got: " << result.output;
}

TEST_F(GeometryEdgeCasesTest, StopOnlyWithHighValueInvalid) {
    // --stop 15 alone: target defaults to 10, so 10 <= 15 → rejected
    auto pq = temp_parquet_path("_edge_stop_high");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --stop 15 --output " + pq);
    EXPECT_NE(result.exit_code, 0)
        << "--stop 15 with default target=10 should be rejected (10 <= 15). Got: "
        << result.output;
}

TEST_F(GeometryEdgeCasesTest, TargetOnlyProducesDifferentLabels) {
    // --target 15 alone (stop=5 default) should differ from default (10, 5)
    auto pq_default = temp_parquet_path("_edge_to_def");
    auto pq_target = temp_parquet_path("_edge_to_tgt");
    track_temp(pq_default);
    track_temp(pq_target);

    auto result_default = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_default);
    auto result_target = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --target 15 --output " + pq_target);

    if (result_default.exit_code != 0 || result_target.exit_code != 0) {
        GTEST_SKIP() << "One or both exports failed";
    }

    auto table_default = read_parquet_table(pq_default);
    auto table_target = read_parquet_table(pq_target);
    ASSERT_NE(table_default, nullptr);
    ASSERT_NE(table_target, nullptr);

    int label_idx_d = table_default->schema()->GetFieldIndex("tb_label");
    int label_idx_t = table_target->schema()->GetFieldIndex("tb_label");
    ASSERT_NE(label_idx_d, -1);
    ASSERT_NE(label_idx_t, -1);

    auto labels_d = extract_doubles(table_default->column(label_idx_d));
    auto labels_t = extract_doubles(table_target->column(label_idx_t));

    auto dist_d = count_labels(labels_d);
    auto dist_t = count_labels(labels_t);

    EXPECT_NE(dist_d, dist_t)
        << "Overriding only --target should change label distribution";
}

// ===========================================================================
// Cross-bar-type: geometry flags work with volume/dollar/tick too
// ===========================================================================

class GeometryCrossBarTypeTest : public GeometryIntegrationTestBase {};

TEST_F(GeometryCrossBarTypeTest, GeometryWithVolumeBarSucceeds) {
    auto pq = temp_parquet_path("_xbar_volume");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type volume --bar-param 100 --target 15 --stop 3 --output " + pq);
    EXPECT_EQ(result.exit_code, 0)
        << "--target/--stop with volume bars should succeed. Got: " << result.output;
}

TEST_F(GeometryCrossBarTypeTest, GeometryWithDollarBarSucceeds) {
    auto pq = temp_parquet_path("_xbar_dollar");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type dollar --bar-param 25000 --target 15 --stop 3 --output " + pq);
    EXPECT_EQ(result.exit_code, 0)
        << "--target/--stop with dollar bars should succeed. Got: " << result.output;
}

TEST_F(GeometryCrossBarTypeTest, GeometryWithTickBarSucceeds) {
    auto pq = temp_parquet_path("_xbar_tick");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type tick --bar-param 50 --target 15 --stop 3 --output " + pq);
    EXPECT_EQ(result.exit_code, 0)
        << "--target/--stop with tick bars should succeed. Got: " << result.output;
}
