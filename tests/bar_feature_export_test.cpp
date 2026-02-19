// bar_feature_export_test.cpp — TDD RED phase tests for bar_feature_export CLI tool
// Spec: .kit/docs/bar-feature-export.md
//
// Tests the CLI interface contract, CSV schema, argument parsing, warmup
// exclusion, NaN return exclusion, and metadata column behavior.
//
// These tests invoke the built binary via std::system / popen and verify
// its outputs. They do NOT require databento or live data — instead they
// test argument parsing, error handling, and CSV header correctness by
// running the tool with controlled arguments and inspecting its output.
//
// The spec's pipeline tests (actual data processing) are integration-level
// and tested by the tool producing valid output on real data. Unit tests
// here focus on the CLI contract and output format.

#include <gtest/gtest.h>

#include "features/bar_features.hpp"         // BarFeatureRow::feature_names()
#include "features/raw_representations.hpp"  // MessageSummary::SUMMARY_SIZE
#include "test_export_helpers.hpp"

#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

// ===========================================================================
// Helpers
// ===========================================================================
namespace {

using export_test_helpers::BINARY_PATH;
using export_test_helpers::run_command;
using export_test_helpers::parse_csv_header;
using export_test_helpers::read_first_line;
using export_test_helpers::read_all_lines;

// Temp output CSV path for tests.
std::string temp_csv_path(const std::string& suffix = "") {
    return (std::filesystem::temp_directory_path() /
            ("bar_feature_export_test" + suffix + ".csv")).string();
}

// The expected CSV header — must match info-decomposition/features.csv exactly.
// 149 columns: 6 metadata + 62 Track A + 40 book_snap + 33 msg_summary
//              + 4 returns + 1 mbo_event_count + 3 tb_labels
std::string expected_csv_header() {
    std::ostringstream ss;
    // Metadata (6)
    ss << "timestamp,bar_type,bar_param,day,is_warmup,bar_index";

    // Track A (62) — from BarFeatureRow::feature_names()
    auto names = BarFeatureRow::feature_names();
    for (const auto& name : names) {
        ss << "," << name;
    }

    // Book snapshot (40)
    for (int i = 0; i < 40; ++i) {
        ss << ",book_snap_" << i;
    }

    // Message summary (33)
    for (size_t i = 0; i < MessageSummary::SUMMARY_SIZE; ++i) {
        ss << ",msg_summary_" << i;
    }

    // Forward returns (4)
    ss << ",return_1,return_5,return_20,return_100";

    // Event count (1)
    ss << ",mbo_event_count";

    // Triple barrier labels (3)
    ss << ",tb_label,tb_exit_type,tb_bars_held";

    return ss.str();
}

}  // anonymous namespace

// ===========================================================================
// Binary Existence — Exit Criteria: binary exists after build
// ===========================================================================

class BinaryExistenceTest : public ::testing::Test {};

TEST_F(BinaryExistenceTest, BinaryExistsAfterBuild) {
    // Exit criteria: `build/bar_feature_export` exists and is executable after build
    EXPECT_TRUE(std::filesystem::exists(BINARY_PATH))
        << "Expected binary at " << BINARY_PATH << " — must be built via CMakeLists.txt";
}

TEST_F(BinaryExistenceTest, BinaryIsExecutable) {
    if (!std::filesystem::exists(BINARY_PATH)) {
        GTEST_SKIP() << "Binary not built yet";
    }
    auto perms = std::filesystem::status(BINARY_PATH).permissions();
    bool is_exec = (perms & std::filesystem::perms::owner_exec) != std::filesystem::perms::none;
    EXPECT_TRUE(is_exec) << "Binary must be executable";
}

// ===========================================================================
// CLI Argument Parsing — Exit Criteria: args parsed, missing/invalid → exit 1
// ===========================================================================

class CLIArgParsingTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(BINARY_PATH)) {
            GTEST_SKIP() << "Binary not built yet";
        }
    }
    void TearDown() override {
        // Clean up temp files
        for (const auto& p : temp_files_) {
            std::filesystem::remove(p);
        }
    }
    void track_temp(const std::string& p) { temp_files_.push_back(p); }
    std::vector<std::string> temp_files_;
};

TEST_F(CLIArgParsingTest, NoArgsReturnsNonZeroExitCode) {
    // Missing/invalid args produce non-zero exit code and usage message
    auto result = run_command(BINARY_PATH);
    EXPECT_NE(result.exit_code, 0)
        << "No arguments should produce non-zero exit code";
}

TEST_F(CLIArgParsingTest, NoArgsShowsUsageMessage) {
    auto result = run_command(BINARY_PATH);
    // Should mention required arguments in usage
    bool has_usage = (result.output.find("bar-type") != std::string::npos ||
                      result.output.find("usage") != std::string::npos ||
                      result.output.find("Usage") != std::string::npos ||
                      result.output.find("--bar-type") != std::string::npos);
    EXPECT_TRUE(has_usage)
        << "Usage message should mention required arguments. Got: " << result.output;
}

TEST_F(CLIArgParsingTest, MissingBarTypeReturnsNonZero) {
    auto csv = temp_csv_path("_missing_bt");
    track_temp(csv);
    auto result = run_command(BINARY_PATH + " --bar-param 5.0 --output " + csv);
    EXPECT_NE(result.exit_code, 0)
        << "Missing --bar-type should produce non-zero exit code";
}

TEST_F(CLIArgParsingTest, MissingBarParamReturnsNonZero) {
    auto csv = temp_csv_path("_missing_bp");
    track_temp(csv);
    auto result = run_command(BINARY_PATH + " --bar-type time --output " + csv);
    EXPECT_NE(result.exit_code, 0)
        << "Missing --bar-param should produce non-zero exit code";
}

TEST_F(CLIArgParsingTest, MissingOutputReturnsNonZero) {
    auto result = run_command(BINARY_PATH + " --bar-type time --bar-param 5.0");
    EXPECT_NE(result.exit_code, 0)
        << "Missing --output should produce non-zero exit code";
}

TEST_F(CLIArgParsingTest, InvalidBarTypeReturnsNonZero) {
    auto csv = temp_csv_path("_invalid_bt");
    track_temp(csv);
    auto result = run_command(
        BINARY_PATH + " --bar-type INVALID --bar-param 5.0 --output " + csv);
    EXPECT_NE(result.exit_code, 0)
        << "Invalid bar type 'INVALID' should produce non-zero exit code";
}

TEST_F(CLIArgParsingTest, ValidBarTypeTimeAccepted) {
    // We just check it doesn't fail on arg parsing itself.
    // It may fail on data loading, but the exit should not be due to arg error.
    // We verify by checking output doesn't contain usage/argument error messages.
    auto csv = temp_csv_path("_valid_time");
    track_temp(csv);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5.0 --output " + csv);
    // If data files don't exist, the tool may still exit non-zero but shouldn't
    // complain about arguments.
    bool arg_error = (result.output.find("Unknown bar type") != std::string::npos ||
                      result.output.find("Invalid bar type") != std::string::npos ||
                      result.output.find("Missing") != std::string::npos);
    EXPECT_FALSE(arg_error)
        << "Valid args should not produce argument errors. Got: " << result.output;
}

TEST_F(CLIArgParsingTest, ValidBarTypeVolumeAccepted) {
    auto csv = temp_csv_path("_valid_volume");
    track_temp(csv);
    auto result = run_command(
        BINARY_PATH + " --bar-type volume --bar-param 100 --output " + csv);
    bool arg_error = (result.output.find("Unknown bar type") != std::string::npos ||
                      result.output.find("Invalid bar type") != std::string::npos ||
                      result.output.find("Missing") != std::string::npos);
    EXPECT_FALSE(arg_error)
        << "bar-type 'volume' should be accepted. Got: " << result.output;
}

TEST_F(CLIArgParsingTest, ValidBarTypeDollarAccepted) {
    auto csv = temp_csv_path("_valid_dollar");
    track_temp(csv);
    auto result = run_command(
        BINARY_PATH + " --bar-type dollar --bar-param 25000 --output " + csv);
    bool arg_error = (result.output.find("Unknown bar type") != std::string::npos ||
                      result.output.find("Invalid bar type") != std::string::npos ||
                      result.output.find("Missing") != std::string::npos);
    EXPECT_FALSE(arg_error)
        << "bar-type 'dollar' should be accepted. Got: " << result.output;
}

TEST_F(CLIArgParsingTest, ValidBarTypeTickAccepted) {
    auto csv = temp_csv_path("_valid_tick");
    track_temp(csv);
    auto result = run_command(
        BINARY_PATH + " --bar-type tick --bar-param 50 --output " + csv);
    bool arg_error = (result.output.find("Unknown bar type") != std::string::npos ||
                      result.output.find("Invalid bar type") != std::string::npos ||
                      result.output.find("Missing") != std::string::npos);
    EXPECT_FALSE(arg_error)
        << "bar-type 'tick' should be accepted. Got: " << result.output;
}

TEST_F(CLIArgParsingTest, OutputFileOpenFailureReturnsNonZero) {
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5.0 --output /nonexistent/dir/out.csv");
    EXPECT_NE(result.exit_code, 0)
        << "File open failure should produce non-zero exit code";
}

// ===========================================================================
// CSV Header Schema — Exit Criteria: header matches features.csv exactly
// ===========================================================================

class CSVHeaderSchemaTest : public ::testing::Test {};

TEST_F(CSVHeaderSchemaTest, ExpectedHeaderHas149Columns) {
    // Verify our reference: 6 + 62 + 40 + 33 + 4 + 1 + 3 = 149
    auto header = expected_csv_header();
    auto cols = parse_csv_header(header);
    EXPECT_EQ(cols.size(), 149u)
        << "Expected CSV header must have exactly 149 columns (6 meta + 62 Track A "
        << "+ 40 book_snap + 33 msg_summary + 4 returns + 1 event_count + 3 tb_labels)";
}

TEST_F(CSVHeaderSchemaTest, MetadataColumnsInOrder) {
    auto header = expected_csv_header();
    auto cols = parse_csv_header(header);
    ASSERT_GE(cols.size(), 6u);
    EXPECT_EQ(cols[0], "timestamp");
    EXPECT_EQ(cols[1], "bar_type");
    EXPECT_EQ(cols[2], "bar_param");
    EXPECT_EQ(cols[3], "day");
    EXPECT_EQ(cols[4], "is_warmup");
    EXPECT_EQ(cols[5], "bar_index");
}

TEST_F(CSVHeaderSchemaTest, TrackAHas62Features) {
    auto names = BarFeatureRow::feature_names();
    EXPECT_EQ(names.size(), 62u)
        << "BarFeatureRow::feature_names() must return exactly 62 names";
}

TEST_F(CSVHeaderSchemaTest, TrackAFeaturesStartAtColumn6) {
    auto header = expected_csv_header();
    auto cols = parse_csv_header(header);
    auto names = BarFeatureRow::feature_names();
    ASSERT_GE(cols.size(), 6u + names.size());
    for (size_t i = 0; i < names.size(); ++i) {
        EXPECT_EQ(cols[6 + i], names[i])
            << "Track A feature at column " << (6 + i) << " should be '" << names[i] << "'";
    }
}

TEST_F(CSVHeaderSchemaTest, BookSnapColumnsAt68Through107) {
    auto header = expected_csv_header();
    auto cols = parse_csv_header(header);
    size_t book_start = 6 + 62;  // metadata + Track A
    ASSERT_GE(cols.size(), book_start + 40);
    for (int i = 0; i < 40; ++i) {
        std::string expected = "book_snap_" + std::to_string(i);
        EXPECT_EQ(cols[book_start + i], expected)
            << "book_snap column at index " << (book_start + i);
    }
}

TEST_F(CSVHeaderSchemaTest, MsgSummaryColumnsAt108Through140) {
    auto header = expected_csv_header();
    auto cols = parse_csv_header(header);
    size_t msg_start = 6 + 62 + 40;
    ASSERT_GE(cols.size(), msg_start + MessageSummary::SUMMARY_SIZE);
    for (size_t i = 0; i < MessageSummary::SUMMARY_SIZE; ++i) {
        std::string expected = "msg_summary_" + std::to_string(i);
        EXPECT_EQ(cols[msg_start + i], expected)
            << "msg_summary column at index " << (msg_start + i);
    }
}

TEST_F(CSVHeaderSchemaTest, ForwardReturnColumnsAt141Through144) {
    auto header = expected_csv_header();
    auto cols = parse_csv_header(header);
    size_t ret_start = 6 + 62 + 40 + 33;
    ASSERT_GE(cols.size(), ret_start + 4);
    EXPECT_EQ(cols[ret_start + 0], "return_1");
    EXPECT_EQ(cols[ret_start + 1], "return_5");
    EXPECT_EQ(cols[ret_start + 2], "return_20");
    EXPECT_EQ(cols[ret_start + 3], "return_100");
}

TEST_F(CSVHeaderSchemaTest, EventCountPrecedesTBColumns) {
    auto header = expected_csv_header();
    auto cols = parse_csv_header(header);
    EXPECT_EQ(cols[145], "mbo_event_count");
    EXPECT_EQ(cols[146], "tb_label");
    EXPECT_EQ(cols[147], "tb_exit_type");
    EXPECT_EQ(cols[148], "tb_bars_held");
}

TEST_F(CSVHeaderSchemaTest, HeaderPrefixMatchesReferenceCSV) {
    // Compare the first 146 columns against the info-decomposition reference CSV.
    // The reference predates the TB label extension (3 extra columns at the end).
    std::vector<std::string> candidates = {
        ".kit/results/info-decomposition/features.csv",
        "../.kit/results/info-decomposition/features.csv",
    };
    std::string ref_path;
    for (const auto& c : candidates) {
        if (std::filesystem::exists(c)) { ref_path = c; break; }
    }
    if (ref_path.empty()) {
        GTEST_SKIP() << "Reference CSV not available";
    }
    auto ref_cols = parse_csv_header(read_first_line(ref_path));
    auto expected_cols = parse_csv_header(expected_csv_header());
    ASSERT_GE(expected_cols.size(), ref_cols.size());
    for (size_t i = 0; i < ref_cols.size(); ++i) {
        EXPECT_EQ(expected_cols[i], ref_cols[i])
            << "Column " << i << " mismatch with reference CSV";
    }
}

// ===========================================================================
// MessageSummary Size Contract
// ===========================================================================

class MessageSummaryContractTest : public ::testing::Test {};

TEST_F(MessageSummaryContractTest, SummarySizeIs33) {
    EXPECT_EQ(MessageSummary::SUMMARY_SIZE, 33u)
        << "MessageSummary::SUMMARY_SIZE must be 33 (10 deciles * 3 actions + 3 aggregates)";
}

// ===========================================================================
// BarFactory Acceptance — valid bar types recognized
// ===========================================================================

#include "bars/bar_factory.hpp"

class BarFactoryContractTest : public ::testing::Test {};

TEST_F(BarFactoryContractTest, TimeBarTypeReturnsNonNull) {
    auto builder = BarFactory::create("time", 5.0);
    EXPECT_NE(builder, nullptr) << "BarFactory must create a builder for type 'time'";
}

TEST_F(BarFactoryContractTest, VolumeBarTypeReturnsNonNull) {
    auto builder = BarFactory::create("volume", 100.0);
    EXPECT_NE(builder, nullptr) << "BarFactory must create a builder for type 'volume'";
}

TEST_F(BarFactoryContractTest, DollarBarTypeReturnsNonNull) {
    auto builder = BarFactory::create("dollar", 25000.0);
    EXPECT_NE(builder, nullptr) << "BarFactory must create a builder for type 'dollar'";
}

TEST_F(BarFactoryContractTest, TickBarTypeReturnsNonNull) {
    auto builder = BarFactory::create("tick", 50.0);
    EXPECT_NE(builder, nullptr) << "BarFactory must create a builder for type 'tick'";
}

TEST_F(BarFactoryContractTest, InvalidBarTypeReturnsNull) {
    auto builder = BarFactory::create("INVALID", 5.0);
    EXPECT_EQ(builder, nullptr)
        << "BarFactory::create must return nullptr for unknown bar type";
}

TEST_F(BarFactoryContractTest, EmptyBarTypeReturnsNull) {
    auto builder = BarFactory::create("", 5.0);
    EXPECT_EQ(builder, nullptr)
        << "BarFactory::create must return nullptr for empty bar type";
}

// ===========================================================================
// CSV Output Structure Tests — verify row/column structure
// Tests below exercise the tool via shell if binary exists, otherwise
// verify the contract at the API level.
// ===========================================================================

class CSVOutputTest : public ::testing::Test {
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

TEST_F(CSVOutputTest, OutputCSVFirstLineIsHeader) {
    // Run tool with valid args (may skip days if data not present)
    auto csv = temp_csv_path("_header_check");
    track_temp(csv);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5.0 --output " + csv);

    if (result.exit_code != 0) {
        GTEST_SKIP() << "Tool exited non-zero (data files may not be present)";
    }

    auto first_line = read_first_line(csv);
    auto expected = expected_csv_header();
    EXPECT_EQ(first_line, expected)
        << "First line of output CSV must be the header matching the spec";
}

TEST_F(CSVOutputTest, AllDataRowsHave149Columns) {
    auto csv = temp_csv_path("_colcount");
    track_temp(csv);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5.0 --output " + csv);

    if (result.exit_code != 0) {
        GTEST_SKIP() << "Tool exited non-zero";
    }

    auto lines = read_all_lines(csv);
    ASSERT_GE(lines.size(), 2u) << "CSV must have at least header + 1 data row";

    auto header_cols = parse_csv_header(lines[0]);
    for (size_t i = 1; i < lines.size(); ++i) {
        auto row_cols = parse_csv_header(lines[i]);
        EXPECT_EQ(row_cols.size(), header_cols.size())
            << "Row " << i << " has " << row_cols.size()
            << " columns, expected " << header_cols.size();
    }
}

// ===========================================================================
// Metadata Column Tests — bar_type and bar_param reflect CLI args
// ===========================================================================

class MetadataColumnsTest : public ::testing::Test {
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

TEST_F(MetadataColumnsTest, BarTypeReflectsCLIArgTime) {
    auto csv = temp_csv_path("_meta_time");
    track_temp(csv);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5.0 --output " + csv);

    if (result.exit_code != 0) {
        GTEST_SKIP() << "Tool exited non-zero";
    }

    auto lines = read_all_lines(csv);
    if (lines.size() < 2) {
        GTEST_SKIP() << "No data rows produced";
    }

    // bar_type is column index 1
    auto row_cols = parse_csv_header(lines[1]);
    ASSERT_GE(row_cols.size(), 3u);
    EXPECT_EQ(row_cols[1], "time")
        << "bar_type column must reflect CLI arg 'time'";
}

TEST_F(MetadataColumnsTest, BarParamReflectsCLIArgForTime) {
    auto csv = temp_csv_path("_meta_param_time");
    track_temp(csv);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + csv);

    if (result.exit_code != 0) {
        GTEST_SKIP() << "Tool exited non-zero";
    }

    auto lines = read_all_lines(csv);
    if (lines.size() < 2) {
        GTEST_SKIP() << "No data rows produced";
    }

    // bar_param is column index 2
    auto row_cols = parse_csv_header(lines[1]);
    ASSERT_GE(row_cols.size(), 3u);
    // bar_param should be "5" or "5.0" (numeric representation of the CLI arg)
    EXPECT_TRUE(row_cols[2] == "5" || row_cols[2] == "5.0" || row_cols[2] == "5.00")
        << "bar_param column should be 5 for time_5s. Got: " << row_cols[2];
}

TEST_F(MetadataColumnsTest, BarTypeReflectsCLIArgVolume) {
    auto csv = temp_csv_path("_meta_volume");
    track_temp(csv);
    auto result = run_command(
        BINARY_PATH + " --bar-type volume --bar-param 100 --output " + csv);

    if (result.exit_code != 0) {
        GTEST_SKIP() << "Tool exited non-zero";
    }

    auto lines = read_all_lines(csv);
    if (lines.size() < 2) {
        GTEST_SKIP() << "No data rows produced";
    }

    auto row_cols = parse_csv_header(lines[1]);
    ASSERT_GE(row_cols.size(), 3u);
    EXPECT_EQ(row_cols[1], "volume")
        << "bar_type column must reflect CLI arg 'volume'";
    EXPECT_TRUE(row_cols[2] == "100" || row_cols[2] == "100.0")
        << "bar_param column should be 100. Got: " << row_cols[2];
}

// ===========================================================================
// Warmup Exclusion — first 50 bars per day excluded from output
// ===========================================================================

class WarmupExclusionTest : public ::testing::Test {
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

TEST_F(WarmupExclusionTest, NoWarmupRowsInOutput) {
    auto csv = temp_csv_path("_warmup");
    track_temp(csv);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5.0 --output " + csv);

    if (result.exit_code != 0) {
        GTEST_SKIP() << "Tool exited non-zero";
    }

    auto lines = read_all_lines(csv);
    if (lines.size() < 2) {
        GTEST_SKIP() << "No data rows produced";
    }

    // is_warmup is column index 4. No row should have is_warmup == "true" or "1"
    for (size_t i = 1; i < lines.size(); ++i) {
        auto cols = parse_csv_header(lines[i]);
        ASSERT_GE(cols.size(), 5u) << "Row " << i << " too short";
        EXPECT_NE(cols[4], "true")
            << "Warmup row found at line " << i << " — warmup bars must be excluded";
        EXPECT_NE(cols[4], "1")
            << "Warmup row found at line " << i << " — warmup bars must be excluded";
    }
}

TEST_F(WarmupExclusionTest, BarIndexStartsAbove49) {
    // Since warmup bars (indices 0-49) are excluded, bar_index in output
    // should always be >= 50
    auto csv = temp_csv_path("_bar_idx");
    track_temp(csv);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5.0 --output " + csv);

    if (result.exit_code != 0) {
        GTEST_SKIP() << "Tool exited non-zero";
    }

    auto lines = read_all_lines(csv);
    if (lines.size() < 2) {
        GTEST_SKIP() << "No data rows produced";
    }

    // bar_index is column index 5
    for (size_t i = 1; i < lines.size(); ++i) {
        auto cols = parse_csv_header(lines[i]);
        ASSERT_GE(cols.size(), 6u) << "Row " << i << " too short";
        int bar_index = std::stoi(cols[5]);
        EXPECT_GE(bar_index, 50)
            << "bar_index " << bar_index << " at line " << i
            << " is < 50 — warmup bars (first 50) must be excluded";
    }
}

// ===========================================================================
// NaN Forward Return Exclusion — bars without valid fwd_return_1 excluded
// ===========================================================================

class NaNReturnExclusionTest : public ::testing::Test {
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

TEST_F(NaNReturnExclusionTest, NoNaNReturn1InOutput) {
    auto csv = temp_csv_path("_nanret");
    track_temp(csv);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5.0 --output " + csv);

    if (result.exit_code != 0) {
        GTEST_SKIP() << "Tool exited non-zero";
    }

    auto lines = read_all_lines(csv);
    if (lines.size() < 2) {
        GTEST_SKIP() << "No data rows produced";
    }

    // return_1 is at column index 6+62+40+33 = 141
    constexpr size_t RETURN_1_COL = 6 + 62 + 40 + 33;
    for (size_t i = 1; i < lines.size(); ++i) {
        auto cols = parse_csv_header(lines[i]);
        ASSERT_GT(cols.size(), RETURN_1_COL) << "Row " << i << " too short";
        EXPECT_NE(cols[RETURN_1_COL], "NaN")
            << "return_1 at row " << i << " is NaN — bars with NaN forward returns "
            << "must be excluded from output";
        EXPECT_NE(cols[RETURN_1_COL], "nan")
            << "return_1 at row " << i << " is nan — bars with NaN forward returns "
            << "must be excluded from output";
    }
}

// ===========================================================================
// Feature Count Contract — verify BarFeatureRow reports 62 features
// ===========================================================================

class FeatureCountTest : public ::testing::Test {};

TEST_F(FeatureCountTest, FeatureCountIs62) {
    EXPECT_EQ(BarFeatureRow::feature_count(), 62u);
}

TEST_F(FeatureCountTest, FeatureNamesCountIs62) {
    EXPECT_EQ(BarFeatureRow::feature_names().size(), 62u);
}

TEST_F(FeatureCountTest, TotalColumnCountIs149) {
    // 6 metadata + 62 Track A + 40 book snap + 33 msg summary + 4 returns + 1 event count + 3 tb labels
    size_t total = 6 + BarFeatureRow::feature_count() + 40 + MessageSummary::SUMMARY_SIZE + 4 + 1 + 3;
    EXPECT_EQ(total, 149u);
}

// ===========================================================================
// BookSnapshotExport Contract — 40 flattened values
// ===========================================================================

class BookSnapshotExportContractTest : public ::testing::Test {};

TEST_F(BookSnapshotExportContractTest, FlattenReturns40Elements) {
    Bar bar{};
    bar.close_mid = 4500.0f;
    for (int i = 0; i < BOOK_DEPTH; ++i) {
        bar.bids[i][0] = 4499.75f - i * 0.25f;
        bar.bids[i][1] = 10.0f;
        bar.asks[i][0] = 4500.25f + i * 0.25f;
        bar.asks[i][1] = 10.0f;
    }
    auto flat = BookSnapshotExport::flatten(bar);
    EXPECT_EQ(flat.size(), 40u)
        << "BookSnapshotExport::flatten must return exactly 40 values";
}

// ===========================================================================
// Exit Code Contract — success = 0, failure = 1
// ===========================================================================

class ExitCodeContractTest : public ::testing::Test {
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

TEST_F(ExitCodeContractTest, SuccessfulRunReturnsZero) {
    // This test only passes when data files are present
    auto csv = temp_csv_path("_exit0");
    track_temp(csv);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5.0 --output " + csv);

    // If the data directory exists and has files, expect exit 0
    if (!std::filesystem::exists("DATA/GLBX-20260207-L953CAPU5B")) {
        GTEST_SKIP() << "Data directory not available";
    }

    EXPECT_EQ(result.exit_code, 0)
        << "Successful run should exit with code 0. Output: " << result.output;
}

TEST_F(ExitCodeContractTest, BadArgsReturnOne) {
    auto result = run_command(BINARY_PATH);
    EXPECT_EQ(result.exit_code, 1)
        << "Missing args should exit with code 1";
}

TEST_F(ExitCodeContractTest, InvalidBarTypeReturnsOne) {
    auto csv = temp_csv_path("_exit1_bt");
    track_temp(csv);
    auto result = run_command(
        BINARY_PATH + " --bar-type nosuchtype --bar-param 5.0 --output " + csv);
    EXPECT_EQ(result.exit_code, 1)
        << "Invalid bar type should exit with code 1";
}

// ===========================================================================
// Column Order Consistency — header and reference CSV agreement
// ===========================================================================

class ColumnOrderTest : public ::testing::Test {};

TEST_F(ColumnOrderTest, FeatureNamesAreStable) {
    // Verify the feature names haven't changed since the spec was written.
    // These are the expected first and last Track A features.
    auto names = BarFeatureRow::feature_names();
    ASSERT_FALSE(names.empty());
    EXPECT_EQ(names.front(), "book_imbalance_1");
    EXPECT_EQ(names.back(), "cancel_concentration");
}

TEST_F(ColumnOrderTest, DepthProfileColumnsInOrder) {
    auto names = BarFeatureRow::feature_names();
    // Find bid_depth_profile_0 and verify 0-9 are contiguous
    auto it0 = std::find(names.begin(), names.end(), "bid_depth_profile_0");
    ASSERT_NE(it0, names.end());
    size_t idx0 = std::distance(names.begin(), it0);
    for (int i = 0; i < 10; ++i) {
        std::string expected = "bid_depth_profile_" + std::to_string(i);
        EXPECT_EQ(names[idx0 + i], expected)
            << "bid_depth_profile columns must be contiguous and ordered 0-9";
    }
}

TEST_F(ColumnOrderTest, AskDepthProfileFollowsBid) {
    auto names = BarFeatureRow::feature_names();
    auto bid_it = std::find(names.begin(), names.end(), "bid_depth_profile_9");
    auto ask_it = std::find(names.begin(), names.end(), "ask_depth_profile_0");
    ASSERT_NE(bid_it, names.end());
    ASSERT_NE(ask_it, names.end());
    EXPECT_EQ(std::distance(bid_it, ask_it), 1)
        << "ask_depth_profile_0 must immediately follow bid_depth_profile_9";
}

// ===========================================================================
// Data Integrity — verify CSV rows contain numeric data in expected columns
// ===========================================================================

class DataIntegrityTest : public ::testing::Test {
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

TEST_F(DataIntegrityTest, TimestampIsPositiveInteger) {
    auto csv = temp_csv_path("_ts_check");
    track_temp(csv);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5.0 --output " + csv);

    if (result.exit_code != 0) {
        GTEST_SKIP() << "Tool exited non-zero";
    }

    auto lines = read_all_lines(csv);
    if (lines.size() < 2) {
        GTEST_SKIP() << "No data rows produced";
    }

    for (size_t i = 1; i < std::min(lines.size(), size_t(10)); ++i) {
        auto cols = parse_csv_header(lines[i]);
        ASSERT_FALSE(cols.empty());
        uint64_t ts = std::stoull(cols[0]);
        EXPECT_GT(ts, 0u)
            << "Timestamp must be a positive integer (nanoseconds since epoch)";
        // Sanity: timestamps should be in 2022 range
        // 2022-01-01 UTC ~= 1640995200000000000 ns
        EXPECT_GT(ts, 1640000000000000000ULL) << "Timestamp too small for 2022";
        EXPECT_LT(ts, 1700000000000000000ULL) << "Timestamp too large for 2022";
    }
}

TEST_F(DataIntegrityTest, DayFieldIsValidDate) {
    auto csv = temp_csv_path("_day_check");
    track_temp(csv);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5.0 --output " + csv);

    if (result.exit_code != 0) {
        GTEST_SKIP() << "Tool exited non-zero";
    }

    auto lines = read_all_lines(csv);
    if (lines.size() < 2) {
        GTEST_SKIP() << "No data rows produced";
    }

    // day is column index 3
    for (size_t i = 1; i < std::min(lines.size(), size_t(10)); ++i) {
        auto cols = parse_csv_header(lines[i]);
        ASSERT_GE(cols.size(), 4u);
        int day = std::stoi(cols[3]);
        EXPECT_GE(day, 20220101) << "Day must be a valid 2022 date";
        EXPECT_LE(day, 20221231) << "Day must be a valid 2022 date";
    }
}

TEST_F(DataIntegrityTest, MboEventCountIsNonNegative) {
    auto csv = temp_csv_path("_evt_check");
    track_temp(csv);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5.0 --output " + csv);

    if (result.exit_code != 0) {
        GTEST_SKIP() << "Tool exited non-zero";
    }

    auto lines = read_all_lines(csv);
    if (lines.size() < 2) {
        GTEST_SKIP() << "No data rows produced";
    }

    // mbo_event_count is at index 145, followed by 3 TB label columns (149 total)
    for (size_t i = 1; i < std::min(lines.size(), size_t(10)); ++i) {
        auto cols = parse_csv_header(lines[i]);
        ASSERT_EQ(cols.size(), 149u) << "Row " << i << " should have 149 columns";
        int event_count = std::stoi(cols[145]);
        EXPECT_GE(event_count, 0)
            << "mbo_event_count must be non-negative at row " << i;
    }
}
