// bidirectional_export_test.cpp — TDD RED phase tests for bidirectional label wiring
// Spec: .kit/docs/bidirectional-export-wiring.md
//
// Tests that bar_feature_export:
//   (1) defaults to bidirectional labels with 152-column Parquet output
//   (2) adds 3 new columns: tb_both_triggered, tb_long_triggered, tb_short_triggered
//   (3) supports --legacy-labels flag for old-style 149-column output
//   (4) preserves existing feature columns unchanged between modes
//   (5) shifts label distribution (more 0s, fewer -1s) under bidirectional
//   (6) maintains correct column ordering
//
// Unit tests verify schema contracts. Integration tests require the binary + data.
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
#include <set>
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

const std::string DATA_DIR = "DATA/GLBX-20260207-L953CAPU5B";

std::string temp_parquet_path(const std::string& suffix = "") {
    return (std::filesystem::temp_directory_path() /
            ("bidir_export_test" + suffix + ".parquet")).string();
}

// Build the expected 152-column schema for bidirectional mode.
// 149 original + 3 new diagnostic columns after tb_bars_held.
std::vector<std::string> expected_bidirectional_columns() {
    std::vector<std::string> cols;

    // Metadata (6)
    cols.push_back("timestamp");
    cols.push_back("bar_type");
    cols.push_back("bar_param");
    cols.push_back("day");
    cols.push_back("is_warmup");
    cols.push_back("bar_index");

    // Track A features (62)
    auto feature_names = BarFeatureRow::feature_names();
    cols.insert(cols.end(), feature_names.begin(), feature_names.end());

    // Book snapshot (40)
    for (int i = 0; i < 40; ++i)
        cols.push_back("book_snap_" + std::to_string(i));

    // Message summary (33)
    for (size_t i = 0; i < MessageSummary::SUMMARY_SIZE; ++i)
        cols.push_back("msg_summary_" + std::to_string(i));

    // Forward returns (4) — Parquet uses fwd_ prefix
    cols.push_back("fwd_return_1");
    cols.push_back("fwd_return_5");
    cols.push_back("fwd_return_20");
    cols.push_back("fwd_return_100");

    // Event count (1)
    cols.push_back("mbo_event_count");

    // Triple barrier labels (3 original)
    cols.push_back("tb_label");
    cols.push_back("tb_exit_type");
    cols.push_back("tb_bars_held");

    // Bidirectional diagnostic columns (3 new)
    cols.push_back("tb_both_triggered");
    cols.push_back("tb_long_triggered");
    cols.push_back("tb_short_triggered");

    return cols;
}

// Build the expected 149-column schema for legacy mode (no new columns).
std::vector<std::string> expected_legacy_columns() {
    std::vector<std::string> cols;

    cols.push_back("timestamp");
    cols.push_back("bar_type");
    cols.push_back("bar_param");
    cols.push_back("day");
    cols.push_back("is_warmup");
    cols.push_back("bar_index");

    auto feature_names = BarFeatureRow::feature_names();
    cols.insert(cols.end(), feature_names.begin(), feature_names.end());

    for (int i = 0; i < 40; ++i)
        cols.push_back("book_snap_" + std::to_string(i));

    for (size_t i = 0; i < MessageSummary::SUMMARY_SIZE; ++i)
        cols.push_back("msg_summary_" + std::to_string(i));

    // Forward returns (4) — Parquet uses fwd_ prefix
    cols.push_back("fwd_return_1");
    cols.push_back("fwd_return_5");
    cols.push_back("fwd_return_20");
    cols.push_back("fwd_return_100");

    cols.push_back("mbo_event_count");

    cols.push_back("tb_label");
    cols.push_back("tb_exit_type");
    cols.push_back("tb_bars_held");

    return cols;
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

// Safely get column index with ASSERT — aborts test if column not found.
// Use via the REQUIRE_COLUMN macro.
#define REQUIRE_COLUMN(table, name, idx_var) \
    int idx_var = (table)->schema()->GetFieldIndex(name); \
    ASSERT_NE(idx_var, -1) << "Column '" << name << "' not found in schema"

bool data_available() {
    return std::filesystem::exists(DATA_DIR) &&
           std::filesystem::is_directory(DATA_DIR);
}

}  // anonymous namespace

// ===========================================================================
// Schema Contract Tests (unit-level — no binary or data required)
// ===========================================================================

class BidirectionalSchemaContractTest : public ::testing::Test {};

TEST_F(BidirectionalSchemaContractTest, BidirectionalSchemaHas152Columns) {
    // Spec: "Total columns: 149 → 152."
    auto cols = expected_bidirectional_columns();
    EXPECT_EQ(cols.size(), 152u)
        << "Bidirectional schema must have 152 columns "
        << "(6 meta + 62 Track A + 40 book_snap + 33 msg_summary "
        << "+ 4 returns + 1 event_count + 3 tb_labels + 3 bidir diagnostics)";
}

TEST_F(BidirectionalSchemaContractTest, LegacySchemaHas149Columns) {
    // Legacy mode must produce the original 149-column schema
    auto cols = expected_legacy_columns();
    EXPECT_EQ(cols.size(), 149u)
        << "Legacy schema must have exactly 149 columns";
}

TEST_F(BidirectionalSchemaContractTest, NewColumnsAreNamedCorrectly) {
    // Spec: exact column names: tb_both_triggered, tb_long_triggered, tb_short_triggered
    auto cols = expected_bidirectional_columns();
    ASSERT_EQ(cols.size(), 152u);
    EXPECT_EQ(cols[149], "tb_both_triggered");
    EXPECT_EQ(cols[150], "tb_long_triggered");
    EXPECT_EQ(cols[151], "tb_short_triggered");
}

TEST_F(BidirectionalSchemaContractTest, NewColumnsAppearAfterTbBarsHeld) {
    // Spec T6: "the 3 new columns appear AFTER tb_bars_held"
    auto cols = expected_bidirectional_columns();
    auto it_bars_held = std::find(cols.begin(), cols.end(), "tb_bars_held");
    ASSERT_NE(it_bars_held, cols.end()) << "tb_bars_held must exist in schema";

    auto idx_bars_held = std::distance(cols.begin(), it_bars_held);
    ASSERT_GE(static_cast<int>(cols.size()), idx_bars_held + 4)
        << "Must have 3 columns after tb_bars_held";

    EXPECT_EQ(cols[idx_bars_held + 1], "tb_both_triggered");
    EXPECT_EQ(cols[idx_bars_held + 2], "tb_long_triggered");
    EXPECT_EQ(cols[idx_bars_held + 3], "tb_short_triggered");
}

TEST_F(BidirectionalSchemaContractTest, First149ColumnsIdenticalBetweenModes) {
    // The first 149 columns must be identical between bidirectional and legacy schemas
    auto bidir_cols = expected_bidirectional_columns();
    auto legacy_cols = expected_legacy_columns();

    ASSERT_GE(bidir_cols.size(), 149u);
    ASSERT_EQ(legacy_cols.size(), 149u);

    for (size_t i = 0; i < 149; ++i) {
        EXPECT_EQ(bidir_cols[i], legacy_cols[i])
            << "Column " << i << " differs: bidir='" << bidir_cols[i]
            << "' legacy='" << legacy_cols[i] << "'";
    }
}

TEST_F(BidirectionalSchemaContractTest, LegacySchemaHasNoNewColumns) {
    auto cols = expected_legacy_columns();
    for (const auto& col : cols) {
        EXPECT_NE(col, "tb_both_triggered")
            << "Legacy schema must not contain tb_both_triggered";
        EXPECT_NE(col, "tb_long_triggered")
            << "Legacy schema must not contain tb_long_triggered";
        EXPECT_NE(col, "tb_short_triggered")
            << "Legacy schema must not contain tb_short_triggered";
    }
}

// ===========================================================================
// CLI Flag Tests (require binary, not data)
// ===========================================================================

class BidirectionalCLITest : public ::testing::Test {
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

TEST_F(BidirectionalCLITest, LegacyLabelsFlagAccepted) {
    // Spec: "--legacy-labels flag should be recognized by the CLI parser"
    // The flag should not produce an "unknown argument" error.
    auto pq = temp_parquet_path("_cli_legacy");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --legacy-labels --output " + pq);

    // The tool may fail due to missing data, but should NOT fail on argument parsing.
    bool arg_error = (result.output.find("Unknown") != std::string::npos ||
                      result.output.find("unknown") != std::string::npos ||
                      result.output.find("unrecognized") != std::string::npos ||
                      result.output.find("Unrecognized") != std::string::npos ||
                      result.output.find("invalid option") != std::string::npos);
    EXPECT_FALSE(arg_error)
        << "--legacy-labels should be a recognized CLI flag. Got: " << result.output;
}

TEST_F(BidirectionalCLITest, LegacyLabelsFlagWithMissingRequiredArgsStillFails) {
    // --legacy-labels alone (without --bar-type, --bar-param, --output) should fail.
    auto result = run_command(BINARY_PATH + " --legacy-labels");
    EXPECT_NE(result.exit_code, 0)
        << "--legacy-labels without required args should still fail";
}

TEST_F(BidirectionalCLITest, DefaultModeNoLegacyFlagNeeded) {
    // Running without --legacy-labels should work (bidirectional is default).
    // Just verify no arg parsing errors; may still fail on data.
    auto pq = temp_parquet_path("_cli_default");
    track_temp(pq);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq);

    bool arg_error = (result.output.find("Unknown") != std::string::npos ||
                      result.output.find("unrecognized") != std::string::npos);
    EXPECT_FALSE(arg_error)
        << "Default mode should not require any extra flag. Got: " << result.output;
}

// ===========================================================================
// Integration test fixture — requires binary + data
// ===========================================================================

class BidirectionalExportTestBase : public ::testing::Test {
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
// T1: Default Export Uses Bidirectional Labels
// ===========================================================================

class T1_DefaultBidirectionalTest : public BidirectionalExportTestBase {};

TEST_F(T1_DefaultBidirectionalTest, ParquetOutputHas152Columns) {
    // Spec T1: "Run bar_feature_export on a small test dataset with no flags"
    // "Verify Parquet output has 152 columns (149 + 3 new)"
    auto pq = temp_parquet_path("_t1_colcount");
    track_temp(pq);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq);
    ASSERT_EQ(result.exit_code, 0) << "Export failed: " << result.output;

    auto table = read_parquet_table(pq);
    ASSERT_NE(table, nullptr) << "Cannot read Parquet file";

    EXPECT_EQ(table->num_columns(), 152)
        << "Default export (bidirectional) must produce 152 columns. "
        << "Got " << table->num_columns();
}

TEST_F(T1_DefaultBidirectionalTest, NewDiagnosticColumnsExist) {
    // Spec T1: "Verify tb_both_triggered, tb_long_triggered, tb_short_triggered columns exist"
    auto pq = temp_parquet_path("_t1_newcols");
    track_temp(pq);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq);
    ASSERT_EQ(result.exit_code, 0) << "Export failed: " << result.output;

    auto table = read_parquet_table(pq);
    ASSERT_NE(table, nullptr);

    auto schema = table->schema();
    EXPECT_NE(schema->GetFieldIndex("tb_both_triggered"), -1)
        << "Column tb_both_triggered must exist in Parquet output";
    EXPECT_NE(schema->GetFieldIndex("tb_long_triggered"), -1)
        << "Column tb_long_triggered must exist in Parquet output";
    EXPECT_NE(schema->GetFieldIndex("tb_short_triggered"), -1)
        << "Column tb_short_triggered must exist in Parquet output";
}

TEST_F(T1_DefaultBidirectionalTest, TbLabelValuesInValidRange) {
    // Spec T1: "Verify tb_label values are in {-1.0, 0.0, 1.0}"
    auto pq = temp_parquet_path("_t1_labelvals");
    track_temp(pq);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq);
    ASSERT_EQ(result.exit_code, 0) << "Export failed: " << result.output;

    auto table = read_parquet_table(pq);
    ASSERT_NE(table, nullptr);
    ASSERT_GT(table->num_rows(), 0);

    int label_idx = table->schema()->GetFieldIndex("tb_label");
    ASSERT_NE(label_idx, -1);

    auto labels = extract_doubles(table->column(label_idx));
    ASSERT_FALSE(labels.empty());

    for (size_t i = 0; i < labels.size(); ++i) {
        EXPECT_TRUE(labels[i] == -1.0 || labels[i] == 0.0 || labels[i] == 1.0)
            << "tb_label at row " << i << " is " << labels[i]
            << " — must be -1.0, 0.0, or 1.0";
    }
}

TEST_F(T1_DefaultBidirectionalTest, NewColumnsAreFloat64) {
    // Spec: new columns are float64 (0.0 or 1.0)
    auto pq = temp_parquet_path("_t1_coltypes");
    track_temp(pq);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq);
    ASSERT_EQ(result.exit_code, 0) << "Export failed: " << result.output;

    auto table = read_parquet_table(pq);
    ASSERT_NE(table, nullptr);

    auto schema = table->schema();

    for (const auto& col_name : {"tb_both_triggered", "tb_long_triggered", "tb_short_triggered"}) {
        int idx = schema->GetFieldIndex(col_name);
        ASSERT_NE(idx, -1) << "Column " << col_name << " not found";
        EXPECT_EQ(schema->field(idx)->type()->id(), arrow::Type::DOUBLE)
            << "Column " << col_name << " must be DOUBLE (float64), got "
            << schema->field(idx)->type()->ToString();
    }
}

// ===========================================================================
// T2: Legacy Flag Produces Old-Style Labels
// ===========================================================================

class T2_LegacyModeTest : public BidirectionalExportTestBase {};

TEST_F(T2_LegacyModeTest, LegacyOutputHas149Columns) {
    // Spec T2: "Run bar_feature_export --legacy-labels ... Verify Parquet output has 149 columns"
    auto pq = temp_parquet_path("_t2_legacy_count");
    track_temp(pq);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --legacy-labels --output " + pq);
    ASSERT_EQ(result.exit_code, 0) << "Legacy export failed: " << result.output;

    auto table = read_parquet_table(pq);
    ASSERT_NE(table, nullptr);

    EXPECT_EQ(table->num_columns(), 149)
        << "--legacy-labels must produce exactly 149 columns (no new columns). "
        << "Got " << table->num_columns();
}

TEST_F(T2_LegacyModeTest, LegacyOutputHasNoNewColumns) {
    // Spec T2: "Verify Parquet output has 149 columns (no new columns)"
    auto pq = temp_parquet_path("_t2_legacy_nocols");
    track_temp(pq);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --legacy-labels --output " + pq);
    ASSERT_EQ(result.exit_code, 0) << "Legacy export failed: " << result.output;

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

TEST_F(T2_LegacyModeTest, LegacyColumnNamesMatchOldSchema) {
    // The 149 columns must match the original schema exactly
    auto pq = temp_parquet_path("_t2_legacy_names");
    track_temp(pq);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --legacy-labels --output " + pq);
    ASSERT_EQ(result.exit_code, 0) << "Legacy export failed: " << result.output;

    auto table = read_parquet_table(pq);
    ASSERT_NE(table, nullptr);

    auto expected = expected_legacy_columns();
    ASSERT_EQ(static_cast<size_t>(table->num_columns()), expected.size());

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(table->field(i)->name(), expected[i])
            << "Legacy column " << i << " name mismatch: got '"
            << table->field(i)->name() << "' expected '" << expected[i] << "'";
    }
}

// ===========================================================================
// T3: New Columns Have Correct Values
// ===========================================================================

class T3_NewColumnValuesTest : public BidirectionalExportTestBase {};

TEST_F(T3_NewColumnValuesTest, DiagnosticColumnsOnlyZeroOrOne) {
    // All diagnostic columns must contain only 0.0 or 1.0
    auto pq = temp_parquet_path("_t3_binary");
    track_temp(pq);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq);
    ASSERT_EQ(result.exit_code, 0) << "Export failed: " << result.output;

    auto table = read_parquet_table(pq);
    ASSERT_NE(table, nullptr);
    ASSERT_GT(table->num_rows(), 0);

    for (const auto& col_name : {"tb_both_triggered", "tb_long_triggered", "tb_short_triggered"}) {
        int idx = table->schema()->GetFieldIndex(col_name);
        ASSERT_NE(idx, -1) << "Column " << col_name << " not found";

        auto values = extract_doubles(table->column(idx));
        for (size_t i = 0; i < values.size(); ++i) {
            EXPECT_TRUE(values[i] == 0.0 || values[i] == 1.0)
                << col_name << " at row " << i << " is " << values[i]
                << " — must be exactly 0.0 or 1.0";
        }
    }
}

TEST_F(T3_NewColumnValuesTest, LongTriggeredConsistentWithLabelPlusOne) {
    // Spec T3: "A bar where only the long race triggers → tb_long_triggered=1,
    //           tb_short_triggered=0, tb_both_triggered=0, tb_label=1"
    // For all rows where tb_label == 1: tb_long_triggered must be 1
    auto pq = temp_parquet_path("_t3_long");
    track_temp(pq);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq);
    ASSERT_EQ(result.exit_code, 0) << "Export failed: " << result.output;

    auto table = read_parquet_table(pq);
    ASSERT_NE(table, nullptr);

    REQUIRE_COLUMN(table, "tb_label", label_idx);
    REQUIRE_COLUMN(table, "tb_long_triggered", long_idx);
    REQUIRE_COLUMN(table, "tb_short_triggered", short_idx);
    REQUIRE_COLUMN(table, "tb_both_triggered", both_idx);

    auto labels = extract_doubles(table->column(label_idx));
    auto long_t = extract_doubles(table->column(long_idx));
    auto short_t = extract_doubles(table->column(short_idx));
    auto both_t = extract_doubles(table->column(both_idx));

    ASSERT_EQ(labels.size(), long_t.size());

    int label_plus_count = 0;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] == 1.0) {
            label_plus_count++;
            EXPECT_EQ(long_t[i], 1.0)
                << "Row " << i << ": tb_label=+1 requires tb_long_triggered=1";
            EXPECT_EQ(short_t[i], 0.0)
                << "Row " << i << ": tb_label=+1 requires tb_short_triggered=0";
            EXPECT_EQ(both_t[i], 0.0)
                << "Row " << i << ": tb_label=+1 requires tb_both_triggered=0";
        }
    }

    EXPECT_GT(label_plus_count, 0)
        << "Must have at least one tb_label=+1 row to validate the invariant";
}

TEST_F(T3_NewColumnValuesTest, ShortTriggeredConsistentWithLabelMinusOne) {
    // Spec T3: "A bar where only the short race triggers → tb_long_triggered=0,
    //           tb_short_triggered=1, tb_both_triggered=0, tb_label=-1"
    auto pq = temp_parquet_path("_t3_short");
    track_temp(pq);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq);
    ASSERT_EQ(result.exit_code, 0) << "Export failed: " << result.output;

    auto table = read_parquet_table(pq);
    ASSERT_NE(table, nullptr);

    REQUIRE_COLUMN(table, "tb_label", label_idx);
    REQUIRE_COLUMN(table, "tb_long_triggered", long_idx);
    REQUIRE_COLUMN(table, "tb_short_triggered", short_idx);
    REQUIRE_COLUMN(table, "tb_both_triggered", both_idx);

    auto labels = extract_doubles(table->column(label_idx));
    auto long_t = extract_doubles(table->column(long_idx));
    auto short_t = extract_doubles(table->column(short_idx));
    auto both_t = extract_doubles(table->column(both_idx));

    int label_minus_count = 0;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] == -1.0) {
            label_minus_count++;
            EXPECT_EQ(long_t[i], 0.0)
                << "Row " << i << ": tb_label=-1 requires tb_long_triggered=0";
            EXPECT_EQ(short_t[i], 1.0)
                << "Row " << i << ": tb_label=-1 requires tb_short_triggered=1";
            EXPECT_EQ(both_t[i], 0.0)
                << "Row " << i << ": tb_label=-1 requires tb_both_triggered=0";
        }
    }

    EXPECT_GT(label_minus_count, 0)
        << "Must have at least one tb_label=-1 row to validate the invariant";
}

TEST_F(T3_NewColumnValuesTest, BothTriggeredImpliesLabelZero) {
    // Spec T3: "A bar where both trigger → tb_long_triggered=1,
    //           tb_short_triggered=1, tb_both_triggered=1, tb_label=0"
    auto pq = temp_parquet_path("_t3_both");
    track_temp(pq);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq);
    ASSERT_EQ(result.exit_code, 0) << "Export failed: " << result.output;

    auto table = read_parquet_table(pq);
    ASSERT_NE(table, nullptr);

    REQUIRE_COLUMN(table, "tb_label", label_idx);
    REQUIRE_COLUMN(table, "tb_long_triggered", long_idx);
    REQUIRE_COLUMN(table, "tb_short_triggered", short_idx);
    REQUIRE_COLUMN(table, "tb_both_triggered", both_idx);

    auto labels = extract_doubles(table->column(label_idx));
    auto long_t = extract_doubles(table->column(long_idx));
    auto short_t = extract_doubles(table->column(short_idx));
    auto both_t = extract_doubles(table->column(both_idx));

    for (size_t i = 0; i < labels.size(); ++i) {
        if (both_t[i] == 1.0) {
            EXPECT_EQ(labels[i], 0.0)
                << "Row " << i << ": tb_both_triggered=1 requires tb_label=0";
            EXPECT_EQ(long_t[i], 1.0)
                << "Row " << i << ": tb_both_triggered=1 requires tb_long_triggered=1";
            EXPECT_EQ(short_t[i], 1.0)
                << "Row " << i << ": tb_both_triggered=1 requires tb_short_triggered=1";
        }
    }
}

TEST_F(T3_NewColumnValuesTest, NeitherTriggeredImpliesAllZero) {
    // Spec T3: "A bar where neither triggers → all three = 0, tb_label=0"
    auto pq = temp_parquet_path("_t3_neither");
    track_temp(pq);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq);
    ASSERT_EQ(result.exit_code, 0) << "Export failed: " << result.output;

    auto table = read_parquet_table(pq);
    ASSERT_NE(table, nullptr);

    REQUIRE_COLUMN(table, "tb_label", label_idx);
    REQUIRE_COLUMN(table, "tb_long_triggered", long_idx);
    REQUIRE_COLUMN(table, "tb_short_triggered", short_idx);
    REQUIRE_COLUMN(table, "tb_both_triggered", both_idx);

    auto labels = extract_doubles(table->column(label_idx));
    auto long_t = extract_doubles(table->column(long_idx));
    auto short_t = extract_doubles(table->column(short_idx));
    auto both_t = extract_doubles(table->column(both_idx));

    int neither_count = 0;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (long_t[i] == 0.0 && short_t[i] == 0.0) {
            neither_count++;
            EXPECT_EQ(both_t[i], 0.0)
                << "Row " << i << ": neither triggered but tb_both_triggered != 0";
            EXPECT_EQ(labels[i], 0.0)
                << "Row " << i << ": neither triggered requires tb_label=0";
        }
    }

    // There should be at least some bars where neither triggers
    EXPECT_GT(neither_count, 0)
        << "Expected some bars where neither long nor short triggers";
}

TEST_F(T3_NewColumnValuesTest, BothTriggeredConsistencyCheck) {
    // tb_both_triggered must equal (tb_long_triggered AND tb_short_triggered)
    auto pq = temp_parquet_path("_t3_consistency");
    track_temp(pq);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq);
    ASSERT_EQ(result.exit_code, 0) << "Export failed: " << result.output;

    auto table = read_parquet_table(pq);
    ASSERT_NE(table, nullptr);

    REQUIRE_COLUMN(table, "tb_long_triggered", long_idx);
    REQUIRE_COLUMN(table, "tb_short_triggered", short_idx);
    REQUIRE_COLUMN(table, "tb_both_triggered", both_idx);

    auto long_t = extract_doubles(table->column(long_idx));
    auto short_t = extract_doubles(table->column(short_idx));
    auto both_t = extract_doubles(table->column(both_idx));

    for (size_t i = 0; i < both_t.size(); ++i) {
        double expected_both = (long_t[i] == 1.0 && short_t[i] == 1.0) ? 1.0 : 0.0;
        EXPECT_EQ(both_t[i], expected_both)
            << "Row " << i << ": tb_both_triggered must equal "
            << "(tb_long_triggered AND tb_short_triggered). "
            << "long=" << long_t[i] << " short=" << short_t[i]
            << " both=" << both_t[i];
    }
}

// ===========================================================================
// T4: Label Distribution Shift Under Bidirectional
// ===========================================================================

class T4_LabelDistributionTest : public BidirectionalExportTestBase {};

TEST_F(T4_LabelDistributionTest, BidirectionalProducesMoreZerosThanLegacy) {
    // Spec T4: "Verify bidirectional mode produces MORE tb_label=0 bars than legacy mode"
    auto pq_bidir = temp_parquet_path("_t4_bidir");
    auto pq_legacy = temp_parquet_path("_t4_legacy");
    track_temp(pq_bidir);
    track_temp(pq_legacy);

    auto r_bidir = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_bidir);
    ASSERT_EQ(r_bidir.exit_code, 0) << "Bidirectional export failed: " << r_bidir.output;

    auto r_legacy = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --legacy-labels --output " + pq_legacy);
    ASSERT_EQ(r_legacy.exit_code, 0) << "Legacy export failed: " << r_legacy.output;

    auto t_bidir = read_parquet_table(pq_bidir);
    auto t_legacy = read_parquet_table(pq_legacy);
    ASSERT_NE(t_bidir, nullptr);
    ASSERT_NE(t_legacy, nullptr);

    // Both should have the same number of rows (same data, different labeling)
    ASSERT_EQ(t_bidir->num_rows(), t_legacy->num_rows())
        << "Both modes must produce the same row count on the same data";

    REQUIRE_COLUMN(t_bidir, "tb_label", bidir_label_idx);
    REQUIRE_COLUMN(t_legacy, "tb_label", legacy_label_idx);

    auto bidir_labels = extract_doubles(t_bidir->column(bidir_label_idx));
    auto legacy_labels = extract_doubles(t_legacy->column(legacy_label_idx));

    int bidir_zeros = std::count(bidir_labels.begin(), bidir_labels.end(), 0.0);
    int legacy_zeros = std::count(legacy_labels.begin(), legacy_labels.end(), 0.0);

    EXPECT_GT(bidir_zeros, legacy_zeros)
        << "Bidirectional mode must produce MORE tb_label=0 bars than legacy. "
        << "bidir_zeros=" << bidir_zeros << " legacy_zeros=" << legacy_zeros;
}

TEST_F(T4_LabelDistributionTest, BidirectionalProducesFewerNegativesThanLegacy) {
    // Spec T4: "Verify bidirectional mode produces FEWER tb_label=-1 bars than legacy mode"
    auto pq_bidir = temp_parquet_path("_t4b_bidir");
    auto pq_legacy = temp_parquet_path("_t4b_legacy");
    track_temp(pq_bidir);
    track_temp(pq_legacy);

    auto r_bidir = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_bidir);
    ASSERT_EQ(r_bidir.exit_code, 0) << "Bidirectional export failed: " << r_bidir.output;

    auto r_legacy = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --legacy-labels --output " + pq_legacy);
    ASSERT_EQ(r_legacy.exit_code, 0) << "Legacy export failed: " << r_legacy.output;

    auto t_bidir = read_parquet_table(pq_bidir);
    auto t_legacy = read_parquet_table(pq_legacy);
    ASSERT_NE(t_bidir, nullptr);
    ASSERT_NE(t_legacy, nullptr);

    REQUIRE_COLUMN(t_bidir, "tb_label", bidir_label_idx);
    REQUIRE_COLUMN(t_legacy, "tb_label", legacy_label_idx);

    auto bidir_labels = extract_doubles(t_bidir->column(bidir_label_idx));
    auto legacy_labels = extract_doubles(t_legacy->column(legacy_label_idx));

    int bidir_neg = std::count(bidir_labels.begin(), bidir_labels.end(), -1.0);
    int legacy_neg = std::count(legacy_labels.begin(), legacy_labels.end(), -1.0);

    EXPECT_LT(bidir_neg, legacy_neg)
        << "Bidirectional mode must produce FEWER tb_label=-1 bars than legacy. "
        << "bidir_neg=" << bidir_neg << " legacy_neg=" << legacy_neg;
}

TEST_F(T4_LabelDistributionTest, AllThreeClassesPresentInBidirectionalMode) {
    // Sanity: all three label classes must exist in bidirectional output
    auto pq = temp_parquet_path("_t4_allclasses");
    track_temp(pq);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq);
    ASSERT_EQ(result.exit_code, 0) << "Export failed: " << result.output;

    auto table = read_parquet_table(pq);
    ASSERT_NE(table, nullptr);

    REQUIRE_COLUMN(table, "tb_label", label_idx);
    auto labels = extract_doubles(table->column(label_idx));

    std::set<double> observed(labels.begin(), labels.end());
    EXPECT_TRUE(observed.count(-1.0) > 0) << "Label -1 must be present";
    EXPECT_TRUE(observed.count(0.0) > 0)  << "Label 0 must be present";
    EXPECT_TRUE(observed.count(1.0) > 0)  << "Label +1 must be present";
}

// ===========================================================================
// T5: No Regression on Existing Columns
// ===========================================================================

class T5_NoRegressionTest : public BidirectionalExportTestBase {};

TEST_F(T5_NoRegressionTest, First149ColumnNamesUnchanged) {
    // Spec T5: "Verify all 149 original columns (features, metadata) are unchanged"
    auto pq = temp_parquet_path("_t5_names");
    track_temp(pq);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq);
    ASSERT_EQ(result.exit_code, 0) << "Export failed: " << result.output;

    auto table = read_parquet_table(pq);
    ASSERT_NE(table, nullptr);
    ASSERT_GE(table->num_columns(), 149)
        << "Bidirectional output must have at least 149 columns";

    auto expected = expected_legacy_columns();
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(table->field(i)->name(), expected[i])
            << "Column " << i << " name changed from '" << expected[i]
            << "' to '" << table->field(i)->name() << "'";
    }
}

TEST_F(T5_NoRegressionTest, FeatureValuesIdenticalBetweenModes) {
    // Spec T5: "close_mid, volatility_50, spread, time_sin columns contain same values
    //           regardless of label mode"
    auto pq_bidir = temp_parquet_path("_t5_bidir");
    auto pq_legacy = temp_parquet_path("_t5_legacy");
    track_temp(pq_bidir);
    track_temp(pq_legacy);

    auto r_bidir = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_bidir);
    ASSERT_EQ(r_bidir.exit_code, 0) << "Bidirectional export failed: " << r_bidir.output;

    auto r_legacy = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --legacy-labels --output " + pq_legacy);
    ASSERT_EQ(r_legacy.exit_code, 0) << "Legacy export failed: " << r_legacy.output;

    auto t_bidir = read_parquet_table(pq_bidir);
    auto t_legacy = read_parquet_table(pq_legacy);
    ASSERT_NE(t_bidir, nullptr);
    ASSERT_NE(t_legacy, nullptr);
    ASSERT_EQ(t_bidir->num_rows(), t_legacy->num_rows());

    // Check specific feature columns that should be identical
    // (close_position = Track A feature; volatility_50, spread, time_sin likewise)
    std::vector<std::string> check_cols = {
        "close_position", "volatility_50", "spread", "time_sin"
    };

    for (const auto& col_name : check_cols) {
        int bidir_idx = t_bidir->schema()->GetFieldIndex(col_name);
        int legacy_idx = t_legacy->schema()->GetFieldIndex(col_name);
        ASSERT_NE(bidir_idx, -1) << "Column " << col_name << " not found in bidirectional output";
        ASSERT_NE(legacy_idx, -1) << "Column " << col_name << " not found in legacy output";

        auto bidir_vals = extract_doubles(t_bidir->column(bidir_idx));
        auto legacy_vals = extract_doubles(t_legacy->column(legacy_idx));
        ASSERT_EQ(bidir_vals.size(), legacy_vals.size());

        for (size_t i = 0; i < bidir_vals.size(); ++i) {
            if (std::isnan(bidir_vals[i]) && std::isnan(legacy_vals[i])) continue;
            EXPECT_EQ(bidir_vals[i], legacy_vals[i])
                << "Column " << col_name << " differs at row " << i
                << ": bidir=" << bidir_vals[i] << " legacy=" << legacy_vals[i];
        }
    }
}

TEST_F(T5_NoRegressionTest, ForwardReturnsIdenticalBetweenModes) {
    // Forward returns must be identical regardless of label mode
    auto pq_bidir = temp_parquet_path("_t5r_bidir");
    auto pq_legacy = temp_parquet_path("_t5r_legacy");
    track_temp(pq_bidir);
    track_temp(pq_legacy);

    auto r_bidir = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_bidir);
    ASSERT_EQ(r_bidir.exit_code, 0);

    auto r_legacy = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --legacy-labels --output " + pq_legacy);
    ASSERT_EQ(r_legacy.exit_code, 0);

    auto t_bidir = read_parquet_table(pq_bidir);
    auto t_legacy = read_parquet_table(pq_legacy);
    ASSERT_NE(t_bidir, nullptr);
    ASSERT_NE(t_legacy, nullptr);

    for (const auto& col_name : {"fwd_return_1", "fwd_return_5", "fwd_return_20", "fwd_return_100"}) {
        auto bidir_vals = extract_doubles(
            t_bidir->column(t_bidir->schema()->GetFieldIndex(col_name)));
        auto legacy_vals = extract_doubles(
            t_legacy->column(t_legacy->schema()->GetFieldIndex(col_name)));
        ASSERT_EQ(bidir_vals.size(), legacy_vals.size());

        for (size_t i = 0; i < bidir_vals.size(); ++i) {
            if (std::isnan(bidir_vals[i]) && std::isnan(legacy_vals[i])) continue;
            EXPECT_EQ(bidir_vals[i], legacy_vals[i])
                << col_name << " differs at row " << i;
        }
    }
}

TEST_F(T5_NoRegressionTest, TimestampsIdenticalBetweenModes) {
    // Timestamps must be identical regardless of label mode
    auto pq_bidir = temp_parquet_path("_t5ts_bidir");
    auto pq_legacy = temp_parquet_path("_t5ts_legacy");
    track_temp(pq_bidir);
    track_temp(pq_legacy);

    auto r_bidir = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_bidir);
    ASSERT_EQ(r_bidir.exit_code, 0);

    auto r_legacy = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --legacy-labels --output " + pq_legacy);
    ASSERT_EQ(r_legacy.exit_code, 0);

    auto t_bidir = read_parquet_table(pq_bidir);
    auto t_legacy = read_parquet_table(pq_legacy);
    ASSERT_NE(t_bidir, nullptr);
    ASSERT_NE(t_legacy, nullptr);
    ASSERT_EQ(t_bidir->num_rows(), t_legacy->num_rows());

    // Compare timestamp columns (first column)
    int ts_idx = 0;
    auto bidir_ts = t_bidir->column(ts_idx);
    auto legacy_ts = t_legacy->column(ts_idx);

    for (int chunk = 0; chunk < bidir_ts->num_chunks(); ++chunk) {
        EXPECT_TRUE(bidir_ts->chunk(chunk)->Equals(legacy_ts->chunk(chunk)))
            << "Timestamp chunk " << chunk << " differs between modes";
    }
}

// ===========================================================================
// T6: Schema Column Order
// ===========================================================================

class T6_SchemaColumnOrderTest : public BidirectionalExportTestBase {};

TEST_F(T6_SchemaColumnOrderTest, NewColumnsAfterTbBarsHeld) {
    // Spec T6: "Verify the 3 new columns appear AFTER tb_bars_held in the Parquet schema"
    auto pq = temp_parquet_path("_t6_order");
    track_temp(pq);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq);
    ASSERT_EQ(result.exit_code, 0) << "Export failed: " << result.output;

    auto table = read_parquet_table(pq);
    ASSERT_NE(table, nullptr);

    auto schema = table->schema();
    int bars_held_idx = schema->GetFieldIndex("tb_bars_held");
    int both_idx = schema->GetFieldIndex("tb_both_triggered");
    int long_idx = schema->GetFieldIndex("tb_long_triggered");
    int short_idx = schema->GetFieldIndex("tb_short_triggered");

    ASSERT_NE(bars_held_idx, -1) << "tb_bars_held not found";
    ASSERT_NE(both_idx, -1) << "tb_both_triggered not found";
    ASSERT_NE(long_idx, -1) << "tb_long_triggered not found";
    ASSERT_NE(short_idx, -1) << "tb_short_triggered not found";

    EXPECT_EQ(both_idx, bars_held_idx + 1)
        << "tb_both_triggered must immediately follow tb_bars_held. "
        << "bars_held_idx=" << bars_held_idx << " both_idx=" << both_idx;
    EXPECT_EQ(long_idx, bars_held_idx + 2)
        << "tb_long_triggered must be 2 positions after tb_bars_held. "
        << "bars_held_idx=" << bars_held_idx << " long_idx=" << long_idx;
    EXPECT_EQ(short_idx, bars_held_idx + 3)
        << "tb_short_triggered must be 3 positions after tb_bars_held. "
        << "bars_held_idx=" << bars_held_idx << " short_idx=" << short_idx;
}

TEST_F(T6_SchemaColumnOrderTest, ColumnNamesExact) {
    // Spec T6: "Verify column names are exact: tb_both_triggered, tb_long_triggered, tb_short_triggered"
    auto pq = temp_parquet_path("_t6_exact");
    track_temp(pq);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq);
    ASSERT_EQ(result.exit_code, 0) << "Export failed: " << result.output;

    auto table = read_parquet_table(pq);
    ASSERT_NE(table, nullptr);
    ASSERT_EQ(table->num_columns(), 152);

    // Last 3 columns must be the new diagnostic columns
    EXPECT_EQ(table->field(149)->name(), "tb_both_triggered");
    EXPECT_EQ(table->field(150)->name(), "tb_long_triggered");
    EXPECT_EQ(table->field(151)->name(), "tb_short_triggered");
}

TEST_F(T6_SchemaColumnOrderTest, BidirectionalSchemaMatchesExpected) {
    // Full schema validation: all 152 column names in order
    auto pq = temp_parquet_path("_t6_full");
    track_temp(pq);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq);
    ASSERT_EQ(result.exit_code, 0) << "Export failed: " << result.output;

    auto table = read_parquet_table(pq);
    ASSERT_NE(table, nullptr);

    auto expected = expected_bidirectional_columns();
    ASSERT_EQ(static_cast<size_t>(table->num_columns()), expected.size())
        << "Expected " << expected.size() << " columns, got " << table->num_columns();

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(table->field(i)->name(), expected[i])
            << "Column " << i << ": expected '" << expected[i]
            << "' got '" << table->field(i)->name() << "'";
    }
}

// ===========================================================================
// Additional robustness checks
// ===========================================================================

class BidirectionalExportRobustnessTest : public BidirectionalExportTestBase {};

TEST_F(BidirectionalExportRobustnessTest, RowCountUnchangedByMode) {
    // Both modes must produce the same number of rows on the same data
    auto pq_bidir = temp_parquet_path("_robust_bidir");
    auto pq_legacy = temp_parquet_path("_robust_legacy");
    track_temp(pq_bidir);
    track_temp(pq_legacy);

    auto r1 = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_bidir);
    ASSERT_EQ(r1.exit_code, 0);

    auto r2 = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --legacy-labels --output " + pq_legacy);
    ASSERT_EQ(r2.exit_code, 0);

    auto t_bidir = read_parquet_table(pq_bidir);
    auto t_legacy = read_parquet_table(pq_legacy);
    ASSERT_NE(t_bidir, nullptr);
    ASSERT_NE(t_legacy, nullptr);

    EXPECT_EQ(t_bidir->num_rows(), t_legacy->num_rows())
        << "Row count must be identical between bidirectional and legacy modes";
}

TEST_F(BidirectionalExportRobustnessTest, TbExitTypeStillValid) {
    // tb_exit_type must still contain valid exit type strings in bidirectional mode
    auto pq = temp_parquet_path("_robust_exittype");
    track_temp(pq);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq);
    ASSERT_EQ(result.exit_code, 0);

    auto table = read_parquet_table(pq);
    ASSERT_NE(table, nullptr);

    REQUIRE_COLUMN(table, "tb_exit_type", exit_type_idx);

    auto col = table->column(exit_type_idx);
    std::set<std::string> valid_types = {"target", "stop", "expiry", "timeout"};

    for (int chunk = 0; chunk < col->num_chunks(); ++chunk) {
        auto arr = std::dynamic_pointer_cast<arrow::StringArray>(col->chunk(chunk));
        ASSERT_NE(arr, nullptr) << "tb_exit_type must be a STRING column";
        for (int64_t i = 0; i < arr->length(); ++i) {
            std::string val = arr->GetString(i);
            EXPECT_TRUE(valid_types.count(val) > 0)
                << "Invalid tb_exit_type '" << val << "' at row " << i
                << " — must be one of: target, stop, expiry, timeout";
        }
    }
}

TEST_F(BidirectionalExportRobustnessTest, TbBarsHeldNonNegative) {
    // tb_bars_held must be non-negative in bidirectional mode
    auto pq = temp_parquet_path("_robust_barsheld");
    track_temp(pq);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq);
    ASSERT_EQ(result.exit_code, 0);

    auto table = read_parquet_table(pq);
    ASSERT_NE(table, nullptr);

    REQUIRE_COLUMN(table, "tb_bars_held", bh_idx);
    auto bars_held = extract_doubles(table->column(bh_idx));

    for (size_t i = 0; i < bars_held.size(); ++i) {
        EXPECT_GE(bars_held[i], 0.0)
            << "tb_bars_held at row " << i << " is " << bars_held[i]
            << " — must be >= 0";
    }
}

TEST_F(BidirectionalExportRobustnessTest, LabelAndTriggeredColumnsAligned) {
    // For every row, exactly one of these four states must hold:
    // (a) tb_label=+1, long=1, short=0, both=0  [long only]
    // (b) tb_label=-1, long=0, short=1, both=0  [short only]
    // (c) tb_label=0,  long=1, short=1, both=1  [both triggered]
    // (d) tb_label=0,  long=0, short=0, both=0  [neither triggered]
    auto pq = temp_parquet_path("_robust_aligned");
    track_temp(pq);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq);
    ASSERT_EQ(result.exit_code, 0);

    auto table = read_parquet_table(pq);
    ASSERT_NE(table, nullptr);

    REQUIRE_COLUMN(table, "tb_label", label_idx);
    REQUIRE_COLUMN(table, "tb_long_triggered", long_idx);
    REQUIRE_COLUMN(table, "tb_short_triggered", short_idx);
    REQUIRE_COLUMN(table, "tb_both_triggered", both_idx);

    auto labels = extract_doubles(table->column(label_idx));
    auto long_t = extract_doubles(table->column(long_idx));
    auto short_t = extract_doubles(table->column(short_idx));
    auto both_t = extract_doubles(table->column(both_idx));

    for (size_t i = 0; i < labels.size(); ++i) {
        bool case_a = (labels[i] == 1.0 && long_t[i] == 1.0 && short_t[i] == 0.0 && both_t[i] == 0.0);
        bool case_b = (labels[i] == -1.0 && long_t[i] == 0.0 && short_t[i] == 1.0 && both_t[i] == 0.0);
        bool case_c = (labels[i] == 0.0 && long_t[i] == 1.0 && short_t[i] == 1.0 && both_t[i] == 1.0);
        bool case_d = (labels[i] == 0.0 && long_t[i] == 0.0 && short_t[i] == 0.0 && both_t[i] == 0.0);

        EXPECT_TRUE(case_a || case_b || case_c || case_d)
            << "Row " << i << " is in invalid state: "
            << "label=" << labels[i]
            << " long=" << long_t[i]
            << " short=" << short_t[i]
            << " both=" << both_t[i]
            << " — must be one of: long-only, short-only, both, or neither";
    }
}
