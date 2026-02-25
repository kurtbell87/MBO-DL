// parquet_export_test.cpp — TDD RED phase tests for Parquet output in bar_feature_export
// Spec: .kit/docs/parquet-export.md
//
// Tests the Parquet output path: schema match (T2), round-trip precision (T1),
// value comparison against CSV (T3), row counts (T4), timestamp ordering (T5/T6),
// warm-up bars (T7), terminal bar consistency (T8), parallel determinism (T9),
// cross-day state isolation (T10), CSV regression (T12), zstd compression (EC-9),
// row group structure, and format detection.
//
// Build requirements:
//   - Apache Arrow C++ (FetchContent) with Parquet and zstd enabled
//   - Link: Arrow::arrow_static, Parquet::parquet_static (or shared equivalents)
//   - GTest
//
// Integration tests require DATA files and should be labeled "integration".

#include <gtest/gtest.h>

// Arrow/Parquet reader — needed to validate exported Parquet files
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
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
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
            ("parquet_export_test" + suffix + ".parquet")).string();
}

std::string temp_csv_path(const std::string& suffix = "") {
    return (std::filesystem::temp_directory_path() /
            ("parquet_export_test" + suffix + ".csv")).string();
}

// Build the expected 149-column schema (matches bar_feature_export CSV header).
std::vector<std::string> expected_column_names() {
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

    // Forward returns (4)
    cols.push_back("return_1");
    cols.push_back("return_5");
    cols.push_back("return_20");
    cols.push_back("return_100");

    // Event count (1)
    cols.push_back("mbo_event_count");

    // Triple barrier labels (3)
    cols.push_back("tb_label");
    cols.push_back("tb_exit_type");
    cols.push_back("tb_bars_held");

    return cols;
}

// Build the expected 149-column Parquet schema. Parquet uses fwd_ prefix on
// forward return columns to avoid collision with Track A return features.
std::vector<std::string> expected_parquet_column_names() {
    auto cols = expected_column_names();
    // Forward return columns are at indices 141-144 (6+62+40+33 = 141)
    constexpr size_t FWD_RETURN_START = 6 + 62 + 40 + 33;
    cols[FWD_RETURN_START + 0] = "fwd_return_1";
    cols[FWD_RETURN_START + 1] = "fwd_return_5";
    cols[FWD_RETURN_START + 2] = "fwd_return_20";
    cols[FWD_RETURN_START + 3] = "fwd_return_100";
    return cols;
}

// Read a Parquet file into an Arrow Table using the Arrow C++ reader.
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

// Read Parquet file-level metadata (compression, row groups).
std::shared_ptr<parquet::FileMetaData> read_parquet_metadata(const std::string& path) {
    auto open_result = arrow::io::ReadableFile::Open(path);
    if (!open_result.ok()) return nullptr;

    auto file_reader = parquet::ParquetFileReader::Open(open_result.ValueOrDie());
    if (!file_reader) return nullptr;

    return file_reader->metadata();
}

bool data_available() {
    return std::filesystem::exists(DATA_DIR) &&
           std::filesystem::is_directory(DATA_DIR);
}

// Extract int64 timestamps from an Arrow column (handles both INT64 and UINT64).
std::vector<int64_t> extract_timestamps(const std::shared_ptr<arrow::ChunkedArray>& col) {
    std::vector<int64_t> timestamps;
    for (int chunk = 0; chunk < col->num_chunks(); ++chunk) {
        auto chunk_arr = col->chunk(chunk);
        if (auto arr = std::dynamic_pointer_cast<arrow::Int64Array>(chunk_arr)) {
            for (int64_t i = 0; i < arr->length(); ++i)
                timestamps.push_back(arr->Value(i));
        } else if (auto uarr = std::dynamic_pointer_cast<arrow::UInt64Array>(chunk_arr)) {
            for (int64_t i = 0; i < uarr->length(); ++i)
                timestamps.push_back(static_cast<int64_t>(uarr->Value(i)));
        }
    }
    return timestamps;
}

}  // anonymous namespace


// ===========================================================================
// Shared fixture base — binary+data check, temp file cleanup
// ===========================================================================

class ParquetExportTestBase : public ::testing::Test {
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
// T1: Parquet Round-Trip — export as Parquet, read back, verify precision
// ===========================================================================

class ParquetRoundTripTest : public ParquetExportTestBase {};

TEST_F(ParquetRoundTripTest, T1_DoubleValuesBitwiseIdenticalAfterRoundTrip) {
    // Export one day as Parquet, read back with Arrow, re-export, compare.
    // "Bitwise identical" means no precision loss from write/read cycle.
    auto pq_path1 = temp_parquet_path("_t1_trip1");
    auto pq_path2 = temp_parquet_path("_t1_trip2");
    track_temp(pq_path1);
    track_temp(pq_path2);

    auto r1 = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_path1);
    ASSERT_EQ(r1.exit_code, 0) << "First export failed: " << r1.output;

    auto r2 = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_path2);
    ASSERT_EQ(r2.exit_code, 0) << "Second export failed: " << r2.output;

    auto table1 = read_parquet_table(pq_path1);
    auto table2 = read_parquet_table(pq_path2);
    ASSERT_NE(table1, nullptr) << "Cannot read first Parquet file";
    ASSERT_NE(table2, nullptr) << "Cannot read second Parquet file";
    ASSERT_GT(table1->num_rows(), 0) << "Parquet file has no rows";
    ASSERT_EQ(table1->num_rows(), table2->num_rows()) << "Row count mismatch";
    ASSERT_EQ(table1->num_columns(), table2->num_columns()) << "Column count mismatch";

    // Compare all DOUBLE columns: values must be bitwise identical
    for (int col = 0; col < table1->num_columns(); ++col) {
        if (table1->field(col)->type()->id() != arrow::Type::DOUBLE) continue;

        auto col1 = table1->column(col);
        auto col2 = table2->column(col);

        for (int chunk = 0; chunk < col1->num_chunks(); ++chunk) {
            auto arr1 = std::static_pointer_cast<arrow::DoubleArray>(col1->chunk(chunk));
            auto arr2 = std::static_pointer_cast<arrow::DoubleArray>(col2->chunk(chunk));
            ASSERT_EQ(arr1->length(), arr2->length());

            for (int64_t row = 0; row < arr1->length(); ++row) {
                if (arr1->IsNull(row) && arr2->IsNull(row)) continue;
                ASSERT_EQ(arr1->IsNull(row), arr2->IsNull(row))
                    << "Null mismatch at col=" << table1->field(col)->name()
                    << " row=" << row;

                double v1 = arr1->Value(row);
                double v2 = arr2->Value(row);
                // NaN == NaN is false, so handle separately
                if (std::isnan(v1) && std::isnan(v2)) continue;
                EXPECT_EQ(0, std::memcmp(&v1, &v2, sizeof(double)))
                    << "Bitwise mismatch at col='" << table1->field(col)->name()
                    << "' row=" << row << " v1=" << v1 << " v2=" << v2;
            }
        }
    }
}


// ===========================================================================
// T2: Schema Match — Parquet column names, order, and types match CSV
// ===========================================================================

class ParquetSchemaTest : public ParquetExportTestBase {};

TEST_F(ParquetSchemaTest, T2_ColumnCountIs149) {
    auto pq_path = temp_parquet_path("_t2_count");
    track_temp(pq_path);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --legacy-labels --output " + pq_path);
    ASSERT_EQ(result.exit_code, 0) << "Export failed: " << result.output;

    auto table = read_parquet_table(pq_path);
    ASSERT_NE(table, nullptr) << "Cannot read Parquet file";

    EXPECT_EQ(table->num_columns(), 149)
        << "Parquet file must have exactly 149 columns in legacy mode "
        << "(6 meta + 62 Track A + 40 book_snap + 33 msg_summary "
        << "+ 4 returns + 1 event_count + 3 tb_labels)";
}

TEST_F(ParquetSchemaTest, T2_ColumnNamesMatchCsvInOrder) {
    auto pq_path = temp_parquet_path("_t2_names");
    track_temp(pq_path);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --legacy-labels --output " + pq_path);
    ASSERT_EQ(result.exit_code, 0) << "Export failed: " << result.output;

    auto table = read_parquet_table(pq_path);
    ASSERT_NE(table, nullptr);

    auto expected = expected_parquet_column_names();
    ASSERT_EQ(static_cast<size_t>(table->num_columns()), expected.size())
        << "Column count mismatch before name comparison";

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(table->field(i)->name(), expected[i])
            << "Column " << i << " name mismatch: Parquet='"
            << table->field(i)->name() << "' expected='" << expected[i] << "'";
    }
}

TEST_F(ParquetSchemaTest, T2_FloatColumnsAreDouble) {
    auto pq_path = temp_parquet_path("_t2_types");
    track_temp(pq_path);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --legacy-labels --output " + pq_path);
    ASSERT_EQ(result.exit_code, 0) << "Export failed: " << result.output;

    auto table = read_parquet_table(pq_path);
    ASSERT_NE(table, nullptr);

    auto expected = expected_parquet_column_names();

    // All numeric feature columns (indices 6+) should be DOUBLE,
    // except bar_type (string), is_warmup (bool/string) which are metadata,
    // and tb_exit_type (string) which is a categorical label.
    // We check columns 6 through 148 (Track A, book_snap, msg_summary, returns,
    // event_count, tb_labels), skipping the known STRING column.
    for (size_t i = 6; i < expected.size(); ++i) {
        if (expected[i] == "tb_exit_type") continue;  // STRING column, not DOUBLE
        auto field = table->field(static_cast<int>(i));
        EXPECT_EQ(field->type()->id(), arrow::Type::DOUBLE)
            << "Column '" << expected[i] << "' at index " << i
            << " should be DOUBLE, got " << field->type()->ToString();
    }
}

TEST_F(ParquetSchemaTest, T2_TimestampColumnIsInt64OrUInt64) {
    auto pq_path = temp_parquet_path("_t2_ts_type");
    track_temp(pq_path);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --legacy-labels --output " + pq_path);
    ASSERT_EQ(result.exit_code, 0) << "Export failed: " << result.output;

    auto table = read_parquet_table(pq_path);
    ASSERT_NE(table, nullptr);

    auto ts_field = table->field(0);
    EXPECT_EQ(ts_field->name(), "timestamp");

    bool is_int64 = (ts_field->type()->id() == arrow::Type::INT64);
    bool is_uint64 = (ts_field->type()->id() == arrow::Type::UINT64);
    EXPECT_TRUE(is_int64 || is_uint64)
        << "Timestamp column should be INT64 or UINT64, got "
        << ts_field->type()->ToString();
}


// ===========================================================================
// T3: CSV vs Parquet Value Comparison — all values within float64 tolerance
// ===========================================================================

class ParquetVsCsvTest : public ParquetExportTestBase {};

TEST_F(ParquetVsCsvTest, T3_AllValuesMatchWithinFloat64Tolerance) {
    // Export same data as both CSV and Parquet, compare all values
    auto csv_path = temp_csv_path("_t3");
    auto pq_path = temp_parquet_path("_t3");
    track_temp(csv_path);
    track_temp(pq_path);

    auto csv_result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --legacy-labels --output " + csv_path);
    ASSERT_EQ(csv_result.exit_code, 0) << "CSV export failed: " << csv_result.output;

    auto pq_result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --legacy-labels --output " + pq_path);
    ASSERT_EQ(pq_result.exit_code, 0) << "Parquet export failed: " << pq_result.output;

    auto csv_lines = read_all_lines(csv_path);
    ASSERT_GE(csv_lines.size(), 2u) << "CSV has no data rows";

    auto table = read_parquet_table(pq_path);
    ASSERT_NE(table, nullptr) << "Cannot read Parquet file";

    // Row counts must match
    int64_t csv_data_rows = static_cast<int64_t>(csv_lines.size()) - 1;
    ASSERT_EQ(table->num_rows(), csv_data_rows)
        << "Row count mismatch: Parquet=" << table->num_rows()
        << " CSV=" << csv_data_rows;

    constexpr double REL_TOL = 1e-10;

    // Compare up to 200 rows (enough to catch systematic errors)
    int64_t check_rows = std::min(table->num_rows(), int64_t(200));

    for (int64_t row = 0; row < check_rows; ++row) {
        auto csv_cols = parse_csv_header(csv_lines[row + 1]);
        ASSERT_EQ(csv_cols.size(), static_cast<size_t>(table->num_columns()))
            << "Column count mismatch at row " << row;

        for (int col = 0; col < table->num_columns(); ++col) {
            auto field = table->field(col);
            auto chunked = table->column(col);

            if (field->type()->id() == arrow::Type::STRING ||
                field->type()->id() == arrow::Type::STRING) {
                auto arr = std::static_pointer_cast<arrow::StringArray>(
                    chunked->chunk(0));
                EXPECT_EQ(arr->GetString(row), csv_cols[col])
                    << "String mismatch at col='" << field->name()
                    << "' row=" << row;
                continue;
            }

            if (field->type()->id() == arrow::Type::BOOL) {
                auto arr = std::static_pointer_cast<arrow::BooleanArray>(
                    chunked->chunk(0));
                std::string expected_str = arr->Value(row) ? "true" : "false";
                EXPECT_EQ(expected_str, csv_cols[col])
                    << "Boolean mismatch at col='" << field->name()
                    << "' row=" << row;
                continue;
            }

            if (field->type()->id() == arrow::Type::DOUBLE) {
                auto arr = std::static_pointer_cast<arrow::DoubleArray>(
                    chunked->chunk(0));
                double pq_val = arr->Value(row);

                if (csv_cols[col] == "NaN" || csv_cols[col] == "nan") {
                    EXPECT_TRUE(std::isnan(pq_val))
                        << "Expected NaN at col='" << field->name()
                        << "' row=" << row << " got=" << pq_val;
                    continue;
                }

                double csv_val = std::stod(csv_cols[col]);

                if (std::isnan(csv_val)) {
                    EXPECT_TRUE(std::isnan(pq_val))
                        << "Expected NaN at col='" << field->name()
                        << "' row=" << row;
                    continue;
                }

                if (csv_val == 0.0) {
                    EXPECT_NEAR(pq_val, 0.0, 1e-15)
                        << "Zero mismatch at col='" << field->name()
                        << "' row=" << row;
                } else {
                    double rel_err = std::abs(pq_val - csv_val) / std::abs(csv_val);
                    EXPECT_LE(rel_err, REL_TOL)
                        << "Value mismatch at col='" << field->name()
                        << "' row=" << row
                        << " csv=" << csv_val << " parquet=" << pq_val
                        << " rel_err=" << rel_err;
                }
            } else if (field->type()->id() == arrow::Type::INT64) {
                auto arr = std::static_pointer_cast<arrow::Int64Array>(
                    chunked->chunk(0));
                int64_t pq_val = arr->Value(row);
                int64_t csv_val = std::stoll(csv_cols[col]);
                EXPECT_EQ(pq_val, csv_val)
                    << "Int64 mismatch at col='" << field->name()
                    << "' row=" << row;
            } else if (field->type()->id() == arrow::Type::UINT64) {
                auto arr = std::static_pointer_cast<arrow::UInt64Array>(
                    chunked->chunk(0));
                uint64_t pq_val = arr->Value(row);
                uint64_t csv_val = std::stoull(csv_cols[col]);
                EXPECT_EQ(pq_val, csv_val)
                    << "UInt64 mismatch at col='" << field->name()
                    << "' row=" << row;
            }
        }
    }
}


// ===========================================================================
// T4: Row Count Match — Parquet row count == CSV row count for same date
// ===========================================================================

class ParquetRowCountTest : public ParquetExportTestBase {};

TEST_F(ParquetRowCountTest, T4_ParquetRowCountMatchesCsvRowCount) {
    auto csv_path = temp_csv_path("_t4");
    auto pq_path = temp_parquet_path("_t4");
    track_temp(csv_path);
    track_temp(pq_path);

    auto csv_result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + csv_path);
    ASSERT_EQ(csv_result.exit_code, 0) << "CSV export failed";

    auto pq_result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_path);
    ASSERT_EQ(pq_result.exit_code, 0) << "Parquet export failed";

    auto csv_lines = read_all_lines(csv_path);
    int64_t csv_data_rows = static_cast<int64_t>(csv_lines.size()) - 1;

    auto table = read_parquet_table(pq_path);
    ASSERT_NE(table, nullptr);

    EXPECT_EQ(table->num_rows(), csv_data_rows)
        << "Row count mismatch: Parquet=" << table->num_rows()
        << " CSV=" << csv_data_rows;
}

TEST_F(ParquetRowCountTest, T4_RowCountPositive) {
    auto pq_path = temp_parquet_path("_t4_positive");
    track_temp(pq_path);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_path);
    ASSERT_EQ(result.exit_code, 0) << "Export failed";

    auto table = read_parquet_table(pq_path);
    ASSERT_NE(table, nullptr);

    EXPECT_GT(table->num_rows(), 0)
        << "Parquet file must have at least one data row";
}


// ===========================================================================
// T5: No Duplicate Timestamps
// ===========================================================================

class ParquetTimestampUniqueTest : public ParquetExportTestBase {};

TEST_F(ParquetTimestampUniqueTest, T5_NoDuplicateTimestamps) {
    auto pq_path = temp_parquet_path("_t5");
    track_temp(pq_path);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_path);
    ASSERT_EQ(result.exit_code, 0) << "Export failed: " << result.output;

    auto table = read_parquet_table(pq_path);
    ASSERT_NE(table, nullptr);
    ASSERT_GT(table->num_rows(), 0);

    auto timestamps = extract_timestamps(table->column(0));
    ASSERT_FALSE(timestamps.empty()) << "No timestamps extracted";

    // Sort and find duplicates
    auto sorted = timestamps;
    std::sort(sorted.begin(), sorted.end());
    auto dup_it = std::adjacent_find(sorted.begin(), sorted.end());

    EXPECT_EQ(dup_it, sorted.end())
        << "Duplicate timestamp found: " << *dup_it;
}


// ===========================================================================
// T6: Timestamps Monotonically Increasing
// ===========================================================================

class ParquetTimestampMonotonicTest : public ParquetExportTestBase {};

TEST_F(ParquetTimestampMonotonicTest, T6_TimestampsStrictlyIncreasing) {
    auto pq_path = temp_parquet_path("_t6");
    track_temp(pq_path);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_path);
    ASSERT_EQ(result.exit_code, 0) << "Export failed: " << result.output;

    auto table = read_parquet_table(pq_path);
    ASSERT_NE(table, nullptr);
    ASSERT_GT(table->num_rows(), 1) << "Need >= 2 rows for monotonicity check";

    auto timestamps = extract_timestamps(table->column(0));
    ASSERT_GE(timestamps.size(), 2u);

    for (size_t i = 1; i < timestamps.size(); ++i) {
        EXPECT_GT(timestamps[i], timestamps[i - 1])
            << "Timestamp not strictly increasing at index " << i
            << ": prev=" << timestamps[i - 1] << " curr=" << timestamps[i];
    }
}


// ===========================================================================
// T7: Warm-up Bars Present — is_warmup column contains both true and false
// ===========================================================================

class ParquetWarmupTest : public ParquetExportTestBase {};

TEST_F(ParquetWarmupTest, T7_IsWarmupColumnExists) {
    auto pq_path = temp_parquet_path("_t7_exists");
    track_temp(pq_path);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_path);
    ASSERT_EQ(result.exit_code, 0) << "Export failed: " << result.output;

    auto table = read_parquet_table(pq_path);
    ASSERT_NE(table, nullptr);
    ASSERT_GE(table->num_columns(), 5);

    EXPECT_EQ(table->field(4)->name(), "is_warmup")
        << "Column 4 should be 'is_warmup'";
}

TEST_F(ParquetWarmupTest, T7_IsWarmupAllFalse_WarmupBarsExcluded) {
    // Warm-up bars are excluded from the export (matching WarmupExclusionTest
    // for CSV). The is_warmup column exists but should contain only 'false'
    // values since warm-up bars are filtered out before writing.
    auto pq_path = temp_parquet_path("_t7_values");
    track_temp(pq_path);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_path);
    ASSERT_EQ(result.exit_code, 0) << "Export failed: " << result.output;

    auto table = read_parquet_table(pq_path);
    ASSERT_NE(table, nullptr);
    ASSERT_GT(table->num_rows(), 0);

    auto warmup_col = table->column(4);
    bool found_true = false;
    bool found_false = false;

    for (int chunk = 0; chunk < warmup_col->num_chunks(); ++chunk) {
        auto chunk_arr = warmup_col->chunk(chunk);

        if (auto bool_arr = std::dynamic_pointer_cast<arrow::BooleanArray>(chunk_arr)) {
            for (int64_t i = 0; i < bool_arr->length(); ++i) {
                if (bool_arr->Value(i)) found_true = true;
                else found_false = true;
            }
        } else if (auto str_arr = std::dynamic_pointer_cast<arrow::StringArray>(chunk_arr)) {
            for (int64_t i = 0; i < str_arr->length(); ++i) {
                auto val = str_arr->GetString(i);
                if (val == "true" || val == "1") found_true = true;
                if (val == "false" || val == "0") found_false = true;
            }
        } else if (auto dbl_arr = std::dynamic_pointer_cast<arrow::DoubleArray>(chunk_arr)) {
            for (int64_t i = 0; i < dbl_arr->length(); ++i) {
                if (dbl_arr->Value(i) != 0.0) found_true = true;
                else found_false = true;
            }
        }
    }

    EXPECT_FALSE(found_true)
        << "is_warmup column should not contain 'true' values "
        << "(warm-up bars are excluded from export, matching CSV behavior)";
    EXPECT_TRUE(found_false)
        << "is_warmup column should contain at least one 'false' value "
        << "(non-warmup bars after warm-up period)";
}


// ===========================================================================
// T8: Terminal Bar Consistency — NaN/sentinel pattern matches CSV
// ===========================================================================

class ParquetTerminalBarTest : public ParquetExportTestBase {};

TEST_F(ParquetTerminalBarTest, T8_TerminalBarForwardReturnsMatchCsv) {
    auto csv_path = temp_csv_path("_t8");
    auto pq_path = temp_parquet_path("_t8");
    track_temp(csv_path);
    track_temp(pq_path);

    auto csv_result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + csv_path);
    ASSERT_EQ(csv_result.exit_code, 0) << "CSV export failed";

    auto pq_result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_path);
    ASSERT_EQ(pq_result.exit_code, 0) << "Parquet export failed";

    auto csv_lines = read_all_lines(csv_path);
    ASSERT_GE(csv_lines.size(), 2u);

    auto table = read_parquet_table(pq_path);
    ASSERT_NE(table, nullptr);

    // Forward return column indices:
    // return_1 = 6+62+40+33 = 141, return_5 = 142, return_20 = 143, return_100 = 144
    constexpr int RETURN_1_COL = 141;
    constexpr int RETURN_100_COL = 144;

    int64_t num_rows = std::min(
        table->num_rows(), static_cast<int64_t>(csv_lines.size() - 1));

    // Check last 20 rows (or all) — terminal bars at end-of-session
    int64_t check_start = std::max(int64_t(0), num_rows - 20);

    for (int64_t row = check_start; row < num_rows; ++row) {
        auto csv_cols = parse_csv_header(csv_lines[row + 1]);
        ASSERT_GE(csv_cols.size(), static_cast<size_t>(RETURN_100_COL + 1));

        for (int col_idx = RETURN_1_COL; col_idx <= RETURN_100_COL; ++col_idx) {
            auto pq_col = table->column(col_idx);
            auto arr = std::dynamic_pointer_cast<arrow::DoubleArray>(
                pq_col->chunk(0));
            ASSERT_NE(arr, nullptr)
                << "Forward return column " << col_idx << " is not DOUBLE";

            double pq_val = arr->Value(row);
            const std::string& csv_val_str = csv_cols[col_idx];

            bool csv_is_nan = (csv_val_str == "NaN" || csv_val_str == "nan" ||
                               csv_val_str == "NAN");

            if (csv_is_nan) {
                EXPECT_TRUE(std::isnan(pq_val))
                    << "Terminal bar at row " << row
                    << " col " << table->field(col_idx)->name()
                    << ": CSV is NaN but Parquet is " << pq_val;
            } else {
                double csv_val = std::stod(csv_val_str);
                if (std::isnan(pq_val)) {
                    ADD_FAILURE()
                        << "Terminal bar at row " << row
                        << " col " << table->field(col_idx)->name()
                        << ": Parquet is NaN but CSV is " << csv_val;
                } else {
                    EXPECT_NEAR(pq_val, csv_val, 1e-10)
                        << "Terminal bar return mismatch at row " << row
                        << " col " << table->field(col_idx)->name();
                }
            }
        }
    }
}


// ===========================================================================
// T9: Parallel vs Sequential — two parallel exports produce identical output
// ===========================================================================

class ParquetParallelTest : public ParquetExportTestBase {};

TEST_F(ParquetParallelTest, T9_ParallelExportsProduceIdenticalOutput) {
    auto pq_path1 = temp_parquet_path("_t9_a");
    auto pq_path2 = temp_parquet_path("_t9_b");
    track_temp(pq_path1);
    track_temp(pq_path2);

    // Launch two exports simultaneously in separate threads
    export_test_helpers::RunResult result1, result2;

    std::thread t1([&]() {
        result1 = run_command(
            BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_path1);
    });
    std::thread t2([&]() {
        result2 = run_command(
            BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_path2);
    });

    t1.join();
    t2.join();

    ASSERT_EQ(result1.exit_code, 0) << "Parallel export A failed: " << result1.output;
    ASSERT_EQ(result2.exit_code, 0) << "Parallel export B failed: " << result2.output;

    // Compare Parquet content (not byte-identical — Parquet metadata may include
    // creation timestamps). Compare row-by-row values instead.
    auto table1 = read_parquet_table(pq_path1);
    auto table2 = read_parquet_table(pq_path2);
    ASSERT_NE(table1, nullptr);
    ASSERT_NE(table2, nullptr);

    ASSERT_EQ(table1->num_rows(), table2->num_rows())
        << "Parallel exports produced different row counts";
    ASSERT_EQ(table1->num_columns(), table2->num_columns())
        << "Parallel exports produced different column counts";

    // Compare all DOUBLE columns value-by-value
    for (int col = 0; col < table1->num_columns(); ++col) {
        if (table1->field(col)->type()->id() != arrow::Type::DOUBLE) continue;

        auto c1 = table1->column(col);
        auto c2 = table2->column(col);

        for (int chunk = 0; chunk < c1->num_chunks(); ++chunk) {
            auto a1 = std::static_pointer_cast<arrow::DoubleArray>(c1->chunk(chunk));
            auto a2 = std::static_pointer_cast<arrow::DoubleArray>(c2->chunk(chunk));

            for (int64_t row = 0; row < a1->length(); ++row) {
                if (std::isnan(a1->Value(row)) && std::isnan(a2->Value(row))) continue;
                EXPECT_EQ(a1->Value(row), a2->Value(row))
                    << "Parallel mismatch at col='"
                    << table1->field(col)->name() << "' row=" << row;
            }
        }
    }
}


// ===========================================================================
// T10: No Cross-Day State Leakage — Day D output identical standalone vs
//      after exporting Day D-1
// ===========================================================================

class ParquetStateLeakageTest : public ParquetExportTestBase {};

TEST_F(ParquetStateLeakageTest, T10_SequentialRunsProduceIdenticalOutput) {
    // Each bar_feature_export invocation is independent (no shared state).
    // Running it twice sequentially must produce identical Parquet output.
    auto pq_run1 = temp_parquet_path("_t10_r1");
    auto pq_run2 = temp_parquet_path("_t10_r2");
    track_temp(pq_run1);
    track_temp(pq_run2);

    auto r1 = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_run1);
    ASSERT_EQ(r1.exit_code, 0) << "First run failed";

    auto r2 = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_run2);
    ASSERT_EQ(r2.exit_code, 0) << "Second run failed";

    auto table1 = read_parquet_table(pq_run1);
    auto table2 = read_parquet_table(pq_run2);
    ASSERT_NE(table1, nullptr);
    ASSERT_NE(table2, nullptr);

    ASSERT_EQ(table1->num_rows(), table2->num_rows())
        << "Different row counts between sequential runs";

    // Compare all DOUBLE columns
    for (int col = 0; col < table1->num_columns(); ++col) {
        if (table1->field(col)->type()->id() != arrow::Type::DOUBLE) continue;

        auto c1 = table1->column(col);
        auto c2 = table2->column(col);

        for (int chunk = 0; chunk < c1->num_chunks(); ++chunk) {
            auto a1 = std::static_pointer_cast<arrow::DoubleArray>(c1->chunk(chunk));
            auto a2 = std::static_pointer_cast<arrow::DoubleArray>(c2->chunk(chunk));

            for (int64_t row = 0; row < a1->length(); ++row) {
                if (std::isnan(a1->Value(row)) && std::isnan(a2->Value(row))) continue;
                EXPECT_EQ(a1->Value(row), a2->Value(row))
                    << "State leakage: value differs between sequential runs at col='"
                    << table1->field(col)->name() << "' row=" << row;
            }
        }
    }
}


// ===========================================================================
// T12: CSV Byte-Identical — CSV output unchanged after Parquet addition
// ===========================================================================

class CsvRegressionTest : public ParquetExportTestBase {};

TEST_F(CsvRegressionTest, T12_CsvHeaderUnchanged) {
    auto csv_path = temp_csv_path("_t12_header");
    track_temp(csv_path);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + csv_path);
    ASSERT_EQ(result.exit_code, 0) << "CSV export failed: " << result.output;

    auto actual_cols = parse_csv_header(read_first_line(csv_path));
    auto expected_cols = expected_column_names();

    ASSERT_EQ(actual_cols.size(), expected_cols.size())
        << "CSV header column count changed after Parquet addition";

    for (size_t i = 0; i < expected_cols.size(); ++i) {
        EXPECT_EQ(actual_cols[i], expected_cols[i])
            << "CSV header column " << i << " changed after Parquet addition";
    }
}

TEST_F(CsvRegressionTest, T12_CsvDataRowsHave149Columns) {
    auto csv_path = temp_csv_path("_t12_rowwidth");
    track_temp(csv_path);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + csv_path);
    ASSERT_EQ(result.exit_code, 0) << "CSV export failed";

    auto lines = read_all_lines(csv_path);
    ASSERT_GE(lines.size(), 2u);

    for (size_t i = 1; i < lines.size(); ++i) {
        auto cols = parse_csv_header(lines[i]);
        EXPECT_EQ(cols.size(), 149u)
            << "CSV data row " << i << " has " << cols.size()
            << " columns, expected 149";
    }
}

TEST_F(CsvRegressionTest, T12_CsvExtensionStillProducesCsvNotParquet) {
    auto csv_path = temp_csv_path("_t12_ext");
    track_temp(csv_path);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + csv_path);
    if (result.exit_code != 0) GTEST_SKIP() << "Export failed";

    // First line should be a CSV header, not Parquet magic bytes
    auto first_line = read_first_line(csv_path);
    ASSERT_FALSE(first_line.empty());
    EXPECT_EQ(first_line.substr(0, 10), "timestamp,")
        << "CSV file should start with 'timestamp,' header";

    // File should NOT have Parquet magic bytes
    std::ifstream f(csv_path, std::ios::binary);
    char magic[4] = {};
    f.read(magic, 4);
    bool is_parquet = (magic[0] == 'P' && magic[1] == 'A' &&
                       magic[2] == 'R' && magic[3] == '1');
    EXPECT_FALSE(is_parquet)
        << ".csv extension must produce CSV output, not Parquet";
}

TEST_F(CsvRegressionTest, T12_CsvMatchesReferenceIfAvailable) {
    // Compare against the known-good reference CSV
    std::string ref_path = ".kit/results/hybrid-model/time_5s.csv";
    if (!std::filesystem::exists(ref_path)) {
        GTEST_SKIP() << "Reference CSV not available at " << ref_path;
    }

    auto csv_path = temp_csv_path("_t12_ref");
    track_temp(csv_path);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + csv_path);
    ASSERT_EQ(result.exit_code, 0) << "CSV export failed";

    // Compare headers (reference predates tb_label columns)
    auto ref_cols = parse_csv_header(read_first_line(ref_path));
    auto new_cols = parse_csv_header(read_first_line(csv_path));
    size_t compare_count = std::min(ref_cols.size(), new_cols.size());
    compare_count = std::min(compare_count, ref_cols.size());

    for (size_t i = 0; i < compare_count; ++i) {
        EXPECT_EQ(new_cols[i], ref_cols[i])
            << "CSV column " << i << " differs from reference";
    }
}


// ===========================================================================
// EC-9: Parquet files use zstd compression
// ===========================================================================

class ParquetCompressionTest : public ParquetExportTestBase {};

TEST_F(ParquetCompressionTest, EC9_ZstdCompressionUsed) {
    auto pq_path = temp_parquet_path("_ec9");
    track_temp(pq_path);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_path);
    ASSERT_EQ(result.exit_code, 0) << "Export failed: " << result.output;

    auto metadata = read_parquet_metadata(pq_path);
    ASSERT_NE(metadata, nullptr) << "Cannot read Parquet metadata";
    ASSERT_GT(metadata->num_row_groups(), 0) << "No row groups found";

    // Check first column of first row group for ZSTD compression
    auto row_group = metadata->RowGroup(0);
    ASSERT_GT(row_group->num_columns(), 0);

    auto col_meta = row_group->ColumnChunk(0);
    EXPECT_EQ(col_meta->compression(), parquet::Compression::ZSTD)
        << "Parquet should use ZSTD compression, got codec="
        << static_cast<int>(col_meta->compression());
}

TEST_F(ParquetCompressionTest, EC9_AllColumnsUseZstd) {
    auto pq_path = temp_parquet_path("_ec9_all");
    track_temp(pq_path);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_path);
    ASSERT_EQ(result.exit_code, 0) << "Export failed: " << result.output;

    auto metadata = read_parquet_metadata(pq_path);
    ASSERT_NE(metadata, nullptr);

    for (int rg = 0; rg < metadata->num_row_groups(); ++rg) {
        auto row_group = metadata->RowGroup(rg);
        for (int col = 0; col < row_group->num_columns(); ++col) {
            auto col_meta = row_group->ColumnChunk(col);
            EXPECT_EQ(col_meta->compression(), parquet::Compression::ZSTD)
                << "Column " << col << " in row group " << rg
                << " should use ZSTD compression, got codec="
                << static_cast<int>(col_meta->compression());
        }
    }
}


// ===========================================================================
// Row Group Structure — one row group per day
// ===========================================================================

class ParquetRowGroupTest : public ParquetExportTestBase {};

TEST_F(ParquetRowGroupTest, AtLeastOneRowGroup) {
    auto pq_path = temp_parquet_path("_rg");
    track_temp(pq_path);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_path);
    ASSERT_EQ(result.exit_code, 0) << "Export failed: " << result.output;

    auto metadata = read_parquet_metadata(pq_path);
    ASSERT_NE(metadata, nullptr);

    EXPECT_GE(metadata->num_row_groups(), 1)
        << "Parquet file must have at least one row group";
}

TEST_F(ParquetRowGroupTest, RowGroupRowsSumToTotal) {
    auto pq_path = temp_parquet_path("_rg_sum");
    track_temp(pq_path);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_path);
    ASSERT_EQ(result.exit_code, 0) << "Export failed: " << result.output;

    auto metadata = read_parquet_metadata(pq_path);
    ASSERT_NE(metadata, nullptr);

    int64_t total_rows = 0;
    for (int rg = 0; rg < metadata->num_row_groups(); ++rg) {
        total_rows += metadata->RowGroup(rg)->num_rows();
    }

    EXPECT_EQ(total_rows, metadata->num_rows())
        << "Sum of row group rows must equal total file rows";
}

TEST_F(ParquetRowGroupTest, EachRowGroupHasNonZeroRows) {
    auto pq_path = temp_parquet_path("_rg_nonempty");
    track_temp(pq_path);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_path);
    ASSERT_EQ(result.exit_code, 0) << "Export failed: " << result.output;

    auto metadata = read_parquet_metadata(pq_path);
    ASSERT_NE(metadata, nullptr);

    for (int rg = 0; rg < metadata->num_row_groups(); ++rg) {
        EXPECT_GT(metadata->RowGroup(rg)->num_rows(), 0)
            << "Row group " << rg << " is empty";
    }
}


// ===========================================================================
// Format Detection — .parquet extension triggers Parquet output
// ===========================================================================

class FormatDetectionTest : public ParquetExportTestBase {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(BINARY_PATH)) GTEST_SKIP() << "Binary not built";
        // Intentionally skip data_available() check — some tests don't need data
    }
};

TEST_F(FormatDetectionTest, ParquetExtensionAccepted) {
    // Tool should accept .parquet extension without argument errors
    auto pq_path = temp_parquet_path("_fmt");
    track_temp(pq_path);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_path);

    // Should not produce format-related argument errors
    bool has_format_error = (
        result.output.find("Unknown output format") != std::string::npos ||
        result.output.find("Unsupported extension") != std::string::npos ||
        result.output.find("Unsupported format") != std::string::npos);
    EXPECT_FALSE(has_format_error)
        << ".parquet extension should be accepted. Got: " << result.output;
}

TEST_F(FormatDetectionTest, ParquetOutputHasMagicBytes) {
    if (!data_available()) GTEST_SKIP() << "Data not available";

    auto pq_path = temp_parquet_path("_fmt_magic");
    track_temp(pq_path);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_path);
    ASSERT_EQ(result.exit_code, 0) << "Export failed: " << result.output;

    // Parquet magic bytes: "PAR1" at start and end of file
    std::ifstream f(pq_path, std::ios::binary | std::ios::ate);
    ASSERT_TRUE(f.is_open()) << "Cannot open Parquet file";

    auto file_size = f.tellg();
    ASSERT_GE(file_size, 8) << "File too small to be valid Parquet";

    // Header magic
    f.seekg(0);
    char header_magic[4] = {};
    f.read(header_magic, 4);
    EXPECT_EQ(std::string(header_magic, 4), "PAR1")
        << "Parquet file should start with 'PAR1' magic bytes";

    // Footer magic
    f.seekg(-4, std::ios::end);
    char footer_magic[4] = {};
    f.read(footer_magic, 4);
    EXPECT_EQ(std::string(footer_magic, 4), "PAR1")
        << "Parquet file should end with 'PAR1' magic bytes";
}

TEST_F(FormatDetectionTest, ParquetFileReadableByArrow) {
    if (!data_available()) GTEST_SKIP() << "Data not available";

    auto pq_path = temp_parquet_path("_fmt_arrow");
    track_temp(pq_path);

    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --legacy-labels --output " + pq_path);
    ASSERT_EQ(result.exit_code, 0) << "Export failed: " << result.output;

    auto table = read_parquet_table(pq_path);
    ASSERT_NE(table, nullptr)
        << "Arrow must be able to read the exported Parquet file";
    EXPECT_GT(table->num_rows(), 0) << "Parquet table should have rows";
    EXPECT_EQ(table->num_columns(), 149)
        << "Parquet table should have 149 columns in legacy mode";
}


// ===========================================================================
// Parquet File Size — should be smaller than CSV due to compression
// ===========================================================================

class ParquetFileSizeTest : public ParquetExportTestBase {};

TEST_F(ParquetFileSizeTest, ParquetWithZstdSmallerThanCsv) {
    auto csv_path = temp_csv_path("_size");
    auto pq_path = temp_parquet_path("_size");
    track_temp(csv_path);
    track_temp(pq_path);

    auto csv_result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + csv_path);
    ASSERT_EQ(csv_result.exit_code, 0);

    auto pq_result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5 --output " + pq_path);
    ASSERT_EQ(pq_result.exit_code, 0);

    auto csv_size = std::filesystem::file_size(csv_path);
    auto pq_size = std::filesystem::file_size(pq_path);

    EXPECT_LT(pq_size, csv_size)
        << "Parquet (zstd compressed) should be smaller than CSV. "
        << "Parquet=" << pq_size << " bytes, CSV=" << csv_size << " bytes";
}


// ===========================================================================
// Column Name Contract — verify expected_column_names() consistency
// (Unit test, no data or Arrow needed — relies only on BarFeatureRow)
// ===========================================================================

class ColumnNameContractTest : public ::testing::Test {};

TEST_F(ColumnNameContractTest, ExpectedColumnCountIs149) {
    auto cols = expected_column_names();
    EXPECT_EQ(cols.size(), 149u)
        << "Expected column count must be 149";
}

TEST_F(ColumnNameContractTest, FirstSixAreMetadata) {
    auto cols = expected_column_names();
    ASSERT_GE(cols.size(), 6u);
    EXPECT_EQ(cols[0], "timestamp");
    EXPECT_EQ(cols[1], "bar_type");
    EXPECT_EQ(cols[2], "bar_param");
    EXPECT_EQ(cols[3], "day");
    EXPECT_EQ(cols[4], "is_warmup");
    EXPECT_EQ(cols[5], "bar_index");
}

TEST_F(ColumnNameContractTest, LastThreeAreTBLabels) {
    auto cols = expected_column_names();
    ASSERT_GE(cols.size(), 3u);
    EXPECT_EQ(cols[146], "tb_label");
    EXPECT_EQ(cols[147], "tb_exit_type");
    EXPECT_EQ(cols[148], "tb_bars_held");
}

TEST_F(ColumnNameContractTest, TrackAFeaturesAre62) {
    auto features = BarFeatureRow::feature_names();
    EXPECT_EQ(features.size(), 62u);
}

TEST_F(ColumnNameContractTest, MsgSummarySizeIs33) {
    EXPECT_EQ(MessageSummary::SUMMARY_SIZE, 33u);
}
