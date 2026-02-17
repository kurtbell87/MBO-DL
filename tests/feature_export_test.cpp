// feature_export_test.cpp — TDD RED phase tests for Feature Export
// Spec: .kit/docs/feature-computation.md §Export Format
//
// Tests for FeatureExporter: CSV export with all Track A/B features, metadata,
// forward returns, rollover exclusion, and file I/O.
//
// No implementation files exist yet — all tests must fail to compile or fail at runtime.

#include <gtest/gtest.h>

// Headers that the implementation must provide (spec §Project Structure):
#include "features/feature_export.hpp"       // FeatureExporter, ExportConfig
#include "features/bar_features.hpp"         // BarFeatureRow, BarFeatureComputer
#include "features/raw_representations.hpp"  // BookSnapshotExport, MessageSummary
#include "backtest/rollover.hpp"             // RolloverCalendar
#include "bars/bar.hpp"                      // Bar

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// ===========================================================================
// Helpers
// ===========================================================================
namespace {

constexpr float TICK_SIZE = 0.25f;

Bar make_bar(float close_mid, uint32_t volume = 100) {
    Bar bar{};
    bar.close_mid = close_mid;
    bar.open_mid = close_mid - 0.25f;
    bar.high_mid = close_mid + 0.25f;
    bar.low_mid = close_mid - 0.50f;
    bar.vwap = close_mid;
    bar.volume = volume;
    bar.spread = 0.25f;
    bar.bar_duration_s = 1.0f;
    bar.time_of_day = 9.5f;
    bar.open_ts = 1641219000000000000ULL;  // 2022-01-03 09:30 ET
    bar.close_ts = bar.open_ts + 1000000000ULL;
    bar.tick_count = volume;
    bar.buy_volume = 60.0f;
    bar.sell_volume = 40.0f;

    for (int i = 0; i < BOOK_DEPTH; ++i) {
        bar.bids[i][0] = close_mid - TICK_SIZE * (0.5f + i);
        bar.bids[i][1] = 10.0f + static_cast<float>(i);
        bar.asks[i][0] = close_mid + TICK_SIZE * (0.5f + i);
        bar.asks[i][1] = 10.0f + static_cast<float>(i);
    }

    bar.add_count = 50;
    bar.cancel_count = 30;
    bar.modify_count = 10;
    bar.trade_event_count = volume;

    return bar;
}

std::vector<Bar> make_bar_sequence(int count, float start_mid = 4500.0f) {
    std::vector<Bar> bars;
    for (int i = 0; i < count; ++i) {
        bars.push_back(make_bar(start_mid + i * 0.25f));
    }
    return bars;
}

BarFeatureRow make_feature_row(int idx = 0) {
    BarFeatureRow row{};
    row.timestamp = 1641219000000000000ULL + idx * 1000000000ULL;
    row.bar_type = "volume";
    row.bar_param = 100.0f;
    row.day = 20220103;
    row.is_warmup = false;

    // Fill with non-zero values
    row.book_imbalance_1 = 0.1f;
    row.book_imbalance_3 = 0.15f;
    row.book_imbalance_5 = 0.2f;
    row.book_imbalance_10 = 0.25f;
    row.net_volume = 20.0f;
    row.spread = 1.0f;
    row.time_sin = 0.5f;
    row.time_cos = 0.866f;
    row.minutes_since_open = 5.0f;
    row.minutes_to_close = 385.0f;

    row.fwd_return_1 = 1.0f;
    row.fwd_return_5 = 5.0f;
    row.fwd_return_20 = 20.0f;
    row.fwd_return_100 = std::numeric_limits<float>::quiet_NaN();

    return row;
}

// Temp file path for test output
std::string temp_csv_path() {
    return std::filesystem::temp_directory_path() / "test_feature_export.csv";
}

// Count lines in a string
int count_lines(const std::string& s) {
    return static_cast<int>(std::count(s.begin(), s.end(), '\n'));
}

// Parse CSV header line into column names
std::vector<std::string> parse_header(const std::string& header_line) {
    std::vector<std::string> cols;
    std::istringstream ss(header_line);
    std::string col;
    while (std::getline(ss, col, ',')) {
        cols.push_back(col);
    }
    return cols;
}

}  // anonymous namespace

// ===========================================================================
// FeatureExporter — Construction Tests
// ===========================================================================

class FeatureExporterConstructionTest : public ::testing::Test {};

TEST_F(FeatureExporterConstructionTest, DefaultConstruction) {
    FeatureExporter exporter;
    (void)exporter;
}

TEST_F(FeatureExporterConstructionTest, ConstructWithConfig) {
    ExportConfig config;
    config.output_path = temp_csv_path();
    config.include_warmup = false;
    FeatureExporter exporter(config);
    (void)exporter;
}

// ===========================================================================
// CSV Header — Column Validation
// ===========================================================================

class CSVHeaderTest : public ::testing::Test {
protected:
    void TearDown() override {
        std::filesystem::remove(temp_csv_path());
    }
};

TEST_F(CSVHeaderTest, HeaderContainsBarMetadata) {
    FeatureExporter exporter;
    auto header = exporter.header_line();
    auto cols = parse_header(header);

    // Must include bar metadata columns
    EXPECT_NE(std::find(cols.begin(), cols.end(), "timestamp"), cols.end());
    EXPECT_NE(std::find(cols.begin(), cols.end(), "bar_type"), cols.end());
    EXPECT_NE(std::find(cols.begin(), cols.end(), "bar_param"), cols.end());
    EXPECT_NE(std::find(cols.begin(), cols.end(), "day"), cols.end());
    EXPECT_NE(std::find(cols.begin(), cols.end(), "is_warmup"), cols.end());
}

TEST_F(CSVHeaderTest, HeaderContainsTrackAFeatures) {
    FeatureExporter exporter;
    auto header = exporter.header_line();
    auto cols = parse_header(header);

    // Spot-check representative features from each category
    EXPECT_NE(std::find(cols.begin(), cols.end(), "book_imbalance_1"), cols.end());
    EXPECT_NE(std::find(cols.begin(), cols.end(), "book_imbalance_10"), cols.end());
    EXPECT_NE(std::find(cols.begin(), cols.end(), "weighted_imbalance"), cols.end());
    EXPECT_NE(std::find(cols.begin(), cols.end(), "spread"), cols.end());
    EXPECT_NE(std::find(cols.begin(), cols.end(), "net_volume"), cols.end());
    EXPECT_NE(std::find(cols.begin(), cols.end(), "kyle_lambda"), cols.end());
    EXPECT_NE(std::find(cols.begin(), cols.end(), "volatility_20"), cols.end());
    EXPECT_NE(std::find(cols.begin(), cols.end(), "momentum"), cols.end());
    EXPECT_NE(std::find(cols.begin(), cols.end(), "volume_surprise"), cols.end());
    EXPECT_NE(std::find(cols.begin(), cols.end(), "time_sin"), cols.end());
    EXPECT_NE(std::find(cols.begin(), cols.end(), "cancel_add_ratio"), cols.end());
    EXPECT_NE(std::find(cols.begin(), cols.end(), "message_rate"), cols.end());
    EXPECT_NE(std::find(cols.begin(), cols.end(), "order_flow_toxicity"), cols.end());
}

TEST_F(CSVHeaderTest, HeaderContainsTrackBBookSnapshot) {
    FeatureExporter exporter;
    auto header = exporter.header_line();
    auto cols = parse_header(header);

    // Track B book snapshot: 40 flattened values (book_snap_0 through book_snap_39)
    EXPECT_NE(std::find(cols.begin(), cols.end(), "book_snap_0"), cols.end());
    EXPECT_NE(std::find(cols.begin(), cols.end(), "book_snap_39"), cols.end());
}

TEST_F(CSVHeaderTest, HeaderContainsTrackBMessageSummaries) {
    FeatureExporter exporter;
    auto header = exporter.header_line();
    auto cols = parse_header(header);

    // Track B message summary columns (at least one)
    EXPECT_NE(std::find(cols.begin(), cols.end(), "msg_summary_0"), cols.end());
}

TEST_F(CSVHeaderTest, HeaderContainsForwardReturns) {
    FeatureExporter exporter;
    auto header = exporter.header_line();
    auto cols = parse_header(header);

    EXPECT_NE(std::find(cols.begin(), cols.end(), "return_1"), cols.end());
    EXPECT_NE(std::find(cols.begin(), cols.end(), "return_5"), cols.end());
    EXPECT_NE(std::find(cols.begin(), cols.end(), "return_20"), cols.end());
    EXPECT_NE(std::find(cols.begin(), cols.end(), "return_100"), cols.end());
}

// ===========================================================================
// CSV Row Formatting
// ===========================================================================

class CSVRowTest : public ::testing::Test {};

TEST_F(CSVRowTest, RowHasSameColumnCountAsHeader) {
    FeatureExporter exporter;
    auto header = exporter.header_line();
    auto header_cols = parse_header(header);

    auto row = make_feature_row();
    auto row_str = exporter.format_row(row);
    auto row_cols = parse_header(row_str);

    EXPECT_EQ(row_cols.size(), header_cols.size())
        << "Row column count (" << row_cols.size() << ") must match header count ("
        << header_cols.size() << ")";
}

TEST_F(CSVRowTest, TimestampIsFirstColumn) {
    FeatureExporter exporter;
    auto row = make_feature_row();
    auto row_str = exporter.format_row(row);
    auto cols = parse_header(row_str);
    EXPECT_FALSE(cols.empty());
    // First column should be the timestamp
    EXPECT_EQ(cols[0], std::to_string(row.timestamp));
}

TEST_F(CSVRowTest, NaNWrittenAsNaN) {
    FeatureExporter exporter;
    auto row = make_feature_row();
    row.fwd_return_100 = std::numeric_limits<float>::quiet_NaN();
    auto row_str = exporter.format_row(row);
    // Should contain "NaN" or "nan" for the NaN field
    EXPECT_TRUE(row_str.find("NaN") != std::string::npos ||
                row_str.find("nan") != std::string::npos);
}

TEST_F(CSVRowTest, IsWarmupWrittenAsBool) {
    FeatureExporter exporter;
    auto row = make_feature_row();
    row.is_warmup = true;
    auto row_str = exporter.format_row(row);
    // Should contain "true" or "1" for is_warmup
    EXPECT_TRUE(row_str.find("true") != std::string::npos ||
                row_str.find(",1,") != std::string::npos);
}

// ===========================================================================
// File Export — CSV
// ===========================================================================

class CSVFileExportTest : public ::testing::Test {
protected:
    void TearDown() override {
        std::filesystem::remove(temp_csv_path());
    }
};

TEST_F(CSVFileExportTest, ExportCreatesFile) {
    std::vector<BarFeatureRow> rows;
    for (int i = 0; i < 5; ++i) {
        rows.push_back(make_feature_row(i));
    }

    ExportConfig config;
    config.output_path = temp_csv_path();
    FeatureExporter exporter(config);
    exporter.export_csv(rows);

    EXPECT_TRUE(std::filesystem::exists(temp_csv_path()));
}

TEST_F(CSVFileExportTest, ExportHasHeaderPlusDataLines) {
    std::vector<BarFeatureRow> rows;
    for (int i = 0; i < 5; ++i) {
        rows.push_back(make_feature_row(i));
    }

    ExportConfig config;
    config.output_path = temp_csv_path();
    FeatureExporter exporter(config);
    exporter.export_csv(rows);

    std::ifstream file(temp_csv_path());
    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    // Header + 5 data lines = 6 lines
    EXPECT_EQ(count_lines(content), 6);
}

TEST_F(CSVFileExportTest, ExportExcludesWarmupWhenConfigured) {
    std::vector<BarFeatureRow> rows;
    for (int i = 0; i < 10; ++i) {
        auto row = make_feature_row(i);
        row.is_warmup = (i < 5);  // First 5 are warmup
        rows.push_back(row);
    }

    ExportConfig config;
    config.output_path = temp_csv_path();
    config.include_warmup = false;
    FeatureExporter exporter(config);
    exporter.export_csv(rows);

    std::ifstream file(temp_csv_path());
    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    // Header + 5 non-warmup data lines = 6 lines
    EXPECT_EQ(count_lines(content), 6);
}

TEST_F(CSVFileExportTest, ExportIncludesWarmupWhenConfigured) {
    std::vector<BarFeatureRow> rows;
    for (int i = 0; i < 10; ++i) {
        auto row = make_feature_row(i);
        row.is_warmup = (i < 5);
        rows.push_back(row);
    }

    ExportConfig config;
    config.output_path = temp_csv_path();
    config.include_warmup = true;
    FeatureExporter exporter(config);
    exporter.export_csv(rows);

    std::ifstream file(temp_csv_path());
    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    // Header + 10 data lines = 11 lines
    EXPECT_EQ(count_lines(content), 11);
}

TEST_F(CSVFileExportTest, EmptyRowsProducesHeaderOnly) {
    std::vector<BarFeatureRow> rows;

    ExportConfig config;
    config.output_path = temp_csv_path();
    FeatureExporter exporter(config);
    exporter.export_csv(rows);

    std::ifstream file(temp_csv_path());
    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    // Just the header line
    EXPECT_EQ(count_lines(content), 1);
}

// ===========================================================================
// Rollover Exclusion
// ===========================================================================

class RolloverExclusionTest : public ::testing::Test {
protected:
    void TearDown() override {
        std::filesystem::remove(temp_csv_path());
    }
};

TEST_F(RolloverExclusionTest, RolloverDaysExcluded) {
    // Build rows spanning a rollover date
    std::vector<BarFeatureRow> rows;
    for (int day = 20220310; day <= 20220320; ++day) {
        auto row = make_feature_row(day - 20220310);
        row.day = day;
        rows.push_back(row);
    }

    RolloverCalendar cal;
    ContractSpec spec;
    spec.rollover_date = 20220315;
    spec.start_date = 20220101;
    spec.end_date = 20220630;
    cal.add_contract(spec);

    ExportConfig config;
    config.output_path = temp_csv_path();
    FeatureExporter exporter(config);
    exporter.export_csv(rows, cal);

    std::ifstream file(temp_csv_path());
    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());

    // Excluded days: 20220312, 20220313, 20220314, 20220315 (4 days)
    // Total rows = 11, excluded = 4, remaining = 7 + header = 8 lines
    EXPECT_EQ(count_lines(content), 8);
}

TEST_F(RolloverExclusionTest, NoRolloverCalendarExportsAll) {
    std::vector<BarFeatureRow> rows;
    for (int i = 0; i < 5; ++i) {
        rows.push_back(make_feature_row(i));
    }

    ExportConfig config;
    config.output_path = temp_csv_path();
    FeatureExporter exporter(config);
    exporter.export_csv(rows);  // No calendar → export all

    std::ifstream file(temp_csv_path());
    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    EXPECT_EQ(count_lines(content), 6);
}

// ===========================================================================
// Export Metadata Validation
// ===========================================================================

class ExportMetadataTest : public ::testing::Test {
protected:
    void TearDown() override {
        std::filesystem::remove(temp_csv_path());
    }
};

TEST_F(ExportMetadataTest, BarTypeIncluded) {
    auto row = make_feature_row();
    row.bar_type = "volume";

    FeatureExporter exporter;
    auto row_str = exporter.format_row(row);

    EXPECT_NE(row_str.find("volume"), std::string::npos);
}

TEST_F(ExportMetadataTest, BarParamIncluded) {
    auto row = make_feature_row();
    row.bar_param = 100.0f;

    FeatureExporter exporter;
    auto row_str = exporter.format_row(row);

    EXPECT_NE(row_str.find("100"), std::string::npos);
}

TEST_F(ExportMetadataTest, DayFieldIncluded) {
    auto row = make_feature_row();
    row.day = 20220103;

    FeatureExporter exporter;
    auto row_str = exporter.format_row(row);

    EXPECT_NE(row_str.find("20220103"), std::string::npos);
}

// ===========================================================================
// ExportConfig Validation
// ===========================================================================

class ExportConfigTest : public ::testing::Test {};

TEST_F(ExportConfigTest, DefaultIncludeWarmupIsFalse) {
    ExportConfig config;
    EXPECT_FALSE(config.include_warmup);
}

TEST_F(ExportConfigTest, DefaultOutputPathIsEmpty) {
    ExportConfig config;
    EXPECT_TRUE(config.output_path.empty());
}

TEST_F(ExportConfigTest, InvalidPathThrowsOnExport) {
    std::vector<BarFeatureRow> rows = {make_feature_row()};

    ExportConfig config;
    config.output_path = "/nonexistent/directory/file.csv";
    FeatureExporter exporter(config);

    EXPECT_THROW(exporter.export_csv(rows), std::runtime_error);
}

// ===========================================================================
// Streaming Export (row-by-row)
// ===========================================================================

class StreamingExportTest : public ::testing::Test {
protected:
    void TearDown() override {
        std::filesystem::remove(temp_csv_path());
    }
};

TEST_F(StreamingExportTest, WriteHeaderThenRows) {
    ExportConfig config;
    config.output_path = temp_csv_path();
    FeatureExporter exporter(config);

    exporter.begin();
    for (int i = 0; i < 3; ++i) {
        exporter.write_row(make_feature_row(i));
    }
    exporter.end();

    std::ifstream file(temp_csv_path());
    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    EXPECT_EQ(count_lines(content), 4);  // header + 3 rows
}

TEST_F(StreamingExportTest, EndWithoutRowsProducesHeaderOnly) {
    ExportConfig config;
    config.output_path = temp_csv_path();
    FeatureExporter exporter(config);

    exporter.begin();
    exporter.end();

    std::ifstream file(temp_csv_path());
    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    EXPECT_EQ(count_lines(content), 1);
}
