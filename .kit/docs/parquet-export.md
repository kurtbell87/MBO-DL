# TDD Spec: Parquet Output for bar_feature_export

**Date:** 2026-02-19
**Priority:** BLOCKING — prerequisite for full-year data export

---

## Context

`bar_feature_export` is an existing C++ tool that exports bar features to CSV. It accepts `--bar-type`, `--bar-param`, and `--output` flags. Currently it writes CSV only.

The goal is to add Parquet output support (Apache Arrow C++) so the tool can write per-day Parquet files for the full 2022 calendar (~250 trading days, ~1.15M bars). The existing CSV path must remain unchanged.

### What exists

- **Binary**: `./build/bar_feature_export --bar-type time --bar-param 5 --output <path>.csv`
- **Reference output**: `.kit/results/hybrid-model/time_5s.csv` — 87,970 bars, 19 days
- **Data**: `DATA/GLBX-20260207-L953CAPU5B/*.dbn.zst` — 312 daily MES MBO files
- **Build system**: CMake with FetchContent (libtorch, databento-cpp, xgboost, GTest)
- **Test framework**: GTest (1003+ unit tests pass, 22 integration tests labeled)

---

## Requirements

### R1: Add Apache Arrow C++ Build Dependency

Add Apache Arrow C++ via FetchContent (matching the existing dependency pattern). Minimal configuration — only what's needed for Parquet writing with zstd compression:

```cmake
set(ARROW_BUILD_STATIC ON)
set(ARROW_PARQUET ON)
set(ARROW_WITH_ZSTD ON)
set(ARROW_COMPUTE OFF)
set(ARROW_DATASET OFF)
set(ARROW_FLIGHT OFF)
set(ARROW_JSON OFF)
set(ARROW_CSV OFF)
set(ARROW_FILESYSTEM ON)
set(PARQUET_BUILD_EXAMPLES OFF)
set(PARQUET_BUILD_EXECUTABLES OFF)
```

**Constraints:**
- Must not modify any existing source files or test files
- Must not break any existing build targets
- All existing 1003+ tests must continue passing after adding the dependency
- Link Arrow/Parquet only to the `bar_feature_export` target (and test targets that need it)

### R2: Parquet Output in bar_feature_export

Add a Parquet output path to `bar_feature_export`:

- **Format detection**: Use the file extension of `--output` to choose format:
  - `.csv` → existing CSV writer (unchanged, byte-identical behavior)
  - `.parquet` → new Parquet writer
- **Schema**: Parquet column names and order must exactly match the CSV columns
- **Types**: Float columns as DOUBLE. Timestamp column as INT64 or UINT64, matching the CSV numeric convention (nanoseconds epoch)
- **Compression**: zstd
- **Row groups**: All rows for one day in a single row group (each invocation exports one day)
- **Invocation**: `./build/bar_feature_export --bar-type time --bar-param 5 --output {path}.parquet`

**The existing CSV output must be completely unchanged.** This is additive only.

### R3: Do NOT modify existing code

Do not modify:
- Bar builder code (bar_builder_base.hpp, time_bar_builder.hpp, etc.)
- Feature computation code (bar_features.hpp, bar_features.cpp)
- Book builder code (book_builder.hpp)
- Oracle replay or backtest code
- Any existing test files

The only files that should change:
- `CMakeLists.txt` (add Arrow dependency, link to targets)
- `tools/bar_feature_export.cpp` (add Parquet writer alongside existing CSV writer)
- New test file(s) for Parquet-specific tests

---

## Tests

Use the existing test framework (GTest). New Parquet tests should be in a new test file. Integration tests that need data files should be labeled `integration` (matching the existing convention).

### Format Tests

**T1: Parquet round-trip** — Export one day as Parquet. Read back using Arrow C++ reader. All double values must be bitwise identical (no precision loss from write/read cycle).

**T2: Schema match** — Parquet file column names and types must match the CSV column names. Same count, same order. Read the Parquet schema (Arrow metadata) and compare against a known reference (the CSV header).

**T3: CSV vs Parquet value comparison** — Export one day as both CSV and Parquet. Read both into memory. All values must match within float64 tolerance (relative error <= 1e-10). This validates the Parquet path produces the same data as the known-good CSV path.

### Export Pipeline Tests

**T4: Row count match** — Single-day Parquet row count equals single-day CSV row count for the same date.

**T5: No duplicate timestamps** — Zero duplicate timestamps within a single-day Parquet file.

**T6: Timestamps monotonically increasing** — All timestamps in a single-day Parquet file are strictly increasing.

**T7: Warm-up bars present** — The `is_warmup` column exists and contains both true and false values (warm-up bars at start of session, non-warmup for the rest).

**T8: Terminal bar consistency** — Forward return columns at end-of-session: terminal bars in Parquet match terminal bars in CSV (same NaN or sentinel pattern for bars where forward returns can't be computed).

### Parallelization Tests

**T9: Parallel vs sequential** — Two days exported as Parquet in parallel produce identical output to sequential export (byte-identical files).

**T10: No cross-day state leakage** — Export day D alone vs export days D-1 then D sequentially. Day D output must be identical in both cases.

### Regression Tests

**T11: All existing tests pass** — The full existing test suite (1003+ unit tests) passes after changes. Verified by running ctest with integration label excluded.

**T12: CSV byte-identical** — CSV export for one day after Parquet addition produces byte-identical output to the same day exported before the Parquet addition. (Save a reference CSV before changes, compare after.)

---

## Exit Criteria

- [ ] EC-1: Apache Arrow C++ builds successfully via FetchContent
- [ ] EC-2: All existing 1003+ unit tests pass unchanged
- [ ] EC-3: `bar_feature_export --output foo.parquet` produces a valid Parquet file
- [ ] EC-4: `bar_feature_export --output foo.csv` produces byte-identical CSV output to before
- [ ] EC-5: Parquet schema (column names, order, types) matches CSV schema
- [ ] EC-6: Parquet values match CSV values within float64 tolerance (T3)
- [ ] EC-7: All new tests T1-T10 pass
- [ ] EC-8: No modifications to existing bar builder, feature, book builder, or test files
- [ ] EC-9: Parquet files use zstd compression
- [ ] EC-10: Build completes without warnings in new code (treat warnings as errors)

---

## Notes for Implementation

- The Arrow FetchContent build can be slow (~10-15 min first time). Be patient.
- Use `arrow::io::FileOutputStream` and `parquet::arrow::FileWriter` for writing.
- The Parquet writer should be a simple function: takes the same data vectors the CSV writer uses, writes them as a Parquet table.
- Test data: use any single day from `DATA/GLBX-20260207-L953CAPU5B/`. The existing 19-day set includes dates like 2022-06-27 through 2022-07-22 (check the data directory for exact dates).
- For T12 (byte-identical CSV): generate a reference CSV from a single day BEFORE making code changes, save it, then compare after changes.
- Integration tests should be labeled so they're excluded from default `ctest` runs (matching existing convention).
