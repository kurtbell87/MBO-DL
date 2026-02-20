# Experiment: Full-Year Time_5s Feature Export (Parquet)

**Date:** 2026-02-19
**Priority:** BLOCKING — prerequisite for full-year model training. Everything else is on hold.

---

## Objective

Scale the existing 19-day feature export (87,970 bars at `.kit/results/hybrid-model/time_5s.csv`) to the full 2022 MES trading calendar. Output: per-day Parquet files with zstd compression, written from C++ using Apache Arrow.

This is a **compute/engineering task**, not a hypothesis-driven experiment.

### Current Status (2026-02-19)

**Phases 0-3 COMPLETE.** Code changes are done:
- `CMakeLists.txt`: Arrow dependency switched from FetchContent → `find_package(Arrow REQUIRED)` / `find_package(Parquet REQUIRED)` using system brew-installed apache-arrow
- `tools/bar_feature_export.cpp`: Parquet writer implemented (detects `.parquet` extension)
- `tests/parquet_export_test.cpp`: 28 GTest tests written

**NEXT: Build (`cmake --build build -j12`) then execute Phases 4-6 (export, validate, report).**

### What exists

- **Export binary**: `./build/bar_feature_export --bar-type time --bar-param 5 --output <path>`
- **Existing output**: `.kit/results/hybrid-model/time_5s.csv` — 87,970 bars, 19 days (CSV — used as schema reference and validation)
- **Data source**: `DATA/GLBX-20260207-L953CAPU5B/` — 312 daily `.dbn.zst` files, MES MBO 2022 (~49 GB)

### What needs to happen

1. Add Apache Arrow C++ (Parquet writer only) as a build dependency
2. Add Parquet output support to bar_feature_export
3. Test the Parquet writer against existing CSV output
4. Export all 2022 RTH trading days in parallel
5. Validate the full dataset

---

## Controls

| Control | Value | Rationale |
|---------|-------|-----------|
| Bar type | time_5s (--bar-type time --bar-param 5) | Locked by R1/R6 synthesis |
| Binary | ./build/bar_feature_export (with Parquet support added) | Phase 8 TDD output + this spec's additions |
| Data source | DATA/GLBX-20260207-L953CAPU5B/*.dbn.zst | Full 2022 MES MBO dataset |

---

## Protocol

### Phase 0: Repository Analysis — COMPLETE

**Status:** Done. 312 .dbn.zst files identified. ~250 RTH trading days.

Before writing any code:

1. Read `tools/bar_feature_export.cpp` and related headers. Understand:
   - How it discovers/selects data files
   - Whether it accepts per-file, per-date, or date-range arguments
   - How rollover days and non-RTH data are handled
   - The current output path (CSV writer location in code)
2. List all 312 `.dbn.zst` files. Extract dates. Filter to valid RTH trading days — exclude weekends, holidays, rollover transition days, non-RTH sessions. Report the final count of exportable days.
3. Identify which 19 days are in the existing time_5s.csv.
4. Verify the project builds and all existing tests pass (1003+). Record the test count.

Write findings before proceeding.

### Phase 1: Add Arrow/Parquet Dependency — COMPLETE

**Status:** Done. CMakeLists.txt updated to use `find_package(Arrow REQUIRED)` / `find_package(Parquet REQUIRED)` with system-installed apache-arrow (`brew install apache-arrow`). FetchContent block removed.

Add Apache Arrow C++ as a build dependency. Minimal configuration:

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

Use whatever dependency mechanism the project already uses (FetchContent, vcpkg, system packages). Match the existing pattern.

**Gate:** Project builds. All existing tests still pass. No existing code modified.

### Phase 2: Parquet Writer Implementation — COMPLETE

**Status:** Done. `tools/bar_feature_export.cpp` now detects `.parquet` extension and writes Parquet output using Arrow C++ (zstd compression, DOUBLE columns, INT64 timestamps). CSV path unchanged.

Add a Parquet output path to bar_feature_export:

- Detect output format from file extension: `.csv` → existing CSV writer (unchanged), `.parquet` → new Parquet writer
- Parquet schema matches CSV columns exactly: same names, same order
- Float columns as DOUBLE (matching C++ double precision)
- Timestamp column as UINT64 or INT64, matching existing CSV convention
- Compression: zstd
- Row group size: all rows for one day in a single row group (simplest, each file is one day)
- Invocation: `./build/bar_feature_export --bar-type time --bar-param 5 --output {date}.parquet`

The existing CSV output path must continue working identically. This is additive, not a replacement.

### Phase 3: TDD — COMPLETE

**Status:** Done. 28 tests written in `tests/parquet_export_test.cpp` (GTest). TDD spec at `.kit/docs/parquet-export.md` executed successfully.

Write tests BEFORE the Parquet writer implementation. Use the existing test framework (Catch2).

**Format tests:**
- T1: Export one day as Parquet via Arrow C++ reader. Read back. All values round-trip correctly (bitwise identical doubles).
- T2: Parquet file column names and types match time_5s.csv column names. Same count, same order.
- T3: Export one day as both CSV and Parquet. Read both. All values match within float64 tolerance (≤1e-10 relative error). This validates the Parquet path against the known-good CSV path.

**Export pipeline tests:**
- T4: Single-day Parquet row count matches single-day CSV row count for the same date.
- T5: No duplicate timestamps within a day.
- T6: Timestamps monotonically increasing within a day.
- T7: Warm-up bars present and correctly flagged (is_warmup column).
- T8: Forward returns at end-of-session: terminal bars handled consistently between CSV and Parquet.

**Parallelization tests:**
- T9: Two days exported as Parquet in parallel produce identical output to sequential export.
- T10: No cross-day state leakage: export day D alone vs. export days D-1,D → day D output is identical.

**Regression tests:**
- T11: All existing tests (1003+) still pass.
- T12: CSV export for one day produces byte-identical output to before the Parquet addition.

Run RED phase: all tests written, all fail (Parquet writer not yet implemented).
Run GREEN phase: implement Parquet writer. All tests pass.

**Gate:** All tests pass. Do not proceed to Phase 4 with any test failures.

### Phase 4: Full Parallel Export

1. Export all valid 2022 RTH trading days as individual Parquet files.
   - N concurrent processes = available CPUs - 1
   - Each process: `./build/bar_feature_export --bar-type time --bar-param 5 --input {dbn_file} --output .kit/results/full-year-export/{date}.parquet`
   - Log stdout/stderr per day to `.kit/results/full-year-export/logs/{date}.log`
   - Track success/failure per day

2. After all days complete, generate `manifest.json`:
   ```json
   {
     "bar_type": "time_5s",
     "compression": "zstd",
     "total_days": 250,
     "total_bars": 1150000,
     "schema_columns": ["timestamp", "open", "high", "low", "close", ...],
     "dates": [
       {"date": "2022-01-03", "bars": 4631, "file": "2022-01-03.parquet"},
       ...
     ],
     "excluded_dates": [
       {"date": "2022-01-17", "reason": "MLK holiday"},
       ...
     ],
     "failed_dates": []
   }
   ```

3. Output directory structure:
   ```
   .kit/results/full-year-export/
   ├── manifest.json
   ├── logs/
   │   ├── 2022-01-03.log
   │   └── ...
   ├── 2022-01-03.parquet
   ├── 2022-01-04.parquet
   └── ...
   ```

**Do NOT merge into a single Parquet file.** Per-day files are the canonical format. Downstream reads use `pl.scan_parquet("full-year-export/*.parquet")` which handles the directory natively with lazy evaluation.

### Phase 5: Validation

1. **Row count**: total bars across all files ≈ 1.15M (±10%). Report per-day min, max, mean, std.
2. **Day count**: total exported days ≥ 240 (accounting for holidays, rollover exclusions).
3. **No gaps**: all weekdays between first and last exported date are either exported or listed in manifest.excluded_dates with a reason.
4. **No duplicates**: zero duplicate timestamps across the entire dataset.
5. **Schema match**: every Parquet file has identical column names and types.
6. **19-day spot check**: for the 19 days that overlap with the original CSV, read both formats and compare all values. Parquet values must match CSV values within float64 tolerance (≤1e-10 relative). Differences beyond this indicate a bug in the Parquet writer.
7. **Downstream read check**: `python3 -c "import polars as pl; df = pl.scan_parquet('.kit/results/full-year-export/*.parquet'); print(df.collect().shape)"` — must succeed and report approximately (1150000, N_columns).

### Phase 6: Report

Write summary to `.kit/results/full-year-export/EXPORT_LOG.md`:
- Total days exported, total bars
- Excluded dates with reasons
- Any failed dates with error details
- Wall-clock time
- How to read the data in C++ and Python
- Parquet schema (column names and types)

---

## Metrics (ALL must be reported)

### Primary
- **total_days_exported**: unique trading days in output (expect ~250)
- **total_rows**: total bar count across all files (expect ~1.15M)
- **schema_match**: all Parquet files have identical columns to time_5s.csv

### Secondary
| Metric | Description |
|--------|-------------|
| per_day_bar_counts | min, max, mean, std of bars per day |
| failed_days | list of days that failed export |
| existing_days_validated | 19 original days match CSV within tolerance |
| duplicate_timestamps | count (expect 0) |
| trading_day_gaps | missing weekdays not accounted for in excluded_dates |
| wall_clock_time | total export time |

---

## Success Criteria (immutable once RUN begins)

- [ ] **SC-1**: total_days_exported >= 240
- [ ] **SC-2**: total_rows >= 1,000,000
- [ ] **SC-3**: schema_match == True (identical column names in identical order, all files)
- [ ] **SC-4**: 19 original days validate against CSV within float64 tolerance (≤1e-10 relative)
- [ ] **SC-5**: duplicate_timestamps == 0
- [ ] **SC-6**: failed_days count <= 5
- [ ] **SC-7**: All Parquet files use zstd compression
- [ ] **SC-8**: Downstream polars read succeeds
- [ ] **SC-9**: All tests (existing + new) pass after the change
- [ ] **SC-10**: manifest.json present with complete metadata

---

## Resource Budget

### Compute Profile
```yaml
compute_type: cpu
estimated_rows: 1150000
model_type: none
sequential_fits: 0
parallelizable: true
memory_gb: 8
gpu_type: none
estimated_wall_hours: 4.0
```

- Max wall-clock: 4 hours (generous for local multi-core execution)
- Parallelism: N-1 CPUs (each day is single-threaded, memory-light)

---

## Abort Criteria

- Arrow dependency fails to build or breaks existing tests → stop. Report build errors. Do not attempt workarounds that modify existing code.
- T3 (CSV vs Parquet comparison) fails → Parquet writer has a bug. Fix before proceeding to export.
- More than 20% of days fail to export → systematic issue. Stop and diagnose.
- Wall-clock exceeds 4 hours → abort remaining days, report what completed.
- Any existing test (1003+) fails after changes → stop. Do not proceed with broken tests.

---

## Scope Constraints

- **DO NOT** modify existing bar builder, feature computation, or book builder code
- **DO NOT** break the existing CSV export path — it must continue working
- **DO NOT** merge per-day Parquet files into a single file
- **DO NOT** export any bar type other than time_5s
- **DO NOT** analyze the exported data or run models
- **DO NOT** touch .kit/experiments/ or other .kit/results/ directories
- **DO** keep the original time_5s.csv intact for validation reference
- **DO** include is_warmup bars (downstream filtering handles them)
