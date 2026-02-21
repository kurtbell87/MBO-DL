# Experiment: Full-Year Time_5s Feature Export (Parquet)

**Date:** 2026-02-19
**Priority:** BLOCKING — prerequisite for full-year model training. Everything downstream is on hold.

---

## Hypothesis

The bar_feature_export binary (with Parquet support from the parquet-export TDD cycle) will produce a complete, schema-consistent Parquet dataset covering all ~250 RTH trading days in 2022, with total rows >= 1,000,000, zero duplicate timestamps, and per-day values matching the reference 19-day CSV (`time_5s.csv`, 87,970 bars) within float64 tolerance (relative error <= 1e-10). Specifically: the existing single-day export pipeline scales linearly to the full year without systematic failures (failed days <= 5) or data corruption.

This is a **data engineering validation**, not a model training experiment. The "hypothesis" is that the export infrastructure is production-ready at full-year scale.

---

## Independent Variables

None. This is a deterministic data export — all parameters are fixed. There is no experimental manipulation.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Bar type | time_5s (`--bar-type time --bar-param 5`) | Locked by R1/R6 synthesis |
| Output format | Parquet (zstd compression) | Phase 8 TDD + parquet-export TDD |
| Date range | Full 2022 (all RTH trading days) | Maximum dataset for downstream model training |

---

## Controls

| Control | Value | Rationale |
|---------|-------|-----------|
| Binary | `./build/bar_feature_export` (Parquet-capable) | Phase 8 TDD output + parquet-export TDD additions |
| Data source | `DATA/GLBX-20260207-L953CAPU5B/*.dbn.zst` | Full 2022 MES MBO dataset, 312 daily files (~49 GB) |
| Reference CSV | `.kit/results/hybrid-model/time_5s.csv` | 87,970 bars, 19 days (S3-backed, SHA256: bf79d49b) |
| Instrument IDs | MESM2=13615 (Q1-Q2), MESU2=10039 (Q3) | Per `DATA/symbology.json` — need verification for Q4 (MESZ2) |
| Session | RTH only (09:30-16:00 ET) | Per DOMAIN_PRIORS.md |
| Snapshot interval | 100ms (10/s) | Per DOMAIN_PRIORS.md |
| Software | Current `main` branch + parquet-export TDD changes | No other code changes between reference CSV and this export |

---

## Metrics (ALL must be reported)

### Primary

- **total_days_exported**: Count of unique trading days with successfully exported Parquet files (expect ~250)
- **total_rows**: Total bar count across all Parquet files (expect ~1.15M, based on 87,970/19 * ~250)
- **schema_match**: Boolean — all Parquet files have identical column names in identical order, matching `time_5s.csv`

### Secondary

| Metric | Description |
|--------|-------------|
| per_day_bar_counts | min, max, mean, std of bars per day across all exported days |
| failed_days | List of days that failed export (with error messages) |
| existing_days_validated | Count of 19 reference days whose Parquet values match CSV within tolerance |
| duplicate_timestamps | Count of duplicate timestamps across entire dataset (expect 0) |
| trading_day_gaps | Count of missing weekdays not accounted for in `manifest.excluded_dates` |
| wall_clock_time | Total export wall-clock time (Phase 4 only) |
| total_disk_size_mb | Total Parquet output size on disk |
| compression_ratio | CSV-equivalent size / Parquet size |

### Sanity Checks

| Check | Expected | Failure interpretation |
|-------|----------|----------------------|
| bars_per_day within [3000, 6000] | ~4,630 mean (from 87,970/19) | Out-of-range days indicate session truncation, holiday data, or export bugs |
| timestamps strictly increasing per day | True for all files | Book builder or bar builder state corruption |
| is_warmup column present | True in all files | Schema drift between CSV and Parquet paths |
| Forward return columns finite for non-terminal bars | NaN/sentinel only at session boundaries | Forward return computation bug |
| Q1-Q4 day counts roughly balanced | ~60-65 days per quarter | Missing rollover days or instrument ID gaps |

---

## Baselines

| Baseline | Source | Value |
|----------|--------|-------|
| 19-day CSV reference | `.kit/results/hybrid-model/time_5s.csv` (S3-backed) | 87,970 bars across 19 days; schema = ground truth |
| Expected day count | CME 2022 RTH trading calendar | ~251 trading days (365 - weekends - ~9 market holidays) |
| Expected bar count | 19-day mean (4,630/day) * ~250 days | ~1,157,500 bars |

The 19-day CSV was produced by the same binary (pre-Parquet), validated in experiments 9B-9E, and serves as the schema reference. Parquet output must be value-identical for overlapping days.

---

## Success Criteria (immutable once RUN begins)

- [ ] **SC-1**: total_days_exported >= 240
- [ ] **SC-2**: total_rows >= 1,000,000
- [ ] **SC-3**: schema_match == True (identical column names in identical order, all files)
- [ ] **SC-4**: 19 original days validate against CSV within float64 tolerance (relative error <= 1e-10)
- [ ] **SC-5**: duplicate_timestamps == 0 across entire dataset
- [ ] **SC-6**: failed_days count <= 5
- [ ] **SC-7**: All Parquet files use zstd compression
- [ ] **SC-8**: Downstream polars read succeeds: `pl.scan_parquet("*.parquet").collect()` returns a DataFrame with shape ~(1.15M, N)
- [ ] **SC-9**: All tests (existing 1003+ unit + 28 new Parquet) pass after changes
- [ ] **SC-10**: manifest.json present with complete metadata (day count, bar counts, excluded dates, failed dates)

---

## Minimum Viable Experiment

Before the full parallel export, validate the pipeline on a minimal subset:

1. **Build gate**: `cmake --build build -j12` succeeds. All tests (1003+ unit + 28 Parquet) pass.
2. **Single new-day export**: Export one day NOT in the original 19-day CSV (e.g., 2022-01-03, the first RTH trading day of 2022) as Parquet. Verify:
   - File is valid Parquet (readable by Arrow C++)
   - Schema matches reference CSV columns
   - Row count in [3000, 6000] range
   - No duplicate timestamps
   - Timestamps monotonically increasing
3. **Cross-format spot check**: Export one of the original 19 days as BOTH Parquet and CSV. Compare values. All must match within float64 tolerance.

**Gate**: If ANY MVE check fails, do NOT proceed to full export. Diagnose and fix first.

---

## Full Protocol

### Phase 0: Repository Analysis — COMPLETE

**Status:** Done. 312 .dbn.zst files identified. ~250 RTH trading days.

1. Read `tools/bar_feature_export.cpp` and related headers. Understand file discovery, date handling, rollover logic, output path.
2. List all 312 `.dbn.zst` files. Extract dates. Filter to valid RTH trading days — exclude weekends, holidays, rollover transition days.
3. Identify which 19 days are in the existing `time_5s.csv`.
4. Verify project builds and all existing tests pass.

### Phase 1: Add Arrow/Parquet Dependency — COMPLETE

**Status:** Done. `CMakeLists.txt` updated to use `find_package(Arrow REQUIRED)` / `find_package(Parquet REQUIRED)` with system-installed apache-arrow (`brew install apache-arrow`). FetchContent approach abandoned per institutional memory (Arrow via FetchContent added 15-30 min per build).

**Gate:** Project builds. All existing tests pass. No existing code modified.

### Phase 2: Parquet Writer Implementation — COMPLETE

**Status:** Done. `tools/bar_feature_export.cpp` detects `.parquet` extension and writes Parquet output using Arrow C++ (zstd compression, DOUBLE columns, INT64 timestamps). CSV path unchanged.

### Phase 3: TDD — COMPLETE

**Status:** Done. 28 tests in `tests/parquet_export_test.cpp` (GTest). TDD spec at `.kit/docs/parquet-export.md` executed successfully.

Tests cover: round-trip fidelity (T1), schema match (T2), CSV-vs-Parquet value comparison (T3), row count match (T4), no duplicate timestamps (T5), monotonic timestamps (T6), warm-up bars (T7), terminal bar consistency (T8), parallel determinism (T9), no cross-day leakage (T10), full regression suite (T11), CSV byte-identity (T12).

**Gate:** All 28 tests pass. Do not proceed to Phase 4 with any test failures.

### Phase 4: Full Parallel Export

1. **Build first**: `cmake --build build -j12`. Verify all tests pass (MVE gate).
2. **Run MVE**: Single new-day export + cross-format spot check (see Minimum Viable Experiment above).
3. **Full export**: Export all valid 2022 RTH trading days as individual Parquet files.
   - Parallelism: N-1 CPUs concurrent
   - Per day: `./build/bar_feature_export --bar-type time --bar-param 5 --input {dbn_file} --output .kit/results/full-year-export/{date}.parquet`
   - Log per day: `.kit/results/full-year-export/logs/{date}.log`
   - Track success/failure per day

4. **Generate manifest.json** after all days complete:
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

5. **Output structure**:
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

**Do NOT merge into a single Parquet file.** Per-day files are canonical. Downstream: `pl.scan_parquet("full-year-export/*.parquet")`.

### Phase 5: Validation

1. **Row count**: total bars across all files ~1.15M (±10%). Report per-day min, max, mean, std.
2. **Day count**: total exported days >= 240.
3. **No gaps**: all weekdays between first and last exported date are either exported or listed in `manifest.excluded_dates` with a reason.
4. **No duplicates**: zero duplicate timestamps across entire dataset.
5. **Schema match**: every Parquet file has identical column names and types.
6. **19-day spot check**: For the 19 days overlapping with the reference CSV, read both formats and compare all values. Must match within float64 tolerance (relative error <= 1e-10).
7. **Downstream read check**: `python3 -c "import polars as pl; df = pl.scan_parquet('.kit/results/full-year-export/*.parquet'); print(df.collect().shape)"` — must succeed and report ~(1,150,000, N_columns).
8. **Quarter balance check**: Count exported days per quarter. Report any quarter with <50 days (would indicate rollover or instrument ID issues).

### Phase 6: Report

Write `.kit/results/full-year-export/EXPORT_LOG.md`:
- Total days exported, total bars
- Excluded dates with reasons
- Failed dates with error details
- Wall-clock time for Phase 4
- Parquet schema (column names and types)
- Per-quarter day counts
- How to read data in Python (`polars`) and C++ (`Arrow`)
- Disk size and compression ratio

---

## Resource Budget

**Tier:** Standard

### Compute Profile
```yaml
compute_type: cpu
estimated_rows: 1150000
model_type: other
sequential_fits: 0
parallelizable: true
memory_gb: 8
gpu_type: none
estimated_wall_hours: 1.5
```

### Wall-Time Estimation

- **Per-day processing**: Read ~157MB `.dbn.zst` → decompress → build order book → compute features at 5s intervals → write ~2-3MB Parquet. Estimated ~2 min/day.
- **Sequential total**: 250 days * 2 min = 500 min = ~8.3 hours
- **Parallel (11 cores)**: 500 / 11 = ~45 min
- **Validation overhead**: ~15 min (read all files, spot checks, manifest generation)
- **Estimated wall clock**: ~1.0–1.5 hours (Phases 4+5+6)
- **Max budget**: 4 hours (abort criterion, generous to avoid false aborts)

### Other Limits
- Max wall-clock time: 4 hours
- Max training runs: 0 (no model training)
- Max seeds: 1 (deterministic export)

---

## Abort Criteria

- **Build failure**: `cmake --build` fails or any test (existing + new) fails → stop. Do not proceed with broken tests.
- **MVE failure**: Single-day Parquet export produces invalid schema, wrong row count, or duplicate timestamps → stop. Parquet writer has a bug. Fix via TDD retry before proceeding.
- **Systematic export failure**: >20% of days fail (>50 days) → stop and diagnose. Likely an instrument ID, rollover, or data format issue.
- **Wall-clock overrun**: Export phase exceeds 4 hours → abort remaining days, report partial results.
- **Schema drift**: Any Parquet file has different columns than the reference CSV → stop. Indicates non-deterministic writer behavior.
- **19-day validation failure**: Any reference day's Parquet values differ from CSV beyond tolerance → stop. Parquet writer is silently corrupting data.

---

## Confounds to Watch For

1. **Instrument rollover transitions**: MES rolls quarterly. DOMAIN_PRIORS.md lists MESM2 (Q1-Q2) and MESU2 (Q3), but the full year requires 4 contracts (MESH2, MESM2, MESU2, MESZ2). If `bar_feature_export` only filters for known instrument IDs, Q4 days (MESZ2) may export with 0 bars. The READ agent should check for instrument_id coverage across all 4 quarters and flag any quarter with anomalously low bar counts.

2. **Half-day sessions**: CME has early closures (day before Thanksgiving, Christmas Eve, July 3). These produce legitimate but short sessions (~2,400 bars vs ~4,600). Should NOT be counted as failures — but the per-day bar count distribution will have a left tail. The validation should expect min_bars_per_day < 3,000 for a few days.

3. **Overnight-only .dbn.zst files**: Some dates may have a .dbn.zst file containing only overnight session data (e.g., if RTH was a holiday but Globex ran overnight). These would produce 0 RTH bars and should be excluded, not counted as export failures.

4. **Cross-day state leakage in parallel export**: Each `bar_feature_export` invocation must be fully independent (no shared state). T10 tested this, but at scale with N-1 concurrent processes, file system contention or memory pressure could cause non-deterministic behavior. Compare: if any day's output differs when re-exported sequentially, there's a leakage bug.

5. **Warm-up bar fraction**: The first ~60s of each day has warm-up bars (book not fully reconstructed). If the warm-up fraction varies significantly across days, downstream train/test splits may have different effective sample sizes. The READ agent should report the warm-up bar fraction per day.

6. **Reference CSV staleness**: The 19-day `time_5s.csv` was generated before the tick-bar-fix TDD cycle. If the tick-bar fix inadvertently changed anything in the time bar code path (it shouldn't — different bar type), the reference comparison would fail for the right reason (detecting an unintended regression) but could be misinterpreted as a Parquet writer bug.

---

## Scope Constraints

- **DO NOT** modify existing bar builder, feature computation, or book builder code
- **DO NOT** break the existing CSV export path — it must continue working
- **DO NOT** merge per-day Parquet files into a single file
- **DO NOT** export any bar type other than time_5s
- **DO NOT** analyze the exported data or run models
- **DO NOT** touch `.kit/experiments/` or other `.kit/results/` directories
- **DO** keep the original `time_5s.csv` intact for validation reference
- **DO** include is_warmup bars (downstream filtering handles them)
- **DO** push large result files (>10MB) to S3 artifact store before committing
