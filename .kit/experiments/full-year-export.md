# Experiment: Full-Year Time_5s Feature Export (Parquet)

## Hypothesis

The existing `bar_feature_export` binary can be parallelized across all 2022 MES RTH trading days to produce a full-year Parquet dataset (~250 days, ~1.16M bars) with identical schema to the existing 19-day export (87,970 bars at `.kit/results/hybrid-model/time_5s.csv`).

## Objective

This is a **compute/data task**, not a hypothesis-driven experiment. The goal is to scale the existing 19-day feature export to the full 2022 trading calendar. **Output format: Parquet** (not CSV) — better compression, typed columns, faster downstream reads at ~1.15M rows.

### What exists

- **Export binary**: `./build/bar_feature_export --bar-type time --bar-param 5 --output <path>`
- **Existing output**: `.kit/results/hybrid-model/time_5s.csv` — 87,970 bars, 19 days (CSV — used as schema reference only)
- **Data source**: `DATA/GLBX-20260207-L953CAPU5B/` — 312 daily `.dbn.zst` files, MES MBO 2022 (~49 GB)
- **Export command from LAST_TOUCH.md**: `./build/bar_feature_export --bar-type time --bar-param 5 --output .kit/results/hybrid-model/time_5s.csv`

### What needs to happen

1. **Understand** why the existing export produced only 19 days from 312 files. Read the binary's source (`tools/bar_feature_export.cpp`) and related infrastructure to understand:
   - How it discovers/selects data files
   - Whether it accepts per-file or per-date arguments
   - How rollover days and non-RTH data are handled
   - Any date range filtering
   - Whether it supports Parquet output natively (TRAJECTORY.md §10.2 mentions `feature_export.hpp` supports "Parquet/CSV")

2. **Export all available days**. Each day is independent — session boundaries reset all state (EWMA, warm-up counters). Parallelize across days using background processes, GNU parallel, xargs, or similar.

3. **Validate and merge** into a single Parquet file at `.kit/results/hybrid-model/time_5s_full.parquet`.

4. **Format**: If the C++ binary supports `--output *.parquet` natively, use that directly. If it only supports CSV, export per-day CSVs then convert to a single Parquet file using Python (pandas/pyarrow). The final deliverable MUST be Parquet.

## Independent Variables

1. **Date**: All 2022 MES RTH trading days available in the data directory

## Controls

| Control | Value | Rationale |
|---------|-------|-----------|
| Bar type | time_5s (--bar-type time --bar-param 5) | Locked by R1/R6 synthesis |
| Binary | ./build/bar_feature_export (existing, already compiled) | Phase 8 TDD output |
| Data source | DATA/GLBX-20260207-L953CAPU5B/*.dbn.zst | Full 2022 MES MBO dataset |

## Metrics (ALL must be reported)

### Primary

1. **total_days_exported**: Number of unique trading days in final output (expect ~250)
2. **total_rows**: Total bar count in final output (expect ~1.15M at ~4,600 bars/day)
3. **schema_match**: Boolean — does the full-year Parquet have identical columns (names and types) to the 19-day CSV?

### Secondary

| Metric | Description |
|--------|-------------|
| per_day_bar_counts | Min, max, mean, std of bars per day |
| failed_days | List of days that failed to export (if any) |
| existing_days_validated | Boolean — do the 19 original days in the full export match the existing file exactly? |
| duplicate_timestamps | Count of duplicate timestamps (expect 0) |
| trading_day_gaps | List of missing weekdays (should only be holidays) |
| wall_clock_time | Total export time |

## Success Criteria

- [ ] **SC-1**: total_days_exported >= 240 (accounting for holidays, rollover exclusions)
- [ ] **SC-2**: total_rows >= 1,000,000
- [ ] **SC-3**: schema_match == True (identical column names in identical order)
- [ ] **SC-4**: existing_days_validated == True (19 original days produce identical rows when compared)
- [ ] **SC-5**: duplicate_timestamps == 0
- [ ] **SC-6**: failed_days count <= 5 (a few failures acceptable; systematic failures are not)
- [ ] **SC-7**: Final output written to `.kit/results/hybrid-model/time_5s_full.parquet` (Parquet format)

## Protocol

### Step 0: Understand the export binary

- Read `tools/bar_feature_export.cpp` and any related headers
- Determine how it selects data files, handles date ranges, rollover
- Determine the correct invocation to export a single day or all days
- Identify if the binary supports per-file input or needs modification

### Step 1: Identify all available trading days

- List all `.dbn.zst` files in `DATA/GLBX-20260207-L953CAPU5B/`
- Extract dates, count them, identify which are RTH trading days
- Identify which 19 days are in the existing export
- List any days that should be excluded (weekends, holidays, rollover transition)

### Step 2: Single-day validation

- Export one day NOT in the existing 19-day set
- Verify output schema matches exactly
- Verify bar count is reasonable (~4,600 bars)

### Step 3: Parallel full export

- Export all remaining days (skip the 19 already done if the binary supports single-day export; otherwise re-export everything)
- Parallelize: N concurrent processes = available CPUs / 2 (leave headroom)
- Write each day to a separate temp file (CSV or Parquet depending on binary support)
- Log stdout/stderr per day
- Track success/failure per day

### Step 4: Merge to Parquet and validate

- If per-day outputs are CSV: load all with pandas/pyarrow, concatenate, write single Parquet
- If per-day outputs are Parquet: merge with pyarrow directly
- Sort by timestamp, write to `.kit/results/hybrid-model/time_5s_full.parquet`
- Use snappy compression (pyarrow default)
- Validate:
  - Row count is reasonable
  - No duplicate timestamps
  - Column schema matches 19-day CSV file (same column names, compatible types)
  - Spot-check: rows for the original 19 days match the existing CSV file
  - Identify any gap days (missing weekdays that aren't holidays)

### Step 5: Report

- Write summary to `.kit/results/full-year-export/analysis.md`
- Write metrics to `.kit/results/full-year-export/metrics.json`

## Resource Budget

**Tier:** Standard

- Max GPU-hours: 0 (CPU only)
- Max wall-clock time: 4 hours (generous for local execution)
- Parallelism: Up to N/2 CPUs for concurrent exports

### Compute Profile
```yaml
compute_type: cpu
estimated_rows: 1150000
model_type: none
sequential_fits: 0
parallelizable: true
memory_gb: 8
gpu_type: none
estimated_wall_hours: 2.0
```

## Abort Criteria

- If more than 20% of days fail to export: something is systematically wrong with the binary or data. Abort and diagnose.
- If the binary cannot be invoked for individual days (only bulk): run in bulk mode and accept the full re-export.
- If wall-clock exceeds 4 hours: abort remaining days, report what completed.

## Deliverables

```
.kit/results/hybrid-model/
  time_5s_full.parquet      # Full-year export (~1.15M rows, Parquet, snappy compression)
  time_5s.csv               # Original 19-day export (kept for validation reference)

.kit/results/full-year-export/
  analysis.md               # Summary report
  metrics.json              # Validation metrics
  per_day_counts.csv        # Bars per day
  failed_days.txt           # Any days that failed (if applicable)
```

## Notes

- Keep the original `time_5s.csv` (19 days) intact for validation reference
- `is_warmup` bars at session start are expected and should be included (downstream filtering handles them)
- EWMA state resets per session — each day is fully independent
- If any day fails, log it and continue. Report failures at the end.
- This is a prerequisite for Phase B (hybrid model training on full-year data)
