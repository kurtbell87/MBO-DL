# bar_feature_export — TDD Spec

## Summary

C++ CLI tool that exports bar-level feature CSVs for arbitrary bar types. Parameterized variant of `info_decomposition_export.cpp`. Pipeline: StreamingBookBuilder → BarFactory → BarFeatureComputer → CSV.

**Template**: `tools/info_decomposition_export.cpp` — copy StreamingBookBuilder, SELECTED_DAYS, MES_CONTRACTS, utility functions verbatim. Replace hardcoded `BarFactory::create("time", 5.0)` with CLI-parameterized call.

## CLI Interface

```
./bar_feature_export --bar-type <type> --bar-param <threshold> --output <csv_path>
```

| Arg | Type | Required | Description |
|-----|------|----------|-------------|
| `--bar-type` | string | yes | Bar type: "time", "volume", "dollar", "tick" |
| `--bar-param` | double | yes | Bar threshold (e.g., 5.0 for time_5s, 100 for volume_100, 25000 for dollar_25k) |
| `--output` | string | yes | Output CSV file path |

Exit code 0 on success, 1 on error (missing args, bad bar type, file open failure).

## Input

- 19 daily `.dbn.zst` files from `DATA/GLBX-20260207-L953CAPU5B/` (same SELECTED_DAYS as `info_decomposition_export.cpp` lines 261-264)
- MBO messages decoded via `databento::DbnFileStore`

## Output

Single CSV file with identical schema to `.kit/results/info-decomposition/features.csv`:

### CSV Header

```
timestamp,bar_type,bar_param,day,is_warmup,bar_index,<62 Track A>,<40 book_snap>,<33 msg_summary>,return_1,return_5,return_20,return_100,mbo_event_count
```

- **Metadata** (6 cols): `timestamp`, `bar_type`, `bar_param`, `day`, `is_warmup`, `bar_index`
- **Track A** (62 cols): from `BarFeatureRow::feature_names()` — same as info_decomposition_export
- **Book snapshot** (40 cols): `book_snap_0` .. `book_snap_39` — from `BookSnapshotExport::flatten(bar)`
- **Message summary** (33 cols): `msg_summary_0` .. `msg_summary_32` — from MBO events
- **Forward returns** (4 cols): `return_1`, `return_5`, `return_20`, `return_100` — from `fwd_return_{1,5,20,100}`
- **Event count** (1 col): `mbo_event_count`

Total: 6 + 62 + 40 + 33 + 4 + 1 = 146 columns.

### Warmup

First 50 bars per day (`WARMUP_BARS = 50`) are excluded from output (same as info_decomposition_export). Bars without valid forward returns (`isnan(fwd_return_1)`) are also excluded.

### bar_type and bar_param metadata columns

These reflect the CLI args, not hardcoded "time"/"5". Example: `--bar-type volume --bar-param 100` → CSV rows have `volume,100`.

## Pipeline (per day)

1. Open `.dbn.zst` file via `databento::DbnFileStore`
2. Feed MBO messages through `StreamingBookBuilder` (identical to info_decomposition_export)
3. Build bars via `BarFactory::create(bar_type, bar_param)` from snapshots
4. Assign MBO event indices to bars (same cursor logic as template)
5. Recount message types per bar from MBO events
6. Compute Track A features via `BarFeatureComputer(0.25f)`
7. Write CSV rows (skip warmup, skip NaN returns)

## Implementation Notes

- **Copy from template**: The entire `StreamingBookBuilder` class, `MES_CONTRACTS`, `SELECTED_DAYS`, `get_instrument_id()`, `date_to_string()`, `date_to_midnight_ns()`, `format_float()` — all identical.
- **No binary event export**: Unlike info_decomposition_export, this tool does NOT write binary `.bin` event files. CSV only.
- **Argument parsing**: Simple `argc`/`argv` loop. No external library needed. Check for `--bar-type`, `--bar-param`, `--output`.
- **CMakeLists.txt**: Add target like existing tools:
  ```cmake
  add_executable(bar_feature_export tools/bar_feature_export.cpp)
  target_include_directories(bar_feature_export PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(bar_feature_export PRIVATE databento::databento)
  ```
- **RTH boundaries**: Same as template — 09:30-16:00 ET.
- **StreamingBookBuilder snapshot interval**: 100ms (100,000,000 ns) — unchanged.

## Exit Criteria

- [ ] `tools/bar_feature_export.cpp` compiles without errors
- [ ] CLI args `--bar-type`, `--bar-param`, `--output` are parsed correctly
- [ ] Missing/invalid args produce non-zero exit code and usage message
- [ ] CSV header matches `info-decomposition/features.csv` exactly (same column names, count, order)
- [ ] `bar_type` and `bar_param` metadata columns reflect CLI args
- [ ] Warmup bars (first 50/day) excluded from output
- [ ] NaN forward-return bars excluded from output
- [ ] Tool registered in CMakeLists.txt and builds as part of `cmake --build build`
- [ ] `build/bar_feature_export` exists and is executable after build
