# bar_feature_export --target/--stop CLI Flags — TDD Spec

## Summary

Add `--target <ticks>` and `--stop <ticks>` CLI flags to `bar_feature_export` so that the triple barrier geometry can be varied per-export. Currently the barrier parameters (target=10, stop=5) are hardcoded. The `compute_bidirectional_tb_label()` function already accepts these as parameters — this spec only exposes them through the CLI.

## Motivation

The label-design-sensitivity experiment needs Parquet files exported at different triple barrier geometries (e.g., 15:3, 12:5, 20:4). Without CLI flags, every geometry requires a code change and rebuild.

## Scope

**Small change — expose existing params through CLI. No new computation logic.**

### New CLI Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--target` | int | 10 | Target (take-profit) barrier in ticks |
| `--stop` | int | 5 | Stop-loss barrier in ticks |

### Expected CLI After Change

```
Usage: bar_feature_export --bar-type <type> --bar-param <threshold> --output <path>

  --bar-type       Bar type: time, volume, dollar, tick
  --bar-param      Bar threshold (e.g., 5.0 for time_5s)
  --output         Output file path (.csv or .parquet)
  --target         Target barrier in ticks (default: 10)
  --stop           Stop barrier in ticks (default: 5)
  --legacy-labels  Use old-style unidirectional labels (149 columns)
```

### Behavior

1. Parse `--target` and `--stop` from command line (integer values)
2. Pass them through to `compute_bidirectional_tb_label()` (which already accepts `target_ticks` and `stop_ticks` parameters)
3. When `--legacy-labels` is used, pass the same target/stop to the legacy label computation
4. Default values (10, 5) produce **byte-identical output** to the current binary
5. Invalid values (target <= 0, stop <= 0, target <= stop) should produce a clear error message and exit with non-zero status

### Files to Modify

- `tools/bar_feature_export.cpp` — Add argument parsing for `--target` and `--stop`, pass to label computation call
- `tests/bar_feature_export_test.cpp` (or equivalent) — Add tests for new flags

### No Changes To

- `triple_barrier.hpp` / `triple_barrier.cpp` — Already parameterized
- `bidirectional_tb_label` — Already accepts target_ticks, stop_ticks
- Parquet schema — No new columns
- Any other source files

## Test Cases

### T1: Default geometry matches current output
Run `bar_feature_export` without `--target`/`--stop` flags. Output must be identical to the current binary (same row count, same label distribution). This confirms the default path is unchanged.

### T2: Non-default geometry produces different labels
Run `bar_feature_export --target 15 --stop 3` on the same input. The label distribution (count of -1, 0, +1) must differ from the default (10, 5) geometry. Specifically:
- Higher target → fewer directional labels (more HOLD)
- Narrower stop → more directional labels (fewer HOLD)
- The net effect of 15:3 vs 10:5 depends on the data, but the distributions MUST differ.

### T3: --help shows new flags
`bar_feature_export --help` (or missing required args) output must include `--target` and `--stop` with descriptions and defaults.

### T4: Invalid values rejected
- `--target 0` → error, non-zero exit
- `--stop -1` → error, non-zero exit
- `--target 3 --stop 5` (target <= stop) → error, non-zero exit

### T5: Flags work with --legacy-labels
`bar_feature_export --target 15 --stop 3 --legacy-labels` produces 149-column output (not 152). The target/stop values are respected in the legacy label computation path too.

## Exit Criteria

- [ ] `bar_feature_export --help` shows `--target` and `--stop` with defaults
- [ ] Default invocation (no --target/--stop) produces identical output to current binary
- [ ] `--target 15 --stop 3` produces Parquet with different label distribution than default
- [ ] Invalid params (target<=0, stop<=0, target<=stop) produce error and non-zero exit
- [ ] `--legacy-labels` works with custom target/stop
- [ ] All existing tests still pass
- [ ] New tests for T1-T5 pass

## Dependencies

- `compute_bidirectional_tb_label()` in `triple_barrier.hpp` (already parameterized)
- `bar_feature_export.cpp` CLI parsing infrastructure (already uses standard arg parsing)
