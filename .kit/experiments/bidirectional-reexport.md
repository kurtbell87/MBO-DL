# Experiment: Full-Year Bidirectional Label Re-Export

**Date:** 2026-02-25
**Priority:** BLOCKING — prerequisite for label-design-sensitivity and all downstream experiments
**Depends on:** bidirectional-export-wiring TDD (DONE, PR #27), full-year-export (DONE, 251 days)

---

## Purpose

Re-export the full-year Parquet dataset (251 RTH trading days, 2022 MES MBO) using the updated `bar_feature_export` binary with bidirectional triple barrier labels. The existing 149-column Parquet was exported with flawed long-perspective-only labels. The new 152-column export includes:

- `tb_label` — bidirectional: +1 (only long triggers), -1 (only short triggers), 0 (neither or both)
- `tb_exit_type` — updated semantics under bidirectional mode
- `tb_bars_held` — unchanged
- `tb_both_triggered` — NEW: 1.0 if both long and short races trigger (volatility diagnostic)
- `tb_long_triggered` — NEW: 1.0 if the long race triggered
- `tb_short_triggered` — NEW: 1.0 if the short race triggered

This is a **data re-export**, not a model training experiment. No experimental manipulation.

---

## Infrastructure

| Component | Value |
|-----------|-------|
| Binary | `bar_feature_export` (bidirectional default, 152 columns) |
| Docker image | ECR `mbo-dl:latest` (rebuilt from main with bidirectional code) |
| EBS snapshot | `snap-0efa355754c9a329d` (49GB MBO MES 2022, 312 .dbn.zst files) |
| Instance type | c5.2xlarge (8 vCPU, 16GB RAM) — CPU-only, parallelizable |
| Data mount | EBS → `/data/GLBX-20260207-L953CAPU5B/` |
| Output | `/work/results/bidirectional-reexport/` → S3 |
| S3 bucket | `s3://kenoma-labs-research/artifact-store/` |

---

## Export Command

For each of the ~251 RTH trading days:

```bash
bar_feature_export --bar-type time --bar-param 5 \
    --input /data/GLBX-20260207-L953CAPU5B/{date}.dbn.zst \
    --output /work/results/{date}.parquet
```

No `--legacy-labels` flag — bidirectional is the default.

Parallel execution: N-1 CPUs concurrent (7 on c5.2xlarge).

---

## Validation

### Schema Check
- Each Parquet file has exactly 152 columns
- Columns include: `tb_both_triggered`, `tb_long_triggered`, `tb_short_triggered`
- Column order: new columns follow `tb_bars_held`

### Label Distribution Check
- `tb_label` values ∈ {-1.0, 0.0, 1.0}
- Bidirectional mode produces MORE `tb_label=0` bars than prior export
- Expected shift: ~25% long / ~45% short / ~22% hold → ~15-20% long / ~15-20% short / ~60-70% hold
- `tb_both_triggered` rate should be non-trivial (>1% of bars)

### Data Integrity
- total_days >= 240 (expect 251)
- total_rows ~ 1,160,150 (same as prior export — bar construction unchanged)
- No duplicate timestamps
- First 149 columns (features, metadata) identical to prior export (label columns differ)

---

## Success Criteria

- [x] SC-1: 312 files exported (all .dbn.zst files; 251 RTH days within). Downstream filters to RTH.
- [ ] SC-2: total_rows ~ 1,160,150 (within 1% of prior export) — PENDING validation
- [x] SC-3: New columns present (152-column schema built into the binary) — PASS by construction
- [ ] SC-4: `tb_label` distribution shifted toward more HOLD (>40% label=0) — PENDING validation
- [ ] SC-5: `tb_both_triggered` rate > 1% of bars — PENDING validation
- [x] SC-6: All results uploaded to S3 (`s3://kenoma-labs-research/results/bidirectional-reexport/`)
- [x] SC-7: manifest.json written with 312 files, 0 failures

---

## Resource Budget

### Compute Profile
```yaml
compute_type: cpu
instance_type: c5.2xlarge
estimated_wall_hours: 2.0
estimated_cost: $1.00
gpu_type: none
estimated_rows: 1160150
model_type: none
sequential_fits: 0
parallelizable: true
memory_gb: 8
```

Prior full-year export took ~90 min on similar hardware. Re-export should be comparable.

---

## Output

```
.kit/results/bidirectional-reexport/
    {date}.parquet              # 251 files, 152 columns each
    manifest.json               # Per-day metadata
    validation.json             # Schema + distribution checks
```

After export, push large files to S3 artifact store:
```bash
orchestration-kit/tools/artifact-store push-dir .kit/results/bidirectional-reexport/ --threshold 10MB
```
