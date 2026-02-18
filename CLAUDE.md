# Project Instructions

WHEN WORKING FROM A PROVIDED SPEC ALWAYS REFER TO THE  ## Exit Criteria SECTION TO CHECK BOXES AND KEEP TRACK OF YOUR WORK


## Path Convention

Kit state files, working directories, and utility scripts live in `.kit/`. Project source code (`src/`, `tests/`, etc.) stays at the project root. The kit prompts reference bare filenames (e.g., `LAST_TOUCH.md`) — the `KIT_STATE_DIR` environment variable tells the scripts to resolve these inside `.kit/`.

Files at project root: `CLAUDE.md`, `.claude/`, `.orchestration-kit.env`, `.gitignore`
Everything else kit-related: `.kit/`

## Available Kits

| Kit | Script | Phases |
|-----|--------|--------|
| **TDD** | `.kit/tdd.sh` | red, green, refactor, ship, full, watch |
| **Research** | `.kit/experiment.sh` | survey, frame, run, read, log, cycle, full, program, status |
| **Math** | `.kit/math.sh` | survey, specify, construct, formalize, prove, audit, log, full, program, status |

## Orchestrator (Advanced)

For cross-kit runs and interop, use the orchestrator:

```bash
source .orchestration-kit.env
orchestration-kit/tools/kit --json <kit> <phase> [args...]
orchestration-kit/tools/kit --json research status
```

Run artifacts land in `orchestration-kit/runs/<run_id>/` — capsules, manifests, logs, events.

## Cross-Kit Interop (Advanced)

```bash
orchestration-kit/tools/kit request --from research --from-phase status --to math --action math.status \
  --run-id <parent_run_id> --json
orchestration-kit/tools/pump --once --request <request_id> --json
```

`--from-phase` is optional; if omitted, `orchestration-kit/tools/pump` infers it from the parent run metadata/events.

### Research → TDD Sub-Cycle

When a research phase needs **new C++ code** (tools, libraries, APIs), spawn a TDD sub-cycle rather than writing code inline. This ensures all new code is regression-tested.

**Pattern**:
1. Write a spec: `.kit/docs/<feature>.md`
2. Run TDD phases (all in background, check exit codes only):
   ```bash
   source .orchestration-kit.env
   .kit/tdd.sh red   .kit/docs/<feature>.md
   .kit/tdd.sh green
   .kit/tdd.sh refactor
   .kit/tdd.sh ship  .kit/docs/<feature>.md
   ```
3. Resume research with the new tested infrastructure available.

**Triggers**: Data extraction tools, new analysis APIs, infrastructure modifications.
**Non-triggers**: Disposable Python scripts, config files, parameter sweeps — these stay in Research kit.

## Cloud / Remote Compute (Optional)

For experiments whose compute profile exceeds local thresholds, use `tools/preflight` and `tools/cloud-run`.

### Pre-flight Check

```bash
orchestration-kit/tools/preflight .kit/experiments/exp-NNN-name.md --json
```

Parses the `Compute Profile` YAML block in the experiment spec and recommends local vs. cloud execution. If cloud is recommended, the RUN agent's prompt includes a **Compute Advisory**.

### Remote Execution (EC2 / RunPod)

```bash
# Launch experiment on cloud
orchestration-kit/tools/cloud-run run "python scripts/run_experiment.py --full" \
    --spec .kit/experiments/exp-NNN-name.md \
    --data-dirs DATA/ \
    --output-dir .kit/results/exp-NNN-name/ \
    --detach

# Check status
orchestration-kit/tools/cloud-run status <run-id>

# Pull results back
orchestration-kit/tools/cloud-run pull <run-id> --output-dir .kit/results/exp-NNN-name/

# List tracked runs
orchestration-kit/tools/cloud-run ls

# Cleanup orphaned resources
orchestration-kit/tools/cloud-run gc

# Force-terminate a run
orchestration-kit/tools/cloud-run terminate <run-id>

# Manage RunPod network volumes
orchestration-kit/tools/cloud-run volume {create,list,delete}
```

**Key flags:**
- `--detach`: Launch and return immediately (for long-running experiments)
- `--data-dirs`: Comma-separated local dirs to upload alongside code
- `--max-hours N`: Auto-terminate safety (default: 12h)
- `--output-dir`: Where to download results locally

**Backend selection:** Configured via environment variables in `.orchestration-kit.env`. Supports AWS EC2 and RunPod. EC2 runs in a `python:3.12-slim` Docker container; RunPod uses `pytorch:2.4.0-py3.11`. Dependencies install via `uv` from `requirements.txt`. Results sync back via S3. **Important:** `--data-dirs` preserves the directory name — if you upload `.../dollar_25k/`, the remote path is `data/dollar_25k/`, not `data/`. Use `--spot` for spot instances (default); `--force` overrides preflight "local" recommendation.

## Global Dashboard (Optional)

```bash
orchestration-kit/tools/dashboard register --orchestration-kit-root ./orchestration-kit --project-root "$(pwd)"
orchestration-kit/tools/dashboard index
orchestration-kit/tools/dashboard serve --host 127.0.0.1 --port 7340
```

Open `http://127.0.0.1:7340` to explore runs across projects and filter by project.

## State Files (in `.kit/`)

| Kit | Read first |
|-----|-----------|
| TDD | `CLAUDE.md` → `.kit/LAST_TOUCH.md` → `.kit/PRD.md` |
| Research | `CLAUDE.md` → `.kit/RESEARCH_LOG.md` → `.kit/QUESTIONS.md` |
| Math | `CLAUDE.md` → `.kit/CONSTRUCTION_LOG.md` → `.kit/CONSTRUCTIONS.md` |

## Working Directories

- `.kit/docs/` — TDD specs
- `.kit/experiments/` — Research experiment specs
- `.kit/results/` — Research + Math results
- `.kit/specs/` — Math specification documents
- `.kit/handoffs/completed/` — Resolved research handoffs
- `.kit/scripts/` — Utility scripts (symlinked from orchestration-kit)

## Don't

- Don't `cd` into `orchestration-kit/` and run kit scripts from there — run from project root.
- Don't `cat` full log files — use `orchestration-kit/tools/query-log`.
- Don't explore the codebase to "understand" it — read state files first.
- **Don't independently verify kit sub-agent work.** Each phase spawns a dedicated sub-agent that does its own verification. Trust the exit code and capsule. Do NOT re-run tests, re-read logs, re-check build output, or otherwise duplicate work the sub-agent already did. Exit 0 + capsule = done. Exit 1 = read the capsule for the failure, don't grep the log.
- Don't read phase log files after a successful phase. Logs are for debugging failures only.

## Orchestrator Discipline (MANDATORY)

You are the orchestrator. Sub-agents do the work. Your job is to sequence phases and react to exit codes. Protect your context window.

1. **Run phases in background, check only the exit code.** Do not read the TaskOutput content — the JSON blob wastes context. Check `status: completed/failed` and `exit_code` only.
2. **Never run Bash for verification.** No `pytest`, `lake build`, `ls`, `cat`, `grep` to check what a sub-agent produced. If the phase exited 0, it worked.
3. **Never read implementation files** the sub-agents wrote (source code, test files, .lean files, experiment scripts). That is their domain. You read only state files (CLAUDE.md, `.kit/LAST_TOUCH.md`, `.kit/RESEARCH_LOG.md`, etc.).
4. **Chain phases by exit code only.** Exit 0 → next phase. Exit 1 → read the capsule (not the log), decide whether to retry or stop.
5. **Never read capsules after success.** Capsules exist for failure diagnosis and interop handoffs. A successful phase needs no capsule read.
6. **Minimize tool calls.** Each Bash call, Read, or Glob adds to your context. If the information isn't needed to decide the next action, don't fetch it.

## Breadcrumb Maintenance (MANDATORY)

After every session that changes the codebase, update:

1. **`.kit/LAST_TOUCH.md`** — Current state and what to do next (TDD).
2. **`.kit/RESEARCH_LOG.md`** — Append experiment results (Research).
3. **`.kit/CONSTRUCTION_LOG.md`** — Progress notes (Math).
4. **This file's "Current State" section** — Keep it current.

## Project: MBO-DL (MES Microstructure Model Suite)

**Master spec**: `completed_specs/ORCHESTRATOR_SPEC.md` (archived) — the single source of truth for all data contracts, model architectures, and build phases.

**Language**: C++20 (data pipeline + 3 models), Python (SSM only — mamba-ssm requires CUDA).
**Build**: CMake. Dependencies via FetchContent (libtorch, databento-cpp, xgboost, GTest).
**Data**: `DATA/GLBX-20260207-L953CAPU5B/` — 312 daily `.dbn.zst` files, MES MBO 2022 (~49 GB). Do NOT read these directly — use `databento::DbnFileStore` C++ API.

### R | API+ (Rithmic) — Available, NOT integrated

R | API+ v13.6.0.0 is installed but **not yet integrated into any source code**. It exists as a future option for live/paper trading connectivity. No targets in CMakeLists.txt currently link against it.

- **Install**: `~/.local/rapi/13.6.0.0/` (include, libs, SSL cert, samples)
- **CMake target**: `RApiPlus::RApiPlus` (found via `cmake/FindRApiPlus.cmake`)
- **SDK docs & samples**: `/Users/brandonbell/Downloads/13.6.0.0/` — contains full Doxygen HTML docs (`doc/html/index.html`), programmer's guide, FAQ, and sample apps (SampleMD, SampleOrder, SampleBar)
- **SSL cert**: `~/.local/rapi/13.6.0.0/etc/rithmic_ssl_cert_auth_params` — required at runtime, path set via `MML_SSL_CLNT_AUTH_FILE` env var
- **Platform**: darwin-20.6-arm64 static libraries (Apple Silicon native)

**Kit state convention**: All kit state files live in `.kit/` (not project root). `KIT_STATE_DIR=".kit"` is set in `.orchestration-kit.env`.

## Current State (updated 2026-02-18, R4c complete)

**VERDICT: GO.** Oracle expectancy extracted on 19 real MES days. Triple barrier passes all 6 success criteria: $4.00/trade expectancy, PF=3.30, WR=64.3%, Sharpe=0.362, net PnL=$19,479. CONDITIONAL GO upgraded to full GO. Triple barrier preferred over first-to-hit.

**R4c (temporal-predictability-completion) — COMPLETE. CONFIRMED (all nulls).** All three gaps from R4/R4b closed. 0/54+ dual threshold passes across tick_50, tick_100 (~10s), tick_250 (~25s), and time_5s extended horizons (h=200/500/1000, ~17-83min). Dollar bars entirely sub-actionable (max ~0.9s/bar at $1M). All Tier 1 AR R² negative. MES is martingale across all bar types and timescales 5s-83min. Temporal encoder dropped permanently with highest confidence. R4 line closed. See `.kit/results/temporal-predictability-completion/analysis.md`.

**R4b (temporal-predictability-event-bars) — COMPLETE. MARGINAL SIGNAL (redundant).** Volume_100: NO SIGNAL (all 36 AR configs negative R², matches time_5s). Dollar_25k: positive AR R² at sub-second horizons (h=1: +0.000633, h=5: +0.000364) but temporal augmentation fails dual threshold (0/48 gaps pass across all bar types). Signal is linear, redundant with static features, at ~140ms HFT timescale. Temporal-Only R²=0.012 (p=0.0005) but adds nothing over static. R4 "no temporal encoder" conclusion is robust across all bar types. See `.kit/results/temporal-predictability-event-bars/analysis.md`.

**R6 (synthesis) complete — CONDITIONAL GO.** CNN + GBT Hybrid architecture recommended. R3 CNN R²=0.132 on structured (20,2) book resolves R2-R3 tension in favor of spatial encoder. Message + temporal encoders dropped. Bar type: time_5s. Horizons: h=1 and h=5. See `.kit/results/synthesis/metrics.json`.

**R4 (temporal-predictability) complete — NO TEMPORAL SIGNAL.** All 36 Tier 1 AR configs produce negative R². MES 5s returns are martingale. Drop SSM/temporal encoder.

**R2 (info-decomposition) complete — FEATURES SUFFICIENT.** No encoder stage passes dual threshold. Best R²=0.0067 on 1-bar horizon only. Book snapshot is sufficient statistic.

**R1 (subordination-test) complete — REFUTED.** Time bars are the baseline; no justification for event-driven bars.

**Spec: `TRAJECTORY.md`** — Kenoma Labs MES Backtest & Feature Discovery. 10 sequential phases (5 engineering, 5 research).

### Phase Sequence

| # | Spec | Kit | Status |
|---|------|-----|--------|
| 1 | `.kit/docs/bar-construction.md` | TDD | **Done** |
| 2 | `.kit/docs/oracle-replay.md` | TDD | **Done** |
| 3 | `.kit/docs/multi-day-backtest.md` | TDD | **Done** |
| R1 | `.kit/experiments/subordination-test.md` | Research | **Done (REFUTED)** |
| 4 | `.kit/docs/feature-computation.md` | TDD | **Done** |
| 5 | `.kit/docs/feature-analysis.md` | TDD | **Done** |
| R2 | `.kit/experiments/info-decomposition.md` | Research | **Done (FEATURES SUFFICIENT)** |
| R3 | `.kit/experiments/book-encoder-bias.md` | Research | **Done (CNN BEST)** |
| R4 | `.kit/experiments/temporal-predictability.md` | Research | **Done (NO TEMPORAL SIGNAL)** |
| 6 | `.kit/experiments/synthesis.md` | Research | **Done (CONDITIONAL GO)** |
| 7 | `.kit/docs/oracle-expectancy.md` | TDD | **Done** |
| 7b | `tools/oracle_expectancy.cpp` | Research | **Done (GO)** |
| 8 | `.kit/docs/bar-feature-export.md` | TDD | **Done** |
| R4b | `.kit/experiments/temporal-predictability-event-bars.md` | Research | **Done (NO SIGNAL — robust)** |
| R4c | `.kit/experiments/temporal-predictability-completion.md` | Research | **Done (CONFIRMED — all nulls)** |

- **Build:** Green.
- **Tests:** 1003/1004 unit tests pass (1 disabled, 1 skipped), 22 integration tests (labeled, excluded from default ctest).
- **Exit criteria audit:** TRAJECTORY.md §13 audited — 21/21 engineering PASS, 13/13 research PASS (R4c closes MI/decay gap).
- **Next task:** Model architecture build spec — CNN+GBT Hybrid, static features, time_5s bars, triple barrier labels. All research phases complete.
