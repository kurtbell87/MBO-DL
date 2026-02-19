# Project Instructions

WHEN WORKING FROM A PROVIDED SPEC ALWAYS REFER TO THE  ## Exit Criteria SECTION TO CHECK BOXES AND KEEP TRACK OF YOUR WORK

## ABSOLUTE RULES — Read This First

**You are the ORCHESTRATOR. You are NOT a developer. You do NOT write code. Ever.**

1. **NEVER write code.** Not C++, not Python, not scripts, not tests, not config files. ALL code is written by kit sub-agents (TDD phases, Research phases). If code needs to exist, delegate to a kit phase that will create it.
2. **NEVER grep, search, or read implementation files.** No `Grep`, no `Read` on `.cpp`, `.py`, `.hpp`, or test files. You read ONLY state files: `CLAUDE.md`, `.kit/LAST_TOUCH.md`, `.kit/RESEARCH_LOG.md`, `.kit/QUESTIONS.md`, `.kit/CONSTRUCTION_LOG.md`, spec files in `.kit/docs/` and `.kit/experiments/`.
3. **NEVER run verification commands.** No `python3 -c`, no `pytest`, no `cmake --build`, no `ctest`, no import checks. Sub-agents verify their own work. Trust exit codes.
4. **NEVER use Write, Edit, or Bash to create or modify source code.** The ONLY files you may write/edit are state files (`.md` files in `.kit/` and `CLAUDE.md`).
5. **ALWAYS delegate through kit phases.** C++ work → `.kit/tdd.sh` (red/green/refactor/ship). Python experiments → `.kit/experiment.sh` (survey/frame/run/read). Math → `.kit/math.sh`.
6. **Your only tools are:** `source .orchestration-kit.env`, `.kit/tdd.sh`, `.kit/experiment.sh`, `.kit/math.sh`, `orchestration-kit/tools/*`, reading/writing state `.md` files, and checking exit codes.
7. **Exit-Criteria in Specs are 1st Class Citizens**, you must always cross off ALL exit-criteria as they are completed and include them in your final report 
**If you catch yourself about to grep a source file, write a line of code, or run a test — STOP. That is a protocol violation. Delegate to a kit phase instead.**


## Git Workflow — Worktrees & Branches (MANDATORY)

**Every session starts on a new branch. No exceptions.**

Before doing any work, create a feature branch. Use **git worktrees** so parallel agents can operate simultaneously without conflicts.

### Session Start Protocol

```bash
# 1. Create a worktree for your task (from project root)
git worktree add ../<project>-<slug> -b <branch-name> main

# 2. cd into the worktree and source the env
cd ../<project>-<slug>
source .orchestration-kit.env

# 3. Do your work on this branch
```

### Branch Naming

| Work type | Pattern | Example |
|-----------|---------|---------|
| Experiment | `experiment/<exp-name>` | `experiment/r3b-event-bar-cnn` |
| TDD feature | `feat/<feature-name>` | `feat/tick-bar-fix` |
| Bug fix | `fix/<description>` | `fix/research-kit-tracking` |
| Infra/chore | `chore/<description>` | `chore/dashboard-scoping` |

### Parallel Agents with Worktrees

Git worktrees let multiple agents work in isolated directories on different branches, all backed by the same repo:

```bash
# Agent A: working on CNN pipeline fix
git worktree add ../MBO-DL-cnn-fix -b fix/cnn-normalization main

# Agent B: working on tick bar construction (parallel, no conflicts)
git worktree add ../MBO-DL-tick-bars -b feat/tick-bar-fix main

# List active worktrees
git worktree list

# After merging, clean up
git worktree remove ../MBO-DL-cnn-fix
```

**Rules:**
- **Never work directly on `main`.** Always branch first.
- **One branch per task.** Don't mix unrelated changes.
- **Merge to main when done** — fast-forward or cherry-pick, then push.
- **Clean up worktrees** after merging: `git worktree remove <path>`.
- **Worktree naming**: `../<project>-<slug>` keeps worktrees adjacent to the main checkout.
- **Each worktree gets its own `.kit/` state** via symlinks — kit scripts resolve relative to `PROJECT_ROOT`.

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

## Phase Orchestration (MANDATORY)

**Never run individual phases manually.** Use block commands that run the full cycle in one shot. This ensures automatic tracking in `program_state.json` and reduces orchestrator tool calls.

### Research Experiments

```bash
# PREFERRED: Full cycle from a pre-written spec (frame→run→read→log)
source .orchestration-kit.env
.kit/experiment.sh cycle .kit/experiments/<spec>.md

# Full cycle including survey (survey→frame→run→read→log)
.kit/experiment.sh full "research question" .kit/experiments/<spec>.md

# Auto-advancing program mode (picks from QUESTIONS.md)
.kit/experiment.sh program

# Check status (reads program_state.json + results dirs)
.kit/experiment.sh status
```

**Do NOT** call `survey`, `frame`, `run`, `read`, `log` individually. Block commands handle sequencing, error recovery, and automatic registration.

### TDD Features

```bash
# PREFERRED: Full cycle in one shot (red→green→refactor→ship)
source .orchestration-kit.env
.kit/tdd.sh full .kit/docs/<feature>.md
```

**Do NOT** call `red`, `green`, `refactor`, `ship` individually unless you need to retry a single failed phase.

### Research → TDD Sub-Cycle

When a research phase needs **new C++ code**, spawn a TDD sub-cycle:

1. Write a spec: `.kit/docs/<feature>.md`
2. Run the full TDD cycle as a single block command:
   ```bash
   source .orchestration-kit.env
   .kit/tdd.sh full .kit/docs/<feature>.md
   ```
3. Resume research with the new tested infrastructure available.

**Triggers**: Data extraction tools, new analysis APIs, infrastructure modifications.
**Non-triggers**: Disposable Python scripts, config files, parameter sweeps — these stay in Research kit.

## Dashboard as Status Interface (MANDATORY)

The project-local dashboard (scoped to this repo only) is the single source of truth for run status. **Do not read capsule files, log files, or events.jsonl directly.** Query the dashboard API instead.

```bash
# Ensure the dashboard is running
source .orchestration-kit.env
orchestration-kit/tools/dashboard ensure-service --wait-seconds 3

# Quick summary — total/running/ok/failed counts
curl -s http://127.0.0.1:7340/api/summary | python3 -m json.tool

# List all runs (filterable)
curl -s 'http://127.0.0.1:7340/api/runs'                    # all runs
curl -s 'http://127.0.0.1:7340/api/runs?status=failed'      # failures only
curl -s 'http://127.0.0.1:7340/api/runs?kit=research'       # research runs only

# Re-index after a phase completes
curl -s -X POST http://127.0.0.1:7340/api/refresh

# Research program status (experiments + questions)
.kit/experiment.sh status
```

### On Failure — Capsule Drill-Down

When a phase fails (exit code != 0):
1. Query the dashboard API for the failed run's `capsule_path`
2. Use `orchestration-kit/tools/query-log` to inspect the capsule with bounded output
3. Do NOT `cat` or `Read` the full log — capsules are the 30-line summary

```bash
# Find the failed run's capsule
curl -s 'http://127.0.0.1:7340/api/runs?status=failed' | python3 -c "
import json,sys
for r in json.load(sys.stdin).get('runs',[]):
    print(f\"{r['run_id']} {r['kit']}/{r['phase']} → {r.get('capsule_path','N/A')}\")
"

# Read the capsule (bounded)
orchestration-kit/tools/query-log tail <capsule_path> 30
```

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

## Research Protocol Lessons (Institutional Memory)

- **Early stopping must use a held-out validation split from training data, never test data.** Using test data for model selection (checkpoint selection via early stopping) is validation leakage that inflates reported R². R3's R²=0.132 was inflated ~36% by this bug (proper-validation R²≈0.084). All experiment specs must specify an 80/20 train/val split for early stopping.
- **Always specify normalization in the experiment spec as concrete operations** (e.g., "divide by TICK_SIZE=0.25" not "normalize to ticks"). Ambiguous normalization language caused three failed reproduction attempts (9B, 9C, 9C diagnostic).
- **Verify bar construction semantics before running experiments.** The C++ `bar_feature_export` tick bar mode counts fixed-rate book snapshots (10/s), not trade events. Diagnostic: (1) check bars_per_day_std — genuine event bars must have non-zero daily variance across trading days; (2) check within-day p10 vs p90 — time bars have p10=p90 (zero variance), genuine event bars don't. R3b wasted a cycle on this. **Blast radius**: every "tick bar" in the research program (R1, R4c, R4d, R3b) was actually time bars. Dollar bars are confirmed genuine (daily counts vary, p10≠p90). Volume bar status is unverified — needs the same diagnostic.
- **Dollar and volume bars are genuine event bars; only tick bars are broken.** R1 metrics contain `bar_count_cv` for all 12 configs — the diagnostic was available from day 1 but nobody checked. Tick bars: CV=0 (broken). Dollar bars: CV=2-6% (genuine). Volume bars: CV=9-10% (genuine). Time bars: CV=0 (correct by design). Additional dollar bar evidence: within-day p10≠p90, daily counts differ from day1 count. Additional volume bar evidence: vol_100 produces 6,087 bars/day vs 2,315 predicted by snapshot-counting model.

## Don't

- **Don't write code.** Not a single line. Not "just a quick check". Not "let me verify the import". ALL code comes from kit sub-agents. Period.
- **Don't run individual phases.** No `experiment.sh survey`, `experiment.sh frame`, `tdd.sh red`, `tdd.sh green` as separate calls. Use block commands: `cycle`, `full`, `program`, `tdd.sh full`. The only exception is retrying a single failed phase.
- **Don't read source files.** No `.cpp`, `.py`, `.hpp`, `.h`, test files. You are the orchestrator, not a code reviewer. Sub-agents handle implementation.
- **Don't grep or search source files.** No `Grep` on implementation code. No `Glob` to find source files. If you need to know what exists, read state files.
- **Don't run verification commands.** No `python3 -c`, `pytest`, `cmake --build`, `ctest`, `import` checks, or any other verification. Sub-agents verify themselves.
- **Don't use Write/Edit/Bash on code files.** The only files you create or edit are `.md` state files in `.kit/` and `CLAUDE.md`.
- **Don't read capsules, logs, or events.jsonl directly.** Query the dashboard API (`curl http://127.0.0.1:7340/api/runs`) for run status. On failure, get the capsule path from the API and use `query-log` to read it bounded. Never `cat` or `Read` raw log/capsule files.
- Don't `cd` into `orchestration-kit/` and run kit scripts from there — run from project root.
- Don't explore the codebase to "understand" it — read state files first.
- **Don't independently verify kit sub-agent work.** Each phase spawns a dedicated sub-agent that does its own verification. Trust the exit code. Do NOT re-run tests, re-read logs, re-check build output, or otherwise duplicate work the sub-agent already did. Exit 0 = done. Exit 1 = query dashboard for capsule, diagnose, retry or stop.

## Orchestrator Discipline (MANDATORY)

You are the orchestrator. Sub-agents do the work. **You write ZERO code. You read ZERO source files. You run ZERO verification commands.**

Your ONLY job: launch block commands, check exit codes via the dashboard, update state files.

1. **Use block commands only.** `experiment.sh cycle`, `experiment.sh full`, `experiment.sh program`, `tdd.sh full`. Never call individual phases (`survey`, `frame`, `run`, `read`, `red`, `green`) unless retrying a single failed phase.
2. **Run phases in background, check only the exit code.** Do not read the TaskOutput content — the JSON blob wastes context. Check `status: completed/failed` and `exit_code` only.
3. **Check status via the dashboard API, not by reading files.** Use `curl http://127.0.0.1:7340/api/summary` for quick counts. Use `curl .../api/runs?status=failed` to find failures. Use `experiment.sh status` for research program overview.
4. **Never run Bash for verification.** No `pytest`, `lake build`, `ls`, `cat`, `grep`, `python3 -c` to check what a sub-agent produced. If the phase exited 0, it worked.
5. **Never read implementation files** the sub-agents wrote (source code, test files, .lean files, experiment scripts, Python modules). That is their domain. You read only state files (CLAUDE.md, `.kit/LAST_TOUCH.md`, `.kit/RESEARCH_LOG.md`, etc.) and spec files (`.kit/docs/*.md`, `.kit/experiments/*.md`).
6. **Never write code.** Not C++, Python, shell scripts, config files, or anything else. If code must be created, it happens inside a kit phase (TDD green creates code, Research run creates scripts). You delegate, you do not implement.
7. **Chain phases by exit code only.** Exit 0 → next phase. Exit 1 → query the dashboard for the failed run's capsule path, read the capsule via `query-log`, decide whether to retry or stop.
8. **Never read capsules or logs directly after success.** On failure, find the capsule via the dashboard API, then read it with `query-log`. Never `cat` or `Read` raw log files.
9. **Minimize tool calls.** Each Bash call, Read, or Glob adds to your context. If the information isn't needed to decide the next action, don't fetch it.
10. **Allowed tool usage summary:**
    - `Bash`: ONLY for `source .orchestration-kit.env && .kit/tdd.sh full ...`, `.kit/experiment.sh cycle|full|program ...`, `orchestration-kit/tools/*`, `curl` to dashboard API, `mkdir -p` for results dirs, `./build/<tool>` for data export.
    - `Read/Edit/Write`: ONLY for `.md` state files in `.kit/` and `CLAUDE.md`.
    - `Grep/Glob`: ONLY for finding state files and spec files. NEVER for source code.

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

## Current State (updated 2026-02-19, R3b Event-Bar CNN — INCONCLUSIVE)

**VERDICT: GO.** Oracle expectancy validated ($4.00/trade). CNN+GBT on time_5s remains the pipeline direction. R3b found that `bar_feature_export` tick bar construction is broken (counts snapshots, not trades), making the event-bar hypothesis untested. CNN spatial R² degrades at slower time frequencies — time_5s is the fastest available and performs best.

**R3b (CNN on Event Bars) — INCONCLUSIVE + SYSTEMIC TICK BAR DEFECT.** Tick bars from bar_feature_export are actually time bars at different frequencies (bars_per_day std=0.0 at all thresholds). Peak R²=0.057 (tick_100 ≈ time_10s), all WORSE than time_5s baseline (0.084). **Blast radius:** Every experiment that used "tick bars" (R1 tick configs, R4c tick_50/100/250, R4d tick_500/3000, R3b) was actually testing time bars at different frequencies. **Dollar and volume bars are genuine** (daily counts vary: dollar CV=2-6%, volume CV=9-10%). R4b dollar_25k and volume_100 results stand. **Only tick bars are void.** RESEARCH_AUDIT "Closed" on bar type should be "Closed for dollar/volume, open for tick only." See `.kit/results/R3b-event-bar-cnn/analysis.md` and RESEARCH_LOG correction notice.

**R3 Reproduction Pipeline Comparison — COMPLETE (REFUTED Step 2 / CONFIRMED Step 1). OUTCOME C.**
- **Step 1 CONFIRMED:** R3's CNN R²=0.132 reproduces exactly. Per-fold correlation with R3 = 0.9997. All 5 folds: train R² in [0.157, 0.196]. CNN spatial signal IS real.
- **Step 2 REFUTED:** features.csv (R3) and time_5s.csv (9C) are **byte-identical** (identity rate=1.0, max diff=0.0). There was never a "Python vs C++ pipeline" — R3 loaded from the same C++ export as 9B/9C.
- **Root cause of 0.132→0.002 gap:** (1) Missing TICK_SIZE division on prices (÷0.25 for tick offsets), (2) Per-fold z-scoring instead of per-day z-scoring on sizes, (3) R3's test-as-validation leakage, (4) Per-fold seed variation.
- **Proper validation R²=0.084** — still 12× higher than R2's flattened MLP R²=0.007, but 36% lower than R3's leaked 0.132.
- **No handoff needed.** C++ export is correct. Fix is in Python training normalization (within research scope).

**Phase 9B (hybrid-model-training) — REFUTED.** CNN R²=-0.002, XGBoost acc=0.41, expectancy -$0.44/trade. GBT-only outperforms hybrid.
- Phase A: TDD cycle all exit 0. C++ TB label export working. 87,970 bars exported.

**R4d (temporal-predictability-dollar-tick-actionable) — COMPLETE. CONFIRMED (5/5 operating points).** 0/38 dual threshold passes across all 5 operating points: dollar $5M/7s, $10M/14s, $50M/69s; tick 500/50s, 3000/300s. Full 7s–300s timescale range now covered (prior run only had 2 points). Dollar $5M AR R²=−0.00035; dollar_50M h=1 marginally positive (+0.0025) but noise (std=6×mean, p=1.0). Calibration table for 10 thresholds produced. R4b's sub-second temporal signal decays to noise by $5M/7s. Cumulative R4 chain: 0/168+ dual threshold passes across 7 bar types, 0.14s–300s. R4 line permanently closed. See `.kit/results/temporal-predictability-dollar-tick-actionable/analysis.md`.

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
| R4d | `.kit/experiments/temporal-predictability-dollar-tick-actionable.md` | Research | **Done (CONFIRMED)** |
| 9A | `.kit/docs/hybrid-model.md` | TDD | **Done** (C++ TB label export) |
| 9B | `.kit/experiments/hybrid-model-training.md` | Research | **Done (REFUTED)** — CNN pipeline broken |
| **9C** | **`.kit/experiments/cnn-reproduction-diagnostic.md`** | **Research** | **Done (REFUTED)** — deviations not root cause; data pipeline suspected |
| **9D** | **`.kit/experiments/r3-reproduction-pipeline-comparison.md`** | **Research** | **Done (Step 1 CONFIRMED, Step 2 REFUTED)** — root cause found: normalization + leakage |
| **R3b** | **`.kit/experiments/r3b-event-bar-cnn.md`** | **Research** | **Done (INCONCLUSIVE)** — bar construction defect; tick bars are time bars |

- **Build:** Green.
- **Tests:** 1003/1004 unit tests pass (1 disabled, 1 skipped), 22 integration tests (labeled, excluded from default ctest).
- **Exit criteria audit:** TRAJECTORY.md §13 audited — 21/21 engineering PASS, 13/13 research PASS (R4c closes MI/decay gap).
- **Next task:** CNN Pipeline Fix — apply TICK_SIZE normalization (÷0.25 on book prices) + per-day z-scoring on sizes in the production training pipeline. Re-attempt CNN+GBT integration with corrected normalization and proper validation. Expected CNN R²≈0.084.
- **Tick bar fix (blocking for event-bar research, non-blocking for model build):** Fix bar_feature_export tick bar construction to count action='T' trade events from MBO stream, not fixed-rate snapshots. Add unit test: "tick bar count per day has non-zero variance across trading days." Validate against T11/T12 export spec tests. Until fixed, RESEARCH_AUDIT bar type status is "Closed for dollar/volume bars, open for tick bars only."
- **Volume bars confirmed genuine** (2026-02-19): R1 metrics show bar_count_cv=9-10% for vol_50/100/200. Cross-check: vol_100 = 6,087 bars/day vs 2,315 predicted by snapshot-counting model (2.6× discrepancy). R4b volume_100 null result is valid.
