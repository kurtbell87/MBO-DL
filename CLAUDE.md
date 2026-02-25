# Project Instructions

WHEN WORKING FROM A PROVIDED SPEC ALWAYS REFER TO THE  ## Exit Criteria SECTION TO CHECK BOXES AND KEEP TRACK OF YOUR WORK

## ABSOLUTE RULES — Read This First

**You are the ORCHESTRATOR. You are NOT a developer. You do NOT write code. Ever.**

1. **NEVER write code.** Not C++, not Python, not scripts, not tests, not config files. ALL code is written by kit sub-agents (TDD phases, Research phases). If code needs to exist, delegate to a kit phase that will create it.
2. **NEVER grep, search, or read implementation files.** No `Grep`, no `Read` on `.cpp`, `.py`, `.hpp`, or test files. You read ONLY state files: `CLAUDE.md`, `.kit/LAST_TOUCH.md`, `.kit/RESEARCH_LOG.md`, `.kit/QUESTIONS.md`, `.kit/CONSTRUCTION_LOG.md`, spec files in `.kit/docs/` and `.kit/experiments/`.
3. **NEVER run verification commands.** No `python3 -c`, no `pytest`, no `cmake --build`, no `ctest`, no import checks. Sub-agents verify their own work. Trust exit codes.
4. **NEVER use Write, Edit, or Bash to create or modify source code.** The ONLY files you may write/edit are state files (`.md` files in `.kit/` and `CLAUDE.md`).
5. **ALWAYS delegate through kit phases.** C++ work → `.kit/tdd.sh` (red/green/refactor/ship). Python experiments → `.kit/experiment.sh` (survey/frame/run/read). Math → `.kit/math.sh`.
6. **Your only tools are:** MCP tools (`kit.tdd`, `kit.research_cycle`, `kit.research_full`, `kit.research_program`, `kit.math`, `kit.status`, `kit.runs`, `kit.capsule`, `kit.research_status`), bash fallbacks (`source .orchestration-kit.env`, `.kit/tdd.sh`, `.kit/experiment.sh`, `.kit/math.sh`, `orchestration-kit/tools/*`), reading/writing state `.md` files, and checking exit codes.
7. **Exit-Criteria in Specs are 1st Class Citizens**, you must always cross off ALL exit-criteria as they are completed and include them in your final report 
**If you catch yourself about to grep a source file, write a line of code, or run a test — STOP. That is a protocol violation. Delegate to a kit phase instead.**

## Compute Execution — Cost-Aware Tiered Strategy

**Choose the cheapest backend that fits the workload. Do NOT default to EC2 on-demand.**

| Workload | Data Size | Backend | Rationale |
|----------|-----------|---------|-----------|
| XGBoost / sklearn / CPU-only | < 1 GB | **Local** | Apple Silicon handles it; free; no cloud overhead |
| PyTorch CNN / GPU training | < 1 GB | **RunPod** (GPU) | Cheaper than EC2 on-demand; no EBS needed for small data |
| Large data processing | > 10 GB | **EC2** (spot preferred) | EBS snapshot pre-loading; spot saves ~70% vs on-demand |
| Full pipeline (build + train) | > 10 GB | **EC2** (Docker/ECR) | Full pipeline needs EBS + ECR image |

- **Claude sub-agent phases** (survey, frame, read, log): always local (LLM conversations, no heavy compute)
- **TDD phases**: local unless build/test is heavy
- **Preflight check**: use `orchestration-kit/tools/preflight` if unsure
- **NEVER use EC2 on-demand for CPU-only experiments on small datasets.** This wastes money.
- **RunPod** requires `RUNPOD_API_KEY` in `.orchestration-kit.env`. Use `--backend runpod` flag with `cloud-run`.
- **EC2 spot** is acceptable for long-running GPU jobs (>4h). Use `--spot` flag.

### Research Hybrid Workflow (Local + Cloud)

`experiment.sh full` and `experiment.sh cycle` handle cloud execution automatically. When `COMPUTE_TARGET=ec2` (set in `.orchestration-kit.env`), RUN phases launch on EC2. Set `COMPUTE_TARGET=local` to run locally instead.

1. The RUN phase injects a **mandatory** cloud-run directive into the sub-agent prompt
2. `sync_results()` runs automatically between RUN and READ, pulling results from cloud-run/S3
3. All composite commands (`cycle`, `full`, `program`) include the sync step

**Use block commands normally:**
```bash
# Preferred: block commands work with EC2 automatically
.kit/experiment.sh cycle <spec>
.kit/experiment.sh full "<question>" <spec>
.kit/experiment.sh program
```

**Manual phase orchestration is only needed if retrying a single failed phase:**
```bash
# Manual RUN + sync (retry scenario only)
orchestration-kit/tools/cloud-run run "python <script>" \
    --spec <spec> \
    --data-dirs .kit/results/full-year-export/ \
    --output-dir .kit/results/<experiment-name>/

# Pull results
orchestration-kit/tools/cloud-run pull <run-id> --output-dir .kit/results/<experiment-name>/

# Then resume
.kit/experiment.sh read "<question>" <spec>
```

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
- **Auto-bootstrap**: The `post-checkout` hook automatically fixes `PROJECT_ROOT` in `.orchestration-kit.env`, restores kit symlinks, and warns about broken artifact symlinks. Install it with: `orchestration-kit/tools/install-hooks`.
- **Subtree, not submodule**: `orchestration-kit/` is a git subtree, so worktrees get the full directory contents (no empty gitlink).

## Path Convention

Kit state files, working directories, and utility scripts live in `.kit/`. Project source code (`src/`, `tests/`, etc.) stays at the project root. The kit prompts reference bare filenames (e.g., `LAST_TOUCH.md`) — the `KIT_STATE_DIR` environment variable tells the scripts to resolve these inside `.kit/`.

Files at project root: `CLAUDE.md`, `.claude/`, `.orchestration-kit.env` (tracked — contains local MCP token, no external secrets), `.gitignore`
Not tracked (gitignored): `.env.local`, `.master-kit.env` — machine-specific overrides with per-machine tokens
Everything else kit-related: `.kit/`

## Available Kits

| Kit | Script | Phases |
|-----|--------|--------|
| **TDD** | `.kit/tdd.sh` | red, green, refactor, ship, full, watch |
| **Research** | `.kit/experiment.sh` | survey, frame, run, read, log, cycle, full, program, status |
| **Math** | `.kit/math.sh` | survey, specify, construct, formalize, prove, audit, log, full, program, status |

## Phase Orchestration (MANDATORY)

**Never run individual phases manually.** Use block commands that run the full cycle in one shot. This ensures automatic tracking in `program_state.json` and reduces orchestrator tool calls.

### Primary Interface: MCP Tools

The orchestration-kit exposes MCP tools via `.mcp.json` (stdio transport). **Use MCP tools as the primary interface** — they handle environment setup, run tracking, and dashboard integration automatically.

#### Execution (fire-and-forget — returns `run_id` immediately)

| MCP Tool | Parameters | Equivalent |
|----------|-----------|------------|
| `kit.tdd` | `spec_path` | `tdd.sh full <spec>` |
| `kit.research_cycle` | `spec_path` | `experiment.sh cycle <spec>` |
| `kit.research_full` | `question`, `spec_path` | `experiment.sh full <q> <spec>` |
| `kit.research_program` | _(none)_ | `experiment.sh program` |
| `kit.research_batch` | `spec_paths` (list) | `experiment.sh batch <spec1> <spec2> ...` |
| `kit.math` | `spec_path` | `math.sh full <spec>` |

#### Dashboard Queries (synchronous — returns data inline)

| MCP Tool | Parameters | Equivalent |
|----------|-----------|------------|
| `kit.status` | _(none)_ | `GET /api/summary` |
| `kit.runs` | `status?`, `kit?`, `phase?`, `limit?` | `GET /api/runs?...` |
| `kit.capsule` | `run_id` | `GET /api/capsule-preview?run_id=` |
| `kit.research_status` | _(none)_ | `experiment.sh status` |

### Fallback: Bash Commands

If MCP tools are unavailable, use bash block commands:

```bash
# Research
source .orchestration-kit.env
.kit/experiment.sh cycle .kit/experiments/<spec>.md
.kit/experiment.sh full "research question" .kit/experiments/<spec>.md
.kit/experiment.sh program
.kit/experiment.sh status

# TDD
source .orchestration-kit.env
.kit/tdd.sh full .kit/docs/<feature>.md
```

**Do NOT** call individual phases (`survey`, `frame`, `run`, `read`, `red`, `green`) unless retrying a single failed phase.

### Research → TDD Sub-Cycle

When a research phase needs **new C++ code**, spawn a TDD sub-cycle:

1. Write a spec: `.kit/docs/<feature>.md`
2. Run: `kit.tdd` with `spec_path=".kit/docs/<feature>.md"` (or bash fallback: `.kit/tdd.sh full .kit/docs/<feature>.md`)
3. Resume research with the new tested infrastructure available.

**Triggers**: Data extraction tools, new analysis APIs, infrastructure modifications.
**Non-triggers**: Disposable Python scripts, config files, parameter sweeps — these stay in Research kit.

## Dashboard as Status Interface (MANDATORY)

The project-local dashboard (scoped to this repo only) is the single source of truth for run status. **Do not read capsule files, log files, or events.jsonl directly.** Use MCP query tools (preferred) or the dashboard API.

### Primary: MCP Tools

- **Quick summary:** `kit.status` → total/running/ok/failed counts
- **List runs:** `kit.runs` → filterable by `status`, `kit`, `phase`, `limit`
- **Failed runs only:** `kit.runs` with `status="failed"`
- **Capsule drill-down:** `kit.capsule` with `run_id="<id>"` → 30-line failure summary
- **Research program:** `kit.research_status` → experiments + questions overview

### On Failure — Capsule Drill-Down

When a phase fails (exit code != 0):
1. `kit.runs` with `status="failed"` → find the failed `run_id`
2. `kit.capsule` with `run_id="<id>"` → read the 30-line failure summary
3. Do NOT `cat` or `Read` the full log — capsules are the summary

### Fallback: curl

```bash
source .orchestration-kit.env
orchestration-kit/tools/dashboard ensure-service --wait-seconds 3
curl -s http://127.0.0.1:7340/api/summary | python3 -m json.tool
curl -s 'http://127.0.0.1:7340/api/runs?status=failed'
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

## S3 Artifact Store

Large experiment result files (CSVs, binary data, model checkpoints) are stored in S3 with content-addressed deduplication. Local files become symlinks to a cache directory. Git tracks the symlinks (tiny) and per-directory `.s3-manifest.json` files.

**S3 bucket:** `s3://kenoma-labs-research/artifact-store/<sha256-prefix-2>/<sha256>.<ext>`
**Local cache:** `.kit/.s3-cache/` (gitignored)

### Commands

```bash
# Push a single file to S3, replace with symlink
orchestration-kit/tools/artifact-store push <file>

# Push all files >10MB in a directory tree
orchestration-kit/tools/artifact-store push-dir .kit/results/ --threshold 10MB

# After clone/checkout: download from S3 and create symlinks
orchestration-kit/tools/artifact-store hydrate

# Check which files are cached/missing
orchestration-kit/tools/artifact-store status

# Verify SHA-256 integrity of cached files
orchestration-kit/tools/artifact-store verify
```

### New Clone / Worktree Workflow

After `git clone` or `git worktree add`, large result files are symlinks pointing to `.kit/.s3-cache/` which is empty. Run:

```bash
orchestration-kit/tools/artifact-store hydrate
```

This downloads all files referenced by `.s3-manifest.json` in `.kit/results/` and creates the symlinks.

### After Running Experiments

If a research phase produces large result files (>10 MB), push them before committing:

```bash
orchestration-kit/tools/artifact-store push-dir .kit/results/<experiment-name>/ --threshold 10MB
git add .kit/results/<experiment-name>/
git commit -m "results: <experiment-name>"
```

## Orchestration-Kit Sync (Subtree)

`orchestration-kit/` is a **git subtree** (not a submodule). This means:
- Worktrees get the full directory contents (no empty gitlink)
- Changes made inside `orchestration-kit/` are normal git commits
- Two-way sync with the upstream repo via `sync-upstream`

```bash
# Check sync status (divergence between local and upstream)
orchestration-kit/tools/sync-upstream status

# Pull latest upstream changes into orchestration-kit/
orchestration-kit/tools/sync-upstream pull

# Push local orchestration-kit/ changes back to upstream
orchestration-kit/tools/sync-upstream push
```

**Upstream remote:** `orchestration-kit-upstream` → `https://github.com/kurtbell87/orchestration-kit.git`
Configured in `.orchestration-kit.env` via `ORCHESTRATION_KIT_UPSTREAM_REMOTE` and `ORCHESTRATION_KIT_UPSTREAM_BRANCH`.

## Project Health Check

```bash
# Full diagnostic — checks repo structure, secrets, symlinks, artifacts, git health
orchestration-kit/tools/project-doctor

# Auto-fix what's possible (missing symlinks, loose objects, remote config)
orchestration-kit/tools/project-doctor --fix

# Machine-readable output
orchestration-kit/tools/project-doctor --json
```

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
- **Verify bar construction semantics before running experiments.** The C++ `bar_feature_export` tick bar mode previously counted fixed-rate book snapshots (10/s), not trade events — **FIXED in tick-bar-fix TDD cycle (2026-02-19).** `book_builder.hpp` now emits `trade_count` per snapshot; `tick_bar_builder.hpp` accumulates trade counts. Diagnostic: (1) check bars_per_day_std — genuine event bars must have non-zero daily variance across trading days; (2) check within-day p10 vs p90 — time bars have p10=p90 (zero variance), genuine event bars don't. R3b wasted a cycle on the old defect. **Blast radius of old defect**: every "tick bar" experiment (R1, R4c, R4d, R3b) was actually time bars — those results are void for tick bars. Dollar and volume bars were always genuine.
- **Always prefer system-installed dependencies (`brew install`) over FetchContent for large C++ libraries.** Arrow C++ via FetchContent added 15-30 min per build iteration and burned 3.5 hours in a single GREEN phase on repeated rebuilds + a source-patching hack. Use `find_package()` for Arrow, Boost, and any library available via homebrew. FetchContent is fine for small libraries (GTest, nlohmann_json). Apache Arrow is installed system-wide: `brew install apache-arrow` → `find_package(Arrow REQUIRED)` / `find_package(Parquet REQUIRED)`.
- **Dollar and volume bars are genuine event bars; tick bars NOW FIXED.** R1 metrics contain `bar_count_cv` for all 12 configs — the diagnostic was available from day 1 but nobody checked. Tick bars: CV=0 (was broken, fixed 2026-02-19). Dollar bars: CV=2-6% (genuine). Volume bars: CV=9-10% (genuine). Time bars: CV=0 (correct by design). Additional dollar bar evidence: within-day p10≠p90, daily counts differ from day1 count. Additional volume bar evidence: vol_100 produces 6,087 bars/day vs 2,315 predicted by snapshot-counting model.

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

Your ONLY job: launch kit phases (via MCP tools or bash fallback), check status via dashboard, update state files.

1. **Use MCP tools or block commands only.** `kit.tdd`, `kit.research_cycle`, `kit.research_full`, `kit.research_program`, `kit.math`. Or bash equivalents: `experiment.sh cycle`, `tdd.sh full`, etc. Never call individual phases unless retrying a single failed phase.
2. **Execution tools are fire-and-forget.** They return a `run_id` immediately. Check completion via `kit.status` or `kit.runs`.
3. **Check status via MCP query tools, not by reading files.** Use `kit.status` for quick counts. Use `kit.runs` with `status="failed"` to find failures. Use `kit.research_status` for research program overview. Fallback: `curl` to dashboard API.
4. **Never run Bash for verification.** No `pytest`, `lake build`, `ls`, `cat`, `grep`, `python3 -c` to check what a sub-agent produced. If the phase exited 0, it worked.
5. **Never read implementation files** the sub-agents wrote (source code, test files, .lean files, experiment scripts, Python modules). That is their domain. You read only state files (CLAUDE.md, `.kit/LAST_TOUCH.md`, `.kit/RESEARCH_LOG.md`, etc.) and spec files (`.kit/docs/*.md`, `.kit/experiments/*.md`).
6. **Never write code.** Not C++, Python, shell scripts, config files, or anything else. If code must be created, it happens inside a kit phase (TDD green creates code, Research run creates scripts). You delegate, you do not implement.
7. **Chain phases by run status only.** `kit.runs` shows status. On failure → `kit.capsule` with the `run_id` → decide whether to retry or stop.
8. **Never read capsules or logs directly after success.** On failure, use `kit.capsule` (or dashboard API + `query-log`). Never `cat` or `Read` raw log files.
9. **Minimize tool calls.** Each Bash call, Read, or Glob adds to your context. If the information isn't needed to decide the next action, don't fetch it.
10. **Allowed tool usage summary:**
    - `MCP tools`: `kit.tdd`, `kit.research_cycle`, `kit.research_full`, `kit.research_program`, `kit.math`, `kit.status`, `kit.runs`, `kit.capsule`, `kit.research_status`. Plus legacy `orchestrator.*` tools.
    - `Bash` (fallback): `source .orchestration-kit.env && .kit/tdd.sh full ...`, `.kit/experiment.sh cycle|full|program ...`, `orchestration-kit/tools/*`, `curl` to dashboard API, `mkdir -p` for results dirs, `./build/<tool>` for data export.
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

## Current State (updated 2026-02-23, Parallel Batch Dispatch — Complete)

**Parallel batch dispatch for cloud-run — COMPLETE (TDD cycle on `tdd/parallel-batch-dispatch` branch).** All components delivered: `batch.py` module, CLI `batch {run,status,pull,ls}` subcommands, MCP `kit.research_batch` tool, `batch_id` tracking in state/remote, `parallelizable` surfacing in preflight, and `experiment.sh batch` shell command. Final piece: `run_batch()` function + case dispatch + help text added to `orchestration-kit/research-kit/experiment.sh`. Tests pass.

**VERDICT: CNN LINE CLOSED FOR CLASSIFICATION. GBT-only is the path forward.** End-to-end CNN classification (Outcome D) — GBT-only beats CNN by 5.9pp accuracy and $0.069 expectancy. CNN spatial signal (R²=0.089 regression) does not encode class-discriminative boundaries. Full-year CPCV (45 splits, 1.16M bars, PBO=0.222): GBT accuracy 0.449, expectancy -$0.064 (base). GBT is **marginally profitable in Q1 (+$0.003) and Q2 (+$0.029)** under base costs — edge exists but consumed by Q3-Q4 losses. Holdout accuracy 0.421, expectancy -$0.204. Next: XGBoost hyperparameter tuning (never optimized, default params from 9B), label design sensitivity, or regime-conditional trading.

**E2E CNN Classification (2026-02-22) — REFUTED (Outcome D).** End-to-end Conv1d CNN on 3-class tb_label with CrossEntropyLoss, full-year CPCV (45 splits, 10 groups, k=2). E2E-CNN accuracy 0.390, GBT-only 0.449 — CNN is 5.9pp WORSE. CNN expectancy -$0.146, GBT -$0.064. PBO=0.222, DSR~0. Holdout (50 days): acc 0.421, exp -$0.204 (GBT). GBT Q1-Q2 marginally positive (+$0.003, +$0.029 base). Walk-forward acc 0.456 (agrees with CPCV). CNN+Features augmented config skipped (wall-clock). Long (+1) recall only 0.21 — model asymmetrically confident on shorts. See `.kit/results/e2e-cnn-classification/metrics.json`.

**Full-Year Export (2026-02-20) — CONFIRMED.** 251/251 days, 1,160,150 bars, 0 failures, 0 duplicates. 10/10 SC pass. Parquet w/ zstd, 255.7 MB, 149 columns. 19/19 reference days validated. Results in S3 artifact store. See `.kit/results/full-year-export/metrics.json`.

**Docker/ECR/EBS Cloud Pipeline (2026-02-21) — VERIFIED E2E.** Dockerfile rewritten (multi-stage, lib isolation). ECR repo + EBS snapshot (49GB MBO data) + IAM profile created. ec2-bootstrap.sh fixed (awk, retry, NVMe, log trap). Full E2E: EC2 → EBS mount → ECR pull → docker run → S3 upload → self-terminate.

**9E (hybrid-model-corrected) — REFUTED (Outcome B).** CNN normalization fix VERIFIED (3rd independent reproduction: R²=0.089 vs 9D's 0.084). All 5 folds within ±0.015 of 9D. But end-to-end pipeline not viable: XGBoost acc=0.419, expectancy=-$0.37/trade (base), PF=0.924. Hybrid > GBT-nobook (+$0.075 exp) and GBT-book (+$0.013 exp). volatility_50 dominates feature importance (19.9 gain). return_5 ranked ~26th (no leakage). CNN acts as denoiser (16-dim embedding beats raw 40-dim book for XGBoost). See `.kit/results/hybrid-model-corrected/analysis.md`.

**R3b-genuine (CNN on Genuine Tick Bars) — CONFIRMED (low confidence).** Tick bar fix validated: all 8 thresholds have CV 0.188–0.467, p10≠p90 (genuine event bars). tick_100 mean OOS R²=0.124 (Δ+0.035 vs 0.089 baseline), but paired t-test p=0.21 — driven by fold 5's anomalous R²=0.259 (excluding fold 5: mean=0.091, COMPARABLE). tick_25 WORSE (0.064), tick_500 WORSE (0.050, 3/5 folds). Inverted-U curve. Not actionable without multi-seed replication. See `.kit/results/r3b-genuine-tick-bars/analysis.md`.

**R3b-original (CNN on Event Bars) — INCONCLUSIVE + SYSTEMIC TICK BAR DEFECT.** Tick bars from bar_feature_export were actually time bars at different frequencies (bars_per_day std=0.0 at all thresholds). Peak R²=0.057 (tick_100 ≈ time_10s), all WORSE than time_5s baseline (0.084). Tick bar fix TDD complete — `bar_feature_export --bar-type tick` now counts action='T' trade events.

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
| **TB-Fix** | **`.kit/docs/tick-bar-fix.md`** | **TDD** | **Done** — tick bars now count trades, not snapshots |
| **9E** | **`.kit/experiments/hybrid-model-corrected.md`** | **Research** | **Done (REFUTED — Outcome B)** — CNN R²=0.089, expectancy=-$0.37/trade |
| **R3b-genuine** | **`.kit/experiments/r3b-genuine-tick-bars.md`** | **Research** | **Done (CONFIRMED low confidence)** — tick_100 R²=0.124, p=0.21, not actionable |
| **FYE** | **`.kit/experiments/full-year-export.md`** | **Research** | **Done (CONFIRMED)** — 251 days, 1.16M bars, 10/10 SC pass |
| **Infra** | **Dockerfile + ec2-bootstrap** | **Chore** | **Done** — Docker/ECR/EBS pipeline verified E2E |
| **10** | **`.kit/experiments/e2e-cnn-classification.md`** | **Research** | **Done (REFUTED — Outcome D)** — GBT beats CNN by 5.9pp; CNN line closed |
| **Batch** | **`.kit/docs/parallel-batch-dispatch.md`** | **TDD** | **Done** — parallel batch dispatch for cloud-run |
| **Batch-sh** | **`.kit/docs/experiment-batch-command.md`** | **TDD** | **Done** — `experiment.sh batch` command |

- **Build:** Green.
- **Tests:** 1003/1004 unit tests pass (1 disabled, 1 skipped) + new tick_bar_fix tests. 22 integration tests (labeled, excluded from default ctest). TDD phases exited 0.
- **Exit criteria audit:** TRAJECTORY.md §13 audited — 21/21 engineering PASS, 13/13 research PASS (R4c closes MI/decay gap).
- **Corrected Hybrid Model COMPLETE (2026-02-19):** CNN normalization fix verified (3rd independent reproduction). R²=0.089 with proper validation. But end-to-end pipeline not economically viable: expectancy=-$0.37/trade (base), PF=0.924. Breakeven RT=$3.37. Hybrid outperforms GBT-only but delta too small to flip sign.
- **Next task options (in priority order):**
  1. **XGBoost hyperparameter tuning on full-year data** — default params from 9B never optimized. GBT already shows Q1-Q2 positive expectancy (+$0.003, +$0.029) with default hyperparams. Grid/random search over max_depth, learning_rate, n_estimators, subsample, colsample, min_child_weight. Most promising path given Outcome D.
  2. **Label design sensitivity** — test wider target (15 ticks) / narrower stop (3 ticks). At 15:3 ratio, breakeven win rate drops to ~42.5% (well below current ~45%). Also test asymmetric cost functions.
  3. **Regime-conditional trading** — Q1-Q2 only strategy. GBT profitable in H1 2022, negative in H2. Cannot validate with only 1 year of data, but could explore what regime features predict profitability.
  4. **2-class formulation** — directional only (merge tb_label=0 into abstain). Long recall is only 0.21 — model struggles with longs. Might perform better as binary short/no-short.
  5. **CNN line CLOSED** — do not revisit CNN for classification. Signal exists for regression but does not transfer.
- **Volume bars confirmed genuine** (2026-02-19): R1 metrics show bar_count_cv=9-10% for vol_50/100/200. R4b volume_100 null result is valid.
- **R3b-genuine tick bars COMPLETE** (2026-02-19): tick_100 R²=0.124 (Δ+0.035 vs 0.089), inverted-U curve, but p=0.21 — statistically fragile, driven by fold 5 anomaly.
