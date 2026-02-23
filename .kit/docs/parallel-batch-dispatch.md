# Parallel Batch Dispatch for Cloud-Run

## Overview

Add batch execution capability to cloud-run so N independent experiments can run simultaneously on separate EC2 instances. Each `remote.run()` already provisions a fresh instance with a unique `run_id` — what's missing is a dispatch layer that launches N experiments in parallel, monitors them all via a unified poll loop, and collects results as they finish.

## IMPORTANT: Language & Framework

- **Language**: Python 3.12 (all new/modified code), Bash (experiment.sh additions)
- **Test framework**: pytest — run with `cd orchestration-kit && python3 -m pytest tests/test_batch.py -v`
- **Test file**: `orchestration-kit/tests/test_batch.py`
- **NO C++ or CMake.** This is pure Python/Bash infrastructure work.
- **NO AWS credentials needed for tests.** Mock all cloud/remote operations in tests.

## Existing Code Contracts (READ THESE FILES FIRST)

Before writing any code, read the existing files to understand the APIs:

- `orchestration-kit/tools/cloud/remote.py` — `run()`, `poll_status()`, `pull_results()`, `list_runs()`, `_generate_run_id()`, `_load_state()`, `_save_state()`
- `orchestration-kit/tools/cloud/state.py` — `register_run()`, `remove_run()`, `list_active_runs()`, `get_run()`, `update_run()`
- `orchestration-kit/tools/cloud-run` — CLI entrypoint using argparse subparsers. `_get_backend()`, `cmd_run()`, etc.
- `orchestration-kit/tools/cloud/spec_parser.py` — `ComputeProfile` dataclass with `parallelizable: bool = False` field (already parsed, never used)
- `orchestration-kit/tools/cloud/preflight.py` — `check()`, `check_spec()`. Returns dict with recommendation, backend, instance_type, cost estimate, reason.
- `orchestration-kit/mcp/server.py` — `MasterKitFacade`, `TOOL_DEFINITIONS` list, `call_tool()` dispatch, `_launch_background()` for fire-and-forget.
- `orchestration-kit/research-kit/experiment.sh` — Shell functions (`run_run`, `sync_results`, `run_cycle`, etc.) with case-dispatch at bottom.

## Detailed Requirements

### 1. NEW: `orchestration-kit/tools/cloud/batch.py` (~200 lines)

Core module for parallel batch dispatch.

#### Batch State

Store batch metadata at `~/.orchestration-kit-cloud/batches/{batch_id}.json`:

```json
{
  "batch_id": "batch-20260223T120000Z-abc12345",
  "runs": {"spec-a.md": "cloud-20260223T120001Z-deadbeef", "spec-b.md": "cloud-20260223T120002Z-cafebabe"},
  "specs": ["spec-a.md", "spec-b.md"],
  "status": "running",
  "started_at": "2026-02-23T12:00:00Z",
  "finished_at": null,
  "max_instances": 5,
  "results": {}
}
```

#### Functions

**`generate_batch_id() -> str`**
Generate unique batch ID: `batch-{YYYYMMDDTHHMMSSZ}-{8-hex-chars}` using `uuid.uuid4().hex[:8]`.

**`_batch_state_dir() -> Path`**
Return `~/.orchestration-kit-cloud/batches/`, creating with `mkdir(parents=True, exist_ok=True)`.

**`save_batch_state(batch_id: str, state: dict) -> Path`**
Atomic write of batch state JSON to `{_batch_state_dir()}/{batch_id}.json`.

**`load_batch_state(batch_id: str) -> dict`**
Load batch state. Raise `FileNotFoundError` if missing.

**`launch_batch(...)` → dict**

```python
def launch_batch(
    *,
    specs: list[str],
    command: str,
    backend: "ComputeBackend",
    backend_name: str,
    project_root: str,
    instance_type: str,
    data_dirs: list[str] | None = None,
    sync_back: str = "results",
    output_base: str | None = None,
    use_spot: bool = True,
    max_hours: float = 12.0,
    env_vars: dict[str, str] | None = None,
    max_instances: int = 5,
    max_cost: float | None = None,
    gpu_mode: bool = False,
    image_tag: str | None = None,
) -> dict:
```

Logic:
1. Validate: `len(specs) <= max_instances`, else raise `ValueError(f"Batch size {len(specs)} exceeds max_instances={max_instances}")`
2. If `max_cost` is set: run `preflight.check_spec()` for each spec, sum `estimated_wall_hours * cost_per_hour_spot` (or similar), reject with `ValueError` if total > `max_cost`. Handle gracefully if a spec has no compute profile (skip cost estimate for that spec).
3. Generate `batch_id` via `generate_batch_id()`
4. For each spec:
   - Derive output_dir: `{output_base}/{Path(spec).stem}/` if output_base provided, else None
   - Call `remote.run(command=command, detach=True, spec_file=spec, local_results_dir=output_dir, batch_id=batch_id, ...)` with all relevant kwargs
   - Collect `run_id` from return dict
5. Save batch state with `status="running"`
6. Enter unified poll loop: every 30s, call `remote.poll_status(run_id)` for each run still in running/provisioning/pending status
7. As each run reaches terminal status (completed/failed/error/terminated): call `remote.pull_results(run_id, output_dir)` if completed, record result
8. When all runs are terminal: update batch state with `status="completed"` (or `"partial"` if any failed), set `finished_at`
9. Return batch state dict

**`poll_batch(batch_id: str) -> dict`**
Load batch state, call `remote.poll_status()` for each run, return updated state dict with per-run status.

**`pull_batch(batch_id: str, output_base: str | None = None) -> dict`**
Pull results for all completed runs in a batch. For each completed run, call `remote.pull_results()`. Return dict mapping spec -> pull status.

**`list_batches() -> list[dict]`**
Glob `_batch_state_dir()` for `batch-*.json`, load each, return sorted most-recent-first by `started_at`.

### 2. MODIFY: `orchestration-kit/tools/cloud/state.py`

**Add `batch_id` parameter to `register_run()`:**

```python
def register_run(
    project_root: str,
    run_id: str,
    *,
    instance_id: str,
    backend: str,
    instance_type: str,
    spec_file: Optional[str] = None,
    launched_at: Optional[str] = None,
    max_hours: float = 12.0,
    batch_id: Optional[str] = None,  # <-- NEW
) -> None:
```

Store `"batch_id": batch_id or ""` in the run entry dict (alongside existing fields).

**Add new function `list_batch_runs()`:**

```python
def list_batch_runs(project_root: str, batch_id: str) -> list[dict]:
    """Return all active runs belonging to a given batch_id."""
    data = _load(project_root)
    results = []
    for rid, entry in data["active_runs"].items():
        if entry.get("batch_id") == batch_id:
            results.append({"run_id": rid, **entry})
    return results
```

### 3. MODIFY: `orchestration-kit/tools/cloud/remote.py`

Add `batch_id: Optional[str] = None` parameter to the `run()` function signature.

Pass it through to `project_state.register_run()`:
```python
project_state.register_run(
    project_root, run_id,
    instance_id=instance_id,
    backend=backend_name,
    instance_type=instance_type,
    spec_file=spec_file,
    launched_at=config.launched_at,
    max_hours=max_hours,
    batch_id=batch_id,  # <-- NEW
)
```

Also store `batch_id` in the run's local state dict (the one saved to `~/.orchestration-kit-cloud/runs/`).

### 4. MODIFY: `orchestration-kit/tools/cloud-run` (CLI)

Add a `batch` subcommand group with 4 sub-subcommands. Add this BEFORE the `args = parser.parse_args()` line in `main()`.

**Subcommand group setup:**

```python
p_batch = sub.add_parser("batch", help="Parallel batch execution")
batch_sub = p_batch.add_subparsers(dest="batch_cmd")
```

**`batch run` subcommand:**

```python
p_br = batch_sub.add_parser("run", help="Launch batch of experiments in parallel")
p_br.add_argument("command", help="Shell command to execute on each instance")
p_br.add_argument("--specs", required=True, help="Comma-separated spec file paths")
p_br.add_argument("--max-instances", type=int, default=5, help="Max concurrent instances (default: 5)")
p_br.add_argument("--max-cost", type=float, help="Reject if estimated total cost exceeds this (USD)")
# Inherit from run:
p_br.add_argument("--backend", choices=["aws", "runpod"])
p_br.add_argument("--instance-type")
p_br.add_argument("--data-dirs")
p_br.add_argument("--sync-back", default="results")
p_br.add_argument("--output-dir")
p_br.add_argument("--project-root")
p_br.add_argument("--on-demand", action="store_true")
p_br.add_argument("--max-hours", type=float, default=DEFAULT_MAX_HOURS)
p_br.add_argument("--env", action="append")
p_br.add_argument("--gpu", action="store_true")
p_br.add_argument("--image-tag")
p_br.add_argument("--json", dest="json_output", action="store_true")
p_br.set_defaults(func=cmd_batch_run)
```

**`batch status` subcommand:**
```python
p_bs = batch_sub.add_parser("status", help="Check batch status")
p_bs.add_argument("batch_id", help="Batch ID")
p_bs.add_argument("--json", dest="json_output", action="store_true")
p_bs.set_defaults(func=cmd_batch_status)
```

**`batch pull` subcommand:**
```python
p_bp = batch_sub.add_parser("pull", help="Pull results for all runs in a batch")
p_bp.add_argument("batch_id", help="Batch ID")
p_bp.add_argument("--output-dir")
p_bp.set_defaults(func=cmd_batch_pull)
```

**`batch ls` subcommand:**
```python
p_bl = batch_sub.add_parser("ls", help="List tracked batches")
p_bl.set_defaults(func=cmd_batch_ls)
```

**Handler functions:**

```python
def cmd_batch_run(args):
    from cloud import batch as batch_mod
    specs = args.specs.split(",")
    backend_name = args.backend or "aws"
    gpu_mode = getattr(args, "gpu", False)
    if gpu_mode:
        backend_name = "aws"
    instance_type = args.instance_type or ("g5.xlarge" if gpu_mode else "c7a.8xlarge")
    backend = _get_backend(backend_name)
    project_root = args.project_root or os.environ.get("PROJECT_ROOT", os.getcwd())
    data_dirs = args.data_dirs.split(",") if args.data_dirs else None
    env_vars = {}
    if args.env:
        for e in args.env:
            k, _, v = e.partition("=")
            env_vars[k] = v
    result = batch_mod.launch_batch(
        specs=specs,
        command=args.command,
        backend=backend,
        backend_name=backend_name,
        project_root=project_root,
        instance_type=instance_type,
        data_dirs=data_dirs,
        sync_back=args.sync_back,
        output_base=args.output_dir,
        use_spot=not args.on_demand,
        max_hours=args.max_hours,
        env_vars=env_vars,
        max_instances=args.max_instances,
        max_cost=args.max_cost,
        gpu_mode=gpu_mode,
        image_tag=getattr(args, "image_tag", None),
    )
    if getattr(args, "json_output", False):
        print(json.dumps(result, indent=2))
    else:
        print(f"Batch {result['batch_id']}: {result['status']}")
        for spec, run_id in result.get("runs", {}).items():
            print(f"  {spec}: {run_id}")

def cmd_batch_status(args):
    from cloud import batch as batch_mod
    result = batch_mod.poll_batch(args.batch_id)
    if getattr(args, "json_output", False):
        print(json.dumps(result, indent=2))
    else:
        print(f"Batch: {result['batch_id']}  Status: {result['status']}")
        for spec, run_id in result.get("runs", {}).items():
            print(f"  {spec}: {run_id}")

def cmd_batch_pull(args):
    from cloud import batch as batch_mod
    result = batch_mod.pull_batch(args.batch_id, args.output_dir)
    print(f"Pulled {len(result)} result(s)")

def cmd_batch_ls(args):
    from cloud import batch as batch_mod
    batches = batch_mod.list_batches()
    if not batches:
        print("No tracked batches.")
        return
    fmt = "{:<40}  {:<12}  {:<6}  {}"
    print(fmt.format("BATCH_ID", "STATUS", "RUNS", "STARTED"))
    print("-" * 80)
    for b in batches:
        print(fmt.format(
            b.get("batch_id", "?"),
            b.get("status", "?"),
            str(len(b.get("specs", []))),
            b.get("started_at", "?")[:19],
        ))
```

Add dispatch for `batch` in main() — after the existing subcmd checks, add:
```python
if args.subcmd == "batch" and not getattr(args, "batch_cmd", None):
    p_batch.print_help()
    sys.exit(1)
```

### 5. MODIFY: `orchestration-kit/research-kit/experiment.sh`

Add `run_batch()` function and update the case dispatch.

**New function** (add before the `# Main` section):

```bash
run_batch() {
  # Run the RUN+sync phase for each spec in parallel via background subshells.
  # Frame and read/log phases are NOT included — they must be run separately
  # because they touch shared state files.
  #
  # Usage: experiment.sh batch <spec1> <spec2> ... <specN>

  if (( $# == 0 )); then
    echo -e "${RED}Usage: experiment.sh batch <spec1> <spec2> ... <specN>${NC}" >&2
    exit 1
  fi

  local specs=("$@")
  local n=${#specs[@]}

  echo ""
  echo -e "${BOLD}${CYAN}======================================================${NC}"
  echo -e "${BOLD}${CYAN}  BATCH MODE -- Parallel RUN+sync for $n specs${NC}"
  echo -e "${BOLD}${CYAN}======================================================${NC}"
  echo ""

  local pids=()
  local spec_for_pid=()

  for spec in "${specs[@]}"; do
    if [[ ! -f "$spec" ]]; then
      echo -e "${RED}Error: Spec file not found: $spec${NC}" >&2
      continue
    fi
    echo -e "  ${GREEN}Launching:${NC} $spec"
    (
      run_run "$spec"
      sync_results "$spec"
    ) &
    pids+=($!)
    spec_for_pid+=("$spec")
  done

  # Wait for all and collect exit codes
  local failed=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo -e "  ${RED}FAILED:${NC} ${spec_for_pid[$i]} (pid ${pids[$i]})"
      failed=$((failed + 1))
    else
      echo -e "  ${GREEN}OK:${NC} ${spec_for_pid[$i]}"
    fi
  done

  echo ""
  echo -e "${BOLD}Batch complete:${NC} $n specs, $failed failure(s)"

  if (( failed > 0 )); then
    return 1
  fi
  return 0
}
```

**Update case dispatch** — add `batch)` case:

In the `case "${1:-help}" in` block at the bottom, add:
```bash
  batch)      shift; run_batch "$@" ;;
```

Add it between the existing `full)` and `status)` cases.

### 6. MODIFY: `orchestration-kit/mcp/server.py`

**Add tool definition** to `TOOL_DEFINITIONS` list (after `kit.research_program`):

```python
{
    "name": "kit.research_batch",
    "description": "Run parallel batch of experiment RUN phases on separate EC2 instances. Returns immediately with run_id. Poll kit.status or kit.runs for completion.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "spec_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of experiment spec paths to run in parallel",
            },
        },
        "required": ["spec_paths"],
        "additionalProperties": False,
    },
},
```

**Add handler method** to `MasterKitFacade`:

```python
def _tool_kit_research_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
    spec_paths = payload.get("spec_paths", [])
    if not isinstance(spec_paths, list) or not spec_paths:
        raise ValueError("spec_paths must be a non-empty list of strings")
    for s in spec_paths:
        if not isinstance(s, str) or not s:
            raise ValueError("each spec_path must be a non-empty string")
    return self._launch_background("research", "batch", [str(s) for s in spec_paths])
```

**Add dispatch** in `call_tool()` — in the "Execution tools" section, add:

```python
if name == "kit.research_batch":
    return self._tool_kit_research_batch(arguments)
```

### 7. MODIFY: `orchestration-kit/tools/cloud/preflight.py`

**In `check()` function:**

Add `"parallelizable": profile.parallelizable` to all return dicts (both local and remote recommendations).

When recommendation is "remote" and `profile.parallelizable` is True, append to the reason string: `" (parallelizable — suitable for batch execution)"`

**In CLI `main()` function:**

When printing remote recommendation, if result has `"parallelizable": True`, add a line:
```python
if result.get("parallelizable"):
    print(f"  Parallelizable: yes (use 'cloud-run batch run' for parallel dispatch)")
```

## Test Requirements

### File: `orchestration-kit/tests/test_batch.py`

All tests must work WITHOUT AWS credentials. Mock all remote/cloud operations.

**Required test cases:**

1. **`test_generate_batch_id_format`** — Verify format matches `batch-{timestamp}-{8hex}` using regex.

2. **`test_save_load_batch_state_roundtrip`** — Save a batch state dict to a temp directory, load it back, verify equality. Use `monkeypatch` to override `_batch_state_dir()`.

3. **`test_launch_batch_exceeds_max_instances`** — Call `launch_batch()` with 6 specs and `max_instances=5`. Assert `ValueError` raised with message about exceeding max_instances.

4. **`test_launch_batch_exceeds_max_cost`** — Mock `preflight.check_spec()` to return estimated costs. Call `launch_batch()` with `max_cost=1.00` where total estimate is $5.00. Assert `ValueError` raised.

5. **`test_launch_batch_success`** — Mock `remote.run(detach=True)` to return fake run states. Mock `remote.poll_status()` to return completed after first poll. Mock `remote.pull_results()`. Call `launch_batch()` with 2 specs. Verify: both specs launched, poll loop ran, results pulled, batch state is "completed".

6. **`test_poll_batch`** — Create a batch state file with 2 run_ids. Mock `remote.poll_status()`. Call `poll_batch()`. Verify per-run status is updated.

7. **`test_pull_batch`** — Create a batch state with completed runs. Mock `remote.pull_results()`. Call `pull_batch()`. Verify pull called for each run.

8. **`test_list_batches_ordering`** — Create 3 batch state files with different `started_at` values. Verify `list_batches()` returns most recent first.

9. **`test_state_register_run_with_batch_id`** — Call `state.register_run()` with `batch_id="batch-test"`. Load state file. Verify `batch_id` field is stored.

10. **`test_state_list_batch_runs`** — Register 3 runs: 2 with batch_id="batch-A", 1 with batch_id="batch-B". Call `list_batch_runs("batch-A")`. Verify returns exactly 2 runs.

11. **`test_cli_batch_subparser_exists`** — Import main parser from cloud-run or construct it, verify `batch run` accepts expected args without error.

12. **`test_preflight_parallelizable_in_output`** — Create a mock spec file with `parallelizable: true` in compute profile. Call `check_spec()`. Verify returned dict contains `"parallelizable": True`.

**Run command**: `cd orchestration-kit && python3 -m pytest tests/test_batch.py -v`

## Non-Goals

- Do NOT modify the existing `remote.run()` polling behavior for single runs
- Do NOT change the instance provisioning logic (one instance per run is correct)
- Do NOT add batch support to RunPod backend (AWS only for now)
- Do NOT implement batch resume/retry (can be added later)

## Exit Criteria

- [ ] `orchestration-kit/tools/cloud/batch.py` exists with `launch_batch()`, `poll_batch()`, `pull_batch()`, `list_batches()`, `generate_batch_id()`, `save_batch_state()`, `load_batch_state()`
- [ ] `orchestration-kit/tools/cloud/state.py` has `batch_id` parameter on `register_run()` and new `list_batch_runs()` function
- [ ] `orchestration-kit/tools/cloud/remote.py` accepts and passes `batch_id` through to state registration
- [ ] `orchestration-kit/tools/cloud-run` has `batch run`, `batch status`, `batch pull`, `batch ls` subcommands
- [ ] `orchestration-kit/research-kit/experiment.sh` has `batch` command with parallel subshells
- [ ] `orchestration-kit/mcp/server.py` has `kit.research_batch` tool definition and handler
- [ ] `orchestration-kit/tools/cloud/preflight.py` surfaces `parallelizable` field in output and adds advisory note
- [ ] All tests pass: `cd orchestration-kit && python3 -m pytest tests/test_batch.py -v`
- [ ] Existing code remains backward compatible (no breaking changes to existing `register_run()`, `run()`, CLI, or MCP APIs)
- [ ] `bash -n orchestration-kit/research-kit/experiment.sh` passes (no syntax errors)
