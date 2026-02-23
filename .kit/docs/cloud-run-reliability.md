# Cloud-Run Reliability Overhaul

## Problem

Untested code ships to EC2, burns 5+ hours of GPU compute, and failures are only discoverable by manually digging through S3 logs. Root causes:

1. **No automated code validation before EC2** — preflight only checks compute profile YAML, not the script itself
2. **No real-time log visibility** — logs upload only at exit (trap), invisible during multi-hour runs
3. **No heartbeat / instance health check** — `poll_status()` only checks S3 for `exit_code`; dead instances show as "running" forever
4. **Results sync only at the end** — partial results lost on crash

## Scope

All changes are in `orchestration-kit/tools/cloud/` and `orchestration-kit/tools/cloud-run`. One update to `.kit/experiment.sh`.

## Implementation

### 1. Bootstrap Sync Daemon (`ec2-bootstrap-gpu.sh`, `ec2-bootstrap.sh`)

Add a background sync daemon between the watchdog setup and experiment execution that handles heartbeat, log streaming, and incremental result sync.

**In `orchestration-kit/tools/cloud/scripts/ec2-bootstrap-gpu.sh`:**

After the existing watchdog setup (around line 83) and before the experiment execution, add:

```bash
# ── Sync daemon: heartbeat + log + incremental results ──────────
(
    _sync_counter=0
    while true; do
        sleep 60
        _sync_counter=$((_sync_counter + 1))

        # Heartbeat: write UTC timestamp to S3 (every 60s)
        date -u +%Y-%m-%dT%H:%M:%SZ | aws s3 cp - "${S3_BASE}/heartbeat" --quiet 2>/dev/null || true

        # Log: upload current log file (every 60s)
        aws s3 cp "$LOGFILE" "${S3_BASE}/experiment.log" --quiet 2>/dev/null || true

        # Results: sync every 5 minutes
        if (( _sync_counter % 5 == 0 )); then
            aws s3 sync /work/results/ "${S3_BASE}/results/" --quiet 2>/dev/null || true
        fi
    done
) &
SYNC_DAEMON_PID=$!
echo "[bootstrap] Sync daemon started (PID=$SYNC_DAEMON_PID)"
```

After the experiment execution completes (around line 91), kill the daemon:

```bash
# Kill sync daemon
if [[ -n "${SYNC_DAEMON_PID:-}" ]]; then
    kill "$SYNC_DAEMON_PID" 2>/dev/null || true
    wait "$SYNC_DAEMON_PID" 2>/dev/null || true
    echo "[bootstrap] Sync daemon stopped"
fi
```

Update the EXIT trap (around line 14) to sync results BEFORE writing exit_code:

```bash
cleanup() {
    local exit_code=$?
    echo "[bootstrap] Cleaning up (exit_code=$exit_code)"

    # Kill sync daemon first
    if [[ -n "${SYNC_DAEMON_PID:-}" ]]; then
        kill "$SYNC_DAEMON_PID" 2>/dev/null || true
        wait "$SYNC_DAEMON_PID" 2>/dev/null || true
    fi

    # Final results sync BEFORE exit_code (so results are available)
    aws s3 sync /work/results/ "${S3_BASE}/results/" --quiet 2>/dev/null || true

    # Upload final log
    aws s3 cp "$LOGFILE" "${S3_BASE}/experiment.log" --quiet 2>/dev/null || true

    # Write exit code LAST (this is the completion signal)
    echo "$exit_code" | aws s3 cp - "${S3_BASE}/exit_code" --quiet

    # Self-terminate (existing behavior)
    ...
}
```

IMPORTANT: The SYNC_DAEMON_PID variable must be initialized as empty before the trap is set, since the trap may fire before the daemon starts:
```bash
SYNC_DAEMON_PID=""
```

**In `orchestration-kit/tools/cloud/scripts/ec2-bootstrap.sh`:**

Same pattern, but sync `/opt/results/` instead of `/work/results/` (this is the non-GPU bootstrap). Apply identical changes:
- Add `SYNC_DAEMON_PID=""` initialization
- Update the cleanup/EXIT trap to kill daemon and sync results before exit_code
- Add sync daemon background loop after watchdog, before experiment
- Kill daemon after experiment completes

### 2. S3 Health & Log Functions (`s3.py`)

Add two new functions to `orchestration-kit/tools/cloud/s3.py`:

**`check_heartbeat(run_id: str) -> dict | None`:**
- Reads `{s3_base}/{run_id}/heartbeat` from S3
- Returns `{"timestamp": "<ISO-8601>", "age_seconds": <int>}` if found
- Returns `None` if no heartbeat exists
- Uses `s3_client.get_object()` — catch `NoSuchKey` / `ClientError` and return None

**`tail_log(run_id: str, lines: int = 50, follow: bool = False) -> str`:**
- Downloads `{s3_base}/{run_id}/experiment.log` from S3
- Returns the last `lines` lines as a string
- If `follow=True`: enters a polling loop (10s interval), re-downloads and prints only NEW lines (track by byte offset or line count). Loop exits when exit_code file appears in S3 or after 30 minutes (safety timeout).
- For first fetch: use `get_object()` and split by newlines
- For follow re-fetch: use `get_object(Range=...)` byte-range if Content-Length increased, to minimize transfer

The S3 base path pattern is: `s3://{bucket}/cloud-run/{run_id}/` — derive from existing config (look at how `poll_status` and `upload_results` construct S3 paths).

### 3. Instance Health Monitoring (`remote.py`)

**Enhance `poll_status()`** (around lines 227-243):

Currently `poll_status()` checks S3 for `exit_code`. When no exit_code is found, it returns `{"status": "running"}`. Enhance it:

When exit_code is not found in S3:
1. Call the backend's `status()` method to get EC2 instance state
2. If instance state is `terminated`, `stopped`, or `shutting-down` but no exit_code → return `{"status": "terminated_no_results", "instance_state": "<state>", "message": "Instance terminated without writing results"}`
3. Call `s3_helper.check_heartbeat(run_id)` — if heartbeat exists, include `last_heartbeat` and `heartbeat_age_seconds` in the response
4. If heartbeat age > 600 seconds (10 min), add `"warning": "heartbeat_stale"` to the response

**Add `gc_stale_runs()` function:**
- List all run state files in S3 with status "running"
- For each, check EC2 instance state via backend API
- If instance is terminated/stopped and no exit_code exists:
  - Write `exit_code=137` to S3 (terminated)
  - Update local state to `terminated_no_results`
- Return count of cleaned-up runs

### 4. Log Tailing CLI (`cloud-run`)

Add a new subcommand `logs` to the `cloud-run` CLI:

```
cloud-run logs <run-id> [--lines N] [--follow]
```

Implementation in `cmd_logs()`:
- Parse `run_id` from args
- Default `--lines` to 50
- Call `s3_helper.tail_log(run_id, lines=args.lines, follow=args.follow)`
- Print the result to stdout
- If `--follow`, the tail_log function handles the polling loop

Add to the argparse subparser:
```python
p_logs = sub.add_parser("logs", help="Tail experiment log from S3")
p_logs.add_argument("run_id", help="Run ID")
p_logs.add_argument("--lines", "-n", type=int, default=50, help="Number of lines (default 50)")
p_logs.add_argument("--follow", "-f", action="store_true", help="Follow log output (poll every 10s)")
p_logs.set_defaults(func=cmd_logs)
```

### 5. Enhanced Status Display (`cloud-run`)

**Enhance `_print_run()` (around line 466):**
- For running instances: show elapsed time since launch (compute from run start timestamp)
- Show cost estimate: `elapsed_hours * hourly_rate` (get hourly rate from instance type in state, or default $0.50/hr for GPU, $0.10/hr for CPU)
- Show last heartbeat time and age if available (call `s3_helper.check_heartbeat()`)
- Show last 3 log lines if available (call `s3_helper.tail_log(run_id, lines=3)`)

**Enhance `cmd_ls()` (around line 185):**
- Add "Elapsed" and "Est. Cost" columns for running instances
- Compute from run start time in state file

**Enhance `cmd_status()` (around line 158):**
- For running instances: show last 5 log lines

### 6. Pre-flight Code Validation (`validate.py`, `cloud-run`, `experiment.sh`)

**New file: `orchestration-kit/tools/cloud/validate.py`** (~80 lines):

```python
"""Pre-flight validation for experiment scripts before cloud execution."""

import ast
import subprocess
import sys
from pathlib import Path


def syntax_check(script: str) -> tuple[bool, str]:
    """Run py_compile on the script. Returns (ok, message)."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", script],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return True, "Syntax OK"
        return False, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "Syntax check timed out"


def import_check(script: str) -> tuple[bool, str]:
    """AST-parse the script, extract imports, verify each is importable."""
    try:
        with open(script) as f:
            tree = ast.parse(f.read())
    except SyntaxError as e:
        return False, f"Parse error: {e}"

    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module.split('.')[0])

    # Skip stdlib and known-available modules
    # Only check top-level package importability
    missing = []
    for mod in sorted(modules):
        try:
            result = subprocess.run(
                [sys.executable, "-c", f"import {mod}"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                missing.append(mod)
        except subprocess.TimeoutExpired:
            missing.append(f"{mod} (timeout)")

    if missing:
        return False, f"Missing imports: {', '.join(missing)}"
    return True, f"All {len(modules)} imports OK"


def smoke_test(script: str, timeout: int = 300) -> tuple[bool, str]:
    """Run `python <script> --smoke-test` with timeout.

    The script should support --smoke-test flag that runs a minimal
    forward pass (1 batch, CPU/MPS) and exits. This catches shape
    mismatches, data loading errors, etc.
    """
    try:
        result = subprocess.run(
            [sys.executable, script, "--smoke-test"],
            capture_output=True, text=True, timeout=timeout,
            env={**__import__('os').environ, "CUDA_VISIBLE_DEVICES": ""}
        )
        if result.returncode == 0:
            return True, "Smoke test passed"
        # Return last 20 lines of stderr for diagnosis
        err_lines = result.stderr.strip().split('\n')
        return False, "Smoke test failed:\n" + '\n'.join(err_lines[-20:])
    except subprocess.TimeoutExpired:
        return False, f"Smoke test timed out after {timeout}s"


def validate_all(script: str, skip_smoke: bool = False) -> tuple[bool, list[tuple[str, bool, str]]]:
    """Run all validation checks. Returns (all_passed, [(check_name, passed, message), ...])."""
    results = []

    ok, msg = syntax_check(script)
    results.append(("syntax", ok, msg))
    if not ok:
        return False, results

    ok, msg = import_check(script)
    results.append(("imports", ok, msg))
    if not ok:
        return False, results

    if not skip_smoke:
        ok, msg = smoke_test(script)
        results.append(("smoke_test", ok, msg))
        if not ok:
            return False, results

    return True, results
```

**Modify `cloud-run` — add `--validate` flag:**

Add to the `run` subparser:
```python
p_run.add_argument("--validate", metavar="SCRIPT", help="Validate script before cloud execution")
p_run.add_argument("--skip-smoke", action="store_true", help="Skip smoke test (syntax + imports only)")
```

In `cmd_run()`, before calling `remote.run()`:
```python
if args.validate:
    from cloud.validate import validate_all
    all_ok, checks = validate_all(args.validate, skip_smoke=args.skip_smoke)
    for name, ok, msg in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: {msg}")
    if not all_ok:
        print("\nValidation failed. Not launching EC2 instance.")
        sys.exit(2)
    print("\nAll validation checks passed. Proceeding with cloud launch.\n")
```

**Modify `.kit/experiment.sh` — add `--validate` to compute directive template:**

In the compute directive template (around line 584), where the `cloud-run run` command template is injected into the sub-agent prompt, add `--validate <script_path>` to the command template. The directive should look like:

```bash
orchestration-kit/tools/cloud-run run --validate <SCRIPT_PATH> "<COMMAND>" \
    --spec <SPEC> \
    --data-dirs <DATA_DIRS> \
    --output-dir <OUTPUT_DIR>
```

The `<SCRIPT_PATH>` should be extracted from the `<COMMAND>` argument — typically the first argument after `python`. If extraction is ambiguous, fall back to no `--validate` flag.

### 7. Stale Run GC (`state.py`, `cloud-run`)

**Add `gc_stale()` to `orchestration-kit/tools/cloud/state.py`:**
- Read the project-local `cloud-state.json`
- For each entry with status "running" or "pending":
  - Check if the run has an exit_code in S3 (call s3 helper)
  - If yes, update local state to match (completed/failed)
  - If the run is older than 24 hours with no heartbeat, mark as stale
- Write updated state back
- Return count of cleaned entries

**Update `cmd_gc()` in `cloud-run`:**
- Call `gc_stale_runs()` from `remote.py` (checks EC2 instance state, writes exit_code=137 for dead instances)
- Call `gc_stale()` from `state.py` (cleans local cloud-state.json)
- Print summary: "Cleaned N stale runs, updated M local state entries"

## Exit Criteria

- [ ] `ec2-bootstrap-gpu.sh` has sync daemon (heartbeat every 60s, log every 60s, results every 5 min)
- [ ] `ec2-bootstrap-gpu.sh` EXIT trap syncs results before writing exit_code
- [ ] `ec2-bootstrap-gpu.sh` kills sync daemon after experiment and in trap
- [ ] `ec2-bootstrap.sh` has same sync daemon pattern (using /opt/results/ path)
- [ ] `s3.py` has `check_heartbeat()` function that reads heartbeat timestamp from S3
- [ ] `s3.py` has `tail_log()` function with follow mode support
- [ ] `remote.py` `poll_status()` checks EC2 instance state when no exit_code found
- [ ] `remote.py` `poll_status()` includes heartbeat info in response
- [ ] `remote.py` has `gc_stale_runs()` function
- [ ] `cloud-run` has `logs` subcommand with `--lines` and `--follow` flags
- [ ] `cloud-run` `_print_run()` shows elapsed time, cost estimate, heartbeat for running instances
- [ ] `cloud-run` `cmd_ls()` shows elapsed and cost columns
- [ ] `cloud-run` `cmd_status()` shows last log lines for running instances
- [ ] `validate.py` exists with `syntax_check()`, `import_check()`, `smoke_test()`, `validate_all()`
- [ ] `cloud-run run` has `--validate` and `--skip-smoke` flags
- [ ] `cloud-run run` exits 2 on validation failure without provisioning EC2
- [ ] `.kit/experiment.sh` compute directive template includes `--validate`
- [ ] `state.py` has `gc_stale()` function
- [ ] `cloud-run gc` calls both `gc_stale_runs()` and `gc_stale()`

## Non-Goals

- No changes to the actual experiment scripts or model code
- No changes to the RunPod backend (focus on EC2/S3 path only)
- No changes to the dashboard or MCP tools
- No new dependencies — use only stdlib + boto3 (already available)
