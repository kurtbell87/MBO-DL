# TDD Spec: Nested Run Visibility (Run Tree + Live Tailing)

**Date:** 2026-02-20
**Priority:** HIGH — foundational infrastructure for orchestrator observability
**Branch:** `feat/nested-run-visibility`

---

## Context

The orchestration-kit MCP server launches kit phases (research, TDD, math) as fire-and-forget subprocesses. Each phase creates a "run" tracked in `orchestration-kit/runs/<run_id>/`. When a research RUN phase spawns a TDD sub-cycle, the child run gets its own `run_id` with `--parent-run-id` linking to the parent.

**Current gap:** The orchestrator has no way to discover child runs or observe the full execution tree. After launching `kit.research_cycle`, you get one `run_id` and can only tail that run's log. If the research RUN phase spawns `tdd.sh full` internally, that child run is invisible unless you scan all runs manually.

### What exists

- `tools/kit` already supports `--parent-run-id` (line 538 of `tools/kit`)
- Each run's `events.jsonl` records structured events (`run_started`, `phase_started`, `phase_finished`, `run_finished`)
- The `run_started` event includes `parent_run_id` field
- Dashboard API at `/api/runs` returns `parent_run_id` per run but has no parent-child query
- `orchestrator_run_info` returns capsules/manifests/logs for a single run

### What's broken (fixed in this branch)

Four bugs were fixed in `orchestration-kit/mcp/server.py` on `feat/full-year-parquet-export` that must be cherry-picked onto this branch:

1. **`--run-id` swallowed by argparse REMAINDER** — options after positional args in `_launch_background` were consumed by `phase_args` instead of parsed. Fix: place `--run-id` before positionals.
2. **DEVNULL error blindness** — `stdout=DEVNULL, stderr=DEVNULL` in `_launch_background` silenced all subprocess errors. Fix: capture to `runs/mcp-launches/{run_id}.log`.
3. **`--reasoning` swallowed by REMAINDER** — same pattern in `_tool_run`. Fix: place before positionals.
4. **Missing `KIT_STATE_DIR`** — MCP subprocess didn't inherit `KIT_STATE_DIR`, causing `experiment.sh` path resolution failure. Fix: auto-detect from project structure, store in `ServerConfig`, forward to subprocesses.

---

## Requirements

### R1: `child_run_spawned` Event in Parent's Event Stream

When `tools/kit` starts a run with `--parent-run-id`, it must append a `child_run_spawned` event to the **parent's** `events.jsonl`:

```json
{
  "ts": "2026-02-20T02:34:40Z",
  "event": "child_run_spawned",
  "parent_run_id": "20260220T023440Z-bd8a4495",
  "child_run_id": "20260220T023510Z-abc12345",
  "child_kit": "tdd",
  "child_phase": "full"
}
```

**Constraints:**
- The parent's `events.jsonl` path is derivable from `--parent-run-id`: `RUNS_DIR / parent_run_id / "events.jsonl"`
- Use the existing `append_event()` function with file locking (already implemented)
- If the parent events file doesn't exist (orphan run), skip silently — don't crash

### R2: Dashboard `/api/runs` Parent-Child Filter

Add `parent_run_id` as a query parameter to the dashboard's `/api/runs` endpoint:

- `GET /api/runs?parent_run_id=X` → returns all runs where `parent_run_id == X`
- Existing filters (`status`, `kit`, `phase`, `limit`) continue to work and can be combined with `parent_run_id`

### R3: `kit.run_tree` MCP Tool

New MCP tool that returns the full run hierarchy for a root run_id:

```json
{
  "name": "kit.run_tree",
  "description": "Get the full execution tree for a run (parent + all descendants with status).",
  "inputSchema": {
    "type": "object",
    "properties": {
      "run_id": {
        "type": "string",
        "description": "The root run ID to get the tree for"
      }
    },
    "required": ["run_id"],
    "additionalProperties": false
  }
}
```

**Response shape:**

```json
{
  "root": {
    "run_id": "20260220T023440Z-bd8a4495",
    "kit": "research",
    "phase": "cycle",
    "status": "running",
    "started_at": "2026-02-20T02:34:40Z",
    "finished_at": null,
    "exit_code": null,
    "children": [
      {
        "run_id": "20260220T023510Z-abc12345",
        "kit": "tdd",
        "phase": "full",
        "status": "ok",
        "started_at": "2026-02-20T02:35:10Z",
        "finished_at": "2026-02-20T02:40:00Z",
        "exit_code": 0,
        "children": []
      }
    ]
  }
}
```

**Implementation:**
1. Read the root run's `events.jsonl` for `child_run_spawned` events
2. For each child, read its `events.jsonl` recursively
3. Build the tree. Max depth: 5 (safety limit)
4. Include status derived from events: `running` (has `run_started` but no `run_finished`), `ok` (exit_code 0), `failed` (exit_code != 0)

### R4: `kit.run_events` MCP Tool

New MCP tool for tailing a run's event stream:

```json
{
  "name": "kit.run_events",
  "description": "Tail recent events from a run's event stream. Use for live monitoring.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "run_id": {
        "type": "string",
        "description": "The run ID to tail events for"
      },
      "last_n": {
        "type": "integer",
        "minimum": 1,
        "maximum": 50,
        "description": "Number of recent events to return (default 10)"
      }
    },
    "required": ["run_id"],
    "additionalProperties": false
  }
}
```

**Response:** Array of the last N events from `events.jsonl`, parsed as JSON objects. This is a lightweight alternative to `query_log` — structured data instead of raw text.

### R5: Cherry-pick MCP Fixes

The four bug fixes from `feat/full-year-parquet-export` (described above) must be present on this branch. Either cherry-pick the relevant commits or manually apply the same changes to `orchestration-kit/mcp/server.py`.

### R6: Do NOT Modify Existing Kit Scripts

Do not modify `experiment.sh`, `tdd.sh`, `math.sh`, or any kit-internal scripts. The changes are confined to:
- `orchestration-kit/tools/kit` (R1: emit `child_run_spawned` event)
- `orchestration-kit/mcp/server.py` (R3, R4: new MCP tools + R5 bug fixes)
- `orchestration-kit/tools/dashboard` (R2: parent-child query)
- New test file(s)

---

## Tests

### Event Tests

**T1: child_run_spawned emitted** — Run `tools/kit` with `--parent-run-id`. Verify the parent's `events.jsonl` contains a `child_run_spawned` event with the correct `child_run_id`, `child_kit`, and `child_phase`.

**T2: orphan parent graceful** — Run `tools/kit` with `--parent-run-id nonexistent-id`. Verify no crash and no `child_run_spawned` event (parent events file doesn't exist).

**T3: child event has correct fields** — Parse the `child_run_spawned` event. Verify `ts`, `parent_run_id`, `child_run_id`, `child_kit`, `child_phase` are all present and correctly typed.

### Run Tree Tests

**T4: single run, no children** — Create a run with no children. `kit.run_tree` returns a tree with one node, empty `children` array.

**T5: parent with one child** — Create a parent run, then a child run with `--parent-run-id`. `kit.run_tree` on the parent returns tree with one child node.

**T6: three-level nesting** — Create grandparent → parent → child chain. `kit.run_tree` on the grandparent returns the full 3-level tree.

**T7: max depth respected** — Create a chain deeper than 5. Verify tree is truncated at depth 5.

**T8: status derivation** — Create runs in various states (running, ok, failed). Verify `kit.run_tree` correctly derives status from events.

### Run Events Tests

**T9: tail events** — Create a run with 5 events. `kit.run_events` with `last_n=3` returns the last 3 events in order.

**T10: empty run** — `kit.run_events` on a run with no events returns empty array.

### Dashboard Tests

**T11: parent_run_id filter** — Create parent + 2 children. Query `/api/runs?parent_run_id=X` returns exactly the 2 children.

**T12: combined filters** — Query `/api/runs?parent_run_id=X&status=failed` returns only failed children.

### MCP Integration Tests

**T13: run_tree tool callable** — Call `kit.run_tree` via MCP. Verify response shape matches spec.

**T14: run_events tool callable** — Call `kit.run_events` via MCP. Verify response is array of event objects.

### Regression

**T15: existing MCP tools unaffected** — `kit.status`, `kit.runs`, `kit.capsule`, `kit.research_status` all still work after changes.

**T16: existing `tools/kit` behavior unchanged** — Runs without `--parent-run-id` don't emit `child_run_spawned` events. No behavior change for non-nested runs.

---

## Exit Criteria

- [ ] EC-1: `child_run_spawned` event emitted to parent's events.jsonl when `--parent-run-id` is set
- [ ] EC-2: Dashboard `/api/runs?parent_run_id=X` returns correct children
- [ ] EC-3: `kit.run_tree` MCP tool returns full hierarchy with status
- [ ] EC-4: `kit.run_events` MCP tool returns last N events as structured JSON
- [ ] EC-5: MCP bug fixes (run-id REMAINDER, DEVNULL, reasoning REMAINDER, KIT_STATE_DIR) present
- [ ] EC-6: Max depth limit (5) enforced in run_tree
- [ ] EC-7: Orphan parent_run_id handled gracefully (no crash)
- [ ] EC-8: All new tests pass
- [ ] EC-9: All existing tests/tools unaffected
- [ ] EC-10: No modifications to kit scripts (experiment.sh, tdd.sh, math.sh)

---

## Notes for Implementation

- `events.jsonl` uses file locking (`fcntl.flock`) — the `append_event()` function already handles this. Use it for writing `child_run_spawned`.
- The dashboard is a Python HTTP server in `orchestration-kit/tools/dashboard`. It reads `events.jsonl` files from `runs/` to build its index.
- For `kit.run_tree`, read events.jsonl directly from the filesystem (like `orchestrator_run_info` does) rather than going through the dashboard API. This avoids indexing delays.
- The MCP server runs in stdio mode for Claude Code. New tools need entries in both `TOOL_DEFINITIONS` and the `call_tool` dispatch.
- Test data: create synthetic run directories with events.jsonl files rather than launching real kit phases. This keeps tests fast and deterministic.
