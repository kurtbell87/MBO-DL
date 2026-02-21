# TDD Spec: Research Kit Cloud Execution Integration

**Date:** 2026-02-21
**Priority:** P0 — blocks all future experiments
**Scope:** `experiment.sh` (orchestration-kit research kit)

---

## Summary

Modify `experiment.sh` so that when `COMPUTE_TARGET=ec2` is set in the environment, the RUN phase **mandates** cloud execution via `orchestration-kit/tools/cloud-run`, and a results-sync step is automatically inserted between RUN and READ in all composite commands (`cycle`, `full`, `program`).

Currently:
- The preflight system injects an advisory into the RUN sub-agent prompt
- The sub-agent may or may not use `cloud-run`
- `run_cycle()`, `run_full()`, and `run_program()` call `run_read()` immediately after `run_run()` — no results sync
- If the RUN sub-agent uses `cloud-run --detach`, results aren't local when READ starts

After this change:
- When `COMPUTE_TARGET=ec2`: the RUN sub-agent prompt **mandates** cloud-run usage (not advisory)
- A `sync_results()` function pulls results from cloud after RUN completes
- `cycle`, `full`, and `program` call `sync_results()` between RUN and READ
- When `COMPUTE_TARGET` is unset or `local`: behavior is unchanged (backward compatible)

---

## Environment Variable

```bash
# In .orchestration-kit.env:
COMPUTE_TARGET="${COMPUTE_TARGET:-local}"   # "local" or "ec2"
```

When `COMPUTE_TARGET=ec2`:
- The RUN phase compute advisory is replaced with a **mandatory directive**
- The sub-agent MUST use `cloud-run run` (not `--detach`) to execute the training script
- The sub-agent should wait for cloud-run to complete and pull results
- After `run_run()` exits, `sync_results()` ensures all result files are local

---

## Changes Required

### 1. New `sync_results()` function

Add after `_phase_summary()` (around line 203):

```bash
sync_results() {
  # Pull results from cloud/S3 after a remote RUN phase.
  # No-op if COMPUTE_TARGET is not ec2.
  local spec_file="$1"
  local results_path
  results_path="$(results_dir_for_spec "$spec_file")"

  if [[ "${COMPUTE_TARGET:-local}" != "ec2" ]]; then
    return 0
  fi

  echo -e "${CYAN}── Syncing results from cloud... ──${NC}"

  # Try cloud-run pull first (if a run-id marker exists)
  local cloud_run_id_file="$results_path/.cloud-run-id"
  if [[ -f "$cloud_run_id_file" ]]; then
    local cloud_run_id
    cloud_run_id=$(cat "$cloud_run_id_file")
    echo -e "  Pulling results for cloud-run: $cloud_run_id"
    local _okit="${ORCHESTRATION_KIT_ROOT:-orchestration-kit}"
    "$_okit/tools/cloud-run" pull "$cloud_run_id" --output-dir "$results_path" || {
      echo -e "${YELLOW}  cloud-run pull failed, trying artifact-store hydrate...${NC}"
    }
  fi

  # Fallback: hydrate any S3 artifact symlinks
  local _okit="${ORCHESTRATION_KIT_ROOT:-orchestration-kit}"
  if [[ -x "$_okit/tools/artifact-store" ]]; then
    "$_okit/tools/artifact-store" hydrate 2>/dev/null || true
  fi

  # Verify results exist
  if [[ -f "$results_path/metrics.json" ]]; then
    echo -e "  ${GREEN}Results synced:${NC} $results_path/metrics.json exists"
  else
    echo -e "  ${YELLOW}Warning: metrics.json not found in $results_path after sync${NC}"
  fi
}
```

### 2. Modify `run_run()` — mandatory cloud directive

In `run_run()` (line 436), modify the compute advisory logic. When `COMPUTE_TARGET=ec2`, override the advisory regardless of preflight recommendation:

After the existing preflight logic (around line 535), add:

```bash
# COMPUTE_TARGET override: if ec2 is mandatory, replace advisory
if [[ "${COMPUTE_TARGET:-local}" == "ec2" ]]; then
  compute_advisory="
## Compute Directive (MANDATORY — EC2)
ALL training and heavy computation MUST run on EC2. Do NOT run model training locally.

Use cloud-run to execute the experiment:
  ${_okit:-orchestration-kit}/tools/cloud-run run \"python <your-script>\" \\
      --spec $spec_file \\
      --data-dirs ${DATA_DIR:-data}/ \\
      --output-dir $results_path/ \\
      --max-hours ${MAX_GPU_HOURS:-4}

IMPORTANT:
- Do NOT use --detach. Wait for the run to complete.
- After cloud-run finishes, pull results:
    ${_okit:-orchestration-kit}/tools/cloud-run pull <run-id> --output-dir $results_path/
- Write the cloud-run run-id to $results_path/.cloud-run-id
- Verify metrics.json exists in $results_path/ before exiting.
- You may run the MVE (minimal viable experiment) locally for fast iteration,
  but the FULL experiment (all CPCV splits, all configs) MUST run on EC2.
- Local-only tasks (data loading verification, normalization checks, small sanity checks) are fine locally."
fi
```

### 3. Modify `run_cycle()` — add sync step

```bash
run_cycle() {
  local spec_file="${1:?Usage: experiment.sh cycle <spec-file>}"

  echo -e "${BOLD}Running experiment cycle: FRAME -> RUN -> READ -> LOG${NC}"
  echo ""

  run_frame "$spec_file"
  echo -e "\n${YELLOW}--- Frame complete. Running experiment... ---${NC}\n"

  run_run "$spec_file"
  sync_results "$spec_file"   # <-- NEW
  echo -e "\n${YELLOW}--- Run complete. Analyzing results... ---${NC}\n"

  run_read "$spec_file"
  echo -e "\n${YELLOW}--- Analysis complete. Logging... ---${NC}\n"

  run_log "$spec_file"

  echo ""
  echo -e "${BOLD}${GREEN}Experiment cycle complete.${NC}"
}
```

### 4. Modify `run_full()` — add sync step

```bash
run_full() {
  local question="${1:?Usage: experiment.sh full <question> <spec-file>}"
  local spec_file="${2:?Usage: experiment.sh full <question> <spec-file>}"

  echo -e "${BOLD}Running full research cycle: SURVEY -> FRAME -> RUN -> READ -> LOG${NC}"
  echo ""

  run_survey "$question"
  echo -e "\n${YELLOW}--- Survey complete. Designing experiment... ---${NC}\n"

  run_frame "$spec_file"
  echo -e "\n${YELLOW}--- Frame complete. Running experiment... ---${NC}\n"

  run_run "$spec_file"
  sync_results "$spec_file"   # <-- NEW
  echo -e "\n${YELLOW}--- Run complete. Analyzing results... ---${NC}\n"

  run_read "$spec_file"
  echo -e "\n${YELLOW}--- Analysis complete. Logging... ---${NC}\n"

  run_log "$spec_file"

  echo ""
  echo -e "${BOLD}${GREEN}Full research cycle complete.${NC}"
}
```

### 5. Modify `run_program()` subshell — add sync step

In the subshell block (around line 1188):

```bash
    (
      run_frame "$spec_file"
      echo -e "\n${YELLOW}--- Frame complete. Running experiment... ---${NC}\n"
      run_run "$spec_file"
      sync_results "$spec_file"   # <-- NEW
      echo -e "\n${YELLOW}--- Run complete. Analyzing results... ---${NC}\n"
      run_read "$spec_file"
    ) || subshell_exit=$?
```

### 6. Update `.orchestration-kit.env`

Add `COMPUTE_TARGET=ec2` to the project's `.orchestration-kit.env` so it's the default for this project.

---

## Exit Criteria

- [ ] `sync_results()` function added to `experiment.sh`
- [ ] `run_run()` injects mandatory cloud directive when `COMPUTE_TARGET=ec2`
- [ ] `run_cycle()` calls `sync_results()` between RUN and READ
- [ ] `run_full()` calls `sync_results()` between RUN and READ
- [ ] `run_program()` calls `sync_results()` between RUN and READ in subshell
- [ ] `COMPUTE_TARGET=ec2` added to `.orchestration-kit.env`
- [ ] When `COMPUTE_TARGET=local` (or unset), behavior is identical to current (backward compatible)
- [ ] `experiment.sh help` output unchanged (no new user-facing commands)

## Non-Goals

- Do NOT modify `cloud-run` itself
- Do NOT modify sub-agent prompts (survey.md, frame.md, run.md, read.md) — the advisory injection in experiment.sh is sufficient
- Do NOT add new CLI flags to experiment.sh — this is env-var driven
- Do NOT change the preflight tool
