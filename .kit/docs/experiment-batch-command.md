# Add `batch` Command to experiment.sh

## Overview

The parallel batch dispatch TDD cycle implemented `cloud-run batch`, `batch.py`, MCP `kit.research_batch`, and all supporting changes. One file was missed: `orchestration-kit/research-kit/experiment.sh` needs a `batch` command for local-parallel RUN+sync dispatch.

## IMPORTANT: Language & Framework

- **Language**: Bash
- **Test**: `bash -n orchestration-kit/research-kit/experiment.sh` (syntax check)
- **NO C++ or CMake.** This is a Bash script modification only.

## Existing Code

Read `orchestration-kit/research-kit/experiment.sh` first. It has:
- Functions: `run_run()`, `sync_results()`, `run_cycle()`, `run_full()`, `run_program()`, etc.
- Case dispatch at the bottom: `case "${1:-help}" in survey) ... full) ... status) ... program) ... esac`
- Color variables: `$RED`, `$GREEN`, `$CYAN`, `$BOLD`, `$NC`, `$YELLOW`

## Requirements

### New function: `run_batch()`

Add this function BEFORE the `# Main` comment section (before the `ensure_dashboard_watchdog` line).

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

### Update case dispatch

In the `case "${1:-help}" in` block at the bottom of the file, add this line between the existing `full)` and `status)` cases:

```bash
  batch)      shift; run_batch "$@" ;;
```

### Update help text

In the `help|*)` case, add this line to the Phases section (after the `full` line):

```
    echo "  batch     <spec1> <spec2> ...   Run RUN+sync in parallel for multiple specs"
```

## Exit Criteria

- [ ] `orchestration-kit/research-kit/experiment.sh` contains `run_batch()` function
- [ ] `orchestration-kit/research-kit/experiment.sh` case dispatch includes `batch)` case
- [ ] `bash -n orchestration-kit/research-kit/experiment.sh` passes (no syntax errors)
- [ ] Help text includes `batch` command
