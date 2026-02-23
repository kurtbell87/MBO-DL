"""Tests for experiment-batch-command spec (.kit/docs/experiment-batch-command.md).

Validates that experiment.sh contains the `batch` command:

  1. run_batch() function: existence, structure, parallel launch, wait, failure tracking
  2. Case dispatch: batch) -> shift; run_batch "$@"
  3. Help text: lists the batch command with usage
  4. Dynamic behavior: no-args error, spec validation, parallel execution, exit codes
  5. Syntax validation: bash -n passes
"""

import os
import re
import subprocess
import textwrap
from functools import lru_cache
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_SH = PROJECT_ROOT / ".kit" / "experiment.sh"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _read_experiment_sh() -> str:
    """Read experiment.sh source code (cached — file is static during test run)."""
    return EXPERIMENT_SH.read_text()


def _extract_function_body(source: str, func_name: str) -> str:
    """Extract the body of a bash function from source code.

    Handles nested braces (including ``${var}`` expansions) by counting
    brace depth.  Returns the full function definition including the
    opening and closing braces.
    """
    pattern = rf"^{re.escape(func_name)}\s*\(\)\s*\{{"
    match = re.search(pattern, source, re.MULTILINE)
    if not match:
        return ""

    # Walk forward from the opening brace, tracking depth.
    brace_depth = 0
    for i in range(match.end() - 1, len(source)):
        if source[i] == "{":
            brace_depth += 1
        elif source[i] == "}":
            brace_depth -= 1
            if brace_depth == 0:
                return source[match.start() : i + 1]

    return source[match.start() :]  # fallback: rest of file


def _extract_case_block(source: str) -> str:
    """Extract the main case dispatch block from experiment.sh."""
    match = re.search(r'case "\$\{1:-help\}" in(.*?)esac', source, re.DOTALL)
    return match.group(1) if match else ""


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    return re.sub(r"\033\[[0-9;]*m", "", text)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def run_batch_body():
    """Extract the run_batch() function body from experiment.sh (once per module)."""
    source = _read_experiment_sh()
    body = _extract_function_body(source, "run_batch")
    assert body, "Could not find run_batch() function in experiment.sh"
    return body


@pytest.fixture
def sandbox(tmp_path):
    """Create a sandboxed environment for testing experiment.sh functions.

    Sets up:
      - Minimal directory structure (.kit/, .claude/prompts/)
      - A git repo (experiment.sh calls git rev-parse)
      - Test spec files
      - A call log for tracking stub invocations
    """
    # Directory structure
    (tmp_path / ".kit" / "experiments").mkdir(parents=True)
    (tmp_path / ".kit" / "results").mkdir(parents=True)
    (tmp_path / ".claude" / "prompts").mkdir(parents=True)
    (tmp_path / ".claude" / "hooks").mkdir(parents=True)

    # Minimal prompt files (referenced by experiment.sh)
    for name in ("survey.md", "frame.md", "run.md", "read.md", "synthesize.md"):
        (tmp_path / ".claude" / "prompts" / name).touch()

    # Git repo (experiment.sh reads git rev-parse --show-toplevel)
    subprocess.run(
        ["git", "init", "-q"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path, check=True, capture_output=True,
    )

    # Stub call log
    stub_log = tmp_path / "stub_calls.log"
    stub_log.touch()

    # Create test spec files
    spec_a = tmp_path / ".kit" / "experiments" / "spec-a.md"
    spec_a.write_text("# Spec A\n## Hypothesis\nTest A\n")
    spec_b = tmp_path / ".kit" / "experiments" / "spec-b.md"
    spec_b.write_text("# Spec B\n## Hypothesis\nTest B\n")
    spec_c = tmp_path / ".kit" / "experiments" / "spec-c.md"
    spec_c.write_text("# Spec C\n## Hypothesis\nTest C\n")

    return {
        "root": tmp_path,
        "spec_a": str(spec_a),
        "spec_b": str(spec_b),
        "spec_c": str(spec_c),
        "stub_log": stub_log,
    }


def _make_sandbox_env(sandbox_data):
    """Build a minimal environment dict for running experiment.sh in a sandbox."""
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(sandbox_data["root"]),
        "KIT_STATE_DIR": ".kit",
        "ORCHESTRATION_KIT_DASHBOARD_AUTOSTART": "0",
    }


def _call_run_batch(sandbox_data, spec_args, run_run_exit=0, sync_exit=0):
    """Source experiment.sh, override run_run/sync_results, call run_batch.

    Returns (returncode, stdout, stderr).

    The function overrides run_run() and sync_results() with lightweight stubs
    that log their calls to the stub_log file and return configurable exit codes.
    """
    env = _make_sandbox_env(sandbox_data)

    stub_log = sandbox_data["stub_log"]
    # Clear previous log contents
    stub_log.write_text("")

    specs_str = " ".join(f'"{s}"' for s in spec_args)

    script = textwrap.dedent(f"""\
        cd "{sandbox_data['root']}"
        source "{EXPERIMENT_SH}" > /dev/null 2>&1

        # Override run_run and sync_results with stubs
        run_run() {{
            echo "run_run $1" >> "{stub_log}"
            return {run_run_exit}
        }}
        sync_results() {{
            echo "sync_results $1" >> "{stub_log}"
            return {sync_exit}
        }}

        run_batch {specs_str}
    """)

    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    return result.returncode, result.stdout, result.stderr


def _call_experiment_sh(sandbox_data, args):
    """Run experiment.sh with given arguments in the sandbox.

    Returns (returncode, stdout, stderr).
    """
    env = _make_sandbox_env(sandbox_data)

    result = subprocess.run(
        ["bash", str(EXPERIMENT_SH)] + args,
        capture_output=True,
        text=True,
        env=env,
        cwd=sandbox_data["root"],
        timeout=30,
    )
    return result.returncode, result.stdout, result.stderr


# ═══════════════════════════════════════════════════════════════
# GROUP 1 — run_batch() Function Existence & Structure
# ═══════════════════════════════════════════════════════════════

class TestRunBatchFunctionExists:
    """Verify run_batch() function is defined with correct structure."""

    def test_run_batch_function_defined(self):
        """experiment.sh must define a run_batch() function."""
        source = _read_experiment_sh()
        assert re.search(r"^run_batch\s*\(\)", source, re.MULTILINE), (
            "run_batch() function definition not found in experiment.sh"
        )

    def test_run_batch_validates_argument_count(self, run_batch_body):
        """run_batch() must check for at least one argument."""
        assert "$#" in run_batch_body, (
            "run_batch() does not check argument count ($#)"
        )

    def test_run_batch_exits_nonzero_on_no_args(self, run_batch_body):
        """run_batch() must exit 1 when called with no arguments."""
        assert "exit 1" in run_batch_body, (
            "run_batch() does not exit 1 on missing arguments"
        )

    def test_run_batch_prints_usage_on_no_args(self, run_batch_body):
        """run_batch() must print a usage message referencing experiment.sh batch."""
        assert "Usage" in run_batch_body, (
            "run_batch() does not print usage message on no args"
        )
        assert "batch" in run_batch_body, (
            "run_batch() usage message does not reference 'batch' command"
        )

    def test_run_batch_checks_spec_file_exists(self, run_batch_body):
        """run_batch() must check if each spec file exists with -f."""
        assert re.search(r'-f\s+"?\$', run_batch_body), (
            "run_batch() does not check spec file existence with -f"
        )

    def test_run_batch_calls_run_run(self, run_batch_body):
        """run_batch() must call run_run() for each spec."""
        assert "run_run" in run_batch_body, (
            "run_batch() does not call run_run"
        )

    def test_run_batch_calls_sync_results(self, run_batch_body):
        """run_batch() must call sync_results() for each spec."""
        assert "sync_results" in run_batch_body, (
            "run_batch() does not call sync_results"
        )

    def test_run_batch_uses_background_subshells(self, run_batch_body):
        """run_batch() must launch subshells in the background with &."""
        # Look for ) & pattern indicating background subshell
        assert re.search(r'\)\s*&', run_batch_body), (
            "run_batch() does not use background subshells (missing ) &)"
        )

    def test_run_batch_collects_pids(self, run_batch_body):
        """run_batch() must track PIDs of background jobs via $!."""
        assert "$!" in run_batch_body, (
            "run_batch() does not collect background PIDs ($!)"
        )

    def test_run_batch_waits_for_all_pids(self, run_batch_body):
        """run_batch() must wait for all background PIDs."""
        assert "wait" in run_batch_body, (
            "run_batch() does not wait for background jobs"
        )

    def test_run_batch_tracks_failure_count(self, run_batch_body):
        """run_batch() must track a failure counter."""
        assert "failed" in run_batch_body, (
            "run_batch() does not track failure count"
        )

    def test_run_batch_returns_nonzero_on_failure(self, run_batch_body):
        """run_batch() must return 1 when any spec fails."""
        assert "return 1" in run_batch_body, (
            "run_batch() does not return 1 on failure"
        )

    def test_run_batch_returns_zero_on_success(self, run_batch_body):
        """run_batch() must return 0 when all specs succeed."""
        assert "return 0" in run_batch_body, (
            "run_batch() does not return 0 on success"
        )

    def test_run_batch_prints_batch_mode_banner(self, run_batch_body):
        """run_batch() must print a BATCH MODE header."""
        assert "BATCH MODE" in run_batch_body, (
            "run_batch() does not print BATCH MODE banner"
        )

    def test_run_batch_prints_spec_count_in_banner(self, run_batch_body):
        """run_batch() banner must include the number of specs."""
        # The spec shows: "Parallel RUN+sync for $n specs"
        assert re.search(r'\$n\b', run_batch_body) or "specs" in run_batch_body, (
            "run_batch() banner does not reference spec count"
        )

    def test_run_batch_prints_completion_summary(self, run_batch_body):
        """run_batch() must print a completion summary."""
        assert "Batch complete" in run_batch_body, (
            "run_batch() does not print 'Batch complete' summary"
        )

    def test_run_batch_placed_before_main_section(self):
        """run_batch() must be defined before the '# Main' section."""
        source = _read_experiment_sh()
        main_match = re.search(r"^# Main", source, re.MULTILINE)
        batch_match = re.search(r"^run_batch\s*\(\)", source, re.MULTILINE)

        assert batch_match, "run_batch() not found in experiment.sh"
        assert main_match, "'# Main' section not found in experiment.sh"
        assert batch_match.start() < main_match.start(), (
            "run_batch() must be defined before the '# Main' section"
        )

    def test_run_batch_skips_missing_spec_with_continue(self, run_batch_body):
        """run_batch() must continue (not exit) when a spec file is missing."""
        assert "continue" in run_batch_body, (
            "run_batch() does not use 'continue' to skip missing specs"
        )

    def test_run_batch_run_run_and_sync_in_same_subshell(self, run_batch_body):
        """run_run and sync_results must be in the same subshell (sequential per-spec)."""
        # Match pattern: ( ... run_run ... sync_results ... ) &
        assert re.search(
            r'\(\s*\n\s*run_run.*\n\s*sync_results.*\n\s*\)\s*&',
            run_batch_body,
        ), (
            "run_run and sync_results must be grouped in a single subshell with &"
        )


# ═══════════════════════════════════════════════════════════════
# GROUP 2 — Case Dispatch
# ═══════════════════════════════════════════════════════════════

class TestCaseDispatch:
    """Verify case dispatch includes batch) case."""

    @pytest.fixture(scope="class")
    def batch_case_line(self):
        """Extract the batch) case line from the case dispatch block."""
        source = _read_experiment_sh()
        case_block = _extract_case_block(source)
        for line in case_block.splitlines():
            if "batch)" in line:
                return line
        return None

    def test_batch_case_exists(self, batch_case_line):
        """The case statement must include a batch) case."""
        assert batch_case_line is not None, (
            "batch) case not found in case dispatch"
        )

    def test_batch_case_calls_run_batch(self, batch_case_line):
        """The batch) case must call run_batch."""
        assert batch_case_line is not None, "batch) case line not found"
        assert "run_batch" in batch_case_line, (
            f"batch) case does not call run_batch: {batch_case_line}"
        )

    def test_batch_case_shifts_args(self, batch_case_line):
        """The batch) case must shift before passing args to run_batch."""
        assert batch_case_line is not None, "batch) case line not found"
        assert "shift" in batch_case_line, (
            f"batch) case does not shift args: {batch_case_line}"
        )

    def test_batch_case_passes_remaining_args(self, batch_case_line):
        """The batch) case must pass "$@" to run_batch after shift."""
        assert batch_case_line is not None, "batch) case line not found"
        assert '"$@"' in batch_case_line, (
            f'batch) case does not pass "$@": {batch_case_line}'
        )

    def test_batch_case_between_full_and_status(self):
        """batch) case must appear between full) and status) in case dispatch."""
        source = _read_experiment_sh()
        case_block = _extract_case_block(source)
        lines = case_block.splitlines()

        full_idx = None
        batch_idx = None
        status_idx = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("full)"):
                full_idx = i
            elif stripped.startswith("batch)"):
                batch_idx = i
            elif stripped.startswith("status)"):
                status_idx = i

        assert full_idx is not None, "full) case not found in dispatch"
        assert batch_idx is not None, "batch) case not found in dispatch"
        assert status_idx is not None, "status) case not found in dispatch"
        assert full_idx < batch_idx < status_idx, (
            f"Expected full({full_idx}) < batch({batch_idx}) < status({status_idx})"
        )


# ═══════════════════════════════════════════════════════════════
# GROUP 3 — Help Text
# ═══════════════════════════════════════════════════════════════

class TestHelpText:
    """Verify help output includes the batch command."""

    @pytest.fixture(scope="class")
    def help_text(self):
        """Extract the help text from the help|*) case in experiment.sh."""
        source = _read_experiment_sh()
        # The help block is in the case statement under help|*)
        match = re.search(
            r'help\|\*\)\s*\n(.*?)(?:\n\s*;;\s*\n\s*esac)',
            source,
            re.DOTALL,
        )
        if match:
            return match.group(1)
        return ""

    def test_help_includes_batch_command(self, help_text):
        """Help text must list 'batch' as a command."""
        assert "batch" in help_text, (
            "Help text does not mention 'batch' command"
        )

    def test_help_batch_shows_spec_args(self, help_text):
        """Help text for batch must show spec file arguments."""
        # Look for something like: batch <spec1> <spec2>
        assert re.search(r'batch.*<spec', help_text), (
            "Help text for batch does not show spec arguments"
        )

    def test_help_batch_mentions_parallel(self, help_text):
        """Help text for batch must mention parallel or RUN+sync."""
        batch_line = ""
        for line in help_text.splitlines():
            if "batch" in line.lower():
                batch_line = line
                break
        assert batch_line, "No line in help text mentions batch"
        assert "parallel" in batch_line.lower() or "RUN+sync" in batch_line, (
            f"Help text for batch does not mention parallel execution: {batch_line}"
        )


# ═══════════════════════════════════════════════════════════════
# GROUP 4 — Dynamic Behavior
# ═══════════════════════════════════════════════════════════════

class TestRunBatchBehavior:
    """Test run_batch() function behavior end-to-end via subprocess."""

    def test_batch_no_args_exits_nonzero(self, sandbox):
        """run_batch() with no arguments must exit with non-zero code."""
        rc, stdout, stderr = _call_run_batch(sandbox, [])
        assert rc != 0, (
            f"Expected non-zero exit for no args, got {rc}"
        )

    def test_batch_no_args_prints_usage_to_stderr(self, sandbox):
        """run_batch() with no arguments must print usage to stderr."""
        rc, stdout, stderr = _call_run_batch(sandbox, [])
        clean = _strip_ansi(stderr)
        assert "Usage" in clean or "batch" in clean, (
            f"Expected usage message in stderr, got: {clean}"
        )

    def test_batch_single_valid_spec_calls_run_run(self, sandbox):
        """run_batch() with one valid spec must call run_run for that spec."""
        rc, stdout, stderr = _call_run_batch(sandbox, [sandbox["spec_a"]])
        log = sandbox["stub_log"].read_text()
        assert f"run_run {sandbox['spec_a']}" in log, (
            f"run_run not called for spec_a. Log: {log}"
        )

    def test_batch_single_valid_spec_calls_sync_results(self, sandbox):
        """run_batch() with one valid spec must call sync_results for that spec."""
        rc, stdout, stderr = _call_run_batch(sandbox, [sandbox["spec_a"]])
        log = sandbox["stub_log"].read_text()
        assert f"sync_results {sandbox['spec_a']}" in log, (
            f"sync_results not called for spec_a. Log: {log}"
        )

    def test_batch_multiple_specs_all_launched(self, sandbox):
        """run_batch() with multiple specs must launch run_run for each."""
        specs = [sandbox["spec_a"], sandbox["spec_b"], sandbox["spec_c"]]
        rc, stdout, stderr = _call_run_batch(sandbox, specs)
        log = sandbox["stub_log"].read_text()
        for spec in specs:
            assert f"run_run {spec}" in log, (
                f"run_run not called for {spec}. Log: {log}"
            )

    def test_batch_multiple_specs_all_synced(self, sandbox):
        """run_batch() with multiple specs must call sync_results for each."""
        specs = [sandbox["spec_a"], sandbox["spec_b"], sandbox["spec_c"]]
        rc, stdout, stderr = _call_run_batch(sandbox, specs)
        log = sandbox["stub_log"].read_text()
        for spec in specs:
            assert f"sync_results {spec}" in log, (
                f"sync_results not called for {spec}. Log: {log}"
            )

    def test_batch_nonexistent_spec_prints_error(self, sandbox):
        """run_batch() must print an error for non-existent spec files."""
        rc, stdout, stderr = _call_run_batch(
            sandbox, ["/nonexistent/spec.md"],
        )
        combined = _strip_ansi(stdout + stderr)
        assert "not found" in combined.lower() or "Error" in combined, (
            f"No error message for non-existent spec. Output: {combined}"
        )

    def test_batch_nonexistent_spec_does_not_launch(self, sandbox):
        """run_batch() must not call run_run for non-existent specs."""
        rc, stdout, stderr = _call_run_batch(
            sandbox, ["/nonexistent/spec.md"],
        )
        log = sandbox["stub_log"].read_text()
        assert "run_run" not in log, (
            f"run_run should not be called for non-existent spec. Log: {log}"
        )

    def test_batch_mixed_valid_invalid_runs_valid(self, sandbox):
        """run_batch() with mixed valid/invalid specs must run the valid ones."""
        specs = [sandbox["spec_a"], "/nonexistent/spec.md", sandbox["spec_b"]]
        rc, stdout, stderr = _call_run_batch(sandbox, specs)
        log = sandbox["stub_log"].read_text()
        assert f"run_run {sandbox['spec_a']}" in log, (
            f"run_run not called for valid spec_a. Log: {log}"
        )
        assert f"run_run {sandbox['spec_b']}" in log, (
            f"run_run not called for valid spec_b. Log: {log}"
        )
        assert "run_run /nonexistent" not in log, (
            f"run_run should not be called for nonexistent spec. Log: {log}"
        )

    def test_batch_all_succeed_returns_zero(self, sandbox):
        """run_batch() must return 0 when all specs succeed."""
        specs = [sandbox["spec_a"], sandbox["spec_b"]]
        rc, stdout, stderr = _call_run_batch(sandbox, specs, run_run_exit=0)
        assert rc == 0, (
            f"Expected exit 0 when all succeed, got {rc}. stderr: {stderr}"
        )

    def test_batch_failed_run_returns_nonzero(self, sandbox):
        """run_batch() must return non-zero when run_run fails for any spec."""
        specs = [sandbox["spec_a"], sandbox["spec_b"]]
        rc, stdout, stderr = _call_run_batch(sandbox, specs, run_run_exit=1)
        assert rc != 0, (
            f"Expected non-zero exit when run_run fails, got {rc}"
        )

    def test_batch_prints_ok_for_successful_specs(self, sandbox):
        """run_batch() must print OK for each successful spec."""
        specs = [sandbox["spec_a"]]
        rc, stdout, stderr = _call_run_batch(sandbox, specs, run_run_exit=0)
        clean = _strip_ansi(stdout)
        assert "OK" in clean, (
            f"Expected 'OK' in output for successful spec. Output: {clean}"
        )

    def test_batch_prints_failed_for_failed_specs(self, sandbox):
        """run_batch() must print FAILED for each failed spec."""
        specs = [sandbox["spec_a"]]
        rc, stdout, stderr = _call_run_batch(sandbox, specs, run_run_exit=1)
        clean = _strip_ansi(stdout)
        assert "FAILED" in clean, (
            f"Expected 'FAILED' in output for failed spec. Output: {clean}"
        )

    def test_batch_summary_shows_spec_count(self, sandbox):
        """run_batch() must show total spec count in the summary."""
        specs = [sandbox["spec_a"], sandbox["spec_b"], sandbox["spec_c"]]
        rc, stdout, stderr = _call_run_batch(sandbox, specs)
        clean = _strip_ansi(stdout)
        assert "3 specs" in clean, (
            f"Expected '3 specs' in summary. Output: {clean}"
        )

    def test_batch_summary_shows_failure_count(self, sandbox):
        """run_batch() must show failure count in the summary."""
        specs = [sandbox["spec_a"], sandbox["spec_b"]]
        rc, stdout, stderr = _call_run_batch(sandbox, specs, run_run_exit=0)
        clean = _strip_ansi(stdout)
        assert "0 failure" in clean, (
            f"Expected '0 failure' in summary. Output: {clean}"
        )

    def test_batch_prints_launch_message_per_spec(self, sandbox):
        """run_batch() must print a launch message for each valid spec."""
        specs = [sandbox["spec_a"], sandbox["spec_b"]]
        rc, stdout, stderr = _call_run_batch(sandbox, specs)
        clean = _strip_ansi(stdout)
        assert "Launching" in clean, (
            f"Expected 'Launching' message. Output: {clean}"
        )

    def test_batch_banner_shows_spec_count(self, sandbox):
        """run_batch() banner must display the number of specs being batched."""
        specs = [sandbox["spec_a"], sandbox["spec_b"]]
        rc, stdout, stderr = _call_run_batch(sandbox, specs)
        clean = _strip_ansi(stdout)
        assert "2 specs" in clean, (
            f"Expected '2 specs' in banner. Output: {clean}"
        )


# ═══════════════════════════════════════════════════════════════
# GROUP 5 — Syntax Validation
# ═══════════════════════════════════════════════════════════════

class TestSyntaxValidation:
    """Verify experiment.sh has no syntax errors after modification."""

    def test_experiment_sh_syntax_valid(self):
        """bash -n experiment.sh must pass with no syntax errors."""
        result = subprocess.run(
            ["bash", "-n", str(EXPERIMENT_SH)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, (
            f"Syntax check failed: {result.stderr}"
        )


# ═══════════════════════════════════════════════════════════════
# GROUP 6 — Integration: experiment.sh batch CLI
# ═══════════════════════════════════════════════════════════════

class TestBatchCLI:
    """Test running 'experiment.sh batch' as a command."""

    def test_batch_command_no_args_exits_nonzero(self, sandbox):
        """'experiment.sh batch' with no specs must exit non-zero."""
        rc, stdout, stderr = _call_experiment_sh(sandbox, ["batch"])
        assert rc != 0, (
            f"Expected non-zero exit for 'experiment.sh batch', got {rc}"
        )

    def test_batch_command_no_args_shows_usage(self, sandbox):
        """'experiment.sh batch' with no specs must show usage."""
        rc, stdout, stderr = _call_experiment_sh(sandbox, ["batch"])
        combined = _strip_ansi(stdout + stderr)
        assert "Usage" in combined or "batch" in combined, (
            f"Expected usage in output for 'experiment.sh batch'. Output: {combined}"
        )

    def test_help_command_lists_batch(self, sandbox):
        """'experiment.sh help' must list the batch command."""
        rc, stdout, stderr = _call_experiment_sh(sandbox, ["help"])
        assert "batch" in stdout, (
            f"'batch' not found in help output: {stdout[:500]}"
        )
