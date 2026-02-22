"""Tests for research-cloud-execution spec (.kit/docs/research-cloud-execution.md).

Validates that experiment.sh correctly integrates mandatory cloud execution:

  1. sync_results() function: existence, no-op on local, cloud-run pull,
     artifact-store fallback, metrics.json reporting
  2. COMPUTE_TARGET=ec2 mandatory directive in run_run()
  3. Pipeline ordering: sync_results between RUN and READ in cycle/full/program
  4. Backward compatibility: unchanged behavior when COMPUTE_TARGET=local/unset
  5. Environment configuration: COMPUTE_TARGET=ec2 in .orchestration-kit.env
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
ORCHESTRATION_ENV = PROJECT_ROOT / ".orchestration-kit.env"


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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sandbox(tmp_path):
    """Create a sandboxed environment for testing experiment.sh functions.

    Sets up:
      - Minimal directory structure (.kit/, .claude/prompts/, orchestration-kit/tools/)
      - A git repo (experiment.sh calls git rev-parse)
      - Stub executables for cloud-run and artifact-store that log their calls
      - A test experiment spec
    """
    # Directory structure
    (tmp_path / ".kit" / "experiments").mkdir(parents=True)
    (tmp_path / ".kit" / "results" / "test-spec").mkdir(parents=True)
    (tmp_path / ".claude" / "prompts").mkdir(parents=True)
    (tmp_path / ".claude" / "hooks").mkdir(parents=True)

    # Minimal prompt files (referenced by experiment.sh but unused by sync_results)
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

    # orchestration-kit/tools stubs
    okit_tools = tmp_path / "orchestration-kit" / "tools"
    okit_tools.mkdir(parents=True)

    # cloud-run stub (success by default)
    cloud_run = okit_tools / "cloud-run"
    cloud_run.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        echo "cloud-run $*" >> "{stub_log}"
    """))
    cloud_run.chmod(0o755)

    # artifact-store stub (success by default)
    artifact_store = okit_tools / "artifact-store"
    artifact_store.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        echo "artifact-store $*" >> "{stub_log}"
    """))
    artifact_store.chmod(0o755)

    # Test spec file
    spec_file = tmp_path / ".kit" / "experiments" / "test-spec.md"
    spec_file.write_text("# Test Experiment Spec\n## Hypothesis\nTest\n")

    return {
        "root": tmp_path,
        "spec": str(spec_file),
        "stub_log": stub_log,
        "okit_tools": okit_tools,
    }


def _call_sync_results(sandbox_data, spec_path, env_overrides=None):
    """Source experiment.sh in a sandbox and call sync_results().

    Returns ``(returncode, stdout, stderr)``.

    The helper sources experiment.sh (which prints help to stdout on
    source with no args) then calls sync_results on the given spec.
    Dashboard auto-start and ORCHESTRATION_KIT_ROOT are controlled so
    that no real infrastructure is contacted.
    """
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(sandbox_data["root"]),
        "KIT_STATE_DIR": ".kit",
        "ORCHESTRATION_KIT_ROOT": str(sandbox_data["root"] / "orchestration-kit"),
        "ORCHESTRATION_KIT_DASHBOARD_AUTOSTART": "0",
    }
    if env_overrides:
        env.update(env_overrides)

    # Remove COMPUTE_TARGET if not explicitly provided (test unset case)
    if env_overrides is not None and "COMPUTE_TARGET" not in env_overrides:
        env.pop("COMPUTE_TARGET", None)

    script = textwrap.dedent(f"""\
        cd "{sandbox_data['root']}"
        source "{EXPERIMENT_SH}" > /dev/null 2>&1
        sync_results "{spec_path}"
    """)

    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    return result.returncode, result.stdout, result.stderr


# ═══════════════════════════════════════════════════════════════
# Shared Fixtures for Source Analysis
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def sync_results_body():
    """Extract the sync_results() function body from experiment.sh (once per module)."""
    source = _read_experiment_sh()
    body = _extract_function_body(source, "sync_results")
    assert body, "Could not extract sync_results function body"
    return body


@pytest.fixture(scope="module")
def run_run_body():
    """Extract the run_run() function body from experiment.sh (once per module)."""
    source = _read_experiment_sh()
    body = _extract_function_body(source, "run_run")
    assert body, "Could not extract run_run function body"
    return body


# ═══════════════════════════════════════════════════════════════
# GROUP 1 — sync_results() Function Existence & Structure
# ═══════════════════════════════════════════════════════════════

class TestSyncResultsFunctionExists:
    """Verify sync_results() function is defined in experiment.sh."""

    def test_sync_results_function_defined(self):
        """experiment.sh must define a sync_results() function."""
        source = _read_experiment_sh()
        assert re.search(r"^sync_results\s*\(\)", source, re.MULTILINE), (
            "sync_results() function definition not found in experiment.sh"
        )

    def test_sync_results_checks_compute_target(self, sync_results_body):
        """sync_results() must reference the COMPUTE_TARGET env var."""
        assert "COMPUTE_TARGET" in sync_results_body, (
            "sync_results() does not reference COMPUTE_TARGET"
        )

    def test_sync_results_returns_early_for_non_ec2(self, sync_results_body):
        """sync_results() must return 0 immediately when target is not ec2."""
        assert "return 0" in sync_results_body, (
            "sync_results() has no early return for non-ec2 targets"
        )

    def test_sync_results_references_cloud_run_pull(self, sync_results_body):
        """sync_results() must attempt cloud-run pull for result retrieval."""
        assert "cloud-run" in sync_results_body and "pull" in sync_results_body, (
            "sync_results() does not reference cloud-run pull"
        )

    def test_sync_results_references_artifact_store_hydrate(self, sync_results_body):
        """sync_results() must call artifact-store hydrate as a fallback."""
        assert "artifact-store" in sync_results_body and "hydrate" in sync_results_body, (
            "sync_results() does not reference artifact-store hydrate"
        )

    def test_sync_results_checks_cloud_run_id_file(self, sync_results_body):
        """sync_results() must look for a .cloud-run-id marker file."""
        assert ".cloud-run-id" in sync_results_body, (
            "sync_results() does not reference .cloud-run-id file"
        )

    def test_sync_results_checks_metrics_json(self, sync_results_body):
        """sync_results() must verify metrics.json exists after sync."""
        assert "metrics.json" in sync_results_body, (
            "sync_results() does not verify metrics.json"
        )


# ═══════════════════════════════════════════════════════════════
# GROUP 2 — COMPUTE_TARGET=ec2 Directive in run_run()
# ═══════════════════════════════════════════════════════════════

class TestComputeDirectiveInRunRun:
    """Verify run_run() injects a mandatory EC2 directive when COMPUTE_TARGET=ec2."""

    def test_run_run_contains_compute_target_check(self, run_run_body):
        """run_run() must check the COMPUTE_TARGET env var."""
        assert "COMPUTE_TARGET" in run_run_body, (
            "run_run() does not reference COMPUTE_TARGET"
        )

    def test_directive_contains_mandatory_label(self, run_run_body):
        """The mandatory directive must label itself MANDATORY and EC2."""
        assert "MANDATORY" in run_run_body and "EC2" in run_run_body, (
            "run_run() missing MANDATORY EC2 label in compute directive"
        )

    def test_directive_forbids_detach(self, run_run_body):
        """The directive must explicitly tell the agent not to use --detach."""
        assert "Do NOT use --detach" in run_run_body, (
            "run_run() directive does not forbid --detach"
        )

    def test_directive_requires_cloud_run_id_write(self, run_run_body):
        """The directive must instruct writing the run-id to .cloud-run-id."""
        assert ".cloud-run-id" in run_run_body, (
            "run_run() directive does not instruct writing .cloud-run-id"
        )

    def test_directive_references_cloud_run_command(self, run_run_body):
        """The directive must reference cloud-run run for execution."""
        assert "cloud-run run" in run_run_body or "cloud-run\" run" in run_run_body, (
            "run_run() directive does not reference cloud-run run command"
        )

    def test_directive_requires_metrics_verification(self, run_run_body):
        """The directive must instruct verifying metrics.json exists."""
        assert "metrics.json" in run_run_body, (
            "run_run() directive does not mention metrics.json verification"
        )

    def test_directive_only_triggers_for_ec2(self, run_run_body):
        """The override block must specifically check for ec2 value."""
        assert re.search(r'COMPUTE_TARGET.*==.*"ec2"', run_run_body) or \
               re.search(r'COMPUTE_TARGET.*==.*ec2', run_run_body), (
            "run_run() does not specifically gate the override on ec2"
        )

    def test_compute_target_override_comes_after_preflight(self, run_run_body):
        """The COMPUTE_TARGET override must appear after preflight logic.

        This ensures the mandatory directive replaces (not is overridden by)
        the preflight advisory.
        """
        lines = run_run_body.splitlines()

        # Find last line referencing preflight
        preflight_lines = [
            i for i, line in enumerate(lines) if "preflight" in line.lower()
        ]
        # Find the COMPUTE_TARGET override block
        override_lines = [
            i for i, line in enumerate(lines)
            if "COMPUTE_TARGET" in line and "ec2" in line
        ]

        assert preflight_lines, "No preflight references found in run_run()"
        assert override_lines, "No COMPUTE_TARGET override found in run_run()"

        last_preflight = max(preflight_lines)
        first_override = min(override_lines)
        assert first_override > last_preflight, (
            f"COMPUTE_TARGET override (line {first_override}) must come after "
            f"preflight logic (last at line {last_preflight})"
        )


# ═══════════════════════════════════════════════════════════════
# GROUP 3 — Pipeline Ordering (sync_results between RUN and READ)
# ═══════════════════════════════════════════════════════════════

class TestPipelineOrdering:
    """Verify sync_results() is called between run_run and run_read in composites."""

    @staticmethod
    def _get_call_order(func_name: str) -> list:
        """Extract ordered list of key function calls from a composite."""
        source = _read_experiment_sh()
        body = _extract_function_body(source, func_name)
        calls = re.findall(
            r"\b(run_survey|run_frame|run_run|sync_results|run_read|run_log)\b",
            body,
        )
        return calls

    def test_run_cycle_calls_sync_results(self):
        """run_cycle() must call sync_results."""
        calls = self._get_call_order("run_cycle")
        assert "sync_results" in calls, "run_cycle() does not call sync_results"

    def test_run_cycle_sync_after_run_before_read(self):
        """run_cycle(): sync_results must appear after run_run, before run_read."""
        calls = self._get_call_order("run_cycle")
        run_idx = calls.index("run_run")
        sync_idx = calls.index("sync_results")
        read_idx = calls.index("run_read")
        assert run_idx < sync_idx < read_idx, (
            f"Expected run_run({run_idx}) < sync_results({sync_idx}) < "
            f"run_read({read_idx}) in run_cycle"
        )

    def test_run_full_calls_sync_results(self):
        """run_full() must call sync_results."""
        calls = self._get_call_order("run_full")
        assert "sync_results" in calls, "run_full() does not call sync_results"

    def test_run_full_sync_after_run_before_read(self):
        """run_full(): sync_results must appear after run_run, before run_read."""
        calls = self._get_call_order("run_full")
        run_idx = calls.index("run_run")
        sync_idx = calls.index("sync_results")
        read_idx = calls.index("run_read")
        assert run_idx < sync_idx < read_idx, (
            f"Expected run_run({run_idx}) < sync_results({sync_idx}) < "
            f"run_read({read_idx}) in run_full"
        )

    def test_run_program_calls_sync_results_in_subshell(self):
        """run_program() subshell must call sync_results between run_run and run_read."""
        source = _read_experiment_sh()
        body = _extract_function_body(source, "run_program")

        # The subshell block is: ( run_frame ... run_run ... run_read ) || subshell_exit
        subshell_match = re.search(
            r"\(\s*\n(.*?)\)\s*\|\|\s*subshell_exit",
            body,
            re.DOTALL,
        )
        assert subshell_match, (
            "run_program() does not contain expected subshell block"
        )
        subshell_body = subshell_match.group(1)

        calls = re.findall(
            r"\b(run_frame|run_run|sync_results|run_read)\b",
            subshell_body,
        )
        assert "sync_results" in calls, (
            "sync_results not called in run_program subshell"
        )
        run_idx = calls.index("run_run")
        sync_idx = calls.index("sync_results")
        read_idx = calls.index("run_read")
        assert run_idx < sync_idx < read_idx, (
            f"Expected run_run({run_idx}) < sync_results({sync_idx}) < "
            f"run_read({read_idx}) in run_program subshell"
        )


# ═══════════════════════════════════════════════════════════════
# GROUP 4 — sync_results() Dynamic Behavior
# ═══════════════════════════════════════════════════════════════

class TestSyncResultsBehavior:
    """Test sync_results() function behavior end-to-end via subprocess."""

    def test_noop_when_compute_target_local(self, sandbox):
        """sync_results() must be a no-op when COMPUTE_TARGET=local."""
        rc, stdout, _ = _call_sync_results(
            sandbox, sandbox["spec"],
            env_overrides={"COMPUTE_TARGET": "local"},
        )
        assert rc == 0, f"Expected exit 0, got {rc}"
        log = sandbox["stub_log"].read_text()
        assert log.strip() == "", (
            f"Expected no stub calls for COMPUTE_TARGET=local, got: {log}"
        )

    def test_noop_when_compute_target_unset(self, sandbox):
        """sync_results() must be a no-op when COMPUTE_TARGET is not set."""
        rc, stdout, _ = _call_sync_results(
            sandbox, sandbox["spec"],
            env_overrides={},  # explicitly no COMPUTE_TARGET
        )
        assert rc == 0, f"Expected exit 0, got {rc}"
        log = sandbox["stub_log"].read_text()
        assert log.strip() == "", (
            f"Expected no stub calls when COMPUTE_TARGET unset, got: {log}"
        )

    def test_calls_artifact_store_hydrate_on_ec2(self, sandbox):
        """sync_results() must call artifact-store hydrate when COMPUTE_TARGET=ec2."""
        rc, stdout, stderr = _call_sync_results(
            sandbox, sandbox["spec"],
            env_overrides={"COMPUTE_TARGET": "ec2"},
        )
        assert rc == 0, f"Expected exit 0, got {rc}. stderr: {stderr}"
        log = sandbox["stub_log"].read_text()
        assert "artifact-store hydrate" in log, (
            f"Expected artifact-store hydrate call, got: {log}"
        )

    def test_calls_cloud_run_pull_when_run_id_exists(self, sandbox):
        """sync_results() must call cloud-run pull when .cloud-run-id file exists."""
        results_dir = sandbox["root"] / ".kit" / "results" / "test-spec"
        (results_dir / ".cloud-run-id").write_text("test-run-12345")

        rc, stdout, stderr = _call_sync_results(
            sandbox, sandbox["spec"],
            env_overrides={"COMPUTE_TARGET": "ec2"},
        )
        assert rc == 0, f"Expected exit 0, got {rc}. stderr: {stderr}"
        log = sandbox["stub_log"].read_text()
        assert "cloud-run pull test-run-12345" in log, (
            f"Expected cloud-run pull with run-id, got: {log}"
        )

    def test_does_not_call_cloud_run_without_run_id(self, sandbox):
        """sync_results() must skip cloud-run pull when no .cloud-run-id exists."""
        rc, stdout, stderr = _call_sync_results(
            sandbox, sandbox["spec"],
            env_overrides={"COMPUTE_TARGET": "ec2"},
        )
        assert rc == 0, f"Expected exit 0, got {rc}. stderr: {stderr}"
        log = sandbox["stub_log"].read_text()
        assert "cloud-run pull" not in log, (
            f"Should not call cloud-run pull without .cloud-run-id, got: {log}"
        )

    def test_reports_success_when_metrics_json_exists(self, sandbox):
        """sync_results() must print a success message when metrics.json is found."""
        results_dir = sandbox["root"] / ".kit" / "results" / "test-spec"
        (results_dir / "metrics.json").write_text('{"test": true}')

        rc, stdout, _ = _call_sync_results(
            sandbox, sandbox["spec"],
            env_overrides={"COMPUTE_TARGET": "ec2"},
        )
        assert rc == 0
        # ANSI color codes may be in the output, strip them for matching
        clean = re.sub(r"\033\[[0-9;]*m", "", stdout)
        assert "metrics.json exists" in clean or "Results synced" in clean, (
            f"Expected success message about metrics.json, got: {clean}"
        )

    def test_warns_when_metrics_json_missing(self, sandbox):
        """sync_results() must print a warning when metrics.json is not found."""
        rc, stdout, _ = _call_sync_results(
            sandbox, sandbox["spec"],
            env_overrides={"COMPUTE_TARGET": "ec2"},
        )
        assert rc == 0, "sync_results should not fail on missing metrics.json"
        clean = re.sub(r"\033\[[0-9;]*m", "", stdout)
        assert "Warning" in clean or "metrics.json not found" in clean, (
            f"Expected warning about missing metrics.json, got: {clean}"
        )

    def test_continues_when_cloud_run_pull_fails(self, sandbox):
        """sync_results() must fall back to artifact-store when cloud-run pull fails."""
        # Make cloud-run stub exit non-zero
        stub_log = sandbox["stub_log"]
        cloud_run = sandbox["okit_tools"] / "cloud-run"
        cloud_run.write_text(textwrap.dedent(f"""\
            #!/usr/bin/env bash
            echo "cloud-run $*" >> "{stub_log}"
            exit 1
        """))
        cloud_run.chmod(0o755)

        # Create .cloud-run-id so cloud-run pull is attempted
        results_dir = sandbox["root"] / ".kit" / "results" / "test-spec"
        (results_dir / ".cloud-run-id").write_text("failing-run-id")

        rc, stdout, stderr = _call_sync_results(
            sandbox, sandbox["spec"],
            env_overrides={"COMPUTE_TARGET": "ec2"},
        )
        assert rc == 0, (
            f"sync_results must not fail when cloud-run pull fails. "
            f"rc={rc}, stderr: {stderr}"
        )
        log = stub_log.read_text()
        # cloud-run pull should have been attempted
        assert "cloud-run pull failing-run-id" in log, (
            f"cloud-run pull was not attempted. Log: {log}"
        )
        # artifact-store hydrate should have been called as fallback
        assert "artifact-store hydrate" in log, (
            f"artifact-store hydrate fallback not called. Log: {log}"
        )

    def test_cloud_run_pull_receives_correct_output_dir(self, sandbox):
        """cloud-run pull must receive the results directory as --output-dir."""
        results_dir = sandbox["root"] / ".kit" / "results" / "test-spec"
        (results_dir / ".cloud-run-id").write_text("run-42")

        _call_sync_results(
            sandbox, sandbox["spec"],
            env_overrides={"COMPUTE_TARGET": "ec2"},
        )
        log = sandbox["stub_log"].read_text()
        # sync_results uses results_dir_for_spec() which returns a relative
        # path ($RESULTS_DIR/$exp_id).  Check that the --output-dir flag
        # points to the correct results subdirectory.
        assert "--output-dir" in log, (
            f"cloud-run pull missing --output-dir flag. Log: {log}"
        )
        assert "test-spec" in log.split("--output-dir")[1], (
            f"cloud-run pull --output-dir does not target test-spec results. Log: {log}"
        )


# ═══════════════════════════════════════════════════════════════
# GROUP 5 — Backward Compatibility
# ═══════════════════════════════════════════════════════════════

class TestBackwardCompatibility:
    """Verify behavior is unchanged when COMPUTE_TARGET is local or unset."""

    @pytest.fixture()
    def help_output(self, sandbox):
        """Run 'experiment.sh help' in the sandbox and return stdout."""
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(sandbox["root"]),
            "KIT_STATE_DIR": ".kit",
            "ORCHESTRATION_KIT_DASHBOARD_AUTOSTART": "0",
        }
        result = subprocess.run(
            ["bash", str(EXPERIMENT_SH), "help"],
            capture_output=True, text=True, env=env,
            cwd=sandbox["root"], timeout=30,
        )
        return result.stdout

    def test_help_output_does_not_mention_sync_results(self, help_output):
        """experiment.sh help must not expose sync_results as a user command."""
        assert "sync_results" not in help_output, (
            "sync_results should not appear in help output (internal function)"
        )

    def test_help_output_does_not_mention_compute_target(self, help_output):
        """experiment.sh help must not reference COMPUTE_TARGET."""
        assert "COMPUTE_TARGET" not in help_output, (
            "COMPUTE_TARGET should not appear in help output"
        )

    def test_help_lists_all_standard_phases(self, help_output):
        """experiment.sh help must list all standard user-facing phases."""
        expected_phases = [
            "survey", "frame", "run", "read", "log",
            "cycle", "full", "status", "program", "synthesize", "watch",
        ]
        for phase in expected_phases:
            assert phase in help_output, (
                f"Standard phase '{phase}' missing from help output"
            )

    def test_help_does_not_add_new_commands(self, help_output):
        """experiment.sh help must not list any new user-facing commands.

        The spec explicitly requires: 'experiment.sh help output unchanged
        (no new user-facing commands)'.
        """
        phase_lines = re.findall(
            r"^\s{2}(\w[\w-]*)\s",
            help_output,
            re.MULTILINE,
        )
        known_phases = {
            "survey", "frame", "run", "read", "log", "cycle", "full",
            "status", "program", "synthesize", "watch",
        }
        # Also allow "complete-handoff" and "validate-handoff" (pre-existing)
        known_phases.update({"complete-handoff", "validate-handoff"})
        unknown = set(phase_lines) - known_phases
        assert not unknown, (
            f"Unexpected new commands in help output: {unknown}"
        )


# ═══════════════════════════════════════════════════════════════
# GROUP 6 — Environment Configuration
# ═══════════════════════════════════════════════════════════════

class TestEnvironmentConfig:
    """Verify .orchestration-kit.env contains COMPUTE_TARGET=ec2."""

    def test_compute_target_ec2_in_env_file(self):
        """COMPUTE_TARGET=ec2 must be exported in .orchestration-kit.env."""
        content = ORCHESTRATION_ENV.read_text()
        assert re.search(
            r'^\s*export\s+COMPUTE_TARGET\s*=\s*["\']?ec2["\']?\s*$',
            content,
            re.MULTILINE,
        ), "COMPUTE_TARGET=ec2 not found as export in .orchestration-kit.env"

    def test_compute_target_is_not_commented_out(self):
        """The COMPUTE_TARGET=ec2 export must not be commented out."""
        content = ORCHESTRATION_ENV.read_text()
        for line in content.splitlines():
            stripped = line.strip()
            if "COMPUTE_TARGET" in stripped and "ec2" in stripped:
                if stripped.startswith("#"):
                    pytest.fail(
                        f"COMPUTE_TARGET=ec2 is commented out: {stripped}"
                    )
                return  # found an active line — pass
        pytest.fail("No active COMPUTE_TARGET=ec2 line in .orchestration-kit.env")

    def test_compute_target_default_is_local(self, sync_results_body):
        """experiment.sh must default COMPUTE_TARGET to 'local' when unset.

        This ensures backward compatibility: projects that don't set the
        variable in their env file get local behavior.
        """
        assert re.search(r'COMPUTE_TARGET:-local', sync_results_body), (
            "sync_results() does not default COMPUTE_TARGET to 'local'"
        )
