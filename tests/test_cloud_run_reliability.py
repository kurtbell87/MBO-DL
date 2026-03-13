"""Tests for cloud-run-reliability spec (.kit/docs/cloud-run-reliability.md).

Validates the cloud-run reliability overhaul:

  1. Bootstrap sync daemon in ec2-bootstrap-gpu.sh and ec2-bootstrap.sh
     (heartbeat every 60s, log every 60s, results every 5 min)
  2. S3 check_heartbeat() and tail_log() functions in s3.py
  3. Enhanced poll_status() with instance health + heartbeat in remote.py
  4. gc_stale_runs() in remote.py
  5. CLI `logs` subcommand with --lines and --follow
  6. Enhanced status display (elapsed, cost, heartbeat, log lines)
  7. Pre-flight code validation (validate.py)
  8. CLI --validate and --skip-smoke flags
  9. experiment.sh compute directive includes --validate
  10. gc_stale() in state.py
  11. cloud-run gc calls both gc_stale_runs() and gc_stale()
"""

import ast
import importlib
import inspect
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from io import StringIO
from pathlib import Path
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLOUD_DIR = PROJECT_ROOT / "orchestration-kit" / "tools" / "cloud"
CLOUD_RUN_CLI = PROJECT_ROOT / "orchestration-kit" / "tools" / "cloud-run"
BOOTSTRAP_GPU = PROJECT_ROOT / "orchestration-kit" / "tools" / "cloud" / "scripts" / "ec2-bootstrap-gpu.sh"
BOOTSTRAP_CPU = PROJECT_ROOT / "orchestration-kit" / "tools" / "cloud" / "scripts" / "ec2-bootstrap.sh"
EXPERIMENT_SH = PROJECT_ROOT / ".kit" / "experiment.sh"

# Ensure cloud module is importable
sys.path.insert(0, str(PROJECT_ROOT / "orchestration-kit" / "tools"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _read_script(path: Path) -> str:
    """Read a script file (cached)."""
    return path.read_text()


@lru_cache(maxsize=1)
def _read_cloud_run_cli() -> str:
    """Read the cloud-run CLI source code (cached)."""
    return CLOUD_RUN_CLI.read_text()


def _extract_function_body(source: str, func_name: str) -> str:
    """Extract the body of a bash function from source code."""
    pattern = rf"^{re.escape(func_name)}\s*\(\)\s*\{{"
    match = re.search(pattern, source, re.MULTILINE)
    if not match:
        return ""
    brace_depth = 0
    for i in range(match.end() - 1, len(source)):
        if source[i] == "{":
            brace_depth += 1
        elif source[i] == "}":
            brace_depth -= 1
            if brace_depth == 0:
                return source[match.start() : i + 1]
    return source[match.start() :]


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 1 — Bootstrap Sync Daemon (ec2-bootstrap-gpu.sh)
# ═══════════════════════════════════════════════════════════════════════════

class TestBootstrapGPUSyncDaemon:
    """Verify ec2-bootstrap-gpu.sh has a background sync daemon for heartbeat,
    log streaming, and incremental result uploads."""

    @pytest.fixture(autouse=True)
    def _load_script(self):
        self.source = _read_script(BOOTSTRAP_GPU)

    def test_sync_daemon_pid_initialized_before_trap(self):
        """SYNC_DAEMON_PID must be initialized as empty before the trap is set.

        This prevents the trap from referencing an undefined variable if it fires
        before the daemon starts.
        """
        lines = self.source.splitlines()
        pid_init_line = None
        trap_line = None
        for i, line in enumerate(lines):
            if re.search(r'SYNC_DAEMON_PID\s*=\s*""', line):
                pid_init_line = i
            if "trap" in line and "EXIT" in line:
                if trap_line is None:
                    trap_line = i
        assert pid_init_line is not None, (
            "SYNC_DAEMON_PID=\"\" initialization not found in ec2-bootstrap-gpu.sh"
        )
        assert trap_line is not None, "EXIT trap not found in ec2-bootstrap-gpu.sh"
        assert pid_init_line < trap_line, (
            f"SYNC_DAEMON_PID init (line {pid_init_line}) must come before "
            f"EXIT trap (line {trap_line})"
        )

    def test_sync_daemon_background_loop_exists(self):
        """A background subshell daemon must exist with 'sleep 60' heartbeat interval."""
        # The daemon is a background subshell: ( ... ) &
        assert re.search(r"\(\s*\n.*?sleep\s+60.*?\)\s*&", self.source, re.DOTALL), (
            "No background sync daemon loop with 'sleep 60' found in ec2-bootstrap-gpu.sh"
        )

    def test_sync_daemon_writes_heartbeat_to_s3(self):
        """The sync daemon must write a UTC timestamp to S3 as heartbeat."""
        assert "heartbeat" in self.source, (
            "No 'heartbeat' reference found in ec2-bootstrap-gpu.sh"
        )
        # Should write ISO-8601 timestamp
        assert re.search(r"date\s+-u\s+\+%Y-%m-%dT%H:%M:%SZ", self.source), (
            "No UTC date formatting for heartbeat found"
        )
        # Must upload to S3 heartbeat path
        assert re.search(r'aws\s+s3\s+cp\s+.*heartbeat', self.source), (
            "No S3 upload of heartbeat file found"
        )

    def test_sync_daemon_uploads_log_every_60s(self):
        """The sync daemon must upload the log file to S3 every 60s."""
        # Inside the daemon loop, should have: aws s3 cp "$LOGFILE" "...experiment.log"
        daemon_match = re.search(
            r"\(\s*\n(.*?)\)\s*&\s*\nSYNC_DAEMON_PID",
            self.source, re.DOTALL,
        )
        assert daemon_match, "Cannot locate sync daemon body"
        daemon_body = daemon_match.group(1)

        assert re.search(r'aws\s+s3\s+cp\s+.*LOGFILE.*experiment\.log', daemon_body), (
            "Sync daemon does not upload LOGFILE as experiment.log"
        )

    def test_sync_daemon_syncs_results_every_5_minutes(self):
        """The sync daemon must sync /work/results/ to S3 every 5 minutes."""
        daemon_match = re.search(
            r"\(\s*\n(.*?)\)\s*&\s*\nSYNC_DAEMON_PID",
            self.source, re.DOTALL,
        )
        assert daemon_match, "Cannot locate sync daemon body"
        daemon_body = daemon_match.group(1)

        # Check for modulo 5 logic
        assert re.search(r'_sync_counter\s*%\s*5', daemon_body) or \
               re.search(r'sync_counter.*5', daemon_body), (
            "Sync daemon does not have 5-minute result sync interval"
        )
        # Must sync /work/results/
        assert "aws s3 sync /work/results/" in daemon_body or \
               re.search(r'aws\s+s3\s+sync\s+/work/results/', daemon_body), (
            "Sync daemon does not sync /work/results/ to S3"
        )

    def test_sync_daemon_pid_captured(self):
        """The daemon's PID must be captured into SYNC_DAEMON_PID."""
        assert re.search(r'SYNC_DAEMON_PID=\$!', self.source), (
            "SYNC_DAEMON_PID=$! not found — daemon PID not captured"
        )

    def test_sync_daemon_killed_after_experiment(self):
        """The sync daemon must be killed after the experiment execution completes."""
        # After experiment finishes (after eval/run section), should kill SYNC_DAEMON_PID
        experiment_line = None
        kill_daemon_line = None
        lines = self.source.splitlines()
        for i, line in enumerate(lines):
            if "EXPERIMENT_COMMAND" in line and ("eval" in line or "bash" in line):
                experiment_line = i
            if experiment_line and re.search(r'kill.*SYNC_DAEMON_PID', line):
                kill_daemon_line = i
                break
        assert kill_daemon_line is not None, (
            "SYNC_DAEMON_PID is not killed after experiment execution"
        )
        assert kill_daemon_line > experiment_line, (
            "Daemon kill must come after experiment execution"
        )

    def test_trap_kills_sync_daemon(self):
        """The EXIT trap / cleanup function must kill the sync daemon."""
        # The trap or cleanup function must reference killing SYNC_DAEMON_PID
        # Find the cleanup/trap block
        trap_block = ""
        if "cleanup()" in self.source:
            trap_block = _extract_function_body(self.source, "cleanup")
        else:
            # Inline trap — look at the trap line
            trap_match = re.search(r"trap\s+'(.*?)'\s+EXIT", self.source)
            if trap_match:
                trap_block = trap_match.group(1)

        assert "SYNC_DAEMON_PID" in trap_block or "SYNC_DAEMON_PID" in self.source.split("cleanup")[0] if "cleanup" in self.source else True, (
            "EXIT trap does not reference SYNC_DAEMON_PID"
        )
        # More robust: look for kill $SYNC_DAEMON_PID in the cleanup/trap area
        assert re.search(r'kill.*SYNC_DAEMON_PID', self.source), (
            "No kill of SYNC_DAEMON_PID found in script"
        )

    def test_trap_syncs_results_before_exit_code(self):
        """The EXIT trap must sync results to S3 BEFORE writing exit_code.

        This ensures partial results are available even on crash.
        """
        # Find the cleanup/trap section
        # Either a cleanup() function or inline trap
        cleanup_source = ""
        if "cleanup()" in self.source or "cleanup ()" in self.source:
            cleanup_source = _extract_function_body(self.source, "cleanup")
        else:
            # Look for the EXIT trap content
            trap_match = re.search(
                r"trap\s+['\"](.+?)['\"].*EXIT",
                self.source, re.DOTALL,
            )
            if trap_match:
                cleanup_source = trap_match.group(1)

        # Results sync should come before exit_code write
        lines = cleanup_source.splitlines() if cleanup_source else self.source.splitlines()
        results_sync_line = None
        exit_code_write_line = None
        for i, line in enumerate(lines):
            if re.search(r'aws\s+s3\s+sync.*/work/results/', line):
                if results_sync_line is None:
                    results_sync_line = i
            if re.search(r'exit_code', line) and re.search(r'aws\s+s3\s+cp', line):
                exit_code_write_line = i

        assert results_sync_line is not None, (
            "No results sync found in trap/cleanup"
        )
        assert exit_code_write_line is not None, (
            "No exit_code write found in trap/cleanup"
        )
        assert results_sync_line < exit_code_write_line, (
            f"Results sync (line {results_sync_line}) must come before "
            f"exit_code write (line {exit_code_write_line}) in trap/cleanup"
        )


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 2 — Bootstrap Sync Daemon (ec2-bootstrap.sh, non-GPU)
# ═══════════════════════════════════════════════════════════════════════════

class TestBootstrapCPUSyncDaemon:
    """Verify ec2-bootstrap.sh (Docker/non-GPU) has the same sync daemon pattern,
    using /opt/results/ instead of /work/results/."""

    @pytest.fixture(autouse=True)
    def _load_script(self):
        self.source = _read_script(BOOTSTRAP_CPU)

    def test_sync_daemon_pid_initialized(self):
        """SYNC_DAEMON_PID must be initialized as empty."""
        assert re.search(r'SYNC_DAEMON_PID\s*=\s*""', self.source), (
            "SYNC_DAEMON_PID=\"\" initialization not found in ec2-bootstrap.sh"
        )

    def test_sync_daemon_background_loop_exists(self):
        """A background sync daemon loop must exist."""
        assert re.search(r"\(\s*\n.*?sleep\s+60.*?\)\s*&", self.source, re.DOTALL), (
            "No background sync daemon with 'sleep 60' found in ec2-bootstrap.sh"
        )

    def test_sync_daemon_writes_heartbeat(self):
        """The sync daemon must write heartbeat to S3."""
        assert re.search(r'aws\s+s3\s+cp\s+.*heartbeat', self.source), (
            "No heartbeat S3 upload found in ec2-bootstrap.sh"
        )

    def test_sync_daemon_uses_opt_results_path(self):
        """The sync daemon must sync /opt/results/ (not /work/results/)."""
        daemon_match = re.search(
            r"\(\s*\n(.*?)\)\s*&\s*\nSYNC_DAEMON_PID",
            self.source, re.DOTALL,
        )
        assert daemon_match, "Cannot locate sync daemon body in ec2-bootstrap.sh"
        daemon_body = daemon_match.group(1)

        assert "/opt/results/" in daemon_body, (
            "Sync daemon in ec2-bootstrap.sh must use /opt/results/ path"
        )

    def test_sync_daemon_killed_after_experiment(self):
        """The sync daemon must be killed after experiment."""
        assert re.search(r'kill.*SYNC_DAEMON_PID', self.source), (
            "SYNC_DAEMON_PID not killed in ec2-bootstrap.sh"
        )

    def test_trap_syncs_results_before_exit_code(self):
        """EXIT trap must sync /opt/results/ before writing exit_code."""
        cleanup_source = ""
        if "cleanup()" in self.source or "cleanup ()" in self.source:
            cleanup_source = _extract_function_body(self.source, "cleanup")
        else:
            trap_match = re.search(
                r"trap\s+['\"](.+?)['\"].*EXIT",
                self.source, re.DOTALL,
            )
            if trap_match:
                cleanup_source = trap_match.group(1)

        lines = cleanup_source.splitlines() if cleanup_source else self.source.splitlines()
        results_sync_line = None
        exit_code_write_line = None
        for i, line in enumerate(lines):
            if re.search(r'aws\s+s3\s+sync.*/opt/results/', line):
                if results_sync_line is None:
                    results_sync_line = i
            if re.search(r'exit_code', line) and re.search(r'aws\s+s3\s+cp', line):
                exit_code_write_line = i

        assert results_sync_line is not None, (
            "No /opt/results/ sync found in ec2-bootstrap.sh trap/cleanup"
        )
        assert exit_code_write_line is not None, (
            "No exit_code write found in ec2-bootstrap.sh trap/cleanup"
        )
        assert results_sync_line < exit_code_write_line, (
            "Results sync must come before exit_code write in ec2-bootstrap.sh"
        )


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 3 — S3 check_heartbeat() Function
# ═══════════════════════════════════════════════════════════════════════════

class TestS3CheckHeartbeat:
    """Verify s3.py has a check_heartbeat() function with correct contract."""

    def test_check_heartbeat_function_exists(self):
        """s3.py must define a check_heartbeat() function."""
        from cloud import s3
        assert hasattr(s3, "check_heartbeat"), (
            "s3.py does not have a check_heartbeat() function"
        )

    def test_check_heartbeat_accepts_run_id(self):
        """check_heartbeat() must accept a run_id parameter."""
        from cloud import s3
        sig = inspect.signature(s3.check_heartbeat)
        assert "run_id" in sig.parameters, (
            "check_heartbeat() does not have a run_id parameter"
        )

    def test_check_heartbeat_returns_dict_when_found(self):
        """check_heartbeat() must return a dict with 'timestamp' and 'age_seconds'
        when the heartbeat file exists in S3."""
        from cloud import s3

        ts = "2026-02-23T10:00:00Z"
        mock_response = {"Body": mock.MagicMock()}
        mock_response["Body"].read.return_value = ts.encode()

        with mock.patch.object(s3, "_get_s3_client", create=True) as mock_client_fn:
            mock_client = mock.MagicMock()
            mock_client.get_object.return_value = mock_response
            mock_client_fn.return_value = mock_client

            # Also try mocking subprocess if the impl uses aws CLI
            with mock.patch("subprocess.run") as mock_run:
                mock_run.return_value = mock.MagicMock(
                    returncode=0, stdout=ts, stderr=""
                )
                result = s3.check_heartbeat("test-run-123")

        assert result is not None, "check_heartbeat() returned None for existing heartbeat"
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert "timestamp" in result, "Result missing 'timestamp' key"
        assert "age_seconds" in result, "Result missing 'age_seconds' key"
        assert isinstance(result["age_seconds"], (int, float)), (
            "age_seconds must be numeric"
        )

    def test_check_heartbeat_returns_none_when_missing(self):
        """check_heartbeat() must return None when no heartbeat file exists."""
        from cloud import s3

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(
                returncode=1, stdout="", stderr="An error occurred (NoSuchKey)"
            )
            result = s3.check_heartbeat("nonexistent-run")

        assert result is None, (
            f"check_heartbeat() should return None for missing heartbeat, got {result}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 4 — S3 tail_log() Function
# ═══════════════════════════════════════════════════════════════════════════

class TestS3TailLog:
    """Verify s3.py has a tail_log() function with correct contract."""

    def test_tail_log_function_exists(self):
        """s3.py must define a tail_log() function."""
        from cloud import s3
        assert hasattr(s3, "tail_log"), (
            "s3.py does not have a tail_log() function"
        )

    def test_tail_log_accepts_run_id(self):
        """tail_log() must accept a run_id parameter."""
        from cloud import s3
        sig = inspect.signature(s3.tail_log)
        assert "run_id" in sig.parameters, (
            "tail_log() does not have a run_id parameter"
        )

    def test_tail_log_has_lines_parameter(self):
        """tail_log() must accept a 'lines' parameter."""
        from cloud import s3
        sig = inspect.signature(s3.tail_log)
        assert "lines" in sig.parameters, (
            "tail_log() does not have a 'lines' parameter"
        )

    def test_tail_log_lines_default_is_50(self):
        """tail_log() 'lines' parameter must default to 50."""
        from cloud import s3
        sig = inspect.signature(s3.tail_log)
        default = sig.parameters["lines"].default
        assert default == 50, (
            f"tail_log() lines default is {default}, expected 50"
        )

    def test_tail_log_has_follow_parameter(self):
        """tail_log() must accept a 'follow' parameter."""
        from cloud import s3
        sig = inspect.signature(s3.tail_log)
        assert "follow" in sig.parameters, (
            "tail_log() does not have a 'follow' parameter"
        )

    def test_tail_log_follow_default_is_false(self):
        """tail_log() 'follow' parameter must default to False."""
        from cloud import s3
        sig = inspect.signature(s3.tail_log)
        default = sig.parameters["follow"].default
        assert default is False, (
            f"tail_log() follow default is {default}, expected False"
        )

    def test_tail_log_returns_string(self):
        """tail_log() must return a string (the log content)."""
        from cloud import s3

        fake_log = "\n".join(f"line {i}" for i in range(100))
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(
                returncode=0, stdout=fake_log, stderr=""
            )
            result = s3.tail_log("test-run-123", lines=10, follow=False)

        assert isinstance(result, str), f"tail_log() must return str, got {type(result)}"

    def test_tail_log_respects_line_count(self):
        """tail_log() must return at most the requested number of lines."""
        from cloud import s3

        fake_log = "\n".join(f"log line {i}" for i in range(200))
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(
                returncode=0, stdout=fake_log, stderr=""
            )
            result = s3.tail_log("test-run-123", lines=20, follow=False)

        result_lines = result.strip().split("\n")
        assert len(result_lines) <= 20, (
            f"tail_log(lines=20) returned {len(result_lines)} lines, expected <= 20"
        )

    def test_tail_log_returns_last_lines(self):
        """tail_log() must return the LAST N lines (tail behavior)."""
        from cloud import s3

        lines_content = [f"line-{i}" for i in range(100)]
        fake_log = "\n".join(lines_content)
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(
                returncode=0, stdout=fake_log, stderr=""
            )
            result = s3.tail_log("test-run-123", lines=5, follow=False)

        # Should contain the last lines
        assert "line-99" in result, (
            "tail_log() does not return the last lines of the log"
        )
        assert "line-0" not in result, (
            "tail_log() should not return early lines when requesting tail"
        )


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 5 — remote.py poll_status() Enhancement
# ═══════════════════════════════════════════════════════════════════════════

class TestPollStatusEnhancement:
    """Verify poll_status() checks EC2 instance state and includes heartbeat."""

    def test_poll_status_checks_instance_state_when_no_exit_code(self):
        """When no exit_code is found in S3, poll_status() must check EC2 instance state."""
        from cloud import remote

        # Set up state with a running instance
        state = {
            "run_id": "test-run-1",
            "status": "running",
            "instance_id": "i-1234567890",
            "backend": "aws",
            "exit_code": None,
            "started_at": "2026-02-23T10:00:00Z",
            "finished_at": None,
        }

        with mock.patch.object(remote, "_load_state", return_value=state):
            with mock.patch.object(remote.s3_helper, "check_exit_code", return_value=None):
                # Mock backend status returning 'terminated'
                with mock.patch.object(remote, "_get_backend_for_run", create=True) as mock_get_backend:
                    mock_backend = mock.MagicMock()
                    mock_backend.status.return_value = "terminated"
                    mock_get_backend.return_value = mock_backend

                    result = remote.poll_status("test-run-1")

        # When instance is terminated without exit_code, should indicate this
        assert result.get("status") in ("terminated_no_results", "failed", "terminated"), (
            f"poll_status() should detect terminated instance, got status={result.get('status')}"
        )

    def test_poll_status_returns_terminated_no_results(self):
        """poll_status() must return terminated_no_results when instance is dead
        and no exit_code exists."""
        from cloud import remote

        state = {
            "run_id": "dead-run",
            "status": "running",
            "instance_id": "i-dead",
            "backend": "aws",
            "exit_code": None,
            "started_at": "2026-02-23T10:00:00Z",
            "finished_at": None,
        }

        with mock.patch.object(remote, "_load_state", return_value=state.copy()):
            with mock.patch.object(remote.s3_helper, "check_exit_code", return_value=None):
                with mock.patch.object(remote.s3_helper, "check_heartbeat", return_value=None):
                    with mock.patch.object(remote, "_get_backend_for_run", create=True) as mock_get_backend:
                        mock_backend = mock.MagicMock()
                        mock_backend.status.return_value = "terminated"
                        mock_get_backend.return_value = mock_backend

                        with mock.patch.object(remote, "_update_state", side_effect=lambda rid, **kw: {**state, **kw}):
                            result = remote.poll_status("dead-run")

        assert result.get("status") == "terminated_no_results", (
            f"Expected status='terminated_no_results', got '{result.get('status')}'"
        )
        assert "instance_state" in result or "message" in result, (
            "Response should include instance_state or message for terminated instances"
        )

    def test_poll_status_includes_heartbeat_info(self):
        """poll_status() must include heartbeat information when available."""
        from cloud import remote

        state = {
            "run_id": "hb-run",
            "status": "running",
            "instance_id": "i-hb",
            "backend": "aws",
            "exit_code": None,
            "started_at": "2026-02-23T10:00:00Z",
            "finished_at": None,
        }
        heartbeat = {"timestamp": "2026-02-23T10:05:00Z", "age_seconds": 30}

        with mock.patch.object(remote, "_load_state", return_value=state):
            with mock.patch.object(remote.s3_helper, "check_exit_code", return_value=None):
                with mock.patch.object(remote.s3_helper, "check_heartbeat", return_value=heartbeat):
                    with mock.patch.object(remote, "_get_backend_for_run", create=True) as mock_get_backend:
                        mock_backend = mock.MagicMock()
                        mock_backend.status.return_value = "running"
                        mock_get_backend.return_value = mock_backend

                        result = remote.poll_status("hb-run")

        assert "last_heartbeat" in result or "heartbeat_age_seconds" in result, (
            "poll_status() must include heartbeat info in response"
        )

    def test_poll_status_warns_on_stale_heartbeat(self):
        """poll_status() must include warning when heartbeat age > 600s."""
        from cloud import remote

        state = {
            "run_id": "stale-run",
            "status": "running",
            "instance_id": "i-stale",
            "backend": "aws",
            "exit_code": None,
            "started_at": "2026-02-23T08:00:00Z",
            "finished_at": None,
        }
        # Stale heartbeat: 15 minutes old
        heartbeat = {"timestamp": "2026-02-23T09:45:00Z", "age_seconds": 900}

        with mock.patch.object(remote, "_load_state", return_value=state):
            with mock.patch.object(remote.s3_helper, "check_exit_code", return_value=None):
                with mock.patch.object(remote.s3_helper, "check_heartbeat", return_value=heartbeat):
                    with mock.patch.object(remote, "_get_backend_for_run", create=True) as mock_get_backend:
                        mock_backend = mock.MagicMock()
                        mock_backend.status.return_value = "running"
                        mock_get_backend.return_value = mock_backend

                        result = remote.poll_status("stale-run")

        assert result.get("warning") == "heartbeat_stale", (
            f"Expected warning='heartbeat_stale' for 900s-old heartbeat, got {result.get('warning')}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 6 — remote.py gc_stale_runs()
# ═══════════════════════════════════════════════════════════════════════════

class TestGcStaleRuns:
    """Verify remote.py has a gc_stale_runs() function."""

    def test_gc_stale_runs_function_exists(self):
        """remote.py must define a gc_stale_runs() function."""
        from cloud import remote
        assert hasattr(remote, "gc_stale_runs"), (
            "remote.py does not have a gc_stale_runs() function"
        )

    def test_gc_stale_runs_returns_count(self):
        """gc_stale_runs() must return a count of cleaned-up runs."""
        from cloud import remote

        # Mock the internals: list_runs returns running entries with dead instances
        runs = [
            {
                "run_id": "stale-1",
                "status": "running",
                "instance_id": "i-dead1",
                "backend": "aws",
            },
        ]
        with mock.patch.object(remote, "list_runs", return_value=runs):
            with mock.patch.object(remote.s3_helper, "check_exit_code", return_value=None):
                with mock.patch.object(remote.s3_helper, "write_marker"):
                    with mock.patch.object(remote, "_update_state"):
                        mock_backend = mock.MagicMock()
                        mock_backend.status.return_value = "terminated"

                        result = remote.gc_stale_runs(mock_backend)

        assert isinstance(result, int), (
            f"gc_stale_runs() must return int, got {type(result)}"
        )

    def test_gc_stale_runs_writes_exit_code_137_for_dead_instances(self):
        """gc_stale_runs() must write exit_code=137 to S3 for terminated instances."""
        from cloud import remote

        runs = [
            {
                "run_id": "dead-run",
                "status": "running",
                "instance_id": "i-dead",
                "backend": "aws",
            },
        ]
        with mock.patch.object(remote, "list_runs", return_value=runs):
            with mock.patch.object(remote.s3_helper, "check_exit_code", return_value=None):
                with mock.patch.object(remote.s3_helper, "write_marker") as mock_write:
                    with mock.patch.object(remote, "_update_state"):
                        mock_backend = mock.MagicMock()
                        mock_backend.status.return_value = "terminated"

                        remote.gc_stale_runs(mock_backend)

        # Should have called write_marker with exit_code=137
        mock_write.assert_called()
        call_args = mock_write.call_args_list
        found_137 = any(
            "exit_code" in str(c) and "137" in str(c)
            for c in call_args
        )
        assert found_137, (
            f"gc_stale_runs() did not write exit_code=137 for dead instance. "
            f"Calls: {call_args}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 7 — CLI `logs` Subcommand
# ═══════════════════════════════════════════════════════════════════════════

class TestCLILogsSubcommand:
    """Verify cloud-run has a `logs` subcommand."""

    @pytest.fixture(autouse=True)
    def _load_cli(self):
        self.source = _read_cloud_run_cli()

    def test_logs_subparser_exists(self):
        """cloud-run must have a 'logs' subparser."""
        assert re.search(r'sub\.add_parser\(\s*"logs"', self.source), (
            "No 'logs' subparser found in cloud-run CLI"
        )

    def test_logs_has_run_id_argument(self):
        """The logs subparser must accept a run_id positional argument."""
        # Find the logs subparser block
        assert re.search(r'"logs".*?add_argument\(\s*"run_id"', self.source, re.DOTALL), (
            "logs subparser does not have a run_id positional argument"
        )

    def test_logs_has_lines_flag(self):
        """The logs subparser must have --lines/-n flag."""
        assert re.search(r'"--lines".*?"-n"', self.source) or \
               re.search(r'"-n".*?"--lines"', self.source), (
            "logs subparser does not have --lines/-n flag"
        )

    def test_logs_lines_default_is_50(self):
        """The --lines flag must default to 50."""
        assert re.search(r'default\s*=\s*50', self.source), (
            "--lines flag does not default to 50"
        )

    def test_logs_has_follow_flag(self):
        """The logs subparser must have --follow/-f flag."""
        assert re.search(r'"--follow".*?"-f"', self.source) or \
               re.search(r'"-f".*?"--follow"', self.source), (
            "logs subparser does not have --follow/-f flag"
        )

    def test_cmd_logs_function_exists(self):
        """cloud-run must define a cmd_logs() function."""
        assert re.search(r'^def\s+cmd_logs\s*\(', self.source, re.MULTILINE), (
            "cmd_logs() function not defined in cloud-run CLI"
        )

    def test_cmd_logs_calls_tail_log(self):
        """cmd_logs() must call s3.tail_log() or s3_helper.tail_log()."""
        assert "tail_log" in self.source, (
            "cmd_logs() does not reference tail_log()"
        )


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 8 — Enhanced Status Display
# ═══════════════════════════════════════════════════════════════════════════

class TestEnhancedStatusDisplay:
    """Verify cloud-run enhanced display for running instances."""

    @pytest.fixture(autouse=True)
    def _load_cli(self):
        self.source = _read_cloud_run_cli()

    def test_print_run_shows_computed_elapsed_for_running(self):
        """_print_run() must compute live elapsed time from started_at for running instances.

        The existing code only shows stored elapsed_seconds (post-completion).
        The spec requires computing live elapsed from the start timestamp for
        currently running instances.
        """
        func_match = re.search(
            r'def\s+_print_run\s*\(.*?\):\s*\n(.*?)(?=\ndef\s|\Z)',
            self.source,
            re.DOTALL,
        )
        assert func_match, "_print_run() function not found"
        body = func_match.group(1)

        # Must compute elapsed from started_at for running instances
        # (not just showing stored elapsed_seconds which only exists after completion)
        has_live_elapsed = (
            "fromisoformat" in body or
            ("datetime" in body and "started_at" in body) or
            ("time.time" in body and "started_at" in body)
        )
        assert has_live_elapsed, (
            "_print_run() does not compute live elapsed time from started_at for running instances"
        )

    def test_print_run_shows_cost_estimate(self):
        """_print_run() must display a cost estimate for running instances."""
        func_match = re.search(
            r'def\s+_print_run\s*\(.*?\):\s*\n(.*?)(?=\ndef\s|\Z)',
            self.source,
            re.DOTALL,
        )
        assert func_match, "_print_run() function not found"
        body = func_match.group(1)

        assert "cost" in body.lower() or "$" in body or "hourly_rate" in body.lower(), (
            "_print_run() does not show cost estimate"
        )

    def test_print_run_shows_heartbeat(self):
        """_print_run() must display heartbeat info for running instances."""
        func_match = re.search(
            r'def\s+_print_run\s*\(.*?\):\s*\n(.*?)(?=\ndef\s|\Z)',
            self.source,
            re.DOTALL,
        )
        assert func_match, "_print_run() function not found"
        body = func_match.group(1)

        assert "heartbeat" in body.lower() or "check_heartbeat" in body, (
            "_print_run() does not show heartbeat info"
        )

    def test_cmd_ls_shows_elapsed_column(self):
        """cmd_ls() must show an 'Elapsed' column for running instances."""
        func_match = re.search(
            r'def\s+cmd_ls\s*\(.*?\):\s*\n(.*?)(?=\ndef\s|\Z)',
            self.source,
            re.DOTALL,
        )
        assert func_match, "cmd_ls() function not found"
        body = func_match.group(1)

        assert "Elapsed" in body or "ELAPSED" in body, (
            "cmd_ls() does not include 'Elapsed' column"
        )

    def test_cmd_ls_shows_cost_column(self):
        """cmd_ls() must show an 'Est. Cost' or 'COST' column."""
        func_match = re.search(
            r'def\s+cmd_ls\s*\(.*?\):\s*\n(.*?)(?=\ndef\s|\Z)',
            self.source,
            re.DOTALL,
        )
        assert func_match, "cmd_ls() function not found"
        body = func_match.group(1)

        assert "Cost" in body or "COST" in body or "cost" in body, (
            "cmd_ls() does not include cost column"
        )

    def test_cmd_status_shows_log_lines(self):
        """cmd_status() must show last log lines for running instances."""
        func_match = re.search(
            r'def\s+cmd_status\s*\(.*?\):\s*\n(.*?)(?=\ndef\s|\Z)',
            self.source,
            re.DOTALL,
        )
        assert func_match, "cmd_status() function not found"
        body = func_match.group(1)

        assert "tail_log" in body or "log" in body.lower(), (
            "cmd_status() does not show log lines for running instances"
        )

    def test_print_run_shows_last_log_lines(self):
        """_print_run() must show last 3 log lines if available."""
        func_match = re.search(
            r'def\s+_print_run\s*\(.*?\):\s*\n(.*?)(?=\ndef\s|\Z)',
            self.source,
            re.DOTALL,
        )
        assert func_match, "_print_run() function not found"
        body = func_match.group(1)

        assert "tail_log" in body or "log_lines" in body or "last.*log" in body.lower(), (
            "_print_run() does not show last log lines"
        )


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 9 — validate.py Module
# ═══════════════════════════════════════════════════════════════════════════

class TestValidateModuleExists:
    """Verify validate.py exists with required functions."""

    def test_validate_module_exists(self):
        """orchestration-kit/tools/cloud/validate.py must exist."""
        validate_path = CLOUD_DIR / "validate.py"
        assert validate_path.exists(), (
            f"validate.py not found at {validate_path}"
        )

    def test_validate_module_importable(self):
        """cloud.validate must be importable."""
        try:
            from cloud import validate
        except ImportError as e:
            pytest.fail(f"Cannot import cloud.validate: {e}")

    def test_syntax_check_function_exists(self):
        """validate.py must define syntax_check()."""
        from cloud import validate
        assert hasattr(validate, "syntax_check"), (
            "validate.py does not define syntax_check()"
        )

    def test_import_check_function_exists(self):
        """validate.py must define import_check()."""
        from cloud import validate
        assert hasattr(validate, "import_check"), (
            "validate.py does not define import_check()"
        )

    def test_smoke_test_function_exists(self):
        """validate.py must define smoke_test()."""
        from cloud import validate
        assert hasattr(validate, "smoke_test"), (
            "validate.py does not define smoke_test()"
        )

    def test_validate_all_function_exists(self):
        """validate.py must define validate_all()."""
        from cloud import validate
        assert hasattr(validate, "validate_all"), (
            "validate.py does not define validate_all()"
        )


class TestSyntaxCheck:
    """Verify syntax_check() behavior."""

    def test_syntax_check_returns_tuple(self):
        """syntax_check() must return (bool, str) tuple."""
        from cloud.validate import syntax_check

        # Use a known-valid script
        result = syntax_check(__file__)
        assert isinstance(result, tuple) and len(result) == 2, (
            f"syntax_check() must return 2-tuple, got {type(result)}"
        )
        assert isinstance(result[0], bool), "First element must be bool"
        assert isinstance(result[1], str), "Second element must be str"

    def test_syntax_check_passes_valid_python(self):
        """syntax_check() must return (True, ...) for valid Python."""
        from cloud.validate import syntax_check

        result = syntax_check(__file__)  # This test file is valid Python
        assert result[0] is True, (
            f"syntax_check() failed on valid Python: {result[1]}"
        )

    def test_syntax_check_fails_invalid_python(self, tmp_path):
        """syntax_check() must return (False, ...) for invalid Python."""
        from cloud.validate import syntax_check

        bad_script = tmp_path / "bad.py"
        bad_script.write_text("def broken(\n  # missing closing paren and colon\n")

        result = syntax_check(str(bad_script))
        assert result[0] is False, (
            "syntax_check() should fail on invalid syntax"
        )
        assert len(result[1]) > 0, "Error message should not be empty"


class TestImportCheck:
    """Verify import_check() behavior."""

    def test_import_check_returns_tuple(self):
        """import_check() must return (bool, str) tuple."""
        from cloud.validate import import_check

        result = import_check(__file__)
        assert isinstance(result, tuple) and len(result) == 2, (
            f"import_check() must return 2-tuple, got {type(result)}"
        )

    def test_import_check_passes_with_available_imports(self, tmp_path):
        """import_check() must return (True, ...) when all imports are available."""
        from cloud.validate import import_check

        good_script = tmp_path / "good.py"
        good_script.write_text("import os\nimport sys\nimport json\n")

        result = import_check(str(good_script))
        assert result[0] is True, (
            f"import_check() failed on script with stdlib imports: {result[1]}"
        )

    def test_import_check_fails_on_missing_import(self, tmp_path):
        """import_check() must return (False, ...) when an import is not available."""
        from cloud.validate import import_check

        bad_script = tmp_path / "missing_import.py"
        bad_script.write_text("import nonexistent_package_xyzzy_12345\n")

        result = import_check(str(bad_script))
        assert result[0] is False, (
            "import_check() should fail when imports are missing"
        )
        assert "nonexistent_package_xyzzy_12345" in result[1], (
            "Error message should name the missing import"
        )

    def test_import_check_extracts_from_import(self, tmp_path):
        """import_check() must handle 'import X' statements."""
        from cloud.validate import import_check

        script = tmp_path / "import_test.py"
        script.write_text("import os\nimport pathlib\n")

        result = import_check(str(script))
        assert result[0] is True, f"Failed on standard imports: {result[1]}"

    def test_import_check_extracts_from_import_from(self, tmp_path):
        """import_check() must handle 'from X import Y' statements."""
        from cloud.validate import import_check

        script = tmp_path / "from_test.py"
        script.write_text("from pathlib import Path\nfrom os.path import join\n")

        result = import_check(str(script))
        assert result[0] is True, f"Failed on from imports: {result[1]}"


class TestSmokeTest:
    """Verify smoke_test() behavior."""

    def test_smoke_test_returns_tuple(self, tmp_path):
        """smoke_test() must return (bool, str) tuple."""
        from cloud.validate import smoke_test

        script = tmp_path / "smoke.py"
        script.write_text(textwrap.dedent("""\
            import sys
            if "--smoke-test" in sys.argv:
                print("smoke ok")
                sys.exit(0)
        """))

        result = smoke_test(str(script))
        assert isinstance(result, tuple) and len(result) == 2, (
            f"smoke_test() must return 2-tuple, got {type(result)}"
        )

    def test_smoke_test_passes_on_zero_exit(self, tmp_path):
        """smoke_test() must return (True, ...) when script exits 0."""
        from cloud.validate import smoke_test

        script = tmp_path / "good_smoke.py"
        script.write_text(textwrap.dedent("""\
            import sys
            if "--smoke-test" in sys.argv:
                sys.exit(0)
        """))

        result = smoke_test(str(script))
        assert result[0] is True, f"smoke_test() failed: {result[1]}"

    def test_smoke_test_fails_on_nonzero_exit(self, tmp_path):
        """smoke_test() must return (False, ...) when script exits non-zero."""
        from cloud.validate import smoke_test

        script = tmp_path / "bad_smoke.py"
        script.write_text(textwrap.dedent("""\
            import sys
            if "--smoke-test" in sys.argv:
                print("Shape mismatch!", file=sys.stderr)
                sys.exit(1)
        """))

        result = smoke_test(str(script))
        assert result[0] is False, (
            "smoke_test() should fail on non-zero exit"
        )

    def test_smoke_test_respects_timeout(self, tmp_path):
        """smoke_test() must fail if script exceeds timeout."""
        from cloud.validate import smoke_test

        script = tmp_path / "slow_smoke.py"
        script.write_text(textwrap.dedent("""\
            import sys, time
            if "--smoke-test" in sys.argv:
                time.sleep(999)
        """))

        result = smoke_test(str(script), timeout=2)
        assert result[0] is False, (
            "smoke_test() should fail on timeout"
        )
        assert "timeout" in result[1].lower(), (
            "Error message should mention timeout"
        )

    def test_smoke_test_disables_cuda(self, tmp_path):
        """smoke_test() must set CUDA_VISIBLE_DEVICES="" to force CPU."""
        from cloud.validate import smoke_test

        script = tmp_path / "cuda_check.py"
        script.write_text(textwrap.dedent("""\
            import os, sys
            if "--smoke-test" in sys.argv:
                cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "NOT_SET")
                if cuda == "":
                    sys.exit(0)
                else:
                    print(f"CUDA_VISIBLE_DEVICES={cuda}", file=sys.stderr)
                    sys.exit(1)
        """))

        result = smoke_test(str(script))
        assert result[0] is True, (
            f"smoke_test() should set CUDA_VISIBLE_DEVICES='': {result[1]}"
        )


class TestValidateAll:
    """Verify validate_all() behavior."""

    def test_validate_all_returns_tuple(self, tmp_path):
        """validate_all() must return (bool, list) tuple."""
        from cloud.validate import validate_all

        script = tmp_path / "valid.py"
        script.write_text("print('hello')\n")

        result = validate_all(str(script), skip_smoke=True)
        assert isinstance(result, tuple) and len(result) == 2, (
            f"validate_all() must return 2-tuple, got {type(result)}"
        )
        assert isinstance(result[0], bool), "First element must be bool"
        assert isinstance(result[1], list), "Second element must be list of check results"

    def test_validate_all_check_results_format(self, tmp_path):
        """Each check result must be (check_name, passed, message)."""
        from cloud.validate import validate_all

        script = tmp_path / "format_test.py"
        script.write_text("import os\nprint(os.getcwd())\n")

        _, checks = validate_all(str(script), skip_smoke=True)
        for check in checks:
            assert len(check) == 3, f"Check result must be 3-tuple, got {check}"
            name, ok, msg = check
            assert isinstance(name, str), f"Check name must be str, got {type(name)}"
            assert isinstance(ok, bool), f"Check passed must be bool, got {type(ok)}"
            assert isinstance(msg, str), f"Check message must be str, got {type(msg)}"

    def test_validate_all_runs_syntax_check(self, tmp_path):
        """validate_all() must include a 'syntax' check."""
        from cloud.validate import validate_all

        script = tmp_path / "syn_test.py"
        script.write_text("x = 1\n")

        _, checks = validate_all(str(script), skip_smoke=True)
        check_names = [c[0] for c in checks]
        assert "syntax" in check_names, (
            f"validate_all() must include 'syntax' check, got {check_names}"
        )

    def test_validate_all_runs_import_check(self, tmp_path):
        """validate_all() must include an 'imports' check."""
        from cloud.validate import validate_all

        script = tmp_path / "imp_test.py"
        script.write_text("import os\n")

        _, checks = validate_all(str(script), skip_smoke=True)
        check_names = [c[0] for c in checks]
        assert "imports" in check_names, (
            f"validate_all() must include 'imports' check, got {check_names}"
        )

    def test_validate_all_short_circuits_on_syntax_failure(self, tmp_path):
        """validate_all() must stop after syntax failure (don't run further checks)."""
        from cloud.validate import validate_all

        bad_script = tmp_path / "bad_syn.py"
        bad_script.write_text("def broken(\n")

        all_ok, checks = validate_all(str(bad_script), skip_smoke=True)
        assert all_ok is False, "Should fail on syntax error"
        check_names = [c[0] for c in checks]
        assert "syntax" in check_names, "Must include syntax check"
        assert "imports" not in check_names, (
            "Must not run import check after syntax failure"
        )

    def test_validate_all_skip_smoke_parameter(self, tmp_path):
        """validate_all(skip_smoke=True) must skip the smoke test."""
        from cloud.validate import validate_all

        script = tmp_path / "skip_smoke.py"
        script.write_text("import os\n")

        _, checks = validate_all(str(script), skip_smoke=True)
        check_names = [c[0] for c in checks]
        assert "smoke_test" not in check_names, (
            "skip_smoke=True should skip smoke_test check"
        )

    def test_validate_all_includes_smoke_test_by_default(self, tmp_path):
        """validate_all(skip_smoke=False) must include the smoke test."""
        from cloud.validate import validate_all

        script = tmp_path / "with_smoke.py"
        script.write_text(textwrap.dedent("""\
            import sys
            if "--smoke-test" in sys.argv:
                sys.exit(0)
        """))

        _, checks = validate_all(str(script), skip_smoke=False)
        check_names = [c[0] for c in checks]
        assert "smoke_test" in check_names, (
            "skip_smoke=False should include smoke_test check"
        )


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 10 — CLI --validate and --skip-smoke Flags
# ═══════════════════════════════════════════════════════════════════════════

class TestCLIValidateFlags:
    """Verify cloud-run run subcommand has --validate and --skip-smoke flags."""

    @pytest.fixture(autouse=True)
    def _load_cli(self):
        self.source = _read_cloud_run_cli()

    def test_validate_flag_exists_on_run(self):
        """cloud-run run must have a --validate flag."""
        # Find the run subparser section and look for --validate
        assert re.search(r'p_run.*add_argument\(\s*"--validate"', self.source, re.DOTALL), (
            "--validate flag not found on the run subparser"
        )

    def test_skip_smoke_flag_exists_on_run(self):
        """cloud-run run must have a --skip-smoke flag."""
        assert re.search(r'p_run.*add_argument\(\s*"--skip-smoke"', self.source, re.DOTALL), (
            "--skip-smoke flag not found on the run subparser"
        )

    def test_cmd_run_calls_validate_all(self):
        """cmd_run() must call validate_all() when --validate is provided."""
        assert "validate_all" in self.source or "validate" in self.source, (
            "cmd_run() does not reference validation logic"
        )

    def test_cmd_run_exits_2_on_validation_failure(self):
        """cmd_run() must exit with code 2 when validation fails."""
        assert "sys.exit(2)" in self.source, (
            "cmd_run() does not exit with code 2 on validation failure"
        )

    def test_cmd_run_prints_check_results(self):
        """cmd_run() must print pass/fail for each validation check."""
        assert "PASS" in self.source and "FAIL" in self.source, (
            "cmd_run() does not print PASS/FAIL for validation checks"
        )

    def test_validation_runs_before_ec2_provision(self):
        """Validation must run BEFORE calling remote.run() to prevent wasted EC2 cost."""
        # Find validate block and remote.run() call — validate must come first
        validate_match = re.search(r'validate_all|validate\(', self.source)
        remote_run_match = re.search(r'remote\.run\(', self.source)
        assert validate_match, "No validation call found in cmd_run()"
        assert remote_run_match, "No remote.run() call found in cmd_run()"
        assert validate_match.start() < remote_run_match.start(), (
            "Validation must execute before remote.run()"
        )


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 11 — experiment.sh --validate in Compute Directive
# ═══════════════════════════════════════════════════════════════════════════

class TestExperimentShValidateDirective:
    """Verify experiment.sh compute directive template includes --validate."""

    def test_run_run_directive_includes_validate(self):
        """run_run() compute directive template must include --validate."""
        source = _read_script(EXPERIMENT_SH)
        body = _extract_function_body(source, "run_run")
        assert body, "Could not extract run_run() function body"

        assert "--validate" in body, (
            "run_run() compute directive does not include --validate flag"
        )


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 12 — state.py gc_stale()
# ═══════════════════════════════════════════════════════════════════════════

class TestStateGcStale:
    """Verify state.py has a gc_stale() function."""

    def test_gc_stale_function_exists(self):
        """state.py must define a gc_stale() function."""
        from cloud import state
        assert hasattr(state, "gc_stale"), (
            "state.py does not define a gc_stale() function"
        )

    def test_gc_stale_returns_count(self):
        """gc_stale() must return a count of cleaned entries."""
        from cloud import state

        result = state.gc_stale(str(Path("/tmp/nonexistent-project-root-xyz")))
        assert isinstance(result, int), (
            f"gc_stale() must return int, got {type(result)}"
        )

    def test_gc_stale_cleans_stale_running_entries(self, tmp_path):
        """gc_stale() must clean entries with status 'running' that have exit codes in S3."""
        from cloud import state

        # Set up project with a running entry
        project_root = str(tmp_path)
        state.register_run(
            project_root, "stale-run-1",
            instance_id="i-stale1",
            backend="aws",
            instance_type="c7a.8xlarge",
        )

        # Mock S3 check to say exit_code exists
        with mock.patch("cloud.s3.check_exit_code", return_value=0):
            result = state.gc_stale(project_root)

        assert result >= 1, (
            f"gc_stale() should have cleaned at least 1 entry, got {result}"
        )

    def test_gc_stale_marks_old_entries_without_heartbeat_as_stale(self, tmp_path):
        """gc_stale() must mark entries older than 24h with no heartbeat as stale."""
        from cloud import state

        project_root = str(tmp_path)
        # Register a run with old timestamp
        old_time = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        state.register_run(
            project_root, "ancient-run",
            instance_id="i-old",
            backend="aws",
            instance_type="c7a.8xlarge",
            launched_at=old_time,
        )

        with mock.patch("cloud.s3.check_exit_code", return_value=None):
            with mock.patch("cloud.s3.check_heartbeat", return_value=None):
                result = state.gc_stale(project_root)

        assert result >= 1, (
            f"gc_stale() should have cleaned old entries without heartbeat, got {result}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 13 — cloud-run gc Integration
# ═══════════════════════════════════════════════════════════════════════════

class TestCloudRunGcIntegration:
    """Verify cloud-run gc calls both gc_stale_runs() and gc_stale()."""

    @pytest.fixture(autouse=True)
    def _load_cli(self):
        self.source = _read_cloud_run_cli()

    def test_cmd_gc_calls_gc_stale_runs(self):
        """cmd_gc() must call gc_stale_runs() from remote.py."""
        func_match = re.search(
            r'def\s+cmd_gc\s*\(.*?\):\s*\n(.*?)(?=\ndef\s|\Z)',
            self.source,
            re.DOTALL,
        )
        assert func_match, "cmd_gc() function not found"
        body = func_match.group(1)

        assert "gc_stale_runs" in body, (
            "cmd_gc() does not call gc_stale_runs()"
        )

    def test_cmd_gc_calls_gc_stale(self):
        """cmd_gc() must call gc_stale() from state.py."""
        func_match = re.search(
            r'def\s+cmd_gc\s*\(.*?\):\s*\n(.*?)(?=\ndef\s|\Z)',
            self.source,
            re.DOTALL,
        )
        assert func_match, "cmd_gc() function not found"
        body = func_match.group(1)

        assert "gc_stale" in body, (
            "cmd_gc() does not call gc_stale()"
        )

    def test_cmd_gc_prints_stale_run_summary(self):
        """cmd_gc() must print a summary specifically about stale run cleanup counts.

        The existing gc only reports backend-level orphaned resources.
        The spec requires reporting both stale runs (gc_stale_runs) and
        local state entries (gc_stale) with counts.
        """
        func_match = re.search(
            r'def\s+cmd_gc\s*\(.*?\):\s*\n(.*?)(?=\ndef\s|\Z)',
            self.source,
            re.DOTALL,
        )
        assert func_match, "cmd_gc() function not found"
        body = func_match.group(1)

        # Must explicitly reference stale runs — the existing "Cleaned:" message is
        # for backend-level resource cleanup, not stale run GC
        assert "stale" in body.lower() and ("run" in body.lower() or "state" in body.lower()), (
            "cmd_gc() does not print a summary about stale run cleanup"
        )
