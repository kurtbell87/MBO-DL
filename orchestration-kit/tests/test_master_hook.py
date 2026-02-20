from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / ".claude" / "hooks" / "pre-tool-use.sh"


class MasterHookTests(unittest.TestCase):
    def run_hook(self, *, tool: str, payload: dict[str, str], env_overrides: dict[str, str]) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(env_overrides)
        env["CLAUDE_TOOL_NAME"] = tool
        env["CLAUDE_TOOL_INPUT"] = json.dumps(payload)
        return subprocess.run(
            [str(HOOK)],
            cwd=str(ROOT),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_large_read_is_blocked_when_not_allowlisted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            big_file = Path(td) / "big.log"
            big_file.write_bytes(b"x" * 1024)
            proc = self.run_hook(
                tool="Read",
                payload={"file_path": str(big_file)},
                env_overrides={
                    "MAX_READ_BYTES": "100",
                    "RUN_ID": "read-block",
                    "READ_BUDGET_STATE_DIR": td,
                },
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("BLOCKED: Read of large file", proc.stderr)

    def test_large_read_allowlist_bypasses_block(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            big_file = Path(td) / "big.log"
            big_file.write_bytes(b"x" * 1024)
            proc = self.run_hook(
                tool="Read",
                payload={"file_path": str(big_file)},
                env_overrides={
                    "MAX_READ_BYTES": "100",
                    "MUST_READ_ALLOWLIST": str(big_file),
                    "RUN_ID": "read-allow",
                    "READ_BUDGET_STATE_DIR": td,
                },
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_read_budget_blocks_unique_file_overflow(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            first = Path(td) / "a.txt"
            second = Path(td) / "b.txt"
            first.write_bytes(b"a" * 80)
            second.write_bytes(b"b" * 80)

            common_env = {
                "MAX_READ_BYTES": "1000",
                "READ_BUDGET_MAX_FILES": "1",
                "READ_BUDGET_MAX_TOTAL_BYTES": "500",
                "RUN_ID": "budget-overflow",
                "READ_BUDGET_STATE_DIR": td,
            }

            proc1 = self.run_hook(tool="Read", payload={"file_path": str(first)}, env_overrides=common_env)
            self.assertEqual(proc1.returncode, 0, proc1.stderr)

            proc2 = self.run_hook(tool="Read", payload={"file_path": str(second)}, env_overrides=common_env)
            self.assertNotEqual(proc2.returncode, 0)
            self.assertIn("BLOCKED: Read budget exceeded", proc2.stderr)

    def test_read_budget_allowlisted_read_does_not_consume_unique_budget(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            capsule = Path(td) / "research_status.md"
            first = Path(td) / "first.txt"
            second = Path(td) / "second.txt"
            capsule.write_bytes(b"capsule")
            first.write_bytes(b"a" * 10)
            second.write_bytes(b"b" * 10)

            common_env = {
                "MAX_READ_BYTES": "1000",
                "READ_BUDGET_MAX_FILES": "1",
                "READ_BUDGET_MAX_TOTAL_BYTES": "500",
                "MUST_READ_ALLOWLIST": str(capsule),
                "RUN_ID": "budget-allowlisted",
                "READ_BUDGET_STATE_DIR": td,
            }

            proc_capsule = self.run_hook(tool="Read", payload={"file_path": str(capsule)}, env_overrides=common_env)
            self.assertEqual(proc_capsule.returncode, 0, proc_capsule.stderr)

            proc_first = self.run_hook(tool="Read", payload={"file_path": str(first)}, env_overrides=common_env)
            self.assertEqual(proc_first.returncode, 0, proc_first.stderr)

            proc_second = self.run_hook(tool="Read", payload={"file_path": str(second)}, env_overrides=common_env)
            self.assertNotEqual(proc_second.returncode, 0)
            self.assertIn("BLOCKED: Read budget exceeded", proc_second.stderr)

    def test_tdd_green_protection_is_enforced(self) -> None:
        proc = self.run_hook(
            tool="Edit",
            payload={"file_path": "tests/foo_test.py"},
            env_overrides={"TDD_PHASE": "green"},
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Cannot edit test files during GREEN phase", proc.stderr)

    def test_research_synthesize_protection_is_enforced(self) -> None:
        proc = self.run_hook(
            tool="Write",
            payload={"file_path": "analysis.md"},
            env_overrides={"EXP_PHASE": "synthesize"},
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("you may only write to SYNTHESIS.md", proc.stderr)

    def test_math_survey_protection_is_enforced(self) -> None:
        proc = self.run_hook(
            tool="Write",
            payload={"file_path": "Foo.lean"},
            env_overrides={"MATH_PHASE": "survey"},
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("SURVEY phase is read-only", proc.stderr)


if __name__ == "__main__":
    unittest.main()
