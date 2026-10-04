"""Regression tests for read failures during single-file ``add``.

The public contract promises that a source which cannot be opened, or
whose bytes cannot be read in full, fails the import with the source
path and the reason — even when the content would have matched an
already-registered digest.  These tests pin that down: the library
raises a handleable failure, the CLI exits non-zero with the error on
standard error and no success output, and a failed import never
registers a record (not even a partial digest paired with the source's
size) nor touches existing samples, batch history or saved split plans.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from vision_workbench.store import DatasetStore


REPO_ROOT = Path(__file__).resolve().parent.parent

CONTENT_A = b"A" * 100000
CONTENT_B = b"B" * 300000
DIGEST_A = hashlib.sha256(CONTENT_A).hexdigest()
DIGEST_B = hashlib.sha256(CONTENT_B).hexdigest()


def _manifest_items(workspace: Path) -> list[dict]:
    manifest_path = workspace / ".vision-workbench" / "manifest.json"
    if not manifest_path.exists():
        return []
    return json.loads(manifest_path.read_text(encoding="utf-8"))["items"]


@contextmanager
def _fail_open(source: Path):
    """Make opening ``source`` fail with an access error (EACCES)."""
    target = str(source.resolve())
    real_open = os.open

    def hooked_open(path, flags, *args, **kwargs):
        if str(path) == target:
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        return real_open(path, flags, *args, **kwargs)

    with mock.patch.object(os, "open", hooked_open):
        yield


class _PartialReadThenError:
    """File-like wrapper: serves one real chunk, then fails the read."""

    def __init__(self, stream) -> None:
        self._stream = stream
        self._first = True

    def read(self, size: int = -1) -> bytes:
        if self._first:
            self._first = False
            return self._stream.read(size)
        raise OSError("simulated read failure")

    def close(self) -> None:
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> bool:
        self.close()
        return False


@contextmanager
def _fail_read_after_first_chunk(source: Path):
    """Let ``source`` open and deliver its first chunk, then fail the read."""
    target = str(source.resolve())
    real_open, real_fdopen = os.open, os.fdopen
    state: dict[str, int] = {}

    def hooked_open(path, flags, *args, **kwargs):
        descriptor = real_open(path, flags, *args, **kwargs)
        if str(path) == target:
            state["descriptor"] = descriptor
        return descriptor

    def hooked_fdopen(descriptor, mode="r", *args, **kwargs):
        stream = real_fdopen(descriptor, mode, *args, **kwargs)
        if descriptor == state.get("descriptor"):
            return _PartialReadThenError(stream)
        return stream

    with mock.patch.object(os, "open", hooked_open), mock.patch.object(
        os, "fdopen", hooked_fdopen
    ):
        yield


class AddReadFailureTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.base = Path(self._temporary.name)
        self.workspace = self.base / "workspace"
        self.store = DatasetStore(self.workspace)

    def _write_source(self, name: str = "img.png", content: bytes = CONTENT_A) -> Path:
        source = self.base / name
        source.write_bytes(content)
        return source

    def test_open_access_error_fails_with_path_and_reason(self) -> None:
        source = self._write_source()

        with _fail_open(source):
            with self.assertRaises(ValueError) as raised:
                self.store.add(source, "cat")

        message = str(raised.exception)
        self.assertIn(str(source.resolve()), message)
        self.assertIn("Permission denied", message)
        self.assertEqual(_manifest_items(self.workspace), [])

    def test_read_error_after_partial_content_fails_and_registers_nothing(
        self,
    ) -> None:
        source = self._write_source()

        with _fail_read_after_first_chunk(source):
            with self.assertRaises(ValueError) as raised:
                self.store.add(source, "cat")

        message = str(raised.exception)
        self.assertIn(str(source.resolve()), message)
        self.assertIn("simulated read failure", message)
        # No record may be pieced together from the partial digest and the
        # source file's size: the failed read registers nothing at all.
        self.assertEqual(_manifest_items(self.workspace), [])

    def _prepare_workspace_with_history(self) -> dict:
        """Register two samples, apply a label batch and save a split plan."""
        first = self.base / "first.png"
        first.write_bytes(CONTENT_A)
        other = self.base / "other.png"
        other.write_bytes(CONTENT_B)
        self.assertTrue(self.store.add(first, "cat").added)
        self.assertTrue(self.store.add(other, "dog").added)
        submitted = self.store.submit_batch(
            {
                "batch": "b1",
                "changes": [
                    {"sha256": DIGEST_A, "old": "cat", "new": "kitten"}
                ],
            }
        )
        self.assertEqual(submitted["status"], "applied")
        plan = self.store.create_split("baseline", 42, ["1/2", "1/4", "1/4"]).plan
        return {
            "first": first,
            "other": other,
            "plan": plan,
            "history": self.store.history(),
            "summary": self.store.summary(),
        }

    def _assert_workspace_preserved(self, state: dict) -> None:
        items = _manifest_items(self.workspace)
        self.assertEqual(len(items), 2)
        by_digest = {item["sha256"]: item for item in items}
        # The originally registered digest, source path, byte size and
        # current label all survive; the failed request's label "bird"
        # was never written anywhere.
        record = by_digest[DIGEST_A]
        self.assertEqual(record["source"], str(state["first"].resolve()))
        self.assertEqual(record["size"], len(CONTENT_A))
        self.assertEqual(record["label"], "kitten")
        self.assertNotIn("bird", json.dumps(items))
        # The unrelated sample is untouched, as are the sample count and
        # the total byte count.
        self.assertEqual(by_digest[DIGEST_B]["label"], "dog")
        self.assertEqual(self.store.summary(), state["summary"])
        # Batch history and the saved split plan are unchanged.
        self.assertEqual(self.store.history(), state["history"])
        self.assertEqual(self.store.get_split("baseline"), state["plan"])

    def test_open_failure_on_known_content_preserves_existing_sample(self) -> None:
        state = self._prepare_workspace_with_history()
        second = self.base / "second.png"
        second.write_bytes(CONTENT_A)  # same content as the registered sample

        with _fail_open(second):
            with self.assertRaises(ValueError) as raised:
                self.store.add(second, "bird")

        message = str(raised.exception)
        self.assertIn(str(second.resolve()), message)
        self.assertIn("Permission denied", message)
        self._assert_workspace_preserved(state)

    def test_read_failure_on_known_content_preserves_existing_sample(self) -> None:
        state = self._prepare_workspace_with_history()
        second = self.base / "second.png"
        second.write_bytes(CONTENT_A)  # same content as the registered sample

        with _fail_read_after_first_chunk(second):
            with self.assertRaises(ValueError) as raised:
                self.store.add(second, "bird")

        message = str(raised.exception)
        self.assertIn(str(second.resolve()), message)
        self.assertIn("simulated read failure", message)
        self._assert_workspace_preserved(state)

    def test_new_content_registers_once_with_full_digest_and_size(self) -> None:
        source = self._write_source()

        result = self.store.add(source, "cat")

        self.assertTrue(result.added)
        self.assertEqual(result.digest, DIGEST_A)
        (item,) = _manifest_items(self.workspace)
        self.assertEqual(item["sha256"], DIGEST_A)
        self.assertEqual(item["size"], len(CONTENT_A))
        self.assertEqual(item["source"], str(source.resolve()))
        self.assertEqual(item["label"], "cat")

    def test_readable_duplicate_reports_already_present_and_keeps_record(
        self,
    ) -> None:
        first = self._write_source("first.png")
        second = self._write_source("second.png")

        added = self.store.add(first, "cat")
        duplicate = self.store.add(second, "dog")

        self.assertTrue(added.added)
        self.assertFalse(duplicate.added)
        self.assertEqual(duplicate.digest, DIGEST_A)
        (item,) = _manifest_items(self.workspace)
        self.assertEqual(item["source"], str(first.resolve()))
        self.assertEqual(item["label"], "cat")


# CLI fault-injection driver: installs the same open/read failures used
# by the library-level tests, then runs the real ``add`` command through
# ``vision_workbench.__main__.main`` in a fresh process so its exit
# status and output streams are observed exactly as a user would see
# them.  Arguments: repo root, workspace, resolved source, fault mode.
_CLI_FAULT_SCRIPT = """
import errno
import os
import sys

repo, workspace, source, fault = sys.argv[1:5]
sys.path.insert(0, repo)

real_open = os.open
real_fdopen = os.fdopen
opened = {}


def hooked_open(path, flags, *args, **kwargs):
    if str(path) == source and fault == "open":
        raise PermissionError(errno.EACCES, "Permission denied", str(path))
    descriptor = real_open(path, flags, *args, **kwargs)
    if str(path) == source:
        opened["descriptor"] = descriptor
    return descriptor


class PartialReadThenError:
    def __init__(self, stream):
        self._stream = stream
        self._first = True

    def read(self, size=-1):
        if self._first:
            self._first = False
            return self._stream.read(size)
        raise OSError("simulated read failure")

    def close(self):
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False


def hooked_fdopen(descriptor, mode="r", *args, **kwargs):
    stream = real_fdopen(descriptor, mode, *args, **kwargs)
    if descriptor == opened.get("descriptor"):
        return PartialReadThenError(stream)
    return stream


os.open = hooked_open
os.fdopen = hooked_fdopen

from vision_workbench.__main__ import main

sys.argv = ["vision-workbench", "add", workspace, source, "--label", "cat"]
raise SystemExit(main())
"""


class AddReadFailureCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.base = Path(self._temporary.name)
        self.workspace = self.base / "workspace"
        self.source = self.base / "img.png"
        self.source.write_bytes(CONTENT_A)

    def _run_cli(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "vision_workbench", *arguments],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def _run_add_with_fault(self, fault: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                "-c",
                _CLI_FAULT_SCRIPT,
                str(REPO_ROOT),
                str(self.workspace),
                str(self.source.resolve()),
                fault,
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def _assert_cli_read_failure(
        self, completed: subprocess.CompletedProcess, reason: str
    ) -> None:
        self.assertEqual(completed.returncode, 1, completed.stderr)
        # No success result of any kind on standard output.
        self.assertEqual(completed.stdout, "")
        self.assertNotIn("added", completed.stdout)
        self.assertNotIn("already present", completed.stdout)
        # The error names this source and the read failure, on standard
        # error, with no unhandled traceback shown to the user.
        self.assertIn(str(self.source.resolve()), completed.stderr)
        self.assertIn(reason, completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)
        self.assertEqual(_manifest_items(self.workspace), [])

    def test_cli_open_failure_exits_nonzero_with_error_on_stderr(self) -> None:
        completed = self._run_add_with_fault("open")
        self._assert_cli_read_failure(completed, "Permission denied")

    def test_cli_read_failure_exits_nonzero_with_error_on_stderr(self) -> None:
        completed = self._run_add_with_fault("read")
        self._assert_cli_read_failure(completed, "simulated read failure")

    def test_cli_success_and_duplicate_output_unchanged(self) -> None:
        added = self._run_cli(
            "add", str(self.workspace), str(self.source), "--label", "cat"
        )
        self.assertEqual(added.returncode, 0, added.stderr)
        self.assertEqual(added.stdout, f"added: {DIGEST_A}\n")
        self.assertEqual(added.stderr, "")

        duplicate = self._run_cli(
            "add", str(self.workspace), str(self.source), "--label", "dog"
        )
        self.assertEqual(duplicate.returncode, 0, duplicate.stderr)
        self.assertEqual(duplicate.stdout, f"already present: {DIGEST_A}\n")
        self.assertEqual(duplicate.stderr, "")


if __name__ == "__main__":
    unittest.main()
