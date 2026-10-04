"""Regression coverage for read failures during single-file ``add``.

A single-file import must read the whole source and compute its full
SHA-256 before it may register a sample.  Two source-side failures are
pinned here:

* the path exists and names a regular file, but opening it fails with an
  access error (``EACCES``);
* the file opens and some bytes are read, but a later read fails.

Neither failure may be masked by content that happens to match an
already-registered digest: the import fails with the source path and the
read reason, no sample or byte is added, existing records keep their
first-registration source and current label, and batch history and saved
split plans stay untouched.  The command line reports such a failure on
standard error with a non-zero status and no success output, never with
an unhandled traceback.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from vision_workbench import confirmed
from vision_workbench.__main__ import main as cli_main
from vision_workbench.store import DatasetStore


REPO_ROOT = Path(__file__).resolve().parents[1]

CONTENT_A = b"A" * 100000
CONTENT_B = b"B" * 70000
DIGEST_A = hashlib.sha256(CONTENT_A).hexdigest()
DIGEST_B = hashlib.sha256(CONTENT_B).hexdigest()

NON_ROOT = not hasattr(os, "geteuid") or os.geteuid() != 0


def _state_path(workspace: Path, name: str) -> Path:
    return workspace / ".vision-workbench" / name


def _manifest_items(workspace: Path) -> list[dict]:
    manifest_path = _state_path(workspace, "manifest.json")
    if not manifest_path.exists():
        return []
    return json.loads(manifest_path.read_text(encoding="utf-8"))["items"]


def _history(workspace: Path) -> dict:
    return json.loads(_state_path(workspace, "batches.json").read_text("utf-8"))


class _PartialFailStream:
    """File-like wrapper that yields part of the first chunk, then fails.

    The first ``read`` returns at most the first 4096 bytes of genuinely
    read source content; every later ``read`` raises ``EIO``.  ``state``
    records how many bytes were actually delivered so a test can prove
    the failure landed in the middle of the read rather than before it.
    """

    def __init__(self, stream: io.BufferedReader, state: dict) -> None:
        self._stream = stream
        self._state = state

    def read(self, size: int = -1) -> bytes:
        self._state["calls"] += 1
        if self._state["calls"] == 1:
            limit = min(size if isinstance(size, int) and size > 0 else 4096, 4096)
            chunk = self._stream.read(limit)
            self._state["partial_bytes"] += len(chunk)
            return chunk
        raise OSError(errno.EIO, "simulated read failure")

    def __enter__(self) -> "_PartialFailStream":
        return self

    def __exit__(self, *exc: object) -> bool:
        self._stream.close()
        return False

    def close(self) -> None:
        self._stream.close()


@contextlib.contextmanager
def _partial_read_failure():
    """Make :func:`confirmed.hash_descriptor` fail after a partial read."""
    real_fdopen = os.fdopen
    state = {"calls": 0, "partial_bytes": 0}

    def failing_fdopen(descriptor: int, *args: object, **kwargs: object):
        return _PartialFailStream(real_fdopen(descriptor, *args, **kwargs), state)

    with mock.patch.object(confirmed.os, "fdopen", failing_fdopen):
        yield state


def _install_open_access_error(source: Path) -> mock._patch:
    """Force the source's ``open`` to fail with ``EACCES``.

    Every other descriptor (workspace lock, journal files) still opens;
    only the confirmed source open is refused.  This works even when the
    tests run as root, where a real chmod would not deny access.
    """
    real_open = os.open
    denied = str(source.resolve())

    def denying_open(path, flags, *args, **kwargs):
        if str(path) == denied:
            raise PermissionError(
                errno.EACCES, "Permission denied (injected)", denied
            )
        return real_open(path, flags, *args, **kwargs)

    return mock.patch.object(os, "open", denying_open)


class AddReadFailureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.workspace = self.base / "workspace"
        self.store = DatasetStore(self.workspace)
        self._locked: list[Path] = []

    def tearDown(self) -> None:
        for path in self._locked:
            try:
                os.chmod(path, 0o644)
            except OSError:
                pass
        self.temp.cleanup()

    def _write(self, name: str, content: bytes) -> Path:
        path = self.base / name
        path.write_bytes(content)
        return path

    def _deny_by_mode(self, path: Path) -> None:
        """Real access denial via mode bits (effective only when non-root)."""
        os.chmod(path, 0o000)
        self._locked.append(path)

    # ------------------------------------------------------------------
    # Failure while opening the source
    # ------------------------------------------------------------------

    @unittest.skipUnless(NON_ROOT, "mode bits do not deny access to root")
    def test_open_access_error_on_new_content_fails_with_path_and_reason(self) -> None:
        source = self._write("new.png", CONTENT_A)

        self._deny_by_mode(source)
        with self.assertRaises(ValueError) as raised:
            self.store.add(source, "cat")

        message = str(raised.exception)
        self.assertIn("cannot read", message)
        self.assertIn(str(source.resolve()), message)
        self.assertIn("Permission denied", message)
        self.assertNotIn("already present", message)
        self.assertEqual(_manifest_items(self.workspace), [])
        self.assertEqual(self.store.summary(), {"items": 0, "bytes": 0, "labels": {}})
        self.assertFalse(_state_path(self.workspace, ".txn.json").exists())

    def test_injected_open_access_error_on_new_content_fails(self) -> None:
        source = self._write("new.png", CONTENT_A)

        with _install_open_access_error(source):
            with self.assertRaises(ValueError) as raised:
                self.store.add(source, "cat")

        message = str(raised.exception)
        self.assertIn(str(source.resolve()), message)
        self.assertIn("Permission denied", message)
        self.assertEqual(_manifest_items(self.workspace), [])
        self.assertFalse(_state_path(self.workspace, ".txn.json").exists())

    @unittest.skipUnless(NON_ROOT, "mode bits do not deny access to root")
    def test_open_access_error_on_duplicate_content_is_not_already_present(
        self,
    ) -> None:
        first = self._write("first.png", CONTENT_A)
        second = self._write("second.png", CONTENT_A)
        self.assertTrue(self.store.add(first, "cat").added)
        other = self._write("other.png", CONTENT_B)
        self.assertTrue(self.store.add(other, "bird").added)

        self._deny_by_mode(second)
        # A different label must never be written, and "already present"
        # must never mask the source failure.
        with self.assertRaises(ValueError) as raised:
            self.store.add(second, "dog")
        message = str(raised.exception)
        self.assertIn(str(second.resolve()), message)
        self.assertIn("Permission denied", message)
        self.assertNotIn("already present", message)

        self._assert_workspace_unchanged_after_failure(first)

    def test_injected_open_error_on_duplicate_preserves_everything(self) -> None:
        first = self._write("first.png", CONTENT_A)
        second = self._write("second.png", CONTENT_A)
        self.assertTrue(self.store.add(first, "cat").added)

        with _install_open_access_error(second):
            with self.assertRaises(ValueError):
                self.store.add(second, "dog")

        items = _manifest_items(self.workspace)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["sha256"], DIGEST_A)
        self.assertEqual(items[0]["source"], str(first.resolve()))
        self.assertEqual(items[0]["size"], len(CONTENT_A))
        self.assertEqual(items[0]["label"], "cat")
        self.assertEqual(self.store.summary()["items"], 1)
        self.assertEqual(self.store.summary()["bytes"], len(CONTENT_A))

    # ------------------------------------------------------------------
    # Failure after some bytes were read
    # ------------------------------------------------------------------

    def test_mid_read_error_on_new_content_fails_and_registers_no_fragment(
        self,
    ) -> None:
        source = self._write("new.png", CONTENT_A)

        with _partial_read_failure() as state:
            with self.assertRaises(ValueError) as raised:
                self.store.add(source, "cat")

        # The failure really landed after bytes were delivered: neither the
        # partial digest plus source size nor anything else was registered.
        self.assertGreater(state["partial_bytes"], 0)
        self.assertLess(state["partial_bytes"], len(CONTENT_A))
        message = str(raised.exception)
        self.assertIn("cannot read", message)
        self.assertIn(str(source.resolve()), message)
        self.assertIn("simulated read failure", message)
        self.assertNotIn("already present", message)
        self.assertEqual(_manifest_items(self.workspace), [])
        self.assertEqual(self.store.summary()["bytes"], 0)
        self.assertFalse(_state_path(self.workspace, ".txn.json").exists())

    def test_mid_read_error_on_duplicate_content_preserves_first_record(
        self,
    ) -> None:
        first = self._write("first.png", CONTENT_A)
        second = self._write("second.png", CONTENT_A)
        self.assertTrue(self.store.add(first, "cat").added)
        other = self._write("other.png", CONTENT_B)
        self.assertTrue(self.store.add(other, "bird").added)

        with _partial_read_failure() as state:
            with self.assertRaises(ValueError) as raised:
                self.store.add(second, "dog")

        self.assertGreater(state["partial_bytes"], 0)
        self.assertIn(str(second.resolve()), str(raised.exception))
        self.assertIn("simulated read failure", str(raised.exception))
        self._assert_workspace_unchanged_after_failure(first)

    def _assert_workspace_unchanged_after_failure(self, first_source: Path) -> None:
        """The duplicate's record and every other sample stay untouched."""
        items = _manifest_items(self.workspace)
        by_digest = {item["sha256"]: item for item in items}
        self.assertEqual(set(by_digest), {DIGEST_A, DIGEST_B})

        first_item = by_digest[DIGEST_A]
        self.assertEqual(first_item["source"], str(first_source.resolve()))
        self.assertEqual(first_item["size"], len(CONTENT_A))
        self.assertEqual(first_item["label"], "cat")
        self.assertNotEqual(first_item["label"], "dog")

        other_item = by_digest[DIGEST_B]
        self.assertEqual(other_item["label"], "bird")
        self.assertEqual(other_item["size"], len(CONTENT_B))

        summary = self.store.summary()
        self.assertEqual(summary["items"], 2)
        self.assertEqual(summary["bytes"], len(CONTENT_A) + len(CONTENT_B))
        self.assertFalse(_state_path(self.workspace, ".txn.json").exists())

    # ------------------------------------------------------------------
    # Batch history and saved split plans survive the failed import
    # ------------------------------------------------------------------

    def test_failed_reads_keep_history_split_plan_and_other_samples(self) -> None:
        first = self._write("first.png", CONTENT_A)
        second = self._write("second.png", CONTENT_A)
        third = self._write("third.png", CONTENT_A)
        other = self._write("other.png", CONTENT_B)
        self.assertTrue(self.store.add(first, "cat").added)
        self.assertTrue(self.store.add(other, "bird").added)

        # A successful label batch creates history the failed import must
        # not touch.
        batch = {
            "batch": "renumber-1",
            "changes": [{"sha256": DIGEST_A, "old": "cat", "new": "kitten"}],
        }
        self.assertEqual(self.store.submit_batch(batch)["status"], "applied")
        self.store.create_split(
            "baseline", 42, ["1/2", "1/2", "0"]
        )

        manifest_before = json.loads(
            _state_path(self.workspace, "manifest.json").read_text("utf-8")
        )
        history_before = _history(self.workspace)
        plan_before = _state_path(
            self.workspace, os.path.join("splits", "baseline.json")
        ).read_text("utf-8")

        # Opening the duplicate fails; the requested label is not applied.
        with _install_open_access_error(second):
            with self.assertRaises(ValueError):
                self.store.add(second, "dog")
        self._assert_preserved_state(
            manifest_before, history_before, plan_before, first
        )

        # A partial-then-failing read of another duplicate is no different.
        with _partial_read_failure():
            with self.assertRaises(ValueError):
                self.store.add(third, "dog")
        self._assert_preserved_state(
            manifest_before, history_before, plan_before, first
        )

        history = self.store.history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["batch"], "renumber-1")
        self.assertFalse(history[0]["undone"])

    def _assert_preserved_state(
        self,
        manifest_before: dict,
        history_before: dict,
        plan_before: str,
        first_source: Path,
    ) -> None:
        manifest_after = json.loads(
            _state_path(self.workspace, "manifest.json").read_text("utf-8")
        )
        self.assertEqual(manifest_after, manifest_before)

        items = {item["sha256"]: item for item in manifest_after["items"]}
        self.assertEqual(set(items), {DIGEST_A, DIGEST_B})
        surviving = items[DIGEST_A]
        self.assertEqual(surviving["source"], str(first_source.resolve()))
        self.assertEqual(surviving["size"], len(CONTENT_A))
        self.assertEqual(surviving["label"], "kitten")
        self.assertEqual(surviving.get("rev"), 1)
        self.assertEqual(items[DIGEST_B]["label"], "bird")

        self.assertEqual(_history(self.workspace), history_before)
        plan_after = _state_path(
            self.workspace, os.path.join("splits", "baseline.json")
        ).read_text("utf-8")
        self.assertEqual(plan_after, plan_before)

        summary = self.store.summary()
        self.assertEqual(summary["items"], 2)
        self.assertEqual(summary["bytes"], len(CONTENT_A) + len(CONTENT_B))

    # ------------------------------------------------------------------
    # Readable sources keep working exactly as before
    # ------------------------------------------------------------------

    def test_readable_new_content_is_registered_once_with_full_digest(self) -> None:
        source = self._write("new.png", CONTENT_A)

        result = self.store.add(source, "cat")
        self.assertTrue(result.added)
        self.assertEqual(result.digest, DIGEST_A)

        (item,) = _manifest_items(self.workspace)
        self.assertEqual(item["sha256"], DIGEST_A)
        self.assertEqual(item["size"], len(CONTENT_A))
        self.assertEqual(item["source"], str(source.resolve()))
        self.assertEqual(item["label"], "cat")

        again = self.store.add(source, "cat")
        self.assertFalse(again.added)
        self.assertEqual(again.digest, DIGEST_A)
        self.assertEqual(len(_manifest_items(self.workspace)), 1)

    def test_readable_duplicate_reports_already_present_and_keeps_metadata(
        self,
    ) -> None:
        first = self._write("first.png", CONTENT_A)
        second = self._write("second.png", CONTENT_A)

        first_result = self.store.add(first, "cat")
        second_result = self.store.add(second, "dog")

        self.assertTrue(first_result.added)
        self.assertFalse(second_result.added)
        self.assertEqual(second_result.digest, DIGEST_A)
        (item,) = _manifest_items(self.workspace)
        self.assertEqual(item["source"], str(first.resolve()))
        self.assertEqual(item["label"], "cat")
        self.assertEqual(item["size"], len(CONTENT_A))


class AddReadFailureCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.workspace = self.base / "workspace"
        self.store = DatasetStore(self.workspace)
        self._locked: list[Path] = []

    def tearDown(self) -> None:
        for path in self._locked:
            try:
                os.chmod(path, 0o644)
            except OSError:
                pass
        self.temp.cleanup()

    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "vision_workbench", *arguments],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    @unittest.skipUnless(NON_ROOT, "mode bits do not deny access to root")
    def test_cli_open_access_error_fails_cleanly(self) -> None:
        first = self.base / "first.png"
        first.write_bytes(CONTENT_A)
        second = self.base / "second.png"
        second.write_bytes(CONTENT_A)
        self.assertTrue(self.store.add(first, "cat").added)

        os.chmod(second, 0o000)
        self._locked.append(second)
        result = self.run_cli(
            "add", str(self.workspace), str(second), "--label", "dog"
        )

        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn("error:", result.stderr)
        self.assertIn(str(second), result.stderr)
        self.assertIn("Permission denied", result.stderr)
        self.assertNotIn("added", result.stdout)
        self.assertNotIn("already present", result.stdout)
        self.assertNotIn("Traceback", result.stderr)

        (item,) = _manifest_items(self.workspace)
        self.assertEqual(item["sha256"], DIGEST_A)
        self.assertEqual(item["source"], str(first.resolve()))
        self.assertEqual(item["size"], len(CONTENT_A))
        self.assertEqual(item["label"], "cat")

    def test_cli_mid_read_error_fails_cleanly(self) -> None:
        first = self.base / "first.png"
        first.write_bytes(CONTENT_A)
        second = self.base / "second.png"
        second.write_bytes(CONTENT_A)
        self.assertTrue(self.store.add(first, "cat").added)

        argv = [
            "vision-workbench",
            "add",
            str(self.workspace),
            str(second),
            "--label",
            "dog",
        ]
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            mock.patch.object(sys, "argv", argv),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
            _partial_read_failure(),
        ):
            status = cli_main()

        self.assertEqual(status, 1)
        self.assertEqual(stdout.getvalue(), "")
        error_text = stderr.getvalue()
        self.assertIn("error:", error_text)
        self.assertIn(str(second), error_text)
        self.assertIn("simulated read failure", error_text)
        self.assertNotIn("added", stdout.getvalue())
        self.assertNotIn("already present", stdout.getvalue())
        self.assertNotIn("Traceback", error_text)

        (item,) = _manifest_items(self.workspace)
        self.assertEqual(item["sha256"], DIGEST_A)
        self.assertEqual(item["source"], str(first.resolve()))
        self.assertEqual(item["label"], "cat")
        self.assertEqual(item["size"], len(CONTENT_A))

    def test_cli_readable_sources_keep_their_success_output(self) -> None:
        first = self.base / "first.png"
        first.write_bytes(CONTENT_A)
        second = self.base / "second.png"
        second.write_bytes(CONTENT_A)

        added = self.run_cli("add", str(self.workspace), str(first), "--label", "cat")
        self.assertEqual(added.returncode, 0, added.stderr)
        self.assertEqual(added.stdout, f"added: {DIGEST_A}\n")
        self.assertEqual(added.stderr, "")

        duplicate = self.run_cli(
            "add", str(self.workspace), str(second), "--label", "dog"
        )
        self.assertEqual(duplicate.returncode, 0, duplicate.stderr)
        self.assertEqual(duplicate.stdout, f"already present: {DIGEST_A}\n")
        self.assertEqual(duplicate.stderr, "")

        (item,) = _manifest_items(self.workspace)
        self.assertEqual(item["source"], str(first.resolve()))
        self.assertEqual(item["label"], "cat")
        self.assertEqual(item["size"], len(CONTENT_A))


if __name__ == "__main__":
    unittest.main()
