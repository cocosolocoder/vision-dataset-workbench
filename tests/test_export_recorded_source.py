"""Exports from the plan-recorded source paths must pin the read.

These tests cover the export run *without* ``--source-dir``: the bytes
streamed into the package must come from the one ordinary regular file
confirmed immediately before reading.  A replacement landing in the gap
between the confirmation and the actual open — another regular file
(even byte-identical with identical size and mtime), a symlink (even
one pointing at the original), or a writerless named pipe — must fail
the whole export with the sample's full SHA-256, the recorded source
path and the concrete reason, never block on the pipe and never accept
the replacement on a matching digest.

A source path that is already a symlink when the export starts keeps
its existing behavior: the link is followed once and the file it names
is what gets pinned and read.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import socket
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from vision_workbench import confirmed
from vision_workbench.exporter import ExportError, export_split
from vision_workbench.store import DatasetStore

REPO_ROOT = Path(__file__).resolve().parents[1]


class RecordedSourceHarness(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / "dataset"
        self.store = DatasetStore(self.root)
        self.store.initialize()
        self._serial = 0

    def tearDown(self) -> None:
        self.temp.cleanup()

    def add_sample(
        self, label: str, content: bytes, name: str | None = None
    ) -> tuple[str, Path]:
        self._serial += 1
        source = self.base / (name or f"sample-{self._serial}.jpg")
        source.write_bytes(content)
        result = self.store.add(source, label)
        self.assertTrue(result.added)
        return result.digest, source

    def recorded_source(self, plan: str = "baseline", index: int = 0) -> Path:
        members = self.store.get_split(plan)["sets"]["train"]["members"]
        return Path(members[index]["source"])

    def create_plan(self, name: str = "baseline",
                    ratios: list[str] | None = None) -> None:
        self.store.create_split(name, 42, ratios or ["1", "0", "0"])

    def export(self, target: Path | None = None,
               skip_unlabeled: bool = False) -> dict:
        target = target or self.base / "out.zip"
        return export_split(
            self.store, "baseline", target, skip_unlabeled=skip_unlabeled
        )

    @contextlib.contextmanager
    def race_open(self, source: Path, mutate):
        """Run ``mutate`` inside the recorded source's real ``os.open``.

        The recorded-source confirm sees a regular file at the lstat; the
        mutation then lands strictly before the descriptor-creating open
        syscall, the exact check/open gap the pinning must close.
        """
        real_open = os.open
        state = {"fired": False}

        def racing_open(path, flags, *args, **kwargs):
            if (
                not state["fired"]
                and str(Path(os.fspath(path))) == str(source)
            ):
                state["fired"] = True
                mutate()
            return real_open(path, flags, *args, **kwargs)

        with mock.patch.object(confirmed.os, "open", side_effect=racing_open):
            yield state

    def run_with_timeout(self, action, timeout: float = 20.0) -> dict:
        """Run ``action`` on a daemon thread; fail instead of hanging."""
        holder: dict = {}

        def work() -> None:
            try:
                holder["value"] = action()
            except BaseException as failure:  # captured, asserted by caller
                holder["error"] = failure

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        thread.join(timeout)
        self.assertFalse(
            thread.is_alive(),
            "export blocked waiting for a writerless named pipe",
        )
        return holder


class RecordedSourceReplacementTest(RecordedSourceHarness):
    def _assert_clean_failure(
        self, failure: Exception, digest: str, source: Path, target: Path
    ) -> None:
        message = str(failure)
        self.assertIn(digest, message)
        self.assertIn(str(source), message)
        self.assertEqual(len(digest), 64)
        self.assertNotIn("Traceback", message)
        self.assertFalse(target.exists())
        leftovers = [
            path.name
            for path in target.parent.iterdir()
            if path.name.startswith(f".{target.name}.")
            and path.name.endswith(".tmp")
        ]
        self.assertEqual(leftovers, [])

    def test_fifo_swapped_between_check_and_open_fails_without_waiting(self) -> None:
        digest, source = self.add_sample("cat", b"abc")
        self.create_plan()
        target = self.base / "out.zip"

        def make_pipe() -> None:
            source.unlink()
            os.mkfifo(source)  # no writer: a blocking open would hang forever

        with self.race_open(source, make_pipe):
            holder = self.run_with_timeout(
                lambda: export_split(self.store, "baseline", target)
            )
        self.assertIn("error", holder)
        failure = holder["error"]
        self.assertIsInstance(failure, ExportError)
        message = str(failure)
        self.assertIn(digest, message)
        self.assertIn(str(source), message)
        self.assertIn("replaced", message)
        self.assertIn("named pipe", message)
        self.assertFalse(target.exists())
        # The pipe was never read: it is still exactly the pipe the
        # racing mutation created, and the package never materialized.
        self.assertTrue(stat_is_fifo(source))

    def test_fifo_present_when_export_starts_is_rejected_not_read(self) -> None:
        digest, source = self.add_sample("cat", b"abc")
        self.create_plan()
        source.unlink()
        os.mkfifo(source)  # no writer
        target = self.base / "out.zip"
        holder = self.run_with_timeout(
            lambda: export_split(self.store, "baseline", target)
        )
        self.assertIn("error", holder)
        failure = holder["error"]
        self.assertIsInstance(failure, ExportError)
        self.assertIn(digest, str(failure))
        self.assertIn(str(source), str(failure))
        self.assertIn("not a regular file", str(failure))
        self.assertIn("named pipe", str(failure))
        self.assertFalse(target.exists())
        self.assertTrue(stat_is_fifo(source))

    def test_identical_file_swapped_in_gap_is_rejected(self) -> None:
        digest, source = self.add_sample("cat", b"abc")
        self.create_plan()
        original = os.lstat(source)
        target = self.base / "out.zip"
        replacement = self.base / "replacement.jpg"
        replacement.write_bytes(b"abc")  # same content, size and...
        os.utime(
            replacement,
            ns=(original.st_atime_ns, original.st_mtime_ns),
        )

        def swap() -> None:
            os.replace(replacement, source)  # ...and same mtime, new inode

        with self.race_open(source, swap):
            with self.assertRaises(ExportError) as caught:
                self.export(target)
        self._assert_clean_failure(caught.exception, digest, source, target)
        self.assertIn("replaced", str(caught.exception))

    def test_symlink_swapped_in_gap_rejected_even_at_original(self) -> None:
        digest, source = self.add_sample("cat", b"abc")
        self.create_plan()
        original = self.base / "original.jpg"
        os.link(source, original)  # keep the confirmed inode alive elsewhere

        target = self.base / "out.zip"

        def make_link() -> None:
            source.unlink()
            source.symlink_to(original)

        with self.race_open(source, make_link):
            with self.assertRaises(ExportError) as caught:
                self.export(target)
        self._assert_clean_failure(caught.exception, digest, source, target)
        self.assertIn("symlink", str(caught.exception))

    def test_failure_aborts_whole_export_after_earlier_samples_copied(self) -> None:
        first_digest, first_source = self.add_sample(
            "cat", b"aaa", name="a.jpg"
        )
        second_digest, second_source = self.add_sample(
            "cat", b"bbb", name="b.jpg"
        )
        self.create_plan()
        target = self.base / "out.zip"
        manifest_before = self.store.manifest_path.read_bytes()
        plan_before = (self.store.splits_directory / "baseline.json").read_bytes()

        # Package entries stream in arc-path order, which sorts by the
        # full digest; race whichever sample is streamed last, so earlier
        # entries have already entered the temporary package when it fails.
        ordered = sorted(
            ((first_digest, first_source), (second_digest, second_source)),
            key=lambda item: item[0],
        )
        raced_digest, raced_source = ordered[-1]
        earlier_digest = ordered[0][0]

        def make_pipe() -> None:
            raced_source.unlink()
            os.mkfifo(raced_source)

        with self.race_open(raced_source, make_pipe):
            holder = self.run_with_timeout(
                lambda: export_split(self.store, "baseline", target)
            )
        self.assertIn("error", holder)
        self.assertIn(raced_digest, str(holder["error"]))
        self.assertNotIn(earlier_digest, str(holder["error"]))
        self.assertFalse(target.exists())
        leftovers = [
            path.name
            for path in target.parent.iterdir()
            if path.name.startswith(f".{target.name}.")
            and path.name.endswith(".tmp")
        ]
        self.assertEqual(leftovers, [])
        # Workspace data and saved plan are never rewritten.
        self.assertEqual(self.store.manifest_path.read_bytes(), manifest_before)
        self.assertEqual(
            (self.store.splits_directory / "baseline.json").read_bytes(),
            plan_before,
        )

    def test_existing_target_kept_when_recorded_source_also_unusable(self) -> None:
        # An occupant at the target name is rejected by the start-up
        # conflict rule before any source is opened, so it is preserved
        # even when the recorded source itself is a writerless FIFO: the
        # command neither blocks on the pipe nor touches the occupant.
        digest, source = self.add_sample("cat", b"abc")
        self.create_plan()
        target = self.base / "out.zip"
        target.write_bytes(b"keep-me")
        source.unlink()
        os.mkfifo(source)

        holder = self.run_with_timeout(
            lambda: export_split(self.store, "baseline", target)
        )
        self.assertIn("error", holder)
        self.assertIsInstance(holder["error"], ExportError)
        self.assertEqual(target.read_bytes(), b"keep-me")
        self.assertTrue(stat.S_ISFIFO(os.lstat(source).st_mode))
        # The failed export does not rewrite the plan's recorded source.
        self.assertEqual(
            self.store.get_split("baseline")["sets"]["train"]["members"][0][
                "sha256"
            ],
            digest,
        )

    def test_no_alternative_same_content_source_is_substituted(self) -> None:
        digest, source = self.add_sample("cat", b"abc")
        # A second, same-content file exists elsewhere; it must not be
        # silently used when the recorded source is replaced.
        twin = self.base / "twin.jpg"
        twin.write_bytes(b"abc")
        self.create_plan()
        target = self.base / "out.zip"

        def make_pipe() -> None:
            source.unlink()
            os.mkfifo(source)

        with self.race_open(source, make_pipe):
            holder = self.run_with_timeout(
                lambda: export_split(self.store, "baseline", target)
            )
        self.assertIn("error", holder)
        self.assertIn(digest, str(holder["error"]))
        self.assertFalse(target.exists())
        # The plan's recorded source path is not rewritten.
        self.assertEqual(self.recorded_source(), source)

    def test_socket_swapped_in_gap_refused_at_open_without_read(self) -> None:
        # A unix datagram socket makes the non-blocking open itself fail
        # with ENXIO: the replacement reason must still name the object.
        digest, source = self.add_sample("cat", b"abc")
        self.create_plan()
        target_file = self.base / "resolved.jpg"
        os.link(source, target_file)
        source.unlink()
        source.symlink_to(target_file)  # swap races the resolved target
        target = self.base / "out.zip"
        self._socket: socket.socket | None = None

        def make_socket() -> None:
            target_file.unlink()
            self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            self._socket.bind(str(target_file))

        try:
            with self.race_open(target_file, make_socket):
                with self.assertRaises(ExportError) as caught:
                    self.export(target)
        finally:
            if self._socket is not None:
                self._socket.close()
        message = str(caught.exception)
        self.assertIn(digest, message)
        # The recorded path (the link), not the resolved target, is named.
        self.assertIn(str(source), message)
        self.assertIn("replaced", message)
        self.assertIn("socket", message)
        self.assertFalse(target.exists())

    def test_symlink_at_export_start_is_followed_and_pinned(self) -> None:
        digest, source = self.add_sample("cat", b"abc")
        self.create_plan()
        # The recorded path is itself a symlink when the export starts;
        # the link target is the confirmed regular file.
        target_file = self.base / "target.jpg"
        os.link(source, target_file)
        source.unlink()
        source.symlink_to(target_file)

        target = self.base / "out.zip"
        result = self.export(target)
        self.assertEqual(result["exported"], 1)
        with zipfile.ZipFile(target) as archive:
            entry = next(
                name for name in archive.namelist() if name.endswith(".jpg")
            )
            self.assertEqual(archive.read(entry), b"abc")
        self.assertEqual(hashlib.sha256(b"abc").hexdigest(), digest)

    def test_normal_export_bytes_unchanged_and_extension_kept(self) -> None:
        self.add_sample("cat", b"abc", name="one.png")
        self.add_sample("dog", b"def", name="two.gif")
        self.create_plan()
        first = self.base / "first.zip"
        second = self.base / "second.zip"
        self.export(first)
        self.export(second)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        with zipfile.ZipFile(first) as archive:
            suffixes = sorted(
                Path(name).suffix
                for name in archive.namelist()
                if not name.endswith("/") and name != "manifest.json"
            )
            self.assertEqual(suffixes, [".gif", ".png"])


class RecordedSourceCliTest(RecordedSourceHarness):
    """The CLI contract for a source replaced in the check/open gap."""

    def test_cli_fifo_race_fails_cleanly(self) -> None:
        digest, source = self.add_sample("cat", b"abc")
        self.create_plan()
        target = self.base / "out.zip"

        driver = self.base / "driver.py"
        driver.write_text(
            textwrap.dedent(
                """
                import sys
                import os
                from pathlib import Path
                from unittest import mock

                workspace, plan, target, source = (
                    sys.argv[1], sys.argv[2],
                    Path(sys.argv[3]), Path(sys.argv[4]),
                )
                sys.path.insert(0, sys.argv[5])

                from vision_workbench import confirmed

                real_open = os.open
                state = {"fired": False}

                def racing_open(path, flags, *args, **kwargs):
                    if (
                        not state["fired"]
                        and str(Path(os.fspath(path))) == str(source)
                    ):
                        state["fired"] = True
                        source.unlink()
                        os.mkfifo(source)
                    return real_open(path, flags, *args, **kwargs)

                sys.argv = [
                    "vision-workbench", "export",
                    workspace, plan, str(target),
                ]
                with mock.patch.object(
                    confirmed.os, "open", side_effect=racing_open
                ):
                    from vision_workbench.__main__ import main
                    raise SystemExit(main())
                """
            )
        )

        env = dict(os.environ)
        completed = subprocess.run(
            [
                sys.executable,
                str(driver),
                str(self.root),
                "baseline",
                str(target),
                str(source),
                str(REPO_ROOT),
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout, "")
        self.assertNotIn("Traceback", completed.stderr)
        self.assertIn(digest, completed.stderr)
        self.assertIn(str(source), completed.stderr)
        self.assertIn("named pipe", completed.stderr)
        self.assertFalse(target.exists())
        leftovers = [
            path.name
            for path in target.parent.iterdir()
            if path.name.startswith(f".{target.name}.")
            and path.name.endswith(".tmp")
        ]
        self.assertEqual(leftovers, [])
        # The saved plan still records the original source path.
        plan = self.store.get_split("baseline")
        self.assertEqual(
            plan["sets"]["train"]["members"][0]["source"], str(source)
        )
        self.assertEqual(
            json.loads(
                (self.store.splits_directory / "baseline.json").read_text()
            )["sets"]["train"]["members"][0]["source"],
            str(source),
        )


def stat_is_fifo(path: Path) -> bool:
    return stat.S_ISFIFO(os.lstat(path).st_mode)


if __name__ == "__main__":
    unittest.main()
