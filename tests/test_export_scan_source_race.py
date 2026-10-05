"""Regression coverage for a source changing while the lookup reads it.

Export with ``--source-dir`` scans the given tree and, for *every*
regular file it meets, confirms the entry as a regular file, opens it
pinned (``O_NOFOLLOW``) and hashes the full content.  The confirmation
must bind the object that is actually hashed: when a path is swapped
between the confirmation and the open, the export has to fail naming
that path rather than read the new occupant.

The gap used to be open during the lookup: the scan opened with plain
blocking semantics and did not prove the descriptor, so a regular file
replaced in that gap by a named pipe made the scan wait for a writer
(forever when none came) and otherwise hash pipe bytes.  These tests
pin the required outcomes:

* a named pipe swapped into the gap — with no writer — fails the
  export immediately; the scan neither waits for a writer nor hashes
  pipe bytes (the no-writer case runs in a subprocess so a hang cannot
  stall the suite);
* a named pipe whose writer already offers the exact planned bytes is
  rejected just the same, and the writer is never connected to;
* another regular file renamed into the gap — even one with identical
  content, size and mtime — is rejected, with no fallback to the other
  same-content copy;
* a symbolic link swapped into the gap is rejected, even when it
  points at same-content bytes;
* the rule applies to *every* file read during the lookup, including
  one unrelated to the plan and even after every planned sample has
  already been found;
* entries that are non-regular (pipes) or symlinks when first
  encountered keep being skipped, exactly as before;
* the failure is a clean abort: nonzero CLI status, empty stdout, a
  clear stderr message without a traceback, no target ZIP, no leftover
  temporary package, and the workspace registration and saved plan
  untouched.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest
import zipfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from vision_workbench.exporter import ExportError, export_split
from vision_workbench.store import DatasetStore
import vision_workbench.exporter as exporter_mod

REPO_ROOT = Path(__file__).resolve().parents[1]


class ScanRaceHarness(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.workspace = self.base / "dataset"
        self.store = DatasetStore(self.workspace)
        self.store.initialize()
        self._serial = 0

    def tearDown(self) -> None:
        self.temp.cleanup()

    def add_sample(self, label: str | None, content: bytes) -> str:
        self._serial += 1
        path = self.base / f"sample-{self._serial}.jpg"
        path.write_bytes(content)
        result = self.store.add(path, label)
        self.assertTrue(result.added)
        return result.digest

    def create_plan(self, name: str = "baseline") -> dict:
        return self.store.create_split(
            name, 42, ["0.5", "0.25", "0.25"]
        ).plan

    def export(self, source_dir: Path, target: Path | None = None):
        target = target or self.base / "out.zip"
        return export_split(
            self.store, "baseline", target, source_dir=source_dir
        ), target

    def temp_leftovers(self) -> list[str]:
        return [
            entry.name
            for entry in self.base.iterdir()
            if entry.name.endswith(".tmp")
        ]

    @contextmanager
    def race_after_inspect(self, victim: Path, mutate):
        """Swap ``victim`` strictly after its scan inspection, pre-open.

        The wrapper sits on the shared confirmation rule and only
        fires for an inspection of ``victim`` reached through its own
        (confirmed) parent directory, so same-named entries elsewhere
        in the tree never trigger it.
        """
        parent_inode = os.lstat(victim.parent).st_ino
        real_inspect = exporter_mod.confirmed.inspect_regular
        state = {"fired": False}

        def racing_inspect(path, *args, **kwargs):
            result = real_inspect(path, *args, **kwargs)
            dir_fd = kwargs.get("dir_fd")
            if (
                not state["fired"]
                and os.fspath(path) == victim.name
                and dir_fd is not None
                and os.fstat(dir_fd).st_ino == parent_inode
            ):
                state["fired"] = True
                mutate()
            return result

        with patch.object(
            exporter_mod.confirmed,
            "inspect_regular",
            side_effect=racing_inspect,
        ):
            yield state

    def replace_with_identical_file(self, victim: Path) -> None:
        replacement = victim.with_name(victim.name + ".swap")
        replacement.write_bytes(victim.read_bytes())
        before = os.lstat(victim)
        os.utime(
            replacement, ns=(before.st_atime_ns, before.st_mtime_ns)
        )
        os.replace(replacement, victim)

    def replace_with_symlink(self, victim: Path, holder: Path) -> None:
        victim.unlink()
        victim.symlink_to(holder)

    def replace_with_fifo(self, victim: Path) -> None:
        victim.unlink()
        os.mkfifo(victim)


class FifoSwapFailsWithoutWaitingTest(ScanRaceHarness):
    """The no-writer FIFO gap is exercised in a subprocess.

    A regression that opens the swapped pipe with plain blocking
    semantics would hang this whole test process; the subprocess
    timeout turns that hang into a fast ordinary failure.
    """

    DRIVER = textwrap.dedent(
        """
        import os
        import sys
        from pathlib import Path
        from unittest import mock

        from vision_workbench import exporter, confirmed
        from vision_workbench.exporter import ExportError
        from vision_workbench.store import DatasetStore

        workspace = Path(sys.argv[1])
        source_dir = Path(sys.argv[2])
        victim = Path(sys.argv[3]).resolve()
        target = Path(sys.argv[4])
        parent_inode = os.lstat(victim.parent).st_ino

        def swap_to_fifo() -> None:
            victim.unlink()
            os.mkfifo(victim)  # no writer: a blocking open hangs here

        real_inspect = confirmed.inspect_regular
        state = {"fired": False}

        def racing_inspect(path, *args, **kwargs):
            result = real_inspect(path, *args, **kwargs)
            dir_fd = kwargs.get("dir_fd")
            if (
                not state["fired"]
                and os.fspath(path) == victim.name
                and dir_fd is not None
                and os.fstat(dir_fd).st_ino == parent_inode
            ):
                state["fired"] = True
                swap_to_fifo()
            return result

        try:
            with mock.patch.object(
                exporter.confirmed, "inspect_regular",
                side_effect=racing_inspect,
            ):
                exporter.export_split(
                    DatasetStore(workspace), "baseline", target,
                    source_dir=source_dir,
                )
        except ExportError as error:
            print(f"error: {error}", file=sys.stderr)
            raise SystemExit(1)
        """
    )

    def _run(self) -> subprocess.CompletedProcess:
        self.add_sample("cat", b"planned-bytes")
        self.create_plan()
        moved = self.base / "moved"
        moved.mkdir()
        match = moved / "match.jpg"
        match.write_bytes(b"planned-bytes")
        victim = moved / "victim.dat"
        victim.write_bytes(b"unrelated")
        target = self.base / "out.zip"

        driver = self.base / "driver.py"
        driver.write_text(self.DRIVER, encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONPATH"] = REPO_ROOT
        proc = subprocess.run(
            [
                sys.executable, str(driver), str(self.workspace),
                str(moved), str(victim), str(target),
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=20,
        )
        return proc, victim, target

    def test_fifo_swap_fails_fast_without_waiting_for_writer(self) -> None:
        proc, victim, target = self._run()
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertEqual(proc.stdout, "")
        self.assertIn("error:", proc.stderr)
        self.assertIn(str(victim), proc.stderr)
        self.assertIn("named pipe", proc.stderr)
        self.assertIn("changed before it was read", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertFalse(target.exists())
        # The pipe is left as the failure found it: never opened or
        # removed by the export.
        self.assertTrue(stat.S_ISFIFO(os.lstat(victim).st_mode))


class FifoWithWriterIsNeverConsumedTest(ScanRaceHarness):
    def test_writer_offering_planned_bytes_is_never_read(self) -> None:
        digest = self.add_sample("cat", b"planned-bytes")
        self.create_plan()
        moved = self.base / "moved"
        moved.mkdir()
        # A regular file genuinely carrying the planned content exists,
        # but sorts after the victim, so accepting the pipe's bytes
        # would have let the pipe win the digest.
        match = moved / "zzz-match.jpg"
        match.write_bytes(b"planned-bytes")
        victim = moved / "aaa-victim.dat"
        victim.write_bytes(b"unrelated")
        target = self.base / "out.zip"

        wrote = threading.Event()
        stop = threading.Event()

        def serve_planned_bytes() -> None:
            # Hold both ends ourselves: then the bytes written stay in
            # the pipe until someone reads them, independent of the
            # exporter's transient non-blocking open.  The exporter
            # must reject after the fstat proof, before any read.
            read_end = os.open(victim, os.O_RDONLY | os.O_NONBLOCK)
            write_end = os.open(victim, os.O_WRONLY)
            os.write(write_end, b"planned-bytes")
            wrote.set()
            try:
                stop.wait(5.0)
            finally:
                os.close(write_end)
                os.close(read_end)

        writer: threading.Thread | None = None

        def swap_and_serve() -> None:
            nonlocal writer
            self.replace_with_fifo(victim)
            writer = threading.Thread(target=serve_planned_bytes, daemon=True)
            writer.start()

        with self.race_after_inspect(victim, swap_and_serve):
            with self.assertRaises(ExportError) as caught:
                self.export(moved, target)

        message = str(caught.exception)
        self.assertIn(str(victim), message)
        self.assertIn("named pipe", message)
        self.assertIn("changed before it was read", message)
        self.assertFalse(target.exists())

        self.assertTrue(wrote.wait(5.0))
        # The exact bytes the writer offered must still be sitting in
        # the pipe: a regression that streamed the pipe would have
        # consumed them (and hashed them as the planned digest).
        drain = os.open(victim, os.O_RDONLY | os.O_NONBLOCK)
        try:
            waiting = os.read(drain, 64)
        except BlockingIOError:
            waiting = b""
        finally:
            os.close(drain)
        self.assertEqual(
            waiting,
            b"planned-bytes",
            "the export consumed bytes from the swapped pipe",
        )
        stop.set()
        assert writer is not None
        writer.join(timeout=5)
        # The regular same-content copy and the workspace are untouched.
        self.assertEqual(match.read_bytes(), b"planned-bytes")
        self.assertTrue(stat.S_ISFIFO(os.lstat(victim).st_mode))
        self.assertIn(
            digest,
            (self.store.splits_directory / "baseline.json").read_text(),
        )


class IdenticalRegularFileSwapTest(ScanRaceHarness):
    def setUp(self) -> None:
        super().setUp()
        self.digest = self.add_sample("cat", b"abc")
        self.create_plan()
        self.moved = self.base / "moved"
        self.moved.mkdir()
        self.victim = self.moved / "aaa-victim.dat"
        self.victim.write_bytes(b"abc")
        # A same-content decoy that sorts after the victim: it must
        # never be silently used after the victim is rejected.
        self.decoy = self.moved / "zzz-decoy.jpg"
        self.decoy.write_bytes(b"abc")
        self.target = self.base / "out.zip"

    def test_identical_content_size_mtime_swap_is_rejected(self) -> None:
        with self.race_after_inspect(
            self.victim, lambda: self.replace_with_identical_file(self.victim)
        ):
            with self.assertRaises(ExportError) as caught:
                self.export(self.moved, self.target)
        message = str(caught.exception)
        self.assertIn(str(self.victim), message)
        self.assertIn("changed before it was read", message)
        self.assertIn("different regular file", message)
        self.assertFalse(self.target.exists())
        self.assertEqual(self.temp_leftovers(), [])
        self.assertTrue(self.decoy.exists())

    def test_unrelated_file_swap_also_fails(self) -> None:
        # The victim carries content the plan does not need at all; its
        # replacement must still abort the whole export.
        victim = self.moved / "aaa-unrelated.dat"
        victim.write_bytes(b"totally unrelated")
        with self.race_after_inspect(
            victim, lambda: self.replace_with_identical_file(victim)
        ):
            with self.assertRaises(ExportError) as caught:
                self.export(self.moved, self.target)
        self.assertIn(str(victim), str(caught.exception))
        self.assertFalse(self.target.exists())


class SymlinkSwapDuringScanTest(ScanRaceHarness):
    def test_symlink_swap_is_rejected_even_to_same_bytes(self) -> None:
        self.add_sample("cat", b"abc")
        self.create_plan()
        moved = self.base / "moved"
        moved.mkdir()
        victim = moved / "victim.dat"
        victim.write_bytes(b"unrelated")
        holder = self.base / "holder.jpg"
        holder.write_bytes(b"unrelated")
        match = moved / "match.jpg"
        match.write_bytes(b"abc")
        target = self.base / "out.zip"

        with self.race_after_inspect(
            victim, lambda: self.replace_with_symlink(victim, holder)
        ):
            with self.assertRaises(ExportError) as caught:
                self.export(moved, target)
        message = str(caught.exception)
        self.assertIn(str(victim), message)
        self.assertIn("symbolic link", message)
        self.assertIn("changed before it was read", message)
        self.assertFalse(target.exists())
        self.assertEqual(self.temp_leftovers(), [])


class UnrelatedChangeAfterAllSamplesFoundTest(ScanRaceHarness):
    def test_change_after_every_planned_sample_was_found_still_fails(self) -> None:
        self.add_sample("cat", b"m")
        self.create_plan()
        moved = self.base / "moved"
        # Directory "00-match" holds the only planned sample; directory
        # "99-junk" holds the unrelated victim that changes.  The walk
        # order is forced below so the match is hashed first.
        match_dir = moved / "00-match"
        match_dir.mkdir(parents=True)
        (match_dir / "match.jpg").write_bytes(b"m")
        junk_dir = moved / "99-junk"
        junk_dir.mkdir()
        victim = junk_dir / "victim.dat"
        victim.write_bytes(b"unrelated")
        target = self.base / "out.zip"

        match_hashed = threading.Event()
        real_hash = exporter_mod._hash_regular_file_at

        def tracking_hash(directory_fd, name, path, *args, **kwargs):
            digest, device, inode = real_hash(
                directory_fd, name, path, *args, **kwargs
            )
            if Path(path).name == "match.jpg":
                match_hashed.set()
            return digest, device, inode

        real_scandir = os.scandir

        class _OrderedEntries:
            def __init__(self, entries) -> None:
                self._entries = entries

            def __iter__(self):
                return iter(self._entries)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def ordered_scandir(path, *args, **kwargs):
            with real_scandir(path, *args, **kwargs) as entries:
                return _OrderedEntries(sorted(entries, key=lambda e: e.name))

        def mutate() -> None:
            # The exact requirement: the planned sample was already
            # found before the unrelated victim changed.
            self.assertTrue(
                match_hashed.is_set(),
                "test setup error: victim raced before the match was read",
            )
            self.replace_with_identical_file(victim)

        with (
            patch.object(
                exporter_mod, "_hash_regular_file_at", side_effect=tracking_hash
            ),
            patch.object(
                exporter_mod.os, "scandir", side_effect=ordered_scandir
            ),
            self.race_after_inspect(victim, mutate),
        ):
            with self.assertRaises(ExportError) as caught:
                self.export(moved, target)
        self.assertTrue(match_hashed.is_set())
        self.assertIn(str(victim), str(caught.exception))
        self.assertFalse(target.exists())
        self.assertEqual(self.temp_leftovers(), [])


class NonRegularAtFirstSightStillSkippedTest(ScanRaceHarness):
    def test_fifo_present_from_the_start_is_skipped(self) -> None:
        digest = self.add_sample("cat", b"abc")
        self.create_plan()
        moved = self.base / "moved"
        moved.mkdir()
        (moved / "match.jpg").write_bytes(b"abc")
        pipe = moved / "strange.pipe"
        os.mkfifo(pipe)  # present before the walk starts; no writer

        def serve_forever() -> None:
            # Blocks in open() until some reader opens the pipe.  If
            # the scan ever read the pipe this thread would finish;
            # while the entry is merely skipped it stays blocked.
            os.open(pipe, os.O_WRONLY)

        writer = threading.Thread(target=serve_forever, daemon=True)
        writer.start()
        result, target = self.export(moved)
        self.assertEqual(result["exported"], 1)
        writer.join(timeout=1)
        self.assertTrue(
            writer.is_alive(),
            "a pre-existing named pipe was opened instead of skipped",
        )
        with zipfile.ZipFile(target) as archive:
            names = [n for n in archive.namelist() if n.endswith(".jpg")]
            self.assertEqual(len(names), 1)
            self.assertEqual(archive.read(names[0]), b"abc")
        self.assertIn(digest, names[0])

    def test_symlink_present_from_the_start_is_skipped(self) -> None:
        self.add_sample("cat", b"abc")
        self.create_plan()
        moved = self.base / "moved"
        moved.mkdir()
        (moved / "match.jpg").write_bytes(b"abc")
        outside = self.base / "outside.jpg"
        outside.write_bytes(b"abc")
        (moved / "link.jpg").symlink_to(outside)
        result, target = self.export(moved)
        self.assertEqual(result["exported"], 1)
        with zipfile.ZipFile(target) as archive:
            names = [n for n in archive.namelist() if n.endswith(".jpg")]
            self.assertEqual(len(names), 1)
            self.assertEqual(archive.read(names[0]), b"abc")


class FailedExportLeavesEverythingUntouchedTest(ScanRaceHarness):
    def test_clean_abort_with_fifo_swap(self) -> None:
        self.add_sample("cat", b"abc")
        self.create_plan()
        moved = self.base / "moved"
        moved.mkdir()
        (moved / "match.jpg").write_bytes(b"abc")
        victim = moved / "victim.dat"
        victim.write_bytes(b"unrelated")
        existing = self.base / "existing.zip"
        existing.write_bytes(b"occupant-keep-me")

        manifest_before = self.store.manifest_path.read_bytes()
        plan_before = (
            self.store.splits_directory / "baseline.json"
        ).read_bytes()

        with self.race_after_inspect(victim, lambda: self.replace_with_fifo(victim)):
            with self.assertRaises(ExportError) as caught:
                self.export(moved, existing)

        message = str(caught.exception)
        self.assertIn(str(victim), message)
        self.assertIn("named pipe", message)
        self.assertEqual(existing.read_bytes(), b"occupant-keep-me")
        self.assertEqual(self.temp_leftovers(), [])
        self.assertEqual(self.store.manifest_path.read_bytes(), manifest_before)
        self.assertEqual(
            (self.store.splits_directory / "baseline.json").read_bytes(),
            plan_before,
        )
        self.assertTrue(stat.S_ISFIFO(os.lstat(victim).st_mode))


class ScanSwapCliTest(ScanRaceHarness):
    DRIVER = textwrap.dedent(
        """
        import os
        import sys
        from pathlib import Path
        from unittest import mock

        from vision_workbench import exporter, confirmed, __main__

        workspace = sys.argv[1]
        source_dir = sys.argv[2]
        victim = Path(sys.argv[3]).resolve()
        target = sys.argv[4]
        parent_inode = os.lstat(victim.parent).st_ino

        def swap_to_fifo() -> None:
            victim.unlink()
            os.mkfifo(victim)

        real_inspect = confirmed.inspect_regular
        state = {"fired": False}

        def racing_inspect(path, *args, **kwargs):
            result = real_inspect(path, *args, **kwargs)
            dir_fd = kwargs.get("dir_fd")
            if (
                not state["fired"]
                and os.fspath(path) == victim.name
                and dir_fd is not None
                and os.fstat(dir_fd).st_ino == parent_inode
            ):
                state["fired"] = True
                swap_to_fifo()
            return result

        sys.argv = [
            "vision-workbench", "export", workspace, "baseline", target,
            "--source-dir", source_dir,
        ]
        with mock.patch.object(
            exporter.confirmed, "inspect_regular", side_effect=racing_inspect
        ):
            raise SystemExit(__main__.main())
        """
    )

    def test_cli_fifo_swap_nonzero_empty_stdout_clear_stderr(self) -> None:
        self.add_sample("cat", b"abc")
        self.create_plan()
        moved = self.base / "moved"
        moved.mkdir()
        (moved / "match.jpg").write_bytes(b"abc")
        victim = moved / "victim.dat"
        victim.write_bytes(b"unrelated")
        target = self.base / "out.zip"
        driver = self.base / "driver.py"
        driver.write_text(self.DRIVER, encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONPATH"] = REPO_ROOT

        proc = subprocess.run(
            [
                sys.executable, str(driver), str(self.workspace),
                str(moved), str(victim), str(target),
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=20,
        )
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertEqual(proc.stdout, "")
        self.assertIn("error:", proc.stderr)
        self.assertIn(str(victim), proc.stderr)
        self.assertIn("named pipe", proc.stderr)
        self.assertIn("changed before it was read", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertFalse(target.exists())
        self.assertTrue(stat.S_ISFIFO(os.lstat(victim).st_mode))


if __name__ == "__main__":
    unittest.main()
