"""Regression coverage for exporting a saved plan from its recorded sources.

Export without ``--source-dir`` copies each sample from the source path
saved in the split plan.  The bytes actually read must come from the one
ordinary regular file confirmed immediately before the read: the path is
inspected, then opened with ``O_NOFOLLOW`` and the descriptor is
fstat()ed and proved to be that exact object.

These tests pin the outcomes when those two steps come apart:

* another regular file — even one with identical content, size and
  modification time — renamed onto the recorded path in the gap is
  rejected as a replaced source, not accepted because the final digest
  happens to match;
* a named pipe swapped into the same gap — even one with no writer —
  fails immediately and explicitly: the export does not wait for data
  and the pipe's content can never enter the package;
* a symbolic link present at the start keeps the long-standing
  behaviour: the link is followed once and the regular file it names is
  the confirmed object for this read.

A failure aborts the whole export: earlier samples already streamed into
the temporary package are discarded with it, the target name is never
published, an existing target keeps its bytes, and the workspace, the
saved plan and the source files are left exactly as they were.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
import zipfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from vision_workbench.splits import SET_NAMES
from vision_workbench.store import DatasetStore
import vision_workbench.exporter as exporter_mod
from vision_workbench.exporter import ExportError, export_split

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

    def add_sample(self, label: str | None, content: bytes) -> str:
        self._serial += 1
        path = self.base / f"sample-{self._serial}.jpg"
        path.write_bytes(content)
        result = self.store.add(path, label)
        self.assertTrue(result.added)
        return result.digest

    def create_plan(self, name: str = "baseline",
                    ratios: list[str] | None = None) -> dict:
        return self.store.create_split(
            name, 42, ratios or ["0.5", "0.25", "0.25"]
        ).plan

    def source_for(self, digest: str, plan: str = "baseline") -> Path:
        payload = self.store.get_split(plan)
        for set_name in SET_NAMES:
            for member in payload["sets"][set_name]["members"]:
                if member["sha256"] == digest:
                    return Path(member["source"])
        raise AssertionError(f"digest {digest} not present in plan {plan}")

    def temp_leftovers(self) -> list[str]:
        return [
            entry.name
            for entry in self.base.iterdir()
            if entry.name.endswith(".tmp")
        ]

    @contextmanager
    def race_after_inspect(self, recorded: Path, mutate):
        """Run ``mutate`` after the recorded source is inspected, pre-open.

        The wrapper sits on the shared confirmation rule, so the swap
        lands strictly between the no-follow lstat that confirmed the
        regular file and the ``O_NOFOLLOW`` open that must open it.
        """
        recorded = Path(recorded).resolve()
        real_inspect = exporter_mod.confirmed.inspect_regular
        state = {"fired": False}

        def racing_inspect(path, *args, **kwargs):
            result = real_inspect(path, *args, **kwargs)
            if not state["fired"] and Path(os.fspath(path)) == recorded:
                state["fired"] = True
                mutate()
            return result

        with patch.object(
            exporter_mod.confirmed,
            "inspect_regular",
            side_effect=racing_inspect,
        ):
            yield state


class SymlinkAtStartKeepsWorkingTest(RecordedSourceHarness):
    def test_symlink_at_start_exports_target_byte_identically(self) -> None:
        digest = self.add_sample("cat", b"cat-one")
        self.create_plan()
        recorded = self.source_for(digest)

        normal = self.base / "normal.zip"
        export_split(self.store, "baseline", normal)

        # The recorded path itself becomes a symlink (to a fresh,
        # byte-identical regular file with its own inode) before export.
        moved = self.base / "moved-copy.jpg"
        moved.write_bytes(b"cat-one")
        recorded.unlink()
        recorded.symlink_to(moved)

        linked = self.base / "linked.zip"
        result = export_split(self.store, "baseline", linked)
        self.assertEqual(result["exported"], 1)
        self.assertEqual(normal.read_bytes(), linked.read_bytes())
        with zipfile.ZipFile(linked) as archive:
            name = next(n for n in archive.namelist() if n.endswith(".jpg"))
            self.assertEqual(archive.read(name), b"cat-one")

    def test_symlink_at_start_to_changed_content_fails_digest_check(self) -> None:
        digest = self.add_sample("cat", b"cat-one")
        self.create_plan()
        recorded = self.source_for(digest)

        moved = self.base / "moved-copy.jpg"
        moved.write_bytes(b"totally different bytes here")
        recorded.unlink()
        recorded.symlink_to(moved)

        target = self.base / "out.zip"
        with self.assertRaises(ExportError) as caught:
            export_split(self.store, "baseline", target)
        message = str(caught.exception)
        self.assertIn(digest, message)
        self.assertIn(str(recorded), message)
        self.assertIn("digest", message)
        self.assertFalse(target.exists())

    def test_plain_export_bytes_unchanged_for_normal_file(self) -> None:
        self.add_sample("cat", b"cat-one")
        self.create_plan()
        first = self.base / "first.zip"
        second = self.base / "second.zip"
        export_split(self.store, "baseline", first)
        export_split(self.store, "baseline", second)
        self.assertEqual(first.read_bytes(), second.read_bytes())


class NonRegularAtStartTest(RecordedSourceHarness):
    def test_fifo_at_start_fails_naming_digest_path_and_reason(self) -> None:
        digest = self.add_sample("cat", b"abc")
        self.create_plan()
        recorded = self.source_for(digest)
        recorded.unlink()
        os.mkfifo(recorded)  # no writer anywhere; inspection must reject it

        target = self.base / "out.zip"
        with self.assertRaises(ExportError) as caught:
            export_split(self.store, "baseline", target)
        message = str(caught.exception)
        self.assertIn(digest, message)
        self.assertIn(str(recorded), message)
        self.assertIn("not a regular file", message)
        self.assertIn("named pipe", message)
        self.assertFalse(target.exists())
        self.assertEqual(self.temp_leftovers(), [])
        # The pipe is left as the failure found it, never unlinked.
        self.assertTrue(stat.S_ISFIFO(os.lstat(recorded).st_mode))

    def test_directory_at_start_fails_not_regular(self) -> None:
        digest = self.add_sample("cat", b"abc")
        self.create_plan()
        recorded = self.source_for(digest)
        recorded.unlink()
        recorded.mkdir()

        target = self.base / "out.zip"
        with self.assertRaises(ExportError) as caught:
            export_split(self.store, "baseline", target)
        message = str(caught.exception)
        self.assertIn(digest, message)
        self.assertIn("not a regular file", message)
        self.assertFalse(target.exists())


class ReplacedBetweenInspectAndOpenTest(RecordedSourceHarness):
    def setUp(self) -> None:
        super().setUp()
        self.digest = self.add_sample("cat", b"abc")
        self.create_plan()
        self.recorded = self.source_for(self.digest)
        self.target = self.base / "out.zip"

    def _assert_clean_failure(self, caught: Exception, *words: str) -> None:
        message = str(caught.exception)
        self.assertIn(self.digest, message)
        self.assertRegex(message, r"[0-9a-f]{64}")
        self.assertIn(str(self.recorded), message)
        for word in words:
            self.assertIn(word, message)
        self.assertFalse(self.target.exists())
        self.assertEqual(self.temp_leftovers(), [])

    def _identical_replacement(self) -> None:
        # Match content, size and (as far as possible) mtime, but be a
        # different object (different inode) renamed onto the path.
        replacement = self.base / "swap.jpg"
        replacement.write_bytes(b"abc")
        stat_before = os.lstat(self.recorded)
        os.utime(
            replacement,
            ns=(stat_before.st_atime_ns, stat_before.st_mtime_ns),
        )
        os.replace(replacement, self.recorded)

    def test_identical_regular_file_swap_is_rejected(self) -> None:
        with self.race_after_inspect(self.recorded, self._identical_replacement):
            with self.assertRaises(ExportError) as caught:
                export_split(self.store, "baseline", self.target)
        self._assert_clean_failure(
            caught, "replaced before it was read", "different regular file"
        )

    def test_symlink_swap_is_rejected_even_pointing_at_same_bytes(self) -> None:
        holder = self.base / "holder.jpg"
        holder.write_bytes(b"abc")

        def make_symlink() -> None:
            self.recorded.unlink()
            self.recorded.symlink_to(holder)

        with self.race_after_inspect(self.recorded, make_symlink):
            with self.assertRaises(ExportError) as caught:
                export_split(self.store, "baseline", self.target)
        self._assert_clean_failure(
            caught, "replaced before it was read", "symbolic link"
        )

    def test_rejected_swap_does_not_hunt_another_copy(self) -> None:
        # A second, still-readable file carries the very same content, but
        # the failed source must never be substituted by anything else.
        alternate = self.base / "alternate.jpg"
        alternate.write_bytes(b"abc")
        with self.race_after_inspect(self.recorded, self._identical_replacement):
            with self.assertRaises(ExportError):
                export_split(self.store, "baseline", self.target)
        self.assertFalse(self.target.exists())
        self.assertTrue(alternate.exists())

    def test_workspace_plan_and_sources_are_untouched_after_swap_failure(self) -> None:
        manifest_before = self.store.manifest_path.read_bytes()
        plan_before = (self.store.splits_directory / "baseline.json").read_bytes()
        with self.race_after_inspect(self.recorded, self._identical_replacement):
            with self.assertRaises(ExportError):
                export_split(self.store, "baseline", self.target)
        self.assertEqual(self.store.manifest_path.read_bytes(), manifest_before)
        self.assertEqual(
            (self.store.splits_directory / "baseline.json").read_bytes(),
            plan_before,
        )
        # The swapped object is what remains at the recorded path; the
        # exporter neither restores nor deletes nor rewrites it, and the
        # plan still records the original path string.
        self.assertEqual(self.source_for(self.digest), self.recorded)

    def test_whole_export_aborts_after_an_earlier_sample_was_streamed(self) -> None:
        # Two labelled samples share one set/class, so their archive
        # entries are ordered purely by digest; race the last-sorted one,
        # guaranteeing at least one earlier entry was already streamed
        # into this run's temporary package before the failure.
        digests = {
            self.add_sample("cat", b"first-bytes"): b"first-bytes",
            self.add_sample("cat", b"second-bytes"): b"second-bytes",
        }
        self.create_plan("multi", ratios=["1", "0", "0"])
        last_digest = max(digests)
        last_source = self.source_for(last_digest, plan="multi")

        target = self.base / "out.zip"
        manifest_before = self.store.manifest_path.read_bytes()
        plan_before = (self.store.splits_directory / "multi.json").read_bytes()

        streamed = []
        real_write_sample = exporter_mod._write_sample

        def tracking_write_sample(archive, arc_name, member, *args, **kwargs):
            real_write_sample(archive, arc_name, member, *args, **kwargs)
            streamed.append(member["sha256"])

        def replace_last() -> None:
            # Prove the ordering assumption the test relies on: by the
            # time the last entry is opened, the other entry is done.
            self.assertNotEqual(streamed, [])
            swap = self.base / "swap-last.jpg"
            swap.write_bytes(digests[last_digest])
            os.replace(swap, last_source)

        with (
            self.race_after_inspect(last_source, replace_last),
            patch.object(
                exporter_mod, "_write_sample", side_effect=tracking_write_sample
            ),
        ):
            with self.assertRaises(ExportError) as caught:
                export_split(self.store, "multi", target)

        message = str(caught.exception)
        self.assertIn(last_digest, message)
        self.assertIn(str(last_source), message)
        self.assertIn("replaced before it was read", message)
        # The earlier entry really had been copied before the abort.
        self.assertTrue(any(d != last_digest for d in streamed))

        # The temporary package is gone and no target was ever published.
        self.assertFalse(target.exists())
        self.assertEqual(self.temp_leftovers(), [])
        self.assertEqual(self.store.manifest_path.read_bytes(), manifest_before)
        self.assertEqual(
            (self.store.splits_directory / "multi.json").read_bytes(),
            plan_before,
        )

    def test_existing_target_occupant_survives_a_failed_export(self) -> None:
        # The conflict rule is unchanged for recorded-source exports: a
        # target already present when the export starts is refused and its
        # bytes are preserved, regardless of what the sources then do.
        digest = self.add_sample("dog", b"dog-bytes")
        self.create_plan("occupied")
        occupant = self.base / "existing.zip"
        occupant.write_bytes(b"occupant-keep-me")
        with self.assertRaises(ExportError) as caught:
            export_split(self.store, "occupied", occupant)
        self.assertIn("target already exists", str(caught.exception))
        self.assertEqual(occupant.read_bytes(), b"occupant-keep-me")
        self.assertEqual(self.temp_leftovers(), [])
        self.assertTrue(self.source_for(digest, plan="occupied").exists())


class FifoSwapMustNotWaitTest(RecordedSourceHarness):
    """The FIFO-in-the-gap case is run in a subprocess.

    A regression that opens the swapped named pipe with plain blocking
    semantics would otherwise hang this whole test process waiting for a
    writer that never comes; the subprocess timeout turns that hang into
    an ordinary, fast failure instead.
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
        recorded = Path(sys.argv[2]).resolve()
        target = Path(sys.argv[3])

        def swap_to_fifo() -> None:
            recorded.unlink()
            os.mkfifo(recorded)  # no writer: a blocking open would hang here

        real_inspect = confirmed.inspect_regular
        state = {"fired": False}

        def racing_inspect(path, *args, **kwargs):
            result = real_inspect(path, *args, **kwargs)
            if not state["fired"] and Path(os.fspath(path)) == recorded:
                state["fired"] = True
                swap_to_fifo()
            return result

        try:
            with mock.patch.object(
                exporter.confirmed, "inspect_regular", side_effect=racing_inspect
            ):
                exporter.export_split(DatasetStore(workspace), "baseline", target)
        except ExportError as error:
            print(f"error: {error}", file=sys.stderr)
            raise SystemExit(1)
        """
    )

    def _run_driver(self) -> subprocess.CompletedProcess:
        digest = self.add_sample("cat", b"abc")
        self.create_plan()
        recorded = self.source_for(digest)
        target = self.base / "out.zip"
        driver = self.base / "driver.py"
        driver.write_text(self.DRIVER, encoding="utf-8")
        env = dict(os.environ)
        repo = str(REPO_ROOT)
        env["PYTHONPATH"] = (
            repo + os.pathsep + env["PYTHONPATH"]
            if env.get("PYTHONPATH")
            else repo
        )
        proc = subprocess.run(
            [sys.executable, str(driver), str(self.root), str(recorded), str(target)],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=20,
        )
        return proc, digest, recorded, target

    def test_fifo_swap_between_inspect_and_open_fails_without_waiting(self) -> None:
        proc, digest, recorded, target = self._run_driver()
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertEqual(proc.stdout, "")
        self.assertIn("error:", proc.stderr)
        self.assertIn(digest, proc.stderr)
        self.assertIn(str(recorded), proc.stderr)
        self.assertIn("named pipe", proc.stderr)
        self.assertIn("replaced before it was read", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertFalse(target.exists())
        # The pipe the attacker put there is never consumed or removed.
        self.assertTrue(stat.S_ISFIFO(os.lstat(recorded).st_mode))


class RecordedSourceCliTest(RecordedSourceHarness):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "vision_workbench", *arguments],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_cli_fifo_source_nonzero_empty_stdout_no_traceback(self) -> None:
        digest = self.add_sample("cat", b"abc")
        self.create_plan()
        recorded = self.source_for(digest)
        recorded.unlink()
        os.mkfifo(recorded)
        target = self.base / "out.zip"

        result = self.run_cli(
            "export", str(self.root), "baseline", str(target)
        )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn("error:", result.stderr)
        self.assertIn(digest, result.stderr)
        self.assertIn(str(recorded), result.stderr)
        self.assertIn("named pipe", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse(target.exists())

    def test_cli_missing_recorded_source_nonzero_empty_stdout(self) -> None:
        digest = self.add_sample("cat", b"abc")
        self.create_plan()
        recorded = self.source_for(digest)
        recorded.unlink()
        target = self.base / "out.zip"

        result = self.run_cli(
            "export", str(self.root), "baseline", str(target)
        )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn(digest, result.stderr)
        self.assertIn(str(recorded), result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
