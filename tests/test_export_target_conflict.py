"""Regression tests for target-name races during export.

An export rejects a target that exists when it starts, but the export can
take a long time (images are copied and compressed); another program may
create a file or symlink at the target name while that work happens.
These tests pin the rule that the target name is owned only at the very
end, once the package is complete and the source checks have passed:

* an occupant appearing at any point up to the final claim fails the
  export as a target-already-exists conflict naming the user's path;
* the occupant is preserved exactly, whatever it is — a foreign file, an
  empty file, a byte-identical copy, or a symlink (even dangling);
* this run's temporary package is removed, the occupant is never moved
  or deleted, and the workspace, plan and label history are untouched;
* the command exits non-zero with no success payload on stdout;
* concurrent exports to one path still have at most one winner.

The rule applies to recorded-source exports, ``--source-dir`` exports,
empty plans and plans whose samples are all skipped.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from vision_workbench.exporter import ExportError, export_split
from vision_workbench.store import DatasetStore
import vision_workbench.exporter as exporter_mod

REPO_ROOT = Path(__file__).resolve().parents[1]


class TargetConflictHarness(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "dataset"
        self.store = DatasetStore(self.root)
        self.store.initialize()
        self._serial = 0

    def tearDown(self) -> None:
        self.temp.cleanup()

    def add_sample(self, label: str | None, content: bytes | None = None) -> str:
        self._serial += 1
        path = self.root.parent / f"sample-{self._serial}.jpg"
        path.write_bytes(
            content if content is not None else f"content-{self._serial}".encode()
        )
        result = self.store.add(path, label)
        self.assertTrue(result.added)
        return result.digest

    def create_plan(self, name: str = "baseline", seed: int = 42,
                    ratios: list[str] | None = None) -> dict:
        return self.store.create_split(
            name, seed, ratios or ["0.5", "0.25", "0.25"]
        ).plan

    def target(self) -> Path:
        return self.root.parent / "out.zip"

    def leftover_temps(self, target: Path) -> list[Path]:
        # This run's temp files use mkstemp with this prefix/suffix in the
        # target's directory; the persistent lock file is not a temp.
        return [
            path
            for path in target.parent.iterdir()
            if path.name.startswith(f".{target.name}.") and path.name.endswith(".tmp")
        ]

    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "vision_workbench", *arguments],
            cwd=REPO_ROOT, text=True, capture_output=True, check=False,
        )


def _during_generation(occupy):
    """Patch the package writer so ``occupy`` runs while the export builds.

    The real write (streaming every image into the temp package) still
    happens; the callback fires after the start-of-export target check
    and before the finished package is claimed, modelling another
    program taking the name while the export is in progress.
    """
    real_write_zip = exporter_mod._write_zip

    def wrapped(stream, *args, **kwargs):
        occupy()
        return real_write_zip(stream, *args, **kwargs)

    return patch.object(exporter_mod, "_write_zip", wrapped)


def _at_last_moment(occupy):
    """Patch the writer so ``occupy`` runs after the package is fully written.

    The callback fires once the whole package has been streamed and
    fsynced (and, for source-dir exports, immediately before the final
    source-tree verification and the claim) — the latest point at which
    another program can still take the name.
    """
    real_write_zip = exporter_mod._write_zip

    def wrapped(stream, *args, **kwargs):
        result = real_write_zip(stream, *args, **kwargs)
        occupy()
        return result

    return patch.object(exporter_mod, "_write_zip", wrapped)


class ExistingTargetTest(TargetConflictHarness):
    def test_existing_file_rejected_preserved_and_named(self) -> None:
        self.add_sample("cat", b"abc")
        self.create_plan()
        target = self.target()
        target.write_bytes(b"foreign content")

        with self.assertRaises(ExportError) as caught:
            export_split(self.store, "baseline", target)
        message = str(caught.exception)
        self.assertIn("target already exists", message)
        self.assertIn(str(target), message)
        # The foreign file and its type are untouched.
        self.assertTrue(target.is_file())
        self.assertEqual(target.read_bytes(), b"foreign content")
        self.assertEqual(self.leftover_temps(target), [])

    def test_dangling_symlink_at_start_rejected_and_link_kept(self) -> None:
        self.add_sample("cat", b"abc")
        self.create_plan()
        target = self.target()
        os.symlink(self.root.parent / "does-not-exist", target)

        with self.assertRaises(ExportError) as caught:
            export_split(self.store, "baseline", target)
        self.assertIn("target already exists", str(caught.exception))
        self.assertIn(str(target), str(caught.exception))
        # The link itself survives and still dangles.
        self.assertTrue(target.is_symlink())
        self.assertEqual(os.readlink(target), str(self.root.parent / "does-not-exist"))
        self.assertFalse(target.exists())
        self.assertEqual(self.leftover_temps(target), [])

    def test_symlink_to_file_at_start_rejected_and_target_kept(self) -> None:
        self.add_sample("cat", b"abc")
        self.create_plan()
        holder = self.root.parent / "keepers-file.bin"
        holder.write_bytes(b"keepers bytes")
        target = self.target()
        os.symlink(holder, target)

        with self.assertRaises(ExportError):
            export_split(self.store, "baseline", target)
        # The link and the file it points at are both preserved.
        self.assertTrue(target.is_symlink())
        self.assertEqual(os.readlink(target), str(holder))
        self.assertEqual(target.read_bytes(), b"keepers bytes")
        self.assertEqual(holder.read_bytes(), b"keepers bytes")
        self.assertEqual(self.leftover_temps(target), [])

    def test_cli_conflict_is_nonzero_with_empty_stdout(self) -> None:
        self.add_sample("cat", b"abc")
        self.create_plan()
        target = self.target()
        os.symlink(self.root.parent / "missing", target)

        result = self.run_cli(
            "export", str(self.root), "baseline", str(target)
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("target already exists", result.stderr)
        # The exact path the user passed must appear in the message.
        self.assertIn(str(target), result.stderr)
        self.assertTrue(target.is_symlink())


class RaceDuringExportTest(TargetConflictHarness):
    def test_file_created_during_copy_fails_and_is_preserved(self) -> None:
        self.add_sample("cat", b"abc")
        self.add_sample("dog", b"def")
        self.create_plan()
        target = self.target()

        with _during_generation(lambda: target.write_bytes(b"foreign file")):
            with self.assertRaises(ExportError) as caught:
                export_split(self.store, "baseline", target)

        self.assertIn("target already exists", str(caught.exception))
        self.assertIn(str(target), str(caught.exception))
        self.assertTrue(target.is_file())
        self.assertEqual(target.read_bytes(), b"foreign file")
        self.assertEqual(self.leftover_temps(target), [])

    def test_empty_file_created_during_copy_still_conflicts(self) -> None:
        self.add_sample("cat", b"abc")
        self.create_plan()
        target = self.target()

        with _during_generation(lambda: target.write_bytes(b"")):
            with self.assertRaises(ExportError):
                export_split(self.store, "baseline", target)

        self.assertTrue(target.is_file())
        self.assertEqual(target.read_bytes(), b"")
        self.assertEqual(self.leftover_temps(target), [])

    def test_byte_identical_occupant_still_conflicts_not_success(self) -> None:
        self.add_sample("cat", b"abc")
        self.create_plan()
        # What the package bytes would be, from an unconstrained export.
        reference = self.root.parent / "reference.zip"
        export_split(self.store, "baseline", reference)
        package_bytes = reference.read_bytes()

        target = self.target()
        with _during_generation(lambda: target.write_bytes(package_bytes)):
            with self.assertRaises(ExportError) as caught:
                export_split(self.store, "baseline", target)

        self.assertIn("target already exists", str(caught.exception))
        self.assertEqual(target.read_bytes(), package_bytes)
        self.assertEqual(self.leftover_temps(target), [])

    def test_dangling_link_created_during_copy_kept_as_a_link(self) -> None:
        self.add_sample("cat", b"abc")
        self.create_plan()
        target = self.target()

        def occupy() -> None:
            os.symlink("/nonexistent/path", target)

        with _during_generation(occupy):
            with self.assertRaises(ExportError) as caught:
                export_split(self.store, "baseline", target)

        self.assertIn("target already exists", str(caught.exception))
        self.assertIn(str(target), str(caught.exception))
        self.assertTrue(target.is_symlink())
        self.assertEqual(os.readlink(target), "/nonexistent/path")
        self.assertFalse(target.exists())
        self.assertEqual(self.leftover_temps(target), [])

    def test_link_to_other_file_created_during_copy_kept(self) -> None:
        self.add_sample("cat", b"abc")
        self.create_plan()
        holder = self.root.parent / "other.png"
        holder.write_bytes(b"other image bytes")
        target = self.target()

        with _during_generation(lambda: os.symlink(holder, target)):
            with self.assertRaises(ExportError):
                export_split(self.store, "baseline", target)

        self.assertTrue(target.is_symlink())
        self.assertEqual(os.readlink(target), str(holder))
        self.assertEqual(target.read_bytes(), b"other image bytes")
        self.assertEqual(self.leftover_temps(target), [])

    def test_occupant_appearing_at_last_moment_is_preserved(self) -> None:
        self.add_sample("cat", b"abc")
        self.create_plan()
        target = self.target()

        with _at_last_moment(lambda: target.write_bytes(b"last second")):
            with self.assertRaises(ExportError) as caught:
                export_split(self.store, "baseline", target)

        self.assertIn("target already exists", str(caught.exception))
        self.assertEqual(target.read_bytes(), b"last second")
        self.assertEqual(self.leftover_temps(target), [])

    def test_claim_refused_between_check_and_link(self) -> None:
        # Drive the atomic claim itself: another program takes the name
        # in the tiny gap after the pre-claim check but inside link(2).
        # The kernel refuses the link with EEXIST, so this must surface
        # as a conflict and leave the new dangling link exactly as made.
        self.add_sample("cat", b"abc")
        self.create_plan()
        target = self.target()
        real_link = os.link

        def racing_link(src, dst):
            if not os.path.lexists(dst):
                os.symlink("/appears/at/link", dst)
            return real_link(src, dst)

        with patch.object(exporter_mod.os, "link", side_effect=racing_link):
            with self.assertRaises(ExportError) as caught:
                export_split(self.store, "baseline", target)

        self.assertIn("target already exists", str(caught.exception))
        self.assertTrue(target.is_symlink())
        self.assertEqual(os.readlink(target), "/appears/at/link")
        self.assertEqual(self.leftover_temps(target), [])

    def test_failed_conflict_leaves_workspace_plan_and_history_untouched(self) -> None:
        digest = self.add_sample("cat", b"abc")
        self.create_plan()
        self.store.submit_batch(
            {"batch": "b1", "changes": [{"sha256": digest, "old": "cat", "new": "kitten"}]}
        )
        manifest_before = self.store.manifest_path.read_bytes()
        plans_before = sorted(p.name for p in self.store.splits_directory.iterdir())
        history_before = [b["batch"] for b in self.store.history()]

        target = self.target()
        with _during_generation(lambda: target.write_bytes(b"x")):
            with self.assertRaises(ExportError):
                export_split(self.store, "baseline", target)

        self.assertEqual(self.store.manifest_path.read_bytes(), manifest_before)
        self.assertEqual(
            sorted(p.name for p in self.store.splits_directory.iterdir()), plans_before
        )
        self.assertEqual(
            [b["batch"] for b in self.store.history()], history_before
        )

    def test_retry_after_conflict_cleared_succeeds_with_same_package(self) -> None:
        self.add_sample("cat", b"abc")
        self.create_plan()
        reference = self.root.parent / "reference.zip"
        export_split(self.store, "baseline", reference)

        target = self.target()
        with _during_generation(lambda: target.write_bytes(b"blocking")):
            with self.assertRaises(ExportError):
                export_split(self.store, "baseline", target)
        target.unlink()

        result = export_split(self.store, "baseline", target)
        self.assertEqual(result["exported"], 1)
        self.assertEqual(target.read_bytes(), reference.read_bytes())
        with zipfile.ZipFile(target) as archive:
            self.assertIn("manifest.json", archive.namelist())


class EmptyAndSkippedConflictTest(TargetConflictHarness):
    def test_empty_plan_start_conflict_keeps_dangling_link(self) -> None:
        self.create_plan("empty")
        target = self.target()
        os.symlink("/nowhere", target)

        with self.assertRaises(ExportError):
            export_split(self.store, "empty", target)
        self.assertTrue(target.is_symlink())
        self.assertEqual(os.readlink(target), "/nowhere")
        self.assertEqual(self.leftover_temps(target), [])

    def test_empty_plan_race_during_generation_conflicts(self) -> None:
        self.create_plan("empty")
        target = self.target()
        with _during_generation(lambda: target.write_bytes(b"taken")):
            with self.assertRaises(ExportError) as caught:
                export_split(self.store, "empty", target)
        self.assertIn("target already exists", str(caught.exception))
        self.assertEqual(target.read_bytes(), b"taken")
        self.assertEqual(self.leftover_temps(target), [])

    def test_all_skipped_samples_race_conflicts(self) -> None:
        self.add_sample(None, b"unlabeled-bytes")
        self.create_plan(ratios=["1", "0", "0"])
        target = self.target()

        with _during_generation(lambda: os.symlink("/dangling", target)):
            with self.assertRaises(ExportError):
                export_split(
                    self.store, "baseline", target, skip_unlabeled=True
                )

        self.assertTrue(target.is_symlink())
        self.assertEqual(os.readlink(target), "/dangling")
        self.assertEqual(self.leftover_temps(target), [])

    def test_empty_plan_without_conflict_still_publishes(self) -> None:
        self.create_plan("empty")
        target = self.target()
        result = export_split(self.store, "empty", target)
        self.assertEqual(result["exported"], 0)
        with zipfile.ZipFile(target) as archive:
            self.assertEqual(json.loads(archive.read("manifest.json"))["samples"], [])


class SourceDirConflictTest(TargetConflictHarness):
    def test_occupant_during_source_dir_copy_preserved(self) -> None:
        self.add_sample("cat", b"cat-bytes")
        self.create_plan()
        # Move the image into a source tree and drop the original path.
        moved = self.root.parent / "moved"
        moved.mkdir()
        located = moved / "renamed.dat"
        located.write_bytes(b"cat-bytes")
        original = Path(
            self.store.get_split("baseline")["sets"]["train"]["members"][0]["source"]
        )
        # The member may live in another set; just remove every recorded
        # source so only --source-dir can satisfy the export.
        for payload in self.store.get_split("baseline")["sets"].values():
            for member in payload["members"]:
                Path(member["source"]).unlink(missing_ok=True)
        self.assertFalse(original.exists())

        target = self.target()
        with _during_generation(lambda: target.write_bytes(b"not the zip")):
            with self.assertRaises(ExportError) as caught:
                export_split(
                    self.store, "baseline", target, source_dir=moved
                )

        self.assertIn("target already exists", str(caught.exception))
        self.assertEqual(target.read_bytes(), b"not the zip")
        # The migrated tree and its file are untouched.
        self.assertEqual(located.read_bytes(), b"cat-bytes")
        self.assertEqual(self.leftover_temps(target), [])

    def test_occupant_after_source_verification_gate_preserved(self) -> None:
        self.add_sample("cat", b"cat-bytes")
        self.create_plan()
        moved = self.root.parent / "moved"
        moved.mkdir()
        (moved / "f.bin").write_bytes(b"cat-bytes")
        for payload in self.store.get_split("baseline")["sets"].values():
            for member in payload["members"]:
                Path(member["source"]).unlink(missing_ok=True)

        target = self.target()
        with _at_last_moment(lambda: os.symlink("/late-dangling", target)):
            with self.assertRaises(ExportError):
                export_split(
                    self.store, "baseline", target, source_dir=moved
                )
        self.assertTrue(target.is_symlink())
        self.assertEqual(os.readlink(target), "/late-dangling")
        self.assertEqual(self.leftover_temps(target), [])


class ConcurrentWorkspacesTest(TargetConflictHarness):
    def test_separate_workstations_same_target_at_most_one_wins(self) -> None:
        target = self.root.parent / "shared.zip"
        workspaces = []
        for index in range(4):
            root = self.root.parent / f"station-{index}"
            station = DatasetStore(root)
            station.initialize()
            source = self.root.parent / f"station-source-{index}.jpg"
            source.write_bytes(f"station-{index}-content".encode())
            station.add(source, f"label-{index}")
            station.create_split("baseline", 1, ["1", "0", "0"])
            workspaces.append(root)

        cli = [
            sys.executable, "-m", "vision_workbench", "export",
        ]
        commands = [
            cli + [str(root), "baseline", str(target)] for root in workspaces
        ]
        processes = [
            subprocess.Popen(
                command, cwd=REPO_ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            for command in commands
        ]
        outcomes = [process.communicate() for process in processes]
        returncodes = [process.returncode for process in processes]

        winners = [code for code in returncodes if code == 0]
        self.assertEqual(len(winners), 1, returncodes)
        for stdout, stderr in outcomes:
            if stdout:
                # Exactly the winner printed a result; losers said
                # nothing on stdout and named the occupied target.
                body = json.loads(stdout)
                self.assertEqual(body["exported"], 1)
            else:
                self.assertIn("target already exists", stderr)
                self.assertIn(str(target), stderr)

        self.assertTrue(target.is_file())
        with zipfile.ZipFile(target) as archive:
            self.assertIn("manifest.json", archive.namelist())
        # No losing export left a temp package behind.
        self.assertEqual(self.leftover_temps(target), [])


if __name__ == "__main__":
    unittest.main()
