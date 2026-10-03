from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
import zipfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from vision_workbench.exporter import (
    ExportError,
    _scan_source_directory,
    _write_sample,
    export_split,
)
from vision_workbench.store import DatasetStore
import vision_workbench.exporter as exporter_mod

REPO_ROOT = Path(__file__).resolve().parents[1]


class SourceDirHarness(unittest.TestCase):
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

    def create_plan(
        self, name: str = "baseline", seed: int = 42, ratios: list[str] | None = None
    ) -> dict:
        result = self.store.create_split(
            name, seed, ratios or ["0.5", "0.25", "0.25"]
        )
        return result.plan

    def export(
        self,
        plan: str = "baseline",
        target: Path | None = None,
        skip_unlabeled: bool = False,
        source_dir: Path | None = None,
    ) -> tuple[dict, Path]:
        target = target or self.root.parent / "out.zip"
        result = export_split(
            self.store,
            plan,
            target,
            skip_unlabeled=skip_unlabeled,
            source_dir=source_dir,
        )
        return result, target

    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "vision_workbench", *arguments],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )


class SourceDirExportTest(SourceDirHarness):
    def test_export_from_moved_tree_byte_identical(self) -> None:
        d1 = self.add_sample("cat", b"cat-one")
        d2 = self.add_sample("dog", b"dog-two")
        self.create_plan()
        normal = self.root.parent / "normal.zip"
        self.export(target=normal)

        # Move the files to a new tree with new names and extensions, then
        # delete the originals: the recorded source paths no longer exist.
        moved = self.root.parent / "moved"
        moved.mkdir()
        (moved / "renamed.cat").write_bytes(b"cat-one")
        (moved / "sub").mkdir()
        (moved / "sub" / "dog.unknown").write_bytes(b"dog-two")
        (moved / "junk.txt").write_bytes(b"unrelated")
        for child in self.root.parent.iterdir():
            if child.name.startswith("sample-") and child.suffix == ".jpg":
                child.unlink()

        via_dir = self.root.parent / "via-dir.zip"
        result, target = self.export(target=via_dir, source_dir=moved)
        self.assertEqual(result["exported"], 2)
        self.assertEqual(normal.read_bytes(), via_dir.read_bytes())
        with zipfile.ZipFile(target) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            by_digest = {s["sha256"]: s["path"] for s in manifest["samples"]}
            self.assertEqual(archive.read(by_digest[d1]), b"cat-one")
            self.assertEqual(archive.read(by_digest[d2]), b"dog-two")

    def test_multiple_copies_uses_sorted_first(self) -> None:
        moved = self.root.parent / "moved"
        moved.mkdir()
        (moved / "b.jpg").write_bytes(b"same")
        (moved / "a.jpg").write_bytes(b"same")
        (moved / "sub").mkdir()
        (moved / "sub" / "c.jpg").write_bytes(b"same")
        resolver = _scan_source_directory(moved)
        digest = hashlib.sha256(b"same").hexdigest()
        self.assertEqual(resolver[digest][0], (moved / "a.jpg").resolve())

    def test_same_name_different_content_not_used(self) -> None:
        d = self.add_sample("cat", b"real-content")
        self.create_plan()
        # The original source is still in place, but the moved tree only
        # has a same-named file with different content.
        moved = self.root.parent / "moved"
        moved.mkdir()
        (moved / "sample-1.jpg").write_bytes(b"different-content")
        target = self.root.parent / "out.zip"
        with self.assertRaises(ExportError) as caught:
            self.export(target=target, source_dir=moved)
        self.assertIn(d, str(caught.exception))
        self.assertIn("no file with matching content", str(caught.exception))
        self.assertFalse(target.exists())

    def test_unrelated_files_ignored_but_tree_still_fully_read(self) -> None:
        d = self.add_sample("cat", b"abc")
        self.create_plan()
        moved = self.root.parent / "moved"
        moved.mkdir()
        (moved / "photo").write_bytes(b"abc")  # no extension at all
        (moved / "random1").write_bytes(b"xyz")
        (moved / "sub").mkdir()
        (moved / "sub" / "random2").write_bytes(b"123")
        result, target = self.export(source_dir=moved)
        self.assertEqual(result["exported"], 1)
        with zipfile.ZipFile(target) as archive:
            names = [n for n in archive.namelist() if n.endswith(".jpg")]
            self.assertEqual(len(names), 1)
            self.assertEqual(names[0], f"train/class_01/{d}.jpg")
            self.assertEqual(archive.read(names[0]), b"abc")

    def test_unreadable_file_fails_with_path(self) -> None:
        self.add_sample("cat", b"abc")
        self.create_plan()
        moved = self.root.parent / "moved"
        moved.mkdir()
        (moved / "good.jpg").write_bytes(b"abc")
        bad = moved / "unreadable.jpg"
        bad.write_bytes(b"xyz")
        bad.chmod(0o000)
        target = self.root.parent / "out.zip"
        with self.assertRaises(ExportError) as caught:
            self.export(target=target, source_dir=moved)
        self.assertIn(str(bad), str(caught.exception))
        self.assertFalse(target.exists())

    def test_missing_source_dir_fails(self) -> None:
        self.add_sample("cat")
        self.create_plan()
        target = self.root.parent / "out.zip"
        with self.assertRaises(ExportError) as caught:
            self.export(target=target, source_dir=self.root.parent / "nope")
        self.assertIn("does not exist", str(caught.exception))
        self.assertFalse(target.exists())

    def test_source_dir_is_file_fails(self) -> None:
        self.add_sample("cat")
        self.create_plan()
        source = self.root.parent / "afile"
        source.write_bytes(b"x")
        target = self.root.parent / "out.zip"
        with self.assertRaises(ExportError) as caught:
            self.export(target=target, source_dir=source)
        self.assertIn("not a directory", str(caught.exception))
        self.assertFalse(target.exists())

    def test_symlinks_skipped_and_not_descended(self) -> None:
        d = self.add_sample("cat", b"abc")
        self.create_plan()
        moved = self.root.parent / "moved"
        moved.mkdir()
        # Symlink to a file with matching content: skipped, not read.
        (moved / "link.jpg").symlink_to(self.root.parent / "sample-1.jpg")
        # Symlink to a directory holding a matching file: not descended.
        real_dir = self.root.parent / "realdir"
        real_dir.mkdir()
        (real_dir / "deep.jpg").write_bytes(b"abc")
        (moved / "dirlink").symlink_to(real_dir)
        target = self.root.parent / "out.zip"
        with self.assertRaises(ExportError) as caught:
            self.export(target=target, source_dir=moved)
        self.assertIn(d, str(caught.exception))
        self.assertFalse(target.exists())

    def test_empty_plan_skips_scan_but_still_checks_dir(self) -> None:
        self.create_plan("empty")
        moved = self.root.parent / "moved"
        moved.mkdir()
        result, target = self.export("empty", source_dir=moved)
        self.assertEqual(result["exported"], 0)
        # A missing directory still fails for an empty plan.
        target2 = self.root.parent / "out2.zip"
        with self.assertRaises(ExportError):
            self.export("empty", target=target2, source_dir=self.root.parent / "nope")
        self.assertFalse(target2.exists())

    def test_all_skipped_skips_scan(self) -> None:
        self.add_sample(None, b"unlabeled")
        self.create_plan()
        moved = self.root.parent / "moved"
        moved.mkdir()  # deliberately empty
        result, target = self.export(skip_unlabeled=True, source_dir=moved)
        self.assertEqual(result["exported"], 0)
        self.assertEqual(result["skipped"], {"train": 1, "validation": 0, "test": 0})
        self.assertTrue(target.exists())

    def test_skip_unlabeled_with_source_dir_does_not_need_their_files(self) -> None:
        d_labeled = self.add_sample("cat", b"labeled")
        self.add_sample(None, b"unlabeled")
        self.create_plan(ratios=["1", "0", "0"])
        moved = self.root.parent / "moved"
        moved.mkdir()
        (moved / "cat.dat").write_bytes(b"labeled")
        # No file for the unlabeled sample exists anywhere.
        result, target = self.export(skip_unlabeled=True, source_dir=moved)
        self.assertEqual(result["exported"], 1)
        self.assertEqual(result["skipped"], {"train": 1, "validation": 0, "test": 0})
        with zipfile.ZipFile(target) as archive:
            self.assertEqual(archive.read(f"train/class_01/{d_labeled}.jpg"), b"labeled")

    def test_manifest_contains_no_absolute_paths(self) -> None:
        self.add_sample("cat", b"abc")
        self.create_plan()
        moved = self.root.parent / "moved"
        moved.mkdir()
        (moved / "x").write_bytes(b"abc")
        result, target = self.export(source_dir=moved)
        with zipfile.ZipFile(target) as archive:
            blob = archive.read("manifest.json").decode("utf-8")
            self.assertNotIn(str(self.root.parent), blob)
            self.assertNotIn(str(moved), blob)
            for name in archive.namelist():
                self.assertFalse(name.startswith("/"))
            manifest = json.loads(blob)
            for sample in manifest["samples"]:
                self.assertFalse(sample["path"].startswith("/"))
                self.assertNotIn(str(moved), sample["path"])

    def test_lookup_not_written_back(self) -> None:
        self.add_sample("cat", b"abc")
        self.create_plan()
        manifest_before = self.store.manifest_path.read_bytes()
        plan_before = (self.store.splits_directory / "baseline.json").read_bytes()
        moved = self.root.parent / "moved"
        moved.mkdir()
        (moved / "x").write_bytes(b"abc")
        self.export(source_dir=moved)
        self.assertEqual(self.store.manifest_path.read_bytes(), manifest_before)
        self.assertEqual(
            (self.store.splits_directory / "baseline.json").read_bytes(), plan_before
        )

    def test_failure_preserves_existing_target(self) -> None:
        self.add_sample("cat", b"abc")
        self.create_plan()
        target = self.root.parent / "out.zip"
        target.write_bytes(b"existing")
        moved = self.root.parent / "moved"
        moved.mkdir()  # no matching file
        with self.assertRaises(ExportError):
            self.export(target=target, source_dir=moved)
        self.assertEqual(target.read_bytes(), b"existing")

    def test_replaced_file_after_lookup_fails(self) -> None:
        path = self.root.parent / "f.jpg"
        path.write_bytes(b"abc")
        digest = hashlib.sha256(b"abc").hexdigest()
        stat_result = os.lstat(path)
        resolver = {digest: (path, stat_result.st_dev, stat_result.st_ino)}
        # Replace the file after the lookup via rename: a distinct inode
        # at the same path (unlink+create could reuse the inode number).
        replacement = self.root.parent / "replacement.jpg"
        replacement.write_bytes(b"abc")
        os.replace(replacement, path)
        member = {
            "sha256": digest,
            "label": "cat",
            "set": "train",
            "source": str(path),
        }
        stream = __import__("io").BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            with self.assertRaises(ExportError) as caught:
                _write_sample(archive, "train/class_01/x", member, resolver)
        self.assertIn("replaced after lookup", str(caught.exception))


class ResolvedCopySwapTest(SourceDirHarness):
    """The opened copy must be the regular file confirmed at lookup time.

    These tests exercise the gap after the directory lookup has finished
    and the selected copy is about to be opened: the selected path may be
    occupied by a different regular file (even byte-identical) or by a
    symlink before or while the actual open pins the confirmed inode.
    """

    def setUp(self) -> None:
        super().setUp()
        self.digest = self.add_sample("cat", b"abc")
        self.create_plan()
        self.moved = self.root.parent / "moved"
        self.moved.mkdir()
        self.selected = self.moved / "a.jpg"
        self.selected.write_bytes(b"abc")
        # A second same-content copy sorts after the selection, so it can
        # never become the resolver's choice but is always available as a
        # decoy that must not be used as a fallback.
        self.other = self.moved / "b.jpg"
        self.other.write_bytes(b"abc")
        self.target = self.root.parent / "out.zip"

    @contextmanager
    def _post_scan_race(self, mutate, on_open: bool = False):
        """Run ``mutate`` strictly after the directory lookup finishes.

        With ``on_open`` False the mutation lands before the copy inspects
        the selected path; with it True it lands inside the copy's open,
        after the pre-open lstat has already passed.
        """
        real_scan = exporter_mod._scan_source_directory
        state = {"scan_done": False, "fired": False}

        def scan_wrapper(root):
            resolver = real_scan(root)
            state["scan_done"] = True
            if not on_open:
                state["fired"] = True
                mutate()
            return resolver

        patches = [
            patch.object(exporter_mod, "_scan_source_directory", side_effect=scan_wrapper)
        ]
        if on_open:
            real_open = os.open

            def racing_open(path, flags, *args, **kwargs):
                if (
                    state["scan_done"]
                    and not state["fired"]
                    and Path(os.fspath(path)) == self.selected
                ):
                    state["fired"] = True
                    mutate()
                return real_open(path, flags, *args, **kwargs)

            patches.append(patch.object(exporter_mod.os, "open", side_effect=racing_open))
        for patched in patches:
            patched.start()
        try:
            yield state
        finally:
            for patched in patches:
                patched.stop()

    def _assert_failed_cleanly(self, caught: Exception) -> None:
        message = str(caught.exception)
        self.assertIn(self.digest, message)
        self.assertIn(str(self.selected), message)
        self.assertIn("replaced after lookup", message)
        self.assertFalse(self.target.exists())

    def _swap_identical_file(self) -> None:
        replacement = self.moved / "swap.jpg"
        replacement.write_bytes(b"abc")
        os.replace(replacement, self.selected)

    def test_identical_copy_replaced_before_open_fails(self) -> None:
        with self._post_scan_race(self._swap_identical_file):
            with self.assertRaises(ExportError) as caught:
                self.export(target=self.target, source_dir=self.moved)
        self._assert_failed_cleanly(caught)

    def test_identical_copy_replaced_between_lstat_and_open_fails(self) -> None:
        with self._post_scan_race(self._swap_identical_file, on_open=True):
            with self.assertRaises(ExportError) as caught:
                self.export(target=self.target, source_dir=self.moved)
        self._assert_failed_cleanly(caught)

    def test_symlink_before_open_fails_even_pointing_at_original(self) -> None:
        original = self.moved / "original.jpg"
        original.write_bytes(b"abc")

        def make_symlink():
            self.selected.unlink()
            self.selected.symlink_to(original)

        with self._post_scan_race(make_symlink):
            with self.assertRaises(ExportError) as caught:
                self.export(target=self.target, source_dir=self.moved)
        self._assert_failed_cleanly(caught)
        self.assertIn("symlink", str(caught.exception))

    def test_symlink_between_lstat_and_open_fails_even_at_same_inode(self) -> None:
        # The link target is the exact inode the lookup selected; the open
        # must still refuse because the path itself became a symlink.
        def make_symlink():
            holder = self.moved / "holder.jpg"
            os.link(self.selected, holder)  # hard link keeps the inode alive
            self.selected.unlink()
            self.selected.symlink_to(holder)

        with self._post_scan_race(make_symlink, on_open=True):
            with self.assertRaises(ExportError) as caught:
                self.export(target=self.target, source_dir=self.moved)
        self._assert_failed_cleanly(caught)
        self.assertIn("symlink", str(caught.exception))

    def test_symlink_to_other_same_content_copy_fails(self) -> None:
        def make_symlink():
            self.selected.unlink()
            self.selected.symlink_to(self.other)

        with self._post_scan_race(make_symlink, on_open=True):
            with self.assertRaises(ExportError) as caught:
                self.export(target=self.target, source_dir=self.moved)
        self._assert_failed_cleanly(caught)
        self.assertIn("symlink", str(caught.exception))

    def test_no_fallback_to_other_copy_after_swap(self) -> None:
        with self._post_scan_race(self._swap_identical_file):
            with self.assertRaises(ExportError):
                self.export(target=self.target, source_dir=self.moved)
        # b.jpg still holds the matching content, but it must not be used.
        self.assertFalse(self.target.exists())

    def test_no_fallback_to_recorded_source_after_swap(self) -> None:
        recorded = Path(
            self.store.get_split("baseline")["sets"]["train"]["members"][0]["source"]
        )
        self.assertTrue(recorded.exists())  # the old recorded source still works
        with self._post_scan_race(self._swap_identical_file):
            with self.assertRaises(ExportError):
                self.export(target=self.target, source_dir=self.moved)
        self.assertFalse(self.target.exists())

    def test_failure_leaves_workspace_plan_and_existing_target_untouched(self) -> None:
        manifest_before = self.store.manifest_path.read_bytes()
        plan_before = (self.store.splits_directory / "baseline.json").read_bytes()
        existing = self.root.parent / "existing.zip"
        existing.write_bytes(b"keep-me")
        with self._post_scan_race(self._swap_identical_file):
            with self.assertRaises(ExportError):
                self.export(target=existing, source_dir=self.moved)
        self.assertEqual(existing.read_bytes(), b"keep-me")
        self.assertEqual(self.store.manifest_path.read_bytes(), manifest_before)
        self.assertEqual(
            (self.store.splits_directory / "baseline.json").read_bytes(), plan_before
        )

    def test_swap_after_open_still_reads_confirmed_inode(self) -> None:
        # A replacement landing after the descriptor is open cannot change
        # what is read: the descriptor pins the confirmed inode, so the
        # export succeeds using the originally confirmed file.
        replacement = self.moved / "swap.jpg"
        replacement.write_bytes(b"abc")
        real_fstat = os.fstat

        def racing_fstat(descriptor, *args, **kwargs):
            result = real_fstat(descriptor, *args, **kwargs)
            if stat.S_ISREG(result.st_mode) and (result.st_dev, result.st_ino) == (
                os.lstat(self.selected).st_dev,
                os.lstat(self.selected).st_ino,
            ):
                if replacement.exists():
                    os.replace(replacement, self.selected)
            return result

        with patch.object(exporter_mod.os, "fstat", side_effect=racing_fstat):
            result, target = self.export(source_dir=self.moved)
        self.assertEqual(result["exported"], 1)
        with zipfile.ZipFile(target) as archive:
            names = [n for n in archive.namelist() if n.endswith(".jpg")]
            self.assertEqual(archive.read(names[0]), b"abc")

    def test_no_swap_export_still_succeeds_byte_identically(self) -> None:
        normal = self.root.parent / "normal.zip"
        self.export(target=normal)
        via_dir = self.root.parent / "via-dir.zip"
        result, target = self.export(target=via_dir, source_dir=self.moved)
        self.assertEqual(result["exported"], 1)
        self.assertEqual(normal.read_bytes(), via_dir.read_bytes())


class DirectorySwapTest(SourceDirHarness):
    """A real subdirectory swapped for a symlink mid-lookup fails the export.

    The scan confirms each directory as real before descending and
    re-confirms it after processing its entries.  These tests swap a
    confirmed directory for a symlink — pointing outside the source
    tree, inside it, or at the moved-away original — in each gap of the
    lookup and require the whole export to fail naming the directory and
    the symlink, never reading, matching or packaging through the link.
    """

    def setUp(self) -> None:
        super().setUp()
        self.digest = self.add_sample("cat", b"abc")
        self.create_plan()
        self.moved = self.root.parent / "moved"
        self.photos = self.moved / "photos"
        self.photos.mkdir(parents=True)
        (self.photos / "cat.jpg").write_bytes(b"abc")
        # The link target holds a same-content copy: even identical bytes
        # must not be read, matched or packaged through the symlink.
        self.decoy = self.root.parent / "decoy"
        self.decoy.mkdir()
        (self.decoy / "cat.jpg").write_bytes(b"abc")
        self.target = self.root.parent / "out.zip"

    def _replace_with_symlink(self, directory: Path, target: Path) -> None:
        os.rename(directory, directory.with_name(directory.name + ".held"))
        directory.symlink_to(target)

    @contextmanager
    def _swap_when_scanned(self, trigger: Path, mutate):
        """Run ``mutate`` right after ``trigger``'s entries are listed.

        The listing still reflects the confirmed real directory; the
        mutation lands before its entries are processed or descended
        into, exercising the confirmed-then-swapped gap.
        """
        real_scandir = os.scandir
        state = {"fired": False}

        def racing_scandir(path, *args, **kwargs):
            entries = list(real_scandir(path, *args, **kwargs))
            if not state["fired"] and Path(os.fspath(path)) == trigger:
                state["fired"] = True
                mutate()
            return iter(entries)

        with patch.object(exporter_mod.os, "scandir", side_effect=racing_scandir):
            yield state

    @contextmanager
    def _swap_when_hashing(self, directory: Path, mutate):
        """Run ``mutate`` right before the first file under ``directory`` is read."""
        real_hash = exporter_mod._hash_regular_file
        state = {"fired": False}

        def racing_hash(path):
            if not state["fired"] and Path(path).parent == directory:
                state["fired"] = True
                mutate()
            return real_hash(path)

        with patch.object(
            exporter_mod, "_hash_regular_file", side_effect=racing_hash
        ):
            yield state

    def _assert_directory_swap_failed(self, caught, directory: Path) -> None:
        message = str(caught.exception)
        self.assertIn(str(directory), message)
        self.assertIn("symlink", message)
        self.assertFalse(self.target.exists())
        self.assertFalse(
            list(self.target.parent.glob(f".{self.target.name}.*.tmp"))
        )

    def test_swap_before_descending_fails(self) -> None:
        def mutate():
            self._replace_with_symlink(self.photos, self.decoy)

        with self._swap_when_scanned(self.photos, mutate):
            with self.assertRaises(ExportError) as caught:
                self.export(target=self.target, source_dir=self.moved)
        self._assert_directory_swap_failed(caught, self.photos)

    def test_swap_after_parent_listing_fails(self) -> None:
        def mutate():
            self._replace_with_symlink(self.photos, self.decoy)

        with self._swap_when_scanned(self.moved, mutate):
            with self.assertRaises(ExportError) as caught:
                self.export(target=self.target, source_dir=self.moved)
        self._assert_directory_swap_failed(caught, self.photos)

    def test_swap_before_reading_files_under_it_fails(self) -> None:
        def mutate():
            self._replace_with_symlink(self.photos, self.decoy)

        with self._swap_when_hashing(self.photos, mutate):
            with self.assertRaises(ExportError) as caught:
                self.export(target=self.target, source_dir=self.moved)
        self._assert_directory_swap_failed(caught, self.photos)

    def test_symlink_to_moved_original_tree_fails(self) -> None:
        # The link points at the directory's own moved-away location,
        # holding the very same files: still not a valid source.
        relocated = self.root.parent / "relocated-photos"

        def mutate():
            os.rename(self.photos, relocated)
            self.photos.symlink_to(relocated)

        with self._swap_when_scanned(self.photos, mutate):
            with self.assertRaises(ExportError) as caught:
                self.export(target=self.target, source_dir=self.moved)
        self._assert_directory_swap_failed(caught, self.photos)

    def test_symlink_target_inside_source_tree_fails(self) -> None:
        inside = self.moved / "elsewhere"
        inside.mkdir()
        (inside / "cat.jpg").write_bytes(b"abc")

        def mutate():
            self._replace_with_symlink(self.photos, inside)

        with self._swap_when_scanned(self.photos, mutate):
            with self.assertRaises(ExportError) as caught:
                self.export(target=self.target, source_dir=self.moved)
        self._assert_directory_swap_failed(caught, self.photos)

    def test_nested_directory_swap_fails(self) -> None:
        inner = self.moved / "a" / "b"
        inner.mkdir(parents=True)
        (inner / "cat.jpg").write_bytes(b"abc")

        def mutate():
            self._replace_with_symlink(inner, self.decoy)

        with self._swap_when_scanned(inner, mutate):
            with self.assertRaises(ExportError) as caught:
                self.export(target=self.target, source_dir=self.moved)
        self._assert_directory_swap_failed(caught, inner)

    def test_no_fallback_to_other_copy_or_recorded_source(self) -> None:
        # A normal same-content copy elsewhere in the tree and the plan's
        # still-readable recorded source must not paper over the change.
        (self.moved / "zz-copy.jpg").write_bytes(b"abc")
        recorded = Path(
            self.store.get_split("baseline")["sets"]["train"]["members"][0]["source"]
        )
        self.assertTrue(recorded.exists())

        def mutate():
            self._replace_with_symlink(self.photos, self.decoy)

        with self._swap_when_scanned(self.photos, mutate):
            with self.assertRaises(ExportError):
                self.export(target=self.target, source_dir=self.moved)
        self.assertFalse(self.target.exists())

    def test_failure_leaves_workspace_plan_and_existing_target_untouched(self) -> None:
        manifest_before = self.store.manifest_path.read_bytes()
        plan_before = (self.store.splits_directory / "baseline.json").read_bytes()
        existing = self.root.parent / "existing.zip"
        existing.write_bytes(b"keep-me")

        def mutate():
            self._replace_with_symlink(self.photos, self.decoy)

        with self._swap_when_scanned(self.photos, mutate):
            with self.assertRaises(ExportError):
                self.export(target=existing, source_dir=self.moved)
        self.assertEqual(existing.read_bytes(), b"keep-me")
        self.assertEqual(self.store.manifest_path.read_bytes(), manifest_before)
        self.assertEqual(
            (self.store.splits_directory / "baseline.json").read_bytes(), plan_before
        )

    def test_no_swap_nested_export_still_succeeds(self) -> None:
        result, target = self.export(source_dir=self.moved)
        self.assertEqual(result["exported"], 1)
        with zipfile.ZipFile(target) as archive:
            names = [n for n in archive.namelist() if n.endswith(".jpg")]
            self.assertEqual(names, [f"train/class_01/{self.digest}.jpg"])
            self.assertEqual(archive.read(names[0]), b"abc")

    def test_cli_symlinked_directory_not_used(self) -> None:
        # A directory that is already a symlink when first encountered is
        # skipped by the standing convention: the export fails for lack of
        # a match, with a non-zero exit, no success output and no target.
        os.rename(self.photos, self.root.parent / "photos.held")
        self.photos.symlink_to(self.decoy)
        result = self.run_cli(
            "export", str(self.root), "baseline", str(self.target),
            "--source-dir", str(self.moved),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertFalse(self.target.exists())


class SourceDirCliTest(SourceDirHarness):
    def test_cli_source_dir_flag(self) -> None:
        self.add_sample("cat", b"abc")
        self.create_plan()
        moved = self.root.parent / "moved"
        moved.mkdir()
        (moved / "renamed").write_bytes(b"abc")
        target = self.root.parent / "cli.zip"
        result = self.run_cli(
            "export", str(self.root), "baseline", str(target),
            "--source-dir", str(moved),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        body = json.loads(result.stdout)
        self.assertEqual(body["exported"], 1)
        self.assertTrue(target.exists())

    def test_cli_missing_source_dir_fails(self) -> None:
        self.add_sample("cat")
        self.create_plan()
        target = self.root.parent / "cli.zip"
        result = self.run_cli(
            "export", str(self.root), "baseline", str(target),
            "--source-dir", str(self.root.parent / "nope"),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source directory", result.stderr)
        self.assertFalse(target.exists())

    def test_concurrent_exports_same_target_one_succeeds(self) -> None:
        self.add_sample("cat", b"abc")
        self.create_plan()
        moved = self.root.parent / "moved"
        moved.mkdir()
        (moved / "x").write_bytes(b"abc")
        target = self.root.parent / "out.zip"
        cli = [
            sys.executable, "-m", "vision_workbench", "export",
            str(self.root), "baseline", str(target),
            "--source-dir", str(moved),
        ]
        processes = [
            subprocess.Popen(
                cli, cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            for _ in range(3)
        ]
        results = [p.wait() for p in processes]
        self.assertEqual(results.count(0), 1, results)
        self.assertTrue(target.exists())
        with zipfile.ZipFile(target) as archive:
            self.assertIn("manifest.json", archive.namelist())


if __name__ == "__main__":
    unittest.main()
