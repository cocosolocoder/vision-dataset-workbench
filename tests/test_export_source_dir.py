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


def _remove_deep_tree(root: Path) -> None:
    """Delete a directory tree without recursive Python calls.

    ``shutil.rmtree`` walks recursively and therefore fails on chains
    deeper than Python's recursion limit; this variant keeps a stack of
    directory descriptors and removes files at once and empty
    directories deepest-first.  Symlinks are unlinked, never traversed.
    A root that is already gone is treated as success.
    """
    if not root.exists():
        return
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    # (parent fd or None for root, this fd, basename or None, entries)
    stack: list = [(None, root_fd, None, os.scandir(root_fd))]
    try:
        while stack:
            parent_fd, fd, name, entries = stack[-1]
            try:
                entry = next(entries)
            except StopIteration:
                entries.close()
                os.close(fd)
                stack.pop()
                if parent_fd is not None:
                    os.rmdir(name, dir_fd=parent_fd)
                continue
            if entry.is_dir(follow_symlinks=False):
                try:
                    child_fd = os.open(
                        entry.name,
                        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                        dir_fd=fd,
                    )
                except PermissionError:
                    # A directory a test made unreadable on purpose:
                    # restore access so the tree can be unwound.
                    os.chmod(entry.name, 0o700, dir_fd=fd)
                    child_fd = os.open(
                        entry.name,
                        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                        dir_fd=fd,
                    )
                stack.append(
                    (fd, child_fd, entry.name, os.scandir(child_fd))
                )
            else:
                os.unlink(entry.name, dir_fd=fd)
    finally:
        for parent_fd, fd, name, entries in stack:
            entries.close()
            os.close(fd)
    os.rmdir(root)


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
        # export succeeds using the originally confirmed file.  The lookup
        # itself proves its descriptors with fstat as well, so the swap is
        # gated until the scan has finished and fires on the copy's prove.
        replacement = self.moved / "swap.jpg"
        replacement.write_bytes(b"abc")
        real_fstat = os.fstat
        real_scan = exporter_mod._scan_source_directory
        state = {"scan_done": False, "fired": False}

        def scan_wrapper(root):
            resolver = real_scan(root)
            state["scan_done"] = True
            return resolver

        def racing_fstat(descriptor, *args, **kwargs):
            result = real_fstat(descriptor, *args, **kwargs)
            if (
                state["scan_done"]
                and not state["fired"]
                and stat.S_ISREG(result.st_mode)
                and (result.st_dev, result.st_ino) == (
                    os.lstat(self.selected).st_dev,
                    os.lstat(self.selected).st_ino,
                )
            ):
                state["fired"] = True
                os.replace(replacement, self.selected)
            return result

        with (
            patch.object(exporter_mod, "_scan_source_directory", side_effect=scan_wrapper),
            patch.object(exporter_mod.os, "fstat", side_effect=racing_fstat),
        ):
            result, target = self.export(source_dir=self.moved)
        self.assertTrue(state["fired"])
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


class DirectoryReplacedBySymlinkTest(SourceDirHarness):
    """A real subdirectory must not be followed once it becomes a symlink.

    A subdirectory that is a genuine directory when the walk first
    reaches it, but is replaced by a symlink before it is entered, while
    its contents are read, after the lookup or while the package is
    copied, must fail the whole export — regardless of where the link
    points (outside the tree, inside it, or at the moved original) and
    regardless of the nesting level at which the change happens.
    """

    CAT = b"cat-photo"

    def setUp(self) -> None:
        super().setUp()
        self.digest = self.add_sample("cat", self.CAT)
        self.create_plan()
        self.moved = self.root.parent / "moved"
        self.target_zip = self.root.parent / "out.zip"
        # The recorded old source still exists and is readable; it must
        # never mask the directory change.
        self.recorded = Path(
            self.store.get_split("baseline")["sets"]["train"]["members"][0]["source"]
        )
        self.assertTrue(self.recorded.exists())

    def _build_tree(self, nested: bool) -> Path:
        """Create moved[/a/b]/photos/cat.jpg plus same-content decoys.

        Decoys: another normal copy elsewhere under the tree and a
        directory outside the tree — the symlink swaps below point at
        directories holding the same image content.
        """
        photos = self.moved / ("a/b/photos" if nested else "photos")
        photos.mkdir(parents=True)
        (photos / "cat.jpg").write_bytes(self.CAT)
        # A normal same-content copy elsewhere in the tree: not a fallback.
        spare = self.moved / "spare"
        spare.mkdir(parents=True)
        (spare / "cat.jpg").write_bytes(self.CAT)
        # The outside directory a swapped link could point at.
        outside = self.root.parent / "outside"
        outside.mkdir()
        (outside / "cat.jpg").write_bytes(self.CAT)
        # The inside directory a swapped link could point at.
        inside = self.moved / "other"
        inside.mkdir()
        (inside / "cat.jpg").write_bytes(self.CAT)
        return photos

    def _swap_to_symlink(self, photos: Path, kind: str) -> None:
        """Replace the real ``photos`` directory with a symlink."""
        moved_away = photos.with_name(photos.name + "-moved")
        os.rename(photos, moved_away)
        if kind == "outside":
            target = self.root.parent / "outside"
        elif kind == "inside":
            target = self.moved / "other"
        elif kind == "moved":
            target = moved_away
        else:  # pragma: no cover - test programming error
            raise AssertionError(kind)
        photos.symlink_to(target)

    @contextmanager
    def _swap_at(self, when: str, photos: Path, mutate):
        patches = []
        if when == "enter":
            # After the parent listed and typed the entry as a real
            # directory, before that directory is entered.
            real_enter = exporter_mod._enter_subdirectory

            def enter_wrap(parent_fd, name, display, *args, **kwargs):
                if Path(os.fspath(display)) == photos:
                    mutate()
                return real_enter(parent_fd, name, display, *args, **kwargs)

            patches.append(
                patch.object(exporter_mod, "_enter_subdirectory", side_effect=enter_wrap)
            )
        elif when == "open-gap":
            # Between the no-follow fstatat of the entry and the openat
            # that enters it.
            real_open = os.open

            def open_wrap(path, flags, *args, **kwargs):
                if (
                    flags & getattr(os, "O_DIRECTORY", 0)
                    and Path(os.fspath(path)).name == photos.name
                ):
                    mutate()
                return real_open(path, flags, *args, **kwargs)

            patches.append(
                patch.object(exporter_mod.os, "open", side_effect=open_wrap)
            )
        elif when == "child":
            # While the directory's children are being processed: after
            # the file beneath it has been read, before the directory's
            # own post-child binding check.
            real_hash = exporter_mod._hash_regular_file_at

            def hash_wrap(directory_fd, name, path, *args, **kwargs):
                result = real_hash(directory_fd, name, path, *args, **kwargs)
                if Path(path).parent == photos:
                    mutate()
                return result

            patches.append(
                patch.object(
                    exporter_mod, "_hash_regular_file_at", side_effect=hash_wrap
                )
            )
        elif when == "after-scan":
            real_scan = exporter_mod._scan_source_directory

            def scan_wrap(root):
                resolver = real_scan(root)
                mutate()
                return resolver

            patches.append(
                patch.object(
                    exporter_mod, "_scan_source_directory", side_effect=scan_wrap
                )
            )
        elif when == "during-read":
            # After the selected file was opened for copying; the final
            # whole-tree gate must still refuse the finished package.
            real_resolve = exporter_mod._open_resolved_source

            def resolve_wrap(source, digest, device, inode):
                stream, size = real_resolve(source, digest, device, inode)
                mutate()
                return stream, size

            patches.append(
                patch.object(
                    exporter_mod,
                    "_open_resolved_source",
                    side_effect=resolve_wrap,
                )
            )
        else:  # pragma: no cover - test programming error
            raise AssertionError(when)
        for patched in patches:
            patched.start()
        try:
            yield
        finally:
            for patched in patches:
                patched.stop()

    def _assert_failed_cleanly(self, caught: Exception, photos: Path) -> None:
        message = str(caught.exception)
        self.assertIn(str(photos), message)
        self.assertIn("symlink", message)
        self.assertIsInstance(caught.exception, ValueError)
        self.assertFalse(self.target_zip.exists())
        leftovers = [
            entry.name
            for entry in self.root.parent.iterdir()
            if entry.name.endswith(".tmp")
        ]
        self.assertEqual(leftovers, [])

    def test_swapped_before_entry_outside_target(self) -> None:
        photos = self._build_tree(nested=False)
        with self._swap_at("enter", photos, lambda: self._swap_to_symlink(photos, "outside")):
            with self.assertRaises(ExportError) as caught:
                self.export(target=self.target_zip, source_dir=self.moved)
        self._assert_failed_cleanly(caught, photos)

    def test_swapped_between_stat_and_open_outside_target(self) -> None:
        photos = self._build_tree(nested=False)
        with self._swap_at("open-gap", photos, lambda: self._swap_to_symlink(photos, "outside")):
            with self.assertRaises(ExportError) as caught:
                self.export(target=self.target_zip, source_dir=self.moved)
        self._assert_failed_cleanly(caught, photos)

    def test_swapped_before_entry_target_inside_tree(self) -> None:
        photos = self._build_tree(nested=False)
        with self._swap_at("enter", photos, lambda: self._swap_to_symlink(photos, "inside")):
            with self.assertRaises(ExportError) as caught:
                self.export(target=self.target_zip, source_dir=self.moved)
        self._assert_failed_cleanly(caught, photos)

    def test_swapped_before_entry_points_at_moved_original(self) -> None:
        photos = self._build_tree(nested=False)
        with self._swap_at("enter", photos, lambda: self._swap_to_symlink(photos, "moved")):
            with self.assertRaises(ExportError) as caught:
                self.export(target=self.target_zip, source_dir=self.moved)
        self._assert_failed_cleanly(caught, photos)

    def test_swapped_while_children_processed(self) -> None:
        photos = self._build_tree(nested=False)
        with self._swap_at("child", photos, lambda: self._swap_to_symlink(photos, "outside")):
            with self.assertRaises(ExportError) as caught:
                self.export(target=self.target_zip, source_dir=self.moved)
        self._assert_failed_cleanly(caught, photos)

    def test_swapped_at_deep_nesting_level(self) -> None:
        photos = self._build_tree(nested=True)
        with self._swap_at("open-gap", photos, lambda: self._swap_to_symlink(photos, "moved")):
            with self.assertRaises(ExportError) as caught:
                self.export(target=self.target_zip, source_dir=self.moved)
        self._assert_failed_cleanly(caught, photos)

    def test_swapped_at_ancestor_level_of_nested_tree(self) -> None:
        photos = self._build_tree(nested=True)
        ancestor = self.moved / "a"
        with self._swap_at(
            "enter", ancestor, lambda: self._swap_to_symlink(ancestor, "outside")
        ):
            with self.assertRaises(ExportError) as caught:
                self.export(target=self.target_zip, source_dir=self.moved)
        self._assert_failed_cleanly(caught, ancestor)

    def test_swapped_during_children_of_nested_directory(self) -> None:
        photos = self._build_tree(nested=True)
        with self._swap_at("child", photos, lambda: self._swap_to_symlink(photos, "outside")):
            with self.assertRaises(ExportError) as caught:
                self.export(target=self.target_zip, source_dir=self.moved)
        self._assert_failed_cleanly(caught, photos)

    def test_swapped_after_lookup_fails_before_copy(self) -> None:
        photos = self._build_tree(nested=False)
        with self._swap_at(
            "after-scan", photos, lambda: self._swap_to_symlink(photos, "moved")
        ):
            with self.assertRaises(ExportError) as caught:
                self.export(target=self.target_zip, source_dir=self.moved)
        self._assert_failed_cleanly(caught, photos)

    def test_swapped_during_read_fails_final_gate(self) -> None:
        photos = self._build_tree(nested=False)
        with self._swap_at(
            "during-read", photos, lambda: self._swap_to_symlink(photos, "moved")
        ):
            with self.assertRaises(ExportError) as caught:
                self.export(target=self.target_zip, source_dir=self.moved)
        # The pinned bytes may have been read, but the directory changed
        # before publication, so the package must not exist.
        self._assert_failed_cleanly(caught, photos)

    def test_swap_after_first_sample_fails_whole_export(self) -> None:
        # Two samples in separate directories: replacing the second
        # sample's directory after the first entry was written must still
        # fail the whole export and publish nothing.
        self.moved.mkdir(parents=True)
        first = self.moved / "one"
        first.mkdir()
        (first / "cat.jpg").write_bytes(self.CAT)
        self.add_sample("dog", b"dog-photo")
        self.create_plan("two")
        two = self.moved / "two"
        two.mkdir(parents=True)
        (two / "dog.jpg").write_bytes(b"dog-photo")
        target = self.root.parent / "multi.zip"

        real_write_sample = exporter_mod._write_sample
        fired = {"done": False}

        def write_wrap(archive, arc_name, member, *args, **kwargs):
            result = real_write_sample(archive, arc_name, member, *args, **kwargs)
            if not fired["done"] and member["sha256"] == self.digest:
                fired["done"] = True
                outside = self.root.parent / "outside-two"
                outside.mkdir()
                (outside / "dog.jpg").write_bytes(b"dog-photo")
                moved_away = two.with_name("two-moved")
                os.rename(two, moved_away)
                two.symlink_to(outside)
            return result

        with patch.object(exporter_mod, "_write_sample", side_effect=write_wrap):
            with self.assertRaises(ExportError) as caught:
                self.export("two", target=target, source_dir=self.moved)
        self.assertTrue(fired["done"])
        message = str(caught.exception)
        self.assertIn(str(two), message)
        self.assertIn("symlink", message)
        self.assertFalse(target.exists())

    def test_failure_preserves_target_and_workspace(self) -> None:
        photos = self._build_tree(nested=False)
        existing = self.root.parent / "existing.zip"
        existing.write_bytes(b"keep-me")
        manifest_before = self.store.manifest_path.read_bytes()
        plan_before = (
            self.store.splits_directory / "baseline.json"
        ).read_bytes()
        with self._swap_at("enter", photos, lambda: self._swap_to_symlink(photos, "outside")):
            with self.assertRaises(ExportError):
                self.export(target=existing, source_dir=self.moved)
        self.assertEqual(existing.read_bytes(), b"keep-me")
        self.assertEqual(self.store.manifest_path.read_bytes(), manifest_before)
        self.assertEqual(
            (self.store.splits_directory / "baseline.json").read_bytes(),
            plan_before,
        )

    def test_directory_swap_rejected_by_scan_directly(self) -> None:
        photos = self._build_tree(nested=False)
        with self._swap_at(
            "open-gap", photos, lambda: self._swap_to_symlink(photos, "inside")
        ):
            with self.assertRaises(ExportError) as caught:
                exporter_mod._scan_source_directory(self.moved)
        self.assertIn(str(photos), str(caught.exception))
        self.assertIn("symlink", str(caught.exception))

    def test_no_swap_keeps_normal_export_behavior(self) -> None:
        photos = self._build_tree(nested=False)
        normal = self.root.parent / "normal.zip"
        self.export(target=normal)
        via_dir = self.root.parent / "via-dir.zip"
        result, target = self.export(target=via_dir, source_dir=self.moved)
        self.assertEqual(result["exported"], 1)
        self.assertEqual(normal.read_bytes(), via_dir.read_bytes())
        with zipfile.ZipFile(target) as archive:
            names = [n for n in archive.namelist() if n.endswith(".jpg")]
            self.assertEqual(archive.read(names[0]), self.CAT)


class DeepTreeSourceDirExportTest(SourceDirHarness):
    """Exports from directory chains deeper than Python's recursion limit."""

    def setUp(self) -> None:
        super().setUp()
        # Roots of chains deeper than Python's recursion limit; removed
        # iteratively in tearDown because shutil.rmtree is itself
        # recursive and cannot unwind such a tree.
        self._deep_roots: list[Path] = []

    def tearDown(self) -> None:
        for root in self._deep_roots:
            _remove_deep_tree(root)
        super().tearDown()

    def _make_chain(self, root: Path, depth: int, name: str = "a") -> Path:
        """Create a ``depth``-level chain of directories without recursion.

        Single-character names keep the absolute path under the platform
        ``PATH_MAX`` even when the depth exceeds Python's recursion limit
        (``Path.mkdir(parents=True)`` is itself recursive and cannot build
        such a chain, so each level is made with one ``os.mkdir``).
        """
        directory = root
        root.mkdir(parents=True, exist_ok=True)
        for _ in range(depth):
            directory = directory / name
            os.mkdir(directory)
        self._deep_roots.append(root)
        return directory

    def _build_moved_tree(self, depth: int) -> tuple[Path, Path, Path]:
        """A deep chain under ``moved`` with a side branch halfway down.

        Returns the tree root, the mid-chain directory carrying the
        ``side`` subdirectory and the leaf directory at the chain's end.
        """
        moved = self.root.parent / "moved-deep"
        moved.mkdir()
        self._deep_roots.append(moved)
        directory = moved
        mid = moved
        for level in range(depth):
            directory = directory / "a"
            os.mkdir(directory)
            if level == depth // 2:
                mid = directory
        side = mid / "side"
        os.mkdir(side)
        return moved, side, directory

    def test_deep_tree_exports_every_sample_byte_identical(self) -> None:
        depth = sys.getrecursionlimit() + 250
        contents = {
            "root": b"root-sample-content",
            "mid": b"mid-side-branch-content",
            "leaf": b"leaf-of-the-long-chain",
        }
        digests = {
            where: self.add_sample(label, content)
            for (where, content), label in zip(
                contents.items(), ["cat", "dog", "bird"]
            )
        }
        self.create_plan()
        normal = self.root.parent / "normal.zip"
        self.export(target=normal)
        manifest_before = self.store.manifest_path.read_bytes()
        plan_before = (
            self.store.splits_directory / "baseline.json"
        ).read_bytes()

        moved, side, leaf = self._build_moved_tree(depth)
        # Root of the new tree, no extension; mid-chain side branch,
        # renamed with a new extension; end of the long chain, no
        # extension.  The recorded sources are deleted, so only the
        # moved tree can supply the bytes.
        (moved / "root-sample").write_bytes(contents["root"])
        (side / "renamed.data").write_bytes(contents["mid"])
        (leaf / "leaf-sample").write_bytes(contents["leaf"])
        for child in self.root.parent.iterdir():
            if child.name.startswith("sample-") and child.suffix == ".jpg":
                child.unlink()

        via_dir = self.root.parent / "via-dir.zip"
        result, target = self.export(target=via_dir, source_dir=moved)

        self.assertEqual(result["exported"], 3)
        # Depth only decides where bytes are found: the package is
        # byte-identical to the one exported from the recorded paths.
        self.assertEqual(normal.read_bytes(), via_dir.read_bytes())
        with zipfile.ZipFile(target) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            by_digest = {s["sha256"]: s for s in manifest["samples"]}
            for where, digest in digests.items():
                entry = by_digest[digest]
                # The package extension still comes from the recorded
                # source, never from the moved file's name.
                self.assertTrue(entry["path"].endswith(".jpg"))
                self.assertEqual(
                    archive.read(entry["path"]), contents[where]
                )
        # The moved locations are not written back into the records.
        self.assertEqual(self.store.manifest_path.read_bytes(), manifest_before)
        self.assertEqual(
            (self.store.splits_directory / "baseline.json").read_bytes(),
            plan_before,
        )

    def test_deep_tree_sorted_first_selection_ignores_depth(self) -> None:
        depth = sys.getrecursionlimit() + 250
        leaf = self._make_chain(self.root.parent / "moved-order", depth)
        moved = leaf.parents[depth - 1]
        digest = hashlib.sha256(b"shared").hexdigest()
        # The deep copy's relative path sorts before the shallow one.
        deep_copy = leaf / "z-deep"
        deep_copy.write_bytes(b"shared")
        shallow_copy = moved / "z-shallow"
        shallow_copy.write_bytes(b"shared")
        resolver = _scan_source_directory(moved)
        self.assertEqual(resolver[digest][0], deep_copy.resolve())
        # A shallow copy whose name sorts first wins instead: the choice
        # follows the slash-separated relative path, not the depth or
        # the order directories were visited in.
        shallow_copy.rename(moved / "0-shallow")
        resolver = _scan_source_directory(moved)
        self.assertEqual(resolver[digest][0], (moved / "0-shallow").resolve())

    def test_deep_unreadable_directory_fails_whole_export(self) -> None:
        depth = sys.getrecursionlimit() + 250
        self.add_sample("cat", b"abc")
        self.create_plan()
        leaf = self._make_chain(self.root.parent / "moved-unreadable", depth)
        moved = leaf.parents[depth - 1]
        # Every sample is findable at the root of the tree...
        (moved / "cat.jpg").write_bytes(b"abc")
        # ...but a directory at the deep end cannot be scanned.
        blocked = leaf / "blocked"
        os.mkdir(blocked)
        blocked.chmod(0)
        target = self.root.parent / "out.zip"
        with self.assertRaises(ExportError) as caught:
            self.export(target=target, source_dir=moved)
        self.assertIn(str(blocked), str(caught.exception))
        self.assertFalse(target.exists())
        leftovers = [
            entry.name
            for entry in self.root.parent.iterdir()
            if entry.name.endswith(".tmp")
        ]
        self.assertEqual(leftovers, [])

    def test_deep_unreadable_file_fails_whole_export(self) -> None:
        depth = sys.getrecursionlimit() + 250
        self.add_sample("cat", b"abc")
        self.create_plan()
        leaf = self._make_chain(self.root.parent / "moved-badfile", depth)
        moved = leaf.parents[depth - 1]
        (moved / "cat.jpg").write_bytes(b"abc")
        bad = leaf / "unreadable"
        bad.write_bytes(b"xyz")
        bad.chmod(0)
        target = self.root.parent / "out.zip"
        with self.assertRaises(ExportError) as caught:
            self.export(target=target, source_dir=moved)
        self.assertIn(str(bad), str(caught.exception))
        self.assertFalse(target.exists())

    def test_deep_directory_swapped_after_scan_fails_export(self) -> None:
        depth = sys.getrecursionlimit() + 250
        self.add_sample("cat", b"abc")
        self.create_plan()
        moved, side, leaf = self._build_moved_tree(depth)
        (leaf / "cat.jpg").write_bytes(b"abc")
        # A confirmed directory halfway down the chain, replaced by a
        # symlink (pointing at the moved original, so even the content
        # behind the link is unchanged) after the lookup finished.
        victim = side.parent
        moved_away = victim.with_name("victim-moved")

        real_scan = exporter_mod._scan_source_directory

        def scan_wrap(root):
            resolver = real_scan(root)
            os.rename(victim, moved_away)
            victim.symlink_to(moved_away)
            return resolver

        target = self.root.parent / "out.zip"
        with patch.object(
            exporter_mod, "_scan_source_directory", side_effect=scan_wrap
        ):
            with self.assertRaises(ExportError) as caught:
                self.export(target=target, source_dir=moved)
        message = str(caught.exception)
        self.assertIn(str(victim), message)
        self.assertIn("symlink", message)
        self.assertFalse(target.exists())

    def test_empty_and_fully_skipped_plans_never_scan_deep_content(self) -> None:
        depth = sys.getrecursionlimit() + 250
        self.create_plan("empty")  # no samples registered yet
        self.add_sample(None, b"unlabeled")
        self.create_plan()
        leaf = self._make_chain(self.root.parent / "moved-skipped", depth)
        moved = leaf.parents[depth - 1]
        # Deep and unreadable content must not matter when nothing has
        # to be exported.
        blocked = leaf / "blocked"
        os.mkdir(blocked)
        blocked.chmod(0)
        result, target = self.export("empty", source_dir=moved)
        self.assertEqual(result["exported"], 0)
        self.assertTrue(target.exists())
        skipped_target = self.root.parent / "skipped.zip"
        result, _ = self.export(
            skip_unlabeled=True, source_dir=moved, target=skipped_target
        )
        self.assertEqual(result["exported"], 0)
        self.assertEqual(
            result["skipped"], {"train": 1, "validation": 0, "test": 0}
        )
        self.assertTrue(skipped_target.exists())

    def test_cli_deep_unreadable_directory_fails_nonzero_quietly(self) -> None:
        depth = sys.getrecursionlimit() + 250
        self.add_sample("cat", b"abc")
        self.create_plan()
        leaf = self._make_chain(self.root.parent / "moved-cli", depth)
        moved = leaf.parents[depth - 1]
        (moved / "cat.jpg").write_bytes(b"abc")
        blocked = leaf / "blocked"
        os.mkdir(blocked)
        blocked.chmod(0)
        target = self.root.parent / "cli.zip"
        result = self.run_cli(
            "export", str(self.root), "baseline", str(target),
            "--source-dir", str(moved),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn(str(blocked), result.stderr)
        self.assertFalse(target.exists())


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
