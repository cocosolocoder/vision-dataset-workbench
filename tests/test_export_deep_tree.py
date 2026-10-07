"""Export with ``--source-dir`` from a tree deeper than Python's recursion limit.

After images are moved into a new directory tree whose directory chain
is deeper than the interpreter's recursion limit, an export must still
find every original by its full SHA-256: files at the root, at the end
of the long chain and on midway side branches all match, renamed or
extension-less files still qualify, duplicate content keeps the
``/``-separated Unicode-code-point-sorted-first copy regardless of visit
order, and the resulting package is byte-identical to an export from the
recorded source paths.  Deep unreadable directories or files still fail
the whole export naming the deep path, symlinks met for the first time
stay skipped, a confirmed deep directory replaced by a symlink during
the scan or the copy still rejects everything, and an empty plan (or
``--skip-unlabeled`` with nothing left to export) only requires the
directory to exist.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from vision_workbench.exporter import ExportError, export_split
from vision_workbench.store import DatasetStore
import vision_workbench.exporter as exporter_mod

REPO_ROOT = Path(__file__).resolve().parents[1]


def remove_deep_tree(root: Path) -> None:
    """Delete a directory tree without recursive Python calls.

    ``shutil.rmtree`` walks recursively and therefore fails on chains
    deeper than Python's recursion limit; this keeps an explicit stack
    of directory descriptors and removes files immediately and empty
    directories deepest-first.  Symlinks are unlinked, never traversed.
    A root that is already gone is treated as success.
    """
    if not root.exists():
        return
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
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


class DeepSourceDirTest(unittest.TestCase):
    # A chain comfortably beyond the interpreter recursion limit while
    # single-character names keep every absolute path under PATH_MAX.
    DEPTH = sys.getrecursionlimit() + 250

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.workspace = self.base / "dataset"
        self.store = DatasetStore(self.workspace)
        self.store.initialize()
        self.moved = self.base / "moved"
        self.moved.mkdir()
        # Roots of deep chains removed iteratively in tearDown.
        self._deep_roots: list[Path] = [self.moved]
        # (path, mode) entries made unreadable during a test; restored
        # before the iterative teardown needs to traverse them.
        self._denied: list[tuple[Path, int]] = []
        self._serial = 0

    def tearDown(self) -> None:
        for path, mode in self._denied:
            try:
                os.chmod(path, mode)
            except OSError:
                pass
        for root in self._deep_roots:
            remove_deep_tree(root)
        self.temp.cleanup()

    def _make_chain(self, root: Path, depth: int, name: str = "a") -> Path:
        """Create a ``depth``-level chain of directories without recursion.

        Single-character names keep the absolute path under PATH_MAX even
        beyond the recursion limit; each level is one ``os.mkdir`` because
        ``Path.mkdir(parents=True)`` is itself recursive.
        """
        directory = root
        root.mkdir(parents=True, exist_ok=True)
        for _ in range(depth):
            directory = directory / name
            os.mkdir(directory)
        return directory

    def _deny(self, path: Path) -> None:
        """Make a deep path unreadable; remember it for teardown cleanup."""
        mode = path.stat().st_mode & 0o7777
        os.chmod(path, 0)
        self._denied.append((path, mode))

    def add_sample(self, content: bytes, label: str, recorded_name: str) -> str:
        self._serial += 1
        path = self.base / "recorded" / recorded_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        result = self.store.add(path, label)
        self.assertTrue(result.added)
        return result.digest

    def export(
        self,
        plan: str = "deep",
        target: Path | None = None,
        skip_unlabeled: bool = False,
        source_dir: Path | None = None,
    ):
        target = target or self.base / "out.zip"
        return (
            export_split(
                self.store,
                plan,
                target,
                skip_unlabeled=skip_unlabeled,
                source_dir=source_dir,
            ),
            target,
        )

    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "vision_workbench", *arguments],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def _build_moved_tree(self) -> tuple[Path, Path]:
        """Place four samples at root, chain end and a midway side branch.

        Recorded sources are ordinary ``.jpg`` paths; in the moved tree
        the same bytes are renamed, extension-less and re-extensioned, so
        matching can only be content-based.  Returns (chain leaf, the
        midway side branch).
        """
        leaf = self._make_chain(self.moved, self.DEPTH)
        # Midway side branch off the long chain.
        branch = leaf.parent / "sibling"
        os.mkdir(branch)
        # Root-level copy: no extension at all.
        (self.moved / "top").write_bytes(b"root-bytes")
        # Chain-end copy: a different, unknown extension.
        (leaf / "leaf.unknown").write_bytes(b"leaf-bytes")
        # Side-branch copy: a different extension.
        (branch / "shot.png").write_bytes(b"branch-bytes")
        # A plain unrelated regular file at the deepest level is still
        # read in full (no extension filter) but never packaged.
        (leaf / "notes.txt").write_bytes(b"not a planned image")
        return leaf, branch

    # ------------------------------------------------------------------
    # Successful deep exports
    # ------------------------------------------------------------------

    def test_deep_tree_finds_samples_at_every_depth_byte_identically(self) -> None:
        d_root = self.add_sample(b"root-bytes", "cat", "root.jpg")
        d_leaf = self.add_sample(b"leaf-bytes", "dog", "leaf.jpg")
        d_branch = self.add_sample(b"branch-bytes", "bird", "branch.jpg")
        self.store.create_split("deep", 1, ["1", "0", "0"])
        normal = self.base / "normal.zip"
        self.export(target=normal)

        self._build_moved_tree()
        via_dir, target = self.export(source_dir=self.moved)

        self.assertEqual(via_dir["exported"], 3)
        self.assertEqual(normal.read_bytes(), target.read_bytes())
        with zipfile.ZipFile(target) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            by_digest = {s["sha256"]: s["path"] for s in manifest["samples"]}
            # Package extensions stay the plan's recorded .jpg even
            # though the moved copies are extension-less or renamed.
            for digest, expected in (
                (d_root, b"root-bytes"),
                (d_leaf, b"leaf-bytes"),
                (d_branch, b"branch-bytes"),
            ):
                arc = by_digest[digest]
                self.assertTrue(arc.endswith(f"{digest}.jpg"), arc)
                self.assertEqual(archive.read(arc), expected)

    def test_duplicate_content_picked_by_sorted_path_not_visit_order(self) -> None:
        leaf, _branch = self._build_moved_tree()
        # Content A: the root copy sorts AFTER the deep copy ('z' > 'a'),
        # even though the walk visits the root copy long before the chain
        # end.
        (self.moved / "zzz-late-root").write_bytes(b"alpha")
        (leaf / "aaa-deep").write_bytes(b"alpha")
        # Content B: the deep copy lives under the chain prefix "a/" but
        # the root name starts with '0' (below 'a'), so the root copy
        # sorts first regardless of visit order.
        (self.moved / "000-early-root").write_bytes(b"beta")
        (leaf / "zzz-deep").write_bytes(b"beta")

        resolver = exporter_mod._scan_source_directory(self.moved.resolve())
        alpha = hashlib.sha256(b"alpha").hexdigest()
        beta = hashlib.sha256(b"beta").hexdigest()
        chosen_alpha = resolver[alpha][0]
        chosen_beta = resolver[beta][0]
        self.assertEqual(chosen_alpha.name, "aaa-deep")
        self.assertEqual(chosen_beta.name, "000-early-root")
        self.assertEqual(
            str(chosen_alpha.relative_to(self.moved.resolve())).replace(os.sep, "/"),
            f"{'a/' * self.DEPTH}aaa-deep",
        )

        # Both contents export through the chosen copies; no
        # "no matching content" failure, and the plan's recorded
        # extensions are kept.
        self.add_sample(b"alpha", "cat", "alpha.jpg")
        self.add_sample(b"beta", "dog", "beta.jpg")
        self.store.create_split("dups", 7, ["1", "0", "0"])
        _result, target = self.export("dups", target=self.base / "dups.zip",
                                      source_dir=self.moved)
        with zipfile.ZipFile(target) as archive:
            entries = {n: archive.read(n) for n in archive.namelist()
                       if n.endswith(".jpg")}
        self.assertIn(b"alpha", entries.values())
        self.assertIn(b"beta", entries.values())

    def test_deep_symlinked_directory_is_not_descended(self) -> None:
        self.add_sample(b"needle", "cat", "needle.jpg")
        self.store.create_split("deep", 1, ["1", "0", "0"])
        leaf, _branch = self._build_moved_tree()
        outside = self.base / "outside"
        outside.mkdir()
        # Only a symlinked directory deep in the tree carries the bytes.
        os.symlink(outside, leaf.parent / "linkdir")
        (outside / "needle").write_bytes(b"needle")
        # A symlink to a matching regular file at depth is skipped too.
        (outside / "needle2").write_bytes(b"needle")
        os.symlink(outside / "needle2", leaf / "linkfile")

        target = self.base / "out.zip"
        with self.assertRaises(ExportError) as caught:
            self.export(target=target, source_dir=self.moved)
        self.assertIn("no file with matching content", str(caught.exception))
        self.assertFalse(target.exists())

    # ------------------------------------------------------------------
    # Failure rules still hold at depth
    # ------------------------------------------------------------------

    def test_unreadable_directory_deep_fails_naming_path_even_if_found(self) -> None:
        self.add_sample(b"root-bytes", "cat", "root.jpg")
        self.store.create_split("deep", 1, ["1", "0", "0"])
        leaf, _branch = self._build_moved_tree()
        # The planned image at the root is already present, but a deep
        # directory that cannot be scanned must still fail the export.
        blocked = leaf.parent.parent
        self._deny(blocked)
        target = self.base / "out.zip"
        with self.assertRaises(ExportError) as caught:
            self.export(target=target, source_dir=self.moved)
        self.assertIn(str(blocked), str(caught.exception))
        self.assertIn("cannot scan directory", str(caught.exception))
        self.assertFalse(target.exists())

    def test_unreadable_file_deep_fails_naming_path_even_if_found(self) -> None:
        self.add_sample(b"root-bytes", "cat", "root.jpg")
        self.store.create_split("deep", 1, ["1", "0", "0"])
        leaf, _branch = self._build_moved_tree()
        bad = leaf / "secret"
        bad.write_bytes(b"cannot read me")
        self._deny(bad)
        target = self.base / "out.zip"
        with self.assertRaises(ExportError) as caught:
            self.export(target=target, source_dir=self.moved)
        self.assertIn(str(bad), str(caught.exception))
        self.assertIn("cannot read source file", str(caught.exception))
        self.assertFalse(target.exists())

    def test_confirmed_deep_directory_replaced_during_scan_fails(self) -> None:
        self.add_sample(b"leaf-bytes", "cat", "leaf.jpg")
        self.store.create_split("deep", 1, ["1", "0", "0"])
        leaf, _branch = self._build_moved_tree()
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "leaf.unknown").write_bytes(b"leaf-bytes")

        def mutate() -> None:
            moved_away = leaf.with_name(leaf.name + "-moved")
            os.rename(leaf, moved_away)
            leaf.symlink_to(outside)

        # After the deep leaf's first file was hashed and before the
        # leaf frame's post-child binding check.  The leaf holds more
        # than one entry, so the swap fires exactly once.
        real_hash = exporter_mod._hash_regular_file_at
        fired = {"done": False}

        def hash_wrap(directory_fd, name, path, *args, **kwargs):
            result = real_hash(directory_fd, name, path, *args, **kwargs)
            if not fired["done"] and Path(path).parent == leaf:
                fired["done"] = True
                mutate()
            return result

        target = self.base / "out.zip"
        with patch.object(
            exporter_mod, "_hash_regular_file_at", side_effect=hash_wrap
        ):
            with self.assertRaises(ExportError) as caught:
                self.export(target=target, source_dir=self.moved)
        self.assertTrue(fired["done"])
        message = str(caught.exception)
        self.assertIn(str(leaf), message)
        self.assertIn("symlink", message)
        self.assertFalse(target.exists())

    def test_deep_directory_swap_after_scan_fails_before_copy(self) -> None:
        self.add_sample(b"leaf-bytes", "cat", "leaf.jpg")
        self.store.create_split("deep", 1, ["1", "0", "0"])
        leaf, _branch = self._build_moved_tree()
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "leaf.unknown").write_bytes(b"leaf-bytes")

        def mutate() -> None:
            moved_away = leaf.with_name(leaf.name + "-moved")
            os.rename(leaf, moved_away)
            leaf.symlink_to(moved_away)

        real_scan = exporter_mod._scan_source_directory

        def scan_wrap(root):
            resolver = real_scan(root)
            mutate()
            return resolver

        target = self.base / "out.zip"
        with patch.object(
            exporter_mod, "_scan_source_directory", side_effect=scan_wrap
        ):
            with self.assertRaises(ExportError) as caught:
                self.export(target=target, source_dir=self.moved)
        message = str(caught.exception)
        self.assertIn(str(leaf), message)
        self.assertIn("symlink", message)
        self.assertFalse(target.exists())

    def test_deep_directory_swap_during_read_fails_final_gate(self) -> None:
        self.add_sample(b"leaf-bytes", "cat", "leaf.jpg")
        self.store.create_split("deep", 1, ["1", "0", "0"])
        leaf, _branch = self._build_moved_tree()
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "leaf.unknown").write_bytes(b"leaf-bytes")

        def mutate() -> None:
            moved_away = leaf.with_name(leaf.name + "-moved")
            os.rename(leaf, moved_away)
            leaf.symlink_to(moved_away)

        # The selected file is opened for the copy; while it streams, the
        # deep directory is swapped.  The pinned file bytes may be read,
        # but the iterative whole-tree gate before publication must fail.
        real_resolve = exporter_mod._open_resolved_source
        fired = {"done": False}

        def resolve_wrap(source, digest, device, inode):
            stream, size = real_resolve(source, digest, device, inode)
            if not fired["done"] and Path(source).parent == leaf:
                fired["done"] = True
                mutate()
            return stream, size

        target = self.base / "out.zip"
        with patch.object(
            exporter_mod, "_open_resolved_source", side_effect=resolve_wrap
        ):
            with self.assertRaises(ExportError) as caught:
                self.export(target=target, source_dir=self.moved)
        self.assertTrue(fired["done"])
        message = str(caught.exception)
        self.assertIn(str(leaf), message)
        self.assertIn("symlink", message)
        self.assertFalse(target.exists())
        self.assertEqual(
            [p.name for p in self.base.iterdir() if p.name.endswith(".tmp")],
            [],
        )

    def test_deep_export_does_not_leak_file_descriptors(self) -> None:
        self.add_sample(b"root-bytes", "cat", "root.jpg")
        self.add_sample(b"leaf-bytes", "dog", "leaf.jpg")
        self.add_sample(b"branch-bytes", "bird", "branch.jpg")
        self.store.create_split("deep", 1, ["1", "0", "0"])
        self._build_moved_tree()
        descriptor_dir = f"/proc/{os.getpid()}/fd"
        before = len(os.listdir(descriptor_dir))
        _result, target = self.export(source_dir=self.moved)
        self.assertTrue(target.exists())
        after = len(os.listdir(descriptor_dir))
        self.assertEqual(after, before)

    # ------------------------------------------------------------------
    # Empty / fully skipped plans only need the directory to exist
    # ------------------------------------------------------------------

    def test_empty_plan_does_not_scan_deep_unreadable_content(self) -> None:
        self.store.create_split("empty", 1, ["1", "0", "0"])
        leaf = self._make_chain(self.moved, self.DEPTH)
        bad_dir = leaf.parent / "secret-dir"
        os.mkdir(bad_dir)
        (bad_dir / "x").write_bytes(b"x")
        self._deny(bad_dir)
        bad_file = leaf / "secret"
        bad_file.write_bytes(b"y")
        self._deny(bad_file)
        result, target = self.export("empty", source_dir=self.moved)
        self.assertEqual(result["exported"], 0)
        self.assertTrue(target.exists())

    def test_skip_unlabeled_does_not_scan_deep_unreadable_content(self) -> None:
        self.add_sample(b"unlabeled", None, "u.jpg")
        self.store.create_split("p", 1, ["1", "0", "0"])
        leaf = self._make_chain(self.moved, self.DEPTH)
        bad = leaf / "secret"
        bad.write_bytes(b"blocked")
        self._deny(bad)
        result, target = self.export(
            "p", target=self.base / "skipped.zip",
            skip_unlabeled=True, source_dir=self.moved,
        )
        self.assertEqual(result["exported"], 0)
        self.assertEqual(
            result["skipped"], {"train": 1, "validation": 0, "test": 0}
        )
        self.assertTrue(target.exists())

    def test_empty_plan_still_requires_real_directory(self) -> None:
        self.store.create_split("empty", 1, ["1", "0", "0"])
        target = self.base / "out.zip"
        with self.assertRaises(ExportError):
            self.export("empty", target=target,
                        source_dir=self.base / "missing")
        self.assertFalse(target.exists())

    # ------------------------------------------------------------------
    # End to end
    # ------------------------------------------------------------------

    def test_cli_deep_tree_export_succeeds(self) -> None:
        self.add_sample(b"root-bytes", "cat", "root.jpg")
        self.add_sample(b"leaf-bytes", "dog", "leaf.jpg")
        self.add_sample(b"branch-bytes", "bird", "branch.jpg")
        self.store.create_split("deep", 1, ["1", "0", "0"])
        self._build_moved_tree()
        target = self.base / "cli.zip"
        result = self.run_cli(
            "export", str(self.workspace), "deep", str(target),
            "--source-dir", str(self.moved),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        body = json.loads(result.stdout)
        self.assertEqual(body["exported"], 3)
        self.assertTrue(target.exists())
        with zipfile.ZipFile(target) as archive:
            names = [n for n in archive.namelist() if n.endswith(".jpg")]
            self.assertEqual(len(names), 3)


if __name__ == "__main__":
    unittest.main()
