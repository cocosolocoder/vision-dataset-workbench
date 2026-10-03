from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from vision_workbench.exporter import (
    ExportError,
    _scan_source_directory,
    _write_sample,
    export_split,
)
from vision_workbench.store import DatasetStore

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

    def test_replacement_with_identical_metadata_fails(self) -> None:
        # A replacement with the same content, size and modification time
        # is still a different file and must be rejected: the digest
        # recheck alone cannot tell the two copies apart.
        path = self.root.parent / "f.jpg"
        path.write_bytes(b"abc")
        digest = hashlib.sha256(b"abc").hexdigest()
        stat_result = os.lstat(path)
        resolver = {digest: (path, stat_result.st_dev, stat_result.st_ino)}
        replacement = self.root.parent / "replacement.jpg"
        replacement.write_bytes(b"abc")
        os.utime(replacement, ns=(stat_result.st_atime_ns, stat_result.st_mtime_ns))
        os.replace(replacement, path)
        self.assertEqual(os.lstat(path).st_mtime_ns, stat_result.st_mtime_ns)
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
        message = str(caught.exception)
        self.assertIn(digest, message)
        self.assertIn(str(path), message)
        self.assertIn("replaced after lookup", message)

    def test_symlink_at_selected_path_fails(self) -> None:
        # Even a symlink pointing at the original, still-intact file is
        # not the confirmed regular file and must be rejected.
        original = self.root.parent / "original.jpg"
        original.write_bytes(b"abc")
        digest = hashlib.sha256(b"abc").hexdigest()
        path = self.root.parent / "f.jpg"
        path.write_bytes(b"abc")
        stat_result = os.lstat(path)
        resolver = {digest: (path, stat_result.st_dev, stat_result.st_ino)}
        path.unlink()
        path.symlink_to(original)
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
        message = str(caught.exception)
        self.assertIn(digest, message)
        self.assertIn(str(path), message)
        self.assertIn("replaced after lookup", message)

    def test_export_fails_when_selected_copy_replaced_before_read(self) -> None:
        # End to end: the sorted-first copy found by the directory scan is
        # replaced by an identical-content file before the copy runs, so
        # the whole export fails and leaves no target.
        d = self.add_sample("cat", b"abc")
        self.create_plan()
        moved = self.root.parent / "moved"
        moved.mkdir()
        selected = moved / "a.jpg"
        selected.write_bytes(b"abc")
        resolver = _scan_source_directory(moved.resolve())
        self.assertEqual(resolver[d][0], selected.resolve())
        replacement = moved / "replacement.jpg"
        replacement.write_bytes(b"abc")
        os.replace(replacement, selected)
        target = self.root.parent / "out.zip"
        member = {
            "sha256": d,
            "label": "cat",
            "set": "train",
            "source": str(selected),
        }
        import io

        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            with self.assertRaises(ExportError) as caught:
                _write_sample(archive, "train/class_01/x", member, resolver)
        self.assertIn(d, str(caught.exception))
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
