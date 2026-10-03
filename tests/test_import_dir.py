"""Whole-directory image import tests.

A directory import registers every accepted image file (``.jpg``, ``.jpeg``,
``.png``, ``.bmp``, ``.gif``, ``.webp``, case-insensitive extension) under a
source directory in one atomic batch: symlinks and other entry types are
skipped, recursion never descends through symlinked directories, candidates
are sorted by slash-separated relative path in Unicode code point order,
duplicate digests keep the first registration, and a label applies only to
genuinely new samples.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from vision_workbench.store import DatasetStore

REPO_ROOT = Path(__file__).resolve().parents[1]

ctx = multiprocessing.get_context("fork")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _import_worker(root: str, source: str, label, recursive: bool, queue) -> None:
    try:
        result = DatasetStore(Path(root)).import_directory(
            Path(source), label, recursive
        )
        queue.put(
            (
                "ok",
                result["added"],
                result["duplicates"],
                result["candidate_count"],
            )
        )
    except Exception as error:  # noqa: BLE001 - report back to parent
        queue.put(("error", str(error), 0, 0))


def _add_worker(root: str, source: str, label, queue) -> None:
    try:
        result = DatasetStore(Path(root)).add(Path(source), label)
        queue.put(("ok", 1 if result.added else 0, 0, 0))
    except Exception as error:  # noqa: BLE001
        queue.put(("error", 0, 0, 0))


def _batch_worker(root: str, number: str, changes: list, queue) -> None:
    try:
        result = DatasetStore(Path(root)).submit_batch(
            {"batch": number, "changes": changes}
        )
        queue.put(("ok", result["status"], 0, 0))
    except Exception as error:  # noqa: BLE001
        queue.put(("error", str(error), 0, 0))


def _drain(queue, count: int) -> list:
    return [queue.get(timeout=30) for _ in range(count)]


class DirectoryImportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.workspace = self.base / "dataset"
        self.source = self.base / "images"
        self.source.mkdir()
        self.store = DatasetStore(self.workspace)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write(self, relative: str, content: bytes) -> Path:
        path = self.source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_filters_extensions_case_insensitive_and_ignores_other_files(self) -> None:
        accepted = ["a.JPG", "b.Jpeg", "c.png", "d.BMP", "e.gif", "f.webp"]
        for name in accepted:
            self._write(name, f"content-{name}".encode())
        self._write("notes.txt", b"not an image")
        self._write("archive.zip", b"not an image")
        # No extension at all: not accepted.
        self._write("noext", b"no extension")

        result = self.store.import_directory(self.source, "cat")

        self.assertEqual(result["candidate_count"], len(accepted))
        self.assertEqual(result["added"], len(accepted))
        self.assertEqual(result["duplicates"], 0)
        self.assertEqual(
            [candidate["path"] for candidate in result["candidates"]],
            sorted(accepted),
        )
        self.assertEqual(self.store.summary()["items"], len(accepted))

    def test_non_recursive_by_default_recursive_flag_includes_subdirs(self) -> None:
        self._write("top.jpg", b"top")
        self._write("sub/nested.png", b"nested")
        self._write("sub/deep/below.gif", b"below")

        top_only = self.store.import_directory(self.source, "cat")
        self.assertEqual(top_only["candidate_count"], 1)
        self.assertEqual(
            [c["path"] for c in top_only["candidates"]], ["top.jpg"]
        )

        # A second store on a fresh workspace for the recursive run.
        workspace2 = self.base / "dataset2"
        result = DatasetStore(workspace2).import_directory(
            self.source, "cat", recursive=True
        )
        self.assertEqual(result["candidate_count"], 3)
        self.assertEqual(
            [c["path"] for c in result["candidates"]],
            ["sub/deep/below.gif", "sub/nested.png", "top.jpg"],
        )
        self.assertEqual(result["added"], 3)

    def test_candidates_sorted_by_slash_relative_path_in_code_point_order(self) -> None:
        self._write("Z.jpg", b"z")
        self._write("a.jpg", b"a")
        self._write("A.jpg", b"upper-a")
        self._write("sub/b.jpg", b"b")
        self._write("sub/a.jpg", b"sub-a")
        self._write("世界.jpg", b"cjk")

        result = self.store.import_directory(self.source, recursive=True)
        paths = [candidate["path"] for candidate in result["candidates"]]
        self.assertEqual(paths, sorted(paths))
        # 'A' (U+0041) sorts before 'Z' before 'a' (U+0061); CJK after ASCII.
        self.assertEqual(
            paths,
            ["A.jpg", "Z.jpg", "a.jpg", "sub/a.jpg", "sub/b.jpg", "世界.jpg"],
        )
        # Every candidate carries its full digest and a status.
        for candidate in result["candidates"]:
            self.assertRegex(candidate["sha256"], r"[0-9a-f]{64}")
            self.assertIn(candidate["status"], ("added", "duplicate"))
        self.assertEqual(
            result["candidate_count"], result["added"] + result["duplicates"]
        )

    def test_symlinked_files_and_directories_are_skipped(self) -> None:
        self._write("real.jpg", b"real")
        self._write("sub/inside.png", b"inside")
        # Symlink to a file inside the tree: never a candidate.
        os.symlink(self.source / "real.jpg", self.source / "link.jpg")
        # Symlink to a directory: never descended into.
        os.symlink(self.source / "sub", self.source / "sub-link")
        # A FIFO (other entry type) is ignored as well.
        os.mkfifo(self.source / "pipe.jpg")

        result = self.store.import_directory(self.source, "cat", recursive=True)
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(
            [c["path"] for c in result["candidates"]],
            ["real.jpg", "sub/inside.png"],
        )
        self.assertEqual(result["added"], 2)

    def test_duplicate_content_in_batch_first_candidate_wins(self) -> None:
        self._write("a.jpg", b"same-bytes")
        self._write("b.jpg", b"same-bytes")
        self._write("c.png", b"same-bytes")
        digest = _sha256(b"same-bytes")

        result = self.store.import_directory(self.source, "cat")
        self.assertEqual(result["added"], 1)
        self.assertEqual(result["duplicates"], 2)
        statuses = {c["path"]: c["status"] for c in result["candidates"]}
        self.assertEqual(statuses, {"a.jpg": "added", "b.jpg": "duplicate",
                                    "c.png": "duplicate"})
        manifest = json.loads(
            (self.workspace / ".vision-workbench" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(manifest["items"]), 1)
        item = manifest["items"][0]
        self.assertEqual(item["sha256"], digest)
        # The sorted-first candidate owns the source and label.
        self.assertEqual(Path(item["source"]).name, "a.jpg")
        self.assertEqual(item["label"], "cat")

    def test_existing_digest_always_duplicate_keeps_original_record(self) -> None:
        original = self.base / "original.jpg"
        original.write_bytes(b"existing-content")
        digest = self.store.add(original, "dog").digest

        # Same content under a new path with a different label.
        self._write("copy.jpg", b"existing-content")
        result = self.store.import_directory(self.source, "cat")
        self.assertEqual(result["added"], 0)
        self.assertEqual(result["duplicates"], 1)
        self.assertEqual(result["candidates"][0]["status"], "duplicate")

        manifest = json.loads(
            (self.workspace / ".vision-workbench" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(manifest["items"]), 1)
        item = manifest["items"][0]
        self.assertEqual(item["sha256"], digest)
        self.assertEqual(item["label"], "dog")
        self.assertEqual(Path(item["source"]).name, "original.jpg")
        # Undo eligibility (revision history) is untouched.
        self.assertEqual(item.get("rev", 0), 0)

    def test_label_applied_only_to_new_samples(self) -> None:
        self._write("new.jpg", b"new")
        self._write("dup.jpg", b"existing")
        (self.base / "seed.jpg").write_bytes(b"existing")
        self.store.add(self.base / "seed.jpg", "bird")

        result = self.store.import_directory(self.source, "kitten")
        statuses = {c["path"]: c["status"] for c in result["candidates"]}
        self.assertEqual(statuses["new.jpg"], "added")
        self.assertEqual(statuses["dup.jpg"], "duplicate")
        manifest = json.loads(
            (self.workspace / ".vision-workbench" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        by_digest = {item["sha256"]: item for item in manifest["items"]}
        self.assertEqual(by_digest[_sha256(b"new")]["label"], "kitten")
        self.assertEqual(by_digest[_sha256(b"existing")]["label"], "bird")

    def test_missing_label_and_empty_string_mean_unlabeled(self) -> None:
        self._write("a.jpg", b"a")
        self._write("b.jpg", b"b")

        store_no_label = DatasetStore(self.base / "ws-no-label")
        result = store_no_label.import_directory(self.source)
        self.assertIsNone(result["label"])
        self.assertEqual(store_no_label.summary()["labels"], {"unlabeled": 2})

        store_empty = DatasetStore(self.base / "ws-empty")
        result = store_empty.import_directory(self.source, "")
        self.assertEqual(result["label"], "")
        self.assertEqual(store_empty.summary()["labels"], {"unlabeled": 2})

    def test_literal_label_including_chinese_kept(self) -> None:
        self._write("a.jpg", b"a")
        result = self.store.import_directory(self.source, "猫")
        self.assertEqual(result["label"], "猫")
        self.assertEqual(self.store.summary()["labels"], {"猫": 1})

    def test_empty_directory_succeeds_with_zero_added(self) -> None:
        empty = self.base / "empty"
        empty.mkdir()
        result = DatasetStore(self.base / "ws").import_directory(empty, "cat")
        self.assertEqual(result["candidate_count"], 0)
        self.assertEqual(result["added"], 0)
        self.assertEqual(result["duplicates"], 0)
        self.assertEqual(result["candidates"], [])
        self.assertEqual(DatasetStore(self.base / "ws").summary()["items"], 0)

    def test_reimporting_same_directory_reports_all_duplicates(self) -> None:
        self._write("a.jpg", b"a")
        self._write("b.jpg", b"b")
        first = self.store.import_directory(self.source, "cat")
        self.assertEqual(first["added"], 2)

        second = self.store.import_directory(self.source, "cat")
        self.assertEqual(second["added"], 0)
        self.assertEqual(second["duplicates"], 2)
        self.assertEqual(
            [c["status"] for c in second["candidates"]],
            ["duplicate", "duplicate"],
        )
        self.assertEqual(self.store.summary()["items"], 2)

    def test_result_directory_is_absolute(self) -> None:
        self._write("a.jpg", b"a")
        result = self.store.import_directory(self.source, "cat")
        self.assertTrue(Path(result["directory"]).is_absolute())
        self.assertEqual(Path(result["directory"]), self.source.resolve())

    def test_missing_source_fails_with_path_and_reason(self) -> None:
        missing = self.base / "does-not-exist"
        with self.assertRaises(ValueError) as context:
            self.store.import_directory(missing, "cat")
        self.assertIn(str(missing), str(context.exception))
        self.assertEqual(self.store.summary()["items"], 0)
        self.assertFalse(self.store.transaction_path.exists())

    def test_source_that_is_a_file_fails(self) -> None:
        source_file = self.base / "source.jpg"
        source_file.write_bytes(b"not-a-dir")
        with self.assertRaises(ValueError) as context:
            self.store.import_directory(source_file, "cat")
        self.assertIn("not a directory", str(context.exception))
        self.assertEqual(self.store.summary()["items"], 0)

    def test_unreadable_subdirectory_fails_whole_batch(self) -> None:
        self._write("top.jpg", b"top")
        locked = self.source / "locked"
        locked.mkdir()
        (locked / "inside.jpg").write_bytes(b"inside")
        try:
            os.chmod(locked, 0o000)
            with self.assertRaises(ValueError) as context:
                self.store.import_directory(self.source, "cat", recursive=True)
            self.assertIn("cannot scan directory", str(context.exception))
            self.assertIn(str(locked), str(context.exception))
            # No partial batch landed.
            self.assertEqual(self.store.summary()["items"], 0)
            self.assertFalse(self.store.transaction_path.exists())
        finally:
            os.chmod(locked, 0o755)

    def test_unreadable_candidate_fails_whole_batch(self) -> None:
        self._write("good.jpg", b"good")
        bad = self._write("bad.jpg", b"bad")
        try:
            os.chmod(bad, 0o000)
            with self.assertRaises(ValueError) as context:
                self.store.import_directory(self.source, "cat")
            self.assertIn("cannot read", str(context.exception))
            self.assertIn("bad.jpg", str(context.exception))
            self.assertEqual(self.store.summary()["items"], 0)
            self.assertFalse(self.store.transaction_path.exists())
        finally:
            os.chmod(bad, 0o644)
        # The good file was not registered either: the batch is all-or-nothing.
        self.assertEqual(self.store.summary()["items"], 0)
        self.assertFalse(self.store.transaction_path.exists())

    def test_file_changed_during_read_fails_whole_batch(self) -> None:
        target = self._write("a.jpg", b"original")
        self._write("b.jpg", b"b")
        real_lstat = os.lstat
        state = {"calls": 0}

        def swap_on_first_lstat(path, *args, **kwargs):
            result = real_lstat(path, *args, **kwargs)
            if Path(path) == target and state["calls"] == 0:
                state["calls"] += 1
                # Replace the file after its before-stat, before the read.
                target.write_bytes(b"changed-content")
            return result

        with mock.patch("vision_workbench.store.os.lstat", swap_on_first_lstat):
            with self.assertRaises(ValueError) as context:
                self.store.import_directory(self.source, "cat")
        self.assertIn("file changed during import", str(context.exception))
        self.assertEqual(self.store.summary()["items"], 0)
        self.assertFalse(self.store.transaction_path.exists())

    def test_subdirectory_swapped_for_symlink_before_descend_fails(self) -> None:
        self._write("top.jpg", b"top")
        self._write("sub/inside.jpg", b"inside")
        elsewhere = self.base / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "other.jpg").write_bytes(b"other")
        real_scandir = os.scandir
        state = {"done": False}

        def swap_on_root_scan(path, *args, **kwargs):
            entries = list(real_scandir(path, *args, **kwargs))
            if not state["done"] and Path(path) == self.source.resolve():
                state["done"] = True
                # Swap the confirmed subdirectory for a symlink after the
                # root scan but before the walk descends into it.
                moved = self.base / "moved-away"
                os.rename(self.source / "sub", moved)
                os.symlink(elsewhere, self.source / "sub")
            return entries

        with mock.patch("vision_workbench.store.os.scandir", swap_on_root_scan):
            with self.assertRaises(ValueError) as context:
                self.store.import_directory(self.source, "cat", recursive=True)
        message = str(context.exception)
        self.assertIn(str(self.source / "sub"), message)
        self.assertIn("symbolic link", message)
        # No partial batch landed: neither the real nor the linked images.
        self.assertEqual(self.store.summary()["items"], 0)
        self.assertFalse(self.store.transaction_path.exists())

    def test_subdirectory_swapped_for_symlink_during_read_fails(self) -> None:
        # The link target may live outside the source tree, inside it, or
        # be the moved-away original directory; every variant fails.
        for variant in ("outside", "inside", "moved-away"):
            with self.subTest(variant=variant):
                source = self.base / f"src-{variant}"
                (source / "sub").mkdir(parents=True)
                (source / "sub" / "inside.jpg").write_bytes(b"inside")
                (source / "top.jpg").write_bytes(b"top")
                store = DatasetStore(self.base / f"ws-{variant}")
                # A digest the workspace already knows: a target whose
                # images are registered duplicates cannot save the batch.
                seed = self.base / f"seed-{variant}.jpg"
                seed.write_bytes(b"seed-content")
                digest = store.add(seed, "bird").digest

                if variant == "inside":
                    target = source / "sibling"
                    target.mkdir()
                else:
                    target = self.base / f"target-{variant}"
                    target.mkdir()
                (target / "other.jpg").write_bytes(b"seed-content")
                # A same-named image with the same bytes: reading through
                # the link would "succeed", so the swap must be detected.
                (target / "inside.jpg").write_bytes(b"inside")

                real_read = store._read_candidate
                state = {"done": False}

                def swap_after_first_read(abs_path, rel_path):
                    result = real_read(abs_path, rel_path)
                    if not state["done"]:
                        state["done"] = True
                        subdir = source / "sub"
                        if variant == "moved-away":
                            moved = self.base / f"moved-{variant}"
                            os.rename(subdir, moved)
                            os.symlink(moved, subdir)
                        else:
                            shutil.rmtree(subdir)
                            os.symlink(target, subdir)
                    return result

                with mock.patch.object(
                    store, "_read_candidate", swap_after_first_read
                ):
                    with self.assertRaises(ValueError) as context:
                        store.import_directory(source, "cat", recursive=True)
                message = str(context.exception)
                self.assertIn(str(source / "sub"), message)
                self.assertIn("symbolic link", message)
                # Nothing new registered; the pre-existing sample is intact.
                self.assertEqual(store.summary()["items"], 1)
                self.assertEqual(store.lookup_label(digest)["label"], "bird")
                self.assertFalse(store.transaction_path.exists())

    def test_failed_import_preserves_labels_history_and_splits(self) -> None:
        seed = self.base / "seed.jpg"
        seed.write_bytes(b"seed")
        digest = self.store.add(seed, "bird").digest
        self.store.submit_batch(
            {"batch": "b1", "changes": [{"sha256": digest, "old": "bird", "new": "eagle"}]}
        )
        plan = self.store.create_split("baseline", 7, [1, 0, 0])
        self._write("sub/inside.jpg", b"inside")
        real_read = self.store._read_candidate
        state = {"done": False}

        def swap_after_first_read(abs_path, rel_path):
            result = real_read(abs_path, rel_path)
            if not state["done"]:
                state["done"] = True
                moved = self.base / "moved-away"
                os.rename(self.source / "sub", moved)
                os.symlink(moved, self.source / "sub")
            return result

        with mock.patch.object(self.store, "_read_candidate", swap_after_first_read):
            with self.assertRaises(ValueError):
                self.store.import_directory(self.source, "cat", recursive=True)

        # Existing samples, labels, batch history and split plans are as
        # they were; the failed batch added nothing.
        self.assertEqual(self.store.summary()["items"], 1)
        self.assertEqual(self.store.lookup_label(digest)["label"], "eagle")
        self.assertEqual([entry["batch"] for entry in self.store.history()], ["b1"])
        self.assertEqual(self.store.undo_batch("b1")["status"], "undone")
        self.assertEqual(self.store.get_split("baseline"), plan.plan)

    def test_non_recursive_import_unaffected_by_subdirectory_swap(self) -> None:
        self._write("top.jpg", b"top")
        self._write("sub/inside.jpg", b"inside")
        real_read = self.store._read_candidate
        state = {"done": False}

        def swap_after_first_read(abs_path, rel_path):
            result = real_read(abs_path, rel_path)
            if not state["done"]:
                state["done"] = True
                moved = self.base / "moved-away"
                os.rename(self.source / "sub", moved)
                os.symlink(moved, self.source / "sub")
            return result

        with mock.patch.object(self.store, "_read_candidate", swap_after_first_read):
            result = self.store.import_directory(self.source, "cat")
        self.assertEqual(result["added"], 1)
        self.assertEqual([c["path"] for c in result["candidates"]], ["top.jpg"])

    def test_new_files_appearing_after_scan_are_left_for_next_import(self) -> None:
        self._write("a.jpg", b"a")
        real_read = self.store._read_candidate
        state = {"done": False}

        def read_then_add(abs_path, rel_path):
            result = real_read(abs_path, rel_path)
            if not state["done"]:
                state["done"] = True
                # Appears after the scan finished; must not be in this batch.
                self._write("z-late.jpg", b"late")
            return result

        with mock.patch.object(self.store, "_read_candidate", read_then_add):
            result = self.store.import_directory(self.source, "cat")
        self.assertEqual(result["added"], 1)
        self.assertEqual(
            [c["path"] for c in result["candidates"]], ["a.jpg"]
        )

        second = self.store.import_directory(self.source, "cat")
        self.assertEqual(second["added"], 1)
        self.assertEqual(second["duplicates"], 1)

    def test_commit_is_one_manifest_state_for_splits(self) -> None:
        self._write("a.jpg", b"a")
        before = self.store.create_split("before", 1, [1, 0, 0])
        self.assertEqual(before.plan["samples"]["total"], 0)

        result = self.store.import_directory(self.source, "cat")
        self.assertEqual(result["added"], 1)

        # The plan created before the batch is untouched.
        self.assertEqual(self.store.get_split("before")["samples"]["total"], 0)
        after = self.store.create_split("after", 2, [1, 0, 0])
        self.assertEqual(after.plan["samples"]["total"], 1)


class DirectoryImportConcurrencyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.workspace = self.base / "dataset"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _make_dir(self, name: str, count: int, content_prefix: bytes) -> Path:
        directory = self.base / name
        directory.mkdir()
        for index in range(count):
            (directory / f"img-{index}.jpg").write_bytes(
                content_prefix + f"-{index}".encode()
            )
        return directory

    def test_concurrent_imports_of_different_directories_all_added(self) -> None:
        dirs = [self._make_dir(f"dir-{i}", 3, f"d{i}".encode()) for i in range(4)]
        queue = ctx.Queue()
        processes = [
            ctx.Process(
                target=_import_worker,
                args=(str(self.workspace), str(directory), f"label-{i}", False, queue),
            )
            for i, directory in enumerate(dirs)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
            self.assertEqual(process.exitcode, 0)

        results = _drain(queue, len(dirs))
        self.assertTrue(all(row[0] == "ok" for row in results), results)
        self.assertEqual(sum(row[1] for row in results), 12)
        self.assertEqual(DatasetStore(self.workspace).summary()["items"], 12)

    def test_concurrent_imports_of_same_directory_serialize(self) -> None:
        directory = self._make_dir("same", 4, b"same")
        queue = ctx.Queue()
        processes = [
            ctx.Process(
                target=_import_worker,
                args=(str(self.workspace), str(directory), f"label-{i}", False, queue),
            )
            for i in range(4)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
            self.assertEqual(process.exitcode, 0)

        results = _drain(queue, 4)
        self.assertTrue(all(row[0] == "ok" for row in results), results)
        added = sum(row[1] for row in results)
        duplicates = sum(row[2] for row in results)
        # Exactly one process's adds survive; the rest report duplicates.
        self.assertEqual(added, 4)
        self.assertEqual(duplicates, 12)
        self.assertEqual(DatasetStore(self.workspace).summary()["items"], 4)

    def test_directory_import_crossing_a_batch_keeps_both(self) -> None:
        store = DatasetStore(self.workspace)
        seed = self.base / "seed.jpg"
        seed.write_bytes(b"seed-content")
        digest = store.add(seed, "cat").digest
        directory = self._make_dir("cross", 5, b"cross")

        queue = ctx.Queue()
        workers = [
            ctx.Process(
                target=_batch_worker,
                args=(str(self.workspace), "b1",
                      [{"sha256": digest, "old": "cat", "new": "dog"}], queue),
            ),
            ctx.Process(
                target=_import_worker,
                args=(str(self.workspace), str(directory), "new", False, queue),
            ),
        ]
        for process in workers:
            process.start()
        for process in workers:
            process.join()
            self.assertEqual(process.exitcode, 0)

        results = _drain(queue, 2)
        self.assertTrue(all(row[0] == "ok" for row in results), results)
        reopened = DatasetStore(self.workspace)
        self.assertEqual(reopened.summary()["items"], 6)
        self.assertEqual(reopened.lookup_label(digest)["label"], "dog")
        self.assertEqual([e["batch"] for e in reopened.history()], ["b1"])
        # The batch's revision is intact: its undo remains eligible.
        self.assertEqual(reopened.undo_batch("b1")["status"], "undone")
        self.assertEqual(reopened.lookup_label(digest)["label"], "cat")

    def test_directory_import_vs_single_add_of_same_content(self) -> None:
        directory = self._make_dir("mix", 3, b"mix")
        single = self.base / "single.jpg"
        single.write_bytes(b"mix-1")  # same bytes as one directory candidate

        queue = ctx.Queue()
        workers = [
            ctx.Process(
                target=_import_worker,
                args=(str(self.workspace), str(directory), "dir-label", False, queue),
            ),
            ctx.Process(
                target=_add_worker,
                args=(str(self.workspace), str(single), "single-label", queue),
            ),
        ]
        for process in workers:
            process.start()
        for process in workers:
            process.join()
            self.assertEqual(process.exitcode, 0)

        results = _drain(queue, 2)
        self.assertTrue(all(row[0] == "ok" for row in results), results)
        total_added = sum(row[1] for row in results)
        # Three directory files + one single file, one shared digest: 3 adds.
        self.assertEqual(total_added, 3)
        summary = DatasetStore(self.workspace).summary()
        self.assertEqual(summary["items"], 3)


class DirectoryImportCLITest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.workspace = self.base / "dataset"
        self.source = self.base / "images"
        self.source.mkdir()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write(self, relative: str, content: bytes) -> None:
        path = self.source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "vision_workbench", *arguments],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

    def test_cli_import_dir_json_and_recursive_flag(self) -> None:
        self._write("a.jpg", b"a")
        self._write("sub/b.png", b"b")

        result = self.run_cli(
            "import-dir", str(self.workspace), str(self.source),
            "--label", "cat", "--recursive",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["added"], 2)
        self.assertEqual(payload["duplicates"], 0)
        self.assertEqual(payload["candidate_count"], 2)
        self.assertEqual(
            [c["path"] for c in payload["candidates"]], ["a.jpg", "sub/b.png"]
        )
        self.assertEqual(payload["label"], "cat")

        summary = json.loads(
            self.run_cli("summary", str(self.workspace)).stdout
        )
        self.assertEqual(summary["items"], 2)
        self.assertEqual(summary["labels"], {"cat": 2})

    def test_cli_import_alias(self) -> None:
        self._write("a.jpg", b"a")
        result = self.run_cli(
            "import", str(self.workspace), str(self.source)
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["added"], 1)
        self.assertIsNone(payload["label"])

    def test_cli_failure_returns_error_exit_code(self) -> None:
        result = self.run_cli(
            "import-dir", str(self.workspace), str(self.base / "missing")
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("error:", result.stderr)
        self.assertIn("does not exist", result.stderr)


if __name__ == "__main__":
    unittest.main()
