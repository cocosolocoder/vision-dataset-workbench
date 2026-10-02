"""Tests for whole-directory image imports (``import_directory``).

Covers candidate discovery (extensions, regular files only, symlink
handling, recursion, ordering), content-based de-duplication inside one
batch and against the workspace, label handling, all-or-nothing failure,
the read-time identity guard, crash recovery and concurrency with other
operations.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
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


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class DirectoryScanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.workspace = self.base / "dataset"
        self.source = self.base / "images"
        self.source.mkdir()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _make(self, relative: str, content: bytes) -> Path:
        path = self.source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_only_image_extensions_case_insensitive_at_top_level(self) -> None:
        for name, content in [
            ("a.JPG", b"a"),
            ("b.jpeg", b"b"),
            ("c.PNG", b"c"),
            ("d.bmp", b"d"),
            ("e.Gif", b"e"),
            ("f.webp", b"f"),
            ("notes.txt", b"x"),
            ("image.jpg.bak", b"x"),
            ("noext", b"x"),
        ]:
            self._make(name, content)

        result = DatasetStore(self.workspace).import_directory(self.source)
        self.assertEqual(
            [entry["path"] for entry in result["results"]],
            ["a.JPG", "b.jpeg", "c.PNG", "d.bmp", "e.Gif", "f.webp"],
        )
        self.assertEqual(result["candidates"], 6)
        self.assertEqual(result["added"], 6)
        self.assertEqual(result["duplicates"], 0)
        self.assertEqual(result["candidates"], result["added"] + result["duplicates"])
        for entry, content in zip(
            result["results"], [b"a", b"b", b"c", b"d", b"e", b"f"]
        ):
            self.assertEqual(entry["sha256"], sha256_bytes(content))
            self.assertEqual(entry["status"], "added")

    def test_default_is_non_recursive_subdirectory_files_skipped(self) -> None:
        self._make("top.jpg", b"top")
        self._make("sub/nested.png", b"nested")
        self._make("sub/deep/down.gif", b"deep")

        result = DatasetStore(self.workspace).import_directory(self.source)
        self.assertEqual([e["path"] for e in result["results"]], ["top.jpg"])

    def test_recursive_descends_real_subdirectories_with_slash_paths(self) -> None:
        self._make("z.jpg", b"z")
        self._make("a/b.jpg", b"ab")
        self._make("a/c.png", b"ac")
        self._make("a/deep/d.webp", b"d")

        result = DatasetStore(self.workspace).import_directory(
            self.source, recursive=True
        )
        self.assertEqual(
            [e["path"] for e in result["results"]],
            ["a/b.jpg", "a/c.png", "a/deep/d.webp", "z.jpg"],
        )
        self.assertTrue(all("/" in e["path"] or e["path"] == "z.jpg" for e in result["results"]))

    def test_symlinked_files_and_directories_never_imported_or_entered(self) -> None:
        self._make("real.jpg", b"real")
        self._make("realdir/inside.png", b"inside")
        os.symlink(self.source / "real.jpg", self.source / "alias.jpg")
        os.symlink(
            self.source / "realdir",
            self.source / "linkdir",
            target_is_directory=True,
        )
        # A dangling symlink is ignored too.
        os.symlink(self.source / "missing.jpg", self.source / "dangling.jpeg")

        result = DatasetStore(self.workspace).import_directory(
            self.source, recursive=True
        )
        self.assertEqual(
            sorted(e["path"] for e in result["results"]),
            ["real.jpg", "realdir/inside.png"],
        )

    def test_special_file_entries_with_image_names_are_skipped(self) -> None:
        self._make("ok.jpg", b"ok")
        os.mkfifo(self.source / "pipe.png")
        os.symlink(self.source / "ok.jpg", self.source / "link.bmp")

        result = DatasetStore(self.workspace).import_directory(self.source)
        self.assertEqual([e["path"] for e in result["results"]], ["ok.jpg"])

    def test_candidates_sorted_by_unicode_code_point_relative_path(self) -> None:
        # Deliberately create out of order; directory read order must not
        # dictate the result order. Mixed case sorts by code point, so
        # uppercase precedes lowercase.
        for name in ["z.png", "a.png", "M.png", "b.png", "A.png"]:
            self._make(name, name.encode())

        result = DatasetStore(self.workspace).import_directory(self.source)
        self.assertEqual(
            [e["path"] for e in result["results"]],
            ["A.png", "M.png", "a.png", "b.png", "z.png"],
        )

    def test_empty_directory_is_a_successful_zero_import(self) -> None:
        result = DatasetStore(self.workspace).import_directory(self.source)
        self.assertEqual(
            result,
            {"candidates": 0, "added": 0, "duplicates": 0, "results": []},
        )
        self.assertEqual(DatasetStore(self.workspace).summary()["items"], 0)


class DirectoryValidationTest(unittest.TestCase):
    def test_missing_source_reports_path_and_reason(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "nope"
            with self.assertRaisesRegex(ValueError, str(missing)):
                DatasetStore(Path(directory) / "ws").import_directory(missing)

    def test_source_that_is_a_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "file.jpg").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "not a directory"):
                DatasetStore(base / "ws").import_directory(base / "file.jpg")


class DirectoryDedupTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.workspace = self.base / "dataset"
        self.source = self.base / "images"
        self.source.mkdir()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _make(self, relative: str, content: bytes) -> Path:
        path = self.source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_identical_content_in_batch_added_once_earliest_path_wins(self) -> None:
        # "a.jpg" sorts before "copy.jpg" before "z.jpg", so a.jpg is source.
        self._make("z.jpg", b"same")
        self._make("copy.jpg", b"same")
        self._make("a.jpg", b"same")
        unique_digest = sha256_bytes(b"same")

        result = DatasetStore(self.workspace).import_directory(self.source, "cat")
        statuses = {e["path"]: e["status"] for e in result["results"]}
        self.assertEqual(
            [e["path"] for e in result["results"]],
            ["a.jpg", "copy.jpg", "z.jpg"],
        )
        self.assertEqual(statuses["a.jpg"], "added")
        self.assertEqual(statuses["copy.jpg"], "duplicate")
        self.assertEqual(statuses["z.jpg"], "duplicate")
        self.assertEqual(result["added"], 1)
        self.assertEqual(result["duplicates"], 2)
        self.assertTrue(
            all(e["sha256"] == unique_digest for e in result["results"])
        )

        manifest = json.loads(
            (self.workspace / ".vision-workbench" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(manifest["items"]), 1)
        item = manifest["items"][0]
        self.assertEqual(item["sha256"], unique_digest)
        self.assertTrue(item["source"].endswith("a.jpg"))
        self.assertEqual(item["label"], "cat")
        self.assertEqual(item["size"], len(b"same"))

    def test_existing_digests_all_report_duplicate_and_keep_metadata(self) -> None:
        store = DatasetStore(self.workspace)
        original = self.base / "original.jpg"
        original.write_bytes(b"shared")
        digest = store.add(original, "dog").digest
        store.submit_batch(
            {
                "batch": "b1",
                "changes": [
                    {"sha256": digest, "old": "dog", "new": "bird"}
                ],
            }
        )

        # New directory paths hold the same content; import with a different
        # label must not overwrite anything.
        self._make("one.jpg", b"shared")
        self._make("sub/two.jpg", b"shared")
        result = store.import_directory(self.source, "cat", recursive=True)

        self.assertTrue(all(e["status"] == "duplicate" for e in result["results"]))
        self.assertEqual(result["added"], 0)
        self.assertEqual(result["duplicates"], 2)
        self.assertEqual(store.summary()["items"], 1)
        self.assertEqual(store.lookup_label(digest)["label"], "bird")
        manifest = json.loads(
            (self.workspace / ".vision-workbench" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(manifest["items"][0]["source"].endswith("original.jpg"))
        # The earlier batch's undo eligibility is untouched.
        self.assertEqual(store.undo_batch("b1")["status"], "undone")
        self.assertEqual(store.lookup_label(digest)["label"], "dog")

    def test_mixed_new_existing_and_in_batch_duplicates(self) -> None:
        store = DatasetStore(self.workspace)
        existing = self.base / "existing.jpg"
        existing.write_bytes(b"existing")
        existing_digest = store.add(existing, "old").digest

        self._make("existing-copy.jpg", b"existing")
        self._make("fresh1.jpg", b"fresh-1")
        self._make("fresh2.jpg", b"fresh-1")  # same content as fresh1
        self._make("fresh3.jpg", b"fresh-3")

        result = store.import_directory(self.source, "new")
        by_path = {e["path"]: e for e in result["results"]}
        self.assertEqual(
            by_path["existing-copy.jpg"]["status"], "duplicate"
        )
        self.assertEqual(by_path["existing-copy.jpg"]["sha256"], existing_digest)
        self.assertEqual(by_path["fresh1.jpg"]["status"], "added")
        self.assertEqual(by_path["fresh2.jpg"]["status"], "duplicate")
        self.assertEqual(by_path["fresh2.jpg"]["sha256"], by_path["fresh1.jpg"]["sha256"])
        self.assertEqual(by_path["fresh3.jpg"]["status"], "added")
        self.assertEqual(result["added"], 2)
        self.assertEqual(result["duplicates"], 2)

        labels = store.summary()["labels"]
        # The pre-existing sample keeps its label; new ones share "new".
        self.assertEqual(labels, {"new": 2, "old": 1})

    def test_reimporting_same_directory_is_idempotent(self) -> None:
        self._make("a.jpg", b"a")
        self._make("b.jpg", b"b")
        store = DatasetStore(self.workspace)
        first = store.import_directory(self.source, "cat")
        second = store.import_directory(self.source, "cat")
        self.assertEqual(first["added"], 2)
        self.assertEqual(second["added"], 0)
        self.assertEqual(second["duplicates"], 2)
        self.assertEqual(
            [e["status"] for e in second["results"]], ["duplicate", "duplicate"]
        )
        self.assertEqual(store.summary()["items"], 2)


class DirectoryLabelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.source = self.base / "images"
        self.source.mkdir()
        (self.source / "a.jpg").write_bytes(b"a")
        (self.source / "b.jpg").write_bytes(b"b")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_no_label_leaves_new_samples_unlabeled(self) -> None:
        store = DatasetStore(self.base / "ws")
        store.import_directory(self.source)
        summary = store.summary()
        self.assertEqual(summary["items"], 2)
        self.assertEqual(summary["labels"], {"unlabeled": 2})

    def test_empty_string_label_means_unlabeled(self) -> None:
        store = DatasetStore(self.base / "ws")
        store.import_directory(self.source, "")
        self.assertEqual(store.summary()["labels"], {"unlabeled": 2})
        for entry in store._read()["items"]:
            self.assertIn(entry["label"], (None, ""))

    def test_labels_are_kept_literally(self) -> None:
        store = DatasetStore(self.base / "ws")
        result = store.import_directory(self.source, "unlabeled")
        # A real class literally named "unlabeled" must not collapse.
        self.assertEqual(store.summary()["labels"], {"unlabeled": 2})
        for entry in result["results"]:
            self.assertEqual(
                store.lookup_label(entry["sha256"])["label"], "unlabeled"
            )

    def test_chinese_label_preserved(self) -> None:
        store = DatasetStore(self.base / "ws")
        store.import_directory(self.source, "猫咪")
        self.assertEqual(store.summary()["labels"], {"猫咪": 2})


class DirectoryFailureAtomicityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.workspace = self.base / "dataset"
        self.source = self.base / "images"
        self.source.mkdir()

    def tearDown(self) -> None:
        # Restore permission so TemporaryDirectory cleanup can succeed.
        for path in self.source.rglob("*"):
            try:
                os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
            except OSError:
                pass
        self.temp.cleanup()

    def test_unreadable_candidate_fails_whole_import_with_no_records(self) -> None:
        (self.source / "good.jpg").write_bytes(b"good")
        locked = self.source / "locked.jpg"
        locked.write_bytes(b"locked")
        os.chmod(locked, 0)
        store = DatasetStore(self.workspace)

        with self.assertRaisesRegex(ValueError, "candidate cannot be read"):
            store.import_directory(self.source)
        self.assertEqual(store.summary()["items"], 0)
        self.assertFalse(
            (self.workspace / ".vision-workbench" / ".txn.json").exists()
        )

    def test_existing_records_and_history_survive_a_failed_import(self) -> None:
        store = DatasetStore(self.workspace)
        keep_source = self.base / "keep.jpg"
        keep_source.write_bytes(b"keep")
        digest = store.add(keep_source, "cat").digest
        store.submit_batch(
            {
                "batch": "b1",
                "changes": [
                    {"sha256": digest, "old": "cat", "new": "dog"}
                ],
            }
        )

        (self.source / "a.jpg").write_bytes(b"a")
        locked = self.source / "b.jpg"
        locked.write_bytes(b"b")
        os.chmod(locked, 0)
        with self.assertRaises(ValueError):
            store.import_directory(self.source, "new")

        self.assertEqual(store.summary()["items"], 1)
        self.assertEqual(store.lookup_label(digest)["label"], "dog")
        self.assertEqual([e["batch"] for e in store.history()], ["b1"])

    def test_file_modified_during_read_fails_the_import(self) -> None:
        target = self.source / "changing.jpg"
        target.write_bytes(b"x" * 4096)
        before_ns = target.stat().st_mtime_ns
        real_factory = hashlib.sha256

        def make_hasher():
            real = real_factory()

            class ChangingHasher:
                triggered = False

                def update(self, chunk: bytes) -> None:
                    if not self.triggered:
                        self.triggered = True
                        # Bump mtime between the pre-read and post-read stat.
                        os.utime(target, ns=(before_ns + 10_000_000_000,) * 2)
                    real.update(chunk)

                def hexdigest(self) -> str:
                    return real.hexdigest()

            return ChangingHasher()

        store = DatasetStore(self.workspace)
        with (
            mock.patch("vision_workbench.store.hashlib.sha256", make_hasher),
            self.assertRaisesRegex(ValueError, "candidate changed during import"),
        ):
            store.import_directory(self.source)
        self.assertEqual(store.summary()["items"], 0)

    def test_file_replaced_during_read_fails_the_import(self) -> None:
        target = self.source / "changing.jpg"
        target.write_bytes(b"x" * 16)
        real_factory = hashlib.sha256

        def make_hasher():
            real = real_factory()

            class ReplacingHasher:
                triggered = False

                def update(self, chunk: bytes) -> None:
                    if not self.triggered:
                        self.triggered = True
                        # Replace contents with a different size mid-read.
                        target.write_bytes(b"y-different-length")
                    real.update(chunk)

                def hexdigest(self) -> str:
                    return real.hexdigest()

            return ReplacingHasher()

        store = DatasetStore(self.workspace)
        with (
            mock.patch("vision_workbench.store.hashlib.sha256", make_hasher),
            self.assertRaisesRegex(ValueError, "candidate changed during import"),
        ):
            store.import_directory(self.source)
        self.assertEqual(store.summary()["items"], 0)

    def test_unscannable_subdirectory_fails_recursive_import(self) -> None:
        (self.source / "top.jpg").write_bytes(b"top")
        hidden = self.source / "hidden"
        hidden.mkdir()
        (hidden / "a.png").write_bytes(b"a")
        os.chmod(hidden, 0)
        store = DatasetStore(self.workspace)
        try:
            with self.assertRaisesRegex(ValueError, "cannot be scanned"):
                store.import_directory(self.source, recursive=True)
        finally:
            os.chmod(hidden, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        self.assertEqual(store.summary()["items"], 0)


class DirectoryRecoveryTest(unittest.TestCase):
    def test_prepared_multi_sample_journal_installs_whole_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            workspace = base / "dataset"
            source = base / "images"
            source.mkdir()
            contents = {f"img-{i}.jpg": f"content-{i}".encode() for i in range(3)}
            for name, content in contents.items():
                (source / name).write_bytes(content)

            store = DatasetStore(workspace)
            # Simulate a crash after the journal was prepared but before the
            # manifest install: manifest missing, .txn.json carrying the full
            # three-sample batch.
            items = [
                {
                    "sha256": sha256_bytes(content),
                    "source": str((source / name).resolve()),
                    "size": len(content),
                    "label": "cat",
                }
                for name, content in contents.items()
            ]
            journal = {
                "manifest": {"schema_version": 1, "items": items},
                "batches": {"schema_version": 1, "batches": []},
            }
            store._write_json_atomic(store.transaction_path, journal)

            reopened = DatasetStore(workspace)
            summary = reopened.summary()
            self.assertEqual(summary["items"], 3)
            self.assertEqual(summary["labels"], {"cat": 3})
            self.assertFalse(reopened.transaction_path.exists())

            # Re-submitting the same directory adds nothing.
            result = reopened.import_directory(source, "cat")
            self.assertEqual(result["added"], 0)
            self.assertEqual(result["duplicates"], 3)


def _import_worker(root: str, source: str, label, recursive: bool, queue) -> None:
    try:
        result = DatasetStore(Path(root)).import_directory(
            Path(source), label, recursive=recursive
        )
        queue.put(("ok", result["added"], result["duplicates"], result["candidates"]))
    except Exception as error:  # noqa: BLE001 - report back to parent
        queue.put(("error", str(error), 0, 0))


def _batch_worker(root: str, number: str, digest: str, old: str, new: str, queue) -> None:
    try:
        result = DatasetStore(Path(root)).submit_batch(
            {
                "batch": number,
                "changes": [{"sha256": digest, "old": old, "new": new}],
            }
        )
        queue.put(("ok", result["status"]))
    except Exception as error:  # noqa: BLE001
        queue.put(("error", str(error)))


class ConcurrentDirectoryImportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_concurrent_imports_overlap_without_losing_or_duplicating(self) -> None:
        workspace = self.base / "dataset"
        sources = []
        shared = b"shared-content"
        for directory_index in range(3):
            source = self.base / f"src-{directory_index}"
            source.mkdir()
            (source / "shared.jpg").write_bytes(shared)
            for file_index in range(3):
                (source / f"unique-{directory_index}-{file_index}.jpg").write_bytes(
                    f"{directory_index}-{file_index}".encode()
                )
            sources.append(source)

        queue = ctx.Queue()
        processes = [
            ctx.Process(
                target=_import_worker,
                args=(str(workspace), str(source), f"label-{i}", False, queue),
            )
            for i, source in enumerate(sources)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
            self.assertEqual(process.exitcode, 0)

        results = [queue.get(timeout=30) for _ in processes]
        self.assertTrue(all(row[0] == "ok" for row in results), results)
        # 9 unique samples plus the single shared content = 10 records.
        self.assertEqual(DatasetStore(workspace).summary()["items"], 10)
        self.assertEqual(sum(row[1] for row in results), 10)
        self.assertEqual(sum(row[2] for row in results), 2)
        for row in results:
            self.assertEqual(row[3], 4)

    def test_directory_import_concurrent_with_batch_keeps_both(self) -> None:
        workspace = self.base / "dataset"
        store = DatasetStore(workspace)
        seeded = self.base / "seed.jpg"
        seeded.write_bytes(b"seed")
        digest = store.add(seeded, "cat").digest

        source = self.base / "images"
        source.mkdir()
        for index in range(5):
            (source / f"new-{index}.jpg").write_bytes(f"new-{index}".encode())

        queue = ctx.Queue()
        workers = [
            ctx.Process(
                target=_batch_worker,
                args=(str(workspace), "b1", digest, "cat", "dog", queue),
            ),
            ctx.Process(
                target=_import_worker,
                args=(str(workspace), str(source), "fresh", False, queue),
            ),
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
            self.assertEqual(worker.exitcode, 0)

        results = [queue.get(timeout=30) for _ in workers]
        self.assertTrue(all(row[0] == "ok" for row in results), results)
        reopened = DatasetStore(workspace)
        self.assertEqual(reopened.summary()["items"], 6)
        self.assertEqual(reopened.lookup_label(digest)["label"], "dog")
        self.assertEqual(reopened.undo_batch("b1")["status"], "undone")
        self.assertEqual(reopened.lookup_label(digest)["label"], "cat")
        # Every newly imported sample kept its own label.
        self.assertEqual(reopened.summary()["labels"]["fresh"], 5)


class ImportDirectoryCliTest(unittest.TestCase):
    def test_cli_emits_result_json_and_reports_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            workspace = base / "ws"
            source = base / "images"
            source.mkdir()
            (source / "a.jpg").write_bytes(b"a")
            (source / "sub").mkdir()
            (source / "sub" / "b.png").write_bytes(b"b")

            env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
            run = [sys.executable, "-m", "vision_workbench"]

            completed = subprocess.run(
                [*run, "import-dir", str(workspace), str(source), "--label", "cat"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["candidates"], 1)
            self.assertEqual(payload["results"][0]["path"], "a.jpg")
            self.assertEqual(payload["results"][0]["status"], "added")

            # Recursive re-import: one duplicate plus one new sample, label
            # omitted so the new sample is unlabeled.
            completed = subprocess.run(
                [*run, "import-dir", "-r", str(workspace), str(source)],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["added"], 1)
            self.assertEqual(payload["duplicates"], 1)
            self.assertEqual(
                [e["path"] for e in payload["results"]], ["a.jpg", "sub/b.png"]
            )

            # A missing directory exits non-zero with a path-bearing message.
            completed = subprocess.run(
                [*run, "import-dir", str(workspace), str(source / "missing")],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 1)
            self.assertIn("missing", completed.stderr)

            # The alias works too.
            completed = subprocess.run(
                [*run, "add-dir", str(workspace), str(source)],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout)["added"], 0)


if __name__ == "__main__":
    unittest.main()
