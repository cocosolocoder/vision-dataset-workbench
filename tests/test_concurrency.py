from __future__ import annotations

import json
import multiprocessing
import tempfile
import unittest
from pathlib import Path

from vision_workbench.store import DatasetStore


class StoreHarness(unittest.TestCase):
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
        path.write_bytes(content if content is not None else f"content-{self._serial}".encode())
        result = self.store.add(path, label)
        self.assertTrue(result.added)
        return result.digest

    def submit(self, number: str, *changes: tuple) -> dict:
        return self.store.submit_batch(
            {
                "batch": number,
                "changes": [
                    {"sha256": digest, "old": old, "new": new}
                    for digest, old, new in changes
                ],
            }
        )


def _add_worker(root: str, path: str, label: str | None, queue) -> None:
    try:
        result = DatasetStore(Path(root)).add(Path(path), label)
        queue.put(("ok", {"digest": result.digest, "added": result.added}))
    except Exception as error:  # noqa: BLE001 - report back to parent
        queue.put(("error", str(error)))


def _batch_worker(root: str, number: str, changes: list, queue) -> None:
    try:
        result = DatasetStore(Path(root)).submit_batch(
            {"batch": number, "changes": changes}
        )
        queue.put(("ok", result))
    except Exception as error:  # noqa: BLE001
        queue.put(("error", str(error)))


def _undo_worker(root: str, number: str, queue) -> None:
    try:
        result = DatasetStore(Path(root)).undo_batch(number)
        queue.put(("ok", result))
    except Exception as error:  # noqa: BLE001
        queue.put(("error", str(error)))


def _split_worker(root: str, name: str, seed: int, ratios: list, queue) -> None:
    try:
        result = DatasetStore(Path(root)).create_split(name, seed, ratios)
        queue.put(("ok", {"created": result.created, "plan": result.plan}))
    except Exception as error:  # noqa: BLE001
        queue.put(("error", str(error)))


def _init_worker(root: str, queue) -> None:
    try:
        DatasetStore(Path(root)).initialize()
        queue.put(("ok", True))
    except Exception as error:  # noqa: BLE001
        queue.put(("error", str(error)))


def _summary_worker(root: str, queue) -> None:
    try:
        queue.put(("ok", DatasetStore(Path(root)).summary()))
    except Exception as error:  # noqa: BLE001
        queue.put(("error", str(error)))


def _drain(queue, count: int) -> list:
    return [queue.get() for _ in range(count)]


class ImportConcurrencyTest(StoreHarness):
    def test_parallel_imports_different_content_all_survive(self) -> None:
        ctx = multiprocessing.get_context("fork")
        queue = ctx.Queue()
        paths = []
        for i in range(12):
            self._serial += 1
            path = self.root.parent / f"p-{i}.jpg"
            path.write_bytes(f"content-{i}".encode())
            paths.append(path)
        processes = [
            ctx.Process(target=_add_worker, args=(str(self.root), str(path), f"lab-{i}", queue))
            for i, path in enumerate(paths)
        ]
        for process in processes:
            process.start()
        results = _drain(queue, len(processes))
        for process in processes:
            process.join()
        self.assertTrue(all(status == "ok" for status, _ in results), results)
        self.assertTrue(all(payload["added"] for _, payload in results), results)
        summary = self.store.summary()
        self.assertEqual(summary["items"], 12)
        self.assertEqual(summary["bytes"], sum(path.stat().st_size for path in paths))

    def test_parallel_imports_same_content_one_added_rest_present(self) -> None:
        ctx = multiprocessing.get_context("fork")
        queue = ctx.Queue()
        paths = []
        for i in range(6):
            self._serial += 1
            path = self.root.parent / f"same-{i}.jpg"
            path.write_bytes(b"identical-bytes")
            paths.append(path)
        processes = [
            ctx.Process(target=_add_worker, args=(str(self.root), str(path), f"label-{i}", queue))
            for i, path in enumerate(paths)
        ]
        for process in processes:
            process.start()
        results = _drain(queue, len(processes))
        for process in processes:
            process.join()
        self.assertTrue(all(status == "ok" for status, _ in results), results)
        payloads = [payload for status, payload in results if status == "ok"]
        self.assertEqual(sum(1 for payload in payloads if payload["added"]), 1)
        self.assertEqual(sum(1 for payload in payloads if not payload["added"]), 5)
        self.assertEqual(len({payload["digest"] for payload in payloads}), 1)
        summary = self.store.summary()
        self.assertEqual(summary["items"], 1)
        manifest = json.loads(self.store.manifest_path.read_text(encoding="utf-8"))
        item = manifest["items"][0]
        self.assertIn(item["label"], {f"label-{i}" for i in range(6)})
        self.assertEqual(Path(item["source"]).read_bytes(), b"identical-bytes")

    def test_reimport_keeps_modified_label_and_undo_eligibility(self) -> None:
        path = self.root.parent / "again.jpg"
        path.write_bytes(b"same")
        first = self.store.add(path, "cat")
        self.assertTrue(first.added)
        digest = first.digest
        applied = self.submit("b1", (digest, "cat", "dog"))
        self.assertEqual(applied["status"], "applied")

        # Re-import with a different label must not overwrite the new label.
        second = self.store.add(path, "kitten")
        self.assertFalse(second.added)
        self.assertEqual(second.digest, digest)
        self.assertEqual(self.store.lookup_label(digest)["label"], "dog")

        # The batch is still undoable after the re-import.
        undone = self.store.undo_batch("b1")
        self.assertEqual(undone["status"], "undone")
        self.assertEqual(self.store.lookup_label(digest)["label"], "cat")

        # Re-import after undo keeps the restored label.
        third = self.store.add(path, "fish")
        self.assertFalse(third.added)
        self.assertEqual(self.store.lookup_label(digest)["label"], "cat")


class ImportBatchInterleaveTest(StoreHarness):
    def test_imports_and_batches_do_not_overwrite_each_other(self) -> None:
        digests = []
        for i in range(3):
            self._serial += 1
            path = self.root.parent / f"pre-{i}.jpg"
            path.write_bytes(f"pre-{i}".encode())
            digests.append(self.store.add(path, "cat").digest)

        ctx = multiprocessing.get_context("fork")
        queue = ctx.Queue()
        add_paths = []
        for i in range(8):
            self._serial += 1
            path = self.root.parent / f"new-{i}.jpg"
            path.write_bytes(f"new-{i}".encode())
            add_paths.append(path)
        add_processes = [
            ctx.Process(target=_add_worker, args=(str(self.root), str(path), "new", queue))
            for path in add_paths
        ]
        batch_processes = [
            ctx.Process(
                target=_batch_worker,
                args=(
                    str(self.root),
                    f"b{i}",
                    [{"sha256": digest, "old": "cat", "new": f"batch-{i}"} for digest in digests],
                    queue,
                ),
            )
            for i in range(4)
        ]
        processes = add_processes + batch_processes
        for process in processes:
            process.start()
        results = _drain(queue, len(processes))
        for process in processes:
            process.join()

        add_results = [
            (status, payload) for status, payload in results if "added" in payload
        ]
        batch_results = [
            (status, payload) for status, payload in results if "added" not in payload
        ]
        self.assertEqual(len(add_results), 8)
        self.assertTrue(all(status == "ok" for status, _ in add_results), add_results)
        self.assertTrue(all(payload["added"] for _, payload in add_results))
        self.assertEqual(sum(1 for status, _ in batch_results if status == "ok"), 1)
        summary = self.store.summary()
        self.assertEqual(summary["items"], 11)
        self.assertEqual(summary["labels"].get("new"), 8)
        manifest = json.loads(self.store.manifest_path.read_text(encoding="utf-8"))
        pre_items = [item for item in manifest["items"] if item["sha256"] in digests]
        self.assertEqual(len(pre_items), 3)
        labels = {item["label"] for item in pre_items}
        self.assertEqual(len(labels), 1)
        self.assertTrue(next(iter(labels)).startswith("batch-"))
        self.assertEqual(len(self.store.history()), 1)

    def test_imports_and_undo_leave_no_lost_samples(self) -> None:
        self._serial += 1
        path = self.root.parent / "u.jpg"
        path.write_bytes(b"undo-me")
        digest = self.store.add(path, "cat").digest
        applied = self.submit("b1", (digest, "cat", "dog"))
        self.assertEqual(applied["status"], "applied")

        ctx = multiprocessing.get_context("fork")
        queue = ctx.Queue()
        add_paths = []
        for i in range(6):
            self._serial += 1
            new_path = self.root.parent / f"un-{i}.jpg"
            new_path.write_bytes(f"un-{i}".encode())
            add_paths.append(new_path)
        add_processes = [
            ctx.Process(target=_add_worker, args=(str(self.root), str(p), "x", queue))
            for p in add_paths
        ]
        undo_process = ctx.Process(target=_undo_worker, args=(str(self.root), "b1", queue))
        processes = add_processes + [undo_process]
        for process in processes:
            process.start()
        results = _drain(queue, len(processes))
        for process in processes:
            process.join()
        self.assertEqual(sum(1 for status, _ in results if status == "ok"), 7, results)
        self.assertEqual(self.store.summary()["items"], 7)
        label = self.store.lookup_label(digest)["label"]
        self.assertIn(label, ("cat", "dog"))
        self.assertEqual(len(self.store.history()), 1)
        # Reopening must not wipe the imported samples.
        self.assertEqual(DatasetStore(self.root).summary()["items"], 7)


class EmptyWorkspaceRaceTest(StoreHarness):
    def test_init_summary_and_first_import_never_revert_to_empty(self) -> None:
        # Start from a truly empty workspace (no manifest written yet).
        fresh = Path(self.temp.name) / "fresh"
        ctx = multiprocessing.get_context("fork")
        queue = ctx.Queue()
        init_processes = [
            ctx.Process(target=_init_worker, args=(str(fresh), queue)) for _ in range(4)
        ]
        summary_processes = [
            ctx.Process(target=_summary_worker, args=(str(fresh), queue)) for _ in range(4)
        ]
        self._serial += 1
        path = self.root.parent / "first.jpg"
        path.write_bytes(b"first")
        add_process = ctx.Process(
            target=_add_worker, args=(str(fresh), str(path), "cat", queue)
        )
        processes = init_processes + summary_processes + [add_process]
        for process in processes:
            process.start()
        results = _drain(queue, len(processes))
        for process in processes:
            process.join()
        self.assertTrue(all(status == "ok" for status, _ in results), results)
        for _, payload in results[4:8]:
            self.assertIn(payload["items"], (0, 1))
        summary = DatasetStore(fresh).summary()
        self.assertEqual(summary["items"], 1)
        self.assertEqual(summary["labels"], {"cat": 1})


class SplitPlanRaceTest(StoreHarness):
    def test_same_name_create_race_one_created_rest_reuse_or_conflict(self) -> None:
        for i in range(4):
            self.add_sample("cat")
        ctx = multiprocessing.get_context("fork")
        queue = ctx.Queue()
        processes = [
            ctx.Process(
                target=_split_worker,
                args=(str(self.root), "race", 7, ["0.5", "0.5", "0"], queue),
            )
            for _ in range(6)
        ]
        for process in processes:
            process.start()
        results = _drain(queue, len(processes))
        for process in processes:
            process.join()
        payloads = [payload for status, payload in results if status == "ok"]
        self.assertEqual(len(payloads), 6, results)
        self.assertEqual(sum(1 for payload in payloads if payload["created"]), 1)
        plans = {json.dumps(payload["plan"], sort_keys=True) for payload in payloads}
        self.assertEqual(len(plans), 1)

        # A different seed conflicts and leaves the original untouched.
        queue2 = ctx.Queue()
        conflict = ctx.Process(
            target=_split_worker, args=(str(self.root), "race", 9, ["0.5", "0.5", "0"], queue2)
        )
        conflict.start()
        conflict.join()
        status, payload = queue2.get()
        self.assertEqual(status, "error")
        self.assertIn("already exists", payload)
        self.assertEqual(self.store.get_split("race")["seed"], 7)

        # A different name is saved normally.
        other = self.store.create_split("other", 7, ["0.5", "0.5", "0"])
        self.assertTrue(other.created)

    def test_split_snapshot_uses_all_before_or_all_after_batch_labels(self) -> None:
        digests = []
        for _ in range(2):
            digests.append(self.add_sample("cat"))
        ctx = multiprocessing.get_context("fork")
        queue = ctx.Queue()
        for round_number in range(12):
            batch = ctx.Process(
                target=_batch_worker,
                args=(
                    str(self.root),
                    f"b{round_number}",
                    [{"sha256": digest, "old": "cat", "new": "dog"} for digest in digests],
                    queue,
                ),
            )
            split = ctx.Process(
                target=_split_worker,
                args=(str(self.root), f"p{round_number}", 0, ["1", "0", "0"], queue),
            )
            processes = [batch, split]
            for process in processes:
                process.start()
            results = _drain(queue, len(processes))
            for process in processes:
                process.join()
            for status, payload in results:
                if status == "ok" and "plan" in payload:
                    distribution = payload["plan"]["samples"]["distribution"]
                    self.assertIn(distribution, ({"cat": 2}, {"dog": 2}), distribution)


class CrashRecoveryTest(StoreHarness):
    def test_crashed_transaction_then_import_keeps_both(self) -> None:
        digest = self.add_sample("cat")
        # Simulate a batch commit that crashed after the journal was written
        # but before the manifest/history installs: the manifest is still old.
        new_manifest = json.loads(self.store.manifest_path.read_text(encoding="utf-8"))
        new_manifest["items"][0]["label"] = "dog"
        new_manifest["items"][0]["rev"] = 1
        history = {
            "schema_version": 1,
            "batches": [
                {
                    "batch": "b1",
                    "records": [
                        {"sha256": digest, "old": "cat", "new": "dog", "changed": True, "rev": 1}
                    ],
                    "changed_count": 1,
                    "undone": False,
                    "undone_at": None,
                }
            ],
        }
        self.store._write_json_atomic(
            self.store.transaction_path, {"manifest": new_manifest, "batches": history}
        )
        self.assertTrue(self.store.transaction_path.exists())

        # An already-open process continues to import without reopening.
        self._serial += 1
        new_path = self.root.parent / "after-crash.jpg"
        new_path.write_bytes(b"after-crash")
        result = self.store.add(new_path, "kitten")
        self.assertTrue(result.added)

        manifest = json.loads(self.store.manifest_path.read_text(encoding="utf-8"))
        items = {item["sha256"]: item for item in manifest["items"]}
        self.assertEqual(items[digest]["label"], "dog")
        self.assertEqual(items[result.digest]["label"], "kitten")
        self.assertEqual(len(items), 2)
        self.assertEqual([entry["batch"] for entry in self.store.history()], ["b1"])
        self.assertFalse(self.store.transaction_path.exists())

        # Reopening must not wipe the imported sample.
        reopened = DatasetStore(self.root)
        self.assertEqual(reopened.summary()["items"], 2)


if __name__ == "__main__":
    unittest.main()
