"""Concurrency and crash-safety regression tests.

These cover multiple local processes operating on one workspace at once:
concurrent imports, imports crossing batch submissions, concurrent
same-name split creation, the empty-workspace initialization race and a
prepared (crashed) transaction observed by processes that already have the
workspace open.
"""

from __future__ import annotations

import json
import multiprocessing
import tempfile
import unittest
from pathlib import Path

from vision_workbench.store import DatasetStore

ctx = multiprocessing.get_context("fork")


def _add_worker(root: str, source: str, label, queue) -> None:
    try:
        store = DatasetStore(Path(root))
        result = store.add(Path(source), label)
        queue.put(("ok", result.added, result.digest, label))
    except Exception as error:  # noqa: BLE001 - report back to parent
        queue.put(("error", str(error), "", label))


def _batch_worker(root: str, number: str, changes: list, queue) -> None:
    try:
        store = DatasetStore(Path(root))
        result = store.submit_batch({"batch": number, "changes": changes})
        queue.put(("ok", result["status"]))
    except Exception as error:  # noqa: BLE001 - report back to parent
        queue.put(("error", str(error)))


def _split_worker(root: str, name: str, seed: int, ratios: list, queue) -> None:
    try:
        store = DatasetStore(Path(root))
        result = store.create_split(name, seed, ratios)
        queue.put(
            (
                "ok",
                result.created,
                result.plan["seed"],
                result.plan["samples"]["total"],
                result.plan["samples"]["distribution"],
            )
        )
    except Exception as error:  # noqa: BLE001 - report back to parent
        queue.put(("error", str(error), seed, 0, {}))


def _init_worker(root: str, barrier, queue) -> None:
    barrier.wait()
    try:
        DatasetStore(Path(root)).initialize()
        queue.put(("init", "ok"))
    except Exception as error:  # noqa: BLE001
        queue.put(("init", f"error: {error}"))


def _summary_worker(root: str, barrier, queue) -> None:
    barrier.wait()
    try:
        summary = DatasetStore(Path(root)).summary()
        queue.put(("summary", summary["items"]))
    except Exception as error:  # noqa: BLE001
        queue.put(("summary", f"error: {error}"))


def _add_barrier_worker(root: str, source: str, label, barrier, queue) -> None:
    barrier.wait()
    _add_worker(root, source, label, queue)


def _drain(queue, count: int) -> list:
    return [queue.get(timeout=30) for _ in range(count)]


class ConcurrentImportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_concurrent_imports_of_different_content_all_added(self) -> None:
        root = self.base / "dataset"
        sources = []
        for index in range(6):
            path = self.base / f"img-{index}.jpg"
            path.write_bytes(f"unique-content-{index}".encode())
            sources.append(path)

        queue = ctx.Queue()
        processes = [
            ctx.Process(
                target=_add_worker,
                args=(str(root), str(path), f"label-{index}", queue),
            )
            for index, path in enumerate(sources)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
            self.assertEqual(process.exitcode, 0)

        results = _drain(queue, len(sources))
        self.assertTrue(all(status == "ok" for status, *_ in results), results)
        self.assertEqual(sum(1 for _, added, *_ in results if added), len(sources))

        store = DatasetStore(root)
        summary = store.summary()
        self.assertEqual(summary["items"], len(sources))
        self.assertEqual(summary["bytes"], sum(len(f"unique-content-{i}") for i in range(6)))

    def test_concurrent_imports_of_same_content_single_add_keeps_first(self) -> None:
        root = self.base / "dataset"
        content = b"identical-bytes"
        # Same content reachable through distinct paths, with distinct labels.
        paths = []
        for index in range(5):
            path = self.base / f"copy-{index}.jpg"
            path.write_bytes(content)
            paths.append(path)
        labels = [f"label-{index}" for index in range(5)]

        queue = ctx.Queue()
        processes = [
            ctx.Process(
                target=_add_worker,
                args=(str(root), str(path), label, queue),
            )
            for path, label in zip(paths, labels)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
            self.assertEqual(process.exitcode, 0)

        results = _drain(queue, len(paths))
        self.assertTrue(all(status == "ok" for status, *_ in results), results)
        added = [row for row in results if row[1]]
        self.assertEqual(len(added), 1, results)
        winner_digest = added[0][2]
        self.assertTrue(all(digest == winner_digest for _, _, digest, _ in results))

        manifest = json.loads(
            (root / ".vision-workbench" / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(manifest["items"]), 1)
        item = manifest["items"][0]
        # The winner's registration is retained; later duplicate reports do
        # not overwrite the source path or the label.
        self.assertIn(item["label"], labels)
        self.assertEqual(Path(item["source"]).read_bytes(), content)

        # A still-later duplicate import (yet another label) reports present
        # and changes nothing.
        again = DatasetStore(root).add(paths[0], "totally-different")
        self.assertFalse(again.added)
        item_after = json.loads(
            (root / ".vision-workbench" / "manifest.json").read_text(encoding="utf-8")
        )["items"][0]
        self.assertEqual(item_after, item)

    def test_duplicate_import_keeps_label_and_undo_eligibility(self) -> None:
        root = self.base / "dataset"
        source = self.base / "one.jpg"
        source.write_bytes(b"only-content")
        store = DatasetStore(root)
        digest = store.add(source, "cat").digest

        store.submit_batch(
            {"batch": "b1", "changes": [{"sha256": digest, "old": "cat", "new": "dog"}]}
        )
        # Re-importing the same content (even with another label) must not
        # roll the label back or disturb the revision that gates undo.
        duplicate = store.add(source, "cat-again")
        self.assertFalse(duplicate.added)
        self.assertEqual(store.lookup_label(digest)["label"], "dog")

        undone = store.undo_batch("b1")
        self.assertEqual(undone["status"], "undone")
        self.assertEqual(store.lookup_label(digest)["label"], "cat")

    def test_imports_crossing_a_batch_keep_label_history_and_samples(self) -> None:
        root = self.base / "dataset"
        store = DatasetStore(root)
        existing = root.parent / "existing.jpg"
        existing.write_bytes(b"existing-content")
        digest = store.add(existing, "cat").digest

        # Many fresh samples imported concurrently with a batch that relabels
        # the already-imported sample.
        new_sources = []
        for index in range(6):
            path = root.parent / f"cross-{index}.jpg"
            path.write_bytes(f"cross-content-{index}".encode())
            new_sources.append(path)

        queue = ctx.Queue()
        changes = [{"sha256": digest, "old": "cat", "new": "dog"}]
        workers = [
            ctx.Process(target=_batch_worker, args=(str(root), "b1", changes, queue))
        ] + [
            ctx.Process(
                target=_add_worker, args=(str(root), str(path), "new", queue)
            )
            for path in new_sources
        ]
        for process in workers:
            process.start()
        for process in workers:
            process.join()
            self.assertEqual(process.exitcode, 0)

        results = _drain(queue, len(workers))
        batch_results = [r for r in results if len(r) == 2]
        add_results = [r for r in results if len(r) == 4]
        self.assertEqual(len(batch_results), 1)
        self.assertEqual(batch_results[0], ("ok", "applied"), batch_results)
        self.assertEqual(sum(1 for r in add_results if r[1]), len(new_sources))

        reopened = DatasetStore(root)
        self.assertEqual(reopened.lookup_label(digest)["label"], "dog")
        self.assertEqual(reopened.summary()["items"], 1 + len(new_sources))
        self.assertEqual([e["batch"] for e in reopened.history()], ["b1"])
        # The batch's revision is intact, so its undo is still eligible even
        # though imports landed around it.
        self.assertEqual(reopened.undo_batch("b1")["status"], "undone")
        self.assertEqual(reopened.lookup_label(digest)["label"], "cat")


class EmptyWorkspaceRaceTest(unittest.TestCase):
    def test_init_summary_and_first_import_cannot_reset_to_empty(self) -> None:
        for round_index in range(5):
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                root = base / "dataset"
                source = base / "first.jpg"
                source.write_bytes(f"first-{round_index}".encode())

                barrier = ctx.Barrier(4)
                queue = ctx.Queue()
                workers = [
                    ctx.Process(target=_init_worker, args=(str(root), barrier, queue)),
                    ctx.Process(target=_init_worker, args=(str(root), barrier, queue)),
                    ctx.Process(
                        target=_summary_worker, args=(str(root), barrier, queue)
                    ),
                    ctx.Process(
                        target=_add_barrier_worker,
                        args=(str(root), str(source), "cat", barrier, queue),
                    ),
                ]
                for process in workers:
                    process.start()
                for process in workers:
                    process.join()
                    self.assertEqual(process.exitcode, 0)

                results = _drain(queue, 4)
                messages = {row[0]: row[1] for row in results}
                self.assertEqual(messages["init"], "ok", results)
                self.assertNotIn("error", str(messages["summary"]), results)

                # The committed first sample must survive: never an empty
                # dataset once the import has reported success.
                final = DatasetStore(root).summary()
                self.assertEqual(final["items"], 1, results)
                self.assertEqual(final["labels"], {"cat": 1}, results)


class ConcurrentSplitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "dataset"
        self.store = DatasetStore(self.root)
        self._serial = 0

    def tearDown(self) -> None:
        self.temp.cleanup()

    def add_sample(self, label: str) -> str:
        self._serial += 1
        path = self.root.parent / f"sample-{self._serial}.jpg"
        path.write_bytes(f"content-{self._serial}-{label}".encode())
        result = self.store.add(path, label)
        self.assertTrue(result.added)
        return result.digest

    def test_concurrent_same_name_identical_requests_one_creates_rest_reuse(self) -> None:
        for label in ("cat", "cat", "dog"):
            self.add_sample(label)

        queue = ctx.Queue()
        workers = [
            ctx.Process(
                target=_split_worker,
                args=(str(self.root), "plan", 7, ["0.5", "0.25", "0.25"], queue),
            )
            for _ in range(4)
        ]
        for process in workers:
            process.start()
        for process in workers:
            process.join()
            self.assertEqual(process.exitcode, 0)

        results = _drain(queue, 4)
        self.assertTrue(all(row[0] == "ok" for row in results), results)
        self.assertEqual(sum(1 for row in results if row[1]), 1, results)
        totals = {row[3] for row in results}
        self.assertEqual(totals, {3})
        distributions = {json.dumps(row[4], sort_keys=True) for row in results}
        self.assertEqual(len(distributions), 1)

    def test_concurrent_same_name_different_requests_conflict_keeps_original(self) -> None:
        for label in ("cat", "dog"):
            self.add_sample(label)

        queue = ctx.Queue()
        seeds = [1, 2, 3, 4]
        workers = [
            ctx.Process(
                target=_split_worker,
                args=(str(self.root), "plan", seed, ["0.5", "0.5", "0"], queue),
            )
            for seed in seeds
        ]
        for process in workers:
            process.start()
        for process in workers:
            process.join()
            self.assertEqual(process.exitcode, 0)

        results = _drain(queue, 4)
        created = [row for row in results if row[0] == "ok" and row[1]]
        conflicts = [row for row in results if row[0] == "error"]
        self.assertEqual(len(created), 1, results)
        self.assertEqual(len(conflicts), 3, results)
        self.assertTrue(
            all("already exists" in row[1] for row in conflicts), results
        )
        # The original is untouched and always reads back complete.
        stored = self.store.get_split("plan")
        self.assertEqual(stored["seed"], created[0][2])
        self.assertIn(stored["seed"], seeds)
        self.assertEqual(stored["samples"]["total"], 2)

    def test_concurrent_different_names_all_saved(self) -> None:
        for label in ("cat", "dog", "bird"):
            self.add_sample(label)

        queue = ctx.Queue()
        names = ["alpha", "beta", "gamma", "delta"]
        workers = [
            ctx.Process(
                target=_split_worker,
                args=(str(self.root), name, 3, ["1/3", "1/3", "1/3"], queue),
            )
            for name in names
        ]
        for process in workers:
            process.start()
        for process in workers:
            process.join()
            self.assertEqual(process.exitcode, 0)

        results = _drain(queue, 4)
        self.assertTrue(all(row[0] == "ok" and row[1] for row in results), results)
        for name in names:
            plan = self.store.get_split(name)
            self.assertEqual(plan["name"], name)
            self.assertEqual(plan["samples"]["total"], 3)

    def test_plan_snapshot_is_one_complete_label_state(self) -> None:
        # A batch that relabels both samples commits as one unit; a plan built
        # around it must see both labels from the same state, never a mix.
        d1 = self.add_sample("cat")
        d2 = self.add_sample("cat")
        self.store.submit_batch(
            {
                "batch": "b1",
                "changes": [
                    {"sha256": d1, "old": "cat", "new": "dog"},
                    {"sha256": d2, "old": "cat", "new": "dog"},
                ],
            }
        )
        after = self.store.create_split("after", 0, [1, 0, 0]).plan
        self.assertEqual(after["samples"]["distribution"], {"dog": 2})


class PreparedTransactionRecoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "dataset"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _prepare_crashed_journal(self, digest: str) -> None:
        """Leave a prepared transaction with the manifest still at old state."""
        store = DatasetStore(self.root)
        source = self.root.parent / "seed.jpg"
        source.write_bytes(b"seed-content")
        store.add(source, "cat")

        new_manifest = {
            "schema_version": 1,
            "items": [
                {
                    "sha256": digest,
                    "source": str(source.resolve()),
                    "size": len(b"seed-content"),
                    "label": "dog",
                    "rev": 1,
                }
            ],
        }
        history = {
            "schema_version": 1,
            "batches": [
                {
                    "batch": "b1",
                    "records": [
                        {
                            "sha256": digest,
                            "old": "cat",
                            "new": "dog",
                            "changed": True,
                            "rev": 1,
                        }
                    ],
                    "changed_count": 1,
                    "undone": False,
                    "undone_at": None,
                }
            ],
        }
        # A store that "already has the workspace open": prepared below by a
        # peer, without reconstructing it.
        self.open_store = DatasetStore(self.root)
        self.open_store._write_json_atomic(
            self.open_store.transaction_path,
            {"manifest": new_manifest, "batches": history},
        )
        self.assertTrue(self.open_store.transaction_path.exists())
        # Manifest on disk is still the pre-crash "cat" state.
        self.assertEqual(
            json.loads(
                self.open_store.manifest_path.read_text(encoding="utf-8")
            )["items"][0]["label"],
            "cat",
        )

    def test_already_open_import_completes_journal_and_survives_reopen(self) -> None:
        digest = hashlib_sha256(b"seed-content")
        self._prepare_crashed_journal(digest)

        # A peer's later import lands through the already-open store; it must
        # first install the prepared transaction, then add without losing the
        # recovered label/history or its own sample.
        later = self.root.parent / "later.jpg"
        later.write_bytes(b"later-content")
        result = self.open_store.add(later, "bird")
        self.assertTrue(result.added)
        self.assertFalse(self.open_store.transaction_path.exists())

        self.assertEqual(self.open_store.lookup_label(digest)["label"], "dog")
        self.assertEqual(self.open_store.summary()["items"], 2)
        self.assertEqual(
            [entry["batch"] for entry in self.open_store.history()], ["b1"]
        )

        # Reopening must not run recovery over the (now absent) journal and
        # erase the sample imported after the crash.
        reopened = DatasetStore(self.root)
        self.assertEqual(reopened.summary()["items"], 2)
        self.assertEqual(reopened.lookup_label(digest)["label"], "dog")
        self.assertEqual([entry["batch"] for entry in reopened.history()], ["b1"])
        labels = reopened.summary()["labels"]
        self.assertEqual(labels["bird"], 1)
        self.assertEqual(labels["dog"], 1)

    def test_already_open_summary_completes_journal(self) -> None:
        digest = hashlib_sha256(b"seed-content")
        self._prepare_crashed_journal(digest)

        summary = self.open_store.summary()
        # Summary observes the complete post-operation state.
        self.assertEqual(summary["items"], 1)
        self.assertEqual(summary["labels"], {"dog": 1})
        self.assertFalse(self.open_store.transaction_path.exists())

        history = self.open_store.history()
        self.assertEqual([entry["batch"] for entry in history], ["b1"])
        self.assertEqual(history[0]["records"][0]["new"], "dog")

        # And the completed state is stable across a fresh open.
        again = DatasetStore(self.root)
        self.assertEqual(again.summary()["labels"], {"dog": 1})
        self.assertEqual(again.lookup_label(digest)["label"], "dog")


def hashlib_sha256(content: bytes) -> str:
    import hashlib

    return hashlib.sha256(content).hexdigest()


if __name__ == "__main__":
    unittest.main()
