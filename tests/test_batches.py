from __future__ import annotations

import json
import multiprocessing
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from vision_workbench.batches import BatchError
from vision_workbench.store import DatasetStore

REPO_ROOT = Path(__file__).resolve().parents[1]


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
        """Submit a batch; each change is (digest, old, new)."""
        return self.store.submit_batch(
            {
                "batch": number,
                "changes": [
                    {"sha256": digest, "old": old, "new": new}
                    for digest, old, new in changes
                ],
            }
        )

    def manifest_labels(self) -> dict[str, str | None]:
        manifest = json.loads(self.store.manifest_path.read_text(encoding="utf-8"))
        return {item["sha256"]: item.get("label") for item in manifest["items"]}


class BatchSubmitTest(StoreHarness):
    def test_submit_changes_labels_and_summary(self) -> None:
        d1 = self.add_sample("cat")
        d2 = self.add_sample("dog")
        before = json.loads(self.store.manifest_path.read_text(encoding="utf-8"))

        result = self.submit("b1", (d1, "cat", "kitten"), (d2, "dog", None))
        self.assertEqual(result["status"], "applied")
        self.assertEqual(result["changed"], 2)
        self.assertEqual(result["unchanged"], 0)

        self.assertEqual(self.store.lookup_label(d1)["label"], "kitten")
        self.assertIsNone(self.store.lookup_label(d2)["label"])
        summary = self.store.summary()
        self.assertEqual(summary["items"], 2)
        self.assertEqual(summary["labels"], {"kitten": 1, "unlabeled": 1})

        # Content digest, source and size are untouched.
        after = json.loads(self.store.manifest_path.read_text(encoding="utf-8"))
        before_items = {i["sha256"]: i for i in before["items"]}
        after_items = {i["sha256"]: i for i in after["items"]}
        for digest, item in after_items.items():
            self.assertEqual(item["source"], before_items[digest]["source"])
            self.assertEqual(item["size"], before_items[digest]["size"])
            self.assertEqual(item["sha256"], before_items[digest]["sha256"])

    def test_null_and_empty_both_unlabeled_literal_unlabeled_kept(self) -> None:
        d1 = self.add_sample("cat")
        d2 = self.add_sample("cat")
        d3 = self.add_sample("cat")
        d4 = self.add_sample("cat")

        # Clear with null, then with the empty string.
        self.submit("b1", (d1, "cat", None))
        self.submit("b2", (d2, "cat", ""))
        self.assertIsNone(self.store.lookup_label(d1)["label"])
        self.assertIsNone(self.store.lookup_label(d2)["label"])

        # A real class literally named "unlabeled" stays distinct.
        self.submit("b3", (d3, "cat", "unlabeled"))
        self.assertEqual(self.store.lookup_label(d3)["label"], "unlabeled")

        # Chinese labels are taken literally.
        self.submit("b4", (d4, "cat", "猫"))
        self.assertEqual(self.store.lookup_label(d4)["label"], "猫")

        # Summary counts both unlabeled spellings and the literal class
        # under "unlabeled"; the split layer keeps the literal class apart.
        summary = self.store.summary()
        self.assertEqual(summary["labels"], {"unlabeled": 3, "猫": 1})
        # Lookup still distinguishes the literal class from null.
        self.assertEqual(self.store.lookup_label(d3)["label"], "unlabeled")

    def test_set_label_on_unlabeled_sample(self) -> None:
        d = self.add_sample(None)
        result = self.submit("b1", (d, None, "cat"))
        self.assertEqual(result["status"], "applied")
        self.assertEqual(self.store.lookup_label(d)["label"], "cat")

    def test_noop_batch_reports_no_changes_and_occupies_no_number(self) -> None:
        d = self.add_sample("cat")
        result = self.submit("b1", (d, "cat", "cat"))
        self.assertEqual(result["status"], "no-changes")
        self.assertEqual(result["changed"], 0)
        self.assertEqual(self.store.history(), [])

        # The number is still free for a real batch.
        real = self.submit("b1", (d, "cat", "dog"))
        self.assertEqual(real["status"], "applied")
        self.assertEqual(self.store.history()[0]["batch"], "b1")

    def test_empty_changes_list_is_noop(self) -> None:
        result = self.store.submit_batch({"batch": "empty", "changes": []})
        self.assertEqual(result["status"], "no-changes")
        self.assertEqual(self.store.history(), [])

    def test_replay_returns_original_result_without_remodifying(self) -> None:
        d = self.add_sample("cat")
        first = self.submit("b1", (d, "cat", "dog"))
        self.assertEqual(first["status"], "applied")

        # Someone changes the label back; the expected old label no longer
        # matches, but replay must still return the original result.
        self.submit("b2", (d, "dog", "cat"))
        replay = self.submit("b1", (d, "cat", "dog"))
        self.assertEqual(replay["status"], "already-applied")
        self.assertEqual(replay["changed"], first["changed"])
        self.assertEqual(replay["batch"], first["batch"])
        # The replay did not modify anything (label stays cat).
        self.assertEqual(self.store.lookup_label(d)["label"], "cat")

    def test_replay_order_and_unlabeled_spelling_are_irrelevant(self) -> None:
        d1 = self.add_sample(None)
        d2 = self.add_sample("")
        self.submit("b1", (d1, None, "cat"), (d2, None, "dog"))

        # Reversed order, "" instead of null for old labels.
        replay = self.store.submit_batch(
            {
                "batch": "b1",
                "changes": [
                    {"sha256": d2, "old": "", "new": "dog"},
                    {"sha256": d1, "old": "", "new": "cat"},
                ],
            }
        )
        self.assertEqual(replay["status"], "already-applied")

    def test_same_number_different_content_conflicts(self) -> None:
        d = self.add_sample("cat")
        self.submit("b1", (d, "cat", "dog"))
        with self.assertRaises(BatchError):
            self.submit("b1", (d, "cat", "fish"))
        # History keeps the original batch only.
        self.assertEqual([e["batch"] for e in self.store.history()], ["b1"])
        self.assertEqual(self.store.lookup_label(d)["label"], "dog")

    def test_history_lists_successes_in_submission_order(self) -> None:
        d1 = self.add_sample("cat")
        d2 = self.add_sample("dog")
        self.submit("a1", (d1, "cat", "kitten"))
        self.submit("a2", (d2, "dog", None))
        history = self.store.history()
        self.assertEqual([e["batch"] for e in history], ["a1", "a2"])
        self.assertEqual(history[0]["changed_count"], 1)
        self.assertEqual(history[0]["records"][0]["old"], "cat")
        self.assertEqual(history[0]["records"][0]["new"], "kitten")
        self.assertEqual(history[1]["records"][0]["old"], "dog")
        self.assertIsNone(history[1]["records"][0]["new"])
        self.assertFalse(history[0]["undone"])

    def test_failed_batch_enters_no_history(self) -> None:
        d = self.add_sample("cat")
        with self.assertRaises(BatchError):
            self.submit("bad", (d, "cat", "dog"), ("deadbeef", "cat", "dog"))
        self.assertEqual(self.store.history(), [])
        self.assertEqual(self.store.lookup_label(d)["label"], "cat")


class BatchValidationTest(StoreHarness):
    def assert_rejected(self, data: object) -> None:
        with self.assertRaises(BatchError):
            self.store.submit_batch(data)
        self.assertEqual(self.store.history(), [])

    def test_file_must_be_object(self) -> None:
        self.assert_rejected(["not", "an", "object"])

    def test_batch_number_must_be_nonempty_string(self) -> None:
        d = self.add_sample("cat")
        for bad in [None, "", "   ", 123]:
            self.assert_rejected({"batch": bad, "changes": [
                {"sha256": d, "old": "cat", "new": "dog"}]})

    def test_changes_must_be_list(self) -> None:
        self.assert_rejected({"batch": "b", "changes": {}})

    def test_change_must_have_required_keys(self) -> None:
        d = self.add_sample("cat")
        self.assert_rejected({"batch": "b", "changes": [
            {"sha256": d, "old": "cat"}]})
        self.assert_rejected({"batch": "b", "changes": [
            {"sha256": d, "new": "dog"}]})

    def test_digest_must_be_nonempty_string(self) -> None:
        self.assert_rejected({"batch": "b", "changes": [
            {"sha256": "", "old": "cat", "new": "dog"}]})
        self.assert_rejected({"batch": "b", "changes": [
            {"sha256": 123, "old": "cat", "new": "dog"}]})

    def test_label_must_be_string_or_null(self) -> None:
        d = self.add_sample("cat")
        for bad_old in [123, True, ["cat"], {"cat": 1}]:
            self.assert_rejected({"batch": "b", "changes": [
                {"sha256": d, "old": bad_old, "new": "dog"}]})
        for bad_new in [123, True, ["dog"]]:
            self.assert_rejected({"batch": "b", "changes": [
                {"sha256": d, "old": "cat", "new": bad_new}]})

    def test_missing_digest_rejects_whole_batch(self) -> None:
        d = self.add_sample("cat")
        with self.assertRaises(BatchError) as ctx:
            self.submit("b", (d, "cat", "dog"), ("f" * 64, "cat", "fish"))
        self.assertIn("not found", str(ctx.exception))
        # Nothing changed.
        self.assertEqual(self.store.lookup_label(d)["label"], "cat")

    def test_duplicate_digest_rejects_whole_batch(self) -> None:
        d = self.add_sample("cat")
        with self.assertRaises(BatchError) as ctx:
            self.store.submit_batch({"batch": "b", "changes": [
                {"sha256": d, "old": "cat", "new": "dog"},
                {"sha256": d, "old": "cat", "new": "fish"},
            ]})
        self.assertIn("duplicate", str(ctx.exception))
        self.assertEqual(self.store.lookup_label(d)["label"], "cat")

    def test_old_label_mismatch_rejects_whole_batch(self) -> None:
        d1 = self.add_sample("cat")
        d2 = self.add_sample("cat")
        with self.assertRaises(BatchError) as ctx:
            self.submit("b", (d1, "cat", "dog"), (d2, "kitten", "fish"))
        message = str(ctx.exception)
        self.assertIn(d2, message)
        self.assertIn("kitten", message)
        self.assertIn("cat", message)
        # Neither sample changed.
        self.assertEqual(self.store.lookup_label(d1)["label"], "cat")
        self.assertEqual(self.store.lookup_label(d2)["label"], "cat")

    def test_null_and_empty_spellings_match_as_old_labels(self) -> None:
        d1 = self.add_sample(None)
        d2 = self.add_sample("")
        # Expected old label "" matches a sample stored as null.
        result = self.store.submit_batch({"batch": "b", "changes": [
            {"sha256": d1, "old": "", "new": "cat"},
            {"sha256": d2, "old": None, "new": "dog"},
        ]})
        self.assertEqual(result["status"], "applied")

    def test_malformed_json_file_rejected(self) -> None:
        d = self.add_sample("cat")
        batch_path = self.root.parent / "bad.json"
        batch_path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(BatchError):
            self.store.submit_batch_file(batch_path)
        self.assertEqual(self.store.lookup_label(d)["label"], "cat")


class BatchUndoTest(StoreHarness):
    def test_undo_restores_old_labels(self) -> None:
        d1 = self.add_sample("cat")
        d2 = self.add_sample("dog")
        self.submit("b1", (d1, "cat", "kitten"), (d2, "dog", None))
        result = self.store.undo_batch("b1")
        self.assertEqual(result["status"], "undone")
        self.assertEqual(result["restored"], 2)
        self.assertEqual(self.store.lookup_label(d1)["label"], "cat")
        self.assertEqual(self.store.lookup_label(d2)["label"], "dog")
        history = self.store.history()
        self.assertTrue(history[0]["undone"])
        self.assertIsNotNone(history[0]["undone_at"])

    def test_undo_only_changed_samples_checked(self) -> None:
        d1 = self.add_sample("cat")
        d2 = self.add_sample("dog")
        # d2 is a no-op in b1.
        self.submit("b1", (d1, "cat", "kitten"), (d2, "dog", "dog"))
        # Modify d2 afterwards; it should not block undo of b1.
        self.submit("b2", (d2, "dog", "fish"))
        result = self.store.undo_batch("b1")
        self.assertEqual(result["status"], "undone")
        self.assertEqual(result["restored"], 1)
        self.assertEqual(self.store.lookup_label(d1)["label"], "cat")
        self.assertEqual(self.store.lookup_label(d2)["label"], "fish")

    def test_undo_rejected_when_sample_modified_afterwards(self) -> None:
        d1 = self.add_sample("cat")
        d2 = self.add_sample("dog")
        self.submit("b1", (d1, "cat", "kitten"), (d2, "dog", "puppy"))
        self.submit("b2", (d2, "puppy", "fish"))
        with self.assertRaises(BatchError) as ctx:
            self.store.undo_batch("b1")
        self.assertIn("modified after", str(ctx.exception))
        # Nothing undone.
        self.assertFalse(self.store.history()[0]["undone"])
        self.assertEqual(self.store.lookup_label(d1)["label"], "kitten")
        self.assertEqual(self.store.lookup_label(d2)["label"], "fish")

    def test_undo_rejected_even_if_label_changed_back(self) -> None:
        d = self.add_sample("cat")
        self.submit("b1", (d, "cat", "dog"))
        self.submit("b2", (d, "dog", "cat"))
        with self.assertRaises(BatchError):
            self.store.undo_batch("b1")
        self.assertEqual(self.store.lookup_label(d)["label"], "cat")

    def test_repeated_undo_is_idempotent(self) -> None:
        d = self.add_sample("cat")
        self.submit("b1", (d, "cat", "dog"))
        first = self.store.undo_batch("b1")
        second = self.store.undo_batch("b1")
        self.assertEqual(first["status"], "undone")
        self.assertEqual(second["status"], "already-undone")
        self.assertEqual(second["restored"], 0)
        # Still one history entry.
        self.assertEqual(len(self.store.history()), 1)

    def test_unknown_batch_number_rejected(self) -> None:
        with self.assertRaises(BatchError) as ctx:
            self.store.undo_batch("ghost")
        self.assertIn("unknown batch number", str(ctx.exception))

    def test_undo_then_new_batch_works(self) -> None:
        d = self.add_sample("cat")
        self.submit("b1", (d, "cat", "dog"))
        self.store.undo_batch("b1")
        result = self.submit("b2", (d, "cat", "fish"))
        self.assertEqual(result["status"], "applied")
        self.assertEqual(self.store.lookup_label(d)["label"], "fish")


class BatchSplitInteractionTest(StoreHarness):
    def test_label_changes_leave_existing_splits_unchanged(self) -> None:
        d = self.add_sample("cat")
        plan = self.store.create_split("p", 0, [1, 0, 0]).plan
        self.submit("b1", (d, "cat", "dog"))
        reopened = self.store.get_split("p")
        self.assertEqual(reopened, plan)
        self.assertEqual(reopened["samples"]["distribution"], {"cat": 1})
        self.assertEqual(
            reopened["sets"]["train"]["members"][0]["label"], "cat"
        )

    def test_new_split_uses_current_labels_and_reuse_rules_apply(self) -> None:
        d = self.add_sample("cat")
        first = self.store.create_split("p", 0, [1, 0, 0])
        self.assertTrue(first.created)
        self.submit("b1", (d, "cat", "dog"))

        # Reusing the same name now conflicts (labels differ).
        with self.assertRaises(Exception):
            self.store.create_split("p", 0, [1, 0, 0])

        # A new plan uses the current label.
        second = self.store.create_split("p2", 0, [1, 0, 0])
        self.assertEqual(second.plan["samples"]["distribution"], {"dog": 1})


class BatchRecoveryTest(StoreHarness):
    def test_interrupted_transaction_is_completed_on_open(self) -> None:
        d = self.add_sample("cat")
        # Simulate a crash after the journal was written but before the
        # manifest/history installs: manifest still old, journal prepared.
        new_manifest = json.loads(self.store.manifest_path.read_text(encoding="utf-8"))
        new_manifest["items"][0]["label"] = "dog"
        new_manifest["items"][0]["rev"] = 1
        history = {
            "schema_version": 1,
            "batches": [
                {
                    "batch": "b1",
                    "records": [
                        {"sha256": d, "old": "cat", "new": "dog", "changed": True, "rev": 1}
                    ],
                    "changed_count": 1,
                    "undone": False,
                    "undone_at": None,
                }
            ],
        }
        journal = {"manifest": new_manifest, "batches": history}
        self.store._write_json_atomic(self.store.transaction_path, journal)
        self.assertTrue(self.store.transaction_path.exists())

        reopened = DatasetStore(self.root)
        self.assertEqual(reopened.lookup_label(d)["label"], "dog")
        self.assertEqual([e["batch"] for e in reopened.history()], ["b1"])
        self.assertFalse(self.store.transaction_path.exists())

    def test_old_workspace_without_batch_files_works(self) -> None:
        # Only manifest.json exists (legacy workspace).
        shutil.rmtree(self.store.batches_path, ignore_errors=True)
        self.assertFalse(self.store.batches_path.exists())
        d = self.add_sample("cat")
        result = self.submit("b1", (d, "cat", "dog"))
        self.assertEqual(result["status"], "applied")


def _submit_worker(root: str, number: str, changes: list, queue) -> None:
    try:
        store = DatasetStore(Path(root))
        result = store.submit_batch(
            {"batch": number, "changes": changes}
        )
        queue.put(("ok", result))
    except Exception as error:  # noqa: BLE001 - report back to parent
        queue.put(("error", str(error)))


class BatchConcurrencyTest(StoreHarness):
    def test_concurrent_both_succeed(self) -> None:
        d1 = self.add_sample("cat")
        d2 = self.add_sample("cat")
        ctx = multiprocessing.get_context("fork")
        queue = ctx.Queue()
        p1 = ctx.Process(
            target=_submit_worker,
            args=(str(self.root), "b1",
                  [{"sha256": d1, "old": "cat", "new": "dog"}], queue),
        )
        p2 = ctx.Process(
            target=_submit_worker,
            args=(str(self.root), "b2",
                  [{"sha256": d2, "old": "cat", "new": "fish"}], queue),
        )
        p1.start()
        p2.start()
        p1.join()
        p2.join()
        results = [queue.get_nowait(), queue.get_nowait()]
        self.assertEqual({status for status, _ in results}, {"ok"})
        labels = self.manifest_labels()
        self.assertEqual(labels[d1], "dog")
        self.assertEqual(labels[d2], "fish")
        self.assertEqual(
            sorted(e["batch"] for e in self.store.history()), ["b1", "b2"]
        )

    def test_concurrent_same_batch_is_idempotent(self) -> None:
        d = self.add_sample("cat")
        ctx = multiprocessing.get_context("fork")
        queue = ctx.Queue()
        changes = [{"sha256": d, "old": "cat", "new": "dog"}]
        processes = [
            ctx.Process(target=_submit_worker,
                        args=(str(self.root), "b1", changes, queue))
            for _ in range(2)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        results = [queue.get_nowait(), queue.get_nowait()]
        self.assertEqual({status for status, _ in results}, {"ok"})
        statuses = {payload["status"] for _, payload in results}
        self.assertEqual(statuses, {"applied", "already-applied"})
        self.assertEqual(self.store.lookup_label(d)["label"], "dog")
        self.assertEqual(len(self.store.history()), 1)

    def test_concurrent_conflicting_batches_one_succeeds(self) -> None:
        d = self.add_sample("cat")
        ctx = multiprocessing.get_context("fork")
        queue = ctx.Queue()
        p1 = ctx.Process(
            target=_submit_worker,
            args=(str(self.root), "b1",
                  [{"sha256": d, "old": "cat", "new": "dog"}], queue),
        )
        p2 = ctx.Process(
            target=_submit_worker,
            args=(str(self.root), "b2",
                  [{"sha256": d, "old": "cat", "new": "fish"}], queue),
        )
        p1.start()
        p2.start()
        p1.join()
        p2.join()
        results = [queue.get_nowait(), queue.get_nowait()]
        statuses = [status for status, _ in results]
        self.assertIn("ok", statuses)
        self.assertIn("error", statuses)
        winner = next(payload for status, payload in results if status == "ok")
        loser_message = next(
            message for status, message in results if status == "error"
        )
        self.assertIn("expected old label", loser_message)
        self.assertEqual(winner["changed"], 1)
        # Exactly one label change landed, history has exactly one batch.
        winning_batch = winner["batch"]
        expected_label = "dog" if winning_batch == "b1" else "fish"
        self.assertEqual(self.store.lookup_label(d)["label"], expected_label)
        self.assertEqual(len(self.store.history()), 1)


class BatchCliTest(StoreHarness):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "vision_workbench", *arguments],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_cli_batch_label_history_undo(self) -> None:
        d = self.add_sample("cat")
        batch_path = self.root.parent / "batch.json"
        batch_path.write_text(
            json.dumps(
                {
                    "batch": "cli1",
                    "changes": [
                        {"sha256": d, "old": "cat", "new": "狗"}
                    ],
                }
            ),
            encoding="utf-8",
        )

        submitted = self.run_cli("batch", str(self.root), str(batch_path))
        self.assertEqual(submitted.returncode, 0, submitted.stderr)
        body = json.loads(submitted.stdout)
        self.assertEqual(body["status"], "applied")
        self.assertEqual(body["changed"], 1)

        looked = self.run_cli("label", str(self.root), d)
        self.assertEqual(looked.returncode, 0, looked.stderr)
        self.assertEqual(json.loads(looked.stdout)["label"], "狗")

        history = self.run_cli("history", str(self.root))
        self.assertEqual(history.returncode, 0, history.stderr)
        entries = json.loads(history.stdout)["batches"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["batch"], "cli1")

        undone = self.run_cli("undo", str(self.root), "cli1")
        self.assertEqual(undone.returncode, 0, undone.stderr)
        self.assertEqual(json.loads(undone.stdout)["status"], "undone")
        self.assertEqual(self.store.lookup_label(d)["label"], "cat")

        again = self.run_cli("undo", str(self.root), "cli1")
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertEqual(json.loads(again.stdout)["status"], "already-undone")

    def test_cli_rejected_batch_reports_error(self) -> None:
        d = self.add_sample("cat")
        batch_path = self.root.parent / "bad.json"
        batch_path.write_text(
            json.dumps(
                {
                    "batch": "bad",
                    "changes": [
                        {"sha256": d, "old": "kitten", "new": "dog"}
                    ],
                }
            ),
            encoding="utf-8",
        )
        rejected = self.run_cli("batch", str(self.root), str(batch_path))
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("kitten", rejected.stderr)
        self.assertEqual(self.store.lookup_label(d)["label"], "cat")

    def test_cli_unknown_undo_reports_error(self) -> None:
        rejected = self.run_cli("undo", str(self.root), "nope")
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("unknown batch number", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
