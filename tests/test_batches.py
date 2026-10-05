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


class BatchUndoCorruptHistoryTest(StoreHarness):
    def _history_on_disk(self) -> dict:
        return json.loads(self.store.batches_path.read_text(encoding="utf-8"))

    def _write_history(self, history: dict) -> None:
        self.store.batches_path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _corrupt_record_rev(self, number: str, digest: str, value: object = None,
                            *, remove: bool = False) -> None:
        """Damage one history record's rev (replace it, or drop the field)."""
        history = self._history_on_disk()
        for entry in history["batches"]:
            if entry["batch"] != number:
                continue
            for record in entry["records"]:
                if record["sha256"] == digest:
                    if remove:
                        del record["rev"]
                    else:
                        record["rev"] = value
        self._write_history(history)

    def _set_record_changed(self, number: str, index: int, changed: bool) -> None:
        """Overwrite one target-batch record's changed flag (history damage)."""
        history = self._history_on_disk()
        entry = next(e for e in history["batches"] if e["batch"] == number)
        entry["records"][index]["changed"] = changed
        self._write_history(history)

    def _duplicate_history_record(self, number: str, index: int) -> None:
        """Append a verbatim copy of one target-batch record to its history."""
        history = self._history_on_disk()
        for entry in history["batches"]:
            if entry["batch"] == number:
                entry["records"].append(dict(entry["records"][index]))
        self._write_history(history)

    def test_duplicate_sample_record_rejects_whole_undo_and_restores_nothing(
        self,
    ) -> None:
        # The batch genuinely changed one sample; a verbatim copy of its
        # history row must not restore it twice, bump its revision twice or
        # report two restored samples.  The whole undo is refused.
        d = self.add_sample("cat")
        self.submit("b1", (d, "cat", "dog"))
        self._duplicate_history_record("b1", 0)

        manifest_before = self.store.manifest_path.read_bytes()
        history_before = self.store.batches_path.read_bytes()
        with self.assertRaises(BatchError) as ctx:
            self.store.undo_batch("b1")
        message = str(ctx.exception)
        self.assertIn("Batch history is corrupted", message)
        self.assertIn("'b1'", message)
        self.assertIn(d, message)
        self.assertIn("record #1", message)
        self.assertIn("record #2", message)

        # Nothing restored; neither file rewritten, pruned or repaired.
        self.assertEqual(self.store.lookup_label(d)["label"], "dog")
        manifest = json.loads(self.store.manifest_path.read_text("utf-8"))
        self.assertEqual(manifest["items"][0]["rev"], 1)
        self.assertEqual(self.store.manifest_path.read_bytes(), manifest_before)
        self.assertEqual(self.store.batches_path.read_bytes(), history_before)
        entry = self._history_on_disk()["batches"][0]
        self.assertFalse(entry["undone"])
        self.assertIsNone(entry["undone_at"])

    def test_duplicate_of_an_unchanged_record_is_still_corruption(self) -> None:
        # A repeated no-op row ("it did not change a label") is a repeat
        # too; the changed record ahead of it is not restored either.
        d1 = self.add_sample("cat")
        d2 = self.add_sample("dog")
        self.submit("b1", (d1, "cat", "kitten"), (d2, "dog", "dog"))
        # Duplicate the no-op d2 row (record index 1) at the back.
        history = self._history_on_disk()
        entry = next(e for e in history["batches"] if e["batch"] == "b1")
        entry["records"].append(dict(entry["records"][1]))
        self._write_history(history)

        manifest_before = self.store.manifest_path.read_bytes()
        history_before = self.store.batches_path.read_bytes()
        with self.assertRaises(BatchError) as ctx:
            self.store.undo_batch("b1")
        message = str(ctx.exception)
        self.assertIn("Batch history is corrupted", message)
        self.assertIn(d2, message)
        self.assertIn("record #2", message)
        self.assertIn("record #3", message)
        self.assertEqual(self.store.lookup_label(d1)["label"], "kitten")
        self.assertEqual(self.store.manifest_path.read_bytes(), manifest_before)
        self.assertEqual(self.store.batches_path.read_bytes(), history_before)

    def test_duplicate_is_judged_by_digest_even_when_copy_differs(self) -> None:
        # The copied row carries different label text; identity is the
        # content digest alone, so it is still a repeat.
        d = self.add_sample("cat")
        self.submit("b1", (d, "cat", "dog"))
        history = self._history_on_disk()
        copy = dict(history["batches"][0]["records"][0])
        copy["new"] = "something-else"
        history["batches"][0]["records"].append(copy)
        self._write_history(history)

        with self.assertRaises(BatchError) as ctx:
            self.store.undo_batch("b1")
        self.assertIn(d, str(ctx.exception))
        self.assertEqual(self.store.lookup_label(d)["label"], "dog")

    def test_restorable_rows_at_front_never_mask_a_later_duplicate(self) -> None:
        d1 = self.add_sample("cat")
        d2 = self.add_sample("dog")
        d3 = self.add_sample("fish")
        self.submit(
            "b1",
            (d1, "cat", "kitten"),
            (d2, "dog", "puppy"),
            (d3, "fish", "guppy"),
        )
        history = self._history_on_disk()
        entry = history["batches"][0]
        entry["records"].append(dict(entry["records"][0]))  # d1 again, #4
        self._write_history(history)

        with self.assertRaises(BatchError) as ctx:
            self.store.undo_batch("b1")
        message = str(ctx.exception)
        self.assertIn(d1, message)
        self.assertIn("record #1", message)
        self.assertIn("record #4", message)
        # The two records before the repeat would restore fine; none do.
        for digest, label in ((d1, "kitten"), (d2, "puppy"), (d3, "guppy")):
            self.assertEqual(self.store.lookup_label(digest)["label"], label)

    def test_duplicate_on_already_undone_batch_is_still_rejected(self) -> None:
        d = self.add_sample("cat")
        self.submit("b1", (d, "cat", "dog"))
        self.assertEqual(self.store.undo_batch("b1")["status"], "undone")
        self._duplicate_history_record("b1", 0)

        history_before = self.store.batches_path.read_bytes()
        with self.assertRaises(BatchError) as ctx:
            self.store.undo_batch("b1")
        message = str(ctx.exception)
        self.assertIn("Batch history is corrupted", message)
        self.assertNotIn("already-undone", message)
        # The existing undone marker is preserved, not rewritten.
        self.assertEqual(self.store.batches_path.read_bytes(), history_before)
        self.assertTrue(self._history_on_disk()["batches"][0]["undone"])

    def test_duplicate_in_another_batch_does_not_block_valid_undo(self) -> None:
        d1 = self.add_sample("cat")
        d2 = self.add_sample("dog")
        self.submit("b1", (d1, "cat", "kitten"))
        self.submit("b2", (d2, "dog", "puppy"))
        self._duplicate_history_record("b2", 0)

        # Only the target batch's records are examined.
        result = self.store.undo_batch("b1")
        self.assertEqual(result["status"], "undone")
        self.assertEqual(result["restored"], 1)
        self.assertEqual(self.store.lookup_label(d1)["label"], "cat")
        # The damaged batch itself is still refused rather than undone.
        with self.assertRaises(BatchError):
            self.store.undo_batch("b2")
        self.assertEqual(self.store.lookup_label(d2)["label"], "puppy")

    def test_same_sample_in_two_batches_still_allows_both_undos(self) -> None:
        # One sample taking part in two different batches over time is
        # legal; an intra-batch repeat must not be confused with it.
        d = self.add_sample("cat")
        self.submit("b1", (d, "cat", "dog"))
        self.assertEqual(self.store.undo_batch("b1")["restored"], 1)
        self.submit("b2", (d, "cat", "fish"))
        self.assertEqual(self.store.undo_batch("b2")["restored"], 1)
        self.assertEqual(self.store.lookup_label(d)["label"], "cat")
        self.assertEqual(self.store.undo_batch("b1")["status"], "already-undone")
        self.assertEqual(self.store.undo_batch("b2")["status"], "already-undone")

    def test_unknown_batch_stays_unknown_even_with_duplicate_batch(self) -> None:
        d = self.add_sample("cat")
        self.submit("b1", (d, "cat", "dog"))
        self._duplicate_history_record("b1", 0)
        with self.assertRaises(BatchError) as ctx:
            self.store.undo_batch("ghost")
        self.assertIn("unknown batch number", str(ctx.exception))

    def test_missing_rev_rejects_whole_undo_and_restores_nothing(self) -> None:
        # The batch changed two samples; the first record is restorable but
        # the second lacks a revision.  The whole undo must be refused and
        # the first sample must keep its post-batch label.
        d1 = self.add_sample("cat")
        d2 = self.add_sample("dog")
        self.submit("b1", (d1, "cat", "kitten"), (d2, "dog", "puppy"))
        self._corrupt_record_rev("b1", d2, remove=True)

        manifest_before = self.store.manifest_path.read_bytes()
        history_before = self.store.batches_path.read_bytes()
        with self.assertRaises(BatchError) as ctx:
            self.store.undo_batch("b1")
        message = str(ctx.exception)
        self.assertIn("Batch history is corrupted", message)
        self.assertIn("'b1'", message)
        self.assertIn(d2, message)
        self.assertIn("missing 'rev'", message)

        # First sample untouched; neither file was rewritten or repaired.
        self.assertEqual(self.store.lookup_label(d1)["label"], "kitten")
        self.assertEqual(self.store.lookup_label(d2)["label"], "puppy")
        self.assertEqual(self.store.manifest_path.read_bytes(), manifest_before)
        self.assertEqual(self.store.batches_path.read_bytes(), history_before)
        entry = self._history_on_disk()["batches"][0]
        self.assertFalse(entry["undone"])
        self.assertIsNone(entry["undone_at"])
        self.assertNotIn("rev", next(
            r for r in entry["records"] if r["sha256"] == d2))

    def test_every_bad_rev_shape_is_rejected(self) -> None:
        for bad in (True, 1.0, "1", None):
            with self.subTest(bad=bad):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory) / "dataset"
                    store = DatasetStore(root)
                    store.initialize()
                    source = Path(directory) / "sample.jpg"
                    source.write_bytes(f"content-{bad!r}".encode())
                    digest = store.add(source, "cat").digest
                    store.submit_batch({
                        "batch": "b1",
                        "changes": [{"sha256": digest, "old": "cat", "new": "dog"}],
                    })
                    history = json.loads(store.batches_path.read_text("utf-8"))
                    history["batches"][0]["records"][0]["rev"] = bad
                    store.batches_path.write_text(
                        json.dumps(history), encoding="utf-8"
                    )
                    with self.assertRaises(BatchError) as ctx:
                        store.undo_batch("b1")
                    message = str(ctx.exception)
                    self.assertIn("Batch history is corrupted", message)
                    self.assertIn(digest, message)
                    # Nothing restored despite the numeric look-alike match.
                    self.assertEqual(store.lookup_label(digest)["label"], "dog")

    def test_true_or_decimal_rev_that_would_compare_equal_still_refused(self) -> None:
        # Before the fix True/1.0/"1" compared equal to revision 1 and the
        # undo went through.  It must now be reported as corruption instead.
        d = self.add_sample("cat")
        self.submit("b1", (d, "cat", "dog"))
        for bad in (True, 1.0, "1"):
            self._corrupt_record_rev("b1", d, bad)
            with self.assertRaises(BatchError) as ctx:
                self.store.undo_batch("b1")
            message = str(ctx.exception)
            self.assertIn("Batch history is corrupted", message)
            self.assertNotIn("modified after", message)
            self.assertEqual(self.store.lookup_label(d)["label"], "dog")
            # Repair for the next sub-case.
            self._corrupt_record_rev("b1", d, 1)

    def test_corruption_on_already_undone_batch_is_still_rejected(self) -> None:
        d = self.add_sample("cat")
        self.submit("b1", (d, "cat", "dog"))
        first = self.store.undo_batch("b1")
        self.assertEqual(first["status"], "undone")
        self._corrupt_record_rev("b1", d, remove=True)

        # A repeated undo must not hide the damage behind already-undone.
        with self.assertRaises(BatchError) as ctx:
            self.store.undo_batch("b1")
        self.assertIn("Batch history is corrupted", str(ctx.exception))
        self.assertIn(d, str(ctx.exception))
        entry = self._history_on_disk()["batches"][0]
        self.assertTrue(entry["undone"])  # marker unchanged, no new write

    def test_corruption_in_another_batch_does_not_block_valid_undo(self) -> None:
        d1 = self.add_sample("cat")
        d2 = self.add_sample("dog")
        self.submit("b1", (d1, "cat", "kitten"))
        self.submit("b2", (d2, "dog", "puppy"))
        self._corrupt_record_rev("b2", d2, True)

        # Only the target batch's integrity matters for its undo.
        result = self.store.undo_batch("b1")
        self.assertEqual(result["status"], "undone")
        self.assertEqual(result["restored"], 1)
        self.assertEqual(self.store.lookup_label(d1)["label"], "cat")
        self.assertEqual(self.store.lookup_label(d2)["label"], "puppy")

    def test_unknown_batch_still_unknown_even_if_history_is_corrupt(self) -> None:
        d = self.add_sample("cat")
        self.submit("b1", (d, "cat", "dog"))
        self._corrupt_record_rev("b1", d, None)
        with self.assertRaises(BatchError) as ctx:
            self.store.undo_batch("ghost")
        self.assertIn("unknown batch number", str(ctx.exception))

    def test_zero_rev_on_unchanged_record_stays_valid(self) -> None:
        d1 = self.add_sample("cat")
        d2 = self.add_sample("dog")
        # d2 is a no-op in b1; its history row pins rev 0.
        self.submit("b1", (d1, "cat", "kitten"), (d2, "dog", "dog"))
        # d2 is later changed by another batch; that must not block b1.
        self.submit("b2", (d2, "dog", "fish"))
        result = self.store.undo_batch("b1")
        self.assertEqual(result["status"], "undone")
        self.assertEqual(result["restored"], 1)
        self.assertEqual(self.store.lookup_label(d1)["label"], "cat")
        self.assertEqual(self.store.lookup_label(d2)["label"], "fish")
        # Repeated undo remains the normal idempotent response.
        self.assertEqual(
            self.store.undo_batch("b1")["status"], "already-undone"
        )


class BatchUndoChangeFlagCorruptionTest(StoreHarness):
    def _history_on_disk(self) -> dict:
        return json.loads(self.store.batches_path.read_text(encoding="utf-8"))

    def _write_history(self, history: dict) -> None:
        self.store.batches_path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _set_record(self, number: str, index: int, **fields: object) -> None:
        history = self._history_on_disk()
        entry = next(e for e in history["batches"] if e["batch"] == number)
        entry["records"][index].update(fields)
        self._write_history(history)

    def test_hidden_change_flagged_unchanged_leaves_dog_and_refuses_undo(self) -> None:
        # The exact motivating damage: cat -> dog is recorded but flagged as
        # unchanged.  Undo must not "succeed" while leaving dog behind.
        d = self.add_sample("cat")
        self.submit("b1", (d, "cat", "dog"))
        self._set_record("b1", 0, changed=False)

        manifest_before = self.store.manifest_path.read_bytes()
        history_before = self.store.batches_path.read_bytes()
        with self.assertRaises(BatchError) as ctx:
            self.store.undo_batch("b1")
        message = str(ctx.exception)
        self.assertIn("Batch history is corrupted", message)
        self.assertIn("'b1'", message)
        self.assertIn(d, message)
        self.assertIn("record #1", message)
        self.assertIn("'changed' is false", message)

        # The sample stays at dog; neither file is rewritten and no undo
        # marker is added.
        self.assertEqual(self.store.lookup_label(d)["label"], "dog")
        manifest = json.loads(self.store.manifest_path.read_text("utf-8"))
        self.assertEqual(manifest["items"][0]["rev"], 1)
        self.assertEqual(self.store.manifest_path.read_bytes(), manifest_before)
        self.assertEqual(self.store.batches_path.read_bytes(), history_before)
        entry = self._history_on_disk()["batches"][0]
        self.assertFalse(entry["undone"])
        self.assertIsNone(entry["undone_at"])

    def test_identical_labels_flagged_changed_would_inflate_restore_count(self) -> None:
        # A genuinely unchanged row flagged changed would be "restored" even
        # though the batch never touched the sample; it must be refused.
        # Craft a batch with one real change (d1) plus a genuine no-op row
        # (d2), then damage the no-op row's flag.
        d = self.add_sample("cat")
        d2 = self.add_sample("dog")
        self.submit("b1", (d, "cat", "dog"), (d2, "dog", "dog"))
        self._set_record("b1", 1, changed=True)  # dog -> dog flagged changed

        manifest_before = self.store.manifest_path.read_bytes()
        history_before = self.store.batches_path.read_bytes()
        with self.assertRaises(BatchError) as ctx:
            self.store.undo_batch("b1")
        message = str(ctx.exception)
        self.assertIn("Batch history is corrupted", message)
        self.assertIn(d2, message)
        self.assertIn("record #2", message)
        self.assertIn("'changed' is true", message)
        # Neither the changed d1 nor the unchanged d2 is restored; the
        # restored count can never include d2.
        self.assertEqual(self.store.lookup_label(d)["label"], "dog")
        self.assertEqual(self.store.lookup_label(d2)["label"], "dog")
        self.assertEqual(self.store.manifest_path.read_bytes(), manifest_before)
        self.assertEqual(self.store.batches_path.read_bytes(), history_before)

    def test_restorable_rows_ahead_of_damage_are_not_restored_first(self) -> None:
        d1 = self.add_sample("cat")
        d2 = self.add_sample("dog")
        d3 = self.add_sample("fish")
        self.submit(
            "b1",
            (d1, "cat", "kitten"),
            (d2, "dog", "puppy"),
            (d3, "fish", "guppy"),
        )
        # The last record hides its real change; the first two would restore
        # fine, but the whole undo must fail before any of them is touched.
        self._set_record("b1", 2, changed=False)

        with self.assertRaises(BatchError) as ctx:
            self.store.undo_batch("b1")
        message = str(ctx.exception)
        self.assertIn(d3, message)
        self.assertIn("record #3", message)
        for digest, label in ((d1, "kitten"), (d2, "puppy"), (d3, "guppy")):
            self.assertEqual(self.store.lookup_label(digest)["label"], label)
        entry = self._history_on_disk()["batches"][0]
        self.assertFalse(entry["undone"])

    def test_null_and_empty_string_swapped_is_not_a_change(self) -> None:
        # A real unlabeled -> cat change followed by a row whose stored old
        # is "" and new null: the two unlabeled spellings must agree with
        # changed=false and reject changed=true.
        d1 = self.add_sample(None)
        self.submit("b1", (d1, None, "cat"))
        # Craft a second, genuinely-unchanged unlabeled row by hand.
        d2 = self.add_sample(None)
        history = self._history_on_disk()
        history["batches"][0]["records"].append(
            {"sha256": d2, "old": "", "new": None, "changed": True, "rev": 0}
        )
        self._write_history(history)
        with self.assertRaises(BatchError) as ctx:
            self.store.undo_batch("b1")
        self.assertIn(d2, str(ctx.exception))
        self.assertIn("record #2", str(ctx.exception))
        # And the same row with changed=false is consistent: undo proceeds
        # and restores only the one sample the batch actually changed.
        self._set_record("b1", 1, changed=False)
        result = self.store.undo_batch("b1")
        self.assertEqual(result["status"], "undone")
        self.assertEqual(result["restored"], 1)
        self.assertIsNone(self.store.lookup_label(d1)["label"])
        self.assertIsNone(self.store.lookup_label(d2)["label"])

    def test_literal_unlabeled_is_a_label_not_the_unlabeled_state(self) -> None:
        d = self.add_sample("unlabeled")
        # Batch records unlabeled(literal) -> null but lies "unchanged".
        self.submit("b1", (d, "unlabeled", None))
        self._set_record("b1", 0, changed=False)
        with self.assertRaises(BatchError) as ctx:
            self.store.undo_batch("b1")
        message = str(ctx.exception)
        self.assertIn("Batch history is corrupted", message)
        self.assertIn(d, message)
        self.assertIn("differ", message)

    def test_case_whitespace_and_separators_are_kept_literal(self) -> None:
        for old, new in (("cat", "Cat"), ("cat", "cat "), ("a/b", "a\\b"), ("猫", "狗")):
            with self.subTest(old=old, new=new):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory) / "dataset"
                    store = DatasetStore(root)
                    store.initialize()
                    source = Path(directory) / "sample.jpg"
                    source.write_bytes(f"content-{old}-{new}".encode())
                    digest = store.add(source, old).digest
                    store.submit_batch({
                        "batch": "b1",
                        "changes": [{"sha256": digest, "old": old, "new": new}],
                    })
                    history = json.loads(store.batches_path.read_text("utf-8"))
                    history["batches"][0]["records"][0]["changed"] = False
                    store.batches_path.write_text(
                        json.dumps(history), encoding="utf-8"
                    )
                    with self.assertRaises(BatchError) as ctx:
                        store.undo_batch("b1")
                    self.assertIn("Batch history is corrupted", str(ctx.exception))
                    self.assertEqual(store.lookup_label(digest)["label"], new)

    def test_judged_from_history_labels_not_current_label(self) -> None:
        # Even when the sample's current label happens to equal the stored
        # old label (so the row "looks" unchanged today), a history row with
        # differing old/new and changed=false is still corruption: the
        # decision never substitutes the current label.
        d = self.add_sample("cat")
        self.submit("b1", (d, "cat", "dog"))
        # A later batch changes dog back to cat.
        self.submit("b2", (d, "dog", "cat"))
        self._set_record("b1", 0, changed=False)
        with self.assertRaises(BatchError) as ctx:
            self.store.undo_batch("b1")
        self.assertIn("Batch history is corrupted", str(ctx.exception))
        self.assertIn(d, str(ctx.exception))
        # It must not be mistaken for the later-modification rejection.
        self.assertNotIn("modified after", str(ctx.exception))
        self.assertEqual(self.store.lookup_label(d)["label"], "cat")

    def test_damage_reported_when_batch_already_undone(self) -> None:
        d = self.add_sample("cat")
        self.submit("b1", (d, "cat", "dog"))
        self.assertEqual(self.store.undo_batch("b1")["status"], "undone")
        self._set_record("b1", 0, changed=False)

        history_before = self.store.batches_path.read_bytes()
        with self.assertRaises(BatchError) as ctx:
            self.store.undo_batch("b1")
        message = str(ctx.exception)
        self.assertIn("Batch history is corrupted", message)
        self.assertIn(d, message)
        self.assertNotIn("already-undone", message)
        # The existing undone marker is preserved; the file is not rewritten.
        self.assertEqual(self.store.batches_path.read_bytes(), history_before)
        self.assertTrue(self._history_on_disk()["batches"][0]["undone"])

    def test_damage_in_another_batch_does_not_block_valid_undo(self) -> None:
        d1 = self.add_sample("cat")
        d2 = self.add_sample("dog")
        self.submit("b1", (d1, "cat", "kitten"))
        self.submit("b2", (d2, "dog", "puppy"))
        self._set_record("b2", 0, changed=False)  # hide b2's change

        result = self.store.undo_batch("b1")
        self.assertEqual(result["status"], "undone")
        self.assertEqual(result["restored"], 1)
        self.assertEqual(self.store.lookup_label(d1)["label"], "cat")
        with self.assertRaises(BatchError):
            self.store.undo_batch("b2")
        self.assertEqual(self.store.lookup_label(d2)["label"], "puppy")

    def test_unknown_batch_stays_unknown_even_with_flag_damage(self) -> None:
        d = self.add_sample("cat")
        self.submit("b1", (d, "cat", "dog"))
        self._set_record("b1", 0, changed=False)
        with self.assertRaises(BatchError) as ctx:
            self.store.undo_batch("ghost")
        self.assertIn("unknown batch number", str(ctx.exception))

    def test_valid_batch_still_restores_only_real_changes_accurately(self) -> None:
        d1 = self.add_sample("cat")
        d2 = self.add_sample("dog")
        # d2 is a genuine no-op in b1; its later change by b2 must not
        # block b1, and the restored count counts d1 alone.
        self.submit("b1", (d1, "cat", "kitten"), (d2, "dog", "dog"))
        self.submit("b2", (d2, "dog", "fish"))
        result = self.store.undo_batch("b1")
        self.assertEqual(result["status"], "undone")
        self.assertEqual(result["restored"], 1)
        self.assertEqual(self.store.lookup_label(d1)["label"], "cat")
        self.assertEqual(self.store.lookup_label(d2)["label"], "fish")
        self.assertEqual(
            self.store.undo_batch("b1")["status"], "already-undone"
        )

    def test_cli_flag_corruption_fails_cleanly_with_nonzero_exit(self) -> None:
        d = self.add_sample("cat")
        self.submit("b1", (d, "cat", "dog"))
        self._set_record("b1", 0, changed=False)

        rejected = subprocess.run(
            [sys.executable, "-m", "vision_workbench", "undo",
             str(self.root), "b1"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertEqual(rejected.stdout, "")
        self.assertIn("Batch history is corrupted", rejected.stderr)
        self.assertIn("b1", rejected.stderr)
        self.assertIn(d, rejected.stderr)
        self.assertIn("record #1", rejected.stderr)
        self.assertNotIn("Traceback", rejected.stderr)
        self.assertEqual(self.store.lookup_label(d)["label"], "dog")


class BatchUndoMissingLabelsTest(StoreHarness):
    def _history_on_disk(self) -> dict:
        return json.loads(self.store.batches_path.read_text(encoding="utf-8"))

    def _write_history(self, history: dict) -> None:
        self.store.batches_path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _drop_record_fields(self, number: str, index: int, *keys: str) -> None:
        """Remove label fields from one target-batch record (history damage)."""
        history = self._history_on_disk()
        entry = next(e for e in history["batches"] if e["batch"] == number)
        for key in keys:
            del entry["records"][index][key]
        self._write_history(history)

    def test_missing_old_rejects_whole_undo_and_restores_nothing(self) -> None:
        d1 = self.add_sample("cat")
        d2 = self.add_sample("dog")
        self.submit("b1", (d1, "cat", "kitten"), (d2, "dog", "puppy"))
        self._drop_record_fields("b1", 1, "old")

        manifest_before = self.store.manifest_path.read_bytes()
        history_before = self.store.batches_path.read_bytes()
        with self.assertRaises(BatchError) as ctx:
            self.store.undo_batch("b1")
        message = str(ctx.exception)
        self.assertIn("Batch history is corrupted", message)
        self.assertIn("'b1'", message)
        self.assertIn(d2, message)
        self.assertIn("record #2", message)
        self.assertIn("missing 'old' label", message)

        # The restorable first record is not restored either; neither file
        # is rewritten, pruned or repaired, and no undo marker appears.
        self.assertEqual(self.store.lookup_label(d1)["label"], "kitten")
        self.assertEqual(self.store.lookup_label(d2)["label"], "puppy")
        self.assertEqual(self.store.manifest_path.read_bytes(), manifest_before)
        self.assertEqual(self.store.batches_path.read_bytes(), history_before)
        entry = self._history_on_disk()["batches"][0]
        self.assertFalse(entry["undone"])
        self.assertIsNone(entry["undone_at"])

    def test_missing_new_rejects_and_names_position(self) -> None:
        d = self.add_sample("cat")
        self.submit("b1", (d, "cat", "dog"))
        self._drop_record_fields("b1", 0, "new")

        with self.assertRaises(BatchError) as ctx:
            self.store.undo_batch("b1")
        message = str(ctx.exception)
        self.assertIn("Batch history is corrupted", message)
        self.assertIn(d, message)
        self.assertIn("record #1", message)
        self.assertIn("missing 'new' label", message)
        self.assertEqual(self.store.lookup_label(d)["label"], "dog")

    def test_missing_both_is_not_an_unlabeled_to_unlabeled_noop(self) -> None:
        # Dropping both labels must not read as an unlabeled->unlabeled
        # record that undo can silently skip.
        d = self.add_sample("cat")
        self.submit("b1", (d, "cat", "dog"))
        self._drop_record_fields("b1", 0, "old", "new")

        history_before = self.store.batches_path.read_bytes()
        with self.assertRaises(BatchError) as ctx:
            self.store.undo_batch("b1")
        message = str(ctx.exception)
        self.assertIn("Batch history is corrupted", message)
        self.assertIn("missing 'old' and 'new' labels", message)
        self.assertEqual(self.store.lookup_label(d)["label"], "dog")
        self.assertEqual(self.store.batches_path.read_bytes(), history_before)

    def test_missing_labels_on_unchanged_record_still_reject(self) -> None:
        # The damaged row is a no-op the undo would never restore; the
        # whole undo is still refused and the changed row keeps its label.
        d1 = self.add_sample("cat")
        d2 = self.add_sample("dog")
        self.submit("b1", (d1, "cat", "kitten"), (d2, "dog", "dog"))
        self._drop_record_fields("b1", 1, "new")

        with self.assertRaises(BatchError) as ctx:
            self.store.undo_batch("b1")
        message = str(ctx.exception)
        self.assertIn("Batch history is corrupted", message)
        self.assertIn(d2, message)
        self.assertIn("record #2", message)
        self.assertEqual(self.store.lookup_label(d1)["label"], "kitten")

    def test_damage_behind_normal_records_rejects_and_restores_nothing(self) -> None:
        d1 = self.add_sample("cat")
        d2 = self.add_sample("dog")
        d3 = self.add_sample("fish")
        self.submit(
            "b1",
            (d1, "cat", "kitten"),
            (d2, "dog", "puppy"),
            (d3, "fish", "guppy"),
        )
        self._drop_record_fields("b1", 2, "old")

        with self.assertRaises(BatchError) as ctx:
            self.store.undo_batch("b1")
        message = str(ctx.exception)
        self.assertIn(d3, message)
        self.assertIn("record #3", message)
        for digest, label in ((d1, "kitten"), (d2, "puppy"), (d3, "guppy")):
            self.assertEqual(self.store.lookup_label(digest)["label"], label)

    def test_explicit_null_and_empty_string_labels_still_undo(self) -> None:
        # Saved null/"" are legal unlabeled spellings, not missing fields.
        d1 = self.add_sample(None)
        d2 = self.add_sample("cat")
        self.submit("b1", (d1, None, "cat"), (d2, "cat", ""))
        result = self.store.undo_batch("b1")
        self.assertEqual(result["status"], "undone")
        self.assertEqual(result["restored"], 2)
        self.assertIsNone(self.store.lookup_label(d1)["label"])
        self.assertEqual(self.store.lookup_label(d2)["label"], "cat")

    def test_literal_unlabeled_class_still_undoes(self) -> None:
        d = self.add_sample("unlabeled")
        self.submit("b1", (d, "unlabeled", "cat"))
        result = self.store.undo_batch("b1")
        self.assertEqual(result["status"], "undone")
        self.assertEqual(self.store.lookup_label(d)["label"], "unlabeled")

    def test_damage_on_already_undone_batch_is_still_rejected(self) -> None:
        d = self.add_sample("cat")
        self.submit("b1", (d, "cat", "dog"))
        self.assertEqual(self.store.undo_batch("b1")["status"], "undone")
        self._drop_record_fields("b1", 0, "old")

        history_before = self.store.batches_path.read_bytes()
        with self.assertRaises(BatchError) as ctx:
            self.store.undo_batch("b1")
        message = str(ctx.exception)
        self.assertIn("Batch history is corrupted", message)
        self.assertNotIn("already-undone", message)
        # The existing undone marker is preserved, not rewritten.
        self.assertEqual(self.store.batches_path.read_bytes(), history_before)
        self.assertTrue(self._history_on_disk()["batches"][0]["undone"])

    def test_damage_in_another_batch_does_not_block_valid_undo(self) -> None:
        d1 = self.add_sample("cat")
        d2 = self.add_sample("dog")
        self.submit("b1", (d1, "cat", "kitten"))
        self.submit("b2", (d2, "dog", "puppy"))
        self._drop_record_fields("b2", 0, "old", "new")

        # Only the target batch's records are examined.
        result = self.store.undo_batch("b1")
        self.assertEqual(result["status"], "undone")
        self.assertEqual(result["restored"], 1)
        self.assertEqual(self.store.lookup_label(d1)["label"], "cat")
        # The damaged batch itself is still refused rather than undone.
        with self.assertRaises(BatchError):
            self.store.undo_batch("b2")
        self.assertEqual(self.store.lookup_label(d2)["label"], "puppy")

    def test_unknown_batch_stays_unknown_even_with_label_damage(self) -> None:
        d = self.add_sample("cat")
        self.submit("b1", (d, "cat", "dog"))
        self._drop_record_fields("b1", 0, "new")
        with self.assertRaises(BatchError) as ctx:
            self.store.undo_batch("ghost")
        self.assertIn("unknown batch number", str(ctx.exception))

    def test_cli_missing_label_fails_cleanly_with_nonzero_exit(self) -> None:
        d = self.add_sample("cat")
        self.submit("b1", (d, "cat", "dog"))
        self._drop_record_fields("b1", 0, "new")

        rejected = subprocess.run(
            [sys.executable, "-m", "vision_workbench", "undo",
             str(self.root), "b1"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertEqual(rejected.stdout, "")
        self.assertIn("Batch history is corrupted", rejected.stderr)
        self.assertIn("b1", rejected.stderr)
        self.assertIn(d, rejected.stderr)
        self.assertIn("record #1", rejected.stderr)
        self.assertIn("missing 'new' label", rejected.stderr)
        self.assertNotIn("Traceback", rejected.stderr)
        self.assertEqual(self.store.lookup_label(d)["label"], "dog")


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

    def test_cli_corrupt_history_undo_fails_cleanly(self) -> None:
        d = self.add_sample("cat")
        d2 = self.add_sample("dog")
        self.submit("b1", (d, "cat", "kitten"), (d2, "dog", "puppy"))
        history_path = self.store.batches_path
        history = json.loads(history_path.read_text(encoding="utf-8"))
        history["batches"][0]["records"][1]["rev"] = True
        history_path.write_text(json.dumps(history), encoding="utf-8")

        rejected = self.run_cli("undo", str(self.root), "b1")
        self.assertNotEqual(rejected.returncode, 0)
        self.assertEqual(rejected.stdout, "")
        self.assertIn("Batch history is corrupted", rejected.stderr)
        self.assertIn("b1", rejected.stderr)
        self.assertIn(d2, rejected.stderr)
        # A clear message, never a program traceback.
        self.assertNotIn("Traceback", rejected.stderr)
        # First sample was not restored either.
        self.assertEqual(self.store.lookup_label(d)["label"], "kitten")
        self.assertEqual(self.store.lookup_label(d2)["label"], "puppy")

    def test_cli_duplicate_history_record_undo_fails_cleanly(self) -> None:
        d = self.add_sample("cat")
        self.submit("b1", (d, "cat", "dog"))
        history_path = self.store.batches_path
        history = json.loads(history_path.read_text(encoding="utf-8"))
        # Copy the one record verbatim, so the sample appears twice.
        history["batches"][0]["records"].append(
            dict(history["batches"][0]["records"][0])
        )
        history_path.write_text(json.dumps(history), encoding="utf-8")

        rejected = self.run_cli("undo", str(self.root), "b1")
        self.assertNotEqual(rejected.returncode, 0)
        self.assertEqual(rejected.stdout, "")
        self.assertIn("Batch history is corrupted", rejected.stderr)
        self.assertIn("b1", rejected.stderr)
        self.assertIn(d, rejected.stderr)
        self.assertIn("record #1", rejected.stderr)
        self.assertIn("record #2", rejected.stderr)
        self.assertNotIn("already-undone", rejected.stderr)
        # A clear message, never a program traceback.
        self.assertNotIn("Traceback", rejected.stderr)
        # The sample was not restored and its revision bumped only once.
        self.assertEqual(self.store.lookup_label(d)["label"], "dog")
        manifest = json.loads(self.store.manifest_path.read_text("utf-8"))
        self.assertEqual(manifest["items"][0]["rev"], 1)


if __name__ == "__main__":
    unittest.main()
