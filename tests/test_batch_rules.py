"""Unit tests for the pure batch submission rules in ``batches``.

These rules deliberately take plain records and callbacks instead of a
:class:`~vision_workbench.store.DatasetStore`, so the conditions that
reject a submission can be tested independently of the workspace save.
"""

from __future__ import annotations

import unittest

from vision_workbench.batches import (
    BatchError,
    BatchResolution,
    apply_resolutions,
    build_history_entry,
    history_record_revision,
    no_changes_result,
    normalize_label,
    reject_unknown_samples,
    replay_result,
    resolve_re_submission,
    result_payload,
    validate_undo_history,
    verify_records,
)


def record(digest: str, old, new) -> dict:
    # Mirror parse_batch_file's output: labels are already normalized.
    return {"sha256": digest, "old": normalize_label(old),
            "new": normalize_label(new)}


class RejectUnknownSamplesTest(unittest.TestCase):
    def test_unknown_digest_reports_first_unknown_record(self) -> None:
        records = [record("aaa", "cat", "dog"), record("bbb", "cat", "fish")]
        with self.assertRaises(BatchError) as ctx:
            reject_unknown_samples(records, {"aaa"}.__contains__)
        self.assertEqual(str(ctx.exception), "sample not found: bbb")

    def test_all_known_passes(self) -> None:
        records = [record("aaa", "cat", "dog"), record("bbb", None, "cat")]
        reject_unknown_samples(records, {"aaa", "bbb"}.__contains__)


class ReSubmissionTest(unittest.TestCase):
    def entry(self, **overrides) -> dict:
        entry = {
            "batch": "b1",
            "records": [
                {"sha256": "d1", "old": "cat", "new": "dog", "changed": True, "rev": 1},
                {"sha256": "d2", "old": "cat", "new": "cat", "changed": False, "rev": 0},
            ],
            "changed_count": 1,
            "undone": False,
            "undone_at": None,
        }
        entry.update(overrides)
        return entry

    def test_free_number_returns_none(self) -> None:
        self.assertIsNone(
            resolve_re_submission("b1", [record("d1", "cat", "dog")], None)
        )

    def test_same_content_replays_original_stats(self) -> None:
        records = [
            record("d2", "cat", "cat"),
            record("d1", "cat", "dog"),
        ]
        result = resolve_re_submission("b1", records, self.entry())
        self.assertEqual(
            result,
            {"batch": "b1", "status": "already-applied",
             "changed": 1, "unchanged": 1, "total": 2},
        )

    def test_null_and_empty_spelling_still_same_content(self) -> None:
        entry = self.entry(
            records=[
                {"sha256": "d1", "old": None, "new": "cat", "changed": True, "rev": 1}
            ],
            changed_count=1,
        )
        result = resolve_re_submission("b1", [record("d1", "", "cat")], entry)
        self.assertEqual(result["status"], "already-applied")

    def test_different_content_conflicts(self) -> None:
        with self.assertRaises(BatchError) as ctx:
            resolve_re_submission(
                "b1", [record("d1", "cat", "fish")], self.entry()
            )
        self.assertIn("already used with different content", str(ctx.exception))

    def test_undone_batch_same_content_still_only_replays(self) -> None:
        # An undone batch must not come back to life through re-submission:
        # equal content replays, it never re-applies.
        result = resolve_re_submission(
            "b1", [record("d1", "cat", "dog"), record("d2", "cat", "cat")],
            self.entry(undone=True),
        )
        self.assertEqual(result["status"], "already-applied")


class VerifyRecordsTest(unittest.TestCase):
    def test_mismatch_rejects_and_names_record_and_both_labels(self) -> None:
        records = [
            record("d1", "cat", "dog"),
            record("d2", "kitten", "fish"),
        ]
        current = {"d1": "cat", "d2": "cat"}
        with self.assertRaises(BatchError) as ctx:
            verify_records(records, current.__getitem__)
        message = str(ctx.exception)
        self.assertIn("d2", message)
        self.assertIn("kitten", message)
        self.assertIn("cat", message)

    def test_one_pass_classifies_changed_and_unchanged(self) -> None:
        records = [
            record("d1", "cat", "dog"),     # changes
            record("d2", "cat", "cat"),     # no-op
            record("d3", None, "cat"),      # unlabeled -> labeled
            record("d4", "cat", None),      # labeled -> unlabeled
        ]
        current = {"d1": "cat", "d2": "cat", "d3": None, "d4": "cat"}
        looked_up: list[str] = []

        def lookup(digest: str):
            looked_up.append(digest)
            return current[digest]

        resolutions = verify_records(records, lookup)
        self.assertEqual(
            resolutions,
            [
                BatchResolution("d1", "cat", "dog", "cat", True),
                BatchResolution("d2", "cat", "cat", "cat", False),
                BatchResolution("d3", None, "cat", None, True),
                BatchResolution("d4", "cat", None, "cat", True),
            ],
        )
        # The current label of every record is read exactly once.
        self.assertEqual(sorted(looked_up), ["d1", "d2", "d3", "d4"])

    def test_empty_string_expected_old_matches_null_current(self) -> None:
        resolutions = verify_records(
            [record("d1", "", "cat")], {"d1": None}.__getitem__
        )
        self.assertEqual(resolutions, [BatchResolution("d1", None, "cat", None, True)])


class ApplyResolutionsTest(unittest.TestCase):
    def test_only_changed_records_mutate_and_revisions_are_recorded(self) -> None:
        items = {
            "d1": {"sha256": "d1", "label": "cat", "rev": 4},
            "d2": {"sha256": "d2", "label": "cat"},  # missing rev counts as 0
        }
        resolutions = [
            BatchResolution("d1", "cat", "dog", "cat", True),
            BatchResolution("d2", "cat", "cat", "cat", False),
        ]
        applied = apply_resolutions(resolutions, items)
        self.assertEqual(items["d1"]["label"], "dog")
        self.assertEqual(items["d1"]["rev"], 5)
        self.assertEqual(items["d2"]["label"], "cat")
        self.assertNotIn("rev", items["d2"])
        self.assertEqual(
            applied,
            [
                {"sha256": "d1", "old": "cat", "new": "dog",
                 "changed": True, "rev": 5},
                {"sha256": "d2", "old": "cat", "new": "cat",
                 "changed": False, "rev": 0},
            ],
        )

    def test_clearing_label_writes_null(self) -> None:
        items = {"d1": {"sha256": "d1", "label": "cat", "rev": 0}}
        apply_resolutions(
            [BatchResolution("d1", "cat", None, "cat", True)], items
        )
        self.assertIsNone(items["d1"]["label"])


class ResultAssemblyTest(unittest.TestCase):
    def applied_records(self) -> list[dict]:
        return [
            {"sha256": "d1", "old": "cat", "new": "dog", "changed": True, "rev": 1},
            {"sha256": "d2", "old": "cat", "new": "cat", "changed": False, "rev": 0},
        ]

    def test_result_payload_keeps_count_meanings(self) -> None:
        self.assertEqual(
            result_payload("b1", "applied", self.applied_records()),
            {"batch": "b1", "status": "applied",
             "changed": 1, "unchanged": 1, "total": 2},
        )

    def test_no_changes_result_counts_every_record_as_unchanged(self) -> None:
        self.assertEqual(
            no_changes_result("b1", 0),
            {"batch": "b1", "status": "no-changes",
             "changed": 0, "unchanged": 0, "total": 0},
        )
        self.assertEqual(
            no_changes_result("b1", 3),
            {"batch": "b1", "status": "no-changes",
             "changed": 0, "unchanged": 3, "total": 3},
        )

    def test_history_entry_marks_fresh_batch_not_undone(self) -> None:
        records = self.applied_records()
        entry = build_history_entry("b1", records)
        self.assertEqual(entry["batch"], "b1")
        self.assertIs(entry["records"], records)
        self.assertEqual(entry["changed_count"], 1)
        self.assertFalse(entry["undone"])
        self.assertIsNone(entry["undone_at"])

    def test_replay_result_matches_re_submission_stats(self) -> None:
        entry = build_history_entry("b1", self.applied_records())
        self.assertEqual(
            replay_result(entry),
            {"batch": "b1", "status": "already-applied",
             "changed": 1, "unchanged": 1, "total": 2},
        )


class HistoryRevisionIntegrityTest(unittest.TestCase):
    def valid_record(self, **overrides) -> dict:
        record = {
            "sha256": "d1", "old": "cat", "new": "dog",
            "changed": True, "rev": 1,
        }
        record.update(overrides)
        return record

    def test_genuine_non_negative_integer_revisions_pass(self) -> None:
        for revision in (0, 1, 42):
            value, problem = history_record_revision(self.valid_record(rev=revision))
            self.assertEqual(value, revision)
            self.assertIsNone(problem)

    def test_zero_revision_is_valid_even_for_unchanged_records(self) -> None:
        value, problem = history_record_revision(
            self.valid_record(changed=False, rev=0)
        )
        self.assertEqual(value, 0)
        self.assertIsNone(problem)

    def test_missing_revision_is_corruption_not_zero(self) -> None:
        record = self.valid_record()
        del record["rev"]
        value, problem = history_record_revision(record)
        self.assertIsNone(value)
        self.assertIn("missing", problem)

    def test_boolean_decimal_string_and_null_revisions_rejected(self) -> None:
        for bad in (True, False, 1.0, 0.0, "1", "0", None, -1, -0.0, "x", [], {}):
            with self.subTest(bad=bad):
                value, problem = history_record_revision(self.valid_record(rev=bad))
                self.assertIsNone(value)
                self.assertIsNotNone(problem)

    def test_boolean_true_never_compares_as_revision_one(self) -> None:
        record = self.valid_record(rev=True)
        value, problem = history_record_revision(record)
        self.assertIsNone(value)
        # The report makes the offending value identifiable.
        self.assertIn("True", problem)

    def test_validate_undo_history_reports_batch_sample_and_problem(self) -> None:
        records = [
            self.valid_record(sha256="d1", rev=1),
            self.valid_record(
                sha256="d2", old="cat", new="cat", changed=False, rev=0
            ),
        ]
        validate_undo_history("b1", records)  # intact history passes

        del records[1]["rev"]
        with self.assertRaises(BatchError) as ctx:
            validate_undo_history("b1", records)
        message = str(ctx.exception)
        self.assertIn("corrupted", message)
        self.assertIn("b1", message)
        self.assertIn("d2", message)
        self.assertIn("rev", message)

    def test_validate_covers_unchanged_records_too(self) -> None:
        # An unmodified record still has to pin its revision; its integer
        # zero is valid, but a missing value on one is still corruption.
        records = [
            self.valid_record(
                sha256="d1", old="cat", new="cat", changed=False
            ),
        ]
        del records[0]["rev"]
        with self.assertRaises(BatchError) as ctx:
            validate_undo_history("b1", records)
        self.assertIn("d1", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
