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
    describe_history_revision_problem,
    no_changes_result,
    normalize_label,
    reject_unknown_samples,
    replay_result,
    require_intact_history_changed_flags,
    require_intact_history_records,
    require_intact_history_revisions,
    resolve_re_submission,
    result_payload,
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


class HistoryRevisionValidationTest(unittest.TestCase):
    def test_zero_and_non_negative_integers_are_valid(self) -> None:
        # Zero stays legal even on an unchanged record.
        self.assertIsNone(describe_history_revision_problem(0, present=True))
        self.assertIsNone(describe_history_revision_problem(1, present=True))
        self.assertIsNone(describe_history_revision_problem(42, present=True))

    def test_missing_field_is_not_treated_as_zero(self) -> None:
        problem = describe_history_revision_problem(None, present=False)
        self.assertEqual(problem, "missing 'rev' field")

    def test_boolean_decimal_string_null_and_negative_are_invalid(self) -> None:
        for value in (True, False, 1.0, 0.0, "1", "0", None, -1):
            problem = describe_history_revision_problem(value, present=True)
            self.assertIsNotNone(problem, value)
            self.assertIn("non-negative integer", problem)

    def test_require_passes_when_every_record_has_a_sound_rev(self) -> None:
        entry = {
            "batch": "b1",
            "records": [
                {"sha256": "d1", "changed": True, "rev": 1},
                {"sha256": "d2", "changed": False, "rev": 0},
            ],
        }
        require_intact_history_revisions(entry)  # must not raise

    def test_require_reports_batch_digest_and_problem(self) -> None:
        entry = {
            "batch": "b1",
            "records": [
                {"sha256": "d1", "changed": True, "rev": 1},
                {"sha256": "d2", "changed": True},  # missing rev
            ],
        }
        with self.assertRaises(BatchError) as ctx:
            require_intact_history_revisions(entry)
        message = str(ctx.exception)
        self.assertIn("Batch history is corrupted", message)
        self.assertIn("'b1'", message)
        self.assertIn("d2", message)
        self.assertIn("missing 'rev'", message)

    def test_require_rejects_spoofing_types_on_unchanged_record_too(self) -> None:
        # Integrity applies to every record, not just changed ones: a
        # boolean or decimal on an unchanged row is still damage.
        for bad in (True, 1.0, "0", None):
            entry = {
                "batch": "b1",
                "records": [{"sha256": "d1", "changed": False, "rev": bad}],
            }
            with self.subTest(bad=bad):
                with self.assertRaises(BatchError) as ctx:
                    require_intact_history_revisions(entry)
                self.assertIn("d1", str(ctx.exception))


class HistoryRecordUniquenessValidationTest(unittest.TestCase):
    def entry(self, records: list[dict], *, number: str = "b1") -> dict:
        return {"batch": number, "records": records}

    def test_distinct_digests_pass(self) -> None:
        require_intact_history_records(
            self.entry(
                [
                    {"sha256": "d1", "changed": True, "rev": 1},
                    {"sha256": "d2", "changed": False, "rev": 0},
                ]
            )
        )

    def test_empty_record_list_passes(self) -> None:
        require_intact_history_records(self.entry([]))

    def test_identical_copied_row_is_corruption_not_a_merge(self) -> None:
        # A verbatim copy of a record that genuinely changed a label is
        # still a repeat: undo must not restore the sample twice.
        row = {"sha256": "d1", "old": "cat", "new": "dog",
               "changed": True, "rev": 1}
        with self.assertRaises(BatchError) as ctx:
            require_intact_history_records(self.entry([dict(row), dict(row)]))
        message = str(ctx.exception)
        self.assertIn("Batch history is corrupted", message)
        self.assertIn("'b1'", message)
        self.assertIn("d1", message)
        self.assertIn("record #1", message)
        self.assertIn("record #2", message)

    def test_duplicate_of_an_unchanged_record_is_corruption(self) -> None:
        # Repeating a no-op row ("record did not actually change a label")
        # is just as invalid as repeating a changed row.
        row = {"sha256": "d2", "old": "dog", "new": "dog",
               "changed": False, "rev": 0}
        with self.assertRaises(BatchError) as ctx:
            require_intact_history_records(self.entry([dict(row), dict(row)]))
        self.assertIn("d2", str(ctx.exception))

    def test_digest_alone_decides_even_when_labels_differ(self) -> None:
        # The repeat carries different label text; identity is by digest.
        with self.assertRaises(BatchError) as ctx:
            require_intact_history_records(
                self.entry(
                    [
                        {"sha256": "d1", "old": "cat", "new": "dog",
                         "changed": True, "rev": 1},
                        {"sha256": "d1", "old": "cat", "new": "fish",
                         "changed": True, "rev": 1},
                    ]
                )
            )
        self.assertIn("d1", str(ctx.exception))

    def test_a_repeat_behind_valid_rows_is_not_masked(self) -> None:
        # The first records are restorable; the duplicate only appears at
        # the back, and positions are counted from the whole list.
        with self.assertRaises(BatchError) as ctx:
            require_intact_history_records(
                self.entry(
                    [
                        {"sha256": "d1", "changed": True, "rev": 1},
                        {"sha256": "d2", "changed": True, "rev": 1},
                        {"sha256": "d3", "changed": True, "rev": 1},
                        {"sha256": "d1", "changed": True, "rev": 1},
                    ]
                )
            )
        message = str(ctx.exception)
        self.assertIn("record #1", message)
        self.assertIn("record #4", message)

    def test_only_first_two_positions_are_named(self) -> None:
        with self.assertRaises(BatchError) as ctx:
            require_intact_history_records(
                self.entry(
                    [
                        {"sha256": "d1", "changed": True, "rev": 1},
                        {"sha256": "d1", "changed": True, "rev": 1},
                        {"sha256": "d1", "changed": True, "rev": 1},
                    ]
                )
            )
        self.assertIn("record #1", str(ctx.exception))
        self.assertIn("record #2", str(ctx.exception))
        self.assertNotIn("record #3", str(ctx.exception))

    def test_batch_number_is_named(self) -> None:
        with self.assertRaises(BatchError) as ctx:
            require_intact_history_records(
                self.entry(
                    [
                        {"sha256": "d1", "changed": True, "rev": 1},
                        {"sha256": "d1", "changed": True, "rev": 1},
                    ],
                    number="renumber-9",
                )
            )
        self.assertIn("'renumber-9'", str(ctx.exception))


class HistoryChangedFlagValidationTest(unittest.TestCase):
    def entry(self, records: list[dict], *, number: str = "b1") -> dict:
        return {"batch": number, "records": records}

    def row(self, digest: str, old, new, changed: bool, *, rev: int = 0) -> dict:
        return {
            "sha256": digest,
            "old": normalize_label(old),
            "new": normalize_label(new),
            "changed": changed,
            "rev": rev,
        }

    def test_consistent_flags_pass(self) -> None:
        require_intact_history_changed_flags(
            self.entry(
                [
                    self.row("d1", "cat", "dog", True, rev=1),
                    self.row("d2", "cat", "cat", False),
                    self.row("d3", None, "cat", True, rev=1),
                    self.row("d4", "cat", None, True, rev=1),
                    self.row("d5", None, None, False),
                    self.row("d6", "", None, False),
                    self.row("d7", None, "", False),
                ]
            )
        )

    def test_empty_record_list_passes(self) -> None:
        require_intact_history_changed_flags(self.entry([]))

    def test_false_flag_with_different_labels_is_corruption(self) -> None:
        # cat -> dog but marked unchanged: undo would leave dog behind.
        with self.assertRaises(BatchError) as ctx:
            require_intact_history_changed_flags(
                self.entry([self.row("d1", "cat", "dog", False, rev=1)])
            )
        message = str(ctx.exception)
        self.assertIn("Batch history is corrupted", message)
        self.assertIn("'b1'", message)
        self.assertIn("d1", message)
        self.assertIn("record #1", message)
        self.assertIn("'changed' is false", message)

    def test_true_flag_with_equal_labels_is_corruption(self) -> None:
        # cat -> cat but marked changed: undo would count a never-modified
        # sample among the restored samples.
        with self.assertRaises(BatchError) as ctx:
            require_intact_history_changed_flags(
                self.entry([self.row("d1", "cat", "cat", True)])
            )
        message = str(ctx.exception)
        self.assertIn("Batch history is corrupted", message)
        self.assertIn("d1", message)
        self.assertIn("record #1", message)
        self.assertIn("'changed' is true", message)

    def test_null_and_empty_string_interchange_is_not_a_change(self) -> None:
        # The two unlabeled spellings differ in no way: either flag
        # direction over a null/"" pair must agree that nothing changed.
        for old, new in ((None, ""), ("", None), ("", ""), (None, None)):
            with self.subTest(old=old, new=new):
                require_intact_history_changed_flags(
                    self.entry([self.row("d1", old, new, False)])
                )
                with self.assertRaises(BatchError):
                    require_intact_history_changed_flags(
                        self.entry([self.row("d1", old, new, True)])
                    )

    def test_literal_empty_string_on_disk_compares_as_unlabeled(self) -> None:
        # History normally stores null, but a file may literally carry "":
        # the raw stored pair None/"" still means no change, and a genuine
        # "" -> "cat" transition still does.
        require_intact_history_changed_flags(
            self.entry(
                [{"sha256": "d1", "old": "", "new": None,
                  "changed": False, "rev": 0}]
            )
        )
        require_intact_history_changed_flags(
            self.entry(
                [{"sha256": "d1", "old": "", "new": "cat",
                  "changed": True, "rev": 1}]
            )
        )
        with self.assertRaises(BatchError):
            require_intact_history_changed_flags(
                self.entry(
                    [{"sha256": "d1", "old": "", "new": "cat",
                      "changed": False, "rev": 1}]
                )
            )

    def test_comparison_keeps_case_whitespace_chinese_and_separators(self) -> None:
        # Each pair is genuinely different under literal comparison, so a
        # false flag is corruption; and an equal literal string marked
        # changed would be too.
        cases = [
            ("cat", "Cat"),
            ("cat", "cat "),
            (" cat", "cat"),
            ("猫", "狗"),
            ("a/b", "a\\b"),
            ("unlabeled", None),
            ("unlabeled", ""),
            ("", "unlabeled"),
        ]
        for old, new in cases:
            with self.subTest(old=old, new=new):
                with self.assertRaises(BatchError):
                    require_intact_history_changed_flags(
                        self.entry([self.row("d1", old, new, False)])
                    )
                require_intact_history_changed_flags(
                    self.entry([self.row("d1", old, new, True, rev=1)])
                )
                # The literal class "unlabeled" equal to itself is no
                # change, like every other identical string.
                require_intact_history_changed_flags(
                    self.entry([self.row("d1", old, old, False)])
                )

    def test_decision_uses_stored_labels_not_current_state(self) -> None:
        # The check takes the history entry alone; nothing about a sample's
        # current label enters the verdict.
        with self.assertRaises(BatchError):
            require_intact_history_changed_flags(
                self.entry([self.row("d1", "cat", "dog", False, rev=1)])
            )

    def test_contradiction_behind_restorable_rows_is_not_masked(self) -> None:
        with self.assertRaises(BatchError) as ctx:
            require_intact_history_changed_flags(
                self.entry(
                    [
                        self.row("d1", "cat", "kitten", True, rev=1),
                        self.row("d2", "dog", "puppy", True, rev=1),
                        self.row("d3", "fish", "fish", True),
                    ]
                )
            )
        message = str(ctx.exception)
        self.assertIn("d3", message)
        self.assertIn("record #3", message)
        self.assertNotIn("d1", message)

    def test_batch_number_is_named(self) -> None:
        with self.assertRaises(BatchError) as ctx:
            require_intact_history_changed_flags(
                self.entry(
                    [self.row("d1", "cat", "dog", False, rev=1)],
                    number="renumber-9",
                )
            )
        self.assertIn("'renumber-9'", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
