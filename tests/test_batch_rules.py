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
    describe_history_change_problem,
    describe_history_revision_problem,
    describe_missing_history_labels,
    history_labels_equal,
    no_changes_result,
    normalize_label,
    reject_unknown_samples,
    replay_result,
    require_intact_history_changes,
    require_intact_history_labels,
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

    def test_empty_string_saved_in_history_matches_null_resubmission(self) -> None:
        # History loaded from disk may have legally saved unlabeled as the
        # empty string (validate_history accepts both spellings); those rows
        # are NOT passed through normalize_label before comparison. The
        # repeat submission parses "" back to None, in another order, and
        # mixes a setting change with an unlabeled-to-unlabeled no-op.
        entry = self.entry(
            records=[
                {"sha256": "d1", "old": "", "new": "cat", "changed": True, "rev": 1},
                {"sha256": "d2", "old": "", "new": "", "changed": False, "rev": 0},
            ],
            changed_count=1,
        )
        result = resolve_re_submission(
            "b1",
            [record("d2", None, None), record("d1", None, "cat")],
            entry,
        )
        self.assertEqual(
            result,
            {"batch": "b1", "status": "already-applied",
             "changed": 1, "unchanged": 1, "total": 2},
        )

    def test_literal_unlabeled_saved_in_history_is_distinct_state(self) -> None:
        # Only null and "" collapse to unlabeled; a real class named
        # "unlabeled" must never compare equal to an unlabeled record.
        entry = self.entry(
            records=[
                {"sha256": "d1", "old": "cat", "new": "",
                 "changed": True, "rev": 1}
            ],
            changed_count=1,
        )
        with self.assertRaises(BatchError) as ctx:
            resolve_re_submission(
                "b1", [record("d1", "cat", "unlabeled")], entry
            )
        self.assertIn("already used with different content", str(ctx.exception))

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


class ReSubmissionHistoryDamageTest(unittest.TestCase):
    """Re-submitting a number whose saved history lacks old/new labels."""

    def entry(self, records: list[dict], *, undone: bool = False) -> dict:
        return {
            "batch": "b1",
            "records": records,
            "changed_count": sum(1 for r in records if r.get("changed")),
            "undone": undone,
            "undone_at": None,
        }

    def _assert_corruption(self, error: Exception, digest: str, position: int,
                           problem: str) -> None:
        message = str(error)
        self.assertIn("Batch history is corrupted", message)
        self.assertIn("'b1'", message)
        self.assertIn(digest, message)
        self.assertIn(f"record #{position}", message)
        self.assertIn(problem, message)
        self.assertNotIn("different content", message)

    def test_missing_old_refused_before_replay(self) -> None:
        # The submission content matches what the batch did; the missing
        # field still refuses instead of replaying the original result.
        entry = self.entry(
            [{"sha256": "d1", "new": "dog", "changed": True, "rev": 1}]
        )
        with self.assertRaises(BatchError) as ctx:
            resolve_re_submission("b1", [record("d1", "cat", "dog")], entry)
        self._assert_corruption(ctx.exception, "d1", 1, "missing 'old' label")

    def test_missing_new_refused_before_replay(self) -> None:
        entry = self.entry(
            [{"sha256": "d1", "old": "cat", "changed": True, "rev": 1}]
        )
        with self.assertRaises(BatchError) as ctx:
            resolve_re_submission("b1", [record("d1", "cat", "dog")], entry)
        self._assert_corruption(ctx.exception, "d1", 1, "missing 'new' label")

    def test_both_missing_is_named_as_both(self) -> None:
        entry = self.entry(
            [{"sha256": "d1", "changed": True, "rev": 1}]
        )
        with self.assertRaises(BatchError) as ctx:
            resolve_re_submission("b1", [record("d1", "cat", "dog")], entry)
        self._assert_corruption(
            ctx.exception, "d1", 1, "missing 'old' and 'new' labels"
        )

    def test_damage_on_an_unchanged_record_is_refused_too(self) -> None:
        # Even a row the batch never changed must carry both labels.
        entry = self.entry(
            [
                {"sha256": "d1", "old": "cat", "new": "dog",
                 "changed": True, "rev": 1},
                {"sha256": "d2", "old": "dog",
                 "changed": False, "rev": 0},
            ]
        )
        with self.assertRaises(BatchError) as ctx:
            resolve_re_submission(
                "b1",
                [record("d1", "cat", "dog"), record("d2", "dog", "dog")],
                entry,
            )
        self._assert_corruption(ctx.exception, "d2", 2, "missing 'new' label")

    def test_a_normal_record_at_the_front_never_masks_a_later_gap(self) -> None:
        entry = self.entry(
            [
                {"sha256": "d1", "old": "cat", "new": "dog",
                 "changed": True, "rev": 1},
                {"sha256": "d2", "old": "dog", "new": "puppy",
                 "changed": True, "rev": 1},
                {"sha256": "d3", "new": "guppy",
                 "changed": True, "rev": 1},
            ]
        )
        with self.assertRaises(BatchError) as ctx:
            resolve_re_submission(
                "b1",
                [
                    record("d1", "cat", "dog"),
                    record("d2", "dog", "puppy"),
                    record("d3", "fish", "guppy"),
                ],
                entry,
            )
        self._assert_corruption(ctx.exception, "d3", 3, "missing 'old' label")

    def test_damage_is_reported_even_when_content_would_conflict(self) -> None:
        # Different content would ordinarily be a number conflict; the
        # integrity refusal must come first.
        entry = self.entry(
            [{"sha256": "d1", "new": "dog", "changed": True, "rev": 1}]
        )
        with self.assertRaises(BatchError) as ctx:
            resolve_re_submission("b1", [record("d1", "cat", "fish")], entry)
        message = str(ctx.exception)
        self.assertIn("Batch history is corrupted", message)
        self.assertIn("d1", message)
        self.assertNotIn("different content", message)

    def test_damage_is_reported_even_when_the_batch_was_undone(self) -> None:
        entry = self.entry(
            [{"sha256": "d1", "old": "cat", "changed": True, "rev": 1}],
            undone=True,
        )
        with self.assertRaises(BatchError) as ctx:
            resolve_re_submission("b1", [record("d1", "cat", "dog")], entry)
        self._assert_corruption(ctx.exception, "d1", 1, "missing 'new' label")

    def test_explicit_null_and_empty_string_still_replay(self) -> None:
        # Explicitly saved null/"" are recorded unlabeled labels, not
        # missing fields, so same-content submission keeps replaying.
        entry = self.entry(
            [
                {"sha256": "d1", "old": None, "new": "cat",
                 "changed": True, "rev": 1},
                {"sha256": "d2", "old": "", "new": "",
                 "changed": False, "rev": 0},
            ]
        )
        result = resolve_re_submission(
            "b1",
            [record("d2", None, None), record("d1", "", "cat")],
            entry,
        )
        self.assertEqual(result["status"], "already-applied")
        self.assertEqual(result["changed"], 1)
        self.assertEqual(result["unchanged"], 1)

    def test_literal_unlabeled_class_keeps_replay_rules(self) -> None:
        entry = self.entry(
            [{"sha256": "d1", "old": "unlabeled", "new": "cat",
              "changed": True, "rev": 1}]
        )
        self.assertEqual(
            resolve_re_submission(
                "b1", [record("d1", "unlabeled", "cat")], entry
            )["status"],
            "already-applied",
        )
        with self.assertRaises(BatchError) as ctx:
            resolve_re_submission("b1", [record("d1", None, "cat")], entry)
        self.assertIn("different content", str(ctx.exception))


class ReSubmissionDuplicateRecordTest(unittest.TestCase):
    """Re-submitting a number whose saved history repeats one digest."""

    def entry(self, records: list[dict], *, undone: bool = False) -> dict:
        return {
            "batch": "b1",
            "records": records,
            "changed_count": sum(1 for r in records if r.get("changed")),
            "undone": undone,
            "undone_at": None,
        }

    def row(self, digest: str = "d1", old="cat", new="dog", *,
            changed: bool = True) -> dict:
        return {"sha256": digest, "old": old, "new": new,
                "changed": changed, "rev": 1 if changed else 0}

    def _assert_duplicate_corruption(self, error: Exception, digest: str,
                                     first: int, second: int) -> None:
        message = str(error)
        self.assertIn("Batch history is corrupted", message)
        self.assertIn("'b1'", message)
        self.assertIn(digest, message)
        self.assertIn(f"record #{first}", message)
        self.assertIn(f"record #{second}", message)
        self.assertNotIn("already used with different content", message)
        self.assertNotIn("already-applied", message)

    def test_verbatim_copy_refused_instead_of_replaying(self) -> None:
        # Same content would otherwise replay; the frozenset comparison
        # collapses the copy and the replay would count the change twice.
        entry = self.entry([self.row(), self.row()])
        with self.assertRaises(BatchError) as ctx:
            resolve_re_submission("b1", [record("d1", "cat", "dog")], entry)
        self._assert_duplicate_corruption(ctx.exception, "d1", 1, 2)

    def test_two_records_with_different_labels_are_still_a_repeat(self) -> None:
        entry = self.entry(
            [self.row(new="dog"), self.row(new="fish")]
        )
        with self.assertRaises(BatchError) as ctx:
            resolve_re_submission(
                "b1", [record("d1", "cat", "fish")], entry
            )
        self._assert_duplicate_corruption(ctx.exception, "d1", 1, 2)

    def test_copy_of_an_unchanged_record_is_corruption(self) -> None:
        entry = self.entry(
            [
                self.row("d2", old="dog", new="dog", changed=False),
                self.row("d2", old="dog", new="dog", changed=False),
            ]
        )
        with self.assertRaises(BatchError) as ctx:
            resolve_re_submission("b1", [record("d2", "dog", "dog")], entry)
        self._assert_duplicate_corruption(ctx.exception, "d2", 1, 2)

    def test_other_samples_between_the_two_occurrences_do_not_hide_it(self) -> None:
        entry = self.entry(
            [
                self.row("d1"),
                self.row("d2"),
                self.row("d3"),
                self.row("d1"),
            ]
        )
        with self.assertRaises(BatchError) as ctx:
            resolve_re_submission(
                "b1",
                [
                    record("d1", "cat", "dog"),
                    record("d2", "cat", "dog"),
                    record("d3", "cat", "dog"),
                ],
                entry,
            )
        self._assert_duplicate_corruption(ctx.exception, "d1", 1, 4)

    def test_only_the_first_two_positions_are_named(self) -> None:
        entry = self.entry([self.row(), self.row(), self.row()])
        with self.assertRaises(BatchError) as ctx:
            resolve_re_submission("b1", [record("d1", "cat", "dog")], entry)
        message = str(ctx.exception)
        self.assertIn("record #1", message)
        self.assertIn("record #2", message)
        self.assertNotIn("record #3", message)

    def test_repeat_refused_even_when_content_would_conflict(self) -> None:
        # Different content would ordinarily be a number conflict; the
        # duplicate-digest refusal must come first.
        entry = self.entry([self.row(), self.row()])
        with self.assertRaises(BatchError) as ctx:
            resolve_re_submission("b1", [record("d1", "cat", "fish")], entry)
        message = str(ctx.exception)
        self.assertIn("Batch history is corrupted", message)
        self.assertNotIn("different content", message)

    def test_repeat_refused_even_when_the_batch_was_undone(self) -> None:
        entry = self.entry([self.row(), self.row()], undone=True)
        with self.assertRaises(BatchError) as ctx:
            resolve_re_submission("b1", [record("d1", "cat", "dog")], entry)
        self._assert_duplicate_corruption(ctx.exception, "d1", 1, 2)

    def test_distinct_samples_still_replay(self) -> None:
        entry = self.entry(
            [self.row("d1"), self.row("d2", old="dog", new="puppy")]
        )
        result = resolve_re_submission(
            "b1",
            [record("d2", "dog", "puppy"), record("d1", "cat", "dog")],
            entry,
        )
        self.assertEqual(result["status"], "already-applied")
        self.assertEqual(result["changed"], 2)


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


class HistoryChangeFlagValidationTest(unittest.TestCase):
    def test_null_and_empty_string_are_the_same_unlabeled_label(self) -> None:
        self.assertTrue(history_labels_equal(None, None))
        self.assertTrue(history_labels_equal(None, ""))
        self.assertTrue(history_labels_equal("", None))
        self.assertTrue(history_labels_equal("", ""))

    def test_everything_else_compares_as_the_exact_string(self) -> None:
        self.assertTrue(history_labels_equal("cat", "cat"))
        self.assertFalse(history_labels_equal("cat", "Cat"))
        self.assertFalse(history_labels_equal("cat", "cat "))
        self.assertFalse(history_labels_equal(" cat", "cat"))
        self.assertFalse(history_labels_equal("a/b", "a\\b"))
        self.assertTrue(history_labels_equal("猫", "猫"))
        self.assertFalse(history_labels_equal("猫", "狗"))
        # A literal class named "unlabeled" is not the unlabeled state.
        self.assertFalse(history_labels_equal("unlabeled", None))
        self.assertFalse(history_labels_equal("unlabeled", ""))

    def row(self, *, old, new, changed) -> dict:
        return {"sha256": "d1", "old": old, "new": new, "changed": changed}

    def test_consistent_rows_have_no_problem(self) -> None:
        self.assertIsNone(
            describe_history_change_problem(self.row(old="cat", new="dog", changed=True))
        )
        self.assertIsNone(
            describe_history_change_problem(self.row(old="cat", new="cat", changed=False))
        )
        # null and "" swapped is not a change, either way around.
        self.assertIsNone(
            describe_history_change_problem(self.row(old=None, new="", changed=False))
        )
        self.assertIsNone(
            describe_history_change_problem(self.row(old="", new=None, changed=False))
        )

    def test_hidden_real_change_is_reported_with_both_labels(self) -> None:
        # cat -> dog but flagged unchanged: undo would leave "dog" behind.
        problem = describe_history_change_problem(
            self.row(old="cat", new="dog", changed=False)
        )
        self.assertIsNotNone(problem)
        self.assertIn("'changed' is false", problem)
        self.assertIn("'cat'", problem)
        self.assertIn("'dog'", problem)

    def test_false_change_on_equal_labels_is_reported(self) -> None:
        problem = describe_history_change_problem(
            self.row(old="cat", new="cat", changed=True)
        )
        self.assertIsNotNone(problem)
        self.assertIn("'changed' is true", problem)
        self.assertIn("'cat'", problem)

    def test_unchanged_unlabeled_spelling_mismatch_is_not_a_change(self) -> None:
        # null/"" are the same state, so changed=true would overstate it.
        problem = describe_history_change_problem(
            self.row(old=None, new="", changed=True)
        )
        self.assertIsNotNone(problem)
        self.assertIn("identical", problem)

    def test_literal_unlabeled_versus_null_is_a_real_change(self) -> None:
        problem = describe_history_change_problem(
            self.row(old="unlabeled", new=None, changed=False)
        )
        self.assertIsNotNone(problem)
        self.assertIn("differ", problem)
        self.assertIsNone(
            describe_history_change_problem(
                self.row(old="unlabeled", new=None, changed=True)
            )
        )

    def test_case_whitespace_separator_differences_are_real_changes(self) -> None:
        for old, new in (("cat", "Cat"), ("cat", "cat "), ("a/b", "a\\b")):
            with self.subTest(old=old, new=new):
                self.assertIsNotNone(
                    describe_history_change_problem(
                        self.row(old=old, new=new, changed=False)
                    )
                )

    def entry(self, records: list[dict], *, number: str = "b1") -> dict:
        return {"batch": number, "records": records}

    def test_require_passes_for_consistent_batch(self) -> None:
        require_intact_history_changes(
            self.entry(
                [
                    {"sha256": "d1", "old": "cat", "new": "dog",
                     "changed": True, "rev": 1},
                    {"sha256": "d2", "old": None, "new": "",
                     "changed": False, "rev": 0},
                ]
            )
        )

    def test_require_empty_record_list_passes(self) -> None:
        require_intact_history_changes(self.entry([]))

    def test_require_names_batch_digest_position_and_reason(self) -> None:
        with self.assertRaises(BatchError) as ctx:
            require_intact_history_changes(
                self.entry(
                    [
                        {"sha256": "d1", "old": "cat", "new": "dog",
                         "changed": True, "rev": 1},
                        {"sha256": "d2", "old": "cat", "new": "dog",
                         "changed": False, "rev": 1},
                    ],
                    number="renumber-7",
                )
            )
        message = str(ctx.exception)
        self.assertIn("Batch history is corrupted", message)
        self.assertIn("'renumber-7'", message)
        self.assertIn("d2", message)
        self.assertIn("record #2", message)
        self.assertIn("'changed' is false", message)

    def test_require_false_change_names_its_position(self) -> None:
        with self.assertRaises(BatchError) as ctx:
            require_intact_history_changes(
                self.entry(
                    [
                        {"sha256": "d1", "old": "cat", "new": "cat",
                         "changed": True, "rev": 0},
                    ]
                )
            )
        message = str(ctx.exception)
        self.assertIn("record #1", message)
        self.assertIn("'changed' is true", message)

    def test_front_rows_never_mask_a_later_contradiction(self) -> None:
        with self.assertRaises(BatchError) as ctx:
            require_intact_history_changes(
                self.entry(
                    [
                        {"sha256": "d1", "old": "cat", "new": "dog",
                         "changed": True, "rev": 1},
                        {"sha256": "d2", "old": "dog", "new": "fish",
                         "changed": True, "rev": 1},
                        {"sha256": "d3", "old": "fish", "new": "guppy",
                         "changed": False, "rev": 1},
                    ]
                )
            )
        message = str(ctx.exception)
        self.assertIn("d3", message)
        self.assertIn("record #3", message)


class HistoryLabelPresenceValidationTest(unittest.TestCase):
    def test_both_labels_present_is_no_problem(self) -> None:
        self.assertIsNone(
            describe_missing_history_labels({"sha256": "d1", "old": "cat", "new": "dog"})
        )

    def test_null_and_empty_string_are_saved_labels_not_missing(self) -> None:
        # Explicit null and "" both mean unlabeled; neither is a gap.
        self.assertIsNone(
            describe_missing_history_labels({"sha256": "d1", "old": None, "new": ""})
        )
        self.assertIsNone(
            describe_missing_history_labels({"sha256": "d1", "old": "", "new": None})
        )
        # A literal class named "unlabeled" is an ordinary saved label.
        self.assertIsNone(
            describe_missing_history_labels(
                {"sha256": "d1", "old": "unlabeled", "new": None}
            )
        )

    def test_each_missing_label_is_named(self) -> None:
        self.assertEqual(
            describe_missing_history_labels({"sha256": "d1", "new": "dog"}),
            "missing 'old' label",
        )
        self.assertEqual(
            describe_missing_history_labels({"sha256": "d1", "old": "cat"}),
            "missing 'new' label",
        )

    def test_both_missing_is_not_an_unlabeled_noop(self) -> None:
        # Two absent keys are two gaps, not an unlabeled->unlabeled record.
        self.assertEqual(
            describe_missing_history_labels({"sha256": "d1"}),
            "missing 'old' and 'new' labels",
        )

    def entry(self, records: list[dict], *, number: str = "b1") -> dict:
        return {"batch": number, "records": records}

    def test_require_passes_for_complete_records(self) -> None:
        require_intact_history_labels(
            self.entry(
                [
                    {"sha256": "d1", "old": "cat", "new": "dog",
                     "changed": True, "rev": 1},
                    {"sha256": "d2", "old": None, "new": "",
                     "changed": False, "rev": 0},
                ]
            )
        )

    def test_require_empty_record_list_passes(self) -> None:
        require_intact_history_labels(self.entry([]))

    def test_require_names_batch_digest_position_and_missing_label(self) -> None:
        with self.assertRaises(BatchError) as ctx:
            require_intact_history_labels(
                self.entry(
                    [
                        {"sha256": "d1", "old": "cat", "new": "dog",
                         "changed": True, "rev": 1},
                        {"sha256": "d2", "new": "puppy",
                         "changed": True, "rev": 1},
                    ],
                    number="renumber-3",
                )
            )
        message = str(ctx.exception)
        self.assertIn("Batch history is corrupted", message)
        self.assertIn("'renumber-3'", message)
        self.assertIn("d2", message)
        self.assertIn("record #2", message)
        self.assertIn("missing 'old' label", message)

    def test_require_names_both_missing_labels(self) -> None:
        with self.assertRaises(BatchError) as ctx:
            require_intact_history_labels(
                self.entry([{"sha256": "d1", "changed": False, "rev": 0}])
            )
        message = str(ctx.exception)
        self.assertIn("record #1", message)
        self.assertIn("missing 'old' and 'new' labels", message)

    def test_unchanged_record_is_checked_too(self) -> None:
        # A no-op row missing 'new' is damage even though undo would
        # never restore it.
        with self.assertRaises(BatchError) as ctx:
            require_intact_history_labels(
                self.entry([{"sha256": "d1", "old": "cat",
                             "changed": False, "rev": 0}])
            )
        self.assertIn("missing 'new' label", str(ctx.exception))

    def test_front_rows_never_mask_a_later_gap(self) -> None:
        with self.assertRaises(BatchError) as ctx:
            require_intact_history_labels(
                self.entry(
                    [
                        {"sha256": "d1", "old": "cat", "new": "dog",
                         "changed": True, "rev": 1},
                        {"sha256": "d2", "old": "dog", "new": "fish",
                         "changed": True, "rev": 1},
                        {"sha256": "d3", "old": "fish",
                         "changed": True, "rev": 1},
                    ]
                )
            )
        message = str(ctx.exception)
        self.assertIn("d3", message)
        self.assertIn("record #3", message)


if __name__ == "__main__":
    unittest.main()
