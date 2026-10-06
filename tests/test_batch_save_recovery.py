"""Regression tests for completing a batch label-update interrupted mid-save.

A batch commits through a write-ahead journal (see
``DatasetStore._commit``): the prepared journal is written first, the
manifest is installed next (samples show their new categories) and the
batch history is installed only after that, before the journal is
removed.  A process that stops in the gap between the manifest install
and the history install therefore leaves a workspace where the samples
already carry the requested labels, ``batches.json`` still lacks the
batch, and the prepared journal is on disk.  The existing recovery
coverage started from a state where neither file had moved; these tests
pin the harder case where the labels already did.

Reopening the workspace (or merely running the next query on a store
that already had it open) must finish exactly that batch:

* every named sample shows the label requested in the submission,
  including records whose target label already equalled the current
  one — those stay in the batch record but do not count as changed and
  do not gain a label revision;
* the history gains the one batch, exactly once, behind every earlier
  batch in the original order, with each record's before/after labels,
  changed flag, pinned revision and the batch changed count matching
  what the submission really did;
* changed samples keep exactly the revision of this one modification —
  recovery never applies the same batch twice;
* content summaries, registered sources and byte sizes are untouched,
  samples outside the request are untouched, a previously saved split
  plan keeps its creation-time labels and members, and earlier undo
  markers keep their saved value.

Recovery itself can meet another storage read/write failure.  Such a
use must fail openly — no query may return a success-shaped answer
whose labels and history disagree — and once storage heals, reopening
must still complete the original batch once, with no dropped or
duplicated history row and no second revision bump.
"""

from __future__ import annotations

import contextlib
import errno
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from vision_workbench.batches import BatchError
from vision_workbench.store import DatasetStore


class _StoppedAfterManifest(Exception):
    """Stands in for the process dying right after the labels were saved."""


class InterruptedBatchHarness(unittest.TestCase):
    """Build a workspace stopped after the manifest install of one batch.

    The seeded workspace holds five samples and two earlier batches —
    one plain, one already undone (so its undo marker must survive
    recovery) — plus a split plan created before the interruption.
    """

    INTERRUPTED_NUMBER = "b-mid"

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

    def _raw_manifest(self) -> dict:
        return json.loads(self.store.manifest_path.read_text(encoding="utf-8"))

    def _raw_history(self) -> dict:
        return json.loads(self.store.batches_path.read_text(encoding="utf-8"))

    def build_scenario(self) -> dict:
        # d1 changes away from its label, d2 changes on top of an earlier
        # revision, d3 is a target-equals-current no-op, d4 only appears in
        # earlier batches and must stay untouched, d5 moves off unlabeled.
        d1 = self.add_sample("cat")
        d2 = self.add_sample("wolf")
        d3 = self.add_sample("fish")
        d4 = self.add_sample("bird")
        d5 = self.add_sample(None)

        self.submit("b-prev", (d2, "wolf", "dog"))
        self.submit("b-undone", (d4, "bird", "hawk"))
        undone = self.store.undo_batch("b-undone")
        self.assertEqual(undone["status"], "undone")
        undone_at = self.store.history()[1]["undone_at"]

        plan = self.store.create_split("plan-a", 0, [1, 0, 0]).plan

        before_items = {
            item["sha256"]: dict(item) for item in self._raw_manifest()["items"]
        }
        total_bytes = sum(item["size"] for item in before_items.values())
        history_before = self._raw_history()
        self.assertEqual(
            [entry["batch"] for entry in history_before["batches"]],
            ["b-prev", "b-undone"],
        )

        payload = {
            "batch": self.INTERRUPTED_NUMBER,
            "changes": [
                {"sha256": d1, "old": "cat", "new": "kitten"},
                {"sha256": d2, "old": "dog", "new": "puppy"},
                {"sha256": d3, "old": "fish", "new": "fish"},
                {"sha256": d5, "old": None, "new": "cat"},
            ],
        }

        # Run the real submission, but stop inside _commit immediately after
        # the manifest install: journal prepared, labels saved, history not.
        real_write = self.store._write_json_atomic

        def stop_after_manifest(path: Path, payload: dict) -> None:
            real_write(path, payload)
            if path == self.store.manifest_path:
                raise _StoppedAfterManifest

        with mock.patch.object(self.store, "_write_json_atomic", stop_after_manifest):
            with self.assertRaises(_StoppedAfterManifest):
                self.store.submit_batch(payload)

        # The straddled state is really on disk: new labels, old history,
        # prepared journal waiting.
        self.assertTrue(self.store.transaction_path.exists())
        interrupted_items = {
            item["sha256"]: item for item in self._raw_manifest()["items"]
        }
        self.assertEqual(interrupted_items[d1]["label"], "kitten")
        self.assertEqual(interrupted_items[d2]["label"], "puppy")
        self.assertEqual(interrupted_items[d3]["label"], "fish")
        self.assertEqual(interrupted_items[d4]["label"], "bird")
        self.assertEqual(interrupted_items[d5]["label"], "cat")
        self.assertEqual(
            [entry["batch"] for entry in self._raw_history()["batches"]],
            ["b-prev", "b-undone"],
        )

        journal = json.loads(
            self.store.transaction_path.read_text(encoding="utf-8")
        )
        expected_entry = journal["batches"]["batches"][-1]
        self.assertEqual(expected_entry["batch"], self.INTERRUPTED_NUMBER)

        return {
            "digests": {"d1": d1, "d2": d2, "d3": d3, "d4": d4, "d5": d5},
            "payload": payload,
            "expected_entry": expected_entry,
            "history_before": history_before,
            "before_items": before_items,
            "total_bytes": total_bytes,
            "plan": plan,
            "undone_at": undone_at,
        }

    def assert_mid_batch_completed(self, store: DatasetStore, sc: dict) -> None:
        d = sc["digests"]

        # All requested samples show the post-request category through the
        # public label query; the unrelated d4 keeps its label.
        expected_labels = {
            d["d1"]: "kitten",
            d["d2"]: "puppy",
            d["d3"]: "fish",
            d["d4"]: "bird",
            d["d5"]: "cat",
        }
        for digest, label in expected_labels.items():
            self.assertEqual(store.lookup_label(digest)["label"], label)

        # Exact category memberships via the public listing query, including
        # that moving d5 off unlabeled leaves no unlabeled sample behind.
        self.assertEqual(
            [s["sha256"] for s in store.find_by_label("kitten")["samples"]],
            [d["d1"]],
        )
        self.assertEqual(
            [s["sha256"] for s in store.find_by_label("puppy")["samples"]],
            [d["d2"]],
        )
        self.assertEqual(
            [s["sha256"] for s in store.find_by_label("fish")["samples"]],
            [d["d3"]],
        )
        self.assertEqual(
            [s["sha256"] for s in store.find_by_label("bird")["samples"]],
            [d["d4"]],
        )
        self.assertEqual(
            [s["sha256"] for s in store.find_by_label("cat")["samples"]],
            [d["d5"]],
        )
        cat_sample = store.find_by_label("cat")["samples"][0]
        self.assertEqual(cat_sample["label"], "cat")
        self.assertEqual(cat_sample["source"], sc["before_items"][d["d5"]]["source"])
        self.assertEqual(cat_sample["size"], sc["before_items"][d["d5"]]["size"])
        self.assertEqual(store.find_by_label("")["samples"], [])

        # Revisions: changed samples carry exactly this batch's one bump
        # (d2 builds on b-prev), the no-op d3 gained no revision at all and
        # the unrelated d4 keeps the revision its own undo left it with.
        items = {item["sha256"]: item for item in json.loads(
            store.manifest_path.read_text(encoding="utf-8")
        )["items"]}
        self.assertEqual(items[d["d1"]].get("rev"), 1)
        self.assertEqual(items[d["d2"]].get("rev"), 2)
        self.assertNotIn("rev", items[d["d3"]])
        self.assertEqual(items[d["d4"]].get("rev"), 2)
        self.assertEqual(items[d["d5"]].get("rev"), 1)

        # Content identity metadata never moves on a label update.
        for digest, before in sc["before_items"].items():
            self.assertEqual(items[digest]["sha256"], before["sha256"])
            self.assertEqual(items[digest]["source"], before["source"])
            self.assertEqual(items[digest]["size"], before["size"])

        # Exactly one new batch, behind both earlier ones in their order.
        entries = store.history()
        numbers = [entry["batch"] for entry in entries]
        self.assertEqual(numbers, ["b-prev", "b-undone", self.INTERRUPTED_NUMBER])
        self.assertEqual(numbers.count(self.INTERRUPTED_NUMBER), 1)
        self.assertEqual(entries[:2], sc["history_before"]["batches"])
        self.assertEqual(entries[-1], sc["expected_entry"])

        # The recovered record describes the real submission: before/after
        # labels, the changed flag and the batch's changed count.
        records = {
            record["sha256"]: record for record in entries[-1]["records"]
        }
        self.assertEqual(records[d["d1"]], {
            "sha256": d["d1"], "old": "cat", "new": "kitten",
            "changed": True, "rev": 1,
        })
        self.assertEqual(records[d["d2"]], {
            "sha256": d["d2"], "old": "dog", "new": "puppy",
            "changed": True, "rev": 2,
        })
        # The no-op stays in the batch record, counts as unchanged and did
        # not move its revision (pinned rev 0, no "rev" on the manifest).
        self.assertEqual(records[d["d3"]], {
            "sha256": d["d3"], "old": "fish", "new": "fish",
            "changed": False, "rev": 0,
        })
        self.assertEqual(records[d["d5"]], {
            "sha256": d["d5"], "old": None, "new": "cat",
            "changed": True, "rev": 1,
        })
        self.assertEqual(entries[-1]["changed_count"], 3)

        # The earlier undo marker and timestamp keep their saved value.
        self.assertTrue(entries[1]["undone"])
        self.assertEqual(entries[1]["undone_at"], sc["undone_at"])

        # Summary reflects the same complete state and the byte total is the
        # registered content, unchanged by relabeling.
        summary = store.summary()
        self.assertEqual(summary["items"], 5)
        self.assertEqual(summary["bytes"], sc["total_bytes"])
        self.assertEqual(summary["labels"], {
            "bird": 1, "cat": 1, "fish": 1, "kitten": 1, "puppy": 1,
        })

        # The pre-interruption split keeps creation-time labels and members.
        plan = store.get_split("plan-a")
        self.assertEqual(plan, sc["plan"])
        plan_labels = {
            member["sha256"]: member["label"]
            for member in plan["sets"]["train"]["members"]
        }
        self.assertEqual(plan_labels[d["d1"]], "cat")
        self.assertEqual(plan_labels[d["d2"]], "dog")
        self.assertEqual(plan_labels[d["d4"]], "bird")
        self.assertEqual(plan_labels[d["d5"]], "")

        self.assertFalse(store.transaction_path.exists())

    def assert_state_stable_across_queries_and_reopens(self, sc: dict) -> None:
        """Queries and further opens neither append nor bump revisions."""
        store = DatasetStore(self.root)
        manifest_bytes = self.store.manifest_path.read_bytes()
        history_bytes = self.store.batches_path.read_bytes()

        store.history()
        store.summary()
        for label in ("kitten", "puppy", "fish", "bird", "cat", ""):
            store.find_by_label(label)
        for digest in sc["digests"].values():
            store.lookup_label(digest)

        DatasetStore(self.root)
        self.assertEqual(self.store.manifest_path.read_bytes(), manifest_bytes)
        self.assertEqual(self.store.batches_path.read_bytes(), history_bytes)
        self.assert_mid_batch_completed(DatasetStore(self.root), sc)


class BatchSaveInterruptionRecoveryTest(InterruptedBatchHarness):
    def test_labels_saved_but_history_not_is_completed_on_reopen(self) -> None:
        sc = self.build_scenario()

        # Opening the workspace finishes the prepared batch automatically.
        reopened = DatasetStore(self.root)
        self.assert_mid_batch_completed(reopened, sc)

        # Continuing to query the same workspace keeps the result.
        reopened.summary()
        self.assertEqual(
            [entry["batch"] for entry in reopened.history()],
            ["b-prev", "b-undone", self.INTERRUPTED_NUMBER],
        )

        # Another open and more queries must not redo the modification:
        # no second history row, no second revision bump.
        self.assert_state_stable_across_queries_and_reopens(sc)

    def test_query_on_already_open_store_completes_interrupted_batch(self) -> None:
        sc = self.build_scenario()

        # No fresh open: the store instance that survived the stopped
        # submission finishes the journal on its next operation, so a plain
        # query on the already-open workspace shows the complete batch.
        summary = self.store.summary()
        self.assertEqual(summary["items"], 5)
        self.assertFalse(self.store.transaction_path.exists())
        self.assert_mid_batch_completed(self.store, sc)
        self.assert_state_stable_across_queries_and_reopens(sc)

    def test_recovered_batch_replays_with_original_counts(self) -> None:
        sc = self.build_scenario()
        reopened = DatasetStore(self.root)
        self.assert_mid_batch_completed(reopened, sc)

        # Resubmitting the same content after recovery is the ordinary
        # replay path: original result counts, nothing applied again.
        replay = reopened.submit_batch(sc["payload"])
        self.assertEqual(replay, {
            "batch": self.INTERRUPTED_NUMBER,
            "status": "already-applied",
            "changed": 3,
            "unchanged": 1,
            "total": 4,
        })
        history = reopened.history()
        self.assertEqual(
            [entry["batch"] for entry in history],
            ["b-prev", "b-undone", self.INTERRUPTED_NUMBER],
        )
        self.assertEqual(history[-1], sc["expected_entry"])
        items = {
            item["sha256"]: item
            for item in json.loads(reopened.manifest_path.read_text("utf-8"))["items"]
        }
        self.assertEqual(items[sc["digests"]["d1"]].get("rev"), 1)
        self.assertEqual(items[sc["digests"]["d2"]].get("rev"), 2)
        self.assertEqual(items[sc["digests"]["d5"]].get("rev"), 1)

    def test_normal_batch_after_recovery_still_reports_counts(self) -> None:
        sc = self.build_scenario()
        reopened = DatasetStore(self.root)
        self.assert_mid_batch_completed(reopened, sc)

        # A later, fully completed batch keeps the established changed vs
        # unchanged meanings, appends once and bumps only its changed sample.
        d = sc["digests"]
        result = reopened.submit_batch(
            {
                "batch": "b-after",
                "changes": [
                    {"sha256": d["d4"], "old": "bird", "new": "hawk"},
                    {"sha256": d["d1"], "old": "kitten", "new": "kitten"},
                ],
            }
        )
        self.assertEqual(result, {
            "batch": "b-after", "status": "applied",
            "changed": 1, "unchanged": 1, "total": 2,
        })
        self.assertEqual(
            [entry["batch"] for entry in reopened.history()],
            ["b-prev", "b-undone", self.INTERRUPTED_NUMBER, "b-after"],
        )
        items = {
            item["sha256"]: item
            for item in json.loads(reopened.manifest_path.read_text("utf-8"))["items"]
        }
        self.assertEqual(items[d["d4"]].get("rev"), 3)
        self.assertEqual(items[d["d1"]].get("rev"), 1)


class RecoveryStorageFailureTest(InterruptedBatchHarness):
    """Recovery meets another read/write failure before storage heals."""

    @contextlib.contextmanager
    def failing_install(self, target_name: str):
        """Make recovery's install of one state file raise EIO."""
        real_write = DatasetStore._write_json_atomic

        def failing_write(self, path: Path, payload: dict) -> None:
            if path.name == target_name:
                raise OSError(errno.EIO, "simulated storage write failure")
            return real_write(self, path, payload)

        with mock.patch.object(DatasetStore, "_write_json_atomic", failing_write):
            yield

    def _assert_every_use_fails(self, error_type: type) -> None:
        # A process that already had the workspace open runs recovery at the
        # start of every operation, so even read-only queries must fail
        # openly instead of observing the straddled files.
        with self.assertRaises(error_type):
            self.store.history()
        with self.assertRaises(error_type):
            self.store.summary()
        with self.assertRaises(error_type):
            self.store.find_by_label("kitten")
        # Reopening fails the same way.
        with self.assertRaises(error_type):
            DatasetStore(self.root)

    def _assert_straddle_untouched(self, sc: dict) -> None:
        # The labels from the interrupted commit are on disk (that is the
        # window this regression covers), the history still lacks the batch
        # and the journal is still waiting; no successful use may observe
        # this state.
        self.assertTrue(self.store.transaction_path.exists())
        self.assertEqual(
            [entry["batch"] for entry in self._raw_history()["batches"]],
            ["b-prev", "b-undone"],
        )
        items = {
            item["sha256"]: item for item in self._raw_manifest()["items"]
        }
        self.assertEqual(items[sc["digests"]["d1"]]["label"], "kitten")
        self.assertEqual(items[sc["digests"]["d2"]]["label"], "puppy")
        self.assertEqual(items[sc["digests"]["d3"]]["label"], "fish")
        self.assertEqual(items[sc["digests"]["d4"]]["label"], "bird")
        self.assertEqual(items[sc["digests"]["d5"]]["label"], "cat")

    def _run_write_failure_case(self, target_name: str) -> None:
        sc = self.build_scenario()

        with self.failing_install(target_name):
            self._assert_every_use_fails(OSError)
            self._assert_straddle_untouched(sc)

        # Storage heals: reopening completes the original batch once, and
        # the result is identical to an uninterrupted submission.
        recovered = DatasetStore(self.root)
        self.assert_mid_batch_completed(recovered, sc)
        self.assert_state_stable_across_queries_and_reopens(sc)

    def test_manifest_install_failure_fails_open_then_completes(self) -> None:
        self._run_write_failure_case("manifest.json")

    def test_history_install_failure_fails_open_then_completes(self) -> None:
        # This is the closest re-encounter with the original crash: labels
        # are installed again first, the history write keeps failing, and no
        # use may report that half state as success.
        self._run_write_failure_case("batches.json")

    def test_journal_read_failure_fails_open_then_completes(self) -> None:
        sc = self.build_scenario()
        transaction = self.store.transaction_path
        real_read_text = Path.read_text

        def fail_journal_read(self, *args, **kwargs):
            if self == transaction:
                raise OSError(errno.EIO, "simulated storage read failure")
            return real_read_text(self, *args, **kwargs)

        manifest_before = self.store.manifest_path.read_bytes()
        history_before = self.store.batches_path.read_bytes()
        with mock.patch.object(Path, "read_text", fail_journal_read):
            # The read failure is reported as a recovery failure, never as a
            # successful query over the straddled state.
            with self.assertRaises(BatchError) as ctx:
                self.store.history()
            self.assertIn("cannot recover interrupted transaction", str(ctx.exception))
            with self.assertRaises(BatchError):
                DatasetStore(self.root)

        # Neither file moved while recovery could not read the journal.
        self.assertTrue(self.store.transaction_path.exists())
        self.assertEqual(self.store.manifest_path.read_bytes(), manifest_before)
        self.assertEqual(self.store.batches_path.read_bytes(), history_before)

        # Once reads work again, the same prepared batch completes once.
        recovered = DatasetStore(self.root)
        self.assert_mid_batch_completed(recovered, sc)
        self.assert_state_stable_across_queries_and_reopens(sc)


if __name__ == "__main__":
    unittest.main()
