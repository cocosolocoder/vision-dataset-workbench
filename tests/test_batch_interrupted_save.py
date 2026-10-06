"""Regression coverage for batch saves interrupted between the two installs.

A batch label update commits through a write-ahead journal: the prepared
manifest + history pair is journaled first, then ``manifest.json`` and
``batches.json`` are installed one after the other.  The existing recovery
tests start from a crash where *neither* file was installed; the scenarios
pinned here start later in the commit, where the program stopped **after
the new labels were saved but before the batch's history entry was** —
the journal still holds both payloads, the manifest on disk already shows
the post-batch labels and the history on disk still lacks the batch.

Two guarantees are covered:

* Reopening (or any query on an already-open store) finishes the
  interrupted batch exactly once: every sample shows the requested
  post-batch labels, the history gains exactly this one batch appended
  after the previously committed ones, and the recorded before/after
  labels, ``changed`` flags, per-record revisions and ``changed_count``
  match what the request really did.  Target-equals-current records are
  still listed in the batch's records but do not count as changes and do
  not bump their sample's revision; actually changed samples keep exactly
  the revision the batch gave them — recovery never applies the same
  change twice.  Content digests, registered sources and byte sizes are
  untouched, samples outside the batch are unaffected, and split plans
  saved earlier keep their creation-time labels and memberships.

* When the recovery itself hits a storage read/write failure, the use
  fails clearly instead of returning a success-looking query result whose
  labels and history disagree.  Once the storage condition is gone, the
  next open still completes the original batch — same labels, complete
  history, no missing or duplicated batch entries, no extra revision
  bumps — and later queries and reopens keep that result, with earlier
  batches' undo flags untouched.
"""

from __future__ import annotations

import contextlib
import errno
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from vision_workbench.batches import BatchError
from vision_workbench.store import DatasetStore

REPO_ROOT = Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def _crash_before_history_save(store: DatasetStore):
    """Stop the commit after the manifest install, before the history one.

    The journal and the new manifest are written for real; the write of
    ``batches.json`` raises, so the submission dies exactly where a
    crashed process would leave it: labels saved, batch history not.
    """
    real_write = DatasetStore._write_json_atomic

    def writing(self: DatasetStore, path: Path, payload: dict) -> None:
        if Path(path) == store.batches_path:
            raise OSError(errno.EIO, "simulated stop before history save")
        return real_write(self, path, payload)

    with mock.patch.object(DatasetStore, "_write_json_atomic", writing):
        yield


@contextlib.contextmanager
def _journal_read_failure(store: DatasetStore):
    """Make reading the prepared journal fail with a storage error."""
    real_read_text = Path.read_text

    def guarded(self: Path, *args: object, **kwargs: object) -> str:
        if self == store.transaction_path:
            raise OSError(errno.EIO, "simulated journal read failure")
        return real_read_text(self, *args, **kwargs)

    with mock.patch.object(Path, "read_text", guarded):
        yield


@contextlib.contextmanager
def _recovery_write_failure(store: DatasetStore):
    """Make the recovery's first install (the manifest) fail."""
    real_write = DatasetStore._write_json_atomic

    def writing(self: DatasetStore, path: Path, payload: dict) -> None:
        if Path(path) == store.manifest_path:
            raise OSError(errno.EIO, "simulated recovery write failure")
        return real_write(self, path, payload)

    with mock.patch.object(DatasetStore, "_write_json_atomic", writing):
        yield


class StoreHarness(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "dataset"
        self.store = DatasetStore(self.root)
        self.store.initialize()
        self._serial = 0

    def tearDown(self) -> None:
        self.temp.cleanup()

    def add_sample(self, label: str | None) -> str:
        self._serial += 1
        path = self.root.parent / f"sample-{self._serial}.jpg"
        path.write_bytes(f"content-{self._serial}".encode())
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

    def crash_submit(self, number: str, *changes: tuple) -> None:
        """Submit a batch whose commit stops before the history install."""
        with _crash_before_history_save(self.store):
            with self.assertRaises(OSError):
                self.submit(number, *changes)

    def disk_manifest_items(self) -> dict[str, dict]:
        """Manifest items straight from disk, keyed by digest (no recovery)."""
        manifest = json.loads(self.store.manifest_path.read_text(encoding="utf-8"))
        return {item["sha256"]: item for item in manifest["items"]}

    def disk_history_batches(self) -> list[dict]:
        """History entries straight from disk (no recovery)."""
        history = json.loads(self.store.batches_path.read_text(encoding="utf-8"))
        return history["batches"]

    def assert_interrupted_state(
        self, expected_labels: dict[str, str | None], committed: list[str]
    ) -> None:
        """Pin the crash premise: labels saved, this batch's history not."""
        self.assertTrue(self.store.transaction_path.exists())
        labels = {
            digest: item.get("label")
            for digest, item in self.disk_manifest_items().items()
        }
        self.assertEqual(labels, expected_labels)
        self.assertEqual(
            [entry["batch"] for entry in self.disk_history_batches()], committed
        )


class InterruptedBatchRecoveryTest(StoreHarness):
    """Recovery when labels were saved but the batch's history was not."""

    def setUp(self) -> None:
        super().setUp()
        # d1, d2 actually change in the interrupted batch; d3's target
        # equals its current label; d4 is only touched by an earlier,
        # normally completed batch.
        self.d1 = self.add_sample("cat")
        self.d2 = self.add_sample("dog")
        self.d3 = self.add_sample("cat")
        self.d4 = self.add_sample("bird")

        # A split plan saved before any batch keeps its creation-time
        # labels and memberships no matter what later batches do.
        self.plan = self.store.create_split("p", 0, [1, 0, 0]).plan

        # A normally completed batch with one real change and one
        # target-equals-current record: the existing count meanings.
        result = self.submit("b1", (self.d4, "bird", "fish"), (self.d2, "dog", "dog"))
        self.assertEqual(result["status"], "applied")
        self.assertEqual(result["changed"], 1)
        self.assertEqual(result["unchanged"], 1)
        self.assertEqual(result["total"], 2)
        self.history_before = self.store.history()
        self.items_before = self.disk_manifest_items()

        # The interrupted batch: two real changes plus one no-op record.
        self.crash_submit(
            "b2",
            (self.d1, "cat", "kitten"),
            (self.d2, "dog", "fish"),
            (self.d3, "cat", "cat"),
        )
        self.assert_interrupted_state(
            {
                self.d1: "kitten",
                self.d2: "fish",
                self.d3: "cat",
                self.d4: "fish",
            },
            ["b1"],
        )

    def test_reopen_completes_batch_exactly_once(self) -> None:
        reopened = DatasetStore(self.root)
        self.assertFalse(self.store.transaction_path.exists())

        # Every batch sample shows the requested post-request label; the
        # no-op sample keeps its label; the unrelated sample keeps the
        # label the earlier normal batch gave it.
        self.assertEqual(reopened.lookup_label(self.d1)["label"], "kitten")
        self.assertEqual(reopened.lookup_label(self.d2)["label"], "fish")
        self.assertEqual(reopened.lookup_label(self.d3)["label"], "cat")
        self.assertEqual(reopened.lookup_label(self.d4)["label"], "fish")
        self.assertEqual(
            reopened.summary()["labels"], {"cat": 1, "fish": 2, "kitten": 1}
        )

        # The public label query reports the recovered labels with the
        # registered source and byte size intact.
        found = reopened.find_by_label("kitten")
        self.assertEqual(found["count"], 1)
        self.assertEqual(found["samples"][0]["sha256"], self.d1)
        self.assertEqual(
            found["samples"][0]["source"], self.items_before[self.d1]["source"]
        )
        self.assertEqual(
            found["samples"][0]["size"], self.items_before[self.d1]["size"]
        )
        self.assertEqual(
            [sample["sha256"] for sample in reopened.find_by_label("cat")["samples"]],
            [self.d3],
        )

        # Exactly one batch was appended, after the previously committed
        # one; the earlier entry is byte-for-byte what it was, undo
        # markers included.
        history = reopened.history()
        self.assertEqual([entry["batch"] for entry in history], ["b1", "b2"])
        self.assertEqual(history[0], self.history_before[0])
        self.assertFalse(history[0]["undone"])
        self.assertIsNone(history[0]["undone_at"])

        # The recovered entry records the request's real outcome.
        entry = history[1]
        self.assertEqual(entry["changed_count"], 2)
        self.assertFalse(entry["undone"])
        self.assertIsNone(entry["undone_at"])
        records = {record["sha256"]: record for record in entry["records"]}
        self.assertEqual(set(records), {self.d1, self.d2, self.d3})
        self.assertEqual(
            records[self.d1],
            {
                "sha256": self.d1,
                "old": "cat",
                "new": "kitten",
                "changed": True,
                "rev": 1,
            },
        )
        self.assertEqual(
            records[self.d2],
            {
                "sha256": self.d2,
                "old": "dog",
                "new": "fish",
                "changed": True,
                "rev": 1,
            },
        )
        # The no-op record is listed but is not a change and did not earn
        # its sample a revision.
        self.assertEqual(
            records[self.d3],
            {
                "sha256": self.d3,
                "old": "cat",
                "new": "cat",
                "changed": False,
                "rev": 0,
            },
        )

        # Revisions: the two real changes carry exactly the revision the
        # batch gave them (recovery must not count the same change twice);
        # the no-op sample's revision was never bumped; the earlier
        # batch's change keeps its own revision.
        items = self.disk_manifest_items()
        self.assertEqual(items[self.d1].get("rev", 0), 1)
        self.assertEqual(items[self.d2].get("rev", 0), 1)
        self.assertEqual(items[self.d3].get("rev", 0), 0)
        self.assertEqual(items[self.d4].get("rev", 0), 1)

        # Content digests, registered sources and byte sizes are as
        # registered; nothing outside labels/revisions moved.
        for digest, before in self.items_before.items():
            item = items[digest]
            self.assertEqual(item["sha256"], before["sha256"])
            self.assertEqual(item["source"], before["source"])
            self.assertEqual(item["size"], before["size"])

        # The split plan saved before the batches keeps its creation-time
        # labels and memberships.
        self.assertEqual(reopened.get_split("p"), self.plan)
        self.assertEqual(
            reopened.get_split("p")["samples"]["distribution"],
            {"bird": 1, "cat": 2, "dog": 1},
        )

    def test_recovered_state_is_stable_across_queries_and_reopens(self) -> None:
        reopened = DatasetStore(self.root)
        first_history = reopened.history()
        first_summary = reopened.summary()

        # Continuing to query and opening the workspace again changes
        # nothing: the batch is finished once, never re-applied.
        again = DatasetStore(self.root)
        self.assertEqual(again.history(), first_history)
        self.assertEqual(again.summary(), first_summary)
        self.assertEqual(again.lookup_label(self.d1)["label"], "kitten")
        items = self.disk_manifest_items()
        self.assertEqual(items[self.d1].get("rev", 0), 1)
        self.assertEqual(items[self.d2].get("rev", 0), 1)
        self.assertEqual(items[self.d3].get("rev", 0), 0)
        self.assertFalse(self.store.transaction_path.exists())

        # The workspace is fully usable afterwards: a later normal batch
        # appends after the recovered one and reports the existing count
        # meanings.
        result = again.submit_batch(
            {
                "batch": "b3",
                "changes": [
                    {"sha256": self.d3, "old": "cat", "new": "bird"},
                    {"sha256": self.d1, "old": "kitten", "new": "kitten"},
                ],
            }
        )
        self.assertEqual(result["status"], "applied")
        self.assertEqual(result["changed"], 1)
        self.assertEqual(result["unchanged"], 1)
        self.assertEqual(
            [entry["batch"] for entry in again.history()], ["b1", "b2", "b3"]
        )
        self.assertEqual(again.history()[0], self.history_before[0])

    def test_cli_queries_observe_the_completed_batch(self) -> None:
        reopened = DatasetStore(self.root)  # completes the recovery
        self.assertEqual(reopened.lookup_label(self.d1)["label"], "kitten")

        def run_cli(*arguments: str) -> subprocess.CompletedProcess:
            return subprocess.run(
                [sys.executable, "-m", "vision_workbench", *arguments],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        label = run_cli("label", str(self.root), self.d1)
        self.assertEqual(label.returncode, 0, label.stderr)
        self.assertEqual(json.loads(label.stdout)["label"], "kitten")

        history = run_cli("history", str(self.root))
        self.assertEqual(history.returncode, 0, history.stderr)
        self.assertEqual(
            [entry["batch"] for entry in json.loads(history.stdout)["batches"]],
            ["b1", "b2"],
        )

        summary = run_cli("summary", str(self.root))
        self.assertEqual(summary.returncode, 0, summary.stderr)
        self.assertEqual(
            json.loads(summary.stdout)["labels"],
            {"cat": 1, "fish": 2, "kitten": 1},
        )


class RecoveryReadWriteFailureTest(StoreHarness):
    """A storage failure during recovery defers but never loses the batch."""

    def setUp(self) -> None:
        super().setUp()
        self.d1 = self.add_sample("cat")
        self.d2 = self.add_sample("dog")
        self.submit("b1", (self.d2, "dog", "fish"))
        self.history_before = self.store.history()
        # An already-open store: its later queries must also finish (or
        # clearly refuse) the interrupted batch rather than observe a
        # half-saved state.
        self.open_store = DatasetStore(self.root)
        self.crash_submit("b2", (self.d1, "cat", "kitten"))
        self.assert_interrupted_state(
            {self.d1: "kitten", self.d2: "fish"}, ["b1"]
        )

    def assert_completed_batch(self) -> None:
        reopened = DatasetStore(self.root)
        self.assertFalse(self.store.transaction_path.exists())
        self.assertEqual(reopened.lookup_label(self.d1)["label"], "kitten")
        self.assertEqual(reopened.lookup_label(self.d2)["label"], "fish")

        history = reopened.history()
        self.assertEqual([entry["batch"] for entry in history], ["b1", "b2"])
        # The earlier batch is preserved as it was, undo markers included.
        self.assertEqual(history[0], self.history_before[0])
        self.assertFalse(history[0]["undone"])
        self.assertIsNone(history[0]["undone_at"])
        # The interrupted batch appears exactly once, with its real
        # outcome and exactly the revision it assigned.
        entry = history[1]
        self.assertEqual(entry["changed_count"], 1)
        self.assertEqual(
            entry["records"],
            [
                {
                    "sha256": self.d1,
                    "old": "cat",
                    "new": "kitten",
                    "changed": True,
                    "rev": 1,
                }
            ],
        )
        self.assertEqual(self.disk_manifest_items()[self.d1].get("rev", 0), 1)
        self.assertEqual(self.disk_manifest_items()[self.d2].get("rev", 0), 1)

        # The result holds across further queries and another reopen.
        again = DatasetStore(self.root)
        self.assertEqual(again.history(), history)
        self.assertEqual(again.lookup_label(self.d1)["label"], "kitten")
        self.assertEqual(self.disk_manifest_items()[self.d1].get("rev", 0), 1)

    def test_journal_read_failure_fails_clearly_then_completes(self) -> None:
        with _journal_read_failure(self.store):
            # Reopening the workspace fails clearly...
            with self.assertRaises(BatchError) as raised:
                DatasetStore(self.root)
            self.assertIn(
                "cannot recover interrupted transaction", str(raised.exception)
            )
            # ...and so does any query on an already-open store: no
            # success-looking result whose labels and history disagree.
            with self.assertRaises(BatchError):
                self.open_store.summary()
            with self.assertRaises(BatchError):
                self.open_store.history()
            with self.assertRaises(BatchError):
                self.open_store.lookup_label(self.d1)

        # The failed attempts changed nothing: the journal is still there
        # and the interrupted on-disk state is untouched.
        self.assert_interrupted_state(
            {self.d1: "kitten", self.d2: "fish"}, ["b1"]
        )

        # Once the storage condition is gone, the original batch still
        # completes — nothing lost, duplicated or re-counted.
        self.assert_completed_batch()

    def test_recovery_write_failure_fails_clearly_then_completes(self) -> None:
        with _recovery_write_failure(self.store):
            with self.assertRaises(OSError):
                DatasetStore(self.root)
            with self.assertRaises(OSError):
                self.open_store.summary()

        self.assert_interrupted_state(
            {self.d1: "kitten", self.d2: "fish"}, ["b1"]
        )

        self.assert_completed_batch()


if __name__ == "__main__":
    unittest.main()
