"""Regression tests for find-label queries interleaved with batch commits.

A user browsing one category (``find-label``) while another local process
submits a label batch must always observe one *complete* registration
state: either everything from before the batch commit or everything from
after it — never a mix of some samples' new labels with other samples'
old labels, and never a corruption report just because a write is in
flight.  The same holds when the batch is rejected: no sample may appear
moved and restored, and the rejection is reported by the batch operation
alone while the classification queries keep succeeding.
"""

from __future__ import annotations

import multiprocessing
import queue as queue_module
import tempfile
import unittest
from pathlib import Path

from vision_workbench.store import DatasetStore

ctx = multiprocessing.get_context("fork")


def _find_label_worker(root: str, labels: list, stop, queue) -> None:
    """Continuously query each label until told to stop, reporting results.

    Every observation is reported as
    ``("ok", label, count, ((sha256, source, size, label), ...))``; any
    failure — including a spurious corruption report — is reported as
    ``("error", message)`` so the parent can fail the test.
    """
    try:
        store = DatasetStore(Path(root))
        while not stop.is_set():
            for label in labels:
                result = store.find_by_label(label)
                queue.put(
                    (
                        "ok",
                        label,
                        result["count"],
                        tuple(
                            (
                                sample["sha256"],
                                sample["source"],
                                sample["size"],
                                sample["label"],
                            )
                            for sample in result["samples"]
                        ),
                    )
                )
    except Exception as error:  # noqa: BLE001 - report back to parent
        queue.put(("error", f"{type(error).__name__}: {error}"))


def _batch_worker(root: str, number: str, changes: list, queue) -> None:
    try:
        store = DatasetStore(Path(root))
        result = store.submit_batch({"batch": number, "changes": changes})
        queue.put(("ok", result["status"]))
    except Exception as error:  # noqa: BLE001 - report back to parent
        queue.put(("error", f"{type(error).__name__}: {error}"))


def _drain(queue) -> list:
    """Collect everything a finished worker left on its queue."""
    rows = []
    while True:
        try:
            rows.append(queue.get_nowait())
        except queue_module.Empty:
            return rows


class FindLabelInterleavingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "dataset"
        self.store = DatasetStore(self.root)
        self.store.initialize()
        self._serial = 0

    def tearDown(self) -> None:
        self.temp.cleanup()

    def add_sample(self, label: str | None) -> tuple[str, str, int]:
        """Add a sample; return (digest, recorded source, size)."""
        self._serial += 1
        path = self.root.parent / f"sample-{self._serial}.jpg"
        data = f"content-{self._serial}".encode()
        path.write_bytes(data)
        result = self.store.add(path, label)
        self.assertTrue(result.added)
        return result.digest, str(path.resolve()), len(data)

    def run_viewer_during(self, labels: list, submitter_args: tuple) -> tuple[list, tuple]:
        """Run a querying process interleaved with one batch submission.

        Returns the viewer's observations and the batch worker's verdict.
        The query queue is drained while the submission runs: the viewer
        produces results far faster than they can be consumed, and an
        undrained queue would block it forever.
        """
        stop = ctx.Event()
        query_queue = ctx.Queue()
        viewer = ctx.Process(
            target=_find_label_worker,
            args=(str(self.root), labels, stop, query_queue),
        )
        viewer.start()

        batch_queue = ctx.Queue()
        submitter = ctx.Process(
            target=_batch_worker, args=(str(self.root),) + submitter_args + (batch_queue,)
        )
        submitter.start()
        observations = []
        while submitter.is_alive():
            observations.extend(_drain(query_queue))
            submitter.join(timeout=0.01)
        self.assertEqual(submitter.exitcode, 0)
        batch_verdict = batch_queue.get(timeout=30)

        stop.set()
        # Keep draining while the viewer winds down: it may be blocked on
        # a full pipe and only observes the stop event once relieved.
        while viewer.is_alive():
            observations.extend(_drain(query_queue))
            viewer.join(timeout=0.01)
        self.assertEqual(viewer.exitcode, 0)
        observations.extend(_drain(query_queue))
        return observations, batch_verdict

    def test_batch_commit_interleaved_with_label_queries(self) -> None:
        cats = [self.add_sample("cat") for _ in range(3)]
        dog = self.add_sample("dog")
        cat_digests = sorted(entry[0] for entry in cats)
        dog_digest = dog[0]
        cat_records = {digest: (source, size) for digest, source, size in cats}

        # A saved split plan snapshots the old labels; those labels must
        # never participate in matching afterwards.
        plan = self.store.create_split("baseline", 0, [1, 0, 0]).plan
        # A source image deleted after registration must not obstruct the
        # classification query either.
        Path(cats[0][1]).unlink()

        changes = [
            {"sha256": digest, "old": "cat", "new": "dog"}
            for digest, _, _ in cats
        ]
        observations, verdict = self.run_viewer_during(["cat", "dog"], ("b1", changes))

        # The batch committed as one unit.
        self.assertEqual(verdict, ("ok", "applied"))

        # Every interleaved observation is one complete state: the three
        # original cats or none, the one original dog or all four — never
        # a mixture, and never an error from the in-flight write.
        self.assertTrue(observations, "viewer produced no observations")
        for row in observations:
            self.assertEqual(row[0], "ok", row)
            _, label, count, samples = row
            digests = [sample[0] for sample in samples]
            # The reported count is always the actual list length, the
            # list is sorted by full digest and lists each digest once.
            self.assertEqual(count, len(samples), row)
            self.assertEqual(digests, sorted(digests), row)
            self.assertEqual(len(set(digests)), len(digests), row)
            if label == "cat":
                self.assertIn(count, (0, 3), row)
                for digest, source, size, sample_label in samples:
                    self.assertEqual(sample_label, "cat", row)
                    # First-registration source and size, even for the
                    # sample whose image file was deleted.
                    self.assertEqual((source, size), cat_records[digest], row)
            else:
                self.assertEqual(label, "dog")
                self.assertIn(count, (1, 4), row)
                # The sample that never joined the batch is always there.
                self.assertIn(dog_digest, digests, row)
                if count == 4:
                    self.assertEqual(digests, sorted(cat_digests + [dog_digest]), row)
                for *_, sample_label in samples:
                    self.assertEqual(sample_label, "dog", row)

        # Queries issued after the batch reported success must reflect the
        # new labels — the saved plan's old "cat" labels do not match.
        self.assertEqual(
            self.store.find_by_label("cat"),
            {"label": "cat", "count": 0, "samples": []},
        )
        dogs = self.store.find_by_label("dog")
        self.assertEqual(dogs["count"], 4)
        self.assertEqual(
            [sample["sha256"] for sample in dogs["samples"]],
            sorted(cat_digests + [dog_digest]),
        )
        self.assertTrue(all(sample["label"] == "dog" for sample in dogs["samples"]))
        # The bystander kept its original label and first-registration
        # source and size.
        bystander = next(s for s in dogs["samples"] if s["sha256"] == dog_digest)
        self.assertEqual((bystander["source"], bystander["size"]), (dog[1], dog[2]))
        # The plan itself is untouched and still holds the old labels.
        self.assertEqual(self.store.get_split("baseline"), plan)

    def test_rejected_batch_never_shows_partial_classification(self) -> None:
        cats = [self.add_sample("cat") for _ in range(3)]
        dog = self.add_sample("dog")
        cat_digests = sorted(entry[0] for entry in cats)

        changes = [
            {"sha256": cats[0][0], "old": "cat", "new": "dog"},
            {"sha256": cats[1][0], "old": "cat", "new": "dog"},
            # This sample's current label is "cat", not the declared old
            # label, so the whole batch must be rejected.
            {"sha256": cats[2][0], "old": "tiger", "new": "dog"},
        ]
        observations, verdict = self.run_viewer_during(["cat", "dog"], ("b1", changes))

        # The rejection is reported by the batch operation, naming the
        # offending sample and both labels.
        self.assertEqual(verdict[0], "error")
        self.assertIn("BatchError", verdict[1])
        self.assertIn(cats[2][0], verdict[1])
        self.assertIn("tiger", verdict[1])

        # Every interleaved observation — and every query afterwards — is
        # the original complete classification: no sample is ever seen
        # moved away and restored, and no query is treated as a failure.
        self.assertTrue(observations, "viewer produced no observations")
        for row in observations:
            self.assertEqual(row[0], "ok", row)
            _, label, count, samples = row
            self.assertEqual(count, len(samples), row)
            if label == "cat":
                self.assertEqual([s[0] for s in samples], cat_digests, row)
                self.assertTrue(all(s[3] == "cat" for s in samples), row)
            else:
                self.assertEqual(label, "dog")
                self.assertEqual([s[0] for s in samples], [dog[0]], row)
                self.assertEqual(samples[0][3], "dog", row)

        self.assertEqual(self.store.find_by_label("cat")["count"], 3)
        self.assertEqual(
            [s["sha256"] for s in self.store.find_by_label("cat")["samples"]],
            cat_digests,
        )
        self.assertEqual(self.store.find_by_label("dog")["count"], 1)
        # The failed batch occupies no number and writes no history.
        self.assertEqual(self.store.history(), [])


if __name__ == "__main__":
    unittest.main()
