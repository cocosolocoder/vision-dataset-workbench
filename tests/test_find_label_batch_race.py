"""Regression coverage for find-label queries interleaved with batch writes.

Another local process can submit a label batch while a category is being
viewed.  Every ``find-label`` query must correspond to one *complete*
registration state: it may see all of a batch's old labels or all of its
new labels, but never some samples of one batch already relabeled while
others keep their old labels, and an in-flight write must never make a
healthy workspace look corrupted.

The scenarios use the README's example: one batch moves three images with
different contents from ``cat`` to ``dog``, while a fourth image that was
already ``dog`` never participates.

* A successful batch makes a ``cat`` query return exactly the original
  three digests or none, and a ``dog`` query return the original one or
  the final four; ``count`` always equals the list length and every
  returned label matches that one complete state.  Queries issued after
  the batch reported success always see the new labels; the sample that
  does not take part keeps its label and first-registration summary.
* A rejected batch (one record's declared old label disagrees with the
  current label) changes nothing, and queries interleaved with that
  submission, or run after it, keep returning the original complete
  classification.  The rejection reason is reported by the batch
  operation; a normal category query is never itself a failure.

The query still reads only the current registrations: matching uses the
full digest ordering, the first registration's source and byte size, and
never the old labels saved in split plans, and a source image that has
since been moved or deleted does not get in the way.  These tests add
coverage only; they change no batch submission rule and add no query
option.
"""

from __future__ import annotations

import json
import multiprocessing
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from vision_workbench.store import DatasetStore

REPO_ROOT = Path(__file__).resolve().parents[1]

ctx = multiprocessing.get_context("fork")

# The store operations take milliseconds; this bounds the wait for one
# hammered batch/query round on a heavily loaded CI machine.
ROUND_TIMEOUT = 60


def _seed_workspace(root: Path) -> dict:
    """Create the four-sample workspace and return its expected records.

    Three samples are registered as ``cat`` (each with distinct content)
    and one as ``dog``.  The mapping carries the full find-label summary a
    result must show in each of the two complete states.
    """
    store = DatasetStore(root)
    store.initialize()
    cats = []
    for index in range(3):
        source = root.parent / f"cat-{index}.jpg"
        data = f"different-cat-content-{index}".encode()
        source.write_bytes(data)
        result = store.add(source, "cat")
        assert result.added
        cats.append(
            {
                "sha256": result.digest,
                "source": str(source.resolve()),
                "size": len(data),
            }
        )
    stays_source = root.parent / "dog-stays.jpg"
    stays_data = b"the-dog-that-never-moves"
    stays_source.write_bytes(stays_data)
    stays_result = store.add(stays_source, "dog")
    assert stays_result.added
    dog_stays = {
        "sha256": stays_result.digest,
        "source": str(stays_source.resolve()),
        "size": len(stays_data),
    }
    return {
        "cats": cats,
        "dog_stays": dog_stays,
        # Full find-label records for each of the two complete states;
        # both are ordered by digest exactly as find-label returns them.
        "before_cat": sorted(
            (
                {"sha256": c["sha256"], "source": c["source"], "size": c["size"],
                 "label": "cat"}
                for c in cats
            ),
            key=lambda sample: sample["sha256"],
        ),
        "after_dog": sorted(
            [
                {"sha256": c["sha256"], "source": c["source"], "size": c["size"],
                 "label": "dog"}
                for c in cats
            ]
            + [{
                "sha256": dog_stays["sha256"],
                "source": dog_stays["source"],
                "size": dog_stays["size"],
                "label": "dog",
            }],
            key=lambda sample: sample["sha256"],
        ),
        "stays_dog": {
            "sha256": dog_stays["sha256"],
            "source": dog_stays["source"],
            "size": dog_stays["size"],
            "label": "dog",
        },
    }


def _changes(digests: list[str], old: str | None, new: str | None) -> list[dict]:
    return [{"sha256": digest, "old": old, "new": new} for digest in digests]


# ---------------------------------------------------------------------------
# Worker processes
# ---------------------------------------------------------------------------


def _batch_worker(root: str, number: str, changes: list, queue) -> None:
    try:
        result = DatasetStore(Path(root)).submit_batch(
            {"batch": number, "changes": changes}
        )
        queue.put(("batch", "ok", result["status"]))
    except Exception as error:  # noqa: BLE001 - report the rejection back
        queue.put(("batch", "error", str(error)))


def _query_worker(root: str, label: str, queue) -> None:
    try:
        result = DatasetStore(Path(root)).find_by_label(label)
        queue.put(("query", "ok", result))
    except Exception as error:  # noqa: BLE001 - a query must never fail here
        queue.put(("query", "error", f"{type(error).__name__}: {error}"))


def _batch_barrier_worker(root, number, changes, barrier, queue) -> None:
    barrier.wait()
    _batch_worker(root, number, changes, queue)


def _query_barrier_worker(root, label, barrier, queue) -> None:
    barrier.wait()
    _query_worker(root, label, queue)


def _flip_worker(
    root: str,
    cat_digests: list[str],
    rounds: int,
    start_barrier,
    sampled_barrier,
    queue,
) -> None:
    """Hold each complete state while readers sample it, then flip once.

    The writer parks at ``sampled_barrier`` until the readers have queried
    the current state, and only then commits the single flip that moves to
    the next state.  Both readers are parked on the same barriers, so under
    any scheduling every reader observes every state and the counts
    alternate; the commit itself stays serialized under the workspace lock.
    After an even number of flips a final forward batch leaves dogs.
    """
    forward = _changes(cat_digests, "cat", "dog")
    backward = _changes(cat_digests, "dog", "cat")
    try:
        store = DatasetStore(Path(root))
        for round_index in range(rounds):
            # State is held here: pre (cat) on even rounds, post (dog) on
            # odd rounds.  Release readers, then wait until each sampled it.
            start_barrier.wait()
            sampled_barrier.wait()
            flip = forward if round_index % 2 == 0 else backward
            store.submit_batch({"batch": f"flip-{round_index}", "changes": flip})
        # Hold and let readers sample the final held state as well.
        start_barrier.wait()
        sampled_barrier.wait()
        # rounds is even, so that held state is pre again; finish on dogs.
        store.submit_batch({"batch": "final", "changes": forward})
        queue.put(("flip", "ok", {}))
    except Exception as error:  # noqa: BLE001
        queue.put(("flip", "error", f"{type(error).__name__}: {error}"))


def _query_loop_worker(
    root: str,
    spec: dict,
    rounds: int,
    start_barrier,
    sampled_barrier,
    queue,
) -> None:
    """Sample cat/dog once per held state, validating every snapshot.

    Between the two barriers the writer is parked on one complete state, so
    the two queries here observe a held state; the loop advances to the
    opposite state next round, guaranteeing both states and a transition
    even when scheduling is pathologically unfair.
    """
    seen = {
        "cat_three": False,
        "cat_zero": False,
        "dog_one": False,
        "dog_four": False,
        "transition": False,
    }
    last_cat = None
    last_dog = None
    try:
        store = DatasetStore(Path(root))
        # rounds flips leave rounds+1 held states to sample.
        for _ in range(rounds + 1):
            start_barrier.wait()
            cat_result = store.find_by_label("cat")
            dog_result = store.find_by_label("dog")
            sampled_barrier.wait()
            _assert_complete_state(spec, cat_result, dog_result)

            cat_count = cat_result["count"]
            dog_count = dog_result["count"]
            seen["cat_three"] |= cat_count == 3
            seen["cat_zero"] |= cat_count == 0
            seen["dog_one"] |= dog_count == 1
            seen["dog_four"] |= dog_count == 4
            if last_cat is not None and cat_count != last_cat:
                seen["transition"] = True
            if last_dog is not None and dog_count != last_dog:
                seen["transition"] = True
            last_cat, last_dog = cat_count, dog_count
        queue.put(("query-loop", "ok", seen))
    except Exception as error:  # noqa: BLE001
        queue.put(("query-loop", "error", f"{type(error).__name__}: {error}"))


def _query_freerun_worker(root: str, spec: dict, iterations: int, queue) -> None:
    """Hammer cat/dog queries with no barrier, overlapping the real commits.

    Unlike the synchronized samplers this worker is never parked relative
    to the writer, so its reads genuinely cross commit windows: that is the
    check that catches a torn read stitching old and new labels into one
    result, or an in-flight write surfacing as corruption.  It makes no
    both-states claim (scheduling may strand it on one side); every single
    snapshot must simply be one complete, internally consistent state.
    """
    try:
        store = DatasetStore(Path(root))
        cat_result = dog_result = None
        for _ in range(iterations):
            cat_result = store.find_by_label("cat")
            dog_result = store.find_by_label("dog")
            _assert_complete_state(spec, cat_result, dog_result)
        queue.put(("freerun", "ok", {}))
    except Exception as error:  # noqa: BLE001
        queue.put(
            ("freerun", "error", f"{type(error).__name__}: {error} {cat_result} {dog_result}")
        )


# ---------------------------------------------------------------------------
# Complete-state assertions
# ---------------------------------------------------------------------------


def _assert_matches_records(result: dict, expected: list[dict]) -> None:
    """count == list length, digest order, and exact per-state summaries."""
    assert result["count"] == len(result["samples"]) == len(expected), result
    digests = [sample["sha256"] for sample in result["samples"]]
    assert digests == sorted(digests), result
    assert digests == [record["sha256"] for record in expected], result
    assert result["samples"] == expected, result


def _assert_complete_state(spec: dict, cat_result: dict, dog_result: dict) -> None:
    """Validate two independent cat/dog queries against complete states.

    Each argument is checked on its own: it must be either the wholly
    pre-batch state or the wholly post-batch state, with an internally
    consistent count/list/labels.  They are deliberately NOT cross-checked
    against each other, since two separate queries may straddle a commit —
    one may see the old state and the other the new one, so a moved digest
    can appear under cat in one and under dog in the other.
    """
    _assert_one_label_complete(spec, cat_result, "cat")
    _assert_one_label_complete(spec, dog_result, "dog")


def _assert_one_label_complete(spec: dict, result: dict, label: str) -> None:
    """One query must be a wholly pre-batch or wholly post-batch snapshot.

    The boundary cases are exactly the README's: cat is three (before) or
    zero (after); dog is one (before, only the untouched sample) or four
    (after, all four).  Every count equals its list length and every
    returned record's label and full summary match that one state.
    """
    stays = spec["stays_dog"]
    assert result["label"] == label, result
    if label == "cat":
        if result["count"] == 3:
            _assert_matches_records(result, spec["before_cat"])
        else:
            assert result["count"] == 0, result
            assert result["samples"] == [], result
        return

    assert label == "dog"
    if result["count"] == 1:
        _assert_matches_records(result, [stays])
        assert result["samples"][0]["sha256"] == stays["sha256"], result
    else:
        assert result["count"] == 4, result
        _assert_matches_records(result, spec["after_dog"])
        assert stays["sha256"] in [
            sample["sha256"] for sample in result["samples"]
        ], result
    # The untouched participant is a dog in both complete states.
    assert all(sample["label"] == "dog" for sample in result["samples"]), result


def _drain(queue, count: int) -> list:
    return [queue.get(timeout=ROUND_TIMEOUT) for _ in range(count)]


class FindLabelBatchInterleaveTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "dataset"
        self.spec = _seed_workspace(self.root)
        self.digests = [c["sha256"] for c in self.spec["cats"]]

    def tearDown(self) -> None:
        self.temp.cleanup()

    # ------------------------------------------------------------------
    # A successful batch, observed by queries crossing it
    # ------------------------------------------------------------------

    def test_query_before_batch_sees_all_old_labels(self) -> None:
        _assert_matches_records(
            DatasetStore(self.root).find_by_label("cat"), self.spec["before_cat"]
        )
        _assert_matches_records(
            DatasetStore(self.root).find_by_label("dog"), [self.spec["stays_dog"]]
        )

    def test_query_after_success_sees_all_new_labels(self) -> None:
        outcome = DatasetStore(self.root).submit_batch(
            {"batch": "move", "changes": _changes(self.digests, "cat", "dog")}
        )
        self.assertEqual(outcome["status"], "applied")
        self.assertEqual(outcome["changed"], 3)

        # A new query after success must not keep returning old members.
        _assert_matches_records(
            DatasetStore(self.root).find_by_label("cat"), []
        )
        _assert_matches_records(
            DatasetStore(self.root).find_by_label("dog"), self.spec["after_dog"]
        )
        # The participant that never took part keeps label and summary.
        self.assertEqual(
            DatasetStore(self.root).lookup_label(self.spec["dog_stays"]["sha256"]),
            {"sha256": self.spec["dog_stays"]["sha256"], "label": "dog"},
        )

    def test_queries_interleaved_with_success_land_on_one_complete_state(self) -> None:
        # Each round aligns one writer committing once with six readers at a
        # barrier, so the readers contend with a real commit: a reader may
        # land before or after it, but an in-flight write must never surface
        # as an error or false corruption, and a category can never be seen
        # half-applied.  Observing both states overall is guaranteed
        # deterministically by the held-state hammer test; this test only
        # requires that every snapshot overlapping a commit is one of the
        # two complete states.
        rounds = 25
        for round_index in range(rounds):
            barrier = ctx.Barrier(7)
            queue = ctx.Queue()
            workers = [
                ctx.Process(
                    target=_batch_barrier_worker,
                    args=(
                        str(self.root),
                        f"move-{round_index}",
                        _changes(self.digests, "cat", "dog"),
                        barrier,
                        queue,
                    ),
                )
            ]
            for label in ("cat", "dog"):
                for _ in range(3):
                    workers.append(
                        ctx.Process(
                            target=_query_barrier_worker,
                            args=(str(self.root), label, barrier, queue),
                        )
                    )
            for process in workers:
                process.start()
            for process in workers:
                process.join()
                self.assertEqual(process.exitcode, 0)

            results = _drain(queue, len(workers))
            errors = [row for row in results if row[1] == "error"]
            self.assertFalse(errors, results)
            batch_rows = [row for row in results if row[0] == "batch"]
            query_rows = [row for row in results if row[0] == "query"]
            self.assertEqual(len(batch_rows), 1, results)
            self.assertEqual(batch_rows[0][2], "applied", results)
            self.assertEqual(len(query_rows), 6, results)

            for _kind, _status, payload in query_rows:
                self.assertEqual(payload["count"], len(payload["samples"]), payload)
                if payload["label"] == "cat":
                    _assert_one_label_complete(self.spec, payload, "cat")
                else:
                    _assert_one_label_complete(self.spec, payload, "dog")

            # Restore the pre-batch state so the next round starts at cat.
            self.assertEqual(
                DatasetStore(self.root).undo_batch(f"move-{round_index}")["status"],
                "undone",
            )

        # After every undo the workspace is back at the original state.
        _assert_matches_records(
            DatasetStore(self.root).find_by_label("cat"), self.spec["before_cat"]
        )
        _assert_matches_records(
            DatasetStore(self.root).find_by_label("dog"), [self.spec["stays_dog"]]
        )

    # ------------------------------------------------------------------
    # Sustained interleaving: one writer flip-flops while readers run
    # ------------------------------------------------------------------

    def test_hammer_queries_against_repeated_batches_always_complete(self) -> None:
        spec = {
            "cats": self.spec["cats"],
            "before_cat": self.spec["before_cat"],
            "after_dog": self.spec["after_dog"],
            "stays_dog": self.spec["stays_dog"],
        }
        rounds = 20  # even, so the held states end pre and the final batch dogs
        # One writer plus two synchronized samplers meet on both barriers.
        start_barrier = ctx.Barrier(3)
        sampled_barrier = ctx.Barrier(3)
        queue = ctx.Queue()
        workers = [
            ctx.Process(
                target=_flip_worker,
                args=(
                    str(self.root),
                    self.digests,
                    rounds,
                    start_barrier,
                    sampled_barrier,
                    queue,
                ),
            ),
            ctx.Process(
                target=_query_loop_worker,
                args=(
                    str(self.root), spec, rounds,
                    start_barrier, sampled_barrier, queue,
                ),
            ),
            ctx.Process(
                target=_query_loop_worker,
                args=(
                    str(self.root), spec, rounds,
                    start_barrier, sampled_barrier, queue,
                ),
            ),
            # Free runners overlap the actual commits; they carry the
            # torn-read / false-corruption atomicity guarantee.
            ctx.Process(
                target=_query_freerun_worker,
                args=(str(self.root), spec, rounds * 50, queue),
            ),
            ctx.Process(
                target=_query_freerun_worker,
                args=(str(self.root), spec, rounds * 50, queue),
            ),
        ]
        for process in workers:
            process.start()
        for process in workers:
            process.join(ROUND_TIMEOUT)
            self.assertEqual(process.exitcode, 0)

        results = _drain(queue, len(workers))
        self.assertTrue(all(row[1] == "ok" for row in results), results)
        sampler_rows = [row for row in results if row[0] == "query-loop"]
        freerun_rows = [row for row in results if row[0] == "freerun"]
        self.assertEqual(len(sampler_rows), 2)
        self.assertEqual(len(freerun_rows), 2)
        for _kind, _status, seen in sampler_rows:
            # Held-state sampling deterministically covers both states and a
            # transition, independent of process scheduling.
            self.assertTrue(seen["cat_three"] and seen["cat_zero"], seen)
            self.assertTrue(seen["dog_one"] and seen["dog_four"], seen)
            self.assertTrue(seen["transition"], seen)

        # Writer's final forward batch leaves a coherent four-dog state.
        _assert_matches_records(
            DatasetStore(self.root).find_by_label("cat"), []
        )
        _assert_matches_records(
            DatasetStore(self.root).find_by_label("dog"), self.spec["after_dog"]
        )

    # ------------------------------------------------------------------
    # Deleted/moved source images and stale split-plan labels
    # ------------------------------------------------------------------

    def test_interleaved_query_works_after_sources_are_deleted(self) -> None:
        for record in self.spec["before_cat"] + [self.spec["stays_dog"]]:
            Path(record["source"]).unlink()

        barrier = ctx.Barrier(9)
        queue = ctx.Queue()
        workers = [
            ctx.Process(
                target=_batch_barrier_worker,
                args=(
                    str(self.root),
                    "move",
                    _changes(self.digests, "cat", "dog"),
                    barrier,
                    queue,
                ),
            )
        ]
        for label in ("cat", "dog"):
            for _ in range(4):
                workers.append(
                    ctx.Process(
                        target=_query_barrier_worker,
                        args=(str(self.root), label, barrier, queue),
                    )
                )
        for process in workers:
            process.start()
        for process in workers:
            process.join()
            self.assertEqual(process.exitcode, 0)

        results = _drain(queue, len(workers))
        self.assertFalse([row for row in results if row[1] == "error"], results)
        for _kind, _status, payload in results:
            if not isinstance(payload, dict):
                continue
            self.assertEqual(payload["count"], len(payload["samples"]), payload)
            if payload["label"] == "cat":
                self.assertIn(payload["count"], (0, 3), payload)
            else:
                self.assertIn(payload["count"], (1, 4), payload)
            # The recorded source/size survive even though the file is gone.
            for sample in payload["samples"]:
                self.assertFalse(Path(sample["source"]).exists())
                self.assertGreaterEqual(sample["size"], 0)

        _assert_matches_records(
            DatasetStore(self.root).find_by_label("dog"), self.spec["after_dog"]
        )

    def test_saved_split_plan_old_labels_never_participate(self) -> None:
        # Snapshot the cat state into a saved plan before the batch.
        plan = DatasetStore(self.root).create_split(
            "baseline", 0, [1, 0, 0]
        ).plan
        self.assertEqual(plan["samples"]["distribution"], {"cat": 3, "dog": 1})

        barrier = ctx.Barrier(3)
        queue = ctx.Queue()
        workers = [
            ctx.Process(
                target=_batch_barrier_worker,
                args=(
                    str(self.root),
                    "move",
                    _changes(self.digests, "cat", "dog"),
                    barrier,
                    queue,
                ),
            ),
            ctx.Process(
                target=_query_barrier_worker,
                args=(str(self.root), "cat", barrier, queue),
            ),
            ctx.Process(
                target=_query_barrier_worker,
                args=(str(self.root), "dog", barrier, queue),
            ),
        ]
        for process in workers:
            process.start()
        for process in workers:
            process.join()
            self.assertEqual(process.exitcode, 0)
        results = _drain(queue, len(workers))
        self.assertFalse([row for row in results if row[1] == "error"], results)

        # Current records end as dogs; the plan's saved cats are not
        # matched by the current-label query and are left saved untouched.
        store = DatasetStore(self.root)
        _assert_matches_records(store.find_by_label("cat"), [])
        _assert_matches_records(store.find_by_label("dog"), self.spec["after_dog"])
        self.assertEqual(
            store.get_split("baseline")["samples"]["distribution"],
            {"cat": 3, "dog": 1},
        )

    # ------------------------------------------------------------------
    # A rejected batch changes nothing visible
    # ------------------------------------------------------------------

    def test_rejected_batch_changes_nothing_and_reports_reason(self) -> None:
        wrong = _changes(self.digests, "cat", "dog")
        # The third record declares an old label the sample does not have.
        wrong[2] = {"sha256": self.digests[2], "old": "bird", "new": "dog"}

        with self.assertRaises(ValueError) as caught:
            DatasetStore(self.root).submit_batch(
                {"batch": "reject", "changes": wrong}
            )
        # The rejection reason is still reported by the batch operation.
        self.assertIn(self.digests[2], str(caught.exception))
        self.assertIn("bird", str(caught.exception))

        store = DatasetStore(self.root)
        _assert_matches_records(store.find_by_label("cat"), self.spec["before_cat"])
        _assert_matches_records(
            store.find_by_label("dog"), [self.spec["stays_dog"]]
        )
        self.assertEqual(store.history(), [])

    def test_queries_interleaved_with_rejected_batch_keep_complete_state(self) -> None:
        wrong = _changes(self.digests, "cat", "dog")
        wrong[1] = {"sha256": self.digests[1], "old": None, "new": "dog"}

        barrier = ctx.Barrier(13)
        queue = ctx.Queue()
        workers = [
            ctx.Process(
                target=_batch_barrier_worker,
                args=(str(self.root), "reject", wrong, barrier, queue),
            )
        ]
        for label in ("cat", "dog"):
            for _ in range(6):
                workers.append(
                    ctx.Process(
                        target=_query_barrier_worker,
                        args=(str(self.root), label, barrier, queue),
                    )
                )
        for process in workers:
            process.start()
        for process in workers:
            process.join()
            self.assertEqual(process.exitcode, 0)
        results = _drain(queue, len(workers))

        # Exactly one batch result, and it is the rejection with a reason.
        batch_rows = [row for row in results if row[0] == "batch"]
        query_rows = [row for row in results if row[0] == "query"]
        self.assertEqual(len(batch_rows), 1, results)
        self.assertEqual(batch_rows[0][1], "error", results)
        self.assertIn(self.digests[1], batch_rows[0][2])
        self.assertEqual(len(query_rows), 12, results)

        # Every interleaved query is a normal, successful result on the one
        # original complete state: cats three, dogs one, never a partial
        # "some moved, some not" view and never a failed query.
        for _kind, status, payload in query_rows:
            self.assertEqual(status, "ok", payload)
            self.assertEqual(payload["count"], len(payload["samples"]), payload)
            if payload["label"] == "cat":
                _assert_matches_records(payload, self.spec["before_cat"])
            else:
                _assert_matches_records(payload, [self.spec["stays_dog"]])

        # After the failed submission nothing moved and no history exists;
        # a normal category query is still not treated as the failure.
        store = DatasetStore(self.root)
        _assert_matches_records(store.find_by_label("cat"), self.spec["before_cat"])
        _assert_matches_records(
            store.find_by_label("dog"), [self.spec["stays_dog"]]
        )
        self.assertEqual(store.history(), [])

    def test_repeated_rejected_batches_leave_state_stable_under_readers(self) -> None:
        queue = ctx.Queue()
        workers = []
        for round_index in range(10):
            wrong = _changes(self.digests, "cat", "dog")
            position = round_index % 3
            wrong[position] = {
                "sha256": self.digests[position],
                "old": "lizard",
                "new": "dog",
            }
            workers.append(
                ctx.Process(
                    target=_batch_worker,
                    args=(str(self.root), f"reject-{round_index}", wrong, queue),
                )
            )
        for index in range(6):
            workers.append(
                ctx.Process(
                    target=_query_worker,
                    args=(
                        str(self.root),
                        "cat" if index % 2 == 0 else "dog",
                        queue,
                    ),
                )
            )
        for process in workers:
            process.start()
        for process in workers:
            process.join()
            self.assertEqual(process.exitcode, 0)
        results = _drain(queue, len(workers))
        rejections = [row for row in results if row[0] == "batch" and row[1] == "error"]
        queries = [row for row in results if row[0] == "query"]
        self.assertEqual(len(rejections), 10, results)
        self.assertTrue(all("lizard" in row[2] for row in rejections), results)
        self.assertEqual(len(queries), 6, results)
        for _kind, status, payload in queries:
            self.assertEqual(status, "ok", payload)
            if payload["label"] == "cat":
                _assert_matches_records(payload, self.spec["before_cat"])
            else:
                _assert_matches_records(payload, [self.spec["stays_dog"]])

        store = DatasetStore(self.root)
        self.assertEqual(store.history(), [])
        _assert_matches_records(store.find_by_label("cat"), self.spec["before_cat"])
        _assert_matches_records(
            store.find_by_label("dog"), [self.spec["stays_dog"]]
        )


class FindLabelBatchInterleaveCliTest(unittest.TestCase):
    """The same guarantees through independent CLI processes."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "dataset"
        self.spec = _seed_workspace(self.root)
        self.digests = [c["sha256"] for c in self.spec["cats"]]
        self.batch_dir = Path(self.temp.name) / "batches"
        self.batch_dir.mkdir()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "vision_workbench", *arguments],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def _spawn_cli(self, *arguments: str) -> subprocess.Popen:
        return subprocess.Popen(
            [sys.executable, "-m", "vision_workbench", *arguments],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _write_batch(self, name: str, changes: list) -> Path:
        path = self.batch_dir / f"{name}.json"
        path.write_text(
            json.dumps({"batch": name, "changes": changes}), encoding="utf-8"
        )
        return path

    def _assert_cli_complete_state(
        self, cat: subprocess.CompletedProcess, dog: subprocess.CompletedProcess
    ) -> None:
        self.assertEqual(cat.returncode, 0, cat.stderr)
        self.assertEqual(dog.returncode, 0, dog.stderr)
        _assert_complete_state(
            self.spec, json.loads(cat.stdout), json.loads(dog.stdout)
        )

    def test_cli_queries_across_a_successful_batch(self) -> None:
        # Boundary anchor: before any commit the CLI reports the pre state.
        cat = self.run_cli("find-label", str(self.root), "cat")
        dog = self.run_cli("find-label", str(self.root), "dog")
        self.assertEqual(json.loads(cat.stdout)["count"], 3)
        self.assertEqual(json.loads(dog.stdout)["count"], 1)
        self._assert_cli_complete_state(cat, dog)

        for round_index in range(15):
            batch_path = self._write_batch(
                f"move-{round_index}", _changes(self.digests, "cat", "dog")
            )
            # Launch the batch and both category queries with overlapping
            # windows so the queries may land before, during or after it.
            batch_proc = self._spawn_cli("batch", str(self.root), str(batch_path))
            cat_proc = self._spawn_cli("find-label", str(self.root), "cat")
            dog_proc = self._spawn_cli("find-label", str(self.root), "dog")
            batch_out, batch_err = batch_proc.communicate()
            cat_out, _cat_err = cat_proc.communicate()
            dog_out, _dog_err = dog_proc.communicate()

            self.assertEqual(batch_proc.returncode, 0, batch_err)
            self.assertEqual(json.loads(batch_out)["status"], "applied")
            cat = subprocess.CompletedProcess(
                cat_proc.args, cat_proc.returncode, cat_out, ""
            )
            dog = subprocess.CompletedProcess(
                dog_proc.args, dog_proc.returncode, dog_out, ""
            )
            # Every overlapping CLI query is a complete state or a clean
            # zero-exit result — never a half-applied list or a failure.
            self.assertEqual(cat.returncode, 0, _cat_err)
            self.assertEqual(dog.returncode, 0, _dog_err)
            self._assert_cli_complete_state(cat, dog)

            # Reset so the next round again races cat -> dog.
            undo = self.run_cli("undo", str(self.root), f"move-{round_index}")
            self.assertEqual(undo.returncode, 0, undo.stderr)

        # Boundary anchor: once a batch reports applied, later CLI queries
        # must reflect the new labels, never the old category members.
        final_path = self._write_batch(
            "final", _changes(self.digests, "cat", "dog")
        )
        applied = self.run_cli("batch", str(self.root), str(final_path))
        self.assertEqual(json.loads(applied.stdout)["status"], "applied")
        cat = self.run_cli("find-label", str(self.root), "cat")
        dog = self.run_cli("find-label", str(self.root), "dog")
        self.assertEqual(json.loads(cat.stdout)["count"], 0)
        self.assertEqual(json.loads(dog.stdout)["count"], 4)
        self._assert_cli_complete_state(cat, dog)

    def test_cli_queries_across_a_rejected_batch_stay_complete(self) -> None:
        wrong = _changes(self.digests, "cat", "dog")
        wrong[0] = {"sha256": self.digests[0], "old": "bird", "new": "dog"}
        batch_path = self._write_batch("reject", wrong)

        for _ in range(10):
            batch_proc = self._spawn_cli("batch", str(self.root), str(batch_path))
            cat_proc = self._spawn_cli("find-label", str(self.root), "cat")
            dog_proc = self._spawn_cli("find-label", str(self.root), "dog")
            _batch_out, batch_err = batch_proc.communicate()
            cat_out, _ = cat_proc.communicate()
            dog_out, _ = dog_proc.communicate()

            # The batch reports the rejection with a non-zero status and a
            # reason naming the offending sample ...
            self.assertNotEqual(batch_proc.returncode, 0)
            self.assertIn(self.digests[0], batch_err)
            # ... while interleaved classification queries stay normal and
            # keep returning the original complete classification.
            cat = subprocess.CompletedProcess(
                cat_proc.args, cat_proc.returncode, cat_out, ""
            )
            dog = subprocess.CompletedProcess(
                dog_proc.args, dog_proc.returncode, dog_out, ""
            )
            self._assert_cli_complete_state(cat, dog)

        cat = self.run_cli("find-label", str(self.root), "cat")
        dog = self.run_cli("find-label", str(self.root), "dog")
        self.assertEqual(json.loads(cat.stdout)["count"], 3)
        self.assertEqual(json.loads(dog.stdout)["count"], 1)
        history = self.run_cli("history", str(self.root))
        self.assertEqual(json.loads(history.stdout), {"batches": []})


if __name__ == "__main__":
    unittest.main()
