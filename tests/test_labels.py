from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from vision_workbench.batches import BatchError
from vision_workbench.store import MANIFEST_SCHEMA_VERSION, DatasetStore

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

    def add_sample(self, label: str | None = None, content: bytes | None = None) -> str:
        self._serial += 1
        path = self.root.parent / f"sample-{self._serial}.jpg"
        path.write_bytes(
            content if content is not None else f"content-{self._serial}".encode()
        )
        result = self.store.add(path, label)
        self.assertTrue(result.added)
        return result.digest

    def batch(self, batch_id: str, changes: list[dict]) -> dict:
        return {"batch_id": batch_id, "changes": changes}

    @staticmethod
    def change(digest: str, old_label, new_label) -> dict:
        return {"sha256": digest, "old_label": old_label, "new_label": new_label}

    def raw_manifest(self) -> dict:
        return json.loads(self.store.manifest_path.read_text(encoding="utf-8"))


class ApplyBatchTest(StoreHarness):
    def test_set_change_and_clear_labels(self) -> None:
        d1 = self.add_sample(None)
        d2 = self.add_sample("cat")
        d3 = self.add_sample("dog")
        before = {
            item["sha256"]: item for item in self.raw_manifest()["items"]
        }

        result = self.store.apply_batch(
            self.batch(
                "b1",
                [
                    self.change(d1, None, "猫"),          # set, Chinese label
                    self.change(d2, "cat", "unlabeled"),  # change to literal class
                    self.change(d3, "dog", None),          # clear
                ],
            )
        )
        self.assertEqual(result.status, "applied")
        self.assertEqual(result.changed, 3)
        self.assertEqual(result.total, 3)
        self.assertTrue(all(c["changed"] for c in result.changes))

        self.assertEqual(self.store.get_label(d1), "猫")
        self.assertEqual(self.store.get_label(d2), "unlabeled")
        self.assertIsNone(self.store.get_label(d3))

        # Summary category counts follow, content metadata does not move.
        labels = self.store.summary()["labels"]
        self.assertEqual(labels, {"unlabeled": 2, "猫": 1})
        after = {item["sha256"]: item for item in self.raw_manifest()["items"]}
        for digest in (d1, d2, d3):
            self.assertEqual(after[digest]["size"], before[digest]["size"])
            self.assertEqual(after[digest]["source"], before[digest]["source"])

    def test_empty_string_and_null_both_unlabeled(self) -> None:
        d1 = self.add_sample(None)
        d2 = self.add_sample("")
        result = self.store.apply_batch(
            self.batch(
                "b",
                [
                    self.change(d1, "", "x"),
                    self.change(d2, None, "x"),
                ],
            )
        )
        self.assertEqual(result.status, "applied")
        self.assertEqual(self.store.get_label(d1), "x")
        self.assertEqual(self.store.get_label(d2), "x")

    def test_unchanged_records_and_all_unchanged_batch(self) -> None:
        d1 = self.add_sample("cat")
        d2 = self.add_sample("dog")
        result = self.store.apply_batch(
            self.batch(
                "b",
                [
                    self.change(d1, "cat", "dog"),
                    self.change(d2, "dog", "dog"),
                ],
            )
        )
        self.assertEqual(result.status, "applied")
        self.assertEqual(result.changed, 1)
        flags = {c["sha256"]: c["changed"] for c in result.changes}
        self.assertTrue(flags[d1])
        self.assertFalse(flags[d2])

        # Whole batch that changes nothing: reported, no history, id free.
        truly_noop = self.store.apply_batch(
            self.batch("zero", [self.change(d1, "dog", "dog")])
        )
        self.assertEqual(truly_noop.status, "unchanged")
        self.assertEqual(truly_noop.changed, 0)
        self.assertEqual(self.store.batch_history()[-1]["batch_id"], "b")
        # The id was not consumed: it can be reused for a real change.
        real = self.store.apply_batch(
            self.batch("zero", [self.change(d1, "dog", "cat")])
        )
        self.assertEqual(real.status, "applied")

    def test_history_records_submission_order_and_shapes(self) -> None:
        d1 = self.add_sample("a")
        d2 = self.add_sample("b")
        self.store.apply_batch(
            self.batch("h1", [self.change(d1, "a", "a1")])
        )
        self.store.apply_batch(
            self.batch(
                "h2", [self.change(d2, "b", "b1"), self.change(d1, "a1", "a2")]
            )
        )
        history = self.store.batch_history()
        self.assertEqual([h["batch_id"] for h in history], ["h1", "h2"])
        self.assertEqual(history[0]["changed_count"], 1)
        self.assertEqual(history[1]["changed_count"], 2)
        self.assertEqual(
            [(c["old_label"], c["new_label"]) for c in history[1]["changes"]],
            [("b", "b1"), ("a1", "a2")],
        )
        self.assertEqual(history[0]["undo"]["status"], "active")


class BatchValidationTest(StoreHarness):
    def assert_rejects(self, payload: dict, code: str) -> None:
        before = self.store.manifest_path.read_text(encoding="utf-8")
        with self.assertRaises(BatchError) as caught:
            self.store.apply_batch(payload)
        self.assertEqual(caught.exception.code, code)
        # Whole batch rejected: nothing on disk changes.
        self.assertEqual(
            self.store.manifest_path.read_text(encoding="utf-8"), before
        )

    def test_structural_failures(self) -> None:
        d = self.add_sample("cat")
        good = self.change(d, "cat", "dog")
        self.assert_rejects(["not", "an", "object"], "invalid_document")
        self.assert_rejects({"changes": [good]}, "invalid_batch_id")
        self.assert_rejects({"batch_id": "  ", "changes": [good]}, "invalid_batch_id")
        self.assert_rejects({"batch_id": "x"}, "invalid_document")
        self.assert_rejects(
            {"batch_id": "x", "changes": "nope"}, "invalid_document"
        )
        self.assert_rejects(
            {"batch_id": "x", "changes": [["nope"]]}, "invalid_change"
        )
        self.assert_rejects(
            {"batch_id": "x", "changes": [{"sha256": d, "new_label": "z"}]},
            "invalid_change",
        )
        self.assert_rejects(
            {"batch_id": "x", "changes": [{"sha256": "", "old_label": None,
                                          "new_label": "z"}]},
            "invalid_digest",
        )

    def test_duplicate_unknown_and_bad_label(self) -> None:
        d1 = self.add_sample("cat")
        self.assert_rejects(
            self.batch(
                "x",
                [self.change(d1, "cat", "dog"), self.change(d1, "dog", "fish")],
            ),
            "duplicate_digest",
        )
        self.assert_rejects(
            self.batch("x", [self.change("a" * 64, None, "dog")]),
            "unknown_digest",
        )
        for bad in (3, 3.0, True, ["cat"], {"x": 1}):
            self.assert_rejects(
                self.batch("x", [self.change(d1, bad, "dog")]), "invalid_label"
            )
            self.assert_rejects(
                self.batch("x", [self.change(d1, "cat", bad)]), "invalid_label"
            )

    def test_expected_label_mismatch_rejects_everything(self) -> None:
        d1 = self.add_sample("cat")
        d2 = self.add_sample("dog")
        with self.assertRaises(BatchError) as caught:
            self.store.apply_batch(
                self.batch(
                    "x",
                    [
                        self.change(d1, "cat", "fish"),
                        self.change(d2, "cat", "fish"),  # d2 is actually dog
                    ],
                )
            )
        self.assertEqual(caught.exception.code, "label_mismatch")
        self.assertEqual(caught.exception.records[0]["index"], 1)
        self.assertEqual(caught.exception.records[0]["sha256"], d2)
        self.assertEqual(self.store.get_label(d1), "cat")
        self.assertEqual(self.store.get_label(d2), "dog")
        self.assertEqual(self.store.batch_history(), [])

    def test_unlabeled_spellings_compare_equal_in_check(self) -> None:
        d = self.add_sample(None)
        # Expected "" while stored null must pass verification; then target
        # equal to current (both unlabeled) makes it a no-op record.
        result = self.store.apply_batch(
            self.batch("x", [self.change(d, "", None)])
        )
        self.assertEqual(result.status, "unchanged")

    def test_failed_batch_leaves_no_history(self) -> None:
        d = self.add_sample("cat")
        for payload in (
            self.batch("f", [self.change("z" * 64, "cat", "dog")]),
            self.batch("f", [self.change(d, "wrong", "dog")]),
        ):
            with self.assertRaises(BatchError):
                self.store.apply_batch(payload)
        self.assertEqual(self.store.batch_history(), [])


class IdempotencyTest(StoreHarness):
    def test_resubmit_same_id_and_content_returns_original(self) -> None:
        d = self.add_sample("cat")
        first = self.store.apply_batch(
            self.batch("dup", [self.change(d, "cat", "dog")])
        )
        # Someone changes the label again in between.
        self.store.apply_batch(
            self.batch("other", [self.change(d, "dog", "fish")])
        )
        again = self.store.apply_batch(
            self.batch("dup", [self.change(d, "cat", "dog")])
        )
        self.assertEqual(again.status, "duplicate")
        self.assertTrue(again.reused)
        # The original result is reported verbatim, including its count.
        self.assertEqual(again.changed, first.changed)
        # The original result is reported verbatim.
        self.assertEqual(
            [c["new_label"] for c in again.changes],
            [c["new_label"] for c in first.changes],
        )
        # The intervening label is untouched.
        self.assertEqual(self.store.get_label(d), "fish")
        self.assertEqual(
            [h["batch_id"] for h in self.store.batch_history()], ["dup", "other"]
        )

    def test_reordering_and_unlabeled_spelling_are_same_content(self) -> None:
        d1 = self.add_sample(None)
        d2 = self.add_sample("cat")
        self.store.apply_batch(
            self.batch(
                "p",
                [
                    self.change(d1, None, "x"),
                    self.change(d2, "cat", ""),
                ],
            )
        )
        self.store.undo_batch("p")
        reordered = self.batch(
            "p",
            [
                self.change(d2, "cat", None),
                self.change(d1, "", "x"),
            ],
        )
        self.assertEqual(
            self.store.apply_batch(reordered).status, "duplicate"
        )

    def test_same_id_different_content_conflicts(self) -> None:
        d1 = self.add_sample("cat")
        d2 = self.add_sample("dog")
        self.store.apply_batch(
            self.batch("p", [self.change(d1, "cat", "x")])
        )
        with self.assertRaises(BatchError) as caught:
            self.store.apply_batch(
                self.batch("p", [self.change(d2, "dog", "x")])
            )
        self.assertEqual(caught.exception.code, "batch_id_conflict")
        self.assertEqual(self.store.get_label(d2), "dog")


class UndoTest(StoreHarness):
    def test_undo_restores_all_old_labels(self) -> None:
        d1 = self.add_sample("cat")
        d2 = self.add_sample("dog")
        self.store.apply_batch(
            self.batch(
                "u",
                [
                    self.change(d1, "cat", "fish"),
                    self.change(d2, "dog", None),
                ],
            )
        )
        result = self.store.undo_batch("u")
        self.assertEqual(result.status, "undone")
        self.assertEqual(result.restored, 2)
        self.assertEqual(self.store.get_label(d1), "cat")
        self.assertEqual(self.store.get_label(d2), "dog")

        record = self.store.batch_history()[0]
        self.assertEqual(record["undo"]["status"], "undone")
        self.assertEqual(record["undo"]["restored"], 2)

        again = self.store.undo_batch("u")
        self.assertEqual(again.status, "already_undone")
        # No new history entry is written.
        self.assertEqual(len(self.store.batch_history()), 1)

    def test_undo_blocked_when_one_sample_changed_again(self) -> None:
        d1 = self.add_sample("cat")
        d2 = self.add_sample("dog")
        self.store.apply_batch(
            self.batch(
                "u",
                [
                    self.change(d1, "cat", "fish"),
                    self.change(d2, "dog", "bird"),
                ],
            )
        )
        # Change d1 elsewhere, then change it back to the same label.
        self.store.apply_batch(
            self.batch("later", [self.change(d1, "fish", "x")])
        )
        self.store.apply_batch(
            self.batch("back", [self.change(d1, "x", "fish")])
        )
        with self.assertRaises(BatchError) as caught:
            self.store.undo_batch("u")
        self.assertEqual(caught.exception.code, "sample_modified_after_batch")
        self.assertEqual(caught.exception.records, [{"sha256": d1}])
        # Labels are left exactly as they were.
        self.assertEqual(self.store.get_label(d1), "fish")
        self.assertEqual(self.store.get_label(d2), "bird")

    def test_changes_to_other_samples_do_not_block_undo(self) -> None:
        d1 = self.add_sample("cat")
        d2 = self.add_sample("dog")
        self.store.apply_batch(
            self.batch("u", [self.change(d1, "cat", "fish")])
        )
        self.store.apply_batch(
            self.batch("other", [self.change(d2, "dog", "bird")])
        )
        result = self.store.undo_batch("u")
        self.assertEqual(result.status, "undone")
        self.assertEqual(self.store.get_label(d1), "cat")
        self.assertEqual(self.store.get_label(d2), "bird")

    def test_undo_unknown_and_unchanged_entries(self) -> None:
        with self.assertRaises(BatchError) as caught:
            self.store.undo_batch("ghost")
        self.assertEqual(caught.exception.code, "unknown_batch")

        d1 = self.add_sample("cat")
        d2 = self.add_sample("dog")
        self.store.apply_batch(
            self.batch(
                "u",
                [
                    self.change(d1, "cat", "fish"),
                    self.change(d2, "dog", "dog"),  # never changed
                ],
            )
        )
        self.store.undo_batch("u")
        self.assertEqual(self.store.get_label(d1), "cat")
        self.assertEqual(self.store.get_label(d2), "dog")


class LabelShowTest(StoreHarness):
    def test_get_label_values_and_unknown(self) -> None:
        d1 = self.add_sample(None)
        d2 = self.add_sample("猫")
        d3 = self.add_sample("unlabeled")
        self.assertIsNone(self.store.get_label(d1))
        self.assertEqual(self.store.get_label(d2), "猫")
        self.assertEqual(self.store.get_label(d3), "unlabeled")
        with self.assertRaises(BatchError) as caught:
            self.store.get_label("0" * 64)
        self.assertEqual(caught.exception.code, "unknown_digest")


class LegacyWorkspaceTest(StoreHarness):
    def test_schema_one_manifest_upgrades_without_reimport(self) -> None:
        d = self.add_sample("cat")
        # Rewind the on-disk manifest to a schema-1 workspace.
        manifest = self.raw_manifest()
        manifest["schema_version"] = 1
        for item in manifest["items"]:
            item.pop("labels_last_batch", None)
        manifest.pop("label_batches", None)
        self.store.manifest_path.write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        reopened = DatasetStore(self.root)
        self.assertEqual(reopened.get_label(d), "cat")
        result = reopened.apply_batch(
            self.batch("legacy", [self.change(d, "cat", "dog")])
        )
        self.assertEqual(result.status, "applied")
        self.assertEqual(self.raw_manifest()["schema_version"], MANIFEST_SCHEMA_VERSION)
        reopened.undo_batch("legacy")
        self.assertEqual(reopened.get_label(d), "cat")


class SplitSnapshotTest(StoreHarness):
    def test_label_changes_do_not_touch_saved_split(self) -> None:
        d = self.add_sample("cat")
        plan = self.store.create_split("snap", 1, [1, 0, 0]).plan
        split_path = self.store.splits_directory / "snap.json"
        saved = split_path.read_text(encoding="utf-8")

        self.store.apply_batch(
            self.batch("b", [self.change(d, "cat", "dog")])
        )
        self.assertEqual(split_path.read_text(encoding="utf-8"), saved)
        viewed = self.store.get_split("snap")
        self.assertEqual(viewed, plan)
        self.assertEqual(viewed["sets"]["train"]["members"][0]["label"], "cat")

        # New splits use the current label; the old name now conflicts.
        with self.assertRaises(Exception):
            self.store.create_split("snap", 1, [1, 0, 0])
        fresh = self.store.create_split("now", 1, [1, 0, 0]).plan
        self.assertEqual(fresh["samples"]["distribution"], {"dog": 1})

        # Undo also leaves the saved split alone.
        self.store.undo_batch("b")
        self.assertEqual(split_path.read_text(encoding="utf-8"), saved)


class AtomicWriteTest(StoreHarness):
    def test_failed_replace_leaves_old_manifest(self) -> None:
        d = self.add_sample("cat")
        before = self.store.manifest_path.read_text(encoding="utf-8")
        original_replace = os.replace

        def fail_replace(src, dst):  # noqa: ANN001
            raise OSError("simulated write failure")

        os.replace = fail_replace
        try:
            with self.assertRaises(OSError):
                self.store.apply_batch(
                    self.batch("x", [self.change(d, "cat", "dog")])
                )
        finally:
            os.replace = original_replace
        self.assertEqual(
            self.store.manifest_path.read_text(encoding="utf-8"), before
        )
        self.assertEqual(self.store.get_label(d), "cat")
        self.assertEqual(self.store.batch_history(), [])
        leftovers = [
            p.name
            for p in self.store.state_directory.iterdir()
            if p.name.endswith(".tmp")
        ]
        self.assertEqual(leftovers, [])

    def test_every_visible_manifest_is_consistent(self) -> None:
        digests = [self.add_sample("c") for _ in range(3)]
        for i, digest in enumerate(digests):
            self.store.apply_batch(
                self.batch(f"b{i}", [self.change(digest, "c", f"l{i}")])
            )
            manifest = self.raw_manifest()
            history = manifest["label_batches"]
            labels = {item["sha256"]: item.get("label") for item in manifest["items"]}
            # Each history entry's claimed result matches visible labels for
            # samples nobody has touched since.
            self.assertEqual(len(history), i + 1)
            self.assertEqual(labels[digest], f"l{i}")


class ConcurrencyTest(StoreHarness):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "vision_workbench", *arguments],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def _write_batch_file(self, name: str, payload: dict) -> Path:
        path = self.root.parent / name
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def test_parallel_disjoint_batches_both_succeed(self) -> None:
        digests = [self.add_sample(None) for _ in range(4)]
        files = [
            self._write_batch_file(
                f"p{i}.json",
                self.batch(f"par-{i}", [self.change(digests[i], None, f"l{i}")]),
            )
            for i in range(4)
        ]
        processes = [
            subprocess.Popen(
                [
                    sys.executable, "-m", "vision_workbench", "label", "apply",
                    str(self.root), str(path),
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for path in files
        ]
        results = [process.communicate() for process in processes]
        codes = [process.returncode for process in processes]
        self.assertEqual(codes, [0] * 4, results)
        history = self.store.batch_history()
        self.assertEqual(sorted(h["batch_id"] for h in history),
                         [f"par-{i}" for i in range(4)])
        for i, digest in enumerate(digests):
            self.assertEqual(self.store.get_label(digest), f"l{i}")

    def test_parallel_overlapping_batches_only_one_wins(self) -> None:
        d = self.add_sample("cat")
        files = [
            self._write_batch_file(
                "o1.json", self.batch("one", [self.change(d, "cat", "fish")])
            ),
            self._write_batch_file(
                "o2.json", self.batch("two", [self.change(d, "cat", "bird")])
            ),
        ]
        processes = [
            subprocess.Popen(
                [
                    sys.executable, "-m", "vision_workbench", "label", "apply",
                    str(self.root), str(path),
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for path in files
        ]
        outputs = [process.communicate() for process in processes]
        codes = sorted(process.returncode for process in processes)
        self.assertEqual(codes, [0, 1], outputs)
        failure_stderr = next(
            output[1] for process, output in zip(processes, outputs)
            if process.returncode != 0
        )
        error = json.loads(failure_stderr)
        self.assertEqual(error["code"], "label_mismatch")
        self.assertEqual(error["records"][0]["sha256"], d)
        self.assertEqual(len(self.store.batch_history()), 1)
        self.assertIn(self.store.get_label(d), {"fish", "bird"})


class LabelCliTest(StoreHarness):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "vision_workbench", *arguments],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_apply_show_history_undo_cli(self) -> None:
        d1 = self.add_sample(None)
        d2 = self.add_sample("cat")
        batch_path = self.root.parent / "batch.json"
        batch_path.write_text(
            json.dumps(
                self.batch(
                    "cli-1",
                    [
                        self.change(d1, None, "猫"),
                        self.change(d2, "cat", None),
                    ],
                ),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        applied = self.run_cli("label", "apply", str(self.root), str(batch_path))
        self.assertEqual(applied.returncode, 0, applied.stderr)
        body = json.loads(applied.stdout)
        self.assertEqual(body["batch_id"], "cli-1")
        self.assertEqual(body["status"], "applied")
        self.assertEqual(body["changed"], 2)

        show = self.run_cli("label", "show", str(self.root), d1)
        self.assertEqual(json.loads(show.stdout)["label"], "猫")
        cleared = self.run_cli("label", "show", str(self.root), d2)
        self.assertIsNone(json.loads(cleared.stdout)["label"])

        history = self.run_cli("label", "history", str(self.root))
        entries = json.loads(history.stdout)["batches"]
        self.assertEqual([e["batch_id"] for e in entries], ["cli-1"])
        self.assertEqual(entries[0]["changed_count"], 2)

        undone = self.run_cli("label", "undo", str(self.root), "cli-1")
        self.assertEqual(json.loads(undone.stdout)["status"], "undone")
        again = self.run_cli("label", "undo", str(self.root), "cli-1")
        self.assertEqual(json.loads(again.stdout)["status"], "already_undone")
        missing = self.run_cli("label", "undo", str(self.root), "nope")
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("unknown_batch", missing.stderr)

    def test_invalid_json_and_rejected_batch_report_code(self) -> None:
        bad = self.root.parent / "bad.json"
        bad.write_bytes("{不是 json".encode("utf-8"))
        result = self.run_cli("label", "apply", str(self.root), str(bad))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid_document", result.stderr)

        d = self.add_sample("cat")
        mismatch = self.root.parent / "m.json"
        mismatch.write_text(
            json.dumps(self.batch("m", [self.change(d, "dog", "fish")])),
            encoding="utf-8",
        )
        rejected = self.run_cli("label", "apply", str(self.root), str(mismatch))
        self.assertNotEqual(rejected.returncode, 0)
        error = json.loads(rejected.stderr)
        self.assertEqual(error["code"], "label_mismatch")
        self.assertEqual(error["records"][0]["sha256"], d)

        show_missing = self.run_cli("label", "show", str(self.root), "f" * 64)
        self.assertNotEqual(show_missing.returncode, 0)
        self.assertIn("unknown_digest", show_missing.stderr)


if __name__ == "__main__":
    unittest.main()
