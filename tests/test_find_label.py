from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from vision_workbench.batches import BatchError
from vision_workbench.store import DatasetStore

REPO_ROOT = Path(__file__).resolve().parents[1]


class FindLabelHarness(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "dataset"
        self.store = DatasetStore(self.root)
        self.store.initialize()
        self._serial = 0

    def tearDown(self) -> None:
        self.temp.cleanup()

    def add_sample(
        self, label: str | None, content: bytes | None = None
    ) -> tuple[str, str, int]:
        """Add a sample; return (digest, recorded source, size)."""
        self._serial += 1
        path = self.root.parent / f"sample-{self._serial}.jpg"
        data = content if content is not None else f"content-{self._serial}".encode()
        path.write_bytes(data)
        result = self.store.add(path, label)
        self.assertTrue(result.added)
        return result.digest, str(path.resolve()), len(data)


class FindLabelStoreTest(FindLabelHarness):
    def test_matches_exact_label_with_full_summaries_sorted_by_digest(self) -> None:
        entries = [self.add_sample("cat") for _ in range(3)]
        self.add_sample("dog")

        result = self.store.find_by_label("cat")

        self.assertEqual(result["label"], "cat")
        self.assertEqual(result["count"], 3)
        self.assertEqual(len(result["samples"]), 3)
        digests = [sample["sha256"] for sample in result["samples"]]
        self.assertEqual(digests, sorted(digests))
        by_digest = {sample["sha256"]: sample for sample in result["samples"]}
        for digest, source, size in entries:
            self.assertEqual(
                by_digest[digest],
                {"sha256": digest, "source": source, "size": size, "label": "cat"},
            )

    def test_matching_is_case_space_and_separator_sensitive(self) -> None:
        target = self.add_sample("cat")[0]
        self.add_sample("Cat")
        self.add_sample("cat ")
        self.add_sample("cat/dog")
        self.add_sample("猫")

        result = self.store.find_by_label("cat")

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["samples"][0]["sha256"], target)

        # Nearby labels are each their own exact category.
        self.assertEqual(self.store.find_by_label("Cat")["count"], 1)
        self.assertEqual(self.store.find_by_label("cat ")["count"], 1)
        self.assertEqual(self.store.find_by_label("cat/dog")["count"], 1)
        chinese = self.store.find_by_label("猫")
        self.assertEqual(chinese["count"], 1)
        self.assertEqual(chinese["samples"][0]["label"], "猫")

    def test_empty_string_queries_unlabeled_and_reports_null(self) -> None:
        d_none, _, _ = self.add_sample(None)
        d_empty, _, _ = self.add_sample("")
        self.add_sample("cat")
        self.add_sample("unlabeled")

        result = self.store.find_by_label("")

        self.assertEqual(result["label"], "")
        self.assertEqual(result["count"], 2)
        self.assertEqual(
            sorted(sample["sha256"] for sample in result["samples"]),
            sorted([d_none, d_empty]),
        )
        self.assertTrue(all(sample["label"] is None for sample in result["samples"]))

    def test_literal_unlabeled_class_is_separate_from_unlabeled(self) -> None:
        self.add_sample(None)
        self.add_sample("")
        literal = self.add_sample("unlabeled")

        result = self.store.find_by_label("unlabeled")

        self.assertEqual(result["count"], 1)
        sample = result["samples"][0]
        self.assertEqual(sample["sha256"], literal[0])
        self.assertEqual(sample["label"], "unlabeled")

        # Summary keeps its existing merged display; the query does not.
        self.assertEqual(
            self.store.summary()["labels"], {"unlabeled": 3}
        )

    def test_empty_workspace_and_missing_label_return_zero_records(self) -> None:
        with tempfile.TemporaryDirectory() as empty:
            fresh = DatasetStore(Path(empty) / "dataset")
            self.assertEqual(
                fresh.find_by_label("cat"),
                {"label": "cat", "count": 0, "samples": []},
            )
            self.assertEqual(
                fresh.find_by_label(""),
                {"label": "", "count": 0, "samples": []},
            )

        self.add_sample("dog")
        self.assertEqual(
            self.store.find_by_label("cat"),
            {"label": "cat", "count": 0, "samples": []},
        )

    def test_duplicate_content_listed_once_with_first_source(self) -> None:
        self._serial += 1
        first = self.root.parent / "first.jpg"
        first.write_bytes(b"identical-content")
        second = self.root.parent / "second.jpg"
        second.write_bytes(b"identical-content")

        first_result = self.store.add(first, "cat")
        second_result = self.store.add(second, "dog")
        self.assertTrue(first_result.added)
        self.assertFalse(second_result.added)

        result = self.store.find_by_label("cat")
        self.assertEqual(result["count"], 1)
        sample = result["samples"][0]
        self.assertEqual(sample["sha256"], first_result.digest)
        self.assertEqual(sample["source"], str(first.resolve()))
        self.assertEqual(sample["label"], "cat")

    def test_batch_change_and_undo_are_reflected(self) -> None:
        moved, _, _ = self.add_sample("cat")
        cleared, _, _ = self.add_sample("dog")

        self.assertEqual(self.store.find_by_label("cat")["count"], 1)
        self.store.submit_batch(
            {
                "batch": "b1",
                "changes": [{"sha256": moved, "old": "cat", "new": "dog"}],
            }
        )
        self.assertEqual(self.store.find_by_label("cat")["count"], 0)
        dogs = self.store.find_by_label("dog")
        self.assertEqual(dogs["count"], 2)
        self.assertEqual({s["sha256"] for s in dogs["samples"]}, {moved, cleared})

        # Clearing a label moves the sample into the unlabeled query.
        self.store.submit_batch(
            {
                "batch": "b2",
                "changes": [{"sha256": cleared, "old": "dog", "new": None}],
            }
        )
        unlabeled = self.store.find_by_label("")
        self.assertEqual([s["sha256"] for s in unlabeled["samples"]], [cleared])
        self.assertTrue(all(s["label"] is None for s in unlabeled["samples"]))
        self.assertEqual(self.store.find_by_label("dog")["count"], 1)

        # Undos are reflected by the next query as well.
        self.store.undo_batch("b2")
        self.assertEqual(self.store.find_by_label("")["count"], 0)
        self.assertEqual(self.store.find_by_label("dog")["count"], 2)
        self.store.undo_batch("b1")
        cats = self.store.find_by_label("cat")
        self.assertEqual(cats["count"], 1)
        self.assertEqual(cats["samples"][0]["sha256"], moved)
        self.assertEqual(self.store.find_by_label("dog")["count"], 1)

    def test_old_label_saved_in_split_plan_does_not_match(self) -> None:
        digest, _, _ = self.add_sample("cat")
        plan = self.store.create_split("p", 0, [1, 0, 0]).plan
        self.store.submit_batch(
            {
                "batch": "b1",
                "changes": [{"sha256": digest, "old": "cat", "new": "dog"}],
            }
        )

        # The query follows the current label only...
        self.assertEqual(self.store.find_by_label("cat")["count"], 0)
        self.assertEqual(self.store.find_by_label("dog")["count"], 1)
        # ...and the saved plan still holds the old label untouched.
        reopened = self.store.get_split("p")
        self.assertEqual(reopened, plan)
        self.assertEqual(
            reopened["sets"]["train"]["members"][0]["label"], "cat"
        )

    def test_moved_or_deleted_source_is_still_returned(self) -> None:
        digest, source, size = self.add_sample("cat")
        Path(source).unlink()
        self.assertFalse(Path(source).exists())

        result = self.store.find_by_label("cat")
        self.assertEqual(result["count"], 1)
        self.assertEqual(
            result["samples"][0],
            {"sha256": digest, "source": source, "size": size, "label": "cat"},
        )

    def test_non_string_query_label_rejected(self) -> None:
        with self.assertRaises(BatchError):
            self.store.find_by_label(None)  # type: ignore[arg-type]


class FindLabelCliTest(FindLabelHarness):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "vision_workbench", *arguments],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_cli_returns_readable_json(self) -> None:
        digest, source, size = self.add_sample("猫")
        self.add_sample(None)

        result = self.run_cli("find-label", str(self.root), "猫")
        self.assertEqual(result.returncode, 0, result.stderr)
        # Chinese text is emitted directly, not \u-escaped.
        self.assertIn("猫", result.stdout)
        body = json.loads(result.stdout)
        self.assertEqual(body["count"], 1)
        self.assertEqual(
            body["samples"][0],
            {"sha256": digest, "source": source, "size": size, "label": "猫"},
        )

    def test_cli_empty_string_queries_unlabeled_with_null_labels(self) -> None:
        self.add_sample(None)
        result = self.run_cli("find-label", str(self.root), "")
        self.assertEqual(result.returncode, 0, result.stderr)
        body = json.loads(result.stdout)
        self.assertEqual(body["label"], "")
        self.assertEqual(body["count"], 1)
        self.assertIsNone(body["samples"][0]["label"])

    def test_cli_no_matches_is_not_an_error(self) -> None:
        result = self.run_cli("find-label", str(self.root), "ghost")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout),
            {"label": "ghost", "count": 0, "samples": []},
        )


if __name__ == "__main__":
    unittest.main()
