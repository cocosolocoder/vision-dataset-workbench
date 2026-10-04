from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from vision_workbench.store import DatasetStore

REPO_ROOT = Path(__file__).resolve().parents[1]


class SamplesHarness(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "dataset"
        self.store = DatasetStore(self.root)
        self.store.initialize()
        self._serial = 0

    def tearDown(self) -> None:
        self.temp.cleanup()

    def add_sample(
        self,
        label: str | None,
        content: bytes | None = None,
        *,
        name: str | None = None,
    ) -> tuple[str, str, int]:
        """Register one sample; return ``(digest, recorded source, size)``."""
        self._serial += 1
        file_name = name or f"sample-{self._serial}.jpg"
        path = self.root.parent / file_name
        data = content if content is not None else f"content-{self._serial}".encode()
        path.write_bytes(data)
        result = self.store.add(path, label)
        self.assertTrue(result.added)
        digest = hashlib.sha256(data).hexdigest()
        self.assertEqual(result.digest, digest)
        return digest, str(path.resolve()), len(data)

    def submit(self, number: str, *changes: tuple) -> None:
        self.store.submit_batch(
            {
                "batch": number,
                "changes": [
                    {"sha256": digest, "old": old, "new": new}
                    for digest, old, new in changes
                ],
            }
        )

    def samples(self, label: str) -> dict:
        return self.store.samples_by_label(label)


class SamplesByLabelTest(SamplesHarness):
    def test_exact_match_returns_full_records(self) -> None:
        d1, source1, size1 = self.add_sample("cat", b"cat-one")
        d2, _source2, _size2 = self.add_sample("dog", b"dog-one")
        d3, source3, size3 = self.add_sample("cat", b"cat-two")

        result = self.samples("cat")
        self.assertEqual(result["count"], 2)
        self.assertEqual(len(result["samples"]), 2)
        # Ordered by full digest ascending.
        self.assertEqual(
            [sample["sha256"] for sample in result["samples"]],
            sorted([d1, d3]),
        )
        by_digest = {sample["sha256"]: sample for sample in result["samples"]}
        for digest, source, size in [(d1, source1, size1), (d3, source3, size3)]:
            record = by_digest[digest]
            self.assertEqual(record["sha256"], digest)
            self.assertEqual(len(record["sha256"]), 64)
            self.assertEqual(record["source"], source)
            self.assertEqual(record["size"], size)
            self.assertEqual(record["label"], "cat")
        # The other category is not mixed in.
        self.assertNotIn(d2, by_digest)

    def test_case_whitespace_and_separators_are_literal(self) -> None:
        variants = ["cat", "Cat", "CAT", "cat ", " cat", " cats/cat", "cats\\cat"]
        for label in variants:
            self.add_sample(label, f"content-{label}".encode())

        for label in variants:
            result = self.samples(label)
            self.assertEqual(result["count"], 1, label)
            self.assertEqual(result["samples"][0]["label"], label)

        # Plain "cat" matches exactly one — no folding into variants.
        self.assertEqual(self.samples("cat")["samples"][0]["sha256"],
                         self.samples("cat")["samples"][0]["sha256"])
        # A separator-bearing label is text, not a path: querying with the
        # other separator or a prefix finds nothing.
        self.assertEqual(self.samples("cats")["count"], 0)
        self.assertEqual(self.samples("cats/cat/")["count"], 0)
        self.assertEqual(self.samples("cats/cat ")["count"], 0)

    def test_non_ascii_labels_preserved(self) -> None:
        digest, _source, _size = self.add_sample("猫")
        self.add_sample("狗")
        result = self.samples("猫")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["samples"][0]["sha256"], digest)
        self.assertEqual(result["samples"][0]["label"], "猫")

    def test_empty_string_lists_null_and_empty_as_null(self) -> None:
        d_null, _, _ = self.add_sample(None, b"unlabeled-null")
        d_empty, _, _ = self.add_sample("", b"unlabeled-empty")
        d_literal, _, _ = self.add_sample("unlabeled", b"literal-class")
        d_cat, _, _ = self.add_sample("cat", b"labeled-cat")

        result = self.samples("")
        self.assertEqual(result["count"], 2)
        self.assertEqual(
            sorted(sample["sha256"] for sample in result["samples"]),
            sorted([d_null, d_empty]),
        )
        for sample in result["samples"]:
            self.assertIsNone(sample["label"])
        self.assertEqual(
            {sample["sha256"] for sample in result["samples"]},
            {d_null, d_empty},
        )
        # Neither the literal "unlabeled" class nor a normal category leaks in.
        self.assertNotIn(d_literal, {s["sha256"] for s in result["samples"]})
        self.assertNotIn(d_cat, {s["sha256"] for s in result["samples"]})

    def test_literal_unlabeled_is_its_own_category(self) -> None:
        d_literal, _, _ = self.add_sample("unlabeled", b"literal")
        self.add_sample(None, b"none")
        self.add_sample("", b"empty")

        result = self.samples("unlabeled")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["samples"][0]["sha256"], d_literal)
        self.assertEqual(result["samples"][0]["label"], "unlabeled")

    def test_count_equals_list_length_and_list_is_sorted(self) -> None:
        digests = [self.add_sample("cat", f"c{i}".encode())[0] for i in range(5)]
        result = self.samples("cat")
        self.assertEqual(result["count"], len(result["samples"]))
        self.assertEqual(
            [sample["sha256"] for sample in result["samples"]], sorted(digests)
        )

    def test_duplicate_content_listed_once_with_first_source(self) -> None:
        first_path = self.root.parent / "first.jpg"
        first_path.write_bytes(b"same-bytes")
        first_result = self.store.add(first_path, "cat")
        self.assertTrue(first_result.added)

        second_path = self.root.parent / "second.jpg"
        second_path.write_bytes(b"same-bytes")
        second_result = self.store.add(second_path, "dog")
        self.assertFalse(second_result.added)

        result = self.samples("cat")
        self.assertEqual(result["count"], 1)
        record = result["samples"][0]
        self.assertEqual(record["sha256"], first_result.digest)
        self.assertEqual(record["source"], str(first_path.resolve()))
        # The duplicate path is never registered, so nothing appears under dog.
        self.assertEqual(self.samples("dog")["count"], 0)

    def test_batch_changes_and_undos_are_reflected(self) -> None:
        d1, _, _ = self.add_sample("cat", b"one")
        d2, _, _ = self.add_sample("cat", b"two")

        self.submit("b1", (d1, "cat", "kitten"), (d2, "cat", None))

        self.assertEqual(self.samples("cat")["count"], 0)
        kitten = self.samples("kitten")
        self.assertEqual(kitten["count"], 1)
        self.assertEqual(kitten["samples"][0]["sha256"], d1)

        unlabeled = self.samples("")
        self.assertEqual(unlabeled["count"], 1)
        self.assertEqual(unlabeled["samples"][0]["sha256"], d2)
        self.assertIsNone(unlabeled["samples"][0]["label"])

        self.store.undo_batch("b1")
        cats = self.samples("cat")
        self.assertEqual(cats["count"], 2)
        self.assertEqual(
            sorted(sample["sha256"] for sample in cats["samples"]),
            sorted([d1, d2]),
        )
        self.assertEqual(self.samples("kitten")["count"], 0)
        self.assertEqual(self.samples("")["count"], 0)

    def test_query_works_after_source_is_deleted(self) -> None:
        digest, source, size = self.add_sample("cat", b"will-vanish")
        recorded = Path(source)
        recorded.unlink()
        self.assertFalse(recorded.exists())

        result = self.samples("cat")
        self.assertEqual(result["count"], 1)
        record = result["samples"][0]
        self.assertEqual(record["sha256"], digest)
        self.assertEqual(record["source"], source)
        self.assertEqual(record["size"], size)
        self.assertEqual(record["label"], "cat")

    def test_split_plan_labels_are_ignored_and_untouched(self) -> None:
        digest, _, _ = self.add_sample("cat", b"split-me")
        self.store.create_split("baseline", 0, [1, 0, 0])

        self.submit("b1", (digest, "cat", "dog"))

        self.assertEqual(self.samples("cat")["count"], 0)
        dogs = self.samples("dog")
        self.assertEqual(dogs["count"], 1)
        self.assertEqual(dogs["samples"][0]["sha256"], digest)

        # The saved plan keeps its snapshot label and is not rewritten.
        plan = self.store.get_split("baseline")
        self.assertEqual(
            plan["sets"]["train"]["members"][0]["label"], "cat"
        )

        # Querying never modifies anything: re-reading yields the same plan.
        self.samples("cat")
        self.samples("dog")
        self.assertEqual(self.store.get_split("baseline"), plan)

    def test_no_match_returns_empty_result(self) -> None:
        self.add_sample("cat", b"only-cat")
        result = self.samples("fish")
        self.assertEqual(result, {"count": 0, "samples": []})

    def test_empty_workspace_returns_zero(self) -> None:
        result = self.samples("cat")
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["samples"], [])
        # The unlabeled query on an initialized empty workspace is empty too.
        self.assertEqual(self.samples("")["count"], 0)

    def test_uninitialized_workspace_query_creates_nothing(self) -> None:
        fresh_root = Path(self.temp.name) / "never-initialized"
        store = DatasetStore(fresh_root)
        self.assertFalse(fresh_root.exists())

        result = store.samples_by_label("cat")
        self.assertEqual(result, {"count": 0, "samples": []})
        # A read-only query materializes no workspace state.
        self.assertFalse(fresh_root.exists())

    def test_query_does_not_trim_or_rewrite_labels(self) -> None:
        d_spaced, _, _ = self.add_sample(" Cat ", b"spaced")
        result = self.samples(" Cat ")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["samples"][0]["sha256"], d_spaced)
        self.assertEqual(result["samples"][0]["label"], " Cat ")
        # Trimming would (wrongly) make this match; it must not.
        self.assertEqual(self.samples("Cat")["count"], 0)


class SamplesCliTest(SamplesHarness):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "vision_workbench", *arguments],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_cli_lists_category_and_unlabeled(self) -> None:
        d_cat, source, size = self.add_sample("猫", b"chinese")
        self.add_sample(None, b"no-label")
        self.add_sample("unlabeled", b"literal")

        category = self.run_cli("samples", str(self.root), "猫")
        self.assertEqual(category.returncode, 0, category.stderr)
        body = json.loads(category.stdout)
        self.assertEqual(body["count"], 1)
        record = body["samples"][0]
        self.assertEqual(record["sha256"], d_cat)
        self.assertEqual(record["source"], source)
        self.assertEqual(record["size"], size)
        self.assertEqual(record["label"], "猫")

        unlabeled = self.run_cli("samples", str(self.root), "")
        self.assertEqual(unlabeled.returncode, 0, unlabeled.stderr)
        body = json.loads(unlabeled.stdout)
        self.assertEqual(body["count"], 1)
        self.assertIsNone(body["samples"][0]["label"])

        literal = self.run_cli("samples", str(self.root), "unlabeled")
        self.assertEqual(literal.returncode, 0, literal.stderr)
        body = json.loads(literal.stdout)
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["samples"][0]["label"], "unlabeled")

    def test_cli_empty_workspace_is_not_an_error(self) -> None:
        fresh = Path(self.temp.name) / "fresh"
        result = self.run_cli("samples", str(fresh), "cat")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout), {"count": 0, "samples": []}
        )

    def test_cli_exact_match_with_trailing_space_argument(self) -> None:
        self.add_sample("cat", b"plain")
        self.add_sample("cat ", b"spaced")

        plain = self.run_cli("samples", str(self.root), "cat")
        self.assertEqual(json.loads(plain.stdout)["count"], 1)

        spaced = self.run_cli("samples", str(self.root), "cat ")
        body = json.loads(spaced.stdout)
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["samples"][0]["label"], "cat ")


if __name__ == "__main__":
    unittest.main()
