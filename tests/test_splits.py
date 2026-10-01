from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from vision_workbench.splits import SETS, SplitError, SplitStore
from vision_workbench.store import DatasetStore

REPO_ROOT = Path(__file__).resolve().parent.parent


def record(digest: str, label=None, source: str | None = None) -> dict:
    return {
        "sha256": digest,
        "label": label,
        "source": source or f"/data/{digest}.jpg",
    }


def make_records(spec: list[tuple[str, int]]) -> list[dict]:
    records = []
    counter = 0
    for label, count in spec:
        for _ in range(count):
            digest = hashlib.sha256(f"{label}-{counter}".encode()).hexdigest()
            records.append(record(digest, label))
            counter += 1
    return records


def check_balance(test: unittest.TestCase, scheme: dict, records: list[dict], ratios: dict) -> None:
    total = len(records)
    # Stratification treats None and the empty string as one unlabeled stratum,
    # separate from any real class literally named "unlabeled".
    def stratum(label):
        return "unlabeled" if label is None or label == "" else label
    by_stratum: dict[str, list[dict]] = {}
    for item in scheme["items"]:
        by_stratum.setdefault(stratum(item["label"]), []).append(item)
    for set_name in SETS:
        count = sum(1 for item in scheme["items"] if item["set"] == set_name)
        test.assertLess(abs(count - ratios[set_name] * total), 1.0)
    for label, members in by_stratum.items():
        for set_name in SETS:
            count = sum(1 for item in members if item["set"] == set_name)
            test.assertLess(abs(count - ratios[set_name] * len(members)), 1.0)


class SplitBalanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.state = Path(self._directory.name) / ".vision-workbench"
        self.store = SplitStore(self.state)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_balances_totals_and_strata(self) -> None:
        records = make_records([("cat", 10), ("dog", 7), ("bird", 3), (None, 5), ("unlabeled", 4)])
        ratios = {"train": 0.5, "validation": 0.25, "test": 0.25}
        scheme = self.store.create(records, "main", 42, ratios)
        check_balance(self, scheme, records, ratios)
        self.assertEqual(scheme["counts"], {"train": 15, "validation": 7, "test": 7})

    def test_one_sample_per_class(self) -> None:
        records = make_records([("cat", 1), ("dog", 1), (None, 1)])
        ratios = {"train": 0.5, "validation": 0.25, "test": 0.25}
        scheme = self.store.create(records, "tiny", 7, ratios)
        check_balance(self, scheme, records, ratios)
        self.assertEqual(sum(scheme["counts"].values()), 3)

    def test_zero_ratio_set_stays_empty(self) -> None:
        records = make_records([("cat", 4), ("dog", 2)])
        ratios = {"train": 0.5, "validation": 0.5, "test": 0.0}
        scheme = self.store.create(records, "no-test", 1, ratios)
        self.assertEqual(scheme["counts"]["test"], 0)
        self.assertEqual(scheme["labels"]["test"], {})
        check_balance(self, scheme, records, ratios)

    def test_empty_dataset(self) -> None:
        scheme = self.store.create([], "empty", 0, {"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3})
        self.assertEqual(scheme["counts"], {"train": 0, "validation": 0, "test": 0})
        self.assertEqual(scheme["items"], [])
        shown = self.store.show("empty")
        self.assertEqual(shown["counts"], {"train": 0, "validation": 0, "test": 0})

    def test_many_random_configurations(self) -> None:
        rng = random.Random(1234)
        for trial in range(30):
            class_count = rng.randint(1, 6)
            spec = []
            for _ in range(class_count):
                label = rng.choice(["cat", "dog", "bird", "car", None, "unlabeled", ""])
                spec.append((label, rng.randint(1, 12)))
            records = make_records(spec)
            train = rng.choice([0.2, 0.25, 0.3, 0.5, 0.6, 0.8])
            validation = rng.choice([0.1, 0.2, 0.25, 0.3])
            test = round(1.0 - train - validation, 10)
            if test < 0 or test > 1:
                continue
            ratios = {"train": train, "validation": validation, "test": test}
            scheme = self.store.create(records, f"trial-{trial}", rng.randint(0, 1000), ratios)
            check_balance(self, scheme, records, ratios)
            self.assertEqual(sum(scheme["counts"].values()), len(records))
            assignments = {item["sha256"]: item["set"] for item in scheme["items"]}
            self.assertEqual(len(assignments), len(records))


class SplitDeterminismTest(unittest.TestCase):
    def _records(self) -> list[dict]:
        return make_records([("cat", 6), ("dog", 4), (None, 3), ("unlabeled", 2)])

    def test_independent_of_order_and_path(self) -> None:
        records = self._records()
        ratios = {"train": 0.6, "validation": 0.2, "test": 0.2}
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_store = SplitStore(Path(first) / ".vision-workbench")
            second_store = SplitStore(Path(second) / ".vision-workbench")
            forward = first_store.create(records, "s", 99, ratios)
            reversed_records = list(reversed(records))
            shuffled = random.Random(5).sample(reversed_records, len(reversed_records))
            backward = second_store.create(shuffled, "s", 99, ratios)
            self.assertEqual(
                {item["sha256"]: item["set"] for item in forward["items"]},
                {item["sha256"]: item["set"] for item in backward["items"]},
            )

    def test_stable_across_restarts(self) -> None:
        records = self._records()
        ratios = {"train": 0.5, "validation": 0.25, "test": 0.25}
        with tempfile.TemporaryDirectory() as directory:
            store = SplitStore(Path(directory) / ".vision-workbench")
            first = store.create(records, "s", 123, ratios)
            second = store.show("s")
            self.assertEqual(
                {item["sha256"]: item["set"] for item in first["items"]},
                {item["sha256"]: item["set"] for item in second["items"]},
            )
            self.assertEqual(
                [item["sha256"] for item in second["items"]],
                sorted(item["sha256"] for item in second["items"]),
            )

    def test_seed_changes_assignment(self) -> None:
        records = make_records([("cat", 5), ("dog", 5)])
        ratios = {"train": 0.5, "validation": 0.25, "test": 0.25}
        with tempfile.TemporaryDirectory() as directory:
            store = SplitStore(Path(directory) / ".vision-workbench")
            first = store.create(records, "a", 1, ratios)
            second = store.create(records, "b", 2, ratios)
            self.assertNotEqual(
                {item["sha256"]: item["set"] for item in first["items"]},
                {item["sha256"]: item["set"] for item in second["items"]},
            )

    def test_unlabeled_separated_from_literal_unlabeled(self) -> None:
        records = make_records([(None, 4), ("", 3), ("unlabeled", 5)])
        ratios = {"train": 0.5, "validation": 0.25, "test": 0.25}
        with tempfile.TemporaryDirectory() as directory:
            store = SplitStore(Path(directory) / ".vision-workbench")
            scheme = store.create(records, "u", 11, ratios)
            check_balance(self, scheme, records, ratios)
            # None and "" form one unlabeled stratum of 7; the real class
            # literally named "unlabeled" forms its own stratum of 5.
            unlabeled = [it for it in scheme["items"] if it["label"] is None or it["label"] == ""]
            real = [it for it in scheme["items"] if it["label"] == "unlabeled"]
            self.assertEqual(len(unlabeled), 7)
            self.assertEqual(len(real), 5)
            for group, size in [(unlabeled, 7), (real, 5)]:
                counts = {s: sum(1 for it in group if it["set"] == s) for s in SETS}
                for s in SETS:
                    self.assertLess(abs(counts[s] - ratios[s] * size), 1.0)


class SplitPersistenceTest(unittest.TestCase):
    def test_scheme_survives_source_deletion_and_modification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            source = root / "image.jpg"
            source.write_bytes(b"original")
            store = DatasetStore(workspace)
            store.add(source, "cat")
            split_store = SplitStore(store.state_directory)
            created = split_store.create(store.items(), "keep", 5, {"train": 0.5, "validation": 0.25, "test": 0.25})

            source.unlink()
            source.write_bytes(b"completely different contents")
            shown = split_store.show("keep")
            self.assertEqual(created["counts"], shown["counts"])
            self.assertEqual(
                {item["sha256"]: item["set"] for item in created["items"]},
                {item["sha256"]: item["set"] for item in shown["items"]},
            )

    def test_new_imports_do_not_change_existing_scheme(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            store = DatasetStore(workspace)
            first_source = root / "a.jpg"
            first_source.write_bytes(b"aaa")
            store.add(first_source, "cat")
            split_store = SplitStore(store.state_directory)
            split_store.create(store.items(), "fixed", 2, {"train": 0.5, "validation": 0.25, "test": 0.25})

            second_source = root / "b.jpg"
            second_source.write_bytes(b"bbb")
            store.add(second_source, "dog")
            shown = split_store.show("fixed")
            self.assertEqual(sum(shown["counts"].values()), 1)
            self.assertEqual(shown["labels"], {"train": {"cat": 1}, "validation": {}, "test": {}})

    def test_old_workspace_without_splits_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            store = DatasetStore(workspace)
            source = root / "a.jpg"
            source.write_bytes(b"aaa")
            store.add(source, "cat")
            split_store = SplitStore(store.state_directory)
            scheme = split_store.create(store.items(), "retro", 3, {"train": 1.0, "validation": 0.0, "test": 0.0})
            self.assertEqual(scheme["counts"], {"train": 1, "validation": 0, "test": 0})


class SplitNameTest(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.store = SplitStore(Path(self._directory.name) / ".vision-workbench")
        self.records = make_records([("cat", 4)])

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_same_name_same_input_returns_existing(self) -> None:
        ratios = {"train": 0.5, "validation": 0.25, "test": 0.25}
        first = self.store.create(self.records, "same", 8, ratios)
        second = self.store.create(self.records, "same", 8, ratios)
        self.assertEqual(first, second)

    def test_same_name_different_records_conflicts(self) -> None:
        ratios = {"train": 0.5, "validation": 0.25, "test": 0.25}
        self.store.create(self.records, "clash", 8, ratios)
        with self.assertRaises(SplitError):
            self.store.create(self.records[:2], "clash", 8, ratios)
        shown = self.store.show("clash")
        self.assertEqual(sum(shown["counts"].values()), 4)

    def test_same_name_different_seed_conflicts(self) -> None:
        ratios = {"train": 0.5, "validation": 0.25, "test": 0.25}
        self.store.create(self.records, "clash", 8, ratios)
        with self.assertRaises(SplitError):
            self.store.create(self.records, "clash", 9, ratios)

    def test_same_name_different_ratios_conflicts(self) -> None:
        self.store.create(self.records, "clash", 8, {"train": 0.5, "validation": 0.25, "test": 0.25})
        with self.assertRaises(SplitError):
            self.store.create(self.records, "clash", 8, {"train": 0.6, "validation": 0.2, "test": 0.2})

    def test_invalid_names_rejected(self) -> None:
        ratios = {"train": 0.5, "validation": 0.25, "test": 0.25}
        for name in ["", "   ", "a/b", "a\\b", ".", "..", "a\x00b"]:
            with self.assertRaises(SplitError, msg=repr(name)):
                self.store.create(self.records, name, 1, ratios)
            with self.assertRaises(SplitError, msg=repr(name)):
                self.store.show(name)

    def test_show_missing_scheme_fails(self) -> None:
        with self.assertRaises(SplitError):
            self.store.show("missing")


class SplitValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.store = SplitStore(Path(self._directory.name) / ".vision-workbench")

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_invalid_ratios(self) -> None:
        records = make_records([("cat", 4)])
        good = {"train": 0.5, "validation": 0.25, "test": 0.25}
        bad_ratios = [
            {"train": 0.5, "validation": 0.25, "test": 0.2},
            {"train": 1.1, "validation": 0.0, "test": 0.0},
            {"train": -0.1, "validation": 0.6, "test": 0.5},
            {"train": 0.5, "validation": 0.5, "test": float("nan")},
            {"train": 0.5, "validation": 0.5, "test": float("inf")},
            {"train": 0.5, "validation": 0.5},
        ]
        for ratios in bad_ratios:
            with self.assertRaises(SplitError, msg=str(ratios)):
                self.store.create(records, "bad", 1, ratios)
        self.store.create(records, "good", 1, good)

    def test_duplicate_identity_fails_without_side_effects(self) -> None:
        digest = hashlib.sha256(b"x").hexdigest()
        records = [record(digest, "cat"), record(digest, "dog")]
        with self.assertRaises(SplitError):
            self.store.create(records, "dup", 1, {"train": 0.5, "validation": 0.25, "test": 0.25})
        self.assertFalse((self.store.splits_directory / "dup.json").exists())

    def test_record_missing_fields_fails(self) -> None:
        digest = hashlib.sha256(b"y").hexdigest()
        with self.assertRaises(SplitError):
            self.store.create([{"sha256": digest}], "missing-source", 1, {"train": 0.5, "validation": 0.25, "test": 0.25})
        with self.assertRaises(SplitError):
            self.store.create([{"label": "cat", "source": "/x"}], "missing-digest", 1, {"train": 0.5, "validation": 0.25, "test": 0.25})

    def test_interrupted_write_never_shows_partial_scheme(self) -> None:
        ratios = {"train": 0.5, "validation": 0.25, "test": 0.25}
        records = make_records([("cat", 4)])
        # A leftover temporary file from an interrupted write must never be
        # mistaken for a scheme; the scheme itself is simply absent.
        self.store.splits_directory.mkdir(parents=True, exist_ok=True)
        temporary = self.store.splits_directory / "partial.json.tmp"
        temporary.write_text('{"schema_version": 1, "name": "partial", "items": [', encoding="utf-8")
        with self.assertRaises(SplitError):
            self.store.show("partial")
        self.store.create(records, "partial", 1, ratios)
        self.assertTrue((self.store.splits_directory / "partial.json").exists())
        self.assertEqual(sum(self.store.show("partial")["counts"].values()), 4)

    def test_non_integer_seed_fails(self) -> None:
        records = make_records([("cat", 2)])
        with self.assertRaises(SplitError):
            self.store.create(records, "seed", 1.5, {"train": 0.5, "validation": 0.25, "test": 0.25})


class SplitCliTest(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "vision_workbench", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT)},
        )

    def test_create_and_show_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            image = Path(directory) / "image.jpg"
            image.write_bytes(b"image-bytes")
            init = self._run("init", str(workspace))
            self.assertEqual(init.returncode, 0, init.stderr)
            add = self._run("add", str(workspace), str(image), "--label", "cat")
            self.assertEqual(add.returncode, 0, add.stderr)
            created = self._run(
                "split", "create", str(workspace),
                "--name", "main", "--seed", "42",
                "--train", "0.5", "--validation", "0.25", "--test", "0.25",
            )
            self.assertEqual(created.returncode, 0, created.stderr)
            payload = json.loads(created.stdout)
            self.assertEqual(payload["name"], "main")
            self.assertEqual(payload["counts"], {"train": 1, "validation": 0, "test": 0})

            shown = self._run("split", "show", str(workspace), "--name", "main")
            self.assertEqual(shown.returncode, 0, shown.stderr)
            payload = json.loads(shown.stdout)
            self.assertEqual(payload["seed"], 42)
            self.assertEqual(payload["ratios"], {"train": 0.5, "validation": 0.25, "test": 0.25})
            self.assertEqual(len(payload["items"]), 1)

    def test_invalid_ratio_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            self._run("init", str(workspace))
            result = self._run(
                "split", "create", str(workspace),
                "--name", "bad", "--seed", "1",
                "--train", "0.5", "--validation", "0.5", "--test", "0.5",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("error", result.stderr)
            self.assertFalse((workspace / ".vision-workbench" / "splits" / "bad.json").exists())

    def test_show_missing_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            self._run("init", str(workspace))
            result = self._run("split", "show", str(workspace), "--name", "ghost")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("error", result.stderr)

    def test_original_commands_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            image = Path(directory) / "image.jpg"
            image.write_bytes(b"image-bytes")
            self.assertEqual(self._run("init", str(workspace)).returncode, 0)
            self.assertEqual(self._run("add", str(workspace), str(image)).returncode, 0)
            summary = self._run("summary", str(workspace))
            self.assertEqual(summary.returncode, 0)
            self.assertEqual(json.loads(summary.stdout)["items"], 1)
            demo = self._run("demo", "--workspace", str(workspace))
            self.assertEqual(demo.returncode, 0)
            self.assertIn("视觉数据集工作台", demo.stdout)


if __name__ == "__main__":
    unittest.main()
