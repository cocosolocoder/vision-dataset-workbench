from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from vision_workbench.batches import BatchError
from vision_workbench.manifest import ManifestError, validate_manifest
from vision_workbench.store import DatasetStore

REPO_ROOT = Path(__file__).resolve().parents[1]

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def record(
    digest: str = DIGEST_A,
    source: str = "/data/a.jpg",
    size: int = 1,
    label: Any = "cat",
    **extra: Any,
) -> dict[str, Any]:
    item = {"sha256": digest, "source": source, "size": size, "label": label}
    item.update(extra)
    return item


def manifest(*items: Any) -> dict[str, Any]:
    return {"schema_version": 1, "items": list(items)}


class ValidateManifestTest(unittest.TestCase):
    def assert_invalid(self, data: Any, *fragments: str) -> None:
        with self.assertRaises(ManifestError) as caught:
            validate_manifest(data)
        message = str(caught.exception)
        for fragment in fragments:
            self.assertIn(fragment, message)

    def test_empty_list_and_zero_byte_samples_are_legal(self) -> None:
        self.assertEqual(validate_manifest(manifest())["items"], [])
        validate_manifest(manifest(record(size=0, label=None)))

    def test_null_and_empty_labels_are_legal_unlabeled(self) -> None:
        validate_manifest(manifest(record(label=None), record(digest=DIGEST_B, label="")))

    def test_extra_fields_are_kept(self) -> None:
        data = manifest(record(rev=3, note="old"), )
        data["created_by"] = "legacy"
        self.assertIs(validate_manifest(data), data)
        self.assertEqual(data["items"][0]["rev"], 3)

    def test_top_level_structure(self) -> None:
        self.assert_invalid([1, 2], "object")
        self.assert_invalid("x", "object")
        for version in (True, 1.0, "1", None, 2):
            self.assert_invalid({"schema_version": version, "items": []})
        self.assert_invalid({"schema_version": 1}, "items")
        self.assert_invalid({"schema_version": 1, "items": {}}, "array")

    def test_each_record_must_be_an_object(self) -> None:
        self.assert_invalid(manifest(42), "sample #1", "object")

    def test_digest_must_be_full_lowercase_hex(self) -> None:
        bad_digests = ("a" * 63, "A" * 64, "g" * 64, "a" * 63 + "A", 123, None)
        for position, digest in enumerate(bad_digests, start=1):
            self.assert_invalid(
                manifest(record(digest=digest)), f"sample #1", "sha256"
            )

    def test_source_must_be_a_non_empty_string(self) -> None:
        self.assert_invalid(
            manifest({"sha256": DIGEST_A, "size": 1, "label": None}), "source"
        )
        self.assert_invalid(manifest(record(source="")), "source")
        self.assert_invalid(manifest(record(source=7)), "source")

    def test_size_must_be_a_non_negative_integer(self) -> None:
        self.assert_invalid(
            manifest({"sha256": DIGEST_A, "source": "/a", "label": None}), "size"
        )
        for size in (True, False, "1", 1.0, -1, None):
            self.assert_invalid(manifest(record(size=size)), "sample #1", "size")

    def test_label_must_be_string_or_null_and_present(self) -> None:
        self.assert_invalid(
            manifest({"sha256": DIGEST_A, "source": "/a", "size": 1}), "label"
        )
        for label in (1, True, [], {}):
            self.assert_invalid(manifest(record(label=label)), "label")

    def test_duplicate_digest_names_digest_and_both_positions(self) -> None:
        # Identical source and label do not make a duplicate registration legal.
        self.assert_invalid(
            manifest(record(), record(digest=DIGEST_B), record(digest=DIGEST_A, source="/c")),
            DIGEST_A,
            "#1",
            "#3",
        )

    def test_valid_records_before_a_bad_one_do_not_mask_it(self) -> None:
        data = manifest(
            record(),
            record(digest=DIGEST_B, source="/b"),
            record(digest="c" * 64, source="/c", size=True),
        )
        self.assert_invalid(data, "sample #3", "size")


class CorruptManifestStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "dataset"
        store = DatasetStore(self.root)
        store.initialize()
        self.manifest_path = store.manifest_path

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_manifest(self, data: Any) -> None:
        self.manifest_path.write_text(json.dumps(data), encoding="utf-8")

    def assert_operation_refused(self, operation: Any) -> None:
        with self.assertRaises(ManifestError):
            operation()

    def test_queries_refuse_a_corrupt_manifest(self) -> None:
        self.write_manifest(manifest(record(size=True)))
        store = DatasetStore(self.root)
        self.assert_operation_refused(store.summary)
        # A query that would not even match the bad record still refuses:
        # the whole list is validated before any sample is used.
        self.assert_operation_refused(lambda: store.find_by_label("nothing-matches"))
        self.assert_operation_refused(lambda: store.lookup_label(DIGEST_A))
        self.assert_operation_refused(
            lambda: store.create_split("p", 1, [1, 0, 0])
        )

    def test_writers_refuse_without_touching_state(self) -> None:
        good = manifest(record())
        self.write_manifest(good)
        source = self.root.parent / "new.jpg"
        source.write_bytes(b"new-image")
        batch_file = self.root.parent / "changes.json"
        batch_file.write_text(
            json.dumps(
                {"batch": "b1", "changes": [
                    {"sha256": DIGEST_A, "old": "cat", "new": "dog"}]}
            ),
            encoding="utf-8",
        )
        for corruption in (
            manifest(record(size=True)),
            manifest(record(), record(digest=DIGEST_A, source="/other")),
        ):
            self.write_manifest(corruption)
            before = self.manifest_path.read_bytes()
            store = DatasetStore(self.root)
            with self.assertRaises(ManifestError):
                store.add(source, "x")
            with self.assertRaises(ManifestError):
                store.submit_batch_file(batch_file)
            self.assertEqual(self.manifest_path.read_bytes(), before)

    def test_initialize_refuses_to_rebuild_a_corrupt_manifest(self) -> None:
        bad = manifest(record(label=5))
        self.write_manifest(bad)
        before = self.manifest_path.read_bytes()
        with self.assertRaises(ManifestError):
            DatasetStore(self.root).initialize()
        self.assertEqual(self.manifest_path.read_bytes(), before)

    def test_corrupt_prepared_transaction_never_replaces_committed_state(self) -> None:
        store = DatasetStore(self.root)
        history = {"schema_version": 1, "batches": []}
        committed = manifest(record())
        self.write_manifest(committed)
        committed_bytes = self.manifest_path.read_bytes()
        # A crashed peer's journal carries an unusable manifest (bool size).
        store._write_json_atomic(
            store.transaction_path,
            {"manifest": manifest(record(size=True)), "batches": history},
        )
        # Opening the workspace finishes a prepared transaction; a broken
        # one is refused at that point instead of being installed.
        with self.assertRaises(BatchError):
            DatasetStore(self.root)
        # Recovery neither installs the broken payload nor drops the journal.
        self.assertEqual(self.manifest_path.read_bytes(), committed_bytes)
        self.assertTrue(store.transaction_path.exists())


class CorruptManifestCliTest(unittest.TestCase):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "vision_workbench", *arguments],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "dataset"
        result = self.run_cli("init", str(self.root))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.manifest_path = self.root / ".vision-workbench" / "manifest.json"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def assert_command_refused(self, *arguments: str) -> str:
        before = self.manifest_path.read_bytes()
        result = self.run_cli(*arguments)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout, "")
        self.assertNotIn("Traceback", result.stderr)
        self.assertIn("manifest", result.stderr.lower())
        self.assertEqual(self.manifest_path.read_bytes(), before)
        return result.stderr

    def test_every_manifest_dependent_command_refuses_cleanly(self) -> None:
        self.manifest_path.write_text(
            json.dumps(manifest(record(size=True))), encoding="utf-8"
        )
        self.assert_command_refused("summary", str(self.root))
        self.assert_command_refused("find-label", str(self.root), "cat")
        self.assert_command_refused("find-label", str(self.root), "no-such-label")
        self.assert_command_refused("label", str(self.root), DIGEST_A)
        self.assert_command_refused("undo", str(self.root), "b1")
        self.assert_command_refused(
            "split", "create", str(self.root), "p",
            "--seed", "1", "--train", "1", "--validation", "0", "--test", "0",
        )

    def test_duplicate_message_names_digest_and_both_positions(self) -> None:
        self.manifest_path.write_text(
            json.dumps(
                manifest(
                    record(),
                    record(digest=DIGEST_B, source="/b", label=None),
                    record(digest=DIGEST_A, source="/c", label="dog"),
                )
            ),
            encoding="utf-8",
        )
        stderr = self.assert_command_refused("summary", str(self.root))
        self.assertIn(DIGEST_A, stderr)
        self.assertIn("#1", stderr)
        self.assertIn("#3", stderr)

    def test_invalid_json_is_a_manifest_error_not_a_traceback(self) -> None:
        self.manifest_path.write_text("{not json", encoding="utf-8")
        self.assert_command_refused("summary", str(self.root))

    def test_valid_legacy_manifest_still_works(self) -> None:
        self.manifest_path.write_text(
            json.dumps(manifest(record(rev=0, tool="legacy"))), encoding="utf-8"
        )
        result = self.run_cli("summary", str(self.root))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["items"], 1)


if __name__ == "__main__":
    unittest.main()
