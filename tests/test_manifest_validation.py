"""Whole-registration-list integrity gating.

Before any feature uses a single sample the complete manifest must
prove usable: an object whose ``schema_version`` is the integer 1 and
whose ``items`` list holds only well-formed, uniquely registered
samples.  These tests pin both the validation rules and the contract
that a damaged manifest rejects every manifest-dependent operation
(cleanly, on standard error, with non-zero status) without changing the
registered data, the label history or a saved split plan — while
features that depend only on saved plans or on the history keep their
existing behavior.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from vision_workbench.manifest import ManifestError, validate_manifest
from vision_workbench.store import DatasetStore

REPO_ROOT = Path(__file__).resolve().parents[1]

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def item(
    digest: str = DIGEST_A,
    *,
    source: str = "/data/a.jpg",
    size: int = 10,
    label: str | None = "cat",
    **extra: object,
) -> dict:
    record = {"sha256": digest, "source": source, "size": size, "label": label}
    record.update(extra)
    return record


def manifest(*items: object) -> dict:
    return {"schema_version": 1, "items": list(items)}


# ---------------------------------------------------------------------------
# Pure validation rules
# ---------------------------------------------------------------------------


class ValidateManifestTest(unittest.TestCase):
    def test_empty_list_and_zero_byte_and_unlabeled_records_are_valid(self) -> None:
        for data in (
            manifest(),
            manifest(item(size=0, label=None)),
            manifest(item(label=""), item(digest=DIGEST_B, label="unlabeled")),
        ):
            self.assertIs(validate_manifest(data), data)

    def test_optional_revision_may_be_missing_and_extra_fields_are_kept(self) -> None:
        data = manifest(item(rev=3, note="keep me"))
        self.assertIs(validate_manifest(data), data)
        self.assertEqual(data["items"][0]["note"], "keep me")
        self.assertEqual(data["items"][0]["rev"], 3)

    def test_payload_must_be_an_object(self) -> None:
        for bad in ([], "manifest", None, 1):
            with self.assertRaises(ManifestError):
                validate_manifest(bad)

    def test_schema_version_must_be_the_integer_1(self) -> None:
        for bad in (None, True, False, 1.0, "1", 0, 2):
            with self.assertRaises(ManifestError) as caught:
                validate_manifest(manifest() | {"schema_version": bad})
            self.assertIn("schema version", str(caught.exception))

        with self.assertRaises(ManifestError) as caught:
            validate_manifest({"items": []})
        self.assertIn("schema version", str(caught.exception))

    def test_items_must_be_present_and_be_a_list(self) -> None:
        with self.assertRaises(ManifestError) as caught:
            validate_manifest({"schema_version": 1})
        self.assertIn("items", str(caught.exception))
        for bad in ({}, "items", None, 3):
            with self.assertRaises(ManifestError):
                validate_manifest({"schema_version": 1, "items": bad})

    def test_sample_must_be_an_object(self) -> None:
        with self.assertRaises(ManifestError) as caught:
            validate_manifest(manifest(item(), ["not", "an", "object"]))
        message = str(caught.exception)
        self.assertIn("#2", message)
        self.assertIn("object", message)

    def test_digest_must_be_full_lowercase_hex(self) -> None:
        bad_digests = [
            None,
            123,
            "",
            "a" * 63,
            "a" * 65,
            "A" * 64,
            "g" * 64,
            DIGEST_A[:-1] + "A",
            " " + "a" * 63,
        ]
        for bad in bad_digests:
            with self.assertRaises(ManifestError) as caught:
                validate_manifest(manifest(item(bad)))  # type: ignore[arg-type]
            message = str(caught.exception)
            self.assertIn("sha256", message)
            self.assertIn("#1", message)

        with self.assertRaises(ManifestError) as caught:
            validate_manifest(manifest({k: v for k, v in item().items()
                                        if k != "sha256"}))
        self.assertIn("missing field 'sha256'", str(caught.exception))

    def test_source_must_be_a_non_empty_string(self) -> None:
        for bad in (None, 1, "", b"/data/a.jpg"):
            with self.assertRaises(ManifestError) as caught:
                validate_manifest(manifest(item(source=bad)))  # type: ignore[arg-type]
            self.assertIn("source", str(caught.exception))

        with self.assertRaises(ManifestError) as caught:
            validate_manifest(manifest({k: v for k, v in item().items()
                                        if k != "source"}))
        self.assertIn("missing field 'source'", str(caught.exception))

    def test_size_must_be_a_non_negative_integer(self) -> None:
        for bad in (True, False, 1.0, 1.5, "1", "-1", None, -1):
            with self.assertRaises(ManifestError) as caught:
                validate_manifest(manifest(item(size=bad)))  # type: ignore[arg-type]
            message = str(caught.exception)
            self.assertIn("size", message)
            self.assertIn("#1", message)

        with self.assertRaises(ManifestError) as caught:
            validate_manifest(manifest({k: v for k, v in item().items()
                                        if k != "size"}))
        self.assertIn("missing field 'size'", str(caught.exception))

    def test_label_must_be_string_or_null(self) -> None:
        for bad in (1, 1.0, True, [], {}):
            with self.assertRaises(ManifestError) as caught:
                validate_manifest(manifest(item(label=bad)))  # type: ignore[arg-type]
            self.assertIn("label", str(caught.exception))

        with self.assertRaises(ManifestError) as caught:
            validate_manifest(manifest({k: v for k, v in item().items()
                                        if k != "label"}))
        self.assertIn("missing field 'label'", str(caught.exception))

    def test_duplicate_digest_names_digest_and_both_positions(self) -> None:
        # Even two byte-for-byte identical records are a failure: the
        # manifest must not silently merge or pick one of them.
        data = manifest(item(), item(DIGEST_B), dict(item()))
        with self.assertRaises(ManifestError) as caught:
            validate_manifest(data)
        message = str(caught.exception)
        self.assertIn(DIGEST_A, message)
        self.assertIn("#1", message)
        self.assertIn("#3", message)

    def test_duplicate_with_different_source_and_label_still_rejected(self) -> None:
        data = manifest(
            item(source="/one.jpg", label="cat"),
            item(DIGEST_B),
            item(source="/two.jpg", label="dog"),
        )
        with self.assertRaises(ManifestError) as caught:
            validate_manifest(data)
        message = str(caught.exception)
        self.assertIn(DIGEST_A, message)
        self.assertIn("#1", message)
        self.assertIn("#3", message)

    def test_error_after_valid_prefix_still_rejects_the_whole_list(self) -> None:
        data = manifest(item(label="cat"), item(size=True))
        with self.assertRaises(ManifestError) as caught:
            validate_manifest(data)
        self.assertIn("#2", str(caught.exception))


# ---------------------------------------------------------------------------
# Store-level gating
# ---------------------------------------------------------------------------


class ManifestHarness(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / "dataset"
        self.store = DatasetStore(self.root)
        self.store.initialize()
        self.serial = 0

    def tearDown(self) -> None:
        self.temp.cleanup()

    @property
    def manifest_path(self) -> Path:
        return self.root / ".vision-workbench" / "manifest.json"

    @property
    def history_path(self) -> Path:
        return self.root / ".vision-workbench" / "batches.json"

    def write_manifest(self, data: object) -> None:
        self.manifest_path.write_text(
            json.dumps(data), encoding="utf-8"
        )

    def add_real_sample(self, label: str | None = "cat") -> str:
        self.serial += 1
        source = self.base / f"sample-{self.serial}.jpg"
        source.write_bytes(f"content-{self.serial}".encode())
        result = self.store.add(source, label)
        self.assertTrue(result.added)
        return result.digest

    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "vision_workbench", *arguments],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )


class CorruptManifestRefusalTest(ManifestHarness):
    def corrupt(self, data: object) -> None:
        self.write_manifest(data)

    def test_every_manifest_dependent_operation_refuses(self) -> None:
        digest = self.add_real_sample("cat")
        # A valid record followed by a structurally damaged one: the
        # damage is later in the list and no query below needs record 2.
        self.corrupt(manifest(item(digest, source="/x.jpg", label="cat"),
                              item(DIGEST_B, label=123)))  # type: ignore[arg-type]

        with self.assertRaises(ManifestError):
            self.store.summary()
        with self.assertRaises(ManifestError):
            # Even the matching-label query, which would never touch the
            # damaged record, must be refused.
            self.store.find_by_label("cat")
        with self.assertRaises(ManifestError):
            self.store.lookup_label(digest)
        with self.assertRaises(ManifestError):
            self.store.create_split("p", 0, [1, 0, 0])
        with self.assertRaises(ManifestError):
            self.store.undo_batch("b1")
        with self.assertRaises(ManifestError):
            self.store.submit_batch(
                {
                    "batch": "b9",
                    "changes": [{"sha256": digest, "old": "cat", "new": "dog"}],
                }
            )

        source = self.base / "new.jpg"
        source.write_bytes(b"new-content")
        with self.assertRaises(ManifestError):
            self.store.add(source, "dog")

        directory = self.base / "images"
        directory.mkdir()
        (directory / "pic.jpg").write_bytes(b"pic")
        with self.assertRaises(ManifestError):
            self.store.import_directory(directory)

    def test_duplicate_records_cannot_be_relabelled_one_at_a_time(self) -> None:
        digest = self.add_real_sample("cat")
        self.corrupt(manifest(
            item(digest, source="/one.jpg", label="cat"),
            item(digest, source="/two.jpg", label="dog"),
        ))
        with self.assertRaises(ManifestError):
            self.store.submit_batch(
                {
                    "batch": "b9",
                    "changes": [{"sha256": digest, "old": "cat", "new": "bird"}],
                }
            )
        reopened = DatasetStore(self.root)
        with self.assertRaises(ManifestError):
            reopened.find_by_label("bird")

    def test_refusal_changes_nothing(self) -> None:
        digest = self.add_real_sample("cat")
        self.store.submit_batch(
            {
                "batch": "b1",
                "changes": [{"sha256": digest, "old": "cat", "new": "dog"}],
            }
        )
        self.store.create_split("keep", 0, [1, 0, 0])
        plan_path = self.root / ".vision-workbench" / "splits" / "keep.json"
        plan_before = plan_path.read_text(encoding="utf-8")

        self.corrupt(manifest(item(digest, label="dog", rev=1),
                              item(DIGEST_B, size="12")))  # type: ignore[arg-type]
        manifest_before = self.manifest_path.read_text(encoding="utf-8")
        history_before = self.history_path.read_text(encoding="utf-8")
        txn_path = self.root / ".vision-workbench" / ".txn.json"

        for operation in (
            lambda: self.store.summary(),
            lambda: self.store.find_by_label("dog"),
            lambda: self.store.lookup_label(digest),
            lambda: self.store.create_split("p2", 0, [1, 0, 0]),
            lambda: self.store.undo_batch("b1"),
        ):
            with self.assertRaises(ManifestError):
                operation()

        # No repair, no default fill-in, no rebuilt empty manifest.
        self.assertEqual(self.manifest_path.read_text(encoding="utf-8"),
                         manifest_before)
        self.assertEqual(self.history_path.read_text(encoding="utf-8"),
                         history_before)
        self.assertEqual(plan_path.read_text(encoding="utf-8"), plan_before)
        self.assertFalse(txn_path.exists())

    def test_unparseable_manifest_is_rejected_as_manifest_problem(self) -> None:
        self.manifest_path.write_text("{broken json", encoding="utf-8")
        with self.assertRaises(ManifestError):
            self.store.summary()

    def test_extra_fields_survive_later_commits(self) -> None:
        digest = self.add_real_sample("cat")
        data = manifest(item(digest, source="/kept.jpg", size=12, label="cat",
                             note="preserve-me"))
        self.write_manifest(data)

        source = self.base / "second.jpg"
        source.write_bytes(b"second-content")
        self.assertTrue(self.store.add(source, None).added)

        reopened = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        by_digest = {row["sha256"]: row for row in reopened["items"]}
        self.assertEqual(by_digest[digest]["note"], "preserve-me")
        self.assertEqual(by_digest[digest]["source"], "/kept.jpg")

    def test_legacy_record_without_revision_supports_batches_and_undo(self) -> None:
        digest = self.add_real_sample("cat")
        data = manifest(item(digest, label="cat"))
        self.write_manifest(data)

        result = self.store.submit_batch(
            {
                "batch": "b1",
                "changes": [{"sha256": digest, "old": "cat", "new": "dog"}],
            }
        )
        self.assertEqual(result["status"], "applied")
        self.assertEqual(self.store.undo_batch("b1")["status"], "undone")
        self.assertEqual(self.store.lookup_label(digest)["label"], "cat")


class PlanOnlyOperationsKeepWorkingTest(ManifestHarness):
    def test_history_split_show_and_export_ignore_manifest_damage(self) -> None:
        digest = self.add_real_sample("cat")
        self.store.submit_batch(
            {
                "batch": "b1",
                "changes": [{"sha256": digest, "old": "cat", "new": "dog"}],
            }
        )
        self.store.create_split("baseline", 1, [1, 0, 0])
        target = self.base / "out.zip"

        # Damage the registration list after the plan and history exist.
        self.write_manifest(manifest(item(digest, label="dog", rev=1),
                                     item(DIGEST_B, size=False)))  # type: ignore[arg-type]

        # History reads only batches.json.
        self.assertEqual([entry["batch"] for entry in self.store.history()], ["b1"])
        # A saved plan is read whole on its own.
        self.assertEqual(self.store.get_split("baseline")["name"], "baseline")
        # Export depends only on the saved plan and its recorded sources.
        from vision_workbench.exporter import export_split

        result = export_split(self.store, "baseline", target)
        self.assertEqual(result["exported"], 1)
        self.assertTrue(target.exists())


class CorruptManifestCliTest(ManifestHarness):
    def setUp(self) -> None:
        super().setUp()
        self.digest = self.add_real_sample("cat")

    def assert_clean_manifest_failure(self, result: subprocess.CompletedProcess) -> None:
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn("manifest", result.stderr.lower())
        self.assertNotIn("Traceback", result.stderr)

    def test_summary_fails_cleanly(self) -> None:
        self.write_manifest(manifest(item(self.digest), item(DIGEST_B, size=1.5)))
        self.assert_clean_manifest_failure(self.run_cli("summary", str(self.root)))

    def test_find_label_fails_cleanly_even_for_a_matching_label(self) -> None:
        self.write_manifest(manifest(
            item(self.digest, label="cat"), item(DIGEST_B, source="")
        ))
        self.assert_clean_manifest_failure(
            self.run_cli("find-label", str(self.root), "cat")
        )

    def test_label_batch_and_split_create_fail_cleanly(self) -> None:
        self.write_manifest(manifest(
            item(self.digest),
            item(DIGEST_B),
            item(DIGEST_B),
        ))
        batch_file = self.base / "changes.json"
        batch_file.write_text(json.dumps({
            "batch": "b9",
            "changes": [{"sha256": self.digest, "old": "cat", "new": "dog"}],
        }), encoding="utf-8")
        self.assert_clean_manifest_failure(
            self.run_cli("label", str(self.root), self.digest)
        )
        self.assert_clean_manifest_failure(
            self.run_cli("batch", str(self.root), str(batch_file))
        )
        self.assert_clean_manifest_failure(
            self.run_cli("split", "create", str(self.root), "p",
                         "--seed", "1", "--train", "1",
                         "--validation", "0", "--test", "0")
        )

    def test_add_fails_cleanly_and_leaves_the_manifest_untouched(self) -> None:
        self.write_manifest(manifest(item(self.digest), item(DIGEST_B, label=9)))
        before = self.manifest_path.read_text(encoding="utf-8")
        source = self.base / "new.jpg"
        source.write_bytes(b"new")
        result = self.run_cli("add", str(self.root), str(source), "--label", "x")
        self.assert_clean_manifest_failure(result)
        self.assertEqual(self.manifest_path.read_text(encoding="utf-8"), before)
        self.assertFalse(
            (self.root / ".vision-workbench" / ".txn.json").exists()
        )

    def test_split_show_history_and_export_still_succeed(self) -> None:
        self.store.create_split("baseline", 1, [1, 0, 0])
        self.store.submit_batch(
            {
                "batch": "b1",
                "changes": [{"sha256": self.digest, "old": "cat", "new": "dog"}],
            }
        )
        self.write_manifest(manifest(item(self.digest, label="dog", rev=1),
                                     item(DIGEST_B, size=True)))

        show = self.run_cli("split", "show", str(self.root), "baseline")
        self.assertEqual(show.returncode, 0, show.stderr)
        history = self.run_cli("history", str(self.root))
        self.assertEqual(history.returncode, 0, history.stderr)

        target = self.base / "pkg.zip"
        export = self.run_cli(
            "export", str(self.root), "baseline", str(target)
        )
        self.assertEqual(export.returncode, 0, export.stderr)
        self.assertTrue(target.exists())


class CorruptJournalRecoveryTest(ManifestHarness):
    def test_corrupt_prepared_manifest_is_not_installed(self) -> None:
        digest = self.add_real_sample("cat")
        good_history = {
            "schema_version": 1,
            "batches": [{
                "batch": "b1",
                "records": [{
                    "sha256": digest, "old": "cat", "new": "dog",
                    "changed": True, "rev": 1,
                }],
                "changed_count": 1,
                "undone": False,
                "undone_at": None,
            }],
        }
        journal = {
            "manifest": manifest(item(digest, label="dog", rev=1),
                                 item(DIGEST_B, size="12")),
            "batches": good_history,
        }
        manifest_before = self.manifest_path.read_text(encoding="utf-8")
        txn_path = self.root / ".vision-workbench" / ".txn.json"
        self.store._write_json_atomic(txn_path, journal)

        with self.assertRaises(ManifestError):
            DatasetStore(self.root)
        # Nothing was installed; the damaged journal stays for inspection.
        self.assertTrue(txn_path.exists())
        self.assertEqual(
            self.manifest_path.read_text(encoding="utf-8"), manifest_before
        )


if __name__ == "__main__":
    unittest.main()
