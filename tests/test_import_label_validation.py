"""Regression tests pinning the ``label`` argument boundary of imports.

Both public Python import entry points — ``DatasetStore.add`` (one image)
and ``DatasetStore.import_directory`` (a directory) — accept only a string
or ``None`` as the import label.  ``None`` and the empty string mean
*unlabeled*; every other string is kept literally.  Numbers, booleans,
lists, dictionaries and any other object are caller-side argument errors:
they must fail with a plain :class:`ValueError` (never a manifest or
batch-history corruption error, never a media-read failure, and never a
normal "already present"/"duplicate"/zero-added result) before the source
is opened or the directory scanned, so no sample is registered and no
existing registration, batch history or saved split plan changes —
including on re-imports, all-duplicate directories and empty directories,
where the call would otherwise have had nothing new to write.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from vision_workbench import confirmed
from vision_workbench.manifest import ManifestError
from vision_workbench.store import DatasetStore


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# (invalid value, name of its concrete type) — booleans must be reported
# as "bool", never as the "int" they compare equal to.
INVALID_LABELS: list[tuple[object, str]] = [
    (0, "int"),
    (1, "int"),
    (-7, "int"),
    (1.0, "float"),
    (1.5, "float"),
    (False, "bool"),
    (True, "bool"),
    ([], "list"),
    (["cat"], "list"),
    ({}, "dict"),
    ({"cat": 1}, "dict"),
    (("cat",), "tuple"),
    (object(), "object"),
]

# The falsy values that must on no account be treated as "no label".
FALSY_INVALID_LABELS = [0, False, []]


def _state_path(workspace: Path, name: str) -> Path:
    return workspace / ".vision-workbench" / name


def _manifest_raw(workspace: Path) -> str:
    return _state_path(workspace, "manifest.json").read_text(encoding="utf-8")


def _manifest_items(workspace: Path) -> list[dict]:
    path = _state_path(workspace, "manifest.json")
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["items"]


def _history_raw(workspace: Path) -> str:
    return _state_path(workspace, "batches.json").read_text(encoding="utf-8")


def _plan_raw(workspace: Path, name: str) -> str:
    return _state_path(workspace, os.path.join("splits", f"{name}.json")).read_text(
        encoding="utf-8"
    )


class ImportLabelValidationTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.workspace = self.base / "workspace"
        self.source = self.base / "images"
        self.source.mkdir()
        self.store = DatasetStore(self.workspace)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_image(self, relative: str, content: bytes) -> Path:
        path = self.source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def _write_file(self, name: str, content: bytes) -> Path:
        path = self.base / name
        path.write_bytes(content)
        return path

    def assertInvalidLabelError(self, raised: Exception, value: object) -> None:
        """The error is a plain ValueError that names the label argument.

        It must identify the ``label`` parameter, the concrete type that
        was received and the allowed types (a string or ``None``) — not a
        corruption error and not a source/media failure.
        """
        # Exactly a built-in ValueError: a ManifestError/BatchError
        # subclass would mean stored data, not the call argument, was
        # blamed.
        self.assertIs(type(raised), ValueError, repr(raised))
        self.assertNotIsInstance(raised, ManifestError)
        message = str(raised)
        self.assertIn("label", message)
        self.assertIn(type(value).__name__, message)
        self.assertIn("string", message)
        self.assertIn("None", message)
        for media_wording in (
            "cannot read",
            "does not exist",
            "not a directory",
            "Not a regular file",
            "already present",
            "duplicate",
            "corrupt",
        ):
            self.assertNotIn(media_wording, message)


class InvalidLabelRejectedTest(ImportLabelValidationTestBase):
    """Every non-string, non-None value is refused by both entry points."""

    def test_add_rejects_every_non_string_label(self) -> None:
        source = self._write_file("a.jpg", b"image-a")
        for value, type_name in INVALID_LABELS:
            with self.subTest(value=value, type=type_name):
                with self.assertRaises(ValueError) as raised:
                    self.store.add(source, value)
                self.assertInvalidLabelError(raised.exception, value)
                self.assertIn(type_name, str(raised.exception))

    def test_import_directory_rejects_every_non_string_label(self) -> None:
        self._write_image("a.jpg", b"image-a")
        for value, type_name in INVALID_LABELS:
            with self.subTest(value=value, type=type_name):
                with self.assertRaises(ValueError) as raised:
                    self.store.import_directory(self.source, value)
                self.assertInvalidLabelError(raised.exception, value)
                self.assertIn(type_name, str(raised.exception))

    def test_falsy_values_are_not_treated_as_unlabeled(self) -> None:
        # 0, False and [] must fail instead of registering an unlabeled
        # sample or becoming a category name.
        source = self._write_file("a.jpg", b"image-a")
        for value in FALSY_INVALID_LABELS:
            with self.subTest(value=value):
                with self.assertRaises(ValueError) as raised:
                    self.store.add(source, value)
                self.assertInvalidLabelError(raised.exception, value)

                self._write_image(f"copy-{value!r}.jpg", b"image-a")
                with self.assertRaises(ValueError) as raised_dir:
                    self.store.import_directory(self.source, value)
                self.assertInvalidLabelError(raised_dir.exception, value)

        self.assertEqual(_manifest_items(self.workspace), [])
        self.assertEqual(
            DatasetStore(self.workspace).summary(),
            {"items": 0, "bytes": 0, "labels": {}},
        )

    def test_boolean_is_named_bool_not_int(self) -> None:
        source = self._write_file("a.jpg", b"image-a")
        for value in (False, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError) as raised:
                    self.store.add(source, value)
                message = str(raised.exception)
                self.assertIn("bool", message)
                # The wording must not fall back on the boolean's integer
                # identity ("got int ...").
                self.assertNotIn("int", message)

    def test_rejected_as_keyword_argument_too(self) -> None:
        # The boundary is on the parameter itself, regardless of call shape.
        source = self._write_file("a.jpg", b"image-a")
        with self.assertRaises(ValueError) as raised:
            self.store.add(source, label=0)
        self.assertInvalidLabelError(raised.exception, 0)
        with self.assertRaises(ValueError) as raised:
            self.store.import_directory(self.source, label=False)
        self.assertInvalidLabelError(raised.exception, False)


class InvalidLabelTakesPrecedenceTest(ImportLabelValidationTestBase):
    """The label is judged before the source, duplicates or the manifest."""

    def test_add_missing_source_still_reports_the_label_first(self) -> None:
        missing = self.base / "never-there.jpg"
        with self.assertRaises(ValueError) as raised:
            self.store.add(missing, 0)
        self.assertInvalidLabelError(raised.exception, 0)

    def test_add_directory_source_still_reports_the_label_first(self) -> None:
        # A directory path would otherwise be "Not a regular file"; the
        # invalid label must be diagnosed before that check.
        with self.assertRaises(ValueError) as raised:
            self.store.add(self.source, False)
        self.assertInvalidLabelError(raised.exception, False)

    def test_add_unreadable_file_still_reports_the_label_first(self) -> None:
        source = self._write_file("locked.jpg", b"image-a")
        os.chmod(source, 0o000)
        try:
            with self.assertRaises(ValueError) as raised:
                self.store.add(source, ["cat"])
            self.assertInvalidLabelError(raised.exception, ["cat"])
        finally:
            os.chmod(source, 0o644)

    def test_import_missing_source_still_reports_the_label_first(self) -> None:
        missing = self.base / "no-such-directory"
        with self.assertRaises(ValueError) as raised:
            self.store.import_directory(missing, 1)
        self.assertInvalidLabelError(raised.exception, 1)

    def test_import_file_source_still_reports_the_label_first(self) -> None:
        source_file = self._write_file("not-a-dir.jpg", b"x")
        with self.assertRaises(ValueError) as raised:
            self.store.import_directory(source_file, {})
        self.assertInvalidLabelError(raised.exception, {})

    def test_label_error_precedes_corrupt_manifest_detection(self) -> None:
        # A broken registration list is a distinct failure mode; with an
        # invalid call argument the argument error is what surfaces.
        self.store.initialize()
        _state_path(self.workspace, "manifest.json").write_text(
            "{not valid json", encoding="utf-8"
        )
        source = self._write_file("a.jpg", b"image-a")

        reopened = DatasetStore(self.workspace)
        with self.assertRaises(ValueError) as raised:
            reopened.add(source, 0)
        self.assertIs(type(raised.exception), ValueError)
        self.assertNotIsInstance(raised.exception, ManifestError)
        self.assertIn("label", str(raised.exception))

        with self.assertRaises(ValueError) as raised:
            reopened.import_directory(self.source, False)
        self.assertIs(type(raised.exception), ValueError)
        self.assertNotIsInstance(raised.exception, ManifestError)
        self.assertIn("label", str(raised.exception))


class InvalidLabelHasNoSideEffectsTest(ImportLabelValidationTestBase):
    """Rejecting the label reads nothing and writes nothing."""

    def test_add_with_bad_label_never_opens_the_source(self) -> None:
        source = self._write_file("a.jpg", b"image-a")
        with mock.patch.object(
            confirmed, "confirm_and_open_regular"
        ) as confirm, mock.patch.object(confirmed, "hash_descriptor") as hasher:
            with self.assertRaises(ValueError):
                self.store.add(source, 0)
        confirm.assert_not_called()
        hasher.assert_not_called()

    def test_import_with_bad_label_never_scans_or_reads(self) -> None:
        self._write_image("a.jpg", b"image-a")
        self._write_image("sub/b.png", b"image-b")
        with (
            mock.patch.object(self.store, "_scan_image_files") as scan,
            mock.patch.object(confirmed, "confirm_and_open_regular") as confirm,
            mock.patch.object(confirmed, "hash_descriptor") as hasher,
        ):
            with self.assertRaises(ValueError):
                self.store.import_directory(self.source, ["nope"], recursive=True)
        scan.assert_not_called()
        confirm.assert_not_called()
        hasher.assert_not_called()

    def test_failure_does_not_initialize_an_unopened_workspace(self) -> None:
        fresh = self.base / "fresh-workspace"
        store = DatasetStore(fresh)
        source = self._write_file("a.jpg", b"image-a")

        with self.assertRaises(ValueError):
            store.add(source, 0)
        self.assertFalse((fresh / ".vision-workbench").exists())

        with self.assertRaises(ValueError):
            store.import_directory(self.source, False)
        self.assertFalse((fresh / ".vision-workbench").exists())

        # Even a missing source must not leave registration data behind.
        with self.assertRaises(ValueError):
            store.import_directory(self.base / "missing-dir", [])
        self.assertFalse((fresh / ".vision-workbench").exists())

    def test_reimporting_registered_image_with_bad_label_is_an_error(self) -> None:
        first = self._write_file("first.jpg", b"existing")
        digest = self.store.add(first, "cat").digest

        second = self._write_file("second.jpg", b"existing")
        # A valid re-import reports "already present"; an invalid label may
        # never borrow that success.
        self.assertFalse(self.store.add(second, "dog").added)
        with self.assertRaises(ValueError) as raised:
            self.store.add(second, 0)
        self.assertInvalidLabelError(raised.exception, 0)

        items = _manifest_items(self.workspace)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["sha256"], digest)
        self.assertEqual(items[0]["source"], str(first.resolve()))
        self.assertEqual(items[0]["label"], "cat")

    def test_all_duplicate_directory_with_bad_label_is_an_error(self) -> None:
        self.store.add(self._write_file("seed.jpg", b"existing"), "cat")
        self._write_image("a.jpg", b"existing")
        self._write_image("b.jpg", b"existing")

        # The same content under a valid label is a normal zero-new result.
        ok = self.store.import_directory(self.source, "dog")
        self.assertEqual(ok["added"], 0)
        self.assertEqual(ok["duplicates"], 2)

        # An invalid label must reject the call rather than return that
        # zero-added success.
        with self.assertRaises(ValueError) as raised:
            self.store.import_directory(self.source, False)
        self.assertInvalidLabelError(raised.exception, False)
        self.assertEqual(self.store.summary()["items"], 1)
        self.assertEqual(self.store.lookup_label(_sha256(b"existing"))["label"], "cat")

    def test_empty_directory_with_bad_label_is_an_error(self) -> None:
        empty = self.base / "empty"
        empty.mkdir()
        with self.assertRaises(ValueError) as raised:
            self.store.import_directory(empty, [])
        self.assertInvalidLabelError(raised.exception, [])
        self.assertFalse((self.workspace / ".vision-workbench").exists())

    def test_mixed_new_and_duplicate_directory_registers_nothing(self) -> None:
        seed = self._write_file("seed.jpg", b"existing")
        digest_existing = self.store.add(seed, "bird").digest
        self._write_image("dup.jpg", b"existing")
        self._write_image("new.jpg", b"brand-new")

        with self.assertRaises(ValueError) as raised:
            self.store.import_directory(self.source, 1)
        self.assertInvalidLabelError(raised.exception, 1)

        items = _manifest_items(self.workspace)
        self.assertEqual([item["sha256"] for item in items], [digest_existing])
        self.assertEqual(items[0]["label"], "bird")
        self.assertEqual(items[0]["source"], str(seed.resolve()))
        self.assertNotIn(_sha256(b"brand-new"), {item["sha256"] for item in items})
        self.assertFalse(_state_path(self.workspace, ".txn.json").exists())

    def test_existing_registrations_history_and_plans_stay_byte_identical(self) -> None:
        first = self._write_file("first.jpg", b"existing")
        digest = self.store.add(first, "bird").digest
        self.store.submit_batch(
            {
                "batch": "b1",
                "changes": [{"sha256": digest, "old": "bird", "new": "eagle"}],
            }
        )
        plan = self.store.create_split("baseline", 7, [1, 0, 0])

        # Conditions under which an import would otherwise write nothing:
        # re-import, an all-duplicate directory and an empty directory —
        # plus a missing source and a mixed (new + duplicate) directory.
        duplicate_copy = self._write_file("dup.jpg", b"existing")
        duplicate_dir = self.base / "all-dup"
        duplicate_dir.mkdir()
        (duplicate_dir / "a.jpg").write_bytes(b"existing")
        empty_dir = self.base / "empty"
        empty_dir.mkdir()
        mixed_dir = self.base / "mixed"
        mixed_dir.mkdir()
        (mixed_dir / "dup.jpg").write_bytes(b"existing")
        (mixed_dir / "new.jpg").write_bytes(b"brand-new")

        manifest_before = _manifest_raw(self.workspace)
        history_before = _history_raw(self.workspace)
        plan_before = _plan_raw(self.workspace, "baseline")

        rejected_calls = [
            lambda: self.store.add(duplicate_copy, 0),
            lambda: self.store.add(self.base / "missing.jpg", False),
            lambda: self.store.import_directory(duplicate_dir, []),
            lambda: self.store.import_directory(empty_dir, {}),
            lambda: self.store.import_directory(self.base / "missing-dir", 1),
            lambda: self.store.import_directory(mixed_dir, True),
        ]
        for call in rejected_calls:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()
            # Nothing moved on disk after each refusal.
            self.assertEqual(_manifest_raw(self.workspace), manifest_before)
            self.assertEqual(_history_raw(self.workspace), history_before)
            self.assertEqual(_plan_raw(self.workspace, "baseline"), plan_before)

        # The registered sample keeps its first source, post-batch label
        # and label revision; history and the saved plan are intact.
        self.assertEqual(self.store.summary()["items"], 1)
        self.assertEqual(self.store.lookup_label(digest)["label"], "eagle")
        items = {item["sha256"]: item for item in _manifest_items(self.workspace)}
        self.assertEqual(items[digest]["source"], str(first.resolve()))
        self.assertEqual(items[digest].get("rev"), 1)
        self.assertEqual([entry["batch"] for entry in self.store.history()], ["b1"])
        self.assertEqual(self.store.get_split("baseline"), plan.plan)
        self.assertFalse(_state_path(self.workspace, ".txn.json").exists())
        self.assertNotIn(
            _sha256(b"brand-new"),
            {item["sha256"] for item in _manifest_items(self.workspace)},
        )


class ValidLabelsKeepTheirMeaningTest(ImportLabelValidationTestBase):
    """Legal labels — ``None``, ``""`` and literal strings — still work."""

    def test_none_and_empty_string_both_mean_unlabeled(self) -> None:
        source_a = self._write_file("a.jpg", b"a")
        source_b = self._write_file("b.jpg", b"b")

        ws_none = self.base / "ws-none"
        self.assertTrue(DatasetStore(ws_none).add(source_a).added)
        item_none = _manifest_items(ws_none)[0]
        self.assertIsNone(item_none["label"])

        ws_empty = self.base / "ws-empty"
        self.assertTrue(DatasetStore(ws_empty).add(source_b, "").added)
        item_empty = _manifest_items(ws_empty)[0]
        self.assertEqual(item_empty["label"], "")

        # Both spellings surface as one unlabeled summary stratum.
        self.assertEqual(
            DatasetStore(ws_none).summary()["labels"], {"unlabeled": 1}
        )
        self.assertEqual(
            DatasetStore(ws_empty).summary()["labels"], {"unlabeled": 1}
        )

        # find-label "" reports both storages with label null.
        self.assertEqual(
            DatasetStore(ws_empty).find_by_label("")["samples"][0]["label"], None
        )

    def test_directory_none_and_empty_string_mean_unlabeled(self) -> None:
        self._write_image("a.jpg", b"a")
        ws_none = self.base / "ws-none"
        result = DatasetStore(ws_none).import_directory(self.source)
        self.assertIsNone(result["label"])
        self.assertIsNone(_manifest_items(ws_none)[0]["label"])

        self._write_image("b.jpg", b"b")
        ws_empty = self.base / "ws-empty"
        result = DatasetStore(ws_empty).import_directory(self.source, "")
        self.assertEqual(result["label"], "")
        labels = {item["label"] for item in _manifest_items(ws_empty)}
        self.assertEqual(labels, {""})

    def test_literal_strings_are_saved_unchanged(self) -> None:
        literals = ["猫", "Cat", "cat", " Cat ", "a/b", "a\\b", "unlabeled"]
        for index, label in enumerate(literals):
            self.store.add(self._write_file(f"f-{index}.jpg", f"c-{index}".encode()), label)

        stored = {item["label"] for item in _manifest_items(self.workspace)}
        self.assertEqual(stored, set(literals))
        # Each literal is independently queryable: no case folding,
        # trimming, path interpretation or rewriting.
        for label in literals:
            with self.subTest(label=label):
                self.assertEqual(self.store.find_by_label(label)["count"], 1)

    def test_literal_unlabeled_is_a_real_class_distinct_from_unlabeled(self) -> None:
        self.store.add(self._write_file("real.jpg", b"real"), "unlabeled")
        self.store.add(self._write_file("none.jpg", b"none"), None)
        self.store.add(self._write_file("empty.jpg", b"empty"), "")

        # The query interface keeps the real class and the unlabeled
        # state apart.
        real = self.store.find_by_label("unlabeled")
        self.assertEqual(real["count"], 1)
        self.assertEqual(real["samples"][0]["label"], "unlabeled")

        unlabeled = self.store.find_by_label("")
        self.assertEqual(unlabeled["count"], 2)
        self.assertEqual(
            {sample["label"] for sample in unlabeled["samples"]}, {None}
        )

    def test_new_directory_samples_get_the_label_duplicates_keep_first(self) -> None:
        first = self._write_file("first.jpg", b"existing")
        digest_existing = self.store.add(first, "bird").digest
        self._write_image("dup.jpg", b"existing")
        self._write_image("new.jpg", b"brand-new")

        result = self.store.import_directory(self.source, "kitten")
        statuses = {c["path"]: c["status"] for c in result["candidates"]}
        self.assertEqual(statuses["dup.jpg"], "duplicate")
        self.assertEqual(statuses["new.jpg"], "added")
        self.assertEqual(result["added"], 1)
        self.assertEqual(result["duplicates"], 1)
        self.assertEqual(result["label"], "kitten")

        items = {item["sha256"]: item for item in _manifest_items(self.workspace)}
        self.assertEqual(items[digest_existing]["label"], "bird")
        self.assertEqual(items[digest_existing]["source"], str(first.resolve()))
        self.assertEqual(items[_sha256(b"brand-new")]["label"], "kitten")

    def test_single_and_directory_import_success_behavior_is_unchanged(self) -> None:
        source = self._write_image("a.jpg", b"a")
        first = self.store.add(source, "cat")
        self.assertTrue(first.added)
        self.assertEqual(first.digest, _sha256(b"a"))
        second = self.store.add(source, "cat")
        self.assertFalse(second.added)
        self.assertEqual(second.digest, first.digest)

        self._write_image("b.jpg", b"b")
        result = self.store.import_directory(self.source, "cat")
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["added"], 1)
        self.assertEqual(result["duplicates"], 1)
        # A second directory import is the normal all-duplicate success.
        again = self.store.import_directory(self.source, "cat")
        self.assertEqual(again["added"], 0)
        self.assertEqual(again["duplicates"], 2)
        self.assertEqual(self.store.summary()["items"], 2)

        # An empty directory is a normal zero-new success with a valid label.
        empty = self.base / "empty"
        empty.mkdir()
        empty_result = self.store.import_directory(empty, None)
        self.assertEqual(empty_result["candidate_count"], 0)
        self.assertEqual(empty_result["added"], 0)
        self.assertEqual(empty_result["duplicates"], 0)


if __name__ == "__main__":
    unittest.main()
