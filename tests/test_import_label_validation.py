"""Regression coverage for the import ``label`` argument boundary.

Both public import entry points — :meth:`DatasetStore.add` (one image) and
:meth:`DatasetStore.import_directory` (a directory batch) — accept a label
that is either a string or ``None`` and nothing else:

* numbers, booleans, lists, dictionaries and every other non-string value
  raise a *plain* :class:`ValueError` (not a manifest/batch/split error)
  whose message names the ``label`` parameter, the received type and the
  allowed types (a string or ``None``);
* the check is the first thing the call does, so its verdict comes ahead
  of every other failure the import could report — an unreadable or
  missing source, an empty or all-duplicate directory, or a corrupted
  workspace registration list;
* nothing is opened, scanned or registered: existing manifest records,
  batch history and saved split plans stay byte-for-byte as they were, and
  an uninitialized workspace gains no ``.vision-workbench`` data merely
  because an invalid-label call failed;
* legal labels keep their existing meaning — ``None`` and ``""`` mean
  *unlabeled*, every other string is saved literally (Chinese text, case,
  surrounding spaces, path separators and the literal class
  ``"unlabeled"`` are never rewritten), and the success behavior of both
  entry points is retained.

The assertions observe behavior only through the public import entry
points (plus reading the workspace files any user could inspect).
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


STATE = ".vision-workbench"

# (value, expected type name in the error message)
INVALID_VALUES = [
    (0, "int"),
    (1, "int"),
    (-3, "int"),
    (1.5, "float"),
    (False, "bool"),
    (True, "bool"),
    ([], "list"),
    (["cat"], "list"),
    ({}, "dict"),
    ({"label": "cat"}, "dict"),
    ((1, 2), "tuple"),
    (object(), "object"),
]

# The falsy values that must never be treated as "unlabeled": only None
# and the empty string carry that meaning.
FALSY_INVALID = [0, False, []]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _state_path(workspace: Path, *parts: str) -> Path:
    return workspace / STATE / Path(*parts)


def _read_manifest(workspace: Path) -> dict:
    return json.loads(_state_path(workspace, "manifest.json").read_text("utf-8"))


def _read_history(workspace: Path) -> dict:
    return json.loads(_state_path(workspace, "batches.json").read_text("utf-8"))


def _assert_label_error(
    test: unittest.TestCase, value: object, error: Exception
) -> None:
    """The error is a plain ValueError naming label, the type and allowed types."""
    # A plain ValueError: exactly ValueError, not a workspace-corruption
    # subclass (ManifestError/BatchError/SplitError are ValueErrors too).
    test.assertIs(type(error), ValueError, repr(value))
    message = str(error)
    test.assertIn("label", message)
    test.assertIn(type(value).__name__, message)
    test.assertIn("None", message)
    test.assertIn("string", message)
    # The received value's repr appears, so e.g. 0 and 1 distinguish
    # themselves rather than both reading as a generic type failure.
    test.assertIn(repr(value), message)


class ImportLabelTypeRejectionTest(unittest.TestCase):
    """Every non-string/non-None label is refused on both entry points."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.workspace = self.base / "dataset"
        self.store = DatasetStore(self.workspace)
        self.image = self.base / "cat.jpg"
        self.image.write_bytes(b"cat-image")
        self.directory = self.base / "images"
        self.directory.mkdir()
        (self.directory / "a.jpg").write_bytes(b"a")
        (self.directory / "b.png").write_bytes(b"b")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_add_rejects_every_non_string_value(self) -> None:
        for value, _ in INVALID_VALUES:
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValueError) as raised:
                    self.store.add(self.image, value)
                _assert_label_error(self, value, raised.exception)

    def test_import_directory_rejects_every_non_string_value(self) -> None:
        for value, _ in INVALID_VALUES:
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValueError) as raised:
                    self.store.import_directory(self.directory, value)
                _assert_label_error(self, value, raised.exception)

    def test_import_directory_rejects_with_recursive_true_too(self) -> None:
        for value in FALSY_INVALID:
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValueError):
                    self.store.import_directory(
                        self.directory, value, recursive=True
                    )

    def test_falsy_values_never_mean_unlabeled(self) -> None:
        # 0, False and [] must fail; they must not be converted to a
        # category name (e.g. "0") nor treated as an omitted label.
        for value in FALSY_INVALID:
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValueError) as raised:
                    self.store.add(self.image, value)
                _assert_label_error(self, value, raised.exception)
                with self.assertRaises(ValueError) as raised:
                    self.store.import_directory(self.directory, value)
                _assert_label_error(self, value, raised.exception)
        # None of those calls registered anything as unlabeled.
        self.assertEqual(
            self.store.summary(), {"items": 0, "bytes": 0, "labels": {}}
        )

    def test_positive_number_and_bool_also_fail(self) -> None:
        # A truthy number would be an especially tempting implicit
        # conversion target ("7" -> class name); pin the rejection.
        for value in (1, 7, True):
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValueError) as raised:
                    self.store.add(self.image, value)
                _assert_label_error(self, value, raised.exception)

    # ------------------------------------------------------------------
    # No I/O happens before the rejection
    # ------------------------------------------------------------------

    def test_add_rejection_opens_neither_source_nor_workspace(self) -> None:
        real_open = os.open
        for value in FALSY_INVALID:
            opened_paths: list[str] = []

            def tracked_open(path, flags, *args, **kwargs):
                opened_paths.append(str(path))
                return real_open(path, flags, *args, **kwargs)

            with mock.patch("vision_workbench.store.os.open", tracked_open):
                with self.assertRaises(ValueError):
                    self.store.add(self.image, value)
            # No descriptor at all — not the source, and not the workspace
            # lock the (never-reached) initialization would create.
            self.assertEqual(opened_paths, [], repr(value))

    def test_add_rejection_does_not_hash_the_source(self) -> None:
        for value in FALSY_INVALID:
            with mock.patch.object(
                confirmed, "hash_descriptor", side_effect=AssertionError("source read")
            ):
                with self.assertRaises(ValueError):
                    self.store.add(self.image, value)

    def test_directory_rejection_does_not_scan(self) -> None:
        for value in FALSY_INVALID:
            with (
                mock.patch(
                    "vision_workbench.store.os.scandir",
                    side_effect=AssertionError("directory scanned"),
                ),
                mock.patch.object(
                    self.store,
                    "_scan_image_files",
                    side_effect=AssertionError("scan reached"),
                ),
                mock.patch.object(
                    self.store,
                    "initialize",
                    side_effect=AssertionError("workspace initialized"),
                ),
            ):
                with self.assertRaises(ValueError):
                    self.store.import_directory(self.directory, value)

    def test_add_rejection_does_not_initialize_workspace(self) -> None:
        for value in FALSY_INVALID:
            with mock.patch.object(
                self.store,
                "initialize",
                side_effect=AssertionError("workspace initialized"),
            ):
                with self.assertRaises(ValueError):
                    self.store.add(self.image, value)

    # ------------------------------------------------------------------
    # Precedence: the invalid-label verdict outranks every other failure
    # ------------------------------------------------------------------

    def test_label_error_precedes_missing_source_on_add(self) -> None:
        missing = self.base / "does-not-exist.jpg"
        with self.assertRaises(ValueError) as raised:
            self.store.add(missing, 0)
        _assert_label_error(self, 0, raised.exception)
        self.assertNotIn("cannot read", str(raised.exception))

    def test_label_error_precedes_missing_source_directory(self) -> None:
        missing = self.base / "no-such-directory"
        with self.assertRaises(ValueError) as raised:
            self.store.import_directory(missing, False)
        _assert_label_error(self, False, raised.exception)
        self.assertNotIn("does not exist", str(raised.exception))

    def test_label_error_precedes_file_used_as_directory(self) -> None:
        with self.assertRaises(ValueError) as raised:
            self.store.import_directory(self.image, [])
        _assert_label_error(self, [], raised.exception)
        self.assertNotIn("not a directory", str(raised.exception))

    def test_label_error_precedes_unreadable_source(self) -> None:
        # Even a source that cannot be opened must report the invalid
        # argument first: this is a caller error, not a media read failure.
        real_open = os.open
        denied = str(self.image.resolve())

        def denying_open(path, flags, *args, **kwargs):
            if str(path) == denied:
                raise PermissionError(13, "Permission denied (injected)")
            return real_open(path, flags, *args, **kwargs)

        with mock.patch("vision_workbench.store.os.open", denying_open):
            with self.assertRaises(ValueError) as raised:
                self.store.add(self.image, {})
        _assert_label_error(self, {}, raised.exception)

    def test_label_error_distinct_from_corrupt_manifest(self) -> None:
        # A healthy workspace first, so a subsequent corrupting edit has
        # something to replace; an invalid label is still an argument error
        # rather than a surfaced ManifestError...
        self.store.initialize()
        manifest_path = _state_path(self.workspace, "manifest.json")
        manifest_path.write_text(
            '{"schema_version": 1, "items": ['
            '{"sha256": "deadbeef", "source": "s", "size": 1, "label": null}'
            "]}",
            encoding="utf-8",
        )
        with self.assertRaises(ValueError) as raised:
            self.store.add(self.image, 5)
        _assert_label_error(self, 5, raised.exception)
        self.assertNotIsInstance(raised.exception, ManifestError)
        with self.assertRaises(ValueError) as raised:
            self.store.import_directory(self.directory, True)
        _assert_label_error(self, True, raised.exception)
        self.assertNotIsInstance(raised.exception, ManifestError)

        # ...whereas a *valid* label on the same workspace surfaces the
        # stored-data corruption, proving the two error kinds stay apart.
        with self.assertRaises(ManifestError):
            self.store.add(self.image, "cat")
        with self.assertRaises(ManifestError):
            self.store.import_directory(self.directory, "cat")


class ImportLabelNoSideEffectsTest(unittest.TestCase):
    """Rejection leaves no registration and no workspace data behind."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.workspace = self.base / "dataset"
        self.source = self.base / "images"
        self.source.mkdir()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write(self, relative: str, content: bytes) -> Path:
        path = self.source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_failure_never_initializes_an_uninitialized_workspace(self) -> None:
        store = DatasetStore(self.workspace)
        self._write("a.jpg", b"a")
        single = self.base / "single.jpg"
        single.write_bytes(b"single")

        for value in (0, False, []):
            with self.assertRaises(ValueError):
                store.add(single, value)
            with self.assertRaises(ValueError):
                store.import_directory(self.source, value)
        # The failed calls alone must not have created registration data.
        self.assertFalse(
            (self.workspace / STATE).exists(),
            os.listdir(self.workspace) if self.workspace.exists() else None,
        )

    def test_reimport_of_registered_image_still_rejects_bad_label(self) -> None:
        store = DatasetStore(self.workspace)
        single = self.base / "single.jpg"
        single.write_bytes(b"single-bytes")
        digest = store.add(single, "cat").digest

        # Without the boundary this would be the harmless "already present"
        # success; it must instead be refused...
        with self.assertRaises(ValueError) as raised:
            store.add(single, 0)
        _assert_label_error(self, 0, raised.exception)

        # ...and the first registration's source, label and revision win.
        items = _read_manifest(self.workspace)["items"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["sha256"], digest)
        self.assertEqual(items[0]["source"], str(single.resolve()))
        self.assertEqual(items[0]["label"], "cat")
        self.assertNotIn("rev", items[0])
        self.assertFalse(store.add(single, "cat").added)

    def test_all_duplicate_directory_still_rejects_bad_label(self) -> None:
        store = DatasetStore(self.workspace)
        self._write("a.jpg", b"a")
        self._write("b.png", b"b")
        first = store.import_directory(self.source, "cat")
        self.assertEqual(first["added"], 2)

        # Every candidate is already known: without the boundary this is a
        # normal zero-added/all-duplicate result.
        with self.assertRaises(ValueError) as raised:
            store.import_directory(self.source, False)
        _assert_label_error(self, False, raised.exception)
        with self.assertRaises(ValueError) as raised:
            store.import_directory(self.source, [])
        _assert_label_error(self, [], raised.exception)

        result = store.import_directory(self.source, "other")
        self.assertEqual(result["added"], 0)
        self.assertEqual(result["duplicates"], 2)
        labels = {
            item["sha256"]: item["label"]
            for item in _read_manifest(self.workspace)["items"]
        }
        self.assertEqual(set(labels.values()), {"cat"})

    def test_empty_directory_still_rejects_bad_label(self) -> None:
        store = DatasetStore(self.base / "empty-ws")
        empty = self.base / "empty-dir"
        empty.mkdir()
        # The legal call is the documented zero-added success...
        ok = store.import_directory(empty, "cat")
        self.assertEqual((ok["added"], ok["duplicates"], ok["candidate_count"]),
                         (0, 0, 0))
        # ...but an invalid label may not ride that empty path to success.
        for value in (0, False, []):
            with self.assertRaises(ValueError) as raised:
                store.import_directory(empty, value)
            _assert_label_error(self, value, raised.exception)

    def test_mixed_new_and_duplicate_directory_registers_nothing(self) -> None:
        store = DatasetStore(self.workspace)
        self._write("dup.jpg", b"known")
        self._write("new.png", b"fresh")
        seed = self.base / "seed.jpg"
        seed.write_bytes(b"known")
        known_digest = store.add(seed, "bird").digest
        before = _read_manifest(self.workspace)

        with self.assertRaises(ValueError) as raised:
            store.import_directory(self.source, 42)
        _assert_label_error(self, 42, raised.exception)

        after = _read_manifest(self.workspace)
        self.assertEqual(after, before)
        items = after["items"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["sha256"], known_digest)
        self.assertEqual(items[0]["label"], "bird")
        self.assertEqual(items[0]["source"], str(seed.resolve()))
        # The genuinely-new content was never registered under the bad label.
        self.assertNotIn(_sha256(b"fresh"), {i["sha256"] for i in items})
        self.assertFalse(_state_path(self.workspace, ".txn.json").exists())

    def test_existing_workspace_state_is_untouched(self) -> None:
        store = DatasetStore(self.workspace)
        seed = self.base / "seed.jpg"
        seed.write_bytes(b"seed-bytes")
        digest = store.add(seed, "bird").digest
        store.submit_batch(
            {
                "batch": "b1",
                "changes": [
                    {"sha256": digest, "old": "bird", "new": "eagle"}
                ],
            }
        )
        plan = store.create_split("baseline", 7, [1, 0, 0])

        manifest_before = _state_path(
            self.workspace, "manifest.json"
        ).read_text("utf-8")
        history_before = _state_path(self.workspace, "batches.json").read_text(
            "utf-8"
        )
        plan_before = _state_path(
            self.workspace, "splits", "baseline.json"
        ).read_text("utf-8")

        single = self.base / "another.jpg"
        single.write_bytes(b"another")
        self._write("a.jpg", b"dir-a")
        for value in (1, True, {}):
            with self.assertRaises(ValueError):
                store.add(single, value)
            with self.assertRaises(ValueError):
                store.import_directory(self.source, value)

        self.assertEqual(
            _state_path(self.workspace, "manifest.json").read_text("utf-8"),
            manifest_before,
        )
        self.assertEqual(
            _state_path(self.workspace, "batches.json").read_text("utf-8"),
            history_before,
        )
        self.assertEqual(
            _state_path(
                self.workspace, "splits", "baseline.json"
            ).read_text("utf-8"),
            plan_before,
        )
        self.assertEqual(store.lookup_label(digest)["label"], "eagle")
        self.assertEqual([e["batch"] for e in store.history()], ["b1"])
        self.assertEqual(store.get_split("baseline"), plan.plan)
        self.assertFalse(_state_path(self.workspace, ".txn.json").exists())
        self.assertEqual(store.summary()["items"], 1)


class LegalLabelsKeepTheirMeaningTest(unittest.TestCase):
    """Valid labels behave exactly as documented on both entry points."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _workspace(self, name: str) -> DatasetStore:
        return DatasetStore(self.base / name)

    def test_none_and_empty_string_mean_unlabeled_on_add(self) -> None:
        image = self.base / "a.jpg"
        image.write_bytes(b"a")
        store = self._workspace("ws")
        self.assertTrue(store.add(image).added)  # omitted -> None
        self.assertFalse(store.add(image, None).added)
        self.assertEqual(
            store.summary(), {"items": 1, "bytes": 1, "labels": {"unlabeled": 1}}
        )

        image2 = self.base / "b.jpg"
        image2.write_bytes(b"b")
        store2 = self._workspace("ws2")
        self.assertTrue(store2.add(image2, "").added)
        self.assertEqual(store2.summary()["labels"], {"unlabeled": 1})

    def test_strings_are_saved_literally_on_add(self) -> None:
        cases = ["猫", "Cat", " cat ", "a/b", "unlabeled", "0", "False", ""]
        store = self._workspace("ws")
        for index, label in enumerate(cases):
            image = self.base / f"img-{index}.jpg"
            image.write_bytes(f"content-{index}".encode())
            self.assertTrue(store.add(image, label).added)
        labels = {
            item["sha256"]: item["label"]
            for item in _read_manifest(self.base / "ws")["items"]
        }
        by_content = {
            _sha256(f"content-{index}".encode()): label
            for index, label in enumerate(cases)
        }
        self.assertEqual(labels, by_content)
        # The literal class "unlabeled" is a real class, not the unlabeled
        # state: summary keeps it distinct from the empty-label sample.
        summary = store.summary()["labels"]
        # Summary merges the unlabeled state and the literal class under
        # one display key ("" or None or "unlabeled" all read "unlabeled")…
        self.assertEqual(summary.get("unlabeled"), 2)
        # …but find-label keeps them apart: the real class is one sample,
        # the genuinely-unlabeled ("") sample is another.
        self.assertEqual(store.find_by_label("unlabeled")["count"], 1)
        self.assertEqual(store.find_by_label("")["count"], 1)

    def test_none_and_empty_string_mean_unlabeled_on_directory(self) -> None:
        source = self.base / "images"
        source.mkdir()
        (source / "a.jpg").write_bytes(b"a")
        (source / "b.png").write_bytes(b"b")

        omitted = self._workspace("ws1").import_directory(source)
        self.assertIsNone(omitted["label"])
        self.assertEqual(omitted["added"], 2)
        self.assertEqual(
            self._workspace("ws1").summary()["labels"], {"unlabeled": 2}
        )

        empty = self._workspace("ws2").import_directory(source, "")
        self.assertEqual(empty["label"], "")
        self.assertEqual(empty["added"], 2)
        self.assertEqual(
            self._workspace("ws2").summary()["labels"], {"unlabeled": 2}
        )

    def test_strings_are_saved_literally_on_directory(self) -> None:
        source = self.base / "images"
        source.mkdir()
        labels = ["猫", "Cat", " cat ", "a/b", "unlabeled"]
        for index, _label in enumerate(labels):
            (source / f"img-{index}.jpg").write_bytes(f"d-{index}".encode())
        store = self._workspace("ws")
        result = store.import_directory(source, "fixed")
        self.assertEqual(result["added"], len(labels))
        # A single batch label applies to every genuinely new sample,
        # stored verbatim.
        self.assertTrue(
            all(
                item["label"] == "fixed"
                for item in _read_manifest(self.base / "ws")["items"]
            )
        )

    def test_literal_unlabeled_is_a_real_class_in_directory_import(self) -> None:
        source = self.base / "images"
        source.mkdir()
        (source / "a.jpg").write_bytes(b"a")
        store = self._workspace("ws")
        result = store.import_directory(source, "unlabeled")
        self.assertEqual(result["added"], 1)
        self.assertEqual(store.find_by_label("unlabeled")["count"], 1)
        # It does not appear among the genuinely unlabeled samples.
        self.assertEqual(store.find_by_label("")["count"], 0)

    def test_new_samples_get_batch_label_duplicates_keep_first(self) -> None:
        source = self.base / "images"
        source.mkdir()
        (source / "dup.jpg").write_bytes(b"shared")
        (source / "new.png").write_bytes(b"new")
        seed = self.base / "seed.jpg"
        seed.write_bytes(b"shared")

        store = self._workspace("ws")
        first_digest = store.add(seed, "bird").digest
        result = store.import_directory(source, "cat")
        self.assertEqual(result["added"], 1)
        self.assertEqual(result["duplicates"], 1)

        by_digest = {
            item["sha256"]: item
            for item in _read_manifest(self.base / "ws")["items"]
        }
        self.assertEqual(by_digest[first_digest]["label"], "bird")
        self.assertEqual(by_digest[first_digest]["source"], str(seed.resolve()))
        self.assertEqual(by_digest[_sha256(b"new")]["label"], "cat")

    def test_single_and_directory_import_success_shapes_are_retained(self) -> None:
        source = self.base / "images"
        source.mkdir()
        (source / "a.jpg").write_bytes(b"a")
        store = self._workspace("ws")
        single = self.base / "s.jpg"
        single.write_bytes(b"s")

        added = store.add(single, "cat")
        self.assertTrue(added.added)
        self.assertEqual(added.digest, _sha256(b"s"))
        again = store.add(single, "dog")
        self.assertFalse(again.added)
        self.assertEqual(again.digest, added.digest)

        result = store.import_directory(source, None)
        self.assertEqual(result["added"], 1)
        self.assertEqual(result["duplicates"], 0)
        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(result["candidates"][0]["path"], "a.jpg")
        self.assertEqual(result["candidates"][0]["sha256"], _sha256(b"a"))
        self.assertEqual(result["candidates"][0]["status"], "added")
        self.assertIsNone(result["label"])


if __name__ == "__main__":
    unittest.main()
