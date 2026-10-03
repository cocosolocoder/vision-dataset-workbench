from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from vision_workbench.store import DatasetStore


CONTENT_A = b"A" * 100000
CONTENT_B = b"B" * 300000
DIGEST_A = hashlib.sha256(CONTENT_A).hexdigest()


def _manifest_items(workspace: Path) -> list[dict]:
    manifest_path = workspace / ".vision-workbench" / "manifest.json"
    if not manifest_path.exists():
        return []
    return json.loads(manifest_path.read_text(encoding="utf-8"))["items"]


class AddConsistencyTest(unittest.TestCase):
    def test_stable_file_imports_and_records_consistent_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "img.png"
            source.write_bytes(CONTENT_A)
            store = DatasetStore(root / "workspace")

            result = store.add(source, "cat")

            self.assertTrue(result.added)
            self.assertEqual(result.digest, DIGEST_A)
            (item,) = _manifest_items(root / "workspace")
            self.assertEqual(item["sha256"], DIGEST_A)
            self.assertEqual(item["size"], len(CONTENT_A))
            self.assertEqual(item["source"], str(source.resolve()))
            self.assertEqual(item["label"], "cat")

    def test_rewrite_during_read_fails_and_registers_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "img.png"
            source.write_bytes(CONTENT_A)
            workspace = root / "workspace"
            store = DatasetStore(workspace)

            real_open, real_lstat = os.open, os.lstat
            state = {"armed": False}

            def hooked_open(path, flags, *args, **kwargs):
                descriptor = real_open(path, flags, *args, **kwargs)
                if str(path) == str(source):
                    # The source is now open for reading; the next lstat is
                    # the post-read stability check.
                    state["armed"] = True
                return descriptor

            def hooked_lstat(path, *args, **kwargs):
                if str(path) == str(source) and state["armed"]:
                    state["armed"] = False
                    source.write_bytes(CONTENT_B)  # in-place rewrite mid-read
                return real_lstat(path, *args, **kwargs)

            with mock.patch.object(os, "open", hooked_open), mock.patch.object(
                os, "lstat", hooked_lstat
            ):
                with self.assertRaises(ValueError) as raised:
                    store.add(source, "cat")
            self.assertIn(str(source), str(raised.exception))
            self.assertEqual(_manifest_items(workspace), [])

    def test_replacement_with_identical_metadata_fails_on_identity(self) -> None:
        # Another file with the same content, size and mtime still has a
        # different identity: swapping it in after the pre-read check must
        # fail the import.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "img.png"
            source.write_bytes(CONTENT_A)
            other = root / "other.png"
            other.write_bytes(CONTENT_A)
            stat = os.lstat(source)
            os.utime(other, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            workspace = root / "workspace"
            store = DatasetStore(workspace)

            real_open = os.open

            def hooked_open(path, flags, *args, **kwargs):
                if str(path) == str(source) and not getattr(
                    hooked_open, "swapped", False
                ):
                    hooked_open.swapped = True
                    os.replace(other, source)
                return real_open(path, flags, *args, **kwargs)

            with mock.patch.object(os, "open", hooked_open):
                with self.assertRaises(ValueError) as raised:
                    store.add(source, "cat")
            self.assertIn(str(source), str(raised.exception))
            self.assertEqual(_manifest_items(workspace), [])

    def test_symlink_swapped_in_after_confirmation_is_not_followed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "img.png"
            source.write_bytes(CONTENT_A)
            target = root / "target.png"
            target.write_bytes(CONTENT_B)
            workspace = root / "workspace"
            store = DatasetStore(workspace)

            real_open = os.open

            def hooked_open(path, flags, *args, **kwargs):
                if str(path) == str(source) and not getattr(
                    hooked_open, "swapped", False
                ):
                    hooked_open.swapped = True
                    source.unlink()
                    source.symlink_to(target)
                return real_open(path, flags, *args, **kwargs)

            with mock.patch.object(os, "open", hooked_open):
                with self.assertRaises(ValueError) as raised:
                    store.add(source, "cat")
            self.assertIn(str(source), str(raised.exception))
            self.assertEqual(_manifest_items(workspace), [])

    def test_symlink_argument_still_imports_and_records_resolved_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sub" / "img.png"
            source.parent.mkdir()
            source.write_bytes(CONTENT_A)
            link = root / "link.png"
            link.symlink_to(source)
            store = DatasetStore(root / "workspace")

            result = store.add(link, "cat")

            self.assertTrue(result.added)
            (item,) = _manifest_items(root / "workspace")
            self.assertEqual(item["source"], str(source.resolve()))

    def test_failed_import_does_not_report_already_present(self) -> None:
        # Content matching an existing digest must not mask a read failure.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "img.png"
            source.write_bytes(CONTENT_A)
            workspace = root / "workspace"
            store = DatasetStore(workspace)
            first = store.add(source, "cat")
            self.assertTrue(first.added)

            source.unlink()  # reading now fails even though the digest is known
            with self.assertRaises(OSError):
                store.add(source, "cat")
            (item,) = _manifest_items(workspace)
            self.assertEqual(item["sha256"], DIGEST_A)
            self.assertEqual(item["label"], "cat")

    def test_duplicate_content_keeps_first_registration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path = root / "first.png"
            first_path.write_bytes(CONTENT_A)
            second_path = root / "second.png"
            second_path.write_bytes(CONTENT_A)
            store = DatasetStore(root / "workspace")

            first = store.add(first_path, "cat")
            second = store.add(second_path, "dog")

            self.assertTrue(first.added)
            self.assertFalse(second.added)
            self.assertEqual(first.digest, second.digest)
            (item,) = _manifest_items(root / "workspace")
            self.assertEqual(item["source"], str(first_path.resolve()))
            self.assertEqual(item["label"], "cat")


if __name__ == "__main__":
    unittest.main()
