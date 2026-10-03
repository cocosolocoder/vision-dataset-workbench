"""Exports whose individual samples reach the classic 2 GiB ZIP limit.

The large samples are sparse files of a known byte pattern: their
apparent size is genuinely 2 GiB (so every reader hashes and copies the
full content), but they occupy almost no disk and their deflated form is
tiny -- the case that used to fail despite the package staying small.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from vision_workbench.exporter import (
    ExportError,
    _entry_needs_zip64,
    export_split,
)
from vision_workbench.store import DatasetStore

REPO_ROOT = Path(__file__).resolve().parents[1]
LIMIT = 2**31  # 2,147,483,648: the spec's 2 GiB boundary

# SHA-256 of exactly LIMIT zero bytes; proves the sample really was the
# full-size content, not a truncated placeholder.
ZEROS_2GIB_SHA256 = (
    "a7c744c13cc101ed66c29f672f92455547889cc586ce6d44fe76ae824958ea51"
)
CHUNK = 1024 * 1024


def write_sparse(path: Path, size: int, fill: bytes = b"\0") -> None:
    """Create a regular file with an apparent ``size`` and a constant fill."""
    with open(path, "wb") as stream:
        if size:
            stream.seek(size - 1)
            stream.write(fill)


def stream_sha256(path_or_file: Path | object, size: int | None = None) -> tuple[str, int]:
    hasher = hashlib.sha256()
    total = 0
    stream = open(path_or_file, "rb") if isinstance(path_or_file, Path) else path_or_file
    try:
        while True:
            chunk = stream.read(CHUNK)
            if not chunk:
                break
            hasher.update(chunk)
            total += len(chunk)
    finally:
        if isinstance(path_or_file, Path):
            stream.close()
    if size is not None:
        assert total == size
    return hasher.hexdigest(), total


def local_extra_fields(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    """Return the raw extra field bytes from one entry's local file header."""
    raw_path = archive.fp.name
    with open(raw_path, "rb") as raw:
        raw.seek(info.header_offset)
        header = raw.read(30)
        (signature,) = struct.unpack_from("<I", header, 0)
        assert signature == 0x04034B50, hex(signature)
        name_length, extra_length = struct.unpack_from("<HH", header, 26)
        raw.seek(name_length, os.SEEK_CUR)
        return raw.read(extra_length)


def has_zip64_extra(extra: bytes) -> bool:
    offset = 0
    while offset + 4 <= len(extra):
        tag, size = struct.unpack_from("<HH", extra, offset)
        if tag == 0x0001:
            return True
        offset += 4 + size
    return False


class Zip64ThresholdTest(unittest.TestCase):
    def test_exact_boundary_matches_zipfile_heuristic(self) -> None:
        # Smallest integer size zipfile's own (size * 1.05 > ZIP64_LIMIT)
        # test turns on at, and the size immediately below it.
        field_limit = zipfile.ZIP64_LIMIT  # 2**31 - 1
        threshold = field_limit * 20 // 21 + 1  # 2,045,222,521
        self.assertFalse(_entry_needs_zip64(threshold - 1))
        self.assertTrue(_entry_needs_zip64(threshold))
        # The exact 2 GiB spec boundary (and anything above) always needs it.
        self.assertTrue(_entry_needs_zip64(LIMIT))
        self.assertTrue(_entry_needs_zip64(LIMIT + 1))
        # Ordinary image sizes never change format.
        self.assertFalse(_entry_needs_zip64(0))
        self.assertFalse(_entry_needs_zip64(1024 * 1024 * 1024))


class LargeExportHarness(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / "dataset"
        self.store = DatasetStore(self.root)
        self.store.initialize()
        self._serial = 0

    def tearDown(self) -> None:
        self.temp.cleanup()

    def add_small(self, label: str | None, content: bytes) -> str:
        self._serial += 1
        path = self.base / f"sample-{self._serial}.jpg"
        path.write_bytes(content)
        result = self.store.add(path, label)
        self.assertTrue(result.added)
        return result.digest

    def add_sparse(self, label: str | None, size: int = LIMIT,
                   suffix: str = ".jpg") -> tuple[str, Path]:
        self._serial += 1
        path = self.base / f"large-{self._serial}{suffix}"
        write_sparse(path, size)
        result = self.store.add(path, label)
        return result.digest, path

    def plan(self, name: str = "baseline", ratios: list[str] | None = None) -> None:
        self.store.create_split(name, 42, ratios or ["1", "0", "0"])

    def export(self, name: str = "baseline", target_name: str = "out.zip",
               source_dir: Path | None = None) -> tuple[dict, Path]:
        target = self.base / target_name
        result = export_split(
            self.store, name, target, source_dir=source_dir
        )
        return result, target


class LargeFileExportTest(LargeExportHarness):
    def test_exact_2gib_compressible_sample_exports_fully(self) -> None:
        digest, _ = self.add_sparse("cat")
        self.assertEqual(digest, ZEROS_2GIB_SHA256)
        self.plan()

        result, target = self.export()
        self.assertEqual(result["exported"], 1)
        self.assertEqual(result["skipped"],
                         {"train": 0, "validation": 0, "test": 0})
        self.assertEqual(result["sets"]["train"]["samples"], 1)
        self.assertEqual(result["sets"]["train"]["distribution"], {"cat": 1})
        # Highly compressible content keeps the package small; the old
        # failure happened at the uncompressed-size boundary regardless.
        self.assertLess(target.stat().st_size, 16 * 1024 * 1024)
        self.assertNoPackageLeftovers(target)

        with zipfile.ZipFile(target) as archive:
            self.assertIsNone(archive.testzip())
            entries = [i for i in archive.infolist() if i.filename.endswith(".jpg")]
            self.assertEqual(len(entries), 1)
            info = entries[0]
            self.assertEqual(info.filename, f"train/class_01/{digest}.jpg")
            self.assertEqual(info.file_size, LIMIT)
            self.assertLess(info.compress_size, LIMIT)
            # ZIP64 format, streamed without a data descriptor.
            self.assertEqual(info.extract_version, 45)
            self.assertEqual(info.flag_bits & 0x08, 0)
            self.assertTrue(has_zip64_extra(info.extra))
            self.assertTrue(has_zip64_extra(local_extra_fields(archive, info)))

            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(manifest["sets"]["train"],
                             {"samples": 1, "skipped": 0})
            self.assertEqual(len(manifest["samples"]), 1)
            sample = manifest["samples"][0]
            self.assertEqual(sample["sha256"], digest)
            self.assertEqual(sample["set"], "train")
            self.assertEqual(sample["label"], "cat")
            self.assertEqual(sample["path"], info.filename)

            # Every extracted byte must match the recorded digest.
            actual, total = stream_sha256(archive.open(info), LIMIT)
            self.assertEqual(total, LIMIT)
            self.assertEqual(actual, digest)

    def test_large_and_normal_samples_share_one_plan_and_package(self) -> None:
        big_digest, _ = self.add_sparse("cat")
        small_cat = self.add_small("cat", b"cat-picture")
        small_dog = self.add_small("dog", b"dog-picture")
        self.plan()

        result, target = self.export()
        self.assertEqual(result["exported"], 3)
        self.assertEqual(result["sets"]["train"]["distribution"],
                         {"cat": 2, "dog": 1})

        with zipfile.ZipFile(target) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(len(manifest["samples"]), 3)
            self.assertEqual(
                sorted((s["sha256"], s["set"]) for s in manifest["samples"]),
                sorted((d, "train") for d in (big_digest, small_cat, small_dog)),
            )
            by_digest = {s["sha256"]: s for s in manifest["samples"]}
            infos = {Path(i.filename).stem: i for i in archive.infolist()
                     if i.filename.endswith(".jpg")}
            self.assertEqual(set(infos), {big_digest, small_cat, small_dog})

            big_info = infos[big_digest]
            self.assertEqual(big_info.file_size, LIMIT)
            self.assertEqual(big_info.extract_version, 45)
            self.assertTrue(has_zip64_extra(big_info.extra))
            # Both samples share the cat class directory; dog has its own.
            self.assertEqual(
                Path(by_digest[big_digest]["path"]).parent,
                Path(by_digest[small_cat]["path"]).parent,
            )
            self.assertNotEqual(
                Path(by_digest[big_digest]["path"]).parent,
                Path(by_digest[small_dog]["path"]).parent,
            )

            for digest, content in (
                (small_cat, b"cat-picture"),
                (small_dog, b"dog-picture"),
            ):
                info = infos[digest]
                self.assertEqual(info.file_size, len(content))
                self.assertEqual(info.extract_version, 20)
                self.assertEqual(info.flag_bits & 0x08, 0)
                self.assertEqual(info.extra, b"")
                self.assertEqual(local_extra_fields(archive, info), b"")
                self.assertEqual(archive.read(info.filename), content)

    def test_reexport_is_byte_identical(self) -> None:
        self.add_sparse("cat")
        self.plan()
        first = self.base / "first.zip"
        second = self.base / "second.zip"
        export_split(self.store, "baseline", first)
        export_split(self.store, "baseline", second)
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_source_dir_export_byte_identical_and_verified(self) -> None:
        digest, original = self.add_sparse("cat")
        self.plan()
        normal = self.base / "normal.zip"
        export_split(self.store, "baseline", normal)

        # Relocate: new tree, new name and extension; original vanishes.
        moved = self.base / "moved"
        moved.mkdir()
        relocated = moved / "picture.renamed"
        shutil.move(str(original), relocated)
        via_dir = self.base / "via-dir.zip"
        result, target = self.export(target_name="via-dir.zip", source_dir=moved)
        self.assertEqual(result["exported"], 1)
        self.assertEqual(normal.read_bytes(), via_dir.read_bytes())

        with zipfile.ZipFile(target) as archive:
            info = [i for i in archive.infolist()
                    if i.filename.endswith(".jpg")][0]
            self.assertEqual(info.file_size, LIMIT)
            actual, total = stream_sha256(archive.open(info), LIMIT)
            self.assertEqual(total, LIMIT)
            self.assertEqual(actual, digest)

    def test_missing_large_source_fails_names_digest_leaves_nothing(self) -> None:
        digest, source = self.add_sparse("cat")
        self.plan()
        source.unlink()
        target = self.base / "out.zip"
        with self.assertRaises(ExportError) as caught:
            export_split(self.store, "baseline", target)
        self.assertIn(digest, str(caught.exception))
        self.assertFalse(target.exists())
        self.assertNoPackageLeftovers(target)

    def test_changed_large_source_fails_names_digest_leaves_nothing(self) -> None:
        digest, source = self.add_sparse("cat")
        self.plan()
        # Same path now carries different, much shorter content: the hash
        # check must reject it even though the package write would fit.
        source.write_bytes(b"not the planned bytes anymore")
        target = self.base / "out.zip"
        with self.assertRaises(ExportError) as caught:
            export_split(self.store, "baseline", target)
        self.assertIn(digest, str(caught.exception))
        self.assertIn("digest", str(caught.exception))
        self.assertFalse(target.exists())
        self.assertNoPackageLeftovers(target)

    def test_existing_target_kept_when_large_export_fails(self) -> None:
        _, source = self.add_sparse("cat")
        self.plan()
        target = self.base / "out.zip"
        target.write_bytes(b"do-not-touch")
        source.unlink()
        with self.assertRaises(ExportError):
            export_split(self.store, "baseline", target)
        self.assertEqual(target.read_bytes(), b"do-not-touch")
        self.assertNoPackageLeftovers(target)

    def assertNoPackageLeftovers(self, target: Path) -> None:
        leftovers = [
            path.name
            for path in target.parent.iterdir()
            if path.name.startswith(f".{target.name}.") and path.name.endswith(".tmp")
        ]
        self.assertEqual(leftovers, [])


class SmallFileFormatCompatibilityTest(LargeExportHarness):
    def _small_plan_zip(self, home: Path) -> tuple[Path, str, str]:
        store = DatasetStore(home / "workspace")
        store.initialize()
        digests = []
        for name, label, content in (
            ("a.jpg", "cat", b"cat-one"),
            ("b.jpg", "dog", b"dog-two"),
        ):
            path = home / name
            path.write_bytes(content)
            digests.append(store.add(path, label).digest)
        store.create_split("baseline", 42, ["0.5", "0.25", "0.25"])
        target = home / "plan.zip"
        export_split(store, "baseline", target)
        return target, digests[0], digests[1]

    def test_small_plan_uses_only_classic_format(self) -> None:
        home = self.base / "home"
        home.mkdir()
        target, d1, d2 = self._small_plan_zip(home)
        blob = target.read_bytes()
        # No ZIP64 end-of-central-directory records anywhere.
        self.assertNotIn(b"PK\x06\x06", blob)
        self.assertNotIn(b"PK\x06\x07", blob)
        with zipfile.ZipFile(target) as archive:
            for info in archive.infolist():
                if info.filename.endswith("/"):
                    continue
                self.assertEqual(info.extract_version, 20, info.filename)
                self.assertEqual(info.flag_bits, 0, info.filename)
                self.assertEqual(info.extra, b"", info.filename)
                self.assertEqual(local_extra_fields(archive, info), b"")
            files = {i.filename for i in archive.infolist()
                     if i.filename.endswith(".jpg")}
            # Both samples land somewhere across the sets, under their
            # shared/distinct class directories; set assignment depends on
            # the seed and is asserted by the plan-level tests.
            self.assertTrue(
                any(f.endswith(f"/class_01/{d1}.jpg") for f in files), files
            )
            self.assertTrue(
                any(f.endswith(f"/class_02/{d2}.jpg") for f in files), files
            )
            self.assertEqual(len(files), 2)

    def test_bytes_match_pre_zip64_implementation(self) -> None:
        # The same small plan exported with the exporter as committed at
        # HEAD (no ZIP64 support) must produce identical bytes.
        if not (REPO_ROOT / ".git").exists() or not shutil.which("git"):
            self.skipTest("requires a git checkout with git available")
        if not shutil.which("tar"):
            self.skipTest("requires tar to materialize the HEAD package")

        old_root = self.base / "old-src"
        old_root.mkdir()
        git_archive = subprocess.Popen(
            ["git", "archive", "HEAD", "vision_workbench"],
            cwd=REPO_ROOT, stdout=subprocess.PIPE,
        )
        try:
            subprocess.run(
                ["tar", "-x", "-C", str(old_root)],
                stdin=git_archive.stdout, check=True,
            )
        finally:
            assert git_archive.stdout is not None
            git_archive.stdout.close()
            self.assertEqual(git_archive.wait(), 0)

        old_home = self.base / "old-home"
        old_home.mkdir()
        env = dict(os.environ, PYTHONPATH=str(old_root))
        script = (
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, sys.argv[2])\n"
            "from vision_workbench.store import DatasetStore\n"
            "from vision_workbench.exporter import export_split\n"
            "base = Path(sys.argv[1])\n"
            "store = DatasetStore(base / 'workspace')\n"
            "store.initialize()\n"
            "for name, label, content in (\n"
            "    ('a.jpg', 'cat', b'cat-one'),\n"
            "    ('b.jpg', 'dog', b'dog-two'),\n"
            "):\n"
            "    path = base / name\n"
            "    path.write_bytes(content)\n"
            "    assert store.add(path, label).added\n"
            "store.create_split('baseline', 42, ['0.5', '0.25', '0.25'])\n"
            "export_split(store, 'baseline', base / 'plan.zip')\n"
        )
        subprocess.run(
            [sys.executable, "-c", script, str(old_home), str(old_root)],
            cwd=REPO_ROOT, env=env, check=True, capture_output=True, text=True,
        )

        current_home = self.base / "current-home"
        current_home.mkdir()
        current_zip, _, _ = self._small_plan_zip(current_home)
        self.assertEqual(current_zip.read_bytes(), (old_home / "plan.zip").read_bytes())


@unittest.skipUnless(shutil.which("unzip"), "requires Info-ZIP unzip")
class InfoZipInteropTest(LargeExportHarness):
    def test_infozip_tests_and_extracts_full_size(self) -> None:
        digest, _ = self.add_sparse("cat")
        self.plan()
        _, target = self.export()

        tested = subprocess.run(
            ["unzip", "-t", str(target)], capture_output=True, text=True
        )
        self.assertEqual(tested.returncode, 0, tested.stderr + tested.stdout)

        extracted_root = self.base / "extracted"
        extracted_root.mkdir()
        subprocess.run(
            ["unzip", "-q", str(target), "-d", str(extracted_root)], check=True
        )
        extracted = extracted_root / "train" / "class_01" / f"{digest}.jpg"
        self.assertEqual(extracted.stat().st_size, LIMIT)
        actual, total = stream_sha256(extracted, LIMIT)
        self.assertEqual(total, LIMIT)
        self.assertEqual(actual, digest)


if __name__ == "__main__":
    unittest.main()
