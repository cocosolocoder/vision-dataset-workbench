from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from vision_workbench.exporting import ExportError
from vision_workbench.splits import SET_NAMES, SplitError
from vision_workbench.store import DatasetStore

REPO_ROOT = Path(__file__).resolve().parents[1]


class ExportHarness(unittest.TestCase):
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
    ) -> tuple[str, Path]:
        self._serial += 1
        path = self.root.parent / (name or f"sample-{self._serial}.jpg")
        path.write_bytes(
            content if content is not None else f"content-{self._serial}".encode()
        )
        result = self.store.add(path, label)
        self.assertTrue(result.added)
        return result.digest, path

    def export_bytes(
        self, plan_name: str = "p", *, skip_unlabeled: bool = False
    ) -> bytes:
        target = self.root.parent / f"out-{plan_name}.zip"
        target.unlink(missing_ok=True)
        self.store.export_plan(plan_name, target, skip_unlabeled=skip_unlabeled)
        return target.read_bytes()

    def unzip(self, blob: bytes) -> dict[str, bytes]:
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            self.assertIsNone(archive.testzip())
            return {info.filename: archive.read(info.filename) for info in archive.infolist()}


class ExportStructureTest(ExportHarness):
    def prepare_plan(self) -> None:
        for label in ("cat", "cat", "dog", "bird", None):
            self.add_sample(label)
        self.store.create_split("p", 42, ["0.6", "0.2", "0.2"])

    def test_archive_layout_is_imagefolder_by_set(self) -> None:
        self.prepare_plan()
        summary = self.store.export_plan(
            "p", self.root.parent / "pkg.zip", skip_unlabeled=True
        )
        with zipfile.ZipFile(self.root.parent / "pkg.zip") as archive:
            names = archive.namelist()
            self.assertIn("manifest.json", names)
            for set_name in SET_NAMES:
                self.assertIn(f"{set_name}/", names)
                for category in ("bird", "cat", "dog"):
                    self.assertIn(f"{set_name}/{category}/", names)
            # Every file sits exactly two levels below its set directory.
            files = [n for n in names if not n.endswith("/") and n != "manifest.json"]
            for name in files:
                parts = name.split("/")
                self.assertEqual(len(parts), 3)
                self.assertIn(parts[0], SET_NAMES)
                self.assertTrue(parts[2])
                self.assertNotIn("..", parts)
                # Normalizing must stay inside the set directory.
                normalized = os.path.normpath(name)
                self.assertTrue(normalized.startswith(parts[0] + "/"))
            manifest = json.loads(archive.read("manifest.json"))
            for sample in manifest["samples"]:
                self.assertIn(sample["path"], names)
                self.assertIn(sample["set"], SET_NAMES)
                # File name is the full digest plus source extension.
                file_name = sample["path"].split("/")[-1]
                self.assertTrue(
                    file_name.startswith(sample["sha256"]),
                    file_name,
                )
                data = archive.read(sample["path"])
                self.assertEqual(hashlib.sha256(data).hexdigest(), sample["sha256"])

        self.assertEqual(summary["plan"], "p")
        self.assertEqual(summary["exported"], 4)
        total_from_sets = sum(summary["sets"][s]["exported"] for s in SET_NAMES)
        self.assertEqual(total_from_sets, 4)
        # The skipped unlabeled sample is reported per set and overall.
        self.assertEqual(sum(summary["sets"][s]["skipped"] for s in SET_NAMES), 1)
        self.assertEqual(sum(summary["skipped"].values()), 1)

    def test_extension_is_preserved_or_dropped(self) -> None:
        self.add_sample("cat", b"a", name="photo.JPEG")
        self.add_sample("cat", b"bb", name="noext")
        self.store.create_split("ext", 0, [1, 0, 0])
        blob = self.export_bytes("ext")
        files = self.unzip(blob)
        image_files = [n for n in files if n.startswith("train/cat/")]
        suffixes = sorted(n.rsplit("/", 1)[-1].split(".", 1)[-1] if "." in n else ""
                          for n in image_files)
        self.assertIn("JPEG", suffixes)
        self.assertIn("", suffixes)

    def test_shared_class_table_and_numbering(self) -> None:
        for label in ("cat", "cat", "dog"):
            self.add_sample(label)
        # Put everything in train; validation/test must still know classes.
        self.store.create_split("p", 1, [1, 0, 0])
        blob = self.export_bytes("p")
        manifest = json.loads(self.unzip(blob)["manifest.json"])
        classes = manifest["classes"]
        self.assertEqual([c["directory"] for c in classes], ["cat", "dog"])
        self.assertEqual([c["index"] for c in classes], [0, 1])
        for set_summary in manifest["sets"]:
            # Every class appears in every set's distribution, even with 0.
            if set_summary["name"] == "train":
                self.assertEqual(set_summary["distribution"], {"cat": 2, "dog": 1})
            else:
                self.assertEqual(set_summary["distribution"], {"cat": 0, "dog": 0})
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            self.assertIn("validation/dog/", archive.namelist())
            self.assertIn("test/cat/", archive.namelist())

    def test_manifest_records_plan_seed_ratios_and_label_mapping(self) -> None:
        self.add_sample("猫/科", b"x")
        self.store.create_split("中文方案", 7, ["1/2", "1/4", "1/4"])
        blob = self.export_bytes("中文方案")
        manifest = json.loads(self.unzip(blob)["manifest.json"].decode("utf-8"))
        self.assertEqual(manifest["plan"], "中文方案")
        self.assertEqual(manifest["seed"], 7)
        self.assertEqual(
            manifest["ratios"],
            {"train": "1/2", "validation": "1/4", "test": "1/4"},
        )
        mapping = {c["label"]: c["directory"] for c in manifest["classes"]}
        self.assertIn("猫/科", mapping)
        self.assertNotIn("/", mapping["猫/科"])
        # No machine-specific absolute path anywhere in the package.
        raw = blob.decode("latin-1")
        self.assertNotIn(str(self.temp.name), raw)
        self.assertNotIn(str(self.root), raw)

    def test_empty_plan_exports_directories_and_empty_manifest(self) -> None:
        self.store.create_split("empty", 0, ["0.5", "0.25", "0.25"])
        target = self.root.parent / "empty.zip"
        summary = self.store.export_plan("empty", target)
        self.assertEqual(summary["exported"], 0)
        with zipfile.ZipFile(target) as archive:
            names = archive.namelist()
            self.assertEqual(sorted(names), ["manifest.json", "test/", "train/", "validation/"])
            manifest = json.loads(archive.read("manifest.json"))
        self.assertEqual(manifest["samples"], [])
        self.assertEqual(manifest["samples_total"], 0)
        self.assertEqual([s["samples"] for s in manifest["sets"]], [0, 0, 0])


class UnlabeledPolicyTest(ExportHarness):
    def test_any_unlabeled_sample_rejects_whole_export(self) -> None:
        digest, _ = self.add_sample(None, b"mystery")
        self.add_sample("cat", b"fine")
        self.store.create_split("p", 0, [1, 0, 0])
        target = self.root.parent / "blocked.zip"
        with self.assertRaises(ExportError) as caught:
            self.store.export_plan("p", target)
        self.assertIn(digest, str(caught.exception))
        self.assertFalse(target.exists())
        self.assertEqual(list(self.root.parent.glob(".*.tmp")), [])

    def test_empty_string_label_is_unlabeled_too(self) -> None:
        digest, _ = self.add_sample("", b"blank")
        self.store.create_split("p", 0, [1, 0, 0])
        with self.assertRaises(ExportError) as caught:
            self.store.export_plan("p", self.root.parent / "z.zip")
        self.assertIn(digest, str(caught.exception))

    def test_literal_unlabeled_class_is_a_normal_category(self) -> None:
        digest, _ = self.add_sample("unlabeled", b"real-class")
        self.store.create_split("p", 0, [1, 0, 0])
        summary = self.store.export_plan("p", self.root.parent / "lit.zip")
        self.assertEqual(summary["exported"], 1)
        files = self.unzip((self.root.parent / "lit.zip").read_bytes())
        self.assertIn(f"train/unlabeled/{digest}.jpg", files)

    def test_skip_unlabeled_counts_per_set_and_does_not_reassign(self) -> None:
        digests = [self.add_sample("cat", f"c{i}".encode())[0] for i in range(4)]
        unlabeled = [self.add_sample(None, f"u{i}".encode())[0] for i in range(3)]
        plan = self.store.create_split("p", 3, ["0.5", "0.25", "0.25"]).plan
        expected_by_set = {
            set_name: [
                m["sha256"]
                for m in plan["sets"][set_name]["members"]
                if m["label"] != ""
            ]
            for set_name in SET_NAMES
        }
        skipped_by_set = {
            set_name: plan["sets"][set_name]["samples"] - len(expected_by_set[set_name])
            for set_name in SET_NAMES
        }
        target = self.root.parent / "skipped.zip"
        summary = self.store.export_plan("p", target, skip_unlabeled=True)
        for set_name in SET_NAMES:
            self.assertEqual(
                summary["sets"][set_name]["skipped"], skipped_by_set[set_name]
            )
            self.assertEqual(
                summary["sets"][set_name]["exported"], len(expected_by_set[set_name])
            )
        with zipfile.ZipFile(target) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            packaged = {s["sha256"] for s in manifest["samples"]}
            by_set = {set_name: set() for set_name in SET_NAMES}
            for sample in manifest["samples"]:
                by_set[sample["set"]].add(sample["sha256"])
        self.assertEqual(set(by_set["train"]), set(expected_by_set["train"]))
        self.assertEqual(set(by_set["validation"]), set(expected_by_set["validation"]))
        self.assertEqual(set(by_set["test"]), set(expected_by_set["test"]))
        self.assertTrue(packaged.isdisjoint(unlabeled))
        self.assertEqual(
            manifest["skipped_unlabeled"],
            {set_name: skipped_by_set[set_name] for set_name in SET_NAMES},
        )


class TrickyLabelTest(ExportHarness):
    def test_distinct_identities_get_distinct_directories(self) -> None:
        cases = ["cat", "Cat", "CAT", "a/b", "a\\b", "狗", "狗 ", "con", "CON"]
        for i, label in enumerate(cases):
            self.add_sample(label, f"v{i}".encode())
        self.store.create_split("p", 0, [1, 0, 0])
        blob = self.export_bytes("p")
        manifest = json.loads(self.unzip(blob)["manifest.json"])
        labels = [c["label"] for c in manifest["classes"]]
        directories = [c["directory"] for c in manifest["classes"]]
        self.assertEqual(sorted(labels), sorted(cases))
        self.assertEqual(len(directories), len(set(directories)))
        self.assertEqual(
            len({d.casefold() for d in directories}), len(directories)
        )
        for directory in directories:
            self.assertNotIn("/", directory)
            self.assertNotIn("\\", directory)
            self.assertNotIn("..", directory)
        names = self.unzip(blob)
        for directory in directories:
            self.assertIn(f"train/{directory}/", names)

    def test_safe_extension_cannot_escape_set_directory(self) -> None:
        from vision_workbench.exporting import safe_extension

        self.assertEqual(safe_extension("a/b.jpg"), ".jpg")
        self.assertEqual(safe_extension("evil..\\x.png"), ".png")
        self.assertEqual(safe_extension("a.bin\x00"), ".bin")
        self.assertEqual(safe_extension(".jpg"), "")
        self.assertNotIn("\\", safe_extension("x.\\jpg"))


class DeterminismTest(ExportHarness):
    def test_identical_inputs_produce_identical_zip_bytes(self) -> None:
        for label, content in [("cat", b"one"), ("cat", b"two"), ("dog", b"three")]:
            self.add_sample(label, content)
        self.store.create_split("p", 99, ["0.5", "0.25", "0.25"])
        first = self.export_bytes("p")

        # Different run time, changed source mtimes, different target name.
        for path in self.root.parent.glob("sample-*"):
            os.utime(path, (1_000, 1_000))
        again = self.export_bytes("p")
        self.assertEqual(first, again)

        # Relocate the whole workspace; the archive must not depend on it.
        copy_parent = self.root.parent / "copy"
        shutil.copytree(self.root, copy_parent / "dataset")
        for path in self.root.parent.glob("sample-*"):
            shutil.copy2(path, copy_parent / path.name)
        moved = DatasetStore(copy_parent / "dataset")
        moved_target = copy_parent / "nested" / "result.zip"
        moved.export_plan("p", moved_target)
        self.assertEqual(moved_target.read_bytes(), first)

    def test_later_changes_do_not_affect_export_of_old_plan(self) -> None:
        first_digest, _ = self.add_sample("cat", b"alpha")
        second_digest, _ = self.add_sample("dog", b"beta")
        plan = self.store.create_split("p", 1, [1, 0, 0])
        before = self.export_bytes("p")

        # Batch relabel, an undo and a new import all happen after saving.
        self.store.submit_batch(
            {
                "batch": "b1",
                "changes": [
                    {"sha256": first_digest, "old": "cat", "new": "kitten"}
                ],
            }
        )
        self.store.undo_batch("b1")
        self.add_sample("bird", b"gamma")
        after = self.export_bytes("p")
        self.assertEqual(before, after)

        manifest = json.loads(self.unzip(after)["manifest.json"])
        packaged = {s["sha256"] for s in manifest["samples"]}
        self.assertEqual(packaged, {first_digest, second_digest})


class SourceFailureTest(ExportHarness):
    def _plan_with_source(self, content: bytes = b"payload") -> tuple[str, Path]:
        digest, path = self.add_sample("cat", content)
        self.store.create_split("p", 0, [1, 0, 0])
        return digest, path

    def test_missing_source_fails_and_leaves_nothing(self) -> None:
        digest, path = self._plan_with_source()
        path.unlink()
        target = self.root.parent / "fail.zip"
        with self.assertRaises(ExportError) as caught:
            self.store.export_plan("p", target)
        self.assertIn(digest, str(caught.exception))
        self.assertFalse(target.exists())
        self.assertEqual(list(self.root.parent.glob(".fail.zip*")), [])

    def test_non_regular_sources_fail(self) -> None:
        digest, path = self._plan_with_source()
        path.unlink()
        os.mkfifo(path)
        with self.assertRaises(ExportError) as caught:
            self.store.export_plan("p", self.root.parent / "fifo.zip")
        self.assertIn("regular file", str(caught.exception))

        os.unlink(path)
        (self.root / "a-directory").mkdir()
        path.symlink_to(self.root / "a-directory")
        with self.assertRaises(ExportError):
            self.store.export_plan("p", self.root.parent / "dir-link.zip")
        path.unlink()
        path.symlink_to(self.root / "missing-target")
        with self.assertRaises(ExportError):
            self.store.export_plan("p", self.root.parent / "broken-link.zip")

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root bypasses permissions")
    def test_unreadable_source_fails(self) -> None:
        digest, path = self._plan_with_source(b"secret")
        path.chmod(0)
        try:
            with self.assertRaises(ExportError):
                self.store.export_plan("p", self.root.parent / "denied.zip")
            self.assertFalse((self.root.parent / "denied.zip").exists())
        finally:
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)

    def test_digest_mismatch_fails_whole_export(self) -> None:
        digest, path = self._plan_with_source(b"original")
        other, other_path = self.add_sample("dog", b"second")
        plan = self.store.get_split("p")
        path.write_bytes(b"completely different bytes")
        target = self.root.parent / "tampered.zip"
        with self.assertRaises(ExportError) as caught:
            self.store.export_plan("p", target)
        self.assertIn(digest, str(caught.exception))
        self.assertIn("digest", str(caught.exception))
        self.assertFalse(target.exists())

    def test_change_during_read_cannot_impersonate_digest(self) -> None:
        digest, path = self._plan_with_source(b"original")
        from vision_workbench import exporting

        real_stream_member = exporting._stream_member

        def tampering_stream_member(archive, info, sample, force_zip64=False):
            # The file changes after verification but while streaming.
            path.write_bytes(b"original-mutated-after-verify")
            return real_stream_member(archive, info, sample, force_zip64)

        with mock.patch.object(exporting, "_stream_member", tampering_stream_member):
            with self.assertRaises(ExportError) as caught:
                self.store.export_plan("p", self.root.parent / "raced.zip")
        self.assertIn(digest, str(caught.exception))
        self.assertFalse((self.root.parent / "raced.zip").exists())

    def test_missing_or_corrupt_plan_fails(self) -> None:
        with self.assertRaises(SplitError):
            self.store.export_plan("ghost", self.root.parent / "g.zip")
        self.add_sample("cat", b"x")
        self.store.create_split("ok", 0, [1, 0, 0])
        plan_path = self.store.splits_directory / "ok.json"
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        del payload["sets"]
        plan_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises((SplitError, ExportError)):
            self.store.export_plan("ok", self.root.parent / "c.zip")
        self.assertFalse((self.root.parent / "c.zip").exists())


class TargetSafetyTest(ExportHarness):
    def test_existing_target_is_refused_and_preserved(self) -> None:
        self.add_sample("cat", b"x")
        self.store.create_split("p", 0, [1, 0, 0])
        target = self.root.parent / "keep.zip"
        target.write_bytes(b"PRECIOUS EXISTING CONTENT")
        with self.assertRaises(ExportError):
            self.store.export_plan("p", target)
        self.assertEqual(target.read_bytes(), b"PRECIOUS EXISTING CONTENT")

        # A symlink target (even to nothing) is also refused, not replaced.
        link = self.root.parent / "link.zip"
        link.symlink_to(self.root.parent / "does-not-exist")
        with self.assertRaises(ExportError):
            self.store.export_plan("p", link)
        self.assertTrue(link.is_symlink())

    def test_failure_removes_temp_and_rerun_succeeds(self) -> None:
        digest, path = self.add_sample("cat", b"ok")
        self.store.create_split("p", 0, [1, 0, 0])

        # A stale temp from an earlier interrupted attempt must not matter.
        (self.root.parent / ".p-retry.zip.abc123.tmp").write_bytes(b"partial")
        path.unlink()
        with self.assertRaises(ExportError):
            self.store.export_plan("p", self.root.parent / "p-retry.zip")
        self.assertFalse((self.root.parent / "p-retry.zip").exists())

        path.write_bytes(b"ok")
        self.store.export_plan("p", self.root.parent / "p-retry.zip")
        self.assertTrue((self.root.parent / "p-retry.zip").exists())
        temps = [
            p
            for p in self.root.parent.glob(".p-retry.zip*.tmp")
            if p.name != ".p-retry.zip.abc123.tmp"
        ]
        self.assertEqual(temps, [])

    def test_workspace_is_not_modified_by_export(self) -> None:
        self.add_sample("cat", b"x")
        self.store.create_split("p", 0, [1, 0, 0])
        before = {
            path: path.read_bytes()
            for path in self.store.state_directory.rglob("*")
            if path.is_file()
        }
        self.export_bytes("p")
        after = {
            path: path.read_bytes()
            for path in self.store.state_directory.rglob("*")
            if path.is_file()
        }
        self.assertEqual(before, after)


class StreamingTest(ExportHarness):
    def test_media_is_streamed_not_buffered(self) -> None:
        big = b"x" * (3 * 1024 * 1024)
        self.add_sample("cat", big, name="big1.bin")
        self.add_sample("cat", b"y" * (2 * 1024 * 1024), name="big2.bin")
        self.store.create_split("p", 0, [1, 0, 0])

        original_writestr = zipfile.ZipFile.writestr
        largest_buffered = 0

        def recording_writestr(self_zip, zinfo_or_arcname, data, *args, **kwargs):
            nonlocal largest_buffered
            if isinstance(data, bytes):
                largest_buffered = max(largest_buffered, len(data))
            return original_writestr(self_zip, zinfo_or_arcname, data, *args, **kwargs)

        target = self.root.parent / "streamed.zip"
        with mock.patch.object(zipfile.ZipFile, "writestr", recording_writestr):
            self.store.export_plan("p", target)
        # Dirs (empty) and manifest (digests only) go through writestr;
        # multi-megabyte media must never be buffered there.
        self.assertLess(largest_buffered, 100_000)
        with zipfile.ZipFile(target) as archive:
            sizes = sorted(
                len(archive.read(name))
                for name in archive.namelist()
                if name.endswith(".bin")
            )
        self.assertEqual(sizes, [2 * 1024 * 1024, 3 * 1024 * 1024])


class ExportCliTest(ExportHarness):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "vision_workbench", *arguments],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_cli_export_success_and_flags(self) -> None:
        for _ in range(3):
            self.add_sample("cat")
        self.add_sample(None)
        self.store.create_split("run", 5, ["0.5", "0.25", "0.25"])
        target = self.root.parent / "cli.zip"

        refused = self.run_cli("export", str(self.root), "run", str(target))
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("unlabeled", refused.stderr)

        ok = self.run_cli(
            "export", str(self.root), "run", str(target), "--skip-unlabeled"
        )
        self.assertEqual(ok.returncode, 0, ok.stderr)
        body = json.loads(ok.stdout)
        self.assertEqual(body["plan"], "run")
        self.assertEqual(body["exported"], 3)
        self.assertEqual(sum(body["skipped"].values()), 1)
        self.assertTrue(target.exists())

        again = self.run_cli(
            "export", str(self.root), "run", str(target), "--skip-unlabeled"
        )
        self.assertNotEqual(again.returncode, 0)
        self.assertIn("already exists", again.stderr)

    def test_concurrent_exports_one_wins(self) -> None:
        for _ in range(4):
            self.add_sample("cat", os.urandom(64))
        self.store.create_split("p", 0, [1, 0, 0])
        target = self.root.parent / "race.zip"

        processes = [
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "vision_workbench",
                    "export",
                    str(self.root),
                    "p",
                    str(target),
                ],
                cwd=REPO_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(8)
        ]
        outcomes = [process.communicate() for process in processes]
        results = [process.returncode for process in processes]
        self.assertEqual(results.count(0), 1, [err for _, err in outcomes])
        self.assertTrue(target.exists())
        with zipfile.ZipFile(target) as archive:
            self.assertIsNone(archive.testzip())


if __name__ == "__main__":
    unittest.main()
