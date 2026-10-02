from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from vision_workbench.exporter import ExportError, export_split
from vision_workbench.splits import SET_NAMES
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

    def add_sample(self, label: str | None, content: bytes | None = None) -> str:
        self._serial += 1
        path = self.root.parent / f"sample-{self._serial}.jpg"
        path.write_bytes(content if content is not None else f"content-{self._serial}".encode())
        result = self.store.add(path, label)
        self.assertTrue(result.added)
        return result.digest

    def create_plan(self, name: str = "baseline", seed: int = 42,
                    ratios: list[str] | None = None) -> dict:
        result = self.store.create_split(
            name, seed, ratios or ["0.5", "0.25", "0.25"]
        )
        return result.plan

    def export(self, plan: str = "baseline", target: Path | None = None,
               skip_unlabeled: bool = False) -> tuple[dict, Path]:
        target = target or self.root.parent / "out.zip"
        result = export_split(self.store, plan, target, skip_unlabeled=skip_unlabeled)
        return result, target

    def read_zip(self, target: Path) -> zipfile.ZipFile:
        return zipfile.ZipFile(target)


class ExportStructureTest(ExportHarness):
    def test_package_layout_manifest_and_content(self) -> None:
        d1 = self.add_sample("cat", b"cat-one")
        d2 = self.add_sample("cat", b"cat-two")
        d3 = self.add_sample("dog", b"dog-one")
        self.create_plan()

        result, target = self.export()
        self.assertEqual(result["plan"], "baseline")
        self.assertEqual(result["exported"], 3)
        self.assertEqual(result["skipped"], {"train": 0, "validation": 0, "test": 0})

        with self.read_zip(target) as archive:
            names = set(archive.namelist())
            for set_name in SET_NAMES:
                self.assertIn(f"{set_name}/", names)
                self.assertIn(f"{set_name}/class_01/", names)
                self.assertIn(f"{set_name}/class_02/", names)
            self.assertIn("manifest.json", names)

            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(manifest["plan"], "baseline")
            self.assertEqual(manifest["seed"], 42)
            self.assertEqual(
                manifest["ratios"],
                {"train": "1/2", "validation": "1/4", "test": "1/4"},
            )
            self.assertEqual(
                manifest["classes"],
                [
                    {"label": "cat", "directory": "class_01"},
                    {"label": "dog", "directory": "class_02"},
                ],
            )
            self.assertEqual(
                {s["sha256"] for s in manifest["samples"]}, {d1, d2, d3}
            )
            contents = {d1: b"cat-one", d2: b"cat-two", d3: b"dog-one"}
            for sample in manifest["samples"]:
                self.assertIn(sample["set"], SET_NAMES)
                self.assertTrue(sample["path"].startswith(f"{sample['set']}/"))
                self.assertEqual(archive.read(sample["path"]), contents[sample["sha256"]])
                # No absolute paths anywhere in the manifest.
                self.assertNotIn(str(self.root), sample["path"])
                self.assertFalse(sample["path"].startswith("/"))

            # File names are digest + source extension.
            for info in archive.infolist():
                if info.filename.endswith(".jpg"):
                    stem = Path(info.filename).stem
                    self.assertRegex(stem, r"^[0-9a-f]{64}$")

        # Per-set distributions match the plan.
        self.assertEqual(result["sets"]["train"]["samples"] +
                         result["sets"]["validation"]["samples"] +
                         result["sets"]["test"]["samples"], 3)
        self.assertEqual(result["sets"]["train"]["distribution"].get("cat", 0) +
                         result["sets"]["validation"]["distribution"].get("cat", 0) +
                         result["sets"]["test"]["distribution"].get("cat", 0), 2)

    def test_class_dirs_shared_even_when_absent_from_a_set(self) -> None:
        # One class only ever lands in train under these ratios; validation
        # and test must still expose the same class directory.
        self.add_sample("cat")
        self.add_sample("cat")
        self.add_sample("dog")
        self.create_plan(ratios=["1", "0", "0"])
        _, target = self.export()
        with self.read_zip(target) as archive:
            names = set(archive.namelist())
            for set_name in SET_NAMES:
                self.assertIn(f"{set_name}/class_01/", names)
                self.assertIn(f"{set_name}/class_02/", names)

    def test_empty_plan_exports_three_dirs_and_empty_manifest(self) -> None:
        self.create_plan("empty")
        result, target = self.export("empty")
        self.assertEqual(result["exported"], 0)
        self.assertEqual(result["skipped"], {"train": 0, "validation": 0, "test": 0})
        with self.read_zip(target) as archive:
            names = set(archive.namelist())
            for set_name in SET_NAMES:
                self.assertIn(f"{set_name}/", names)
                self.assertEqual(
                    [n for n in names if n.startswith(f"{set_name}/") and n != f"{set_name}/"],
                    [],
                )
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(manifest["samples"], [])
            self.assertEqual(manifest["classes"], [])
            for set_name in SET_NAMES:
                self.assertEqual(manifest["sets"][set_name],
                                 {"samples": 0, "skipped": 0})

    def test_chinese_label_keeps_identity(self) -> None:
        self.add_sample("猫")
        self.add_sample("狗")
        self.create_plan()
        _, target = self.export()
        with self.read_zip(target) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            labels = {c["label"]: c["directory"] for c in manifest["classes"]}
            self.assertEqual(set(labels), {"猫", "狗"})
            self.assertEqual(len(set(labels.values())), 2)

    def test_label_with_path_separator_keeps_identity_without_escape(self) -> None:
        self.add_sample("a/b")
        self.add_sample("a\\b")
        self.create_plan()
        _, target = self.export()
        with self.read_zip(target) as archive:
            names = set(archive.namelist())
            # No entry may escape its set directory.
            for name in names:
                if name.endswith("/") or name == "manifest.json":
                    continue
                parts = name.split("/")
                self.assertEqual(len(parts), 3, name)
                self.assertIn(parts[0], SET_NAMES)
                self.assertTrue(parts[1].startswith("class_"))
            manifest = json.loads(archive.read("manifest.json"))
            labels = {c["label"] for c in manifest["classes"]}
            self.assertEqual(labels, {"a/b", "a\\b"})

    def test_case_only_different_labels_get_distinct_dirs(self) -> None:
        self.add_sample("Cat")
        self.add_sample("cat")
        self.create_plan()
        _, target = self.export()
        with self.read_zip(target) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            dirs = {c["label"]: c["directory"] for c in manifest["classes"]}
            self.assertNotEqual(dirs["Cat"], dirs["cat"])

    def test_literal_unlabeled_is_a_normal_class(self) -> None:
        self.add_sample("unlabeled")
        self.create_plan()
        result, target = self.export()
        self.assertEqual(result["exported"], 1)
        self.assertEqual(result["skipped"], {"train": 0, "validation": 0, "test": 0})
        with self.read_zip(target) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual([c["label"] for c in manifest["classes"]], ["unlabeled"])
            self.assertEqual(manifest["samples"][0]["label"], "unlabeled")


class UnlabeledExportTest(ExportHarness):
    def test_unlabeled_rejected_by_default_naming_digest(self) -> None:
        d = self.add_sample(None)
        self.add_sample("cat")
        self.create_plan()
        target = self.root.parent / "out.zip"
        with self.assertRaises(ExportError) as caught:
            export_split(self.store, "baseline", target)
        self.assertIn(d, str(caught.exception))
        self.assertIn("--skip-unlabeled", str(caught.exception))
        self.assertFalse(target.exists())

    def test_skip_unlabeled_counts_per_set_and_keeps_labeled_counts(self) -> None:
        # All samples land in train under 1/0/0; the unlabeled one must be
        # skipped, not redistributed.
        d_labeled = self.add_sample("cat", b"labeled")
        d_unlabeled = self.add_sample(None, b"unlabeled")
        self.create_plan(ratios=["1", "0", "0"])
        plan = self.store.get_split("baseline")
        result, target = self.export(skip_unlabeled=True)

        self.assertEqual(result["exported"], 1)
        self.assertEqual(result["skipped"], {"train": 1, "validation": 0, "test": 0})
        self.assertEqual(result["sets"]["train"]["samples"], 1)
        self.assertEqual(result["sets"]["train"]["distribution"], {"cat": 1})

        with self.read_zip(target) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(manifest["sets"]["train"]["skipped"], 1)
            self.assertEqual(manifest["sets"]["validation"]["skipped"], 0)
            self.assertEqual(manifest["sets"]["test"]["skipped"], 0)
            self.assertEqual(len(manifest["samples"]), 1)
            self.assertEqual(manifest["samples"][0]["sha256"], d_labeled)
            self.assertEqual(archive.read(manifest["samples"][0]["path"]), b"labeled")

    def test_skip_empty_string_label(self) -> None:
        self.add_sample("")
        self.create_plan()
        result, _ = self.export(skip_unlabeled=True)
        self.assertEqual(result["exported"], 0)
        self.assertEqual(result["skipped"], {"train": 1, "validation": 0, "test": 0})


class DeterminismTest(ExportHarness):
    def test_reexport_bytes_identical(self) -> None:
        self.add_sample("cat", b"abc")
        self.add_sample("dog", b"def")
        self.create_plan()
        first = self.root.parent / "first.zip"
        second = self.root.parent / "second.zip"
        export_split(self.store, "baseline", first)
        export_split(self.store, "baseline", second)
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_bytes_independent_of_source_mtime(self) -> None:
        self.add_sample("cat", b"abc")
        self.create_plan()
        first = self.root.parent / "first.zip"
        second = self.root.parent / "second.zip"
        export_split(self.store, "baseline", first)
        source = self.store.get_split("baseline")["sets"]["train"]["members"][0]["source"]
        os.utime(source, (0, 0))
        export_split(self.store, "baseline", second)
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_bytes_independent_of_workspace_location(self) -> None:
        self.add_sample("cat", b"abc")
        self.create_plan()
        first = self.root.parent / "first.zip"
        export_split(self.store, "baseline", first)

        relocated = Path(self.temp.name) / "relocated"
        shutil_copytree(self.root, relocated)
        second = relocated.parent / "second.zip"
        export_split(DatasetStore(relocated), "baseline", second)
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_later_imports_and_label_changes_do_not_change_export(self) -> None:
        d = self.add_sample("cat", b"abc")
        self.create_plan()
        first = self.root.parent / "first.zip"
        export_split(self.store, "baseline", first)

        # Later import and a label change on the exported sample.
        self.add_sample("dog", b"woof")
        self.store.submit_batch(
            {"batch": "b1", "changes": [{"sha256": d, "old": "cat", "new": "kitten"}]}
        )
        second = self.root.parent / "second.zip"
        export_split(self.store, "baseline", second)
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_export_does_not_modify_workspace(self) -> None:
        self.add_sample("cat")
        self.create_plan()
        manifest_before = self.store.manifest_path.read_bytes()
        plans_before = sorted(p.name for p in self.store.splits_directory.iterdir())
        self.export()
        self.assertEqual(self.store.manifest_path.read_bytes(), manifest_before)
        self.assertEqual(
            sorted(p.name for p in self.store.splits_directory.iterdir()), plans_before
        )


class SourceFailureTest(ExportHarness):
    def test_missing_source_fails_with_digest_and_leaves_no_target(self) -> None:
        d = self.add_sample("cat", b"abc")
        self.create_plan()
        source = Path(self.store.get_split("baseline")["sets"]["train"]["members"][0]["source"])
        source.unlink()
        target = self.root.parent / "out.zip"
        with self.assertRaises(ExportError) as caught:
            export_split(self.store, "baseline", target)
        self.assertIn(d, str(caught.exception))
        self.assertFalse(target.exists())

    def test_source_not_regular_file_fails(self) -> None:
        d = self.add_sample("cat", b"abc")
        self.create_plan()
        source = Path(self.store.get_split("baseline")["sets"]["train"]["members"][0]["source"])
        source.unlink()
        source.mkdir()
        target = self.root.parent / "out.zip"
        with self.assertRaises(ExportError) as caught:
            export_split(self.store, "baseline", target)
        self.assertIn(d, str(caught.exception))
        self.assertIn("not a regular file", str(caught.exception))
        self.assertFalse(target.exists())

    def test_source_content_mismatch_fails_and_leaves_no_target(self) -> None:
        d = self.add_sample("cat", b"abc")
        self.create_plan()
        source = Path(self.store.get_split("baseline")["sets"]["train"]["members"][0]["source"])
        source.write_bytes(b"changed-after-plan")
        target = self.root.parent / "out.zip"
        with self.assertRaises(ExportError) as caught:
            export_split(self.store, "baseline", target)
        self.assertIn(d, str(caught.exception))
        self.assertIn("digest", str(caught.exception))
        self.assertFalse(target.exists())

    def test_missing_plan_fails(self) -> None:
        target = self.root.parent / "out.zip"
        with self.assertRaises(Exception):
            export_split(self.store, "nope", target)
        self.assertFalse(target.exists())

    def test_corrupted_plan_fails(self) -> None:
        self.add_sample("cat")
        self.create_plan()
        plan_path = self.store.splits_directory / "baseline.json"
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        payload["sets"]["train"]["members"][0]["sha256"] = "not-a-digest"
        plan_path.write_text(json.dumps(payload), encoding="utf-8")
        target = self.root.parent / "out.zip"
        with self.assertRaises(ExportError):
            export_split(self.store, "baseline", target)
        self.assertFalse(target.exists())


class TargetHandlingTest(ExportHarness):
    def test_existing_target_rejected_and_preserved(self) -> None:
        self.add_sample("cat")
        self.create_plan()
        target = self.root.parent / "out.zip"
        export_split(self.store, "baseline", target)
        original = target.read_bytes()
        with self.assertRaises(ExportError):
            export_split(self.store, "baseline", target)
        self.assertEqual(target.read_bytes(), original)

    def test_concurrent_exports_to_same_target_one_succeeds(self) -> None:
        self.add_sample("cat", b"abc")
        self.create_plan()
        target = self.root.parent / "out.zip"
        cli = [sys.executable, "-m", "vision_workbench", "export",
               str(self.root), "baseline", str(target)]
        processes = [
            subprocess.Popen(cli, cwd=REPO_ROOT, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            for _ in range(3)
        ]
        results = [p.wait() for p in processes]
        self.assertEqual(results.count(0), 1, results)
        self.assertTrue(target.exists())
        with zipfile.ZipFile(target) as archive:
            self.assertIn("manifest.json", archive.namelist())


class CliTest(ExportHarness):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "vision_workbench", *arguments],
            cwd=REPO_ROOT, text=True, capture_output=True, check=False,
        )

    def test_cli_export_success_json(self) -> None:
        self.add_sample("cat")
        self.add_sample("dog")
        self.create_plan()
        target = self.root.parent / "cli.zip"
        result = self.run_cli("export", str(self.root), "baseline", str(target))
        self.assertEqual(result.returncode, 0, result.stderr)
        body = json.loads(result.stdout)
        self.assertEqual(body["plan"], "baseline")
        self.assertEqual(body["exported"], 2)
        self.assertEqual(body["skipped"], {"train": 0, "validation": 0, "test": 0})
        self.assertTrue(target.exists())

    def test_cli_export_reports_unlabeled_and_skip_flag(self) -> None:
        self.add_sample(None)
        self.create_plan()
        target = self.root.parent / "cli.zip"
        rejected = self.run_cli("export", str(self.root), "baseline", str(target))
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("unlabeled", rejected.stderr)
        self.assertFalse(target.exists())

        skipped = self.run_cli(
            "export", str(self.root), "baseline", str(target), "--skip-unlabeled"
        )
        self.assertEqual(skipped.returncode, 0, skipped.stderr)
        body = json.loads(skipped.stdout)
        self.assertEqual(body["exported"], 0)
        self.assertEqual(body["skipped"], {"train": 1, "validation": 0, "test": 0})


class SourceDirExportTest(ExportHarness):
    def setUp(self) -> None:
        super().setUp()
        self.moved = Path(self.temp.name) / "moved"

    def move_sources(self, renames: dict[str, str] | None = None) -> Path:
        """Move every recorded source into self.moved, optionally renaming."""
        plan = self.store.get_split("baseline")
        sources = []
        for set_name in SET_NAMES:
            for member in plan["sets"][set_name]["members"]:
                sources.append(member["source"])
        for index, source in enumerate(sources):
            old = Path(source)
            new_name = (renames or {}).get(old.name, old.name)
            new_path = self.moved / new_name
            new_path.parent.mkdir(parents=True, exist_ok=True)
            old.replace(new_path)
        return self.moved

    def test_export_from_moved_tree_byte_identical(self) -> None:
        self.add_sample("cat", b"cat-one")
        self.add_sample("dog", b"dog-one")
        self.create_plan()
        baseline = self.root.parent / "baseline.zip"
        export_split(self.store, "baseline", baseline)

        moved = self.move_sources()
        target = self.root.parent / "out.zip"
        result = export_split(
            self.store, "baseline", target, source_dir=moved
        )
        self.assertEqual(result["exported"], 2)
        self.assertEqual(target.read_bytes(), baseline.read_bytes())

    def test_renamed_files_match_by_content_and_keep_plan_extension(self) -> None:
        d = self.add_sample("cat", b"cat-one")
        self.create_plan()
        baseline = self.root.parent / "baseline.zip"
        export_split(self.store, "baseline", baseline)

        moved = self.move_sources(renames={"sample-1.jpg": "nested/renamed.bin"})
        target = self.root.parent / "out.zip"
        export_split(self.store, "baseline", target, source_dir=moved)
        self.assertEqual(target.read_bytes(), baseline.read_bytes())
        with self.read_zip(target) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            entry = [s for s in manifest["samples"] if s["sha256"] == d][0]
            # Extension still comes from the plan, not the new file name.
            self.assertTrue(entry["path"].endswith(".jpg"), entry["path"])
            self.assertNotIn(str(moved), json.dumps(manifest))

    def test_duplicate_copies_pick_first_relative_path(self) -> None:
        self.add_sample("cat", b"cat-one")
        self.create_plan()
        moved = self.move_sources()
        # Extra copies of the same content; the export must not change.
        (moved / "aaa").mkdir()
        (moved / "aaa" / "copy.dat").write_bytes(b"cat-one")
        (moved / "zzz").mkdir()
        (moved / "zzz" / "copy.dat").write_bytes(b"cat-one")
        first = self.root.parent / "first.zip"
        export_split(self.store, "baseline", first, source_dir=moved)

        # Only the lexicographically first copy is read: corrupting the
        # others after a fresh scan must not matter, and the bytes are
        # identical to a copy-free export.
        (moved / "zzz" / "copy.dat").write_bytes(b"corrupted")
        (moved / "aaa" / "copy.dat").replace(moved / "sole.dat")
        import shutil
        shutil.rmtree(moved / "aaa")
        shutil.rmtree(moved / "zzz")
        second = self.root.parent / "second.zip"
        export_split(self.store, "baseline", second, source_dir=moved)
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_unrelated_files_not_exported(self) -> None:
        self.add_sample("cat", b"cat-one")
        self.create_plan()
        moved = self.move_sources()
        (moved / "stray.txt").write_bytes(b"not in the plan")
        target = self.root.parent / "out.zip"
        result = export_split(self.store, "baseline", target, source_dir=moved)
        self.assertEqual(result["exported"], 1)
        with self.read_zip(target) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(len(manifest["samples"]), 1)

    def test_missing_sample_in_source_dir_fails_with_digest(self) -> None:
        self.add_sample("cat", b"cat-one")
        self.add_sample("dog", b"dog-one")
        self.create_plan()
        moved = self.move_sources()
        # Remove the dog sample's file from the new tree.
        for path in moved.rglob("*"):
            if path.is_file() and path.read_bytes() == b"dog-one":
                path.unlink()
        target = self.root.parent / "out.zip"
        with self.assertRaises(ExportError) as caught:
            export_split(self.store, "baseline", target, source_dir=moved)
        message = str(caught.exception)
        self.assertFalse(target.exists())
        # Exactly the missing sample's digest must be named.
        plan = self.store.get_split("baseline")
        digests = [
            m["sha256"]
            for s in SET_NAMES
            for m in plan["sets"][s]["members"]
        ]
        named = [digest for digest in digests if digest in message]
        self.assertEqual(len(named), 1, message)

    def test_same_name_different_content_does_not_substitute(self) -> None:
        self.add_sample("cat", b"cat-one")
        self.create_plan()
        moved = self.move_sources()
        # Replace the moved file with different content under the same name.
        for path in moved.rglob("*"):
            if path.is_file():
                path.write_bytes(b"cat-one-but-not-really")
        target = self.root.parent / "out.zip"
        with self.assertRaises(ExportError):
            export_split(self.store, "baseline", target, source_dir=moved)
        self.assertFalse(target.exists())

    def test_original_path_not_used_as_fallback(self) -> None:
        self.add_sample("cat", b"cat-one")
        self.create_plan()
        # Copy (not move) so the recorded source path still works.
        plan = self.store.get_split("baseline")
        source = Path(plan["sets"]["train"]["members"][0]["source"])
        self.moved.mkdir()
        (self.moved / "unrelated.dat").write_bytes(b"other content")
        target = self.root.parent / "out.zip"
        with self.assertRaises(ExportError) as caught:
            export_split(self.store, "baseline", target, source_dir=self.moved)
        self.assertTrue(source.exists())  # original still available, unused
        self.assertFalse(target.exists())

    def test_symlinks_are_skipped(self) -> None:
        self.add_sample("cat", b"cat-one")
        self.create_plan()
        moved = self.move_sources()
        real_files = [p for p in moved.rglob("*") if p.is_file()]
        self.assertEqual(len(real_files), 1)
        real = real_files[0]
        hidden = Path(self.temp.name) / "hidden"
        hidden.mkdir()
        real.replace(hidden / "real.jpg")
        os.symlink(hidden / "real.jpg", moved / "link.jpg")
        os.symlink(hidden, moved / "linkdir")
        target = self.root.parent / "out.zip"
        with self.assertRaises(ExportError):
            export_split(self.store, "baseline", target, source_dir=moved)
        self.assertFalse(target.exists())

    def test_source_dir_must_exist_and_be_a_directory(self) -> None:
        self.add_sample("cat", b"cat-one")
        self.create_plan()
        target = self.root.parent / "out.zip"
        missing = Path(self.temp.name) / "nope"
        with self.assertRaises(ExportError) as caught:
            export_split(self.store, "baseline", target, source_dir=missing)
        self.assertIn(str(missing), str(caught.exception))
        not_dir = Path(self.temp.name) / "a-file"
        not_dir.write_bytes(b"x")
        with self.assertRaises(ExportError) as caught:
            export_split(self.store, "baseline", target, source_dir=not_dir)
        self.assertIn(str(not_dir), str(caught.exception))
        self.assertFalse(target.exists())

    def test_empty_plan_only_checks_source_dir_exists(self) -> None:
        self.create_plan("empty")
        self.moved.mkdir()
        (self.moved / "anything.dat").write_bytes(b"irrelevant")
        target = self.root.parent / "out.zip"
        result = export_split(self.store, "empty", target, source_dir=self.moved)
        self.assertEqual(result["exported"], 0)
        with self.read_zip(target) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(manifest["samples"], [])

        missing_target = self.root.parent / "other.zip"
        with self.assertRaises(ExportError):
            export_split(
                self.store, "empty", missing_target,
                source_dir=Path(self.temp.name) / "nope",
            )
        self.assertFalse(missing_target.exists())

    def test_all_skipped_only_checks_source_dir_exists(self) -> None:
        self.add_sample(None, b"unlabeled")
        self.create_plan(ratios=["1", "0", "0"])
        self.moved.mkdir()
        target = self.root.parent / "out.zip"
        result = export_split(
            self.store, "baseline", target,
            skip_unlabeled=True, source_dir=self.moved,
        )
        self.assertEqual(result["exported"], 0)
        self.assertEqual(result["skipped"], {"train": 1, "validation": 0, "test": 0})

    def test_skip_unlabeled_needs_no_source_for_skipped(self) -> None:
        self.add_sample("cat", b"labeled")
        self.add_sample(None, b"unlabeled")
        self.create_plan(ratios=["1", "0", "0"])
        moved = self.move_sources()
        # Remove the unlabeled sample's file; only the labeled one remains.
        for path in moved.rglob("*"):
            if path.is_file() and path.read_bytes() == b"unlabeled":
                path.unlink()
        target = self.root.parent / "out.zip"
        result = export_split(
            self.store, "baseline", target,
            skip_unlabeled=True, source_dir=moved,
        )
        self.assertEqual(result["exported"], 1)
        self.assertEqual(result["skipped"], {"train": 1, "validation": 0, "test": 0})

    def test_selected_file_replaced_before_copy_fails(self) -> None:
        self.add_sample("cat", b"cat-one")
        self.create_plan()
        moved = self.move_sources()

        import vision_workbench.exporter as exporter

        original_hash = exporter._hash_file

        def corrupting_hash(path: Path) -> str:
            digest = original_hash(path)
            # Replace the file after it has been indexed, before the copy.
            Path(path).write_bytes(b"swapped-after-scan")
            return digest

        exporter._hash_file = corrupting_hash
        try:
            target = self.root.parent / "out.zip"
            with self.assertRaises(ExportError) as caught:
                export_split(self.store, "baseline", target, source_dir=moved)
            self.assertIn("digest", str(caught.exception))
            self.assertFalse(target.exists())
        finally:
            exporter._hash_file = original_hash

    def test_source_dir_does_not_write_to_workspace(self) -> None:
        self.add_sample("cat", b"cat-one")
        self.create_plan()
        moved = self.move_sources()
        manifest_before = self.store.manifest_path.read_bytes()
        plans_before = {
            p.name: p.read_bytes() for p in self.store.splits_directory.iterdir()
        }
        export_split(
            self.store, "baseline", self.root.parent / "out.zip",
            source_dir=moved,
        )
        self.assertEqual(self.store.manifest_path.read_bytes(), manifest_before)
        self.assertEqual(
            {p.name: p.read_bytes() for p in self.store.splits_directory.iterdir()},
            plans_before,
        )

    def test_cli_export_with_source_dir(self) -> None:
        self.add_sample("cat", b"cat-one")
        self.create_plan()
        moved = self.move_sources()
        target = self.root.parent / "cli.zip"
        result = subprocess.run(
            [sys.executable, "-m", "vision_workbench", "export",
             str(self.root), "baseline", str(target),
             "--source-dir", str(moved)],
            cwd=REPO_ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        body = json.loads(result.stdout)
        self.assertEqual(body["exported"], 1)
        self.assertTrue(target.exists())


def shutil_copytree(src: Path, dst: Path) -> None:
    import shutil

    shutil.copytree(src, dst)


if __name__ == "__main__":
    unittest.main()
