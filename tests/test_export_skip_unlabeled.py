"""Regression coverage for exporting a saved plan with ``--skip-unlabeled``.

The rule under test is narrow: samples whose label was null or the empty
string *when the plan was saved* are omitted from the package, while every
other sample is packaged exactly as the plan records it.  These tests pin
the contract from three directions at once, for the train, validation and
test sets independently:

* the result payload printed on the terminal (total, per-set samples and
  class distribution, per-set skip counts);
* the ``manifest.json`` carried inside the ZIP (sample list, per-set
  exported/skipped counts, class list);
* the ZIP entries and the bytes actually packaged.

The three must agree with each other and with the members of the saved
plan: kept samples keep their original set, label and file content,
skipped samples appear nowhere, skip counts stay attributed to each
sample's original set (remaining samples are never moved to approach the
old ratios), and every decision uses the labels snapshotted in the plan
rather than the workspace's current labels.  A genuine class literally
named ``"unlabeled"`` is a normal class and must never be omitted.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path

from vision_workbench.exporter import ExportError, export_split
from vision_workbench.splits import SET_NAMES
from vision_workbench.store import DatasetStore

REPO_ROOT = Path(__file__).resolve().parents[1]


class SkipUnlabeledHarness(unittest.TestCase):
    def setUp(self) -> None:
        self._workspaces: list[DatasetStore] = []
        self.temp_root = Path(self._make_temp())
        self.serial = 0
        self.store = self.fresh_store("main")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def _make_temp(self) -> str:
        return tempfile.mkdtemp(prefix="skip-unlabeled-")

    def fresh_store(self, name: str) -> DatasetStore:
        root = self.temp_root / name
        store = DatasetStore(root)
        store.initialize()
        self._workspaces.append(store)
        return store

    def add_sample(
        self, store: DatasetStore, label: str | None, content: bytes | None = None
    ) -> str:
        self.serial += 1
        path = self.temp_root / f"sample-{self.serial}.jpg"
        path.write_bytes(
            content if content is not None else f"content-{self.serial}".encode()
        )
        result = store.add(path, label)
        self.assertTrue(result.added)
        return result.digest

    def content_of(self, digest: str) -> bytes:
        for path in self.temp_root.glob("sample-*.jpg"):
            if hashlib.sha256(path.read_bytes()).hexdigest() == digest:
                return path.read_bytes()
        raise AssertionError(f"no source file found for {digest}")

    def export(
        self,
        store: DatasetStore,
        plan: str,
        target_name: str,
        skip_unlabeled: bool = False,
        source_dir: Path | None = None,
    ) -> tuple[dict, Path]:
        target = self.temp_root / target_name
        result = export_split(
            store,
            plan,
            target,
            skip_unlabeled=skip_unlabeled,
            source_dir=source_dir,
        )
        return result, target

    def read_package(self, target: Path) -> tuple[dict, set[str], dict[str, bytes]]:
        """Return (manifest, all entry names, packaged file bytes by path)."""
        with zipfile.ZipFile(target) as archive:
            names = set(archive.namelist())
            manifest = json.loads(archive.read("manifest.json"))
            files = {
                info.filename: archive.read(info.filename)
                for info in archive.infolist()
                if not info.is_dir() and info.filename != "manifest.json"
            }
        return manifest, names, files

    def rewrite_plan(
        self, store: DatasetStore, name: str, assignments: dict[str, list[str]]
    ) -> dict:
        """Rewrite a saved plan, moving its members to the given sets.

        ``assignments`` must partition every existing member.  All summary
        statistics (per-set samples/distribution and the overall total and
        distribution) are rebuilt from the moved member records, which keep
        their snapshotted labels and sources.
        """
        plan_path = store.splits_directory / f"{name}.json"
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        records: dict[str, dict] = {}
        for set_name in SET_NAMES:
            for member in payload["sets"][set_name]["members"]:
                records[member["sha256"]] = member

        assigned = [digest for set_name in SET_NAMES for digest in assignments[set_name]]
        self.assertEqual(sorted(assigned), sorted(records), "plan rewrite must partition all members")

        sets: dict[str, dict] = {}
        overall: Counter[str] = Counter()
        for set_name in SET_NAMES:
            members = [records[digest] for digest in assignments[set_name]]
            distribution = Counter(member["label"] for member in members)
            sets[set_name] = {
                "samples": len(members),
                "distribution": dict(sorted(distribution.items())),
                "members": members,
            }
            overall.update(distribution)
        payload["sets"] = sets
        payload["samples"] = {
            "total": len(records),
            "distribution": dict(sorted(overall.items())),
        }
        plan_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def members_by_set(self, plan: dict, label: str | None = None) -> dict[str, list[dict]]:
        result = {set_name: [] for set_name in SET_NAMES}
        for set_name in SET_NAMES:
            for member in plan["sets"][set_name]["members"]:
                if label is None or member["label"] == label:
                    result[set_name].append(member)
        return result

    def assert_package_consistent(
        self,
        result: dict,
        target: Path,
        plan: dict,
        expected_kept: dict[str, list[dict]],
        expected_skipped: dict[str, int],
    ) -> tuple[dict, set[str], dict[str, bytes]]:
        """Cross-check result payload, manifest and ZIP against expectations."""
        self.assertEqual(result["plan"], plan["name"])
        self.assertTrue(target.exists())

        total_kept = sum(len(members) for members in expected_kept.values())
        total_skipped = sum(expected_skipped.values())
        plan_total = sum(
            plan["sets"][set_name]["samples"] for set_name in SET_NAMES
        )
        self.assertEqual(total_kept + total_skipped, plan_total)

        # Terminal result: totals and per-set statistics describe only the
        # records actually packaged; skip counts stay on their origin sets.
        self.assertEqual(result["exported"], total_kept)
        self.assertEqual(result["skipped"], dict(expected_skipped))
        expected_distributions: dict[str, dict[str, int]] = {}
        for set_name in SET_NAMES:
            distribution = Counter(
                member["label"] for member in expected_kept[set_name]
            )
            expected_distributions[set_name] = dict(sorted(distribution.items()))
            self.assertEqual(
                result["sets"][set_name]["samples"],
                len(expected_kept[set_name]),
                set_name,
            )
            self.assertEqual(
                result["sets"][set_name]["distribution"],
                expected_distributions[set_name],
                set_name,
            )

        manifest, names, files = self.read_package(target)

        # All three set directories are always present, even when a set
        # exported zero samples.
        for set_name in SET_NAMES:
            self.assertIn(f"{set_name}/", names)

        # Manifest header is the saved plan's, untouched.
        self.assertEqual(manifest["plan"], plan["name"])
        self.assertEqual(manifest["seed"], plan["seed"])
        self.assertEqual(manifest["ratios"], plan["ratios"])

        # Per-set manifest counts agree with the terminal result.
        for set_name in SET_NAMES:
            self.assertEqual(
                manifest["sets"][set_name],
                {
                    "samples": len(expected_kept[set_name]),
                    "skipped": expected_skipped[set_name],
                },
                set_name,
            )

        kept_labels = {
            member["label"]
            for members in expected_kept.values()
            for member in members
        }
        self.assertEqual(
            [entry["label"] for entry in manifest["classes"]],
            sorted(kept_labels),
        )
        label_directories = {
            entry["label"]: entry["directory"] for entry in manifest["classes"]
        }

        # The manifest sample list is exactly the kept plan members.
        sample_records = {
            entry["sha256"]: entry for entry in manifest["samples"]
        }
        expected_records = {
            member["sha256"]: (set_name, member)
            for set_name in SET_NAMES
            for member in expected_kept[set_name]
        }
        self.assertEqual(set(sample_records), set(expected_records))
        self.assertEqual(len(manifest["samples"]), total_kept)

        skipped_digests = {
            member["sha256"]
            for set_name in SET_NAMES
            for member in plan["sets"][set_name]["members"]
            if member["sha256"] not in expected_records
        }
        self.assertEqual(len(skipped_digests), total_skipped)

        packaged_paths: set[str] = set()
        for digest, entry in sample_records.items():
            set_name, member = expected_records[digest]
            # Kept samples keep the set, label and identity the saved plan
            # gave them; nothing was redistributed to restore old ratios.
            self.assertEqual(entry["set"], set_name)
            self.assertEqual(entry["label"], member["label"])
            self.assertEqual(entry["sha256"], member["sha256"])
            expected_path = (
                f"{set_name}/{label_directories[member['label']]}/"
                f"{member['sha256']}{Path(member['source']).suffix}"
            )
            self.assertEqual(entry["path"], expected_path)
            packaged_paths.add(expected_path)

        # Every packaged file corresponds to a manifest entry, and only to
        # manifest entries; bytes are unchanged from the recorded source.
        self.assertEqual(set(files), packaged_paths)
        for digest, entry in sample_records.items():
            self.assertEqual(files[entry["path"]], self.content_of(digest))

        # Skipped samples appear neither in the sample list nor as files.
        for digest in skipped_digests:
            self.assertNotIn(digest, sample_records)
            self.assertTrue(
                all(digest not in name for name in names),
                f"skipped sample {digest} packaged under some entry",
            )

        # Shared numbered class directories exist under every set for every
        # class that survived, including sets with zero surviving samples.
        for directory in label_directories.values():
            for set_name in SET_NAMES:
                self.assertIn(f"{set_name}/{directory}/", names)

        return manifest, names, files

    def make_mixed_plan(self, store: DatasetStore, name: str) -> tuple[dict, dict]:
        """Create a plan mixing labeled and unlabeled samples in all sets.

        With seed 11 and equal thirds, six ``cat`` samples plus three
        samples of the genuine class literally named ``"unlabeled"`` plus
        six null-label and three empty-string-label samples stratify to
        exactly 2 cat / 1 literal-unlabeled / 3 unlabeled per set.

        Returns ``(plan, originals)`` where ``originals`` maps each digest
        to the label it was imported with (``None`` for the null spelling).
        """
        originals: dict[str, str | None] = {}
        for content in (f"cat-{i}" for i in range(6)):
            originals[self.add_sample(store, "cat", content.encode())] = "cat"
        for content in (f"lit-{i}" for i in range(3)):
            originals[
                self.add_sample(store, "unlabeled", content.encode())
            ] = "unlabeled"
        for content in (f"null-{i}" for i in range(6)):
            originals[self.add_sample(store, None, content.encode())] = None
        for content in (f"empty-{i}" for i in range(3)):
            originals[self.add_sample(store, "", content.encode())] = ""

        plan = store.create_split(name, seed=11, ratios=["1/3", "1/3", "1/3"]).plan
        for set_name in SET_NAMES:
            distribution = Counter(
                member["label"]
                for member in plan["sets"][set_name]["members"]
            )
            self.assertEqual(
                dict(distribution), {"": 3, "cat": 2, "unlabeled": 1}, set_name
            )
        return plan, originals


class SkipUnlabeledMixedTest(SkipUnlabeledHarness):
    def test_skip_mixed_plan_stats_manifest_and_package_agree(self) -> None:
        plan, originals = self.make_mixed_plan(self.store, "mixed")

        # Both unlabeled spellings are stored as the empty string in the
        # saved plan, while the genuine "unlabeled" class stays literal.
        null_and_empty = [d for d, label in originals.items() if label in (None, "")]
        literal = [d for d, label in originals.items() if label == "unlabeled"]
        saved_labels = {
            member["sha256"]: member["label"]
            for set_name in SET_NAMES
            for member in plan["sets"][set_name]["members"]
        }
        for digest in null_and_empty:
            self.assertEqual(saved_labels[digest], "")
        for digest in literal:
            self.assertEqual(saved_labels[digest], "unlabeled")
        self.assertEqual(Counter(saved_labels[d] for d in null_and_empty), Counter({"": 9}))

        # Move the literal-unlabeled members from validation and test into
        # train, so validation/test each have exactly two labeled records
        # and three unlabeled ones (the spec's worked example).  The
        # exporter must not perform any movement of its own.
        literal_by_set = self.members_by_set(plan, "unlabeled")
        assignments = {
            set_name: [
                member["sha256"]
                for member in plan["sets"][set_name]["members"]
            ]
            for set_name in SET_NAMES
        }
        for set_name in ("validation", "test"):
            assignments[set_name] = [
                digest
                for digest in assignments[set_name]
                if digest not in literal
            ]
            assignments["train"].extend(
                member["sha256"] for member in literal_by_set[set_name]
            )
        plan = self.rewrite_plan(self.store, "mixed", assignments)

        result, target = self.export(
            self.store, "mixed", "mixed.zip", skip_unlabeled=True
        )

        kept = self.members_by_set(plan)
        for set_name in SET_NAMES:
            kept[set_name] = [
                member for member in kept[set_name] if member["label"] != ""
            ]
        self.assertEqual([len(members) for members in kept.values()], [5, 2, 2])

        # Worked example, set by set: validation and test keep exactly the
        # two labeled records and report three skips of their own; train's
        # three skips are train's own unlabeled members.
        self.assertEqual(result["exported"], 9)
        self.assertEqual(
            result["skipped"], {"train": 3, "validation": 3, "test": 3}
        )
        for set_name in ("validation", "test"):
            self.assertEqual(result["sets"][set_name]["samples"], 2)
            self.assertEqual(
                result["sets"][set_name]["distribution"], {"cat": 2}
            )
        self.assertEqual(result["sets"]["train"]["samples"], 5)
        self.assertEqual(
            result["sets"]["train"]["distribution"],
            {"cat": 2, "unlabeled": 3},
        )

        self.assert_package_consistent(
            result,
            target,
            plan,
            kept,
            {"train": 3, "validation": 3, "test": 3},
        )

        # The three genuine "unlabeled"-class samples are packaged as
        # ordinary labeled samples, under a class directory of their own.
        manifest, _, files = self.read_package(target)
        label_directories = {
            entry["label"]: entry["directory"] for entry in manifest["classes"]
        }
        self.assertEqual(
            sorted(label_directories), ["cat", "unlabeled"]
        )
        packaged_digests = {entry["sha256"] for entry in manifest["samples"]}
        self.assertEqual(set(literal) <= packaged_digests, True)
        for digest in literal:
            entry = next(
                entry for entry in manifest["samples"] if entry["sha256"] == digest
            )
            self.assertEqual(entry["label"], "unlabeled")
            self.assertEqual(
                entry["path"],
                f"train/{label_directories['unlabeled']}/{digest}.jpg",
            )
            self.assertEqual(files[entry["path"]], self.content_of(digest))

        # Skipped digests: all nine null/empty samples, and each skip is
        # charged to the set the saved plan named for that sample.
        skipped_by_set = {set_name: 0 for set_name in SET_NAMES}
        for digest in null_and_empty:
            skipped_by_set[self._set_of(plan, digest)] += 1
        # The plan still records each skipped sample in its original set.
        for digest in null_and_empty:
            self.assertIn(digest, [
                m["sha256"]
                for m in plan["sets"][self._set_of(plan, digest)]["members"]
            ])
        self.assertEqual(
            skipped_by_set, {"train": 3, "validation": 3, "test": 3}
        )

    @staticmethod
    def _set_of(plan: dict, digest: str) -> str:
        for set_name in SET_NAMES:
            if any(
                member["sha256"] == digest
                for member in plan["sets"][set_name]["members"]
            ):
                return set_name
        raise AssertionError(f"{digest} missing from plan")


class SkipUnlabeledSpellingsTest(SkipUnlabeledHarness):
    def test_null_and_empty_string_skipped_per_origin_set(self) -> None:
        # One null-labeled, one empty-string-labeled and one literal
        # "unlabeled" sample; place each in a different set, regardless of
        # where stratification naturally landed them.
        d_null = self.add_sample(self.store, None, b"null-bytes")
        d_empty = self.add_sample(self.store, "", b"empty-bytes")
        d_literal = self.add_sample(
            self.store, "unlabeled", b"literal-bytes"
        )
        plan = self.store.create_split(
            "spellings", seed=1, ratios=["1", "0", "0"]
        ).plan
        plan = self.rewrite_plan(
            self.store,
            "spellings",
            {"train": [d_null], "validation": [d_empty], "test": [d_literal]},
        )

        result, target = self.export(
            self.store, "spellings", "spellings.zip", skip_unlabeled=True
        )
        self.assertEqual(result["exported"], 1)
        self.assertEqual(
            result["skipped"], {"train": 1, "validation": 1, "test": 0}
        )
        kept = {
            "train": [],
            "validation": [],
            "test": [plan["sets"]["test"]["members"][0]],
        }
        manifest, _, files = self.assert_package_consistent(
            result,
            target,
            plan,
            kept,
            {"train": 1, "validation": 1, "test": 0},
        )
        self.assertEqual(len(manifest["samples"]), 1)
        entry = manifest["samples"][0]
        self.assertEqual(entry["sha256"], d_literal)
        self.assertEqual(entry["label"], "unlabeled")
        self.assertEqual(entry["set"], "test")
        self.assertTrue(entry["path"].startswith("test/class_"))
        self.assertEqual(files[entry["path"]], b"literal-bytes")


class SkipUnlabeledEmptySetsTest(SkipUnlabeledHarness):
    def test_set_emptied_by_skipping_keeps_directory_and_zero_stats(self) -> None:
        d_cat_1 = self.add_sample(self.store, "cat", b"cat-one")
        d_cat_2 = self.add_sample(self.store, "cat", b"cat-two")
        d_cat_3 = self.add_sample(self.store, "cat", b"cat-three")
        d_un_1 = self.add_sample(self.store, None, b"un-one")
        d_un_2 = self.add_sample(self.store, "", b"un-two")
        plan = self.store.create_split(
            "hollow", seed=3, ratios=["1", "0", "0"]
        ).plan
        plan = self.rewrite_plan(
            self.store,
            "hollow",
            {
                "train": [d_cat_1, d_cat_2],
                "validation": [d_un_1, d_un_2],
                "test": [d_cat_3],
            },
        )

        result, target = self.export(
            self.store, "hollow", "hollow.zip", skip_unlabeled=True
        )
        self.assertEqual(result["exported"], 3)
        self.assertEqual(
            result["skipped"], {"train": 0, "validation": 2, "test": 0}
        )
        self.assertEqual(result["sets"]["validation"]["samples"], 0)
        self.assertEqual(result["sets"]["validation"]["distribution"], {})

        kept = {
            "train": [plan["sets"]["train"]["members"][0],
                      plan["sets"]["train"]["members"][1]],
            "validation": [],
            "test": [plan["sets"]["test"]["members"][0]],
        }
        manifest, names, files = self.assert_package_consistent(
            result,
            target,
            plan,
            kept,
            {"train": 0, "validation": 2, "test": 0},
        )
        # No packaged file or manifest sample lands in the emptied set;
        # its set directory (and the shared class directory) still exist.
        self.assertIn("validation/", names)
        self.assertIn("validation/class_01/", names)
        self.assertFalse(
            [entry for entry in manifest["samples"] if entry["set"] == "validation"]
        )
        self.assertFalse([path for path in files if path.startswith("validation/")])
        self.assertEqual(
            sorted(entry["set"] for entry in manifest["samples"]),
            ["test", "train", "train"],
        )

    def test_entire_plan_skipped_exports_empty_package_with_per_set_skips(self) -> None:
        d1 = self.add_sample(self.store, None, b"only-null-1")
        d2 = self.add_sample(self.store, "", b"only-empty")
        d3 = self.add_sample(self.store, None, b"only-null-3")
        plan = self.store.create_split(
            "allun", seed=5, ratios=["1/3", "1/3", "1/3"]
        ).plan
        # Three members, one stratum: equal thirds give one per set.
        self.assertEqual(
            [plan["sets"][set_name]["samples"] for set_name in SET_NAMES],
            [1, 1, 1],
        )
        plan_sets = {
            set_name: [member["sha256"] for member in plan["sets"][set_name]["members"]]
            for set_name in SET_NAMES
        }
        self.assertEqual(sorted(digest for digests in plan_sets.values() for digest in digests),
                         sorted([d1, d2, d3]))

        result, target = self.export(
            self.store, "allun", "allun.zip", skip_unlabeled=True
        )
        self.assertEqual(result["exported"], 0)
        self.assertEqual(
            result["skipped"], {"train": 1, "validation": 1, "test": 1}
        )
        for set_name in SET_NAMES:
            self.assertEqual(result["sets"][set_name]["samples"], 0)
            self.assertEqual(result["sets"][set_name]["distribution"], {})

        empty = {set_name: [] for set_name in SET_NAMES}
        manifest, names, files = self.assert_package_consistent(
            result, target, plan, empty,
            {"train": 1, "validation": 1, "test": 1},
        )
        # All samples skipped: empty sample and class lists, no files.
        self.assertEqual(manifest["samples"], [])
        self.assertEqual(manifest["classes"], [])
        self.assertEqual(files, {})
        # The three set directories are still published.
        for set_name in SET_NAMES:
            self.assertIn(f"{set_name}/", names)
        for digest in (d1, d2, d3):
            self.assertTrue(all(digest not in name for name in names))

    def test_entire_plan_skipped_with_source_dir_still_succeeds(self) -> None:
        d1 = self.add_sample(self.store, None, b"src-null-1")
        d2 = self.add_sample(self.store, "", b"src-empty")
        d3 = self.add_sample(self.store, None, b"src-null-3")
        self.store.create_split(
            "allun-src", seed=5, ratios=["1/3", "1/3", "1/3"]
        )
        # The lookup directory is deliberately empty: with every sample
        # skipped there is nothing to match or read, so it is never scanned.
        source_dir = self.temp_root / "lookup"
        source_dir.mkdir()
        result, target = self.export(
            self.store,
            "allun-src",
            "allun-src.zip",
            skip_unlabeled=True,
            source_dir=source_dir,
        )
        self.assertEqual(result["exported"], 0)
        self.assertEqual(
            result["skipped"], {"train": 1, "validation": 1, "test": 1}
        )
        manifest, names, files = self.read_package(target)
        self.assertEqual(manifest["samples"], [])
        self.assertEqual(manifest["classes"], [])
        self.assertEqual(files, {})
        self.assertTrue(all(f"{set_name}/" in names for set_name in SET_NAMES))
        for digest in (d1, d2, d3):
            self.assertTrue(all(digest not in name for name in names))


class SkipUnlabeledRejectionTest(SkipUnlabeledHarness):
    def _plan_with_one_unlabeled(
        self, unlabeled_value: str | None
    ) -> tuple[DatasetStore, str, str, Path, str]:
        store = self.fresh_store(f"reject-{unlabeled_value!r}")
        self.add_sample(store, "cat", b"keep")
        d_unlabeled = self.add_sample(store, unlabeled_value, b"drop")
        plan_name = f"plan-{unlabeled_value!r}"
        plan = store.create_split(
            plan_name, seed=1, ratios=["1", "0", "0"]
        ).plan
        unlabeled_set = next(
            set_name
            for set_name in SET_NAMES
            for member in plan["sets"][set_name]["members"]
            if member["sha256"] == d_unlabeled
        )
        target = self.temp_root / f"reject-{unlabeled_value!r}.zip"
        return store, plan_name, d_unlabeled, target, unlabeled_set

    def test_without_flag_null_and_empty_string_each_reject_entire_export(self) -> None:
        for unlabeled_value in (None, ""):
            with self.subTest(unlabeled=unlabeled_value):
                store, plan_name, digest, target, set_name = (
                    self._plan_with_one_unlabeled(unlabeled_value)
                )
                with self.assertRaises(ExportError) as caught:
                    export_split(store, plan_name, target)
                message = str(caught.exception)
                # The rejection names the concrete sample and its set, and
                # points at the opt-in flag; no package is produced.
                self.assertIn(digest, message)
                self.assertIn(set_name, message)
                self.assertIn("--skip-unlabeled", message)
                self.assertFalse(target.exists())

    def test_labeled_only_plan_still_exports_without_flag(self) -> None:
        self.add_sample(self.store, "cat", b"cat")
        self.add_sample(self.store, "unlabeled", b"literal-class")
        self.store.create_split("clean", seed=1, ratios=["1", "0", "0"])
        result, target = self.export(self.store, "clean", "clean.zip")
        self.assertEqual(result["exported"], 2)
        self.assertEqual(
            result["skipped"], {"train": 0, "validation": 0, "test": 0}
        )
        self.assertTrue(target.exists())


class SavedPlanSnapshotTest(SkipUnlabeledHarness):
    def test_export_uses_saved_labels_despite_workspace_changes(self) -> None:
        # Build the mixed three-set plan, then flip labels in the
        # workspace after the plan was saved:
        #   * a plan-unlabeled sample becomes labeled;
        #   * a plan-labeled sample becomes unlabeled.
        plan, originals = self.make_mixed_plan(self.store, "snap")
        plan_before = self.store.get_split("snap")
        plan_file = self.store.splits_directory / "snap.json"
        plan_bytes_before = plan_file.read_bytes()

        digest_now_labeled = next(
            digest for digest, label in originals.items() if label is None
        )
        digest_now_unlabeled = next(
            digest
            for digest, label in originals.items()
            if label == "cat"
        )
        self.store.submit_batch(
            {
                "batch": "b-flip",
                "changes": [
                    {"sha256": digest_now_labeled, "old": None, "new": "cat"},
                    {"sha256": digest_now_unlabeled, "old": "cat", "new": None},
                ],
            }
        )
        # Sanity: the workspace really did change.
        self.assertEqual(
            self.store.lookup_label(digest_now_labeled)["label"], "cat"
        )
        self.assertIsNone(
            self.store.lookup_label(digest_now_unlabeled)["label"]
        )

        set_of_now_labeled = SkipUnlabeledMixedTest._set_of(
            plan_before, digest_now_labeled
        )
        set_of_now_unlabeled = SkipUnlabeledMixedTest._set_of(
            plan_before, digest_now_unlabeled
        )

        # The plan still snapshots the old labels: the sample that was
        # unlabeled at save time is skipped even though it is labeled now,
        # and the sample that was labeled at save time is kept with its
        # saved label even though the workspace dropped it.
        result, target = self.export(
            self.store, "snap", "snap.zip", skip_unlabeled=True
        )
        self.assertEqual(result["exported"], 9)
        self.assertEqual(
            result["skipped"], {"train": 3, "validation": 3, "test": 3}
        )
        manifest, _, files = self.read_package(target)
        packaged = {entry["sha256"]: entry for entry in manifest["samples"]}
        self.assertNotIn(digest_now_labeled, packaged)
        kept_entry = packaged[digest_now_unlabeled]
        self.assertEqual(kept_entry["label"], "cat")
        self.assertEqual(kept_entry["set"], set_of_now_unlabeled)
        self.assertEqual(files[kept_entry["path"]], self.content_of(digest_now_unlabeled))
        # The now-labeled sample is still counted as skipped against the
        # set named in the saved plan.
        self.assertEqual(result["skipped"][set_of_now_labeled], 3)

        # Without the flag the plan's own labels still force a rejection,
        # naming the concrete plan-unlabeled sample — the workspace relabel
        # cannot quietly unblock the export.
        rejected_target = self.temp_root / "snap-rejected.zip"
        with self.assertRaises(ExportError) as caught:
            export_split(self.store, "snap", rejected_target)
        unlabeled_in_plan = sorted(
            member["sha256"]
            for set_name in SET_NAMES
            for member in plan_before["sets"][set_name]["members"]
            if member["label"] == ""
        )
        self.assertIn(unlabeled_in_plan[0], str(caught.exception))
        self.assertFalse(rejected_target.exists())

        # The saved plan and its statistics were not rewritten by either
        # export attempt.
        self.assertEqual(plan_file.read_bytes(), plan_bytes_before)
        self.assertEqual(
            json.dumps(self.store.get_split("snap"), sort_keys=True),
            json.dumps(plan_before, sort_keys=True),
        )


class SkipUnlabeledCliTest(SkipUnlabeledHarness):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "vision_workbench", *arguments],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_cli_skip_unlabeled_output_matches_manifest(self) -> None:
        plan, _ = self.make_mixed_plan(self.store, "climix")
        target = self.temp_root / "climix.zip"

        rejected = self.run_cli(
            "export", str(self.store.root), "climix", str(target)
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("unlabeled", rejected.stderr)
        self.assertFalse(target.exists())

        skipped = self.run_cli(
            "export",
            str(self.store.root),
            "climix",
            str(target),
            "--skip-unlabeled",
        )
        self.assertEqual(skipped.returncode, 0, skipped.stderr)
        body = json.loads(skipped.stdout)
        self.assertEqual(body["exported"], 9)
        self.assertEqual(
            body["skipped"], {"train": 3, "validation": 3, "test": 3}
        )
        for set_name in SET_NAMES:
            self.assertEqual(
                body["sets"][set_name]["distribution"],
                {"cat": 2, "unlabeled": 1},
            )
            self.assertEqual(body["sets"][set_name]["samples"], 3)

        manifest, names, files = self.read_package(target)
        self.assertEqual(len(manifest["samples"]), 9)
        self.assertEqual(len(files), 9)
        for set_name in SET_NAMES:
            self.assertEqual(
                manifest["sets"][set_name],
                {"samples": 3, "skipped": 3},
            )
        # Result payload and manifest agree on every count.
        self.assertEqual(body["exported"], len(manifest["samples"]))
        self.assertEqual(body["plan"], manifest["plan"])


if __name__ == "__main__":
    unittest.main()
