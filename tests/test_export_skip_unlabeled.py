"""Regression coverage for ``--skip-unlabeled`` export.

The rule under test is narrow: when exporting a saved split plan, samples
whose *saved-plan* label is missing (``null``/``""`` at plan creation,
both normalized to ``""`` in the plan) are omitted from the package
instead of rejecting the whole export.  Everything else is locked down:

* the kept samples keep the set, label and file content the saved plan
  gave them -- nothing is moved between sets to re-approach the ratios;
* the per-set skip counts are charged to the set each skipped sample
  belonged to in the plan, never to another set;
* the terminal result payload, ``manifest.json`` and the files actually
  present in the ZIP agree on exported totals, per-set sample counts,
  skip counts and class distributions;
* a literal class named ``"unlabeled"`` is an ordinary class;
* a set emptied by skipping, and a plan whose every sample is skipped,
  still export with the three set directories retained;
* without the flag the export is still rejected as a whole, naming the
  sample, and the target is never created;
* the decision reads the saved plan's labels only -- later workspace
  label changes neither change who is skipped nor rewrite the plan.

Plans on disk store every label as a string (``null``/``""`` are
normalized to ``""`` when the plan is created), so deliberately crafted
plans spell unlabeled members as ``""``; the ``null`` spelling is
covered through real ``create_split`` snapshots.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from vision_workbench.exporter import ExportError, export_split
from vision_workbench.splits import SET_NAMES
from vision_workbench.store import DatasetStore

REPO_ROOT = Path(__file__).resolve().parents[1]

UNLABELED = ""


class SkipUnlabeledHarness(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "dataset"
        self.store = DatasetStore(self.root)
        self.store.initialize()
        self._serial = 0
        # Expected packaged bytes keyed by digest, for every real source
        # created through stage_file().
        self.contents: dict[str, bytes] = {}

    def tearDown(self) -> None:
        self.temp.cleanup()

    def stage_file(self, token: str, content: bytes | None = None) -> tuple[str, Path]:
        """Create one real source file; return ``(digest, source_path)``."""
        self._serial += 1
        path = self.root.parent / f"source-{self._serial}-{token}.jpg"
        data = content if content is not None else f"bytes:{token}:{self._serial}".encode()
        path.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        self.contents[digest] = data
        return digest, path

    def add_sample(self, label: str | None, content: bytes | None = None) -> str:
        """Import a sample into the workspace the normal way."""
        self._serial += 1
        path = self.root.parent / f"sample-{self._serial}.jpg"
        data = content if content is not None else f"content-{self._serial}".encode()
        path.write_bytes(data)
        result = self.store.add(path, label)
        self.assertTrue(result.added)
        self.contents[result.digest] = data
        return result.digest

    def member(
        self, digest: str, label: str, source: Path | str | None = None
    ) -> dict[str, str]:
        """Build one plan member; skipped members may point at no file."""
        if source is None:
            # A path that never exists: skipped samples must never be read,
            # so an export that only omits such members still succeeds.
            source = self.root.parent / "never-read" / f"{digest}.jpg"
        return {"sha256": digest, "label": label, "source": str(source)}

    def write_plan(
        self,
        name: str,
        sets: dict[str, list[dict[str, str]]],
        seed: int = 7,
        ratios: tuple[str, str, str] | None = None,
    ) -> dict[str, Any]:
        """Write a valid split plan with exact per-set member lists.

        ``ratios`` defaults to the exact proportions of the given member
        lists, so a crafted plan always satisfies the saved-ratio rule
        that plan reads enforce; pass explicit ratios to pin them.
        """
        if ratios is None:
            total = sum(len(sets.get(set_name, [])) for set_name in SET_NAMES)
            ratios = tuple(
                f"{len(sets.get(set_name, []))}/{total}" for set_name in SET_NAMES
            )  # type: ignore[assignment]
        set_payloads: dict[str, Any] = {}
        overall: Counter[str] = Counter()
        for set_name in SET_NAMES:
            members = sets.get(set_name, [])
            distribution = Counter(member["label"] for member in members)
            overall.update(distribution)
            set_payloads[set_name] = {
                "samples": len(members),
                "distribution": dict(sorted(distribution.items())),
                "members": members,
            }
        plan = {
            "schema_version": 1,
            "name": name,
            "seed": seed,
            "ratios": {
                set_name: ratios[index] for index, set_name in enumerate(SET_NAMES)
            },
            "samples": {
                "total": sum(overall.values()),
                "distribution": dict(sorted(overall.items())),
            },
            "sets": set_payloads,
        }
        plan_path = self.store.splits_directory / f"{name}.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        # Re-read through the store so a malformed crafted plan fails the
        # test rather than the export under test.
        return self.store.get_split(name)

    def export(
        self, name: str = "mixed", *, skip: bool
    ) -> tuple[dict[str, Any], Path]:
        target = self.root.parent / f"{name}.zip"
        result = export_split(self.store, name, target, skip_unlabeled=skip)
        return result, target

    def assert_package_consistency(
        self,
        result: dict[str, Any],
        target: Path,
        plan: dict[str, Any],
    ) -> None:
        """Result payload, manifest and ZIP contents must tell one story."""
        with zipfile.ZipFile(target) as archive:
            names = set(archive.namelist())
            manifest = json.loads(archive.read("manifest.json"))

            plan_members = {
                member["sha256"]: member
                for set_name in SET_NAMES
                for member in plan["sets"][set_name]["members"]
            }
            member_set = {
                member["sha256"]: set_name
                for set_name in SET_NAMES
                for member in plan["sets"][set_name]["members"]
            }
            kept = [
                member
                for member in plan_members.values()
                if member["label"] != UNLABELED
            ]
            skipped = [
                member
                for member in plan_members.values()
                if member["label"] == UNLABELED
            ]

            # ---- aggregate totals across the three surfaces ----
            self.assertEqual(result["exported"], len(kept))
            self.assertEqual(result["exported"], len(manifest["samples"]))
            file_names = {
                name
                for name in names
                if not name.endswith("/") and name != "manifest.json"
            }
            self.assertEqual(result["exported"], len(file_names))

            expected_skipped = {
                set_name: sum(
                    1
                    for member in plan["sets"][set_name]["members"]
                    if member["label"] == UNLABELED
                )
                for set_name in SET_NAMES
            }
            self.assertEqual(result["skipped"], expected_skipped)

            # ---- per-set agreement: result vs manifest vs ZIP ----
            manifest_entries = {entry["sha256"]: entry for entry in manifest["samples"]}
            self.assertEqual(set(manifest_entries), {m["sha256"] for m in kept})

            for set_name in SET_NAMES:
                plan_members_in_set = plan["sets"][set_name]["members"]
                kept_in_set = [
                    m for m in plan_members_in_set if m["label"] != UNLABELED
                ]
                # Nothing moved between sets to make up the ratios: each
                # kept sample is still in its saved-plan set, and skips
                # are charged to the set that owned the sample.
                self.assertEqual(
                    result["sets"][set_name]["samples"], len(kept_in_set)
                )
                self.assertEqual(
                    manifest["sets"][set_name]["samples"], len(kept_in_set)
                )
                self.assertEqual(
                    manifest["sets"][set_name]["skipped"],
                    expected_skipped[set_name],
                )
                self.assertEqual(
                    result["sets"][set_name]["samples"]
                    + result["skipped"][set_name],
                    len(plan_members_in_set),
                )

                expected_distribution = dict(
                    Counter(member["label"] for member in kept_in_set)
                )
                self.assertEqual(
                    result["sets"][set_name]["distribution"],
                    dict(sorted(expected_distribution.items())),
                )
                entries_in_set = [
                    entry
                    for entry in manifest["samples"]
                    if entry["set"] == set_name
                ]
                self.assertEqual(
                    dict(Counter(entry["label"] for entry in entries_in_set)),
                    expected_distribution,
                )
                files_in_set = {name for name in file_names if name.startswith(f"{set_name}/")}
                self.assertEqual(len(files_in_set), len(kept_in_set))

            # ---- the three set directories are always retained ----
            for set_name in SET_NAMES:
                self.assertIn(f"{set_name}/", names)

            # ---- every manifest entry matches plan, path and bytes ----
            label_to_directory = {
                cls["label"]: cls["directory"] for cls in manifest["classes"]
            }
            self.assertEqual(
                {cls["label"] for cls in manifest["classes"]},
                {member["label"] for member in kept},
            )
            for entry in manifest["samples"]:
                planned = plan_members[entry["sha256"]]
                planned_set = member_set[entry["sha256"]]
                # Kept identity: same set, same label, same digest.
                self.assertEqual(entry["set"], planned_set)
                self.assertEqual(entry["label"], planned["label"])
                suffix = Path(planned["source"]).suffix
                self.assertEqual(
                    entry["path"],
                    f"{planned_set}/"
                    f"{label_to_directory[planned['label']]}/"
                    f"{planned['sha256']}{suffix}",
                )
                self.assertIn(entry["path"], file_names)
                data = archive.read(entry["path"])
                self.assertEqual(data, self.contents[entry["sha256"]])
                self.assertEqual(
                    hashlib.sha256(data).hexdigest(), entry["sha256"]
                )

            # ---- omitted samples leave no trace in the package ----
            for member in skipped:
                self.assertNotIn(member["sha256"], manifest_entries)
                for name in names:
                    self.assertNotIn(member["sha256"], name)


class SkipUnlabeledExportTest(SkipUnlabeledHarness):
    def test_skips_charged_to_each_original_set_with_matching_package(self) -> None:
        # Train: two labeled + three unlabeled; validation and test carry
        # labeled and unlabeled members as well, so a skip count leaking
        # between sets would be visible.
        cat_train, cat_src = self.stage_file("cat-train")
        dog_train, dog_src = self.stage_file("dog-train")
        skip_sources: list[dict[str, str]] = []
        for token in ("t-skip-1", "t-skip-2", "t-skip-3"):
            digest, _ = self.stage_file(token)
            skip_sources.append(
                self.member(digest, UNLABELED, self.root.parent / "gone" / f"{token}.jpg")
            )

        bird_val, bird_src = self.stage_file("bird-val")
        val_skip_1, _ = self.stage_file("v-skip-1")
        val_skip_2, _ = self.stage_file("v-skip-2")

        cat_test, cat_test_src = self.stage_file("cat-test")
        fish_test, fish_src = self.stage_file("fish-test")
        test_skip, _ = self.stage_file("te-skip-1")

        plan = self.write_plan(
            "mixed",
            {
                "train": [
                    self.member(cat_train, "cat", cat_src),
                    self.member(dog_train, "dog", dog_src),
                    *skip_sources,
                ],
                "validation": [
                    self.member(bird_val, "bird", bird_src),
                    self.member(val_skip_1, UNLABELED),
                    self.member(val_skip_2, UNLABELED),
                ],
                "test": [
                    self.member(cat_test, "cat", cat_test_src),
                    self.member(fish_test, "fish", fish_src),
                    self.member(test_skip, UNLABELED),
                ],
            },
        )

        result, target = self.export("mixed", skip=True)

        self.assertEqual(result["exported"], 5)
        self.assertEqual(
            result["skipped"], {"train": 3, "validation": 2, "test": 1}
        )
        self.assertEqual(result["sets"]["train"]["samples"], 2)
        self.assertEqual(
            result["sets"]["train"]["distribution"], {"cat": 1, "dog": 1}
        )
        self.assertEqual(result["sets"]["validation"]["samples"], 1)
        self.assertEqual(
            result["sets"]["validation"]["distribution"], {"bird": 1}
        )
        self.assertEqual(result["sets"]["test"]["samples"], 2)
        self.assertEqual(
            result["sets"]["test"]["distribution"], {"cat": 1, "fish": 1}
        )

        with zipfile.ZipFile(target) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(
                manifest["sets"],
                {
                    "train": {"samples": 2, "skipped": 3},
                    "validation": {"samples": 1, "skipped": 2},
                    "test": {"samples": 2, "skipped": 1},
                },
            )
            self.assertEqual(
                sorted(entry["sha256"] for entry in manifest["samples"]),
                sorted([cat_train, dog_train, bird_val, cat_test, fish_test]),
            )

        self.assert_package_consistency(result, target, plan)

    def test_null_and_empty_string_labels_both_skipped_from_real_plan(self) -> None:
        # Both unlabeled spellings enter through the public import path
        # and are normalized into the saved plan as "".
        d_cat = self.add_sample("cat", b"cat-a")
        d_dog = self.add_sample("dog", b"dog-a")
        d_none = self.add_sample(None, b"none-label")
        d_empty = self.add_sample("", b"empty-label")

        plan = self.store.create_split(
            "baseline", 42, ["1", "0", "0"]
        ).plan
        # Saved labels for both are the empty string (never null).
        saved = {
            member["sha256"]: member["label"]
            for member in plan["sets"]["train"]["members"]
        }
        self.assertEqual(saved[d_none], "")
        self.assertEqual(saved[d_empty], "")

        result, target = self.export("baseline", skip=True)
        self.assertEqual(result["exported"], 2)
        self.assertEqual(
            result["skipped"], {"train": 2, "validation": 0, "test": 0}
        )
        self.assertEqual(result["sets"]["train"]["samples"], 2)

        with zipfile.ZipFile(target) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            kept = {entry["sha256"] for entry in manifest["samples"]}
            self.assertEqual(kept, {d_cat, d_dog})
            self.assertNotIn(d_none, kept)
            self.assertNotIn(d_empty, kept)
            for name in archive.namelist():
                self.assertNotIn(d_none, name)
                self.assertNotIn(d_empty, name)

        self.assert_package_consistency(result, target, plan)

    def test_skip_counts_follow_real_stratified_assignment(self) -> None:
        # A real stratified plan spreads unlabeled members across sets;
        # the reported skip counts must match where the plan put them.
        self.add_sample("cat", b"cat-1")
        self.add_sample("cat", b"cat-2")
        self.add_sample("dog", b"dog-1")
        self.add_sample("dog", b"dog-2")
        self.add_sample(None, b"none-1")
        self.add_sample(None, b"none-2")
        self.add_sample("", b"empty-1")
        self.add_sample("", b"empty-2")

        plan = self.store.create_split(
            "baseline", 11, ["1/2", "1/4", "1/4"]
        ).plan
        expected_skipped = {
            set_name: sum(
                1
                for member in plan["sets"][set_name]["members"]
                if member["label"] == ""
            )
            for set_name in SET_NAMES
        }
        self.assertEqual(sum(expected_skipped.values()), 4)

        result, target = self.export("baseline", skip=True)
        self.assertEqual(result["exported"], 4)
        self.assertEqual(result["skipped"], expected_skipped)
        for set_name in SET_NAMES:
            self.assertEqual(
                result["sets"][set_name]["samples"]
                + result["skipped"][set_name],
                plan["sets"][set_name]["samples"],
            )
        self.assert_package_consistency(result, target, plan)

    def test_literal_unlabeled_class_is_kept_alongside_skipped_members(self) -> None:
        cat_d, cat_src = self.stage_file("cat")
        lit_1, lit_1_src = self.stage_file("literal-1")
        lit_2, lit_2_src = self.stage_file("literal-2")
        val_lit, val_lit_src = self.stage_file("val-literal")
        skip_1, _ = self.stage_file("skip-1")
        skip_2, _ = self.stage_file("skip-2")
        val_skip, _ = self.stage_file("val-skip")

        plan = self.write_plan(
            "mixed",
            {
                "train": [
                    self.member(cat_d, "cat", cat_src),
                    self.member(lit_1, "unlabeled", lit_1_src),
                    self.member(lit_2, "unlabeled", lit_2_src),
                    self.member(skip_1, UNLABELED),
                    self.member(skip_2, UNLABELED),
                ],
                "validation": [
                    self.member(val_lit, "unlabeled", val_lit_src),
                    self.member(val_skip, UNLABELED),
                ],
                "test": [],
            },
        )

        result, target = self.export("mixed", skip=True)
        self.assertEqual(result["exported"], 4)
        self.assertEqual(
            result["skipped"], {"train": 2, "validation": 1, "test": 0}
        )
        self.assertEqual(
            result["sets"]["train"]["distribution"], {"cat": 1, "unlabeled": 2}
        )
        self.assertEqual(
            result["sets"]["validation"]["distribution"], {"unlabeled": 1}
        )
        self.assertEqual(result["sets"]["test"]["distribution"], {})

        with zipfile.ZipFile(target) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            labels = {cls["label"] for cls in manifest["classes"]}
            self.assertEqual(labels, {"cat", "unlabeled"})
            kept = {entry["sha256"]: entry["label"] for entry in manifest["samples"]}
            self.assertEqual(
                kept,
                {
                    cat_d: "cat",
                    lit_1: "unlabeled",
                    lit_2: "unlabeled",
                    val_lit: "unlabeled",
                },
            )

        self.assert_package_consistency(result, target, plan)

    def test_set_emptied_by_skipping_reports_zero_and_empty_distribution(self) -> None:
        # Test set originally holds only unlabeled members; after skipping
        # it has zero samples but keeps its skip count and its directory.
        cat_d, cat_src = self.stage_file("only-cat")
        test_skips = []
        for token in ("te-empty-1", "te-empty-2"):
            digest, _ = self.stage_file(token)
            test_skips.append(self.member(digest, UNLABELED))

        plan = self.write_plan(
            "mixed",
            {
                "train": [self.member(cat_d, "cat", cat_src)],
                "validation": [],
                "test": test_skips,
            },
        )

        result, target = self.export("mixed", skip=True)
        self.assertEqual(result["exported"], 1)
        self.assertEqual(
            result["skipped"], {"train": 0, "validation": 0, "test": 2}
        )
        self.assertEqual(result["sets"]["test"]["samples"], 0)
        self.assertEqual(result["sets"]["test"]["distribution"], {})

        with zipfile.ZipFile(target) as archive:
            names = set(archive.namelist())
            self.assertIn("test/", names)
            # The emptied set still mirrors shared class directories (the
            # normal cross-set layout), but it must contain no sample
            # files of its own.
            self.assertEqual(
                [
                    name
                    for name in names
                    if name.startswith("test/") and not name.endswith("/")
                ],
                [],
            )
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(manifest["sets"]["test"], {"samples": 0, "skipped": 2})
            self.assertEqual(
                [entry for entry in manifest["samples"] if entry["set"] == "test"],
                [],
            )

        self.assert_package_consistency(result, target, plan)

    def test_all_samples_skipped_exports_empty_package_with_per_set_skips(self) -> None:
        # Every member is unlabeled and points at paths that do not exist:
        # a successful export proves skipped samples are never opened.
        members: dict[str, list[dict[str, str]]] = {}
        per_set = {"train": 2, "validation": 1, "test": 3}
        for set_name, count in per_set.items():
            entries = []
            for index in range(count):
                digest, _ = self.stage_file(f"{set_name}-{index}")
                entries.append(self.member(digest, UNLABELED))
            members[set_name] = entries

        plan = self.write_plan("all-unlabeled", members)

        result, target = self.export("all-unlabeled", skip=True)
        self.assertEqual(result["exported"], 0)
        self.assertEqual(
            result["skipped"], {"train": 2, "validation": 1, "test": 3}
        )
        for set_name in SET_NAMES:
            self.assertEqual(result["sets"][set_name]["samples"], 0)
            self.assertEqual(result["sets"][set_name]["distribution"], {})

        self.assertTrue(target.exists())
        with zipfile.ZipFile(target) as archive:
            names = set(archive.namelist())
            # Exactly the three set directories and the manifest; no class
            # directories and no sample files.
            self.assertEqual(names, {"train/", "validation/", "test/", "manifest.json"})
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(manifest["samples"], [])
            self.assertEqual(manifest["classes"], [])
            self.assertEqual(
                manifest["sets"],
                {
                    "train": {"samples": 0, "skipped": 2},
                    "validation": {"samples": 0, "skipped": 1},
                    "test": {"samples": 0, "skipped": 3},
                },
            )

        self.assert_package_consistency(result, target, plan)

    def test_without_flag_rejects_entire_export_naming_sample_and_set(self) -> None:
        skip_a, _ = self.stage_file("reject-a")
        skip_b, _ = self.stage_file("reject-b")
        cat_d, cat_src = self.stage_file("reject-cat")
        self.write_plan(
            "mixed",
            {
                "train": [self.member(skip_a, UNLABELED)],
                "validation": [
                    self.member(cat_d, "cat", cat_src),
                    self.member(skip_b, UNLABELED),
                ],
                "test": [],
            },
        )
        target = self.root.parent / "mixed.zip"
        with self.assertRaises(ExportError) as caught:
            export_split(self.store, "mixed", target)

        message = str(caught.exception)
        # The digest named is the sorted-first unlabeled member, and its
        # set is reported; the message tells the user how to proceed.
        first_digest = min(skip_a, skip_b)
        self.assertIn(first_digest, message)
        self.assertIn("--skip-unlabeled", message)
        expected_set = "train" if first_digest == skip_a else "validation"
        self.assertIn(expected_set, message)
        self.assertFalse(target.exists())

    def test_without_flag_empty_string_in_real_plan_still_rejects(self) -> None:
        d_none = self.add_sample(None, b"none")
        d_empty = self.add_sample("", b"empty")
        self.add_sample("cat", b"cat")
        self.store.create_split("baseline", 42, ["1", "0", "0"])

        target = self.root.parent / "baseline.zip"
        with self.assertRaises(ExportError) as caught:
            export_split(self.store, "baseline", target)
        message = str(caught.exception)
        self.assertIn(min(d_none, d_empty), message)
        self.assertIn("--skip-unlabeled", message)
        self.assertFalse(target.exists())

    def test_skip_decision_uses_saved_plan_labels_despite_workspace_changes(self) -> None:
        d_now_labeled = self.add_sample(None, b"was-unlabeled")
        d_now_unlabeled = self.add_sample("cat", b"was-cat")
        d_cat = self.add_sample("cat", b"steady-cat")
        plan = self.store.create_split("baseline", 42, ["1", "0", "0"]).plan

        def plan_label(digest: str) -> str:
            for member in plan["sets"]["train"]["members"]:
                if member["sha256"] == digest:
                    return member["label"]
            raise KeyError(digest)

        self.assertEqual(plan_label(d_now_labeled), "")
        self.assertEqual(plan_label(d_now_unlabeled), "cat")

        # After the plan is saved, flip both labels in the workspace.
        self.store.submit_batch(
            {
                "batch": "b1",
                "changes": [
                    {"sha256": d_now_labeled, "old": None, "new": "cat"},
                    {"sha256": d_now_unlabeled, "old": "cat", "new": None},
                ],
            }
        )

        # Without the flag the saved plan still forces rejection, naming
        # the sample that was unlabeled *when the plan was saved*.
        target = self.root.parent / "baseline.zip"
        with self.assertRaises(ExportError) as caught:
            export_split(self.store, "baseline", target)
        self.assertIn(d_now_labeled, str(caught.exception))
        self.assertFalse(target.exists())

        # With the flag the saved-plan decision stands: the sample now
        # labeled in the workspace is still skipped; the sample that lost
        # its label after saving is still kept with its saved label.
        result, target = self.export("baseline", skip=True)
        self.assertEqual(result["exported"], 2)
        self.assertEqual(
            result["skipped"], {"train": 1, "validation": 0, "test": 0}
        )
        with zipfile.ZipFile(target) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            kept = {entry["sha256"]: entry["label"] for entry in manifest["samples"]}
            self.assertEqual(
                kept, {d_now_unlabeled: "cat", d_cat: "cat"}
            )
            names = archive.namelist()
            for name in names:
                self.assertNotIn(d_now_labeled, name)
            self.assertEqual(
                archive.read(
                    next(
                        entry["path"]
                        for entry in manifest["samples"]
                        if entry["sha256"] == d_now_unlabeled
                    )
                ),
                b"was-cat",
            )

        # The saved plan itself was not rewritten by either export.
        reloaded = self.store.get_split("baseline")
        self.assertEqual(reloaded, plan)

    def test_skip_export_does_not_modify_workspace_or_plan(self) -> None:
        skip_d, _ = self.stage_file("workspace-skip")
        cat_d, cat_src = self.stage_file("workspace-cat")
        self.write_plan(
            "mixed",
            {
                "train": [
                    self.member(cat_d, "cat", cat_src),
                    self.member(skip_d, UNLABELED),
                ],
                "validation": [],
                "test": [],
            },
        )
        plan_path = self.store.splits_directory / "mixed.json"
        manifest_before = self.store.manifest_path.read_bytes()
        plan_before = plan_path.read_bytes()
        plans_before = sorted(p.name for p in self.store.splits_directory.iterdir())

        self.export("mixed", skip=True)

        self.assertEqual(self.store.manifest_path.read_bytes(), manifest_before)
        self.assertEqual(plan_path.read_bytes(), plan_before)
        self.assertEqual(
            sorted(p.name for p in self.store.splits_directory.iterdir()),
            plans_before,
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

    def test_cli_skip_reports_per_set_counts_matching_package(self) -> None:
        cat_d, cat_src = self.stage_file("cli-cat")
        dog_d, dog_src = self.stage_file("cli-dog")
        skip_d, _ = self.stage_file("cli-skip")
        val_skip, _ = self.stage_file("cli-val-skip")
        plan = self.write_plan(
            "mixed",
            {
                "train": [
                    self.member(cat_d, "cat", cat_src),
                    self.member(dog_d, "dog", dog_src),
                    self.member(skip_d, UNLABELED),
                ],
                "validation": [self.member(val_skip, UNLABELED)],
                "test": [],
            },
        )

        target = self.root.parent / "cli.zip"

        rejected = self.run_cli(
            "export", str(self.root), "mixed", str(target)
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertFalse(target.exists())

        skipped = self.run_cli(
            "export", str(self.root), "mixed", str(target), "--skip-unlabeled"
        )
        self.assertEqual(skipped.returncode, 0, skipped.stderr)
        body = json.loads(skipped.stdout)
        self.assertEqual(body["exported"], 2)
        self.assertEqual(
            body["skipped"], {"train": 1, "validation": 1, "test": 0}
        )
        self.assertEqual(body["sets"]["train"]["samples"], 2)
        self.assertEqual(
            body["sets"]["train"]["distribution"], {"cat": 1, "dog": 1}
        )
        self.assertEqual(body["sets"]["validation"]["samples"], 0)
        self.assertEqual(body["sets"]["validation"]["distribution"], {})

        with zipfile.ZipFile(target) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(body["exported"], len(manifest["samples"]))
            for set_name in SET_NAMES:
                self.assertEqual(
                    body["sets"][set_name]["samples"],
                    manifest["sets"][set_name]["samples"],
                )
                self.assertEqual(
                    body["skipped"][set_name],
                    manifest["sets"][set_name]["skipped"],
                )
            kept = {entry["sha256"] for entry in manifest["samples"]}
            self.assertEqual(kept, {cat_d, dog_d})
            names = archive.namelist()
            for omitted in (skip_d, val_skip):
                self.assertFalse(any(omitted in name for name in names))


if __name__ == "__main__":
    unittest.main()
