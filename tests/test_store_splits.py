from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from vision_workbench.splits import SET_NAMES, SplitError
from vision_workbench.store import DatasetStore

REPO_ROOT = Path(__file__).resolve().parents[1]


class StoreHarness(unittest.TestCase):
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


class SplitPlanStoreTest(StoreHarness):
    def test_create_reports_counts_and_distributions(self) -> None:
        for _ in range(5):
            self.add_sample("cat")
        for _ in range(3):
            self.add_sample("dog")
        result = self.store.create_split("v1", 42, ["0.6", "0.2", "0.2"])
        self.assertTrue(result.created)
        plan = result.plan
        self.assertEqual(plan["name"], "v1")
        self.assertEqual(plan["seed"], 42)
        self.assertEqual(plan["ratios"], {"train": "3/5", "validation": "1/5", "test": "1/5"})
        self.assertEqual(plan["samples"]["total"], 8)
        totals = {name: plan["sets"][name]["samples"] for name in SET_NAMES}
        self.assertEqual(sum(totals.values()), 8)
        self.assertEqual(totals["train"], 5)
        # Stratification: each set has the same 5:3 shape within <1.
        for name in SET_NAMES:
            dist = plan["sets"][name]["distribution"]
            self.assertEqual(
                dist["cat"] + dist.get("dog", 0), plan["sets"][name]["samples"]
            )

    def test_reuse_vs_conflict(self) -> None:
        self.add_sample("cat")
        self.add_sample("dog")
        first = self.store.create_split("p", 1, [0.5, 0.5, 0])
        self.assertTrue(first.created)
        again = self.store.create_split("p", 1, [0.5, 0.5, 0])
        self.assertFalse(again.created)
        self.assertEqual(again.plan, first.plan)

        with self.assertRaises(SplitError):
            self.store.create_split("p", 2, [0.5, 0.5, 0])
        with self.assertRaises(SplitError):
            self.store.create_split("p", 1, [0.6, 0.4, 0])

        # Conflict must not overwrite the original.
        unchanged = self.store.get_split("p")
        self.assertEqual(unchanged["seed"], 1)
        self.assertEqual(unchanged["ratios"]["train"], "1/2")

        # Importing new samples then re-creating with the same name conflicts.
        self.add_sample("bird")
        with self.assertRaises(SplitError):
            self.store.create_split("p", 1, [0.5, 0.5, 0])

    def test_plan_is_fixed_after_new_imports_and_uses_old_labels(self) -> None:
        d1 = self.add_sample("cat")
        plan = self.store.create_split("fixed", 0, [1, 0, 0]).plan
        d2 = self.add_sample("dog")
        viewed = self.store.get_split("fixed")
        identities = {
            member["sha256"]
            for set_payload in viewed["sets"].values()
            for member in set_payload["members"]
        }
        self.assertEqual(identities, {d1})
        self.assertNotIn(d2, identities)
        self.assertEqual(viewed["samples"], {"total": 1, "distribution": {"cat": 1}})

        # Source file moved/deleted: the plan still loads unchanged.
        source = Path(viewed["sets"]["train"]["members"][0]["source"])
        source.unlink()
        viewed_after = self.store.get_split("fixed")
        self.assertEqual(viewed_after, viewed)

    def test_relocated_workspace_and_restart_keep_assignments(self) -> None:
        digests = []
        for label, count in [("a", 4), ("b", 3), ("c", 2)]:
            for _ in range(count):
                digests.append(self.add_sample(label))
        plan = self.store.create_split("portable", 9, [0.7, 0.2, 0.1]).plan

        copy_root = self.root.parent / "copied"
        shutil.copytree(self.root, copy_root)
        reopened = DatasetStore(copy_root).get_split("portable")
        self.assertEqual(reopened["sets"], plan["sets"])

        # Reconstructing from a manifest in a different order gives the
        # same stored JSON (samples are arranged by digest).
        self.assertEqual(json.dumps(reopened, sort_keys=True), json.dumps(plan, sort_keys=True))

    def test_members_sorted_by_digest(self) -> None:
        for _ in range(6):
            self.add_sample(None)
        plan = self.store.create_split("ordered", 3, [0.5, 0.25, 0.25]).plan
        for set_name in SET_NAMES:
            digests = [m["sha256"] for m in plan["sets"][set_name]["members"]]
            self.assertEqual(digests, sorted(digests))

    def test_empty_dataset_plan(self) -> None:
        result = self.store.create_split("empty", 0, [0.5, 0.25, 0.25])
        self.assertTrue(result.created)
        self.assertEqual(result.plan["samples"], {"total": 0, "distribution": {}})
        for name in SET_NAMES:
            self.assertEqual(
                result.plan["sets"][name],
                {"samples": 0, "distribution": {}, "members": []},
            )
        self.assertEqual(self.store.get_split("empty"), result.plan)

    def test_unlabeled_and_literal_class_separated(self) -> None:
        self.add_sample(None)
        self.add_sample("")
        self.add_sample("unlabeled")
        plan = self.store.create_split("u", 0, [1, 0, 0]).plan
        self.assertEqual(
            plan["samples"]["distribution"], {"": 2, "unlabeled": 1}
        )
        dist = plan["sets"]["train"]["distribution"]
        self.assertEqual(dist.get(""), 2)
        self.assertEqual(dist.get("unlabeled"), 1)

    def test_get_missing_plan_fails(self) -> None:
        with self.assertRaises(SplitError):
            self.store.get_split("nope")

    def test_invalid_names_rejected(self) -> None:
        for bad in ["", "  ", "a/b", "a\\b", ".", ".."]:
            with self.assertRaises(SplitError):
                self.store.create_split(bad, 0, [1, 0, 0])
        if self.store.splits_directory.exists():
            self.assertEqual(list(self.store.splits_directory.iterdir()), [])

    def test_bad_ratios_and_seed_leave_no_plan_or_manifest_change(self) -> None:
        self.add_sample("cat")
        manifest_before = self.store.manifest_path.read_text(encoding="utf-8")
        with self.assertRaises(SplitError):
            self.store.create_split("bad", 0, [0.5, 0.5, 0.5])
        with self.assertRaises(SplitError):
            self.store.create_split("bad", 1.5, [1, 0, 0])
        self.assertEqual(
            self.store.manifest_path.read_text(encoding="utf-8"), manifest_before
        )
        if self.store.splits_directory.exists():
            leftovers = [p for p in self.store.splits_directory.iterdir() if not p.name.startswith(".")]
            self.assertEqual(leftovers, [])

    def test_corrupt_plan_rejected(self) -> None:
        self.add_sample("cat")
        self.store.create_split("ok", 0, [1, 0, 0])
        plan_path = self.store.splits_directory / "ok.json"
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        # Drop a required field.
        del payload["sets"]["train"]["members"]
        plan_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(SplitError):
            self.store.get_split("ok")

    def corrupt_plan(self, mutate) -> tuple[Path, str]:
        """Create a valid plan, apply ``mutate`` to its payload, return path/text."""
        self.store.create_split("ok", 0, [1, 0, 0])
        plan_path = self.store.splits_directory / "ok.json"
        original = plan_path.read_text(encoding="utf-8")
        payload = json.loads(original)
        mutate(payload)
        plan_path.write_text(json.dumps(payload), encoding="utf-8")
        return plan_path, original

    def test_set_distribution_must_be_an_object(self) -> None:
        self.add_sample("cat")
        for bad in ([["cat", 1]], "cat", None):
            with self.subTest(bad=bad):
                plan_path, original = self.corrupt_plan(
                    lambda payload: payload["sets"]["train"].__setitem__(
                        "distribution", bad
                    )
                )
                with self.assertRaises(SplitError) as caught:
                    self.store.get_split("ok")
                self.assertIn("train", str(caught.exception))
                plan_path.write_text(original, encoding="utf-8")

    def test_missing_set_distribution_rejected(self) -> None:
        self.add_sample("cat")
        # Even an empty set may not silently treat a missing distribution
        # as an empty object.
        self.corrupt_plan(
            lambda payload: [
                payload["sets"][name].pop("distribution")
                for name in SET_NAMES
            ]
        )
        with self.assertRaises(SplitError) as caught:
            self.store.get_split("ok")
        self.assertIn("train", str(caught.exception))

    def test_set_sample_count_must_be_a_plain_integer(self) -> None:
        self.add_sample("cat")
        for bad in (True, 1.0, "1", None, -1):
            with self.subTest(bad=bad):
                plan_path, original = self.corrupt_plan(
                    lambda payload: payload["sets"]["train"].__setitem__(
                        "samples", bad
                    )
                )
                with self.assertRaises(SplitError) as caught:
                    self.store.get_split("ok")
                self.assertIn("train", str(caught.exception))
                plan_path.write_text(original, encoding="utf-8")

    def test_category_counts_must_be_plain_integers(self) -> None:
        self.add_sample("cat")
        for bad in (True, 1.0, "1", None, -1):
            with self.subTest(bad=bad):
                plan_path, original = self.corrupt_plan(
                    lambda payload: payload["sets"]["train"]["distribution"]
                    .__setitem__("cat", bad)
                )
                with self.assertRaises(SplitError) as caught:
                    self.store.get_split("ok")
                self.assertIn("train", str(caught.exception))
                plan_path.write_text(original, encoding="utf-8")

    def test_overall_statistics_must_be_plain_integers(self) -> None:
        self.add_sample("cat")
        for bad in (True, 1.0, "1", None, -1):
            with self.subTest(bad=bad):
                plan_path, original = self.corrupt_plan(
                    lambda payload: payload["samples"].__setitem__("total", bad)
                )
                with self.assertRaises(SplitError):
                    self.store.get_split("ok")
                plan_path.write_text(original, encoding="utf-8")

    def test_overall_distribution_must_be_an_object_of_counts(self) -> None:
        self.add_sample("cat")
        for bad in ([["cat", 1]], "cat", None, {"cat": True}, {"cat": 1.0}):
            with self.subTest(bad=bad):
                plan_path, original = self.corrupt_plan(
                    lambda payload: payload["samples"].__setitem__(
                        "distribution", bad
                    )
                )
                with self.assertRaises(SplitError):
                    self.store.get_split("ok")
                plan_path.write_text(original, encoding="utf-8")

    def test_rejected_read_does_not_rewrite_plan(self) -> None:
        self.add_sample("cat")
        plan_path, _ = self.corrupt_plan(
            lambda payload: payload["sets"]["train"].__setitem__("samples", True)
        )
        corrupted = plan_path.read_text(encoding="utf-8")
        with self.assertRaises(SplitError):
            self.store.get_split("ok")
        self.assertEqual(plan_path.read_text(encoding="utf-8"), corrupted)

    def test_statistics_must_match_the_saved_members(self) -> None:
        self.add_sample("cat")
        self.add_sample("dog")
        self.corrupt_plan(
            lambda payload: payload["sets"]["train"].__setitem__(
                "distribution", {"cat": 2}
            )
        )
        with self.assertRaises(SplitError) as caught:
            self.store.get_split("ok")
        self.assertIn("train", str(caught.exception))

    def test_partial_write_never_visible(self) -> None:
        self.add_sample("cat")
        plan_path = self.store.splits_directory / "ghost.json"
        # A leftover temp file must not be readable as a plan.
        self.store.splits_directory.mkdir(parents=True, exist_ok=True)
        (self.store.splits_directory / ".ghost.json.tmp").write_text("{partial", encoding="utf-8")
        with self.assertRaises(SplitError):
            self.store.get_split("ghost")

    def test_legacy_workspace_without_splits_directory(self) -> None:
        # An old workspace (only manifest.json) works without re-import.
        self.assertTrue(self.store.manifest_path.exists())
        self.assertFalse(self.store.splits_directory.exists())
        result = self.store.create_split("legacy", 0, [1, 0, 0])
        self.assertTrue(result.created)


class CliTest(StoreHarness):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "vision_workbench", *arguments],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_cli_create_show_and_errors(self) -> None:
        for _ in range(4):
            self.add_sample("cat")
        for _ in range(2):
            self.add_sample("dog")

        created = self.run_cli("split", "create", str(self.root), "run1",
                               "--seed", "7", "--train", "0.5",
                               "--validation", "0.25", "--test", "0.25")
        self.assertEqual(created.returncode, 0, created.stderr)
        body = json.loads(created.stdout)
        self.assertEqual(body["name"], "run1")
        self.assertEqual(body["status"], "created")
        self.assertEqual(sum(s["samples"] for s in body["sets"].values()), 6)

        shown = self.run_cli("split", "show", str(self.root), "run1")
        self.assertEqual(shown.returncode, 0, shown.stderr)
        full = json.loads(shown.stdout)
        self.assertEqual(full["seed"], 7)
        self.assertEqual(full["ratios"]["train"], "1/2")
        members = [
            (m["sha256"], m["label"], m["source"])
            for set_name in SET_NAMES
            for m in full["sets"][set_name]["members"]
        ]
        self.assertEqual(len(members), 6)
        # Samples are stably ordered by digest within each set.
        for set_name in SET_NAMES:
            digests = [m["sha256"] for m in full["sets"][set_name]["members"]]
            self.assertEqual(digests, sorted(digests))

        missing = self.run_cli("split", "show", str(self.root), "missing")
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("not found", missing.stderr)

        bad = self.run_cli("split", "create", str(self.root), "run1",
                           "--seed", "8", "--train", "0.5",
                           "--validation", "0.25", "--test", "0.25")
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("already exists", bad.stderr)

        invalid = self.run_cli("split", "create", str(self.root), "a/b",
                               "--seed", "1", "--train", "0.5",
                               "--validation", "0.5", "--test", "0.5")
        self.assertNotEqual(invalid.returncode, 0)

    def test_cli_show_and_export_reject_corrupt_statistics(self) -> None:
        self.add_sample("cat")
        created = self.run_cli("split", "create", str(self.root), "run1",
                               "--seed", "7", "--train", "1",
                               "--validation", "0", "--test", "0")
        self.assertEqual(created.returncode, 0, created.stderr)

        plan_path = self.store.splits_directory / "run1.json"
        original = plan_path.read_text(encoding="utf-8")
        payload = json.loads(original)
        payload["sets"]["validation"]["distribution"] = None
        payload["sets"]["test"]["samples"] = True
        plan_path.write_text(json.dumps(payload), encoding="utf-8")
        corrupted = plan_path.read_text(encoding="utf-8")

        shown = self.run_cli("split", "show", str(self.root), "run1")
        self.assertEqual(shown.returncode, 1)
        self.assertEqual(shown.stdout, "")
        self.assertIn("error:", shown.stderr)
        self.assertIn("validation", shown.stderr)
        self.assertNotIn("Traceback", shown.stderr)

        target = self.root.parent / "out.zip"
        exported = self.run_cli("export", str(self.root), "run1", str(target))
        self.assertEqual(exported.returncode, 1)
        self.assertIn("error:", exported.stderr)
        self.assertNotIn("Traceback", exported.stderr)
        self.assertFalse(target.exists())
        # The failed reads must not have rewritten the plan.
        self.assertEqual(plan_path.read_text(encoding="utf-8"), corrupted)
        plan_path.write_text(original, encoding="utf-8")
        restored = self.run_cli("split", "show", str(self.root), "run1")
        self.assertEqual(restored.returncode, 0, restored.stderr)

    def test_existing_commands_unchanged(self) -> None:
        image = self.root.parent / "pic.jpg"
        image.write_bytes(b"hello")
        added = self.run_cli("add", str(self.root), str(image), "--label", "cat")
        self.assertEqual(added.returncode, 0, added.stderr)
        summary = self.run_cli("summary", str(self.root))
        self.assertEqual(summary.returncode, 0, summary.stderr)
        body = json.loads(summary.stdout)
        self.assertEqual(body["items"], 1)
        self.assertEqual(body["labels"], {"cat": 1})


if __name__ == "__main__":
    unittest.main()
