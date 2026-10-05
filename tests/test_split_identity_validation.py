"""Identity-format validation of saved split plans.

Every member ``sha256`` in a saved plan must be exactly 64 lowercase
hexadecimal characters — the same spelling the registration manifest
uses.  ``split show``, ``export`` (even with ``--skip-unlabeled``) and
``split create`` under an existing name must all reject any other shape
(truncated, padded, uppercase, non-hex, whitespace-bearing, empty,
missing or non-string), naming the plan, the set and the member's
1-based position inside that set.  The check reads the saved plan only:
a format-correct plan whose images moved or whose samples left the
manifest stays valid, and nothing is rewritten when a bad plan is
refused.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from vision_workbench.exporter import ExportError, export_split
from vision_workbench.splits import SET_NAMES, SplitError, invalid_digest_reason
from vision_workbench.store import DatasetStore

REPO_ROOT = Path(__file__).resolve().parents[1]


class IdentityHarness(unittest.TestCase):
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

    def digest(self, token: str) -> str:
        return hashlib.sha256(f"plan-{token}".encode()).hexdigest()

    def raw_member(self, digest: Any, label: str = "cat") -> dict[str, Any]:
        return {
            "sha256": digest,
            "label": label,
            "source": str(self.root.parent / f"{digest!s}.jpg"),
        }

    def write_raw_plan(
        self,
        name: str,
        members_by_set: dict[str, list[dict[str, Any]]],
        ratios: tuple[str, str, str] = ("1", "0", "0"),
        seed: int = 0,
    ) -> Path:
        """Write a plan whose statistics exactly match its raw members."""
        from collections import Counter

        set_payloads: dict[str, Any] = {}
        overall: Counter[str] = Counter()
        for set_name in SET_NAMES:
            members = members_by_set.get(set_name, [])
            distribution = Counter(
                member["label"] for member in members if isinstance(member, dict)
            )
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
                "total": sum(
                    len(members_by_set.get(set_name, [])) for set_name in SET_NAMES
                ),
                "distribution": dict(sorted(overall.items())),
            },
            "sets": set_payloads,
        }
        plan_path = self.store.splits_directory / f"{name}.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return plan_path

    def good_train_plan(self, name: str, bad_at: int, bad_value: Any) -> Path:
        """All-train plan (ratios 1/0/0) with one bad identity at ``bad_at``."""
        members: list[dict[str, Any]] = []
        for index in range(1, 4):
            if index == bad_at:
                member = self.raw_member(bad_value)
                # Non-string/None values survive JSON; a missing field is
                # produced by deleting the key after writing.
                if bad_value == "__missing__":
                    del member["sha256"]
            else:
                member = self.raw_member(self.digest(f"good-{name}-{index}"))
            members.append(member)
        return self.write_raw_plan(name, {"train": members})

    def assert_identity_error(
        self, name: str, set_name: str, position: int, value: Any
    ) -> SplitError:
        with self.assertRaises(SplitError) as caught:
            self.store.get_split(name)
        message = str(caught.exception)
        self.assertIn("corrupted", message)
        self.assertIn(name, message)
        self.assertIn(set_name, message)
        self.assertIn(f"member {position}", message)
        self.assertIn("sha256", message)
        # The bad spelling itself is shown (the message renders it with
        # repr(), so compare against the same repr for tabs/spaces).
        if isinstance(value, str) and value and value != "__missing__":
            self.assertIn(repr(value)[1:-1], message)
        return caught.exception


class InvalidDigestReasonTest(unittest.TestCase):
    def test_valid_lowercase_digest_accepted(self) -> None:
        digest = hashlib.sha256(b"x").hexdigest()
        self.assertIsNone(invalid_digest_reason(digest))
        self.assertEqual(len(digest), 64)

    def test_each_bad_shape_has_a_reason(self) -> None:
        good = hashlib.sha256(b"x").hexdigest()
        bad_values = [
            good[:-1],                  # truncated
            good + "a",                 # too long
            good.upper(),               # uppercase
            "",                         # empty
            None,                       # missing
            123,                        # non-string number
            True,                       # boolean
            ["not", "a", "string"],     # other type
            " " + good,                 # leading space
            good + " ",                 # trailing space
            good[:-1] + "g",            # non-hex character at the end
            good[:8] + "\t" + good[9:], # embedded tab
        ]
        for value in bad_values:
            with self.subTest(value=repr(value)):
                self.assertIsNotNone(invalid_digest_reason(value))


class SavedPlanIdentityTest(IdentityHarness):
    BAD_STRINGS = {
        "truncated": lambda d: d[:-1],
        "too-long": lambda d: d + "a",
        "non-hex": lambda d: d[:-1] + "g",
        "uppercase": lambda d: d.upper(),
        "leading-space": lambda d: " " + d,
        "trailing-space": lambda d: d + " ",
        "embedded-tab": lambda d: d[:8] + "\t" + d[9:],
    }

    def test_string_corruptions_rejected_in_train_with_position(self) -> None:
        for case, mutate in self.BAD_STRINGS.items():
            with self.subTest(case=case):
                name = f"bad-{case}"
                bad_value = mutate(self.digest(f"bad-{case}"))
                plan_path = self.good_train_plan(name, bad_at=3, bad_value=bad_value)
                original = plan_path.read_bytes()
                self.assert_identity_error(name, "train", 3, bad_value)
                # The rejected read never rewrites the plan.
                self.assertEqual(plan_path.read_bytes(), original)

    def test_empty_missing_and_non_string_identities_rejected(self) -> None:
        for case, value in [
            ("empty", ""),
            ("missing", "__missing__"),
            ("number", 12345),
            ("null", None),
        ]:
            with self.subTest(case=case):
                name = f"bad-{case}"
                plan_path = self.good_train_plan(name, bad_at=2, bad_value=value)
                self.assert_identity_error(name, "train", 2, value)

    def test_error_reported_for_every_set_and_later_positions(self) -> None:
        # Two members per set under equal thirds: 2 vs 6*1/3 = 2 exactly,
        # so the plan is proportionally sound apart from the one bad
        # identity.  Earlier valid members must not mask a later one.
        for set_name, position in [
            ("train", 2),
            ("validation", 1),
            ("validation", 2),
            ("test", 1),
        ]:
            with self.subTest(set_name=set_name, position=position):
                name = f"pos-{set_name}-{position}"
                members_by_set: dict[str, list[dict[str, Any]]] = {}
                for each_set in SET_NAMES:
                    members_by_set[each_set] = [
                        self.raw_member(self.digest(f"{name}-{each_set}-{i}"))
                        for i in range(1, 3)
                    ]
                bad_value = self.digest(f"{name}-bad")[:-1]
                members_by_set[set_name][position - 1] = self.raw_member(bad_value)
                self.write_raw_plan(
                    name, members_by_set, ratios=("1/3", "1/3", "1/3")
                )
                self.assert_identity_error(name, set_name, position, bad_value)

    def test_sets_are_checked_in_train_validation_test_order(self) -> None:
        # Bad members in validation and test, train clean: validation is
        # reported first.
        members_by_set = {
            "train": [self.raw_member(self.digest("ord-t"))],
            "validation": [self.raw_member(self.digest("ord-v")[:-1])],
            "test": [self.raw_member(self.digest("ord-x").upper())],
        }
        # Keep statistics consistent with the (bad) members; identity is
        # checked before proportion rules.
        self.write_raw_plan(
            "order", members_by_set, ratios=("1/3", "1/3", "1/3")
        )
        error = self.assert_identity_error(
            "order", "validation", 1, members_by_set["validation"][0]["sha256"]
        )
        self.assertNotIn("test", str(error))

    def test_unlabeled_member_identity_must_be_valid_too(self) -> None:
        # A valid labeled member followed by an unlabeled member with a
        # truncated identity: the unlabeled status never exempts it.
        members = [
            self.raw_member(self.digest("lab-ok"), label="cat"),
            self.raw_member(self.digest("unlab-bad")[:-1], label=""),
        ]
        self.write_raw_plan("unlab", {"train": members})
        self.assert_identity_error(
            "unlab", "train", 2, members[1]["sha256"]
        )

    def test_no_normalization_accepts_uppercase_registered_sample(self) -> None:
        # The real sample exists in the manifest with a lowercase digest;
        # the plan spelling it uppercase is still corruption, even though
        # the image is present and the digest could be "fixed" by lowercasing.
        digest = self.add_sample("cat", b"uppercase-case")
        members = [self.raw_member(digest.upper())]
        plan_path = self.write_raw_plan("upper", {"train": members})
        corrupted_bytes = plan_path.read_bytes()
        with self.assertRaises(SplitError) as caught:
            self.store.get_split("upper")
        self.assertIn("lowercase hexadecimal", str(caught.exception))
        self.assertIn("member 1", str(caught.exception))
        # The rejected read never rewrites or "repairs" the plan.
        self.assertEqual(plan_path.read_bytes(), corrupted_bytes)

    def test_non_object_member_is_rejected_without_attribute_error(self) -> None:
        plan_path = self.write_raw_plan(
            "notobject", {"train": ["not-an-object"]}
        )
        with self.assertRaises(SplitError) as caught:
            self.store.get_split("notobject")
        message = str(caught.exception)
        self.assertIn("corrupted", message)
        self.assertIn("train", message)
        self.assertIn("member 1", message)
        # The file is left exactly as saved.
        self.assertTrue(plan_path.read_text(encoding="utf-8").strip().endswith("}"))

    def test_empty_plan_stays_valid(self) -> None:
        self.write_raw_plan(
            "empty",
            {"train": [], "validation": [], "test": []},
            ratios=("1/2", "1/4", "1/4"),
        )
        plan = self.store.get_split("empty")
        self.assertEqual(plan["samples"]["total"], 0)


class SameNameCreateRejectsCorruptionTest(IdentityHarness):
    def corrupt_plan(self, name: str = "p") -> tuple[Path, bytes]:
        members = [
            self.raw_member(self.digest("p-good")),
            self.raw_member(self.digest("p-bad")[:-1]),
        ]
        plan_path = self.write_raw_plan(name, {"train": members})
        return plan_path, plan_path.read_bytes()

    def test_matching_create_reports_corruption_not_reuse(self) -> None:
        self.add_sample("cat")
        plan_path, original = self.corrupt_plan()
        manifest_before = self.store.manifest_path.read_bytes()

        with self.assertRaises(SplitError) as caught:
            self.store.create_split("p", 0, [1, 0, 0])
        message = str(caught.exception)
        self.assertIn("corrupted", message)
        self.assertIn("p", message)
        self.assertIn("train", message)
        self.assertIn("member 2", message)
        self.assertNotIn("already exists", message)

        self.assertEqual(plan_path.read_bytes(), original)
        self.assertEqual(self.store.manifest_path.read_bytes(), manifest_before)

    def test_differing_create_reports_corruption_not_conflict(self) -> None:
        plan_path, original = self.corrupt_plan()
        with self.assertRaises(SplitError) as caught:
            self.store.create_split("p", 99, ["0.5", "0.25", "0.25"])
        message = str(caught.exception)
        self.assertIn("corrupted", message)
        self.assertNotIn("already exists", message)
        self.assertNotIn("name conflict", message)
        # The corrupt plan is preserved, not replaced by the new request.
        self.assertEqual(plan_path.read_bytes(), original)


class ExportIdentityCorruptionTest(IdentityHarness):
    def corrupt_plan(self, name: str = "badid") -> tuple[Path, bytes, str]:
        members = [
            self.raw_member(self.digest("e-good")),
            self.raw_member(self.digest("e-bad").upper()),
        ]
        plan_path = self.write_raw_plan(name, {"train": members})
        return plan_path, plan_path.read_bytes(), members[1]["sha256"]

    def test_export_rejects_with_export_error_and_no_package(self) -> None:
        plan_path, original, bad_value = self.corrupt_plan()
        target = self.root.parent / "badid.zip"
        with self.assertRaises(ExportError) as caught:
            export_split(self.store, "badid", target)
        message = str(caught.exception)
        self.assertIn("corrupted", message)
        self.assertIn("badid", message)
        self.assertIn("train", message)
        self.assertIn("member 2", message)
        self.assertIn(bad_value, message)
        self.assertFalse(target.exists())
        leftovers = [
            p.name for p in target.parent.iterdir() if p.name.startswith(".badid")
        ]
        self.assertEqual(leftovers, [])
        self.assertEqual(plan_path.read_bytes(), original)

    def test_skip_unlabeled_cannot_skip_identity_corruption(self) -> None:
        plan_path, original, _ = self.corrupt_plan()
        target = self.root.parent / "badid-skip.zip"
        with self.assertRaises(ExportError) as caught:
            export_split(self.store, "badid", target, skip_unlabeled=True)
        self.assertIn("corrupted", str(caught.exception))
        self.assertIn("member 2", str(caught.exception))
        self.assertFalse(target.exists())
        self.assertEqual(plan_path.read_bytes(), original)

    def test_source_dir_export_rejects_before_scanning(self) -> None:
        self.corrupt_plan()
        source_dir = self.root.parent / "sources"
        source_dir.mkdir()
        target = self.root.parent / "badid-src.zip"
        with self.assertRaises(ExportError) as caught:
            export_split(
                self.store, "badid", target, source_dir=source_dir
            )
        self.assertIn("corrupted", str(caught.exception))
        self.assertIn("member 2", str(caught.exception))
        self.assertFalse(target.exists())

    def test_missing_plan_still_reports_not_found(self) -> None:
        target = self.root.parent / "missing.zip"
        with self.assertRaises(ExportError) as caught:
            export_split(self.store, "nope", target)
        self.assertIn("not found", str(caught.exception))
        self.assertFalse(target.exists())


class SavedPlanJudgesFormatOnlyTest(IdentityHarness):
    def test_format_correct_plan_survives_later_workspace_changes(self) -> None:
        digest = self.add_sample("cat", b"plan-content")
        plan = self.store.create_split("keep", 0, [1, 0, 0]).plan
        self.assertEqual(plan["sets"]["train"]["members"][0]["sha256"], digest)

        # Image moved/deleted, label changed, new sample imported: the
        # format-valid saved plan is not corruption.
        Path(plan["sets"]["train"]["members"][0]["source"]).unlink()
        self.store.submit_batch(
            {"batch": "b1", "changes": [{"sha256": digest, "old": "cat", "new": "dog"}]}
        )
        self.add_sample("bird", b"newcomer")

        viewed = self.store.get_split("keep")
        self.assertEqual(viewed, plan)

    def test_hand_written_valid_plan_readable_without_registration(self) -> None:
        # Members need not exist in the current manifest at all.
        digest = self.digest("ghost")
        self.write_raw_plan("ghost", {"train": [self.raw_member(digest)]})
        viewed = self.store.get_split("ghost")
        self.assertEqual(
            viewed["sets"]["train"]["members"][0]["sha256"], digest
        )


class CliIdentityCorruptionTest(IdentityHarness):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "vision_workbench", *arguments],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def corrupt_plan(self, name: str = "broken") -> tuple[Path, bytes]:
        members = [
            self.raw_member(self.digest("cli-good-1")),
            self.raw_member(self.digest("cli-good-2")),
            self.raw_member(self.digest("cli-bad")[:-1]),
        ]
        plan_path = self.write_raw_plan(name, {"train": members})
        return plan_path, plan_path.read_bytes()

    def test_show_export_create_all_fail_readably(self) -> None:
        plan_path, original = self.corrupt_plan()

        shown = self.run_cli("split", "show", str(self.root), "broken")
        self.assertEqual(shown.returncode, 1)
        self.assertEqual(shown.stdout, "")
        self.assertIn("error:", shown.stderr)
        for fragment in ("corrupted", "broken", "train", "member 3"):
            self.assertIn(fragment, shown.stderr)
        self.assertNotIn("Traceback", shown.stderr)

        target = self.root.parent / "broken.zip"
        exported = self.run_cli(
            "export", str(self.root), "broken", str(target)
        )
        self.assertEqual(exported.returncode, 1)
        self.assertEqual(exported.stdout, "")
        self.assertIn("error:", exported.stderr)
        for fragment in ("corrupted", "broken", "train", "member 3"):
            self.assertIn(fragment, exported.stderr)
        self.assertNotIn("Traceback", exported.stderr)
        self.assertFalse(target.exists())
        leftovers = [
            p.name
            for p in target.parent.iterdir()
            if p.name.startswith(".broken.zip")
        ]
        self.assertEqual(leftovers, [])

        skipped = self.run_cli(
            "export", str(self.root), "broken", str(target), "--skip-unlabeled"
        )
        self.assertEqual(skipped.returncode, 1)
        self.assertEqual(skipped.stdout, "")
        self.assertIn("corrupted", skipped.stderr)
        self.assertNotIn("Traceback", skipped.stderr)
        self.assertFalse(target.exists())

        # Same-name create with matching inputs: corruption, not reuse.
        recreated = self.run_cli(
            "split", "create", str(self.root), "broken",
            "--seed", "0", "--train", "1", "--validation", "0", "--test", "0",
        )
        self.assertEqual(recreated.returncode, 1)
        self.assertEqual(recreated.stdout, "")
        self.assertIn("corrupted", recreated.stderr)
        self.assertNotIn("already exists", recreated.stderr)
        self.assertNotIn("Traceback", recreated.stderr)

        # Same-name create with differing inputs: corruption, not conflict.
        conflicted = self.run_cli(
            "split", "create", str(self.root), "broken",
            "--seed", "5", "--train", "0.5",
            "--validation", "0.25", "--test", "0.25",
        )
        self.assertEqual(conflicted.returncode, 1)
        self.assertEqual(conflicted.stdout, "")
        self.assertIn("corrupted", conflicted.stderr)
        self.assertNotIn("already exists", conflicted.stderr)

        # Nothing was rewritten: plan and an empty manifest are intact.
        self.assertEqual(plan_path.read_bytes(), original)
        self.assertEqual(
            json.loads(self.store.manifest_path.read_text(encoding="utf-8"))["items"],
            [],
        )

    def test_show_accepts_valid_plan_not_in_registry(self) -> None:
        # Show judges the saved plan alone: members need not exist in the
        # current manifest, as long as every identity is well-formed.
        digest = self.digest("fine")
        self.write_raw_plan("fine", {"train": [self.raw_member(digest)]})
        shown = self.run_cli("split", "show", str(self.root), "fine")
        self.assertEqual(shown.returncode, 0, shown.stderr)
        body = json.loads(shown.stdout)
        self.assertEqual(
            body["sets"]["train"]["members"][0]["sha256"], digest
        )


if __name__ == "__main__":
    unittest.main()
