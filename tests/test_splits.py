from __future__ import annotations

import unittest
from fractions import Fraction

from vision_workbench.splits import (
    SET_NAMES,
    SplitError,
    allocation_table,
    assign,
    category_key,
    category_name,
    parse_ratio,
    seeded_order,
    validate_name,
    validate_ratios,
)


def samples(counts: dict[str | object, int]) -> list[dict]:
    """Build manifest-like samples; key None means the unlabeled stratum."""
    built: list[dict] = []
    serial = 0
    for label, count in counts.items():
        for _ in range(count):
            serial += 1
            digest = f"{serial:08d}"
            built.append({"sha256": digest, "label": None if label is None else label})
    return built


class AllocationPropertyTest(unittest.TestCase):
    def assertMarginalBounds(
        self, counts: dict, ratios: tuple[Fraction, Fraction, Fraction]
    ) -> None:
        table = allocation_table(counts, ratios)
        total = sum(counts.values())
        for key, n_c in counts.items():
            row = table[key]
            self.assertEqual(sum(row), n_c)
            for j, ratio in enumerate(ratios):
                expected = Fraction(n_c) * ratio
                self.assertLess(
                    abs(Fraction(row[j]) - expected),
                    1,
                    f"class row {key} set {SET_NAMES[j]}: {row[j]} vs {expected}",
                )
        for j, ratio in enumerate(ratios):
            column = sum(table[key][j] for key in counts)
            expected = Fraction(total) * ratio
            self.assertLess(
                abs(Fraction(column) - expected),
                1,
                f"total set {SET_NAMES[j]}: {column} vs {expected}",
            )

    def test_many_distributions_and_ratios(self) -> None:
        ratio_sets = [
            (Fraction(1, 3), Fraction(1, 3), Fraction(1, 3)),
            (Fraction(7, 10), Fraction(2, 10), Fraction(1, 10)),
            (Fraction(0), Fraction(0), Fraction(1)),
            (Fraction(1), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(1, 2), Fraction(1, 2)),
            (Fraction(1, 7), Fraction(2, 7), Fraction(4, 7)),
            (Fraction(1, 2), Fraction(1, 3), Fraction(1, 6)),
        ]
        distributions = [
            {"a": 1},
            {"a": 1, "b": 1},
            {"a": 2},
            {"a": 1, "b": 1, "c": 1},
            {"a": 2, "b": 1},
            {"a": 3, "b": 2, "c": 1},
            {"a": 5, "b": 7, "c": 11, "d": 1},
            {str(i): 1 for i in range(20)},
            {"a": 1, "b": 2, "c": 4, "d": 8},
            {"x": 100, "y": 1},
            {"x": 100, "y": 2},
            {str(k): (k % 5) + 1 for k in range(12)},
        ]
        for counts in distributions:
            for ratios in ratio_sets:
                with self.subTest(counts=counts, ratios=ratios):
                    self.assertMarginalBounds(counts, ratios)

    def test_single_sample_classes_with_thirds(self) -> None:
        # Every class has one sample; totals must still stay within <1.
        counts = {f"c{i}": 1 for i in range(10)}
        table = allocation_table(
            counts, (Fraction(1, 3), Fraction(1, 3), Fraction(1, 3))
        )
        totals = [sum(table[c][j] for c in counts) for j in range(3)]
        self.assertEqual(sum(totals), 10)
        for j in range(3):
            self.assertLess(abs(Fraction(totals[j]) - Fraction(10, 3)), 1)

    def test_empty_dataset(self) -> None:
        table = allocation_table(
            {}, (Fraction(1, 3), Fraction(1, 3), Fraction(1, 3))
        )
        self.assertEqual(table, {})


class AssignTest(unittest.TestCase):
    def test_partition_and_zero_sets_empty(self) -> None:
        data = samples({"a": 4, "b": 3})
        assignment = assign(data, [0, Fraction(1, 2), Fraction(1, 2)], seed=7)
        self.assertEqual(set(assignment), {s["sha256"] for s in data})
        self.assertNotIn("train", set(assignment.values()))
        self.assertTrue(all(value in SET_NAMES for value in assignment.values()))

    def test_unlabeled_distinct_from_literal_class(self) -> None:
        data = [
            {"sha256": "1", "label": None},
            {"sha256": "2", "label": ""},
            {"sha256": "3", "label": "unlabeled"},
        ]
        assignment = assign(data, [1, 0, 0], seed=0)
        self.assertEqual(set(assignment.values()), {"train"})
        # None and "" share a stratum; the literal label is its own stratum.
        self.assertIs(category_key(None), category_key(""))
        self.assertIsNot(category_key("unlabeled"), category_key(None))
        self.assertEqual(category_name(category_key(None)), "")
        self.assertEqual(category_name(category_key("unlabeled")), "unlabeled")

    def test_deterministic_regardless_of_order(self) -> None:
        data = samples({"a": 6, "b": 5, "c": 3})
        first = assign(data, [0.7, 0.2, 0.1], seed=42)
        reordered = list(reversed(data))
        second = assign(reordered, ["0.7", "0.2", "0.1"], seed=42)
        self.assertEqual(first, second)

    def test_seed_changes_selection_on_larger_dataset(self) -> None:
        data = samples({f"c{k}": 6 for k in range(8)})
        ratios = [Fraction(8, 10), Fraction(1, 10), Fraction(1, 10)]
        assignments = {
            seed: tuple(sorted((d, a[d]) for d in (s["sha256"] for s in data)))
            for seed, a in ((s, assign(data, ratios, s)) for s in range(6))
        }
        # At least two seeds must pick different samples for the small sets.
        self.assertGreater(len(set(assignments.values())), 1)

    def test_small_dataset_seed_may_be_identical(self) -> None:
        data = samples({"a": 1})
        self.assertEqual(
            assign(data, [1, 0, 0], seed=0), assign(data, [1, 0, 0], seed=999)
        )

    def test_seeded_order_only_depends_on_set(self) -> None:
        digests = [f"{i:04d}" for i in range(20)]
        self.assertEqual(
            seeded_order(digests, 5), seeded_order(list(reversed(digests)), 5)
        )
        self.assertEqual(seeded_order(digests, 5), seeded_order(digests, 5))

    def test_duplicate_identity_rejected(self) -> None:
        data = [
            {"sha256": "1", "label": "a"},
            {"sha256": "1", "label": "b"},
        ]
        with self.assertRaises(SplitError):
            assign(data, [1, 0, 0], seed=0)

    def test_missing_identity_rejected(self) -> None:
        with self.assertRaises(SplitError):
            assign([{"label": "a"}], [1, 0, 0], seed=0)


class ValidationTest(unittest.TestCase):
    def test_ratio_rules(self) -> None:
        self.assertEqual(validate_ratios([0, 1, 0]), (Fraction(0), Fraction(1), Fraction(0)))
        self.assertEqual(parse_ratio("1/3"), Fraction(1, 3))
        self.assertEqual(parse_ratio(0.1), Fraction(1, 10))
        for bad in ["nan", "inf", "-inf", "1.0.0", "abc", -0.01, 1.01]:
            with self.assertRaises(SplitError):
                parse_ratio(bad)
        with self.assertRaises(SplitError):
            parse_ratio(True)
        with self.assertRaises(SplitError):
            validate_ratios([0.34, 0.33, 0.32])  # 0.99, not 1
        self.assertEqual(
            validate_ratios(["0.34", "0.33", "0.33"]),
            (Fraction(34, 100), Fraction(33, 100), Fraction(33, 100)),
        )

    def test_seed_must_be_integer(self) -> None:
        with self.assertRaises(SplitError):
            assign([], [1, 0, 0], 1.5)
        with self.assertRaises(SplitError):
            assign([], [1, 0, 0], True)

    def test_name_rules(self) -> None:
        for name in ["ok", "a b", "plan_v2", "..." ]:
            validate_name(name)
        for bad in ["", "   ", "a/b", "a\\b", ".", ".."]:
            with self.assertRaises(SplitError):
                validate_name(bad)


if __name__ == "__main__":
    unittest.main()
