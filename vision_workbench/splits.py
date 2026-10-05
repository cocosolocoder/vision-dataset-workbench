"""Stratified train/validation/test split planning.

A plan allocates every sample to exactly one of three sets.  Exact rational
arithmetic (:mod:`fractions`) drives the allocation so that, for every
class *and* for the dataset as a whole, each set's sample count differs
from its proportional expectation by strictly less than one.

Assignments derive solely from sample identities (content digests), labels
and the integer seed, so they are independent of import order, manifest
ordering, workspace location and process restarts.
"""

from __future__ import annotations

import hashlib
import math
import re
from fractions import Fraction
from typing import Any, Sequence

# Sentinel for samples without a label.  An object (not the string
# "unlabeled") keeps a real class literally named "unlabeled" separate.
UNLABELED = object()

SET_NAMES = ("train", "validation", "test")

# A sample identity is a full SHA-256 digest: exactly 64 lowercase
# hexadecimal characters.  The same spelling the registration manifest
# enforces is required inside saved split plans, so a truncated digest,
# an uppercase spelling, surrounding whitespace or a stray non-hex digit
# can never be repaired by truncating, padding, stripping or changing
# case.
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")


class SplitError(ValueError):
    """A split parameter or record is invalid."""


def invalid_digest_reason(value: Any) -> str | None:
    """Return why ``value`` is not a full lowercase SHA-256 digest, else None.

    The accepted spelling is exactly 64 lowercase hexadecimal characters,
    matching the registered sample identity format.  A non-string value
    (including a missing ``None``), an empty string, a digest of any other
    length, an uppercase spelling or surrounding whitespace is rejected
    with a concrete reason; the value is never normalized into acceptance.
    """
    if value is None:
        return (
            "the field is missing or null; expected a string of 64 "
            "lowercase hexadecimal characters"
        )
    if not isinstance(value, str):
        return (
            f"it is a {type(value).__name__}, not a string; expected a "
            "string of 64 lowercase hexadecimal characters"
        )
    if not value:
        return "it is empty; expected 64 lowercase hexadecimal characters"
    if value != value.strip() or any(char in " \t\r\n" for char in value):
        return (
            "it contains surrounding or embedded whitespace; expected 64 "
            "lowercase hexadecimal characters without spaces"
        )
    if len(value) != 64:
        return (
            f"it has {len(value)} characters instead of 64; expected 64 "
            "lowercase hexadecimal characters"
        )
    if _DIGEST_RE.fullmatch(value) is None:
        # Length is exactly 64 here, so every remaining problem is a
        # non-hex or uppercase character.
        return (
            "it must contain only lowercase hexadecimal characters "
            "(digits 0-9 and letters a-f)"
        )
    return None


def category_key(label: Any) -> Any:
    """Stratum key for a manifest label value.

    ``None`` and the empty string both mean "unlabeled", matching
    :meth:`DatasetStore.summary`; any other label is taken literally.
    """
    if label is None or label == "":
        return UNLABELED
    return label


def category_name(key: Any) -> str:
    """Render a stratum key for JSON output.

    The unlabeled stratum renders as the empty string (which mirrors the
    manifest's empty-label meaning); a genuine class literally named
    ``"unlabeled"`` therefore stays distinguishable.
    """
    return "" if key is UNLABELED else str(key)


def _sort_key(key: Any) -> tuple[int, str]:
    return (0 if key is UNLABELED else 1, category_name(key))


def validate_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise SplitError("Split plan name must not be empty or only whitespace")
    if "/" in name or "\\" in name:
        raise SplitError("Split plan name must not contain '/' or '\\'")
    if name in (".", ".."):
        raise SplitError("Split plan name must not be '.' or '..'")
    return name


def parse_ratio(value: Any) -> Fraction:
    """Parse one ratio as a finite number in the closed interval [0, 1]."""
    if isinstance(value, bool):
        raise SplitError(f"Ratio must be a number, got {value!r}")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SplitError(f"Ratio must be finite, got {value!r}")
        # Use the shortest decimal that round-trips, so 0.1 means 1/10
        # rather than its binary approximation.
        value = repr(value)
    try:
        if isinstance(value, Fraction):
            ratio = value
        elif isinstance(value, int):
            ratio = Fraction(value)
        elif isinstance(value, str):
            ratio = Fraction(value)
        else:
            raise SplitError(f"Ratio must be a number, got {value!r}")
    except (ValueError, ZeroDivisionError, ArithmeticError, OverflowError):
        raise SplitError(f"Ratio must be a finite number, got {value!r}") from None
    if ratio < 0 or ratio > 1:
        raise SplitError(
            f"Ratio must be between 0 and 1 (inclusive), got {value!r}"
        )
    return ratio


def validate_ratios(ratios: Sequence[Any]) -> tuple[Fraction, Fraction, Fraction]:
    if len(ratios) != 3:
        raise SplitError("Exactly three ratios (train, validation, test) are required")
    parsed = tuple(parse_ratio(value) for value in ratios)
    if sum(parsed, Fraction(0)) != 1:
        raise SplitError("Ratios must sum to exactly 1")
    return parsed  # type: ignore[return-value]


def allocation_table(
    strata_counts: dict[Any, int], ratios: tuple[Fraction, Fraction, Fraction]
) -> dict[Any, tuple[int, int, int]]:
    """Allocate integer ``(train, validation, test)`` counts per stratum.

    Construction: every cell starts at ``floor(N_c * p_s)``; each stratum
    then distributes its leftover free units (at most two per row) and each
    column total must land on ``floor`` or ``ceil`` of its own expectation.
    A free unit may only enter a non-integral cell.  An integral placement
    always exists (controlled rounding of the fractional-parts table); the
    small max-flow below searches the deterministic floor/ceil choices.
    Every used cell ends at ``floor`` or ``ceil`` of its expectation and
    every column total at ``floor``/``ceil`` of its, hence every
    discrepancy is strictly below one.
    """
    strata = sorted(strata_counts, key=_sort_key)
    total = sum(strata_counts.values())

    floors: dict[Any, list[int]] = {}
    parts: dict[Any, list[Fraction]] = {}
    for c in strata:
        expectation = [Fraction(strata_counts[c]) * ratios[j] for j in range(3)]
        floors[c] = [e.numerator // e.denominator for e in expectation]
        parts[c] = [e - floors[c][j] for j, e in enumerate(expectation)]

    # Fractional column sums S_j; a column receives floor(S_j) or
    # ceil(S_j) free units.  sum_j S_j is the integer number of free
    # units, so the fractional parts sum to an integer k: exactly k
    # non-integral columns round up.  A column whose S_j is integral
    # cannot round up (that would be a discrepancy of exactly one), so
    # the size-k subsets are tried until a flow matches every row.
    column_sums = [sum((parts[c][j] for c in strata), Fraction(0)) for j in range(3)]
    column_base = [s.numerator // s.denominator for s in column_sums]
    row_supply = {c: int(sum(parts[c])) for c in strata}
    extra_needed = sum(row_supply.values()) - sum(column_base)
    eligible = [j for j in range(3) if column_sums[j] != column_base[j]]

    flows = _controlled_rounding(
        strata, parts, row_supply, column_base, extra_needed, eligible
    )

    table: dict[Any, tuple[int, int, int]] = {}
    for c in strata:
        table[c] = tuple(floors[c][j] + flows[c][j] for j in range(3))  # type: ignore[assignment]
        if sum(table[c]) != strata_counts[c]:
            raise SplitError("Internal error: stratum count mismatch")  # pragma: no cover
    for j in range(3):
        column_total = sum(table[c][j] for c in strata)
        expect_total = Fraction(total) * ratios[j]
        if abs(Fraction(column_total) - expect_total) >= 1:
            raise SplitError("Internal error: total count mismatch")  # pragma: no cover
    return table


def _controlled_rounding(
    strata: Sequence[Any],
    parts: dict[Any, list[Fraction]],
    row_supply: dict[Any, int],
    column_base: list[int],
    extra_needed: int,
    eligible: Sequence[int],
) -> dict[Any, list[int]]:
    """Place every row's free units on non-integral cells.

    Column j receives ``column_base[j]`` units, plus one for each member
    of a size-``extra_needed`` subset of eligible columns (its total then
    rounds ``S_j`` to floor or ceil).  Candidate subsets are tried in a
    fixed order; the fractional parts themselves prove the rounded
    polytope has an integral vertex, so one candidate always yields a
    full-value flow.
    """
    from itertools import combinations

    if extra_needed == 0:
        candidates: list[tuple[int, ...]] = [()]
    else:
        candidates = list(combinations(eligible, extra_needed))

    last_error: Exception | None = None
    for extra in candidates:
        demand = column_base[:]
        for j in extra:
            demand[j] += 1
        try:
            return _integral_flow(strata, parts, row_supply, demand)
        except SplitError as error:  # candidate infeasible, try next
            last_error = error
    raise SplitError(
        f"Internal error: cannot satisfy split proportions: {last_error}"
    )  # pragma: no cover


def _integral_flow(
    strata: Sequence[Any],
    parts: dict[Any, list[Fraction]],
    row_supply: dict[Any, int],
    column_demand: list[int],
) -> dict[Any, list[int]]:
    """Max-flow placement; raises unless every free unit is placed."""
    source = ("s",)
    sink = ("t",)
    row_nodes = [("r", i) for i in range(len(strata))]
    col_nodes = [("c", j) for j in range(3)]
    residual: dict[Any, dict[Any, int]] = {source: {}, sink: {}}
    for node in row_nodes + col_nodes:
        residual[node] = {}

    def add_edge(u: Any, v: Any, capacity: int) -> None:
        residual[u][v] = capacity
        residual[v].setdefault(u, 0)

    required = 0
    for i, c in enumerate(strata):
        supply = row_supply[c]
        required += supply
        if supply:
            add_edge(source, row_nodes[i], supply)
        for j in sorted(range(3), key=lambda k: (-parts[c][k], k)):
            if parts[c][j] > 0:
                add_edge(row_nodes[i], col_nodes[j], 1)
    for j, node in enumerate(col_nodes):
        add_edge(node, sink, column_demand[j])

    pushed = 0
    while pushed < required:
        path = _augmenting_path(residual, source, sink, row_nodes, col_nodes)
        if path is None:
            raise SplitError("no feasible integral placement")
        for u, v in zip(path, path[1:]):
            residual[u][v] -= 1
            residual[v][u] += 1
        pushed += 1

    flows = {c: [0, 0, 0] for c in strata}
    for i, c in enumerate(strata):
        for j, col in enumerate(col_nodes):
            # Forward edge capacity was 1; residual reverse flow of 1
            # means the unit was placed in this cell.
            flows[c][j] = residual[col].get(row_nodes[i], 0)
    return flows


def _augmenting_path(
    residual: dict[Any, dict[Any, int]],
    source: Any,
    sink: Any,
    row_nodes: Sequence[Any],
    col_nodes: Sequence[Any],
) -> list[Any] | None:
    """Breadth-first augmenting path with fixed node traversal order."""
    order = [source, *row_nodes, *col_nodes, sink]
    previous: dict[Any, Any] = {source: None}
    queue = [source]
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        for neighbor in order:
            if neighbor in previous:
                continue
            if residual.get(node, {}).get(neighbor, 0) > 0:
                previous[neighbor] = node
                if neighbor == sink:
                    path = [sink]
                    while path[-1] is not source:
                        path.append(previous[path[-1]])
                    path.reverse()
                    return path
                queue.append(neighbor)
    return None


def seeded_order(digests: Sequence[str], seed: int) -> list[str]:
    """Deterministic seed-driven order of the given digests.

    Each digest gets a sort key ``H(seed || digest)`` from SHA-256, with
    the digest itself breaking ties.  Unlike :mod:`random`, this stream is
    defined by the hash algorithm itself, so it is stable across Python
    versions and only depends on the digest set and the seed.
    """
    seed_bytes = str(seed).encode("utf-8")
    return sorted(
        digests,
        key=lambda digest: (
            hashlib.sha256(seed_bytes + digest.encode("utf-8")).digest(),
            digest,
        ),
    )


def assign(samples: Sequence[dict[str, Any]], ratios: Sequence[Any], seed: Any) -> dict[str, str]:
    """Return ``{digest: set_name}`` for the given manifest-like samples.

    Each sample mapping must contain ``sha256`` and ``label`` keys.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SplitError(f"Seed must be an integer, got {seed!r}")
    ratio_values = validate_ratios(ratios)

    digests: list[str] = []
    seen: set[str] = set()
    strata_counts: dict[Any, int] = {}
    stratum_of: dict[str, Any] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            raise SplitError("Every sample record must be an object")
        digest = sample.get("sha256")
        if not isinstance(digest, str) or not digest:
            raise SplitError("Every sample needs a non-empty 'sha256' identity")
        if digest in seen:
            raise SplitError(f"Duplicate sample identity in manifest: {digest}")
        seen.add(digest)
        key = category_key(sample.get("label"))
        stratum_of[digest] = key
        strata_counts[key] = strata_counts.get(key, 0) + 1
        digests.append(digest)

    table = allocation_table(strata_counts, ratio_values)
    positions = {digest: i for i, digest in enumerate(seeded_order(digests, seed))}

    assignment: dict[str, str] = {}
    for key, row in table.items():
        members = sorted(
            (d for d in digests if stratum_of[d] == key),
            key=lambda d: positions[d],
        )
        start = 0
        for j, count in enumerate(row):
            for digest in members[start : start + count]:
                assignment[digest] = SET_NAMES[j]
            start += count
    return assignment
