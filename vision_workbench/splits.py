"""Persistent train/validation/test split schemes for vision datasets.

A split scheme records, at creation time, every sample's identity (content
digest), classification label, source path, and which set it belongs to.
Schemes are stored as JSON files under ``.vision-workbench/splits/`` and are
never updated by later imports.

The assignment is stratified: for every class (unlabeled samples form their
own stratum, separate from any real class literally named ``unlabeled``),
the number of samples allocated to each set differs from the expected count
(ratio x stratum size) by strictly less than one. The same holds for the
total counts per set.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

SETS = ("train", "validation", "test")

_SCHEMA_VERSION = 1
_SPLIT_DIRECTORY = "splits"
_UNLABELED = ("unlabeled", "")


class SplitError(ValueError):
    """Raised when a split scheme cannot be created, read, or applied."""


def _validate_name(name: Any) -> str:
    if not isinstance(name, str) or not name:
        raise SplitError("Split name must not be empty")
    if name.strip() == "":
        raise SplitError("Split name must not be only whitespace")
    if "/" in name or "\\" in name:
        raise SplitError("Split name must not contain '/' or '\\'")
    if name in (".", ".."):
        raise SplitError("Split name must not be '.' or '..'")
    if any(ord(character) < 32 for character in name):
        raise SplitError("Split name must not contain control characters")
    return name


def _validate_ratios(ratios: Any) -> dict[str, float]:
    if not isinstance(ratios, dict):
        raise SplitError("Ratios must be an object mapping set names to numbers")
    missing = [set_name for set_name in SETS if set_name not in ratios]
    if missing:
        raise SplitError(f"Missing ratios for: {', '.join(missing)}")
    validated: dict[str, float] = {}
    for set_name in SETS:
        value = ratios[set_name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SplitError(f"Ratio for {set_name} must be a finite number")
        value = float(value)
        if not math.isfinite(value):
            raise SplitError(f"Ratio for {set_name} must be finite")
        if not 0.0 <= value <= 1.0:
            raise SplitError(f"Ratio for {set_name} must be between 0 and 1")
        validated[set_name] = value
    total = sum(validated.values())
    if not math.isfinite(total) or abs(total - 1.0) > 1e-9:
        raise SplitError(f"Ratios must sum to 1 (got {total!r})")
    return validated


def _stratum_key(label: Any) -> tuple[str, str]:
    if label is None or label == "":
        return _UNLABELED
    return ("label", str(label))


def _largest_remainder(quotas: dict[str, float], total: int) -> dict[str, int]:
    """Distribute ``total`` seats proportionally to ``quotas``.

    Each set receives either the floor or the ceiling of its quota, so the
    result differs from every quota by strictly less than one. Ties are
    broken in the fixed set order for determinism.
    """

    floors = {set_name: math.floor(quotas[set_name]) for set_name in SETS}
    remainder = total - sum(floors.values())
    if remainder < 0 or remainder > len(SETS):
        raise SplitError("Cannot allocate split counts")
    order = sorted(
        SETS,
        key=lambda set_name: (
            -(quotas[set_name] - floors[set_name]),
            SETS.index(set_name),
        ),
    )
    targets = dict(floors)
    for set_name in order[:remainder]:
        targets[set_name] += 1
    return targets


def _bipartite_flow(
    supplies: dict[Any, int],
    edges: Iterable[tuple[Any, str]],
    demands: dict[str, int],
) -> set[tuple[Any, str]]:
    """Find a 0/1 flow matching row supplies to set demands.

    Rows are stratum keys; an edge (key, set_name) means the stratum may send
    one sample to that set. Returns the set of edges carrying flow.
    """

    source = "<source>"
    sink = "<sink>"
    usable_edges = [
        (row, set_name)
        for row, set_name in edges
        if supplies.get(row, 0) > 0 and demands.get(set_name, 0) > 0
    ]
    nodes = {source, sink}
    for row, supply in supplies.items():
        if supply > 0:
            nodes.add(row)
    for set_name, demand in demands.items():
        if demand > 0:
            nodes.add(set_name)
    for row, set_name in usable_edges:
        nodes.add(row)
        nodes.add(set_name)

    capacity: dict[Any, dict[Any, int]] = {node: {} for node in nodes}
    for row, set_name in usable_edges:
        capacity[row][set_name] = capacity[row].get(set_name, 0) + 1
    for row, supply in supplies.items():
        if supply > 0 and row in nodes:
            capacity[source][row] = supply
    for set_name, demand in demands.items():
        if demand > 0 and set_name in nodes:
            capacity[set_name][sink] = demand

    while True:
        parent: dict[Any, Any] = {source: None}
        queue = [source]
        while queue and sink not in parent:
            node = queue.pop(0)
            for neighbor, residual in capacity[node].items():
                if residual > 0 and neighbor not in parent:
                    parent[neighbor] = node
                    queue.append(neighbor)
        if sink not in parent:
            break
        path: list[tuple[Any, Any]] = []
        node = sink
        while node != source:
            previous = parent[node]
            path.append((previous, node))
            node = previous
        bottleneck = min(capacity[row][set_name] for row, set_name in path)
        for row, set_name in path:
            capacity[row][set_name] -= bottleneck
            capacity[set_name][row] = capacity[set_name].get(row, 0) + bottleneck

    used = set()
    for row, set_name in usable_edges:
        if capacity[row].get(set_name, 0) == 0:
            used.add((row, set_name))
    return used


def _allocate(
    strata: dict[tuple[str, str], list[dict[str, Any]]],
    ratios: dict[str, float],
) -> dict[tuple[str, str], dict[str, int]]:
    """Compute per-stratum, per-set sample counts.

    Global targets are set by largest remainder. Each stratum first gets the
    floor of its quota; the remaining seats are filled by a bipartite flow
    that respects both the stratum totals and the global targets. Every
    resulting count is the floor or ceiling of its quota, hence differs by
    strictly less than one.
    """

    total = sum(len(members) for members in strata.values())
    targets = _largest_remainder(
        {set_name: ratios[set_name] * total for set_name in SETS}, total
    )

    base: dict[tuple[str, str], dict[str, int]] = {}
    supplies: dict[tuple[str, str], int] = {}
    edges: list[tuple[tuple[str, str], str]] = []
    demands = dict(targets)

    for key, members in strata.items():
        count = len(members)
        per_set: dict[str, int] = {}
        for set_name in SETS:
            quota = ratios[set_name] * count
            per_set[set_name] = math.floor(quota)
            if math.ceil(quota) > math.floor(quota):
                edges.append((key, set_name))
        base[key] = per_set
        supplies[key] = count - sum(per_set.values())
        for set_name in SETS:
            demands[set_name] -= per_set[set_name]

    if any(demand < 0 for demand in demands.values()):
        raise SplitError("Cannot allocate split counts")

    used = _bipartite_flow(supplies, edges, demands)
    if len(used) != sum(supplies.values()):
        raise SplitError("Cannot allocate split counts")

    allocation: dict[tuple[str, str], dict[str, int]] = {}
    for key in strata:
        allocation[key] = {
            set_name: base[key][set_name] + (1 if (key, set_name) in used else 0)
            for set_name in SETS
        }
    return allocation


def _assign(
    records: list[dict[str, Any]],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, str]:
    """Assign each sample to a set, stratified by label.

    The result depends only on the sample identities, their labels, the
    ratios, and the seed: manifest order, workspace path, and restart count
    never influence it.
    """

    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        strata[_stratum_key(record["label"])].append(record)
    allocation = _allocate(strata, ratios)

    assignment: dict[str, str] = {}
    for key, members in strata.items():
        counts = allocation[key]
        ordered = sorted(members, key=lambda record: record["sha256"])
        seed_material = json.dumps(
            {"seed": seed, "stratum": list(key)},
            ensure_ascii=False,
            sort_keys=True,
        )
        seed_digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
        random.Random(int(seed_digest, 16)).shuffle(ordered)
        offset = 0
        for set_name in SETS:
            for record in ordered[offset : offset + counts[set_name]]:
                assignment[record["sha256"]] = set_name
            offset += counts[set_name]
    return assignment


class SplitStore:
    """Reads and writes split scheme files in a workspace state directory."""

    def __init__(self, state_directory: Path) -> None:
        self.state_directory = state_directory
        self.splits_directory = state_directory / _SPLIT_DIRECTORY

    def create(
        self,
        records: list[dict[str, Any]],
        name: str,
        seed: int,
        ratios: Any,
    ) -> dict[str, Any]:
        name = _validate_name(name)
        ratios = _validate_ratios(ratios)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise SplitError("Seed must be an integer")

        normalized = self._normalize_records(records)
        path = self._scheme_path(name)
        if path.exists():
            existing = self._read(path)
            if self._matches(existing, normalized, seed, ratios):
                return self._response(existing)
            raise SplitError(f"Split name already exists: {name!r}")

        assignment = _assign(normalized, ratios, seed)
        scheme = {
            "schema_version": _SCHEMA_VERSION,
            "name": name,
            "seed": seed,
            "ratios": {set_name: ratios[set_name] for set_name in SETS},
            "items": [
                {
                    "sha256": record["sha256"],
                    "label": record["label"],
                    "source": record["source"],
                    "set": assignment[record["sha256"]],
                }
                for record in normalized
            ],
        }
        self._write_atomic(path, scheme)
        return self._response(scheme)

    def show(self, name: str) -> dict[str, Any]:
        name = _validate_name(name)
        path = self._scheme_path(name)
        if not path.exists():
            raise SplitError(f"Split does not exist: {name!r}")
        return self._response(self._read(path))

    def _scheme_path(self, name: str) -> Path:
        return self.splits_directory / f"{name}.json"

    @staticmethod
    def _normalize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                raise SplitError("Manifest record is not an object")
            digest = record.get("sha256")
            if not isinstance(digest, str) or not digest:
                raise SplitError("Manifest record missing sha256")
            if digest in seen:
                raise SplitError(f"Duplicate sample identity: {digest}")
            seen.add(digest)
            label = record.get("label")
            if label is not None and not isinstance(label, str):
                raise SplitError(f"Record {digest} has an invalid label")
            source = record.get("source")
            if not isinstance(source, str) or not source:
                raise SplitError(f"Record {digest} missing source")
            normalized.append(
                {"sha256": digest, "label": label, "source": source}
            )
        normalized.sort(key=lambda record: record["sha256"])
        return normalized

    @staticmethod
    def _matches(
        scheme: dict[str, Any],
        records: list[dict[str, Any]],
        seed: int,
        ratios: dict[str, float],
    ) -> bool:
        if scheme.get("seed") != seed:
            return False
        stored_ratios = scheme.get("ratios", {})
        if not isinstance(stored_ratios, dict):
            return False
        for set_name in SETS:
            stored = stored_ratios.get(set_name)
            if not isinstance(stored, (int, float)) or isinstance(stored, bool):
                return False
            if abs(float(stored) - ratios[set_name]) > 1e-12:
                return False
        current = {(record["sha256"], record["label"]) for record in records}
        stored_pairs: set[tuple[str, Any]] = set()
        for item in scheme.get("items", []):
            if not isinstance(item, dict) or "sha256" not in item:
                return False
            stored_pairs.add((item["sha256"], item.get("label")))
        return current == stored_pairs

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SplitError(f"Cannot read split scheme: {error}") from error
        if not isinstance(data, dict) or data.get("schema_version") != _SCHEMA_VERSION:
            raise SplitError("Unsupported split scheme")
        if not isinstance(data.get("items"), list):
            raise SplitError("Corrupt split scheme")
        for item in data["items"]:
            if not isinstance(item, dict):
                raise SplitError("Corrupt split scheme")
            if not isinstance(item.get("sha256"), str) or not item["sha256"]:
                raise SplitError("Corrupt split scheme")
            if item.get("label") is not None and not isinstance(item.get("label"), str):
                raise SplitError("Corrupt split scheme")
            if item.get("set") not in SETS:
                raise SplitError("Corrupt split scheme")
        return data

    @staticmethod
    def _write_atomic(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.tmp")
        try:
            temporary.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
        except OSError as error:
            raise SplitError(f"Cannot write split scheme: {error}") from error

    @classmethod
    def _response(cls, scheme: dict[str, Any]) -> dict[str, Any]:
        items = sorted(scheme["items"], key=lambda item: item["sha256"])
        counts = {set_name: 0 for set_name in SETS}
        labels = {set_name: Counter() for set_name in SETS}
        for item in items:
            set_name = item["set"]
            counts[set_name] += 1
            label = item.get("label")
            label_key = "unlabeled" if label is None or label == "" else label
            labels[set_name][label_key] += 1
        return {
            "name": scheme["name"],
            "seed": scheme["seed"],
            "ratios": {
                set_name: scheme["ratios"][set_name] for set_name in SETS
            },
            "counts": counts,
            "labels": {
                set_name: dict(sorted(labels[set_name].items()))
                for set_name in SETS
            },
            "items": [
                {
                    "sha256": item["sha256"],
                    "label": item.get("label"),
                    "set": item["set"],
                }
                for item in items
            ],
        }
