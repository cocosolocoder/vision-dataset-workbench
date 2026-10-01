from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator

import fcntl

from .batches import (
    LABEL_SCHEMA_VERSION,
    BatchError,
    content_fingerprint,
    labels_equal,
    normalize_label,
    parse_batch_document,
)
from .splits import (
    SET_NAMES,
    SplitError,
    assign,
    category_key,
    category_name,
    validate_name,
    validate_ratios,
)


STATE_DIRECTORY = ".vision-workbench"
MANIFEST_NAME = "manifest.json"
LOCK_NAME = ".lock"
SPLITS_DIRECTORY = "splits"
SPLIT_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = LABEL_SCHEMA_VERSION


@dataclass(frozen=True)
class ImportResult:
    digest: str
    added: bool


@dataclass(frozen=True)
class SplitPlanResult:
    name: str
    created: bool
    plan: dict[str, Any]


@dataclass(frozen=True)
class BatchApplyResult:
    batch_id: str
    status: str  # "applied" | "unchanged" | "duplicate"
    changed: int
    total: int
    changes: tuple[dict[str, Any], ...]
    reused: bool = False


@dataclass(frozen=True)
class BatchUndoResult:
    batch_id: str
    status: str  # "undone" | "already_undone"
    restored: int


class DatasetStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.state_directory = self.root / STATE_DIRECTORY
        self.manifest_path = self.state_directory / MANIFEST_NAME
        self.lock_path = self.state_directory / LOCK_NAME
        self.splits_directory = self.state_directory / SPLITS_DIRECTORY

    def initialize(self) -> None:
        self.state_directory.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            self._write(
                {
                    "schema_version": MANIFEST_SCHEMA_VERSION,
                    "items": [],
                    "label_batches": [],
                }
            )

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Hold an exclusive cross-process lock for a read-modify-write.

        A plain lock file inside the state directory serializes two local
        processes; combined with atomic manifest replacement, neither
        successful batch can be lost and conflicting batches cannot both
        commit.
        """
        self.state_directory.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)

    def add(self, source: Path, label: str | None = None) -> ImportResult:
        self.initialize()
        file_path = source.resolve(strict=True)
        if not file_path.is_file():
            raise ValueError(f"Not a regular file: {source}")
        digest = self._digest(file_path)
        with self._locked():
            manifest = self._read()
            if any(item["sha256"] == digest for item in manifest["items"]):
                return ImportResult(digest=digest, added=False)
            manifest["items"].append(
                {
                    "sha256": digest,
                    "source": str(file_path),
                    "size": file_path.stat().st_size,
                    "label": normalize_label(label),
                }
            )
            manifest["items"].sort(key=lambda item: item["sha256"])
            self._write(manifest)
        return ImportResult(digest=digest, added=True)

    def summary(self) -> dict[str, Any]:
        self.initialize()
        items = self._read()["items"]
        labels = Counter(item["label"] or "unlabeled" for item in items)
        return {
            "items": len(items),
            "bytes": sum(item["size"] for item in items),
            "labels": dict(sorted(labels.items())),
        }

    def get_label(self, digest: str) -> str | None:
        """Return the current label of one sample (``None`` if unlabeled)."""
        self.initialize()
        items = self._read()["items"]
        for item in items:
            if item["sha256"] == digest:
                return normalize_label(item.get("label"))
        raise BatchError(
            f"Sample not found: {digest}",
            code="unknown_digest",
            records=[{"sha256": digest}],
        )

    # ------------------------------------------------------------------
    # Batch label changes
    # ------------------------------------------------------------------

    def apply_batch(self, payload: Any) -> BatchApplyResult:
        """Validate and apply one label-change batch, all or nothing.

        Structural validation runs before any state is touched; semantic
        checks (existing digests, expected labels) and the commit run under
        an exclusive lock so two local processes serialize.  A batch id
        that has already succeeded replays its original result when the
        content matches (order and null/"" spelling do not matter), and
        conflicts otherwise.
        """
        batch_id, changes = parse_batch_document(payload)
        self.initialize()
        with self._locked():
            manifest = self._read()
            history = manifest.setdefault("label_batches", [])

            previous = self._find_history(history, batch_id)
            if previous is not None:
                return self._replay_previous(previous, changes)

            items_by_digest = {item["sha256"]: item for item in manifest["items"]}
            for index, change in enumerate(changes):
                location = {"index": index, "sha256": change["sha256"]}
                item = items_by_digest.get(change["sha256"])
                if item is None:
                    raise BatchError(
                        f"Sample not found: {change['sha256']}",
                        code="unknown_digest",
                        records=[location],
                    )
                if not labels_equal(item.get("label"), change["old_label"]):
                    raise BatchError(
                        "Current label does not match expected old label for "
                        f"{change['sha256']}: found "
                        f"{normalize_label(item.get('label'))!r}, expected "
                        f"{change['old_label']!r}",
                        code="label_mismatch",
                        records=[location],
                    )

            applied_changes: list[dict[str, Any]] = []
            changed_digests: list[str] = []
            for change in changes:
                item = items_by_digest[change["sha256"]]
                current = normalize_label(item.get("label"))
                target = change["new_label"]
                applied_changes.append(
                    {
                        "sha256": change["sha256"],
                        "old_label": current,
                        "new_label": target,
                        "changed": current != target,
                    }
                )
                if current != target:
                    changed_digests.append(change["sha256"])

            if not changed_digests:
                # No-op batches are reported but consume neither history
                # nor the batch id.
                return BatchApplyResult(
                    batch_id=batch_id,
                    status="unchanged",
                    changed=0,
                    total=len(changes),
                    changes=tuple(applied_changes),
                )

            for change in changes:
                item = items_by_digest[change["sha256"]]
                target = change["new_label"]
                if normalize_label(item.get("label")) != target:
                    item["label"] = target
                    # Stamp every actually changed sample: only a later
                    # label touch clears this, which is what gates undo.
                    item["labels_last_batch"] = batch_id

            record = {
                "batch_id": batch_id,
                "fingerprint": content_fingerprint(changes),
                "changed_count": len(changed_digests),
                "changes": applied_changes,
                "undo": {"status": "active", "restored": None},
            }
            history.append(record)
            self._write(manifest)
            return BatchApplyResult(
                batch_id=batch_id,
                status="applied",
                changed=len(changed_digests),
                total=len(changes),
                changes=tuple(applied_changes),
            )

    def undo_batch(self, batch_id: str) -> BatchUndoResult:
        """Restore the labels of one successful batch.

        Succeeds only when every sample the batch really changed is still
        stamped with this batch id, i.e. none of them was modified
        afterwards (even if it was changed back to the same label).  Later
        changes to other samples do not block the undo.  Repeating an
        undo reports the prior undo without writing again; an unknown id
        is an explicit error.
        """
        if not isinstance(batch_id, str) or not batch_id.strip():
            raise BatchError(
                "Batch id must be a non-empty string", code="invalid_batch_id"
            )
        self.initialize()
        with self._locked():
            manifest = self._read()
            history = manifest.setdefault("label_batches", [])
            record = self._find_history(history, batch_id)
            if record is None:
                raise BatchError(
                    f"Unknown batch id: {batch_id}", code="unknown_batch"
                )
            undo = record.setdefault("undo", {})
            if undo.get("status") == "undone":
                return BatchUndoResult(
                    batch_id=batch_id,
                    status="already_undone",
                    restored=0,
                )

            items_by_digest = {item["sha256"]: item for item in manifest["items"]}
            touched_after: list[dict[str, Any]] = []
            for change in record["changes"]:
                if not change["changed"]:
                    continue
                digest = change["sha256"]
                item = items_by_digest.get(digest)
                if item is None or item.get("labels_last_batch") != batch_id:
                    touched_after.append({"sha256": digest})
            if touched_after:
                raise BatchError(
                    "Cannot undo batch: at least one changed sample was "
                    "modified afterwards",
                    code="sample_modified_after_batch",
                    records=touched_after,
                )

            for change in record["changes"]:
                if not change["changed"]:
                    continue
                items_by_digest[change["sha256"]]["label"] = change["old_label"]
                # The restoration itself is a later touch for undo-gating
                # purposes; re-undo returns "already_undone".
                items_by_digest[change["sha256"]]["labels_last_batch"] = None
            undo["status"] = "undone"
            undo["restored"] = record["changed_count"]
            self._write(manifest)
            return BatchUndoResult(
                batch_id=batch_id,
                status="undone",
                restored=record["changed_count"],
            )

    def batch_history(self) -> list[dict[str, Any]]:
        """Successful batches in submission order, undo state included."""
        self.initialize()
        history = self._read().setdefault("label_batches", [])
        return [
            {
                "batch_id": record["batch_id"],
                "changed_count": record["changed_count"],
                "changes": [dict(change) for change in record["changes"]],
                "undo": dict(record.get("undo", {"status": "active"})),
            }
            for record in history
        ]

    @staticmethod
    def _find_history(
        history: list[dict[str, Any]], batch_id: str
    ) -> dict[str, Any] | None:
        for record in history:
            if record["batch_id"] == batch_id:
                return record
        return None

    def _replay_previous(
        self, record: dict[str, Any], changes: list[dict[str, Any]]
    ) -> BatchApplyResult:
        """Return the original result for a resubmitted successful batch."""
        if record.get("fingerprint") != content_fingerprint(changes):
            raise BatchError(
                f"Batch id {record['batch_id']!r} was already used with "
                "different changes",
                code="batch_id_conflict",
            )
        return BatchApplyResult(
            batch_id=record["batch_id"],
            status="duplicate",
            changed=record["changed_count"],
            total=len(record["changes"]),
            changes=tuple(dict(change) for change in record["changes"]),
            reused=True,
        )

    # ------------------------------------------------------------------
    # Split plans
    # ------------------------------------------------------------------

    def create_split(
        self, name: str, seed: int, ratios: list[float | int | str | Fraction]
    ) -> SplitPlanResult:
        """Create (or reuse) a named train/validation/test plan.

        The plan snapshots every sample currently in the manifest.  A
        second call with the same name returns the stored plan unchanged
        when the inputs match; otherwise it reports a name conflict.
        """
        validate_name(name)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise SplitError(f"Seed must be an integer, got {seed!r}")
        fraction_ratios = validate_ratios(ratios)

        self.initialize()
        items = self._read()["items"]

        # assign() validates identities and performs the stratification.
        assignment = assign(items, fraction_ratios, seed)

        snapshot = [
            {
                "sha256": item["sha256"],
                "label": item.get("label"),
                "source": item.get("source"),
            }
            for item in sorted(items, key=lambda item: item["sha256"])
        ]
        plan_payload = self._build_plan(name, seed, fraction_ratios, snapshot, assignment)

        plan_path = self._split_path(name)
        if plan_path.exists():
            existing = self._read_split(plan_path)
            if self._same_inputs(existing, seed, fraction_ratios, snapshot):
                return SplitPlanResult(name=name, created=False, plan=existing)
            raise SplitError(
                f"Split plan {name!r} already exists with different seed, "
                "ratios or samples; choose another name"
            )

        self._write_json_atomic(plan_path, plan_payload)
        return SplitPlanResult(name=name, created=True, plan=plan_payload)

    def get_split(self, name: str) -> dict[str, Any]:
        """Return the stored plan payload, or raise if it does not exist."""
        validate_name(name)
        plan_path = self._split_path(name)
        if not plan_path.exists():
            raise SplitError(f"Split plan not found: {name}")
        return self._read_split(plan_path)

    def _split_path(self, name: str) -> Path:
        # validate_name() has already ruled out path separators and "..".
        return self.splits_directory / f"{name}.json"

    def _build_plan(
        self,
        name: str,
        seed: int,
        ratios: tuple[Fraction, Fraction, Fraction],
        snapshot: list[dict[str, Any]],
        assignment: dict[str, str],
    ) -> dict[str, Any]:
        sets = {set_name: [] for set_name in SET_NAMES}
        for sample in snapshot:
            set_name = assignment[sample["sha256"]]
            sets[set_name].append(
                {
                    "sha256": sample["sha256"],
                    "label": category_name(category_key(sample["label"])),
                    "source": sample["source"],
                }
            )

        set_summaries: dict[str, Any] = {}
        for set_name in SET_NAMES:
            members = sets[set_name]
            distribution = Counter(member["label"] for member in members)
            set_summaries[set_name] = {
                "samples": len(members),
                "distribution": dict(sorted(distribution.items())),
                "members": [
                    {
                        "sha256": member["sha256"],
                        "label": member["label"],
                        "source": member["source"],
                    }
                    for member in members
                ],
            }

        overall = Counter(
            category_name(category_key(sample["label"])) for sample in snapshot
        )
        return {
            "schema_version": SPLIT_SCHEMA_VERSION,
            "name": name,
            "seed": seed,
            "ratios": {
                set_name: str(ratios[i]) for i, set_name in enumerate(SET_NAMES)
            },
            "samples": {
                "total": len(snapshot),
                "distribution": dict(sorted(overall.items())),
            },
            "sets": set_summaries,
        }

    @staticmethod
    def _same_inputs(
        existing: dict[str, Any],
        seed: int,
        ratios: tuple[Fraction, Fraction, Fraction],
        snapshot: list[dict[str, Any]],
    ) -> bool:
        """Whether a stored plan was created from the exact same inputs."""
        if existing.get("seed") != seed:
            return False
        stored_ratios = existing.get("ratios")
        if not isinstance(stored_ratios, dict):
            return False
        for i, set_name in enumerate(SET_NAMES):
            stored = stored_ratios.get(set_name)
            if not isinstance(stored, str) or Fraction(stored) != ratios[i]:
                return False

        # Reuse requires the exact same sample identities with the exact
        # same creation-time labels; recorded sources are informational and
        # do not participate in the identity (the snapshot itself is the
        # decision input, per the feature contract).
        current = sorted(
            (sample["sha256"], category_name(category_key(sample.get("label"))))
            for sample in snapshot
        )
        stored: list[tuple[str, str]] = []
        sets = existing["sets"]
        for set_name in SET_NAMES:
            for member in sets[set_name]["members"]:
                stored.append((member["sha256"], member.get("label")))
        return sorted(stored) == current

    def _read_split(self, plan_path: Path) -> dict[str, Any]:
        try:
            data = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SplitError(f"Cannot read split plan: {error}") from error
        self._validate_split_payload(data)
        return data

    @staticmethod
    def _validate_split_payload(data: Any) -> None:
        if not isinstance(data, dict):
            raise SplitError("Split plan record is malformed")
        if data.get("schema_version") != SPLIT_SCHEMA_VERSION:
            raise SplitError("Unsupported split plan schema version")
        if not isinstance(data.get("name"), str):
            raise SplitError("Split plan record is missing its name")
        if not isinstance(data.get("seed"), int) or isinstance(data.get("seed"), bool):
            raise SplitError("Split plan record is missing its seed")
        ratios = data.get("ratios")
        if not isinstance(ratios, dict):
            raise SplitError("Split plan record is missing its ratios")
        ratio_values = []
        for set_name in SET_NAMES:
            value = ratios.get(set_name)
            if not isinstance(value, str):
                raise SplitError(f"Split plan record is missing ratio for {set_name}")
            ratio_values.append(value)
        try:
            validate_ratios(ratio_values)
        except SplitError as error:
            raise SplitError(f"Split plan record has invalid ratios: {error}") from error

        samples = data.get("samples")
        if (
            not isinstance(samples, dict)
            or not isinstance(samples.get("total"), int)
            or isinstance(samples.get("total"), bool)
        ):
            raise SplitError("Split plan record is missing its sample summary")
        if not isinstance(samples.get("distribution"), dict):
            raise SplitError("Split plan record is missing its class distribution")
        sets = data.get("sets")
        if not isinstance(sets, dict):
            raise SplitError("Split plan record is missing its set assignments")

        seen: set[str] = set()
        overall: Counter[str] = Counter()
        for set_name in SET_NAMES:
            set_payload = sets.get(set_name)
            if not isinstance(set_payload, dict):
                raise SplitError(f"Split plan record is missing set {set_name}")
            members = set_payload.get("members")
            if not isinstance(members, list):
                raise SplitError(f"Split plan record is missing members for {set_name}")
            if set_payload.get("samples") != len(members):
                raise SplitError(f"Split plan record has a bad count for {set_name}")
            distribution = Counter()
            for member in members:
                if (
                    not isinstance(member, dict)
                    or not isinstance(member.get("sha256"), str)
                    or not isinstance(member.get("label"), str)
                    or not isinstance(member.get("source"), str)
                ):
                    raise SplitError(f"Split plan record has a bad member in {set_name}")
                digest = member["sha256"]
                if digest in seen:
                    raise SplitError(
                        f"Split plan record lists {digest} in more than one set"
                    )
                seen.add(digest)
                distribution[member["label"]] += 1
                overall[member["label"]] += 1
            if dict(sorted(distribution.items())) != {
                key: value for key, value in sorted(set_payload.get("distribution", {}).items())
            }:
                raise SplitError(
                    f"Split plan record has a mismatched distribution for {set_name}"
                )
        if samples["total"] != len(seen):
            raise SplitError("Split plan record has a mismatched sample total")
        if dict(sorted(overall.items())) != {
            key: value for key, value in sorted(samples["distribution"].items())
        }:
            raise SplitError("Split plan record has a mismatched class distribution")

    def _write_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        """Write via a temp file + fsync + atomic replace.

        A reader therefore observes either no plan or the complete plan,
        never a partial one.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            self._fsync_directory(path.parent)
        except OSError:
            temporary.unlink(missing_ok=True)
            raise

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        try:
            descriptor = os.open(directory, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot read dataset manifest: {error}") from error
        if (
            not isinstance(data, dict)
            or data.get("schema_version") not in (1, MANIFEST_SCHEMA_VERSION)
            or not isinstance(data.get("items"), list)
        ):
            raise ValueError("Unsupported dataset manifest")
        # Older workspaces predate label-batch history; they upgrade on
        # the next write without requiring a re-import.
        data.setdefault("label_batches", [])
        if not isinstance(data["label_batches"], list):
            raise ValueError("Unsupported dataset manifest")
        return data

    def _write(self, data: dict[str, Any]) -> None:
        data["schema_version"] = MANIFEST_SCHEMA_VERSION
        self._write_json_atomic(self.manifest_path, data)

    @staticmethod
    def _digest(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
