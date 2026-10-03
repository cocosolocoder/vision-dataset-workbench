from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

from .batches import (
    BatchError,
    content_key,
    empty_history,
    normalize_label,
    parse_batch_file,
    render_label,
    validate_history,
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
SPLITS_DIRECTORY = "splits"
BATCHES_NAME = "batches.json"
LOCK_NAME = ".lock"
TRANSACTION_NAME = ".txn.json"
SPLIT_SCHEMA_VERSION = 1

# Image extensions accepted by whole-directory import, matched case-insensitively.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


@dataclass(frozen=True)
class ImportResult:
    digest: str
    added: bool


@dataclass(frozen=True)
class SplitPlanResult:
    name: str
    created: bool
    plan: dict[str, Any]


class DatasetStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.state_directory = self.root / STATE_DIRECTORY
        self.manifest_path = self.state_directory / MANIFEST_NAME
        self.splits_directory = self.state_directory / SPLITS_DIRECTORY
        self.batches_path = self.state_directory / BATCHES_NAME
        self.lock_path = self.state_directory / LOCK_NAME
        self.transaction_path = self.state_directory / TRANSACTION_NAME
        # A crash mid-commit can leave a prepared transaction; finish it
        # before any operation observes the workspace.
        self._recover_if_needed()

    def initialize(self) -> None:
        # The empty-manifest decision is made under the workspace lock so a
        # concurrent first import cannot have its sample replaced by an empty
        # dataset created from a stale "manifest missing" observation.
        with self._locked(create=True):
            if not self.manifest_path.exists():
                self._write_json_atomic(
                    self.manifest_path, {"schema_version": 1, "items": []}
                )

    def add(self, source: Path, label: str | None = None) -> ImportResult:
        self.initialize()
        file_path = source.resolve(strict=True)
        if not file_path.is_file():
            raise ValueError(f"Not a regular file: {source}")
        # Hashing is pure I/O on the source file; do it before taking the
        # workspace lock so concurrent imports of different files hash in
        # parallel and only serialize for the manifest check-and-commit.
        # The digest and the size must describe one stable file: the read
        # fails if the file's identity, size or modification time changes
        # around or during the read, so a source being rewritten or
        # replaced is rejected instead of registering a digest/size pair
        # that never coexisted.
        digest, size = self._read_stable(file_path, str(file_path))
        with self._locked(create=True):
            manifest = self._read()
            if any(item["sha256"] == digest for item in manifest["items"]):
                # Re-import never overwrites the registered source, label or
                # revision: the first report of a digest owns its metadata.
                return ImportResult(digest=digest, added=False)
            manifest["items"].append(
                {
                    "sha256": digest,
                    "source": str(file_path),
                    "size": size,
                    "label": label,
                }
            )
            manifest["items"].sort(key=lambda item: item["sha256"])
            # Commit through the same journal as batches so the import is
            # crash-safe and cannot clobber a batch committed concurrently.
            self._commit(manifest, self._read_history())
            return ImportResult(digest=digest, added=True)

    def import_directory(
        self, source: Path, label: str | None = None, recursive: bool = False
    ) -> dict[str, Any]:
        """Import every accepted image file under a source directory at once.

        Candidates are regular files (never symlinks or other entry types)
        whose lower-cased extension is one of :data:`IMAGE_EXTENSIONS`;
        recursion stays on the top level unless ``recursive`` is set and
        never descends through symlinked directories.  They are ordered by
        path relative to the source directory, with ``/`` separators,
        compared by Unicode code point.

        Every candidate is read in full and identified by its full
        SHA-256 digest.  A digest already in the workspace, or first seen
        on an earlier candidate in this same import, is reported as a
        duplicate: the earliest sorted candidate owns the registration and
        later paths never overwrite its source or label.  The supplied
        label is attached only to genuinely new samples; ``None`` and the
        empty string both mean unlabeled.

        The whole batch commits as one journal transaction.  The import
        fails entirely (no new samples) when the source is missing or not
        a directory, a directory that should be visited cannot be scanned,
        a candidate cannot be read in full, or a candidate's identity,
        size or modification time changes while it is being read.
        """
        self.initialize()
        source_path = Path(source)
        if not source_path.exists():
            raise ValueError(f"source directory does not exist: {source}")
        if not source_path.is_dir():
            raise ValueError(f"source is not a directory: {source}")
        root = source_path.resolve()

        rel_paths = self._scan_image_files(root, recursive)
        # Read (and re-stat) every candidate before taking the workspace
        # lock: I/O overlaps with peers, and a file that changes mid-read
        # fails the whole batch before anything is committed.
        candidates = [
            self._read_candidate(root / rel_path, rel_path) for rel_path in rel_paths
        ]

        with self._locked(create=True):
            manifest = self._read()
            history = self._read_history()
            known = {item["sha256"] for item in manifest["items"]}

            results: list[dict[str, Any]] = []
            new_items: list[dict[str, Any]] = []
            for rel_path, abs_path, digest, size in candidates:
                if digest in known:
                    status = "duplicate"
                else:
                    status = "added"
                    known.add(digest)
                    new_items.append(
                        {
                            "sha256": digest,
                            "source": str(abs_path.resolve()),
                            "size": size,
                            "label": label,
                        }
                    )
                results.append(
                    {"path": rel_path, "sha256": digest, "status": status}
                )

            if new_items:
                manifest["items"].extend(new_items)
                manifest["items"].sort(key=lambda item: item["sha256"])
                # One journal commit for the whole batch: an interruption
                # leaves either the old state or every new sample together.
                self._commit(manifest, history)

        added = sum(1 for result in results if result["status"] == "added")
        duplicates = len(results) - added
        return {
            "directory": str(root),
            "recursive": recursive,
            "label": label,
            "candidate_count": len(results),
            "added": added,
            "duplicates": duplicates,
            "candidates": results,
        }

    def _scan_image_files(self, root: Path, recursive: bool) -> list[str]:
        """Return accepted files as ``/``-separated paths relative to root.

        Only real regular files qualify; symlinks and other entry types are
        skipped, and recursion never follows symlinked directories.  Any
        directory that should be visited but cannot be scanned fails the
        whole import with the path and reason.
        """
        rel_paths: list[str] = []

        def walk(directory: Path) -> None:
            try:
                entries = list(os.scandir(directory))
            except OSError as error:
                raise ValueError(
                    f"cannot scan directory {directory}: {error}"
                ) from error
            for entry in entries:
                try:
                    is_file = entry.is_file(follow_symlinks=False)
                except OSError as error:
                    raise ValueError(
                        f"cannot inspect entry {entry.path}: {error}"
                    ) from error
                if is_file:
                    if Path(entry.name).suffix.lower() in IMAGE_EXTENSIONS:
                        rel_paths.append(
                            os.path.relpath(entry.path, root).replace(os.sep, "/")
                        )
                elif recursive:
                    try:
                        is_dir = entry.is_dir(follow_symlinks=False)
                    except OSError as error:
                        raise ValueError(
                            f"cannot inspect entry {entry.path}: {error}"
                        ) from error
                    if is_dir:
                        walk(Path(entry.path))

        walk(root)
        # Unicode code point order on the slash-separated relative path.
        rel_paths.sort()
        return rel_paths

    def _read_candidate(
        self, path: Path, rel_path: str
    ) -> tuple[str, Path, str, int]:
        """Read one candidate in full, verifying identity/size/mtime around it.

        Returns ``(rel_path, absolute_path, digest, size)``.  A mismatch or
        read failure raises before the batch is committed.
        """
        digest, size = self._read_stable(path, rel_path)
        return rel_path, path, digest, size

    @staticmethod
    def _read_stable(path: Path, display: str) -> tuple[str, int]:
        """Read one file in full and return ``(digest, size)`` for it.

        The digest and size are guaranteed to describe the same stable
        regular file: the path is lstat()ed before and after the read, the
        open uses ``O_NOFOLLOW`` so a path swapped for a symlink cannot
        redirect the read to another file, and the opened descriptor is
        fstat()ed to prove it is the inode that was confirmed before the
        read.  Any identity, size or modification-time change — in-place
        rewrites, appends, truncation, delete-and-recreate, or replacing
        the path with another file, even one with identical content, size
        and mtime — fails with the path and reason, as do a vanishing
        path, a non-regular file and any open or read error.
        """
        try:
            before = os.lstat(path)
        except OSError as error:
            raise ValueError(f"cannot read {display}: {error}") from error
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"cannot read {display}: not a regular file")
        try:
            # O_NOFOLLOW guarantees the bytes come from the inode just
            # lstat()ed, even if the path is swapped for a symlink meanwhile.
            descriptor = os.open(
                path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
        except OSError as error:
            raise ValueError(f"cannot read {display}: {error}") from error
        try:
            opened = os.fstat(descriptor)
        except OSError as error:
            os.close(descriptor)
            raise ValueError(f"cannot read {display}: {error}") from error
        # The path may have been replaced by another regular file between
        # the lstat() and the open(); the descriptor must belong to the
        # inode that was confirmed, or the bytes read would not come from
        # the file this import agreed to read.
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            os.close(descriptor)
            raise ValueError(
                f"file changed during import: {display} "
                "(identity, size or modification time changed)"
            )
        hasher = hashlib.sha256()
        try:
            with os.fdopen(descriptor, "rb") as stream:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    hasher.update(chunk)
        except OSError as error:
            raise ValueError(f"cannot read {display}: {error}") from error
        try:
            after = os.lstat(path)
        except OSError as error:
            raise ValueError(f"cannot read {display}: {error}") from error
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError(
                f"file changed during import: {display} "
                "(identity, size or modification time changed)"
            )
        return hasher.hexdigest(), before.st_size

    def summary(self) -> dict[str, Any]:
        self.initialize()
        with self._locked():
            items = self._read()["items"]
        labels = Counter(item["label"] or "unlabeled" for item in items)
        return {
            "items": len(items),
            "bytes": sum(item["size"] for item in items),
            "labels": dict(sorted(labels.items())),
        }

    # ------------------------------------------------------------------
    # Batch label updates
    # ------------------------------------------------------------------

    def lookup_label(self, digest: str) -> dict[str, Any]:
        """Return ``{"sha256": digest, "label": str | None}`` for a sample.

        ``None`` means unlabeled (the manifest may store ``null`` or ``""``).
        """
        with self._locked():
            manifest = self._read()
            for item in manifest["items"]:
                if item["sha256"] == digest:
                    return {"sha256": digest, "label": normalize_label(item.get("label"))}
        raise BatchError(f"sample not found: {digest}")

    def submit_batch_file(self, path: Path) -> dict[str, Any]:
        """Submit a batch from a UTF-8 JSON file."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BatchError(f"cannot read batch file: {error}") from error
        return self.submit_batch(data)

    def submit_batch(self, data: Any) -> dict[str, Any]:
        """Validate and apply a batch label update.

        The whole batch is verified against the current manifest before any
        change is written: missing digests, duplicate records, bad label
        types and old-label mismatches reject the batch with no changes.
        A successful batch is recorded in history; re-submitting the same
        number with the same content returns the original result without
        touching anything.
        """
        number, records = parse_batch_file(data)
        with self._locked(create=True):
            manifest = self._read()
            history = self._read_history()
            items = {item["sha256"]: item for item in manifest["items"]}

            for record in records:
                if record["sha256"] not in items:
                    raise BatchError(f"sample not found: {record['sha256']}")

            existing = next(
                (entry for entry in history["batches"] if entry["batch"] == number),
                None,
            )
            if existing is not None:
                if content_key(existing["records"]) == content_key(records):
                    return self._replay_result(existing)
                raise BatchError(
                    f"batch number {number!r} is already used with different content"
                )

            for record in records:
                current = normalize_label(items[record["sha256"]].get("label"))
                if current != record["old"]:
                    raise BatchError(
                        f"change for {record['sha256']}: expected old label "
                        f"{render_label(record['old'])!r} but current label is "
                        f"{render_label(current)!r}"
                    )

            applied_records: list[dict[str, Any]] = []
            changed_count = 0
            for record in records:
                item = items[record["sha256"]]
                current = normalize_label(item.get("label"))
                changed = current != record["new"]
                revision = self._revision(item)
                if changed:
                    item["label"] = record["new"]
                    revision += 1
                    item["rev"] = revision
                    changed_count += 1
                applied_records.append(
                    {**record, "changed": changed, "rev": revision}
                )

            if changed_count == 0:
                # No-op batches occupy no number and write no history.
                return {
                    "batch": number,
                    "status": "no-changes",
                    "changed": 0,
                    "unchanged": len(records),
                    "total": len(records),
                }

            entry = {
                "batch": number,
                "records": applied_records,
                "changed_count": changed_count,
                "undone": False,
                "undone_at": None,
            }
            history["batches"].append(entry)
            self._commit(manifest, history)
            return {
                "batch": number,
                "status": "applied",
                "changed": changed_count,
                "unchanged": len(records) - changed_count,
                "total": len(records),
            }

    def undo_batch(self, number: str) -> dict[str, Any]:
        """Undo a successful batch by number.

        Only samples the batch actually changed are checked: each must have
        the same label revision it had right after the batch (no later
        modification, even one that restored the same label).  Repeated
        undos are idempotent and do not add history.
        """
        with self._locked(create=True):
            manifest = self._read()
            history = self._read_history()
            entry = next(
                (entry for entry in history["batches"] if entry["batch"] == number),
                None,
            )
            if entry is None:
                raise BatchError(f"unknown batch number: {number!r}")
            if entry["undone"]:
                return {"batch": number, "status": "already-undone", "restored": 0}

            items = {item["sha256"]: item for item in manifest["items"]}
            for record in entry["records"]:
                if not record["changed"]:
                    continue
                item = items.get(record["sha256"])
                if item is None:
                    raise BatchError(
                        f"cannot undo batch {number!r}: sample {record['sha256']} "
                        "is no longer in the manifest"
                    )
                current = normalize_label(item.get("label"))
                if self._revision(item) != record["rev"] or current != record["new"]:
                    raise BatchError(
                        f"cannot undo batch {number!r}: sample {record['sha256']} "
                        "was modified after the batch"
                    )

            restored = 0
            for record in entry["records"]:
                if not record["changed"]:
                    continue
                item = items[record["sha256"]]
                item["label"] = record["old"]
                item["rev"] = self._revision(item) + 1
                restored += 1

            entry["undone"] = True
            entry["undone_at"] = datetime.now(timezone.utc).isoformat()
            self._commit(manifest, history)
            return {"batch": number, "status": "undone", "restored": restored}

    def history(self) -> list[dict[str, Any]]:
        """Return successful batch entries in submission order."""
        with self._locked():
            return self._read_history()["batches"]

    @staticmethod
    def _replay_result(entry: dict[str, Any]) -> dict[str, Any]:
        changed = sum(1 for record in entry["records"] if record["changed"])
        return {
            "batch": entry["batch"],
            "status": "already-applied",
            "changed": changed,
            "unchanged": len(entry["records"]) - changed,
            "total": len(entry["records"]),
        }

    @staticmethod
    def _revision(item: dict[str, Any]) -> int:
        value = item.get("rev", 0)
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    # ------------------------------------------------------------------
    # Workspace locking and transactional commits
    # ------------------------------------------------------------------

    @contextmanager
    def _workspace_lock(self):
        """Exclusive workspace lock for check-and-write transactions.

        ``flock`` is advisory but sufficient for local processes: holders
        serialize, so every read-modify-write observes the state left by the
        previous completed operation.
        """
        self.state_directory.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    @contextmanager
    def _locked(self, *, create: bool = False):
        """Hold the workspace lock and finish any leftover prepared journal.

        Recovery runs once the lock is held, so a long-lived process that
        only imports, queries or creates plans still installs the prepared
        transaction of a process that crashed on this machine, without
        needing to reopen the workspace.

        Read callers leave ``create`` false: on a workspace whose state
        directory does not yet exist there is no journal to finish and
        nothing for a writer to hold, so the query proceeds (and fails or
        reports empty exactly as before) without materializing one.
        """
        if not create and not self.state_directory.exists():
            yield
            return
        with self._workspace_lock():
            self._recover_transaction_locked()
            yield

    def _recover_if_needed(self) -> None:
        """Finish a prepared transaction left by a crashed process."""
        if not self.transaction_path.exists():
            return
        with self._locked():
            pass

    def _recover_transaction_locked(self) -> None:
        if not self.transaction_path.exists():
            return
        try:
            journal = json.loads(self.transaction_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BatchError(f"cannot recover interrupted transaction: {error}") from error
        manifest = journal.get("manifest")
        history = journal.get("batches")
        if not isinstance(manifest, dict) or not isinstance(history, dict):
            raise BatchError("cannot recover interrupted transaction: journal is malformed")
        # Only install a coherent pair, so recovery can never leave the
        # manifest and the history in two different states.
        if manifest.get("schema_version") != 1 or not isinstance(
            manifest.get("items"), list
        ):
            raise BatchError(
                "cannot recover interrupted transaction: journal manifest is malformed"
            )
        validate_history(history)
        # Idempotent installs: both files end up at the journal's version.
        self._write_json_atomic(self.manifest_path, manifest)
        self._write_json_atomic(self.batches_path, history)
        try:
            self.transaction_path.unlink()
        except OSError:
            pass

    def _commit(self, manifest: dict[str, Any], history: dict[str, Any]) -> None:
        """Atomically install the new manifest and history together.

        A write-ahead journal holding both full payloads is written and
        fsynced first; a crash at any point leaves enough to redo the
        missing install on the next open, so the two files never disagree.
        """
        journal = {"manifest": manifest, "batches": history}
        self._write_json_atomic(self.transaction_path, journal)
        self._write_json_atomic(self.manifest_path, manifest)
        self._write_json_atomic(self.batches_path, history)
        try:
            self.transaction_path.unlink()
        except OSError:
            pass

    def _read_history(self) -> dict[str, Any]:
        if not self.batches_path.exists():
            return empty_history()
        try:
            data = json.loads(self.batches_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BatchError(f"cannot read batch history: {error}") from error
        return validate_history(data)

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
        plan_path = self._split_path(name)

        with self._locked(create=True):
            # Sample identities and labels are read from one complete
            # manifest state, so a batch committing concurrently is observed
            # either entirely (all its label changes) or not at all.
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
            plan_payload = self._build_plan(
                name, seed, fraction_ratios, snapshot, assignment
            )

            # The check-and-create is atomic under the same lock, so among
            # concurrent creations of one name exactly one reports success;
            # matching requests reuse it and differing ones conflict.
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
        if data.get("schema_version") != 1 or not isinstance(data.get("items"), list):
            raise ValueError("Unsupported dataset manifest")
        return data
