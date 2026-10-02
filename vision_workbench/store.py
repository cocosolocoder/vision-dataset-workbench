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

# Ordinary image files accepted by a directory import; compared case
# insensitively against the final path suffix.
IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
)
# Identity fields checked before and after reading so a file that is
# replaced, truncated or touched mid-scan fails the whole import.
_FILE_IDENTITY_FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns")


@dataclass(frozen=True)
class ImportResult:
    digest: str
    added: bool


@dataclass(frozen=True)
class DirectoryCandidate:
    """One scanned image file: its source path and ``/``-separated relative
    path as presented in the import result."""

    path: Path
    relative: str


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
        digest = self._digest(file_path)
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
                    "size": file_path.stat().st_size,
                    "label": label,
                }
            )
            manifest["items"].sort(key=lambda item: item["sha256"])
            # Commit through the same journal as batches so the import is
            # crash-safe and cannot clobber a batch committed concurrently.
            self._commit(manifest, self._read_history())
            return ImportResult(digest=digest, added=True)

    def import_directory(
        self,
        source: Path,
        label: str | None = None,
        *,
        recursive: bool = False,
    ) -> dict[str, Any]:
        """Register every image file in ``source`` in one atomic batch.

        Only regular, non-symlink entries with an image extension
        (``.jpg``/``.jpeg``/``.png``/``.bmp``/``.gif``/``.webp``, case
        insensitive) are candidates; symlink directories are never entered
        and recursion is opt-in. Candidates are sorted by their ``/``-joined
        path relative to ``source`` using Unicode code points, so the order
        is stable and independent of the local directory order.

        Files appearing after the scan completes are left for a later
        import. Several candidates with identical content register once,
        sourced from the earliest candidate; every other candidate reports
        as a duplicate, as do candidates whose digest is already in the
        workspace. The given label applies only to genuinely new samples;
        it never overrides an existing record. ``None`` and ``""`` both
        mean unlabeled.

        The new samples commit together through the write-ahead journal:
        a missing/non-directory source, an unscannable directory, an
        unreadable candidate or a file whose identity, size or modification
        time changes while being read fails the whole import without
        touching the manifest or history.
        """
        normalized_label = normalize_label(label)
        source_root = Path(source)
        self._validate_source_directory(source_root)
        candidates = self._scan_candidates(source_root, recursive=recursive)

        # Hashing is pure I/O on the source tree; do it before taking the
        # workspace lock, exactly like a single-file add, so concurrent
        # imports hash in parallel. Reading each candidate is bracketed by
        # identity/size/mtime checks, so a changed file fails the batch
        # before anything could be committed.
        scanned: list[dict[str, Any]] = []
        for candidate in candidates:
            digest, size, file_path = self._read_candidate(candidate)
            scanned.append(
                {
                    "relative": candidate.relative,
                    "path": file_path,
                    "digest": digest,
                    "size": size,
                }
            )

        self.initialize()
        with self._locked(create=True):
            # Duplicate decisions are made against the workspace state at
            # the moment the batch takes effect, so concurrent commits are
            # never clobbered and never double-registered.
            manifest = self._read()
            known = {item["sha256"] for item in manifest["items"]}

            outcomes: list[dict[str, Any]] = []
            chosen: dict[str, dict[str, Any]] = {}
            batch_digests: set[str] = set()
            for entry in scanned:
                digest = entry["digest"]
                if digest in known or digest in batch_digests:
                    status = "duplicate"
                else:
                    status = "added"
                    batch_digests.add(digest)
                    chosen[digest] = entry
                outcomes.append(
                    {
                        "path": entry["relative"],
                        "sha256": digest,
                        "status": status,
                    }
                )

            for digest in sorted(chosen):
                entry = chosen[digest]
                manifest["items"].append(
                    {
                        "sha256": digest,
                        "source": str(entry["path"]),
                        "size": entry["size"],
                        "label": normalized_label,
                    }
                )
            manifest["items"].sort(key=lambda item: item["sha256"])

            # One journal commit installs the whole set together; on
            # interruption recovery shows either the old manifest or the
            # complete new one, never a partial import.
            if chosen:
                self._commit(manifest, self._read_history())

        added = sum(1 for outcome in outcomes if outcome["status"] == "added")
        return {
            "candidates": len(outcomes),
            "added": added,
            "duplicates": len(outcomes) - added,
            "results": outcomes,
        }

    # Backwards-friendly alias mirroring the single-file ``add`` name.
    def add_directory(
        self,
        source: Path,
        label: str | None = None,
        *,
        recursive: bool = False,
    ) -> dict[str, Any]:
        return self.import_directory(source, label, recursive=recursive)

    @staticmethod
    def _validate_source_directory(source: Path) -> None:
        try:
            stat_result = source.stat()
        except OSError as error:
            raise ValueError(f"source directory cannot be accessed: {source}: {error}") from error
        if not stat.S_ISDIR(stat_result.st_mode):
            raise ValueError(f"source is not a directory: {source}")

    @staticmethod
    def _scan_candidates(source: Path, *, recursive: bool) -> list[DirectoryCandidate]:
        """Return all importable files under ``source`` in display order.

        Only the source layer is read by default; ``recursive`` also
        descends into real subdirectories but never into symlinked ones.
        Symlinks and other non-regular entries are skipped, including
        image-named symlinks.
        """
        found: list[DirectoryCandidate] = []

        def visit(directory: Path, prefix: str) -> None:
            try:
                with os.scandir(directory) as entries:
                    current = list(entries)
            except OSError as error:
                raise ValueError(
                    f"source directory cannot be scanned: {directory}: {error}"
                ) from error
            for entry in current:
                relative = (
                    entry.name if not prefix else f"{prefix}/{entry.name}"
                )
                try:
                    is_symlink = entry.is_symlink()
                except OSError as error:
                    raise ValueError(
                        f"source entry cannot be inspected: {directory / entry.name}: {error}"
                    ) from error
                if is_symlink:
                    continue
                try:
                    is_directory = entry.is_dir(follow_symlinks=False)
                    is_file = entry.is_file(follow_symlinks=False)
                except OSError as error:
                    raise ValueError(
                        f"source entry cannot be inspected: {directory / entry.name}: {error}"
                    ) from error
                if is_directory:
                    if recursive:
                        visit(Path(entry.path), relative)
                    continue
                if is_file and Path(entry.name).suffix.lower() in IMAGE_EXTENSIONS:
                    found.append(
                        DirectoryCandidate(
                            path=Path(entry.path), relative=relative
                        )
                    )

        visit(source, "")
        found.sort(key=lambda candidate: candidate.relative)
        return found

    @staticmethod
    def _read_candidate(candidate: DirectoryCandidate) -> tuple[str, int, Path]:
        """Hash one candidate, failing unless it stays identical across the read.

        Returns ``(digest, size, resolved path)``. A non-regular entry, an
        unreadable file, or any change in device/inode/size/mtime between
        the pre-read and post-read stat raises, which aborts the whole
        import.
        """
        path = candidate.path
        try:
            # lstat (not stat): a regular file swapped for a symlink while
            # the batch runs must abort rather than silently read through
            # the link.
            before = path.lstat()
        except OSError as error:
            raise ValueError(
                f"candidate cannot be read: {candidate.relative}: {error}"
            ) from error
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(
                f"candidate is not a regular file: {candidate.relative}"
            )

        hasher = hashlib.sha256()
        try:
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
        except OSError as error:
            raise ValueError(
                f"candidate cannot be read: {candidate.relative}: {error}"
            ) from error

        try:
            after = path.lstat()
        except OSError as error:
            raise ValueError(
                f"candidate changed during import: {candidate.relative}: {error}"
            ) from error
        changed = tuple(getattr(after, field) for field in _FILE_IDENTITY_FIELDS)
        expected = tuple(getattr(before, field) for field in _FILE_IDENTITY_FIELDS)
        if changed != expected:
            raise ValueError(
                f"candidate changed during import: {candidate.relative}: "
                "file identity, size or modification time differs"
            )
        return hasher.hexdigest(), before.st_size, path.resolve()

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

    @staticmethod
    def _digest(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
