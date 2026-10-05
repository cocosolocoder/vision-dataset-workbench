from __future__ import annotations

import errno
import fcntl
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

from . import confirmed
from .manifest import ManifestError, validate_manifest
from .batches import (
    BatchError,
    apply_resolutions,
    build_history_entry,
    empty_history,
    item_revision,
    no_changes_result,
    normalize_label,
    parse_batch_file,
    reject_unknown_samples,
    require_intact_history_changed_flags,
    require_intact_history_records,
    require_intact_history_revisions,
    resolve_re_submission,
    result_payload,
    validate_history,
    verify_records,
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

# Flags for pinning a directory before scanning or reading through it:
# O_DIRECTORY so only a real directory opens, O_NOFOLLOW so a symbolic
# link swapped onto the path is refused instead of followed.
_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


def _import_confirmation_error(
    failure: confirmed.ConfirmationError, display: str
) -> ValueError:
    """Translate a shared confirmation failure into import's existing error.

    Import reports every non-replacement problem as ``cannot read
    <path>: …`` — a non-regular object at the pre-open inspection
    (including a symlink swapped onto the path by then) is “not a
    regular file”, and a refused no-follow open keeps its underlying
    ``OSError`` wording — while an object that replaced the confirmed
    file between its inspection and the pinned open is “file changed
    during import …”.  The shared rule supplies only the structured
    reason; this entry point keeps the public text and the source path.
    """
    if failure.stage is confirmed.ConfirmStage.INSPECT and failure.reason in (
        confirmed.ConfirmReason.NOT_REGULAR,
        confirmed.ConfirmReason.NOW_SYMLINK,
    ):
        return ValueError(f"cannot read {display}: not a regular file")
    if failure.reason is confirmed.ConfirmReason.WRONG_IDENTITY:
        return ValueError(
            f"file changed during import: {display} "
            "(identity, size or modification time changed)"
        )
    if failure.error is not None:
        return ValueError(f"cannot read {display}: {failure.error}")
    return ValueError(f"cannot read {display}: not a regular file")


@dataclass(frozen=True)
class ImportResult:
    digest: str
    added: bool


@dataclass(frozen=True)
class SplitPlanResult:
    name: str
    created: bool
    plan: dict[str, Any]


def _is_count(value: Any) -> bool:
    """Whether ``value`` can serve as a sample or category count.

    Only a genuine non-negative integer qualifies: booleans, floats
    (even integral ones like ``1.0``), strings and ``None`` are rejected
    even when they would compare equal to the correct count.
    """
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validated_distribution(distribution: Any, description: str) -> dict[str, Any]:
    """Return a plan's class distribution, requiring a JSON object of counts.

    A missing distribution — or an array, string or ``null`` in its
    place — is a plan error, never silently treated as an empty
    distribution, and every category count must be a non-negative
    integer.  ``description`` names the statistic (and its set, when
    the distribution belongs to one) for the error message.
    """
    if not isinstance(distribution, dict):
        raise SplitError(f"Split plan record is missing {description}")
    for category, count in distribution.items():
        if not _is_count(count):
            raise SplitError(
                f"Split plan record has an invalid count for category "
                f"{category!r} in {description}"
            )
    return distribution


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
            else:
                # An existing manifest is never rebuilt, emptied or
                # normalized away — but initializing over a structurally
                # broken one must not report success, so the corruption is
                # surfaced here exactly as for any other operation.
                self._read()

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
        a candidate cannot be read in full, a candidate's identity,
        size or modification time changes while it is being read, or a
        subdirectory confirmed as a real directory is replaced by a
        symbolic link before its scan or before its candidates finish
        reading — the link target's images are never imported or even
        reported as duplicates.
        """
        self.initialize()
        source_path = Path(source)
        if not source_path.exists():
            raise ValueError(f"source directory does not exist: {source}")
        if not source_path.is_dir():
            raise ValueError(f"source is not a directory: {source}")
        root = source_path.resolve()

        rel_paths, confirmed_dirs = self._scan_image_files(root, recursive)
        # Read (and re-stat) every candidate before taking the workspace
        # lock: I/O overlaps with peers, and a file that changes mid-read
        # fails the whole batch before anything is committed.  A confirmed
        # subdirectory swapped for a symlink is never read through: every
        # candidate's ancestor directories are re-checked before it is
        # opened, the open itself is pinned to the confirmed ancestor
        # directories with O_NOFOLLOW descriptors, and the confirmed
        # directories are checked once more after all reads, so the link
        # target's images cannot be imported or even reported as
        # duplicates.
        candidates = []
        for rel_path in rel_paths:
            ancestor = root
            for part in Path(rel_path).parts[:-1]:
                ancestor = ancestor / part
                self._raise_if_now_symlink(ancestor)
            candidates.append(self._read_candidate(root / rel_path, rel_path))
        for directory in confirmed_dirs:
            self._raise_if_now_symlink(directory)

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

    def _scan_image_files(
        self, root: Path, recursive: bool
    ) -> tuple[list[str], list[Path]]:
        """Return accepted files and the real directories descended into.

        The first element lists accepted files as ``/``-separated paths
        relative to root; the second lists every subdirectory confirmed as
        a real directory and descended into (empty unless ``recursive``).

        Only real regular files qualify; symlinks and other entry types are
        skipped, and recursion never follows symlinked directories.  Any
        directory that should be visited but cannot be scanned fails the
        whole import with the path and reason.  Every confirmed
        subdirectory is opened through its pinned parent directory with
        ``O_DIRECTORY | O_NOFOLLOW`` and scanned through that descriptor,
        so a subdirectory replaced by a symbolic link at any moment before
        or during its scan fails the whole import instead of the link
        target being scanned.  The confirmed list lets the caller
        re-verify the directories once more after the candidate reads.
        """
        rel_paths: list[str] = []
        confirmed_dirs: list[Path] = []

        def classify(
            directory: Path, fd_or_path: Any
        ) -> list[tuple[str, bool, bool]]:
            try:
                entries = list(os.scandir(fd_or_path))
            except OSError as error:
                raise ValueError(
                    f"cannot scan directory {directory}: {error}"
                ) from error
            found: list[tuple[str, bool, bool]] = []
            for entry in entries:
                try:
                    is_file = entry.is_file(follow_symlinks=False)
                    is_dir = (
                        not is_file
                        and recursive
                        and entry.is_dir(follow_symlinks=False)
                    )
                except OSError as error:
                    raise ValueError(
                        f"cannot inspect entry {directory / entry.name}: {error}"
                    ) from error
                found.append((entry.name, is_file, is_dir))
            return found

        def walk(
            directory: Path,
            rel_prefix: str,
            parent_fd: int,
            entries: list[tuple[str, bool, bool]],
        ) -> None:
            for name, is_file, is_dir in entries:
                if is_file:
                    if Path(name).suffix.lower() in IMAGE_EXTENSIONS:
                        rel_paths.append(rel_prefix + name)
                elif is_dir:
                    subdirectory = directory / name
                    # Open the confirmed subdirectory through the pinned
                    # parent with O_NOFOLLOW and scan the descriptor
                    # itself: a symlink swapped onto the path by now fails
                    # the whole import here, and the link target is never
                    # scanned.
                    try:
                        child_fd = self._open_real_directory(
                            name, parent_fd, subdirectory
                        )
                    except OSError as error:
                        raise ValueError(
                            f"cannot scan directory {subdirectory}: {error}"
                        ) from error
                    try:
                        child_entries = classify(subdirectory, child_fd)
                        confirmed_dirs.append(subdirectory)
                        walk(
                            subdirectory,
                            rel_prefix + name + "/",
                            child_fd,
                            child_entries,
                        )
                    finally:
                        os.close(child_fd)

        try:
            root_fd = self._open_real_directory(str(root), None, root)
        except OSError as error:
            raise ValueError(f"cannot scan directory {root}: {error}") from error
        try:
            walk(root, "", root_fd, classify(root, root))
        finally:
            os.close(root_fd)
        # Unicode code point order on the slash-separated relative path.
        rel_paths.sort()
        return rel_paths, confirmed_dirs

    @staticmethod
    def _open_real_directory(name: str, dir_fd: int | None, path: Path) -> int:
        """Open a confirmed directory, refusing a symlink swapped onto it.

        The open uses ``O_DIRECTORY | O_NOFOLLOW`` — relative to the
        pinned parent descriptor when ``dir_fd`` is given — so a symbolic
        link (or any non-directory) that replaced the confirmed directory
        is never descended through.  A symlink swap raises the import's
        "directory changed" error naming the replaced path; any other
        open failure propagates as the underlying :class:`OSError`.
        """
        try:
            return os.open(name, _DIRECTORY_OPEN_FLAGS, dir_fd=dir_fd)
        except OSError as error:
            # Linux reports a symlink opened with O_NOFOLLOW|O_DIRECTORY
            # as ENOTDIR rather than ELOOP, so confirm via lstat() either
            # way.
            if error.errno in (errno.ELOOP, errno.ENOTDIR):
                try:
                    current = os.lstat(path)
                except OSError:
                    current = None
                if current is not None and stat.S_ISLNK(current.st_mode):
                    raise ValueError(
                        f"directory changed during import: {path} "
                        "is now a symbolic link"
                    ) from error
            raise

    @staticmethod
    def _raise_if_now_symlink(directory: Path) -> None:
        """Fail if a previously confirmed real directory is now a symlink.

        A path that vanished is left to the candidate reads (or ignored
        when the directory held no candidates); only the symlink swap is
        rejected here, because following the link would import the target's
        images against the no-symlinked-directories rule.
        """
        try:
            current = os.lstat(directory)
        except OSError:
            return
        if stat.S_ISLNK(current.st_mode):
            raise ValueError(
                f"directory changed during import: {directory} "
                "is now a symbolic link"
            )

    def _read_candidate(
        self, path: Path, rel_path: str
    ) -> tuple[str, Path, str, int]:
        """Read one candidate in full, verifying identity/size/mtime around it.

        Returns ``(rel_path, absolute_path, digest, size)``.  A mismatch or
        read failure raises before the batch is committed.

        Every ancestor directory of the candidate is re-opened from the
        source root with ``O_DIRECTORY | O_NOFOLLOW``, each relative to its
        pinned parent, and the candidate itself is opened relative to its
        pinned parent directory: a subdirectory replaced by a symbolic
        link since the scan fails the whole import here, and the link
        target's images are never opened — no matter when the swap lands.
        """
        parts = Path(rel_path).parts
        root = path
        for _ in parts:
            root = root.parent
        try:
            fd = self._open_real_directory(str(root), None, root)
        except OSError as error:
            raise ValueError(f"cannot read {rel_path}: {error}") from error
        try:
            ancestor = root
            for part in parts[:-1]:
                ancestor = ancestor / part
                try:
                    next_fd = self._open_real_directory(part, fd, ancestor)
                except OSError as error:
                    raise ValueError(
                        f"cannot read {rel_path}: {error}"
                    ) from error
                os.close(fd)
                fd = next_fd
            digest, size = self._read_stable(
                path, rel_path, dir_fd=fd, name=parts[-1]
            )
        finally:
            os.close(fd)
        return rel_path, path, digest, size

    @staticmethod
    def _read_stable(
        path: Path,
        display: str,
        *,
        dir_fd: int | None = None,
        name: str | None = None,
    ) -> tuple[str, int]:
        """Read one file in full and return ``(digest, size)`` for it.

        The digest and size are guaranteed to describe the same stable
        regular file: the path is lstat()ed without following symlinks
        and opened with ``O_NOFOLLOW``, the opened descriptor is
        fstat()ed to prove it is the inode that was confirmed, and the
        path is lstat()ed again after the read, requiring the same
        object with the same size and modification time.  Any identity,
        size or modification-time change — in-place rewrites, appends,
        truncation, delete-and-recreate, or replacing the path with
        another file, even one with identical content, size and mtime —
        fails with the path and reason, as do a vanishing path, a
        non-regular file and any open or read error.

        The confirm/open/prove rule itself lives in :mod:`confirmed`
        and is shared with the source-directory export; only the
        post-read stability requirement and the import wording are
        specific to this entry point.

        When ``dir_fd`` and ``name`` are given (whole-directory import),
        the pre-open inspection and the post-read re-inspection stay on
        the path, but the open itself is ``name`` relative to the pinned
        parent directory descriptor: an ancestor directory replaced by a
        symbolic link can never redirect the open into the link target.
        """
        try:
            if dir_fd is None:
                opened_file = confirmed.confirm_and_open_regular(path)
            else:
                confirmed_status = confirmed.inspect_regular(path)
                descriptor = confirmed.open_without_follow(name, dir_fd=dir_fd)
                opened_file = confirmed.ConfirmedOpen(
                    descriptor,
                    confirmed_status,
                    confirmed.prove_opened_regular(descriptor, confirmed_status),
                )
        except confirmed.ConfirmationError as failure:
            raise _import_confirmation_error(failure, display) from failure
        try:
            digest = confirmed.hash_descriptor(opened_file.descriptor)
        except OSError as error:
            raise ValueError(f"cannot read {display}: {error}") from error
        # Import, unlike export, also requires the path itself to still
        # name the same unchanged file after the read: an in-place
        # rewrite, append, truncation or replacement that landed while
        # the bytes were streamed must not register a digest/size pair
        # that never coexisted.
        try:
            after = os.lstat(path)
        except OSError as error:
            raise ValueError(f"cannot read {display}: {error}") from error
        if not confirmed.same_identity_size_mtime(opened_file.confirmed, after):
            raise ValueError(
                f"file changed during import: {display} "
                "(identity, size or modification time changed)"
            )
        return digest, opened_file.confirmed.st_size

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

    def find_by_label(self, label: str) -> dict[str, Any]:
        """List samples currently registered under one exact category label.

        Matching is exact on the current manifest label: case, surrounding
        whitespace and path separators are all kept literally, so ``"cat"``
        never matches ``"Cat"`` or ``"cat "``.  The empty string queries
        unlabeled samples — records whose stored label is ``None`` or ``""``
        — and every matching record's label is reported uniformly as
        ``None``.  A genuine class literally named ``"unlabeled"`` is
        queried with that text and never mixed with unlabeled samples,
        unlike :meth:`summary` which merges them under one display key.

        Each record provides the full SHA-256 digest, the source path saved
        at first registration, the byte size and the current label;
        records are sorted ascending by full digest and ``count`` always
        equals the list length.  The query reads the registered records
        only — it never opens a source file, and saved split plans' old
        labels are neither matched nor rewritten — so a moved, deleted or
        unreadable source is still found, and a batch update or undo
        committed before the call is immediately reflected.
        """
        if not isinstance(label, str):
            raise BatchError("query label must be a string")
        want_unlabeled = label == ""
        samples: list[dict[str, Any]] = []
        self.initialize()
        with self._locked():
            items = self._read()["items"]
        for item in items:
            stored = item.get("label")
            if want_unlabeled:
                if not (stored is None or stored == ""):
                    continue
                current = None
            else:
                if stored != label:
                    continue
                current = stored
            samples.append(
                {
                    "sha256": item["sha256"],
                    "source": item["source"],
                    "size": item["size"],
                    "label": current,
                }
            )
        samples.sort(key=lambda sample: sample["sha256"])
        return {"label": label, "count": len(samples), "samples": samples}

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

        Parsing happens before the lock; inside the lock the submission
        runs through clearly separated stages, each with one job:

        1. **Unknown samples** — every named digest must be registered; the
           first unknown one rejects the batch with nothing written.
        2. **Number identity** — a batch number already in history either
           replays the first submission's result (same content) or is a
           conflict (different content); neither touches the manifest.
        3. **Verification/classification** — each current label is read
           once: an expected-old mismatch rejects the batch, and the same
           reading decides whether that record actually changes.
        4. **Application** — only records classified as changed are
           mutated; if none are, the batch occupies no number and writes
           no history.
        5. **History and save** — the entry and result are assembled from
           the applied records, and manifest plus history commit together
           through the journal.

        No stage mutates anything before every earlier stage has passed, so
        a rejection can neither leave partially updated samples nor a
        partial history entry.
        """
        number, records = parse_batch_file(data)
        with self._locked(create=True):
            manifest = self._read()
            history = self._read_history()
            items = {item["sha256"]: item for item in manifest["items"]}

            # Stage 1: every record must name a registered sample.
            reject_unknown_samples(records, items.__contains__)

            # Stage 2: a known number is replayed or conflicts; a fresh
            # number proceeds without examining the current labels.
            existing = next(
                (entry for entry in history["batches"] if entry["batch"] == number),
                None,
            )
            replay = resolve_re_submission(number, records, existing)
            if replay is not None:
                return replay

            # Stage 3: verify expected old labels and classify changed vs
            # unchanged in a single current-label pass.
            resolutions = verify_records(
                records,
                lambda digest: normalize_label(items[digest].get("label")),
            )

            # Stage 4: apply is the only mutation.  No changes at all
            # occupies no batch number and writes no history.
            if not any(resolution.changed for resolution in resolutions):
                return no_changes_result(number, len(records))
            applied_records = apply_resolutions(resolutions, items)

            # Stage 5: assemble history, then save manifest and history
            # together through the journal.
            entry = build_history_entry(number, applied_records)
            history["batches"].append(entry)
            self._commit(manifest, history)
            return result_payload(number, "applied", applied_records)

    def undo_batch(self, number: str) -> dict[str, Any]:
        """Undo a successful batch by number.

        Only samples the batch actually changed are checked: each must have
        the same label revision it had right after the batch (no later
        modification, even one that restored the same label).  Repeated
        undos are idempotent and do not add history.

        Integrity comes before either verdict: the target batch must list
        every sample exactly once, carry a sound pinned revision on each
        record, and have every record's ``changed`` flag agree with the
        before/after labels stored on that record.  A duplicated digest, a
        bad row, or a flag contradicting its stored labels rejects the undo
        even when the batch is already marked undone, so corruption is
        never hidden behind an ``already-undone`` success or confused with
        a sample modified after the batch.
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

            # Validate the target batch's integrity — one row per sample,
            # sound pinned revisions, and changed flags that match the
            # stored before/after labels — before the already-undone
            # short-circuit and the later-modification checks: a duplicated,
            # corrupt or internally contradictory history record is refusal,
            # not an undo, a no-op or an unknown batch, and nothing is
            # rewritten.
            require_intact_history_records(entry)
            require_intact_history_revisions(entry)
            require_intact_history_changed_flags(entry)

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
                # record["rev"] is now known to be a genuine non-negative
                # integer, so a boolean/decimal/string can never compare
                # equal to the sample's current revision.
                if item_revision(item) != record["rev"] or current != record["new"]:
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
                item["rev"] = item_revision(item) + 1
                restored += 1

            entry["undone"] = True
            entry["undone_at"] = datetime.now(timezone.utc).isoformat()
            self._commit(manifest, history)
            return {"batch": number, "status": "undone", "restored": restored}

    def history(self) -> list[dict[str, Any]]:
        """Return successful batch entries in submission order."""
        with self._locked():
            return self._read_history()["batches"]

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
        if not isinstance(history, dict):
            raise BatchError("cannot recover interrupted transaction: journal is malformed")
        # Only install a coherent, fully valid pair, so recovery can never
        # replace the committed manifest with a structurally broken sample
        # list (a missing field, a bad field type or a duplicated digest):
        # the check runs before either file is rewritten, leaving both the
        # last committed registration and its history untouched.
        try:
            validate_manifest(manifest)
        except ManifestError as error:
            raise BatchError(
                f"cannot recover interrupted transaction: {error}"
            ) from error
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
        if not isinstance(samples, dict):
            raise SplitError("Split plan record is missing its sample summary")
        if not _is_count(samples.get("total")):
            raise SplitError("Split plan record has an invalid sample total")
        overall_distribution = _validated_distribution(
            samples.get("distribution"), "its class distribution"
        )
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
            if not _is_count(set_payload.get("samples")):
                raise SplitError(
                    f"Split plan record has an invalid sample count for {set_name}"
                )
            set_distribution = _validated_distribution(
                set_payload.get("distribution"),
                f"its class distribution for {set_name}",
            )
            distribution: Counter[str] = Counter()
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
            if set_payload["samples"] != len(members):
                raise SplitError(f"Split plan record has a bad count for {set_name}")
            if dict(sorted(distribution.items())) != dict(
                sorted(set_distribution.items())
            ):
                raise SplitError(
                    f"Split plan record has a mismatched distribution for {set_name}"
                )
        if samples["total"] != len(seen):
            raise SplitError("Split plan record has a mismatched sample total")
        if dict(sorted(overall.items())) != dict(
            sorted(overall_distribution.items())
        ):
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
        """Read and fully validate the current manifest.

        Every caller — queries included — gets only a manifest whose
        version and entire sample list are structurally sound: a missing
        field, a wrong field type or a duplicated digest rejects the whole
        operation before any sample is used, rather than surfacing as a
        runtime exception on one record or silently acting on only one of
        two duplicate registrations.
        """
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ManifestError(
                f"Dataset manifest is corrupted: cannot read manifest.json: {error}"
            ) from error
        return validate_manifest(data)
