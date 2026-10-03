"""Portable offline export of a saved split plan.

An export package is a self-contained ZIP: one directory per set
(``train``/``validation``/``test``), each containing the same numbered
ImageFolder class directories, plus a UTF-8 JSON manifest.  Samples are
stored under ``<set>/<class>/<full-sha256><source-extension>``.

The export is driven solely by the saved plan: its sample identities,
labels, set assignments and recorded source paths.  Later imports, label
changes or undos never affect an exported package, and exporting never
writes to the workspace.

When ``source_dir`` is given, files are read from that directory tree
instead of the recorded source paths: every regular file under the
directory (no extension filter; symlinks and other non-regular entries
are skipped and symlinked directories are never descended) is read in
full and identified by content digest, and each exported sample is
matched to a file whose full SHA-256 matches the plan.  The selected
copy is pinned through the open (lstat without symlink follow,
``O_NOFOLLOW`` and an fstat of the descriptor), so the bytes read come
from the exact regular file confirmed at lookup time; a replacement
landing after the lookup — another regular file, even with identical
content, or a symlink, even pointing at the original file — fails the
export.  Directories are held to the same rule during the lookup: each
directory is confirmed to be a real directory (not a symlink) right
before it is scanned and re-confirmed after its entries have been
processed, so a subdirectory swapped for a symlink mid-lookup — wherever
the link points, even at the moved-away original tree — fails the whole
export naming the directory instead of being searched through.  The new
directory only changes where bytes are read from; sample identities,
labels, set assignments and package file extensions remain the plan's.
The lookup is used for this one export only and is never written back.

ZIP bytes are deterministic for a given plan, options and source content:
entry metadata uses fixed values, entries are written in a fixed order and
source modification times are never consulted.

A single sample whose bytes reach the classic ZIP 32-bit size limit
(2 GiB, or close enough that its deflated form could) is written as a
ZIP64 entry: the local file header carries a ZIP64 extra field with the
real sizes while the entry is streamed, so no sample has to fit in
memory and highly compressible multi-gigabyte content is supported even
when the resulting package stays small.  Plans containing only smaller
samples use exactly the classic format they always did, byte for byte.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from .splits import SET_NAMES

EXPORT_SCHEMA_VERSION = 1

# Fixed ZIP entry metadata: deterministic across machines and runs.
_FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)
_CREATE_SYSTEM_UNIX = 3
_FILE_ATTR = 0o100644 << 16
_DIR_ATTR = (0o40755 << 16) | 0x10  # MS-DOS directory flag
_CHUNK_SIZE = 1024 * 1024
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")


class ExportError(ValueError):
    """An export cannot be produced."""


def _entry_needs_zip64(file_size: int) -> bool:
    """Whether one sample entry must use the ZIP64 local header.

    Mirrors zipfile's own pre-streaming heuristic (the uncompressed size
    plus a 5% margin, compared with the 32-bit size field limit): the
    margin covers both a size landing at the exact limit and deflate
    growth past it on incompressible content, while everything below it
    keeps the classic header byte for byte.
    """
    return file_size * 21 > zipfile.ZIP64_LIMIT * 20


def export_split(
    store: Any,
    plan_name: str,
    target: Path,
    skip_unlabeled: bool = False,
    source_dir: Path | None = None,
) -> dict[str, Any]:
    """Export a saved split plan to a new ZIP at ``target``.

    Returns a result payload describing the plan, the number of samples
    actually exported, the class distribution per set and the per-set skip
    counts.  Raises :class:`ExportError` (a ``ValueError``) on any failure;
    the target path is never created in that case.

    When ``source_dir`` is given, sample bytes are read from that directory
    tree instead of the plan's recorded source paths: the directory is
    walked (regular files only, no extension filter, never through
    symlinks), every regular file is hashed in full, and each exported
    sample is matched to a file with the same full SHA-256 digest.  The
    plan still decides identities, labels, set assignments and package
    file extensions; the lookup is used for this export only.
    """
    plan = store.get_split(plan_name)
    members = _collect_members(plan)

    unlabeled = [member for member in members if member["label"] == ""]
    if unlabeled and not skip_unlabeled:
        first = min(unlabeled, key=lambda member: member["sha256"])
        raise ExportError(
            f"sample {first['sha256']} in set {first['set']} has no label; "
            "pass --skip-unlabeled to export without unlabeled samples"
        )

    exported = [member for member in members if member["label"] != ""]
    skipped_counts = {set_name: 0 for set_name in SET_NAMES}
    for member in unlabeled:
        skipped_counts[member["set"]] += 1

    classes = _build_classes(exported)
    manifest = _build_manifest(plan, classes, exported, skipped_counts)
    distributions = _distributions(exported)

    resolver = None
    if source_dir is not None:
        source_dir = Path(source_dir)
        if not source_dir.exists():
            raise ExportError(f"source directory does not exist: {source_dir}")
        if not source_dir.is_dir():
            raise ExportError(f"source is not a directory: {source_dir}")
        # An empty plan (or every sample skipped) needs no files at all;
        # the directory still has to exist and be one.
        if exported:
            resolver = _scan_source_directory(source_dir.resolve())
            for member in exported:
                if member["sha256"] not in resolver:
                    raise ExportError(
                        f"sample {member['sha256']} in set {member['set']}: "
                        f"no file with matching content (full SHA-256 "
                        f"{member['sha256']}) found under source directory "
                        f"{source_dir}"
                    )

    _write_package(target, classes, exported, manifest, resolver)

    return {
        "plan": plan["name"],
        "exported": len(exported),
        "skipped": skipped_counts,
        "sets": {
            set_name: {
                "samples": distributions[set_name]["samples"],
                "distribution": dict(
                    sorted(distributions[set_name]["distribution"].items())
                ),
            }
            for set_name in SET_NAMES
        },
        "target": str(target.resolve()),
    }


def _collect_members(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten plan members and validate the fields export depends on."""
    members: list[dict[str, Any]] = []
    for set_name in SET_NAMES:
        payload = plan["sets"][set_name]
        for member in payload["members"]:
            digest = member.get("sha256")
            label = member.get("label")
            source = member.get("source")
            if not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
                raise ExportError(
                    f"split plan {plan.get('name')!r} is corrupted: "
                    f"sample {digest!r} is not a full SHA-256 digest"
                )
            if not isinstance(label, str):
                raise ExportError(
                    f"split plan {plan.get('name')!r} is corrupted: "
                    f"sample {digest} has a non-string label"
                )
            if not isinstance(source, str) or not source:
                raise ExportError(
                    f"split plan {plan.get('name')!r} is corrupted: "
                    f"sample {digest} has no recorded source path"
                )
            members.append(
                {
                    "sha256": digest,
                    "label": label,
                    "source": source,
                    "set": set_name,
                }
            )
    return members


def _build_classes(members: list[dict[str, Any]]) -> dict[str, str]:
    """Map every exported label to one shared numbered class directory.

    All sets use the same numbering; labels are sorted by their literal
    text so Chinese labels, labels containing path separators and labels
    differing only in case each get a distinct, collision-free directory.
    """
    labels = sorted({member["label"] for member in members})
    width = max(2, len(str(len(labels)))) if labels else 2
    return {label: f"class_{index:0{width}d}" for index, label in enumerate(labels, 1)}


def _build_manifest(
    plan: dict[str, Any],
    classes: dict[str, str],
    members: list[dict[str, Any]],
    skipped_counts: dict[str, int],
) -> dict[str, Any]:
    set_summaries = {}
    for set_name in SET_NAMES:
        set_members = [member for member in members if member["set"] == set_name]
        set_summaries[set_name] = {
            "samples": len(set_members),
            "skipped": skipped_counts[set_name],
        }

    sample_entries = []
    for member in sorted(members, key=lambda item: item["sha256"]):
        arc_path = _sample_path(member, classes[member["label"]])
        sample_entries.append(
            {
                "sha256": member["sha256"],
                "label": member["label"],
                "set": member["set"],
                "path": arc_path,
            }
        )

    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "plan": plan["name"],
        "seed": plan["seed"],
        "ratios": {set_name: plan["ratios"][set_name] for set_name in SET_NAMES},
        "classes": [
            {"label": label, "directory": directory}
            for label, directory in sorted(classes.items())
        ],
        "sets": set_summaries,
        "samples": sample_entries,
    }


def _sample_path(member: dict[str, Any], class_directory: str) -> str:
    """Package-relative path for a sample: ``<set>/<class>/<digest><ext>``."""
    extension = Path(member["source"]).suffix
    return f"{member['set']}/{class_directory}/{member['sha256']}{extension}"


def _distributions(members: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for set_name in SET_NAMES:
        set_members = [member for member in members if member["set"] == set_name]
        result[set_name] = {
            "samples": len(set_members),
            "distribution": Counter(member["label"] for member in set_members),
        }
    return result


# A resolver maps a full SHA-256 digest to the absolute path of the
# sorted-first regular file carrying that content, plus the file's
# identity (device, inode) captured during the lookup.  The copy opens
# that exact inode pinned through the open, so a swap after the lookup —
# another regular file, even with identical content, or a symlink, even
# one pointing at the original file — cannot be read in its place.
Resolver = dict[str, tuple[Path, int, int]]


def _scan_source_directory(root: Path) -> Resolver:
    """Walk ``root`` and map every digest to the sorted-first copy.

    The directory and all its subdirectories are searched for regular
    files only: there is no extension filter, symlinks and other
    non-regular entries are skipped, and recursion never descends through
    symlinked directories.  Every regular file found is read in full and
    identified by its content digest; when several files share content,
    the one whose slash-separated path relative to ``root`` sorts first
    by Unicode code point owns the digest.  A directory that cannot be
    scanned, an entry that cannot be inspected or a file that cannot be
    read in full fails the export with the path and reason.

    A directory confirmed as a real directory by its parent's listing is
    re-checked without following symlinks right before it is scanned and
    again after its entries have been processed: if it has been swapped
    for a symlink in between — at any depth, wherever the link points —
    the lookup fails naming the directory, so the link target is never
    searched, matched or packaged in place of the confirmed tree.
    """
    found: dict[str, tuple[str, Path, int, int]] = {}

    def walk(directory: Path) -> None:
        confirmed = _confirm_real_directory(directory)
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            raise ExportError(f"cannot scan directory {directory}: {error}") from error
        for entry in entries:
            try:
                is_file = entry.is_file(follow_symlinks=False)
            except OSError as error:
                raise ExportError(
                    f"cannot inspect entry {entry.path}: {error}"
                ) from error
            if is_file:
                rel = os.path.relpath(entry.path, root).replace(os.sep, "/")
                digest, device, inode = _hash_regular_file(Path(entry.path))
                current = found.get(digest)
                if current is None or rel < current[0]:
                    found[digest] = (rel, Path(entry.path), device, inode)
            else:
                try:
                    is_dir = entry.is_dir(follow_symlinks=False)
                except OSError as error:
                    raise ExportError(
                        f"cannot inspect entry {entry.path}: {error}"
                    ) from error
                if is_dir:
                    walk(Path(entry.path))
        _confirm_directory_unchanged(directory, confirmed)

    walk(root)
    return {
        digest: (path, device, inode)
        for digest, (_, path, device, inode) in found.items()
    }


def _confirm_real_directory(directory: Path) -> tuple[int, int]:
    """Confirm ``directory`` is still a real directory and pin its identity.

    A subdirectory is confirmed as a real directory by its parent's
    listing before it is descended into; if it has been swapped for a
    symlink in the gap, descending would search the link target, which
    the never-through-symlinks rule forbids, so the whole lookup fails
    naming the directory and the symlink.  Returns the confirmed
    ``(device, inode)`` so a later re-check can spot a swap landing
    while the directory's entries are being processed.
    """
    try:
        stat_result = os.lstat(directory)
    except OSError as error:
        raise ExportError(f"cannot scan directory {directory}: {error}") from error
    if stat.S_ISLNK(stat_result.st_mode):
        raise ExportError(
            f"directory changed during lookup: {directory} is now a symlink"
        )
    if not stat.S_ISDIR(stat_result.st_mode):
        raise ExportError(f"cannot scan directory {directory}: not a directory")
    return stat_result.st_dev, stat_result.st_ino


def _confirm_directory_unchanged(
    directory: Path, confirmed: tuple[int, int]
) -> None:
    """Re-verify a scanned directory before its results are trusted.

    The directory's entries were listed and its files read through the
    path confirmed on entry; if the path has been swapped for a symlink
    in the meantime — even one pointing at the moved-away original tree,
    with byte-identical content — later opens through it would read the
    link target, so the whole lookup (and the export) must fail naming
    the directory and the reason.  A swap for a different real directory
    is rejected the same way.
    """
    try:
        stat_result = os.lstat(directory)
    except OSError as error:
        raise ExportError(f"cannot scan directory {directory}: {error}") from error
    if stat.S_ISLNK(stat_result.st_mode):
        raise ExportError(
            f"directory changed during lookup: {directory} is now a symlink"
        )
    if not stat.S_ISDIR(stat_result.st_mode) or (
        stat_result.st_dev,
        stat_result.st_ino,
    ) != confirmed:
        raise ExportError(
            f"directory changed during lookup: {directory} was replaced"
        )


def _hash_regular_file(path: Path) -> tuple[str, int, int]:
    """Read one file in full, returning ``(digest, device, inode)``.

    The bytes are read with ``O_NOFOLLOW`` so a symlink swapped in after
    the lookup cannot redirect the read; any failure raises with the path
    and reason.
    """
    try:
        stat_result = os.lstat(path)
    except OSError as error:
        raise ExportError(f"cannot read source file {path}: {error}") from error
    if not stat.S_ISREG(stat_result.st_mode):
        raise ExportError(f"cannot read source file {path}: not a regular file")
    try:
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
    except OSError as error:
        raise ExportError(f"cannot read source file {path}: {error}") from error
    try:
        hasher = hashlib.sha256()
        with os.fdopen(descriptor, "rb") as stream:
            while True:
                chunk = stream.read(_CHUNK_SIZE)
                if not chunk:
                    break
                hasher.update(chunk)
    except OSError as error:
        raise ExportError(f"cannot read source file {path}: {error}") from error
    return hasher.hexdigest(), stat_result.st_dev, stat_result.st_ino


def _write_package(
    target: Path,
    classes: dict[str, str],
    members: list[dict[str, Any]],
    manifest: dict[str, Any],
    resolver: dict[str, tuple[Path, int, int]] | None = None,
) -> None:
    """Write the ZIP atomically: lock the target name, stream to a temp file.

    The target is never overwritten: an existing target is rejected and a
    temp file is used until the whole package has been written and fsynced,
    so a failure or interruption leaves no incomplete target.  An flock on
    a lock file next to the target serializes concurrent exports to the
    same path.
    """
    target = Path(target)
    parent = target.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ExportError(f"cannot create target directory {parent}: {error}") from error

    lock_path = target.with_name(f".{target.name}.lock")
    try:
        lock_descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError as error:
        raise ExportError(f"cannot lock target {target}: {error}") from error

    temp_path: Path | None = None
    try:
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        if os.path.lexists(target):
            raise ExportError(f"target already exists: {target}")

        fd, temp_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=parent
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                _write_zip(stream, classes, members, manifest, resolver)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, target)
            temp_path = None
            _fsync_directory(parent)
        except BaseException:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise
    except OSError as error:
        raise ExportError(f"cannot write export package: {error}") from error
    finally:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(lock_descriptor)


def _write_zip(
    stream: Any,
    classes: dict[str, str],
    members: list[dict[str, Any]],
    manifest: dict[str, Any],
    resolver: Resolver | None = None,
) -> None:
    """Stream the package into ``stream`` without holding media in memory."""
    directory_entries = _directory_entries(classes)
    file_entries = sorted(
        (_sample_path(member, classes[member["label"]]), member)
        for member in members
    )

    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(directory_entries):
            archive.writestr(_directory_info(name), b"")
        for arc_name, member in file_entries:
            _write_sample(archive, arc_name, member, resolver)
        archive.writestr(_file_info("manifest.json"), _manifest_bytes(manifest))


def _directory_entries(classes: dict[str, str]) -> set[str]:
    entries = {f"{set_name}/" for set_name in SET_NAMES}
    for class_directory in classes.values():
        for set_name in SET_NAMES:
            entries.add(f"{set_name}/{class_directory}/")
    return entries


def _write_sample(
    archive: zipfile.ZipFile,
    arc_name: str,
    member: dict[str, Any],
    resolver: Resolver | None = None,
) -> None:
    digest = member["sha256"]
    hasher = hashlib.sha256()
    if resolver is not None:
        # Source-directory export: read from the matched copy only.  The
        # plan's recorded source path is deliberately not consulted, so a
        # still-usable original location cannot paper over a missing
        # match.  The opened object is pinned to the regular inode
        # confirmed at lookup time, so a swap after that confirmation
        # cannot be copied in its place.
        try:
            source, expected_device, expected_inode = resolver[digest]
        except KeyError:
            raise ExportError(
                f"sample {digest} in set {member['set']}: no file with "
                f"matching content found under the source directory"
            ) from None
        source_stream, source_size = _open_resolved_source(
            source, digest, expected_device, expected_inode
        )
    else:
        source = Path(member["source"])
        try:
            stat_result = os.stat(source)
        except OSError as error:
            raise ExportError(
                f"sample {digest}: cannot stat source file {source}: {error}"
            ) from error
        if not stat.S_ISREG(stat_result.st_mode):
            raise ExportError(
                f"sample {digest}: source is not a regular file: {source}"
            )
        try:
            source_stream = source.open("rb")
        except OSError as error:
            raise ExportError(
                f"sample {digest}: cannot read source file {source}: {error}"
            ) from error
        source_size = stat_result.st_size

    # Sizes are unknown to the ZIP layer until the entry has been streamed,
    # and the classic local file header cannot describe a 2 GiB-or-larger
    # entry, so decide the ZIP64 format up front from the confirmed file
    # size (with the same 5% headroom zipfile itself uses, which also
    # covers deflate growth on incompressible data).  Smaller entries open
    # without ZIP64, keeping their archive bytes byte-identical to the
    # classic format.  The size is only a format decision: the bytes
    # actually read are always hash-verified below.
    info = _file_info(arc_name)
    info.file_size = source_size
    force_zip64 = _entry_needs_zip64(source_size)
    try:
        with source_stream, archive.open(info, "w", force_zip64=force_zip64) as target_stream:
            while True:
                chunk = source_stream.read(_CHUNK_SIZE)
                if not chunk:
                    break
                hasher.update(chunk)
                target_stream.write(chunk)
    except OSError as error:
        raise ExportError(f"sample {digest}: cannot read source file {source}: {error}") from error
    except RuntimeError as error:
        # The file grew past the 32-bit limit after the stat: the chosen
        # header cannot describe it, so this package cannot be finished.
        # Fail the whole export naming the sample instead of emitting a
        # truncated entry.
        raise ExportError(
            f"sample {digest}: source file {source} changed while it was "
            f"being read and is now too large to describe in the planned "
            f"entry: {error}"
        ) from error

    actual = hasher.hexdigest()
    if actual != digest:
        raise ExportError(
            f"sample {digest}: source file content changed or is corrupt: "
            f"expected digest {digest}, got {actual}"
        )


def _open_resolved_source(
    source: Path, digest: str, expected_device: int, expected_inode: int
) -> tuple[Any, int]:
    """Open the lookup-confirmed copy, pinned to its confirmed inode.

    The selected path can be remapped in the gap between the directory
    lookup and this copy.  The path is lstat()ed without following
    symlinks and is opened with ``O_NOFOLLOW``; the opened descriptor is
    then fstat()ed, so the decision covers the object that is actually
    read, not just one pre-open look at the path.  A stream is returned
    only when that object is a regular file carrying the device/inode
    confirmed at lookup time, together with that confirmed file's size.

    A different regular file renamed onto the path — even one with
    identical content, size and modification time — or a symlink — even
    one pointing at the original file — fails with the sample's full
    digest, the selected source path and the reason.
    """
    try:
        before = os.lstat(source)
    except OSError as error:
        raise ExportError(
            f"sample {digest}: cannot stat source file {source}: {error}"
        ) from error
    if stat.S_ISLNK(before.st_mode):
        raise ExportError(
            f"sample {digest}: source file was replaced after lookup: "
            f"{source} is now a symlink"
        )
    if not stat.S_ISREG(before.st_mode):
        raise ExportError(
            f"sample {digest}: source is not a regular file: {source}"
        )
    if (before.st_dev, before.st_ino) != (expected_device, expected_inode):
        raise ExportError(
            f"sample {digest}: source file was replaced after lookup: {source}"
        )
    try:
        # O_NOFOLLOW makes the open fail instead of following a symlink
        # swapped onto the path after the lstat() above.
        descriptor = os.open(
            source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise ExportError(
                f"sample {digest}: source file was replaced after lookup: "
                f"{source} is now a symlink"
            ) from error
        raise ExportError(
            f"sample {digest}: cannot read source file {source}: {error}"
        ) from error
    try:
        opened = os.fstat(descriptor)
    except OSError as error:
        os.close(descriptor)
        raise ExportError(
            f"sample {digest}: cannot read source file {source}: {error}"
        ) from error
    # Prove the descriptor itself belongs to the confirmed regular inode:
    # the path could have been remapped in the lstat()-to-open() gap.
    if (opened.st_dev, opened.st_ino) != (
        expected_device,
        expected_inode,
    ) or not stat.S_ISREG(opened.st_mode):
        os.close(descriptor)
        raise ExportError(
            f"sample {digest}: source file was replaced after lookup: {source}"
        )
    try:
        return os.fdopen(descriptor, "rb"), opened.st_size
    except OSError as error:
        os.close(descriptor)
        raise ExportError(
            f"sample {digest}: cannot read source file {source}: {error}"
        ) from error


def _file_info(arc_name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(arc_name, date_time=_FIXED_DATE_TIME)
    info.create_system = _CREATE_SYSTEM_UNIX
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = _FILE_ATTR
    return info


def _directory_info(arc_name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(arc_name, date_time=_FIXED_DATE_TIME)
    info.create_system = _CREATE_SYSTEM_UNIX
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = _DIR_ATTR
    return info


def _manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


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
