"""Portable offline export of a saved split plan.

An export package is a self-contained ZIP: one directory per set
(``train``/``validation``/``test``), each containing the same numbered
ImageFolder class directories, plus a UTF-8 JSON manifest.  Samples are
stored under ``<set>/<class>/<full-sha256><source-extension>``.

The export is driven solely by the saved plan: its sample identities,
labels, set assignments and recorded source paths.  Later imports, label
changes or undos never affect an exported package, and exporting never
writes to the workspace.

Without ``source_dir`` each sample is copied from its recorded source
path.  A path that is a symbolic link when the export starts is followed
once, exactly like a single-file import, and the regular file the link
then names is the object confirmed for this read.  From that point the
read is pinned with the same inspect / open-no-follow / fstat-prove rule
as a source-directory copy: the concrete target is lstat()ed, opened with
``O_NOFOLLOW`` plus ``O_NONBLOCK`` and proved by descriptor to be that
exact regular file.  Another regular file renamed onto the path in the
gap — even one with identical content, size and modification time — is
rejected as a replaced source rather than read, and a named pipe swapped
into the gap (even with no writer) is opened without blocking and then
rejected, so the export neither waits for data nor packages pipe bytes.
Failures name the sample's full digest, the recorded source path and the
reason; the recorded path is never rewritten and no same-content
replacement is substituted.

When ``source_dir`` is given, files are read from that directory tree
instead of the recorded source paths: every regular file under the
directory (no extension filter; symlinks and other non-regular entries
are skipped and symlinked directories are never descended) is read in
full and identified by content digest, and each exported sample is
matched to a file whose full SHA-256 matches the plan.  Each scan read
is itself pinned with the same inspect / open-no-follow-nonblocking /
fstat-prove rule as the package copy: the exact regular file the walk
typed is the object whose bytes are hashed, so a named pipe swapped
into the gap before the lookup read — even one with no writer, or one
whose writer would supply the planned bytes — is opened without
blocking and rejected before any byte is consumed, and another
regular file landing there (even byte-, size- and mtime-identical) is
rejected the same way.  The selected copy is pinned again through the
open when it is streamed into the package (lstat without symlink
follow, ``O_NOFOLLOW`` and an fstat of the descriptor), so the bytes
read come from the exact regular file confirmed at lookup time; a
replacement landing after the lookup — another regular file, even
with identical content, or a symlink, even pointing at the original
file — fails the export.

The walk itself is anchored to directory descriptors: the confirmed
root is held open for the whole export and each directory is entered
with ``openat`` (``O_NOFOLLOW`` plus ``O_DIRECTORY``) relative to its
parent's descriptor and verified with ``fstatat`` on both the
parent-relative name and the opened descriptor, so a real subdirectory
cannot be followed through even if its path is swapped for a symlink in
the gap between its first inspection and its entry.  A subdirectory
confirmed real during the walk that is a symlink by the time it is
entered — whether the link points outside the tree, inside it, or at
the moved original — fails the whole export naming that path and the
symlink; the same applies to a swap landing while its children are
being processed.

Both the lookup walk and the copy-phase directory verification run on
an explicit stack of directory frames rather than Python call frames,
so a chain deeper than the interpreter's recursion limit is searched in
full and verified in full: files at the root, on side branches and at
the end of a long chain are all found, hashed and matched, and a
binding change at any depth still keeps the package from being
published.  Display and relative paths are carried as strings on those
frames, so handling a deep name never needs a deep Python call stack
even to render the path.

The protection extends to the package copy that follows the lookup.
Before each matched file is opened its ancestor directories are
re-derived and re-pinned through the same descriptor-anchored checks,
and after every sample has been streamed every directory recorded
during the walk is verified once more before the package is published:
a real directory replaced by a symlink after the lookup — before a
sample is opened, while a sample is read or between samples — still
fails the export, and a byte-equal file behind the new link is never
read, matched or packaged.  Entries that are symlinks when first
encountered keep being skipped exactly as before.

The new directory only changes where bytes are read from; sample
identities, labels, set assignments and package file extensions remain
the plan's.  The lookup is used for this one export only and is never
written back.

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
import stat
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from . import confirmed
from .splits import SET_NAMES, SplitError, invalid_digest_reason

EXPORT_SCHEMA_VERSION = 1

# Fixed ZIP entry metadata: deterministic across machines and runs.
_FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)
_CREATE_SYSTEM_UNIX = 3
_FILE_ATTR = 0o100644 << 16
_DIR_ATTR = (0o40755 << 16) | 0x10  # MS-DOS directory flag
_CHUNK_SIZE = 1024 * 1024
# Descriptor-relative, no-symlink-follow directory opens: POSIX and
# available on the platforms this tool runs on (0 where a flag is absent).
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)


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
    sample is matched to a file with the same full SHA-256 digest.  Both
    the lookup hashes and the later copy use the same pinned
    inspect/open/prove rule: a named pipe or a different regular file
    (even a byte-identical one) swapped in after a scan entry was typed
    fails the export before its bytes are read or waited on, just as a
    swap landing after the lookup does when the selected copy is opened.
    The plan still decides identities, labels, set assignments and
    package file extensions; the lookup is used for this export only.

    Without ``source_dir`` each sample is read from its recorded source
    path, pinned to the one ordinary regular file confirmed immediately
    before the read: a symlink present at the start is followed once,
    while a replacement landing in the confirm-to-open gap — another
    regular file, even a byte-identical one, or a named pipe, even one
    with no writer — fails the whole export instead of being read.

    The walk is anchored to directory descriptors, so a subdirectory that
    was a real directory when the walk reached it but has become a symlink
    before it is entered or while its contents are read fails the export
    naming that path; the link is never followed.  The walk and the
    copy-phase verification iterate an explicit stack rather than Python
    call frames, so a moved tree whose directory chain is deeper than
    the interpreter's recursion limit is still searched and verified in
    full instead of failing partway.
    """
    try:
        plan = store.get_split(plan_name)
    except SplitError as error:
        # Reading a saved plan already enforces the same identity and
        # proportion rules as "split show"; export reports them through
        # its own error type while keeping the message (plan name, set,
        # member position and the concrete problem) intact.  This runs
        # before the unlabeled check, so --skip-unlabeled can never
        # bypass a corrupted identity.
        raise ExportError(str(error)) from error
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
            resolved_root = source_dir.resolve()
            # Keep the confirmed root open and pinned for the whole
            # export: the lookup and the later package copy both resolve
            # through this descriptor, so a subdirectory turned into a
            # symlink after the lookup cannot be traversed while reading
            # the package bytes either.
            try:
                root_fd = os.open(resolved_root, os.O_RDONLY | _O_DIRECTORY)
            except OSError as error:
                raise ExportError(
                    f"cannot scan directory {resolved_root}: {error}"
                ) from error
            try:
                resolver = _scan_source_directory(resolved_root)
                for member in exported:
                    if member["sha256"] not in resolver:
                        raise ExportError(
                            f"sample {member['sha256']} in set {member['set']}: "
                            f"no file with matching content (full SHA-256 "
                            f"{member['sha256']}) found under source directory "
                            f"{source_dir}"
                        )
                directory_ids = getattr(resolver, "directory_ids", None)
                anchor = (
                    _SourceAnchor(root_fd, resolved_root, directory_ids)
                    if directory_ids is not None
                    else None
                )
                _write_package(
                    target,
                    classes,
                    exported,
                    manifest,
                    resolver,
                    source_anchor=anchor,
                )
            finally:
                os.close(root_fd)
        else:
            _write_package(target, classes, exported, manifest, None)
    else:
        _write_package(target, classes, exported, manifest, None)

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
        for position, member in enumerate(payload["members"], start=1):
            digest = member.get("sha256")
            label = member.get("label")
            source = member.get("source")
            digest_problem = invalid_digest_reason(digest)
            if digest_problem is not None:
                raise ExportError(
                    f"split plan {plan.get('name')!r} is corrupted: "
                    f"{set_name} member {position} has an invalid 'sha256' "
                    f"identity {digest!r}: {digest_problem}"
                )
            if not isinstance(label, str):
                raise ExportError(
                    f"split plan {plan.get('name')!r} is corrupted: "
                    f"{set_name} member {position} (sha256 {digest}) has a "
                    "non-string label"
                )
            if not isinstance(source, str) or not source:
                raise ExportError(
                    f"split plan {plan.get('name')!r} is corrupted: "
                    f"{set_name} member {position} (sha256 {digest}) has no "
                    "recorded source path"
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
# identity (device, inode) of the proved object actually hashed during
# the lookup.  The copy opens that exact inode pinned through the open,
# so a swap after the lookup — another regular file, even with
# identical content, or a symlink, even one pointing at the original
# file — cannot be read in its place; swaps landing in the gap inside
# the lookup itself are rejected by the scan before the digest is
# recorded.
Resolver = dict[str, tuple[Path, int, int]]


class _ScanResult(dict):
    """The digest resolver, also carrying every real directory seen.

    Behaves as the plain digest-to-file resolver mapping, with an extra
    ``directory_ids`` attribute: each key is a ``/``-separated path
    relative to the root (``""`` for the root itself) and each value is
    ``(display path, device, inode)`` as observed during the walk.
    """

    def __init__(
        self,
        files: dict[str, tuple[Path, int, int]],
        directories: dict[str, tuple[Path, int, int]],
    ) -> None:
        super().__init__(files)
        self.directory_ids = directories


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

    Each file read is itself pinned through the shared
    inspect / open-no-follow-with-nonblocking / fstat-prove rule: the
    bytes hashed come from the exact regular file the walk typed, and
    an object swapped onto the name in that gap — a named pipe (even
    with no writer), a symlink or another regular file, even a
    byte-identical one — fails the export before any of its bytes are
    read, naming the path and what it became.

    Every step is anchored to directory descriptors rather than path
    strings: a child is typed and entered with ``fstatat``/``openat``
    relative to the descriptor of the directory it was listed from, and
    each entered directory is pinned for the whole time its contents are
    processed.  A real subdirectory whose path is swapped for a symlink
    after it was listed but before or while it is entered therefore can
    never be followed — the open refuses the link and the export fails
    naming that directory — and a swap landing while its children are
    being read is detected when its path binding is re-checked after the
    children have been processed.  Only entries that are already
    symlinks when first listed are skipped, preserving the original
    convention.

    The descent uses an explicit stack of pinned directory frames
    rather than recursive calls: a directory chain deeper than the
    interpreter's recursion limit is walked in full — files at the root,
    on side branches and at the end of the chain all hashed and offered
    as matches — and a confirmed directory's descriptor stays open for
    exactly as long as its subtree is visited, as with a recursive
    walk.  Each frame carries its display and ``/``-relative path as
    plain strings, so a deep name never has to be rendered through a
    deep Python call chain.
    """
    try:
        root_fd = os.open(root, os.O_RDONLY | _O_DIRECTORY)
    except OSError as error:
        raise ExportError(f"cannot scan directory {root}: {error}") from error
    try:
        return _scan_tree(root_fd, root)
    finally:
        os.close(root_fd)


def _scan_tree(root_fd: int, root: Path) -> _ScanResult:
    """Walk the tree reached through the already-open, pinned ``root_fd``.

    The descent runs on an explicit stack of frames rather than Python
    call frames: a chain of directories deeper than the interpreter's
    recursion limit is searched in full — files at the root, on side
    branches and at the end of a long chain are all found — instead of
    aborting partway with a ``RecursionError``.  A confirmed directory's
    descriptor stays open for exactly as long as its subtree is visited,
    just as with a recursive walk, and every descriptor this walk owns is
    closed when its frame finishes or when any step fails.

    Display paths and ``/``-separated relative paths are carried as
    incrementally built strings on the frames.  They are never derived
    from a deep :class:`~pathlib.Path` while Python frames are stacked
    deep (even stringifying such a path would exhaust the remaining
    call-stack allowance); a :class:`~pathlib.Path` is wrapped around a
    frame's string only at the fixed shallow depth of the helper that
    needs it for an error message.
    """
    found: dict[str, tuple[str, Path, int, int]] = {}
    directories: dict[str, tuple[Path, int, int]] = {}
    try:
        root_status = os.fstat(root_fd)
    except OSError as error:
        raise ExportError(f"cannot scan directory {root}: {error}") from error
    directories[""] = (root, root_status.st_dev, root_status.st_ino)

    root_display = os.fspath(root)
    # One frame per directory being processed, deepest last:
    # (descriptor, parent descriptor or None for the root, basename in
    # the parent or None, display path, '/'-relative key, entry basenames
    # left to visit).  The names are reversed so pop() visits scandir
    # order; each name is typed (fstatat, no symlink follow) only when
    # popped, matching the former recursive loop: a name that is a
    # symlink the first time the walk actually inspects it is skipped,
    # while a name already confirmed as a real directory is pinned
    # through :func:`_enter_subdirectory`.  The root frame owns no
    # descriptor: ``root_fd`` is closed by the caller.
    frames: list[tuple[int, int | None, str | None, str, str, list[str]]] = [
        (root_fd, None, None, root_display, "", _list_entry_names(root_fd, root_display))
    ]
    try:
        while frames:
            (
                directory_fd,
                parent_fd,
                dir_name,
                display,
                rel,
                entry_names,
            ) = frames[-1]
            if not entry_names:
                frames.pop()
                if parent_fd is not None:
                    # All children were reached through this pinned
                    # descriptor.  Re-prove the parent-relative name still
                    # binds to this same directory before releasing it: a
                    # symlink (or anything else) swapped onto it while the
                    # children were being processed must fail the export
                    # even though nothing was read through that link.
                    try:
                        _assert_still_same_directory(
                            parent_fd, dir_name, Path(display), directory_fd
                        )
                    except BaseException:
                        os.close(directory_fd)
                        raise
                    os.close(directory_fd)
                continue
            child_name = entry_names.pop()
            child_display = f"{display}/{child_name}"
            child_rel = f"{rel}/{child_name}" if rel else child_name
            try:
                status = os.stat(
                    child_name, dir_fd=directory_fd, follow_symlinks=False
                )
            except OSError as error:
                raise ExportError(
                    f"cannot inspect entry {child_display}: {error}"
                ) from error
            mode = status.st_mode
            if stat.S_ISLNK(mode):
                # A symlink when first encountered: skipped, never read.
                continue
            if stat.S_ISREG(mode):
                digest, device, inode = _hash_regular_file_at(
                    directory_fd, child_name, Path(child_display)
                )
                current = found.get(digest)
                if current is None or child_rel < current[0]:
                    found[digest] = (
                        child_rel,
                        Path(child_display),
                        device,
                        inode,
                    )
            elif stat.S_ISDIR(mode):
                child_fd = _enter_subdirectory(
                    directory_fd, child_name, Path(child_display)
                )
                try:
                    try:
                        pinned = os.fstat(child_fd)
                    except OSError as error:
                        raise ExportError(
                            f"cannot scan directory {child_display}: {error}"
                        ) from error
                    child_names = _list_entry_names(child_fd, child_display)
                except BaseException:
                    os.close(child_fd)
                    raise
                directories[child_rel] = (
                    Path(child_display),
                    pinned.st_dev,
                    pinned.st_ino,
                )
                frames.append(
                    (
                        child_fd,
                        directory_fd,
                        child_name,
                        child_display,
                        child_rel,
                        child_names,
                    )
                )
            # Any other entry type is skipped as before.
    except BaseException:
        # Mid-walk failure: release every confirmed descendant
        # descriptor still held by an unfinished frame (each exactly
        # once); the root descriptor itself belongs to the caller.
        for frame in frames:
            if frame[1] is not None:
                os.close(frame[0])
        raise
    files = {
        digest: (path, device, inode)
        for digest, (_, path, device, inode) in found.items()
    }
    return _ScanResult(files, directories)


def _list_entry_names(directory_fd: int, display: str) -> list[str]:
    """List one pinned directory's entry basenames for stack processing.

    Returns the names in reversed listing order, so a frame's ``pop()``
    visits them in scandir order.  Only the names are read here; each
    entry is typed without following symlinks when it is actually
    visited, so a directory that cannot be listed fails the export
    naming its path and the reason.
    """
    try:
        names = [entry.name for entry in os.scandir(directory_fd)]
    except OSError as error:
        raise ExportError(f"cannot scan directory {display}: {error}") from error
    names.reverse()
    return names


def _enter_subdirectory(parent_fd: int, name: str, display: Path) -> int:
    """Open a listed subdirectory relative to ``parent_fd`` and pin it.

    The parent listed ``name`` as a real directory, so anything other
    than that same directory observed here means the path changed after
    the listing.  This is the first confirmation of the directory: no
    identity was recorded yet, so the fresh inspection itself sets the
    expectation the opened descriptor is then proved against.  A symlink
    swapped in at any point fails the export identifying ``display`` as
    a directory that changed into a symlink, and its target — outside
    the tree, inside it, or the moved original — is never touched.
    """
    return _open_confirmed_directory(
        parent_fd, name, display, expected=None, during_scan=True
    )


def _open_confirmed_directory(
    parent_fd: int,
    name: str,
    display: Path,
    expected: tuple[int, int] | None,
    during_scan: bool,
) -> int:
    """Open ``name`` relative to ``parent_fd`` and pin the directory.

    The single confirmation rule shared by the lookup and the package
    copy: the name is inspected with ``fstatat`` without following
    symlinks and then opened with ``openat`` using ``O_NOFOLLOW`` and
    ``O_DIRECTORY``; the descriptor is fstat()ed and proved to be the
    directory the inspection named.  ``expected`` carries the
    ``(device, inode)`` the binding must match — the identity recorded
    during the lookup when the copy re-checks an already confirmed
    directory, or ``None`` when the lookup confirms a directory for the
    first time and the fresh inspection itself is the expectation.

    ``during_scan`` selects the phase the error messages report:
    failures while walking the tree say "during scan" and "cannot scan",
    failures while copying the package say "during export" and "cannot
    inspect".  A symlink swapped in at any point in this sequence fails
    the export identifying ``display`` and what it became; the link's
    target is never touched.
    """
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        raise ExportError(f"cannot inspect directory {display}: {error}") from error
    if expected is None:
        expected = (before.st_dev, before.st_ino)
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISDIR(before.st_mode)
        or (before.st_dev, before.st_ino) != expected
    ):
        raise _directory_changed_message(
            display, before.st_mode, during_scan=during_scan
        )
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | _O_NOFOLLOW | _O_DIRECTORY,
            dir_fd=parent_fd,
        )
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.ENOTDIR):
            # The path changed between the fstatat and the open; inspect
            # it once more to report the concrete reason.
            raise _directory_changed_error(
                parent_fd, name, display, during_scan=during_scan
            ) from error
        raise _directory_access_error(display, error, during_scan) from error
    try:
        opened = os.fstat(descriptor)
    except OSError as error:
        os.close(descriptor)
        raise _directory_access_error(display, error, during_scan) from error
    if not stat.S_ISDIR(opened.st_mode) or (
        opened.st_dev,
        opened.st_ino,
    ) != expected:
        os.close(descriptor)
        raise _directory_changed_error(
            parent_fd, name, display, during_scan=during_scan
        )
    return descriptor


def _directory_access_error(
    display: Path, error: OSError, during_scan: bool
) -> ExportError:
    """Report a directory that cannot be entered, naming the phase."""
    action = "scan" if during_scan else "inspect"
    return ExportError(f"cannot {action} directory {display}: {error}")


def _assert_still_same_directory(
    parent_fd: int, name: str, display: Path, directory_fd: int
) -> None:
    """Re-check that ``name`` in the parent still names ``directory_fd``."""
    try:
        now = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        raise ExportError(f"cannot inspect directory {display}: {error}") from error
    try:
        pinned = os.fstat(directory_fd)
    except OSError as error:
        raise ExportError(f"cannot scan directory {display}: {error}") from error
    if (now.st_dev, now.st_ino) != (pinned.st_dev, pinned.st_ino):
        raise _directory_changed_message(
            display, now.st_mode, during_scan=True
        )


def _directory_changed_error(
    parent_fd: int, name: str, display: Path, during_scan: bool
) -> ExportError:
    """Build the change error, inspecting the swapped path once for the reason."""
    try:
        now = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        mode: int | None = now.st_mode
    except OSError:
        mode = None
    return _directory_changed_message(display, mode, during_scan=during_scan)


def _directory_changed_message(
    display: Path, mode: int | None, during_scan: bool
) -> ExportError:
    """Fail naming the directory that changed and what it became."""
    phase = "during scan" if during_scan else "during export"
    if mode is not None and stat.S_ISLNK(mode):
        reason = "is now a symlink"
    elif mode is not None and stat.S_ISDIR(mode):
        reason = "was replaced by a different directory"
    else:
        reason = "no longer points to the directory being scanned"
    return ExportError(f"directory changed {phase}: {display} {reason}")


class _SourceAnchor:
    """Pinned view of the confirmed source tree, used while copying.

    Holds the root descriptor for the whole export and the identity of
    every directory observed during the lookup.  Each verification
    re-derives directory bindings from descriptors (``fstatat`` and
    ``openat`` with ``O_NOFOLLOW``/``O_DIRECTORY``) instead of trusting
    path strings, so a real directory replaced by a symlink after the
    lookup — before a sample is opened, while it is read or between
    samples — fails the export even when the link leads to byte-equal
    content.  :meth:`verify_all` runs once more after every sample has
    been copied, so the package is never published when a binding
    changed.
    """

    def __init__(
        self,
        root_fd: int,
        root: Path,
        directories: dict[str, tuple[Path, int, int]],
    ) -> None:
        self.root_fd = root_fd
        self.root = root
        self.directories = directories

    def verify_file(self, source: Path) -> None:
        """Prove the root and all ancestor directories of ``source``."""
        parts = source.relative_to(self.root).parts[:-1]
        self._descend(tuple(parts))

    def verify_all(self) -> None:
        """Prove the root and every directory recorded during the lookup.

        A single descriptor-anchored traversal over the recorded
        parent/child structure opens each directory exactly once, so a
        tree of any depth costs one descent rather than one per
        directory.  Every open is an ``openat`` relative to the pinned
        parent, so a symlink swapped in at any level is refused.

        The traversal runs on an explicit stack rather than Python call
        frames, so a tree deeper than the interpreter's recursion limit
        is still verified in full: the finished package is never
        published when a binding changed at any depth.
        """
        self._check_root()
        children: dict[str, list[tuple[str, str]]] = {}
        for key in self.directories:
            if not key:
                continue
            parent_key, _, name = key.rpartition("/")
            children.setdefault(parent_key, []).append((name, key))
        for entries in children.values():
            entries.sort()
        # One frame per directory whose recorded children remain to
        # open: (pinned descriptor, display, child entries left).  A
        # descendant descriptor is released as soon as its subtree is
        # done, so the number held open is bounded by the current depth
        # of the traversal, never by the whole tree.  If an open raises
        # mid-traversal, every descendant descriptor still held is
        # released while unwinding (the root descriptor is owned by the
        # export and is never closed here).
        stack: list[tuple[int, Path, list[tuple[str, str]]]] = [
            (self.root_fd, self.root, list(children.get("", ())))
        ]
        try:
            while stack:
                parent_fd, display, pending = stack[-1]
                if not pending:
                    stack.pop()
                    if parent_fd is not self.root_fd:
                        os.close(parent_fd)
                    continue
                name, key = pending.pop()
                child_display = display / name
                descriptor = self._open_checked(
                    parent_fd, child_display.name, child_display,
                    *self.directories[key][1:],
                )
                stack.append(
                    (descriptor, child_display, list(children.get(key, ())))
                )
        except BaseException:
            for descriptor, _display, _pending in stack:
                if descriptor is not self.root_fd:
                    os.close(descriptor)
            raise

    def _check_root(self) -> None:
        _, root_dev, root_inode = self.directories[""]
        try:
            pinned_root = os.fstat(self.root_fd)
        except OSError as error:
            raise ExportError(
                f"cannot inspect directory {self.root}: {error}"
            ) from error
        if (pinned_root.st_dev, pinned_root.st_ino) != (
            root_dev,
            root_inode,
        ) or not stat.S_ISDIR(pinned_root.st_mode):
            raise _directory_changed_message(
                self.root, pinned_root.st_mode, during_scan=False
            )

    def _descend(self, parts: tuple[str, ...]) -> None:
        # Start from the root descriptor held open since before the
        # lookup: it pins the confirmed root inode regardless of later
        # path changes.  Only intermediate children are opened here, and
        # each is closed before returning (the root descriptor itself is
        # owned by the export and must not be closed).  The descent is a
        # plain loop with the '/'-relative key carried along, so a chain
        # deeper than the interpreter's recursion limit needs no Python
        # call frames (and no deep relative-path derivation) per level.
        self._check_root()
        cursor = self.root_fd
        display = self.root
        key = ""
        owned: list[int] = []
        try:
            for part in parts:
                display = display / part
                key = f"{key}/{part}" if key else part
                expected = self.directories.get(key)
                if expected is None:
                    raise ExportError(
                        f"directory changed during export: {display} is "
                        f"not part of the scanned tree"
                    )
                cursor = self._open_checked(cursor, part, display, *expected[1:])
                owned.append(cursor)
        finally:
            for descriptor in owned:
                os.close(descriptor)

    def _open_checked(
        self,
        parent_fd: int,
        name: str,
        display: Path,
        expected_dev: int,
        expected_inode: int,
    ) -> int:
        """Open one directory component, refusing any changed binding.

        This is the copy-phase half of the shared confirmation rule: the
        directory was already confirmed during the lookup, so the binding
        must still carry the recorded identity, and failures are reported
        as happening during the export rather than the scan.
        """
        return _open_confirmed_directory(
            parent_fd,
            name,
            display,
            expected=(expected_dev, expected_inode),
            during_scan=False,
        )


# Open-relative-to-a-directory-descriptor support (openat/fstatat with
# dir_fd, O_NOFOLLOW, O_DIRECTORY) is provided by the module-level
# _O_NOFOLLOW / _O_DIRECTORY flags above.


def _hash_regular_file_at(
    directory_fd: int, name: str, path: Path
) -> tuple[str, int, int]:
    """Read one directory-relative file in full, pinned through the open.

    The name is inspected with ``fstatat`` relative to the pinned
    directory descriptor and the exact regular file confirmed there is
    then opened with ``openat`` plus ``O_NOFOLLOW`` and ``O_NONBLOCK``;
    the descriptor is fstat()ed and proved to be that same object
    before any byte is read.  Closing the inspect/open gap this way
    means an object swapped in after the walk typed the entry — a
    named pipe, even one with no writer or one whose writer would
    supply the planned bytes, a symlink, or another regular file, even
    one with identical content, size and modification time — fails the
    export immediately: the open never waits for a writer, the swapped
    object's bytes are never hashed, and the error names this path.
    The recorded identity is the proved descriptor's
    ``(device, inode)`` — the object the bytes actually came from —
    which the copy later independently re-pins before streaming the
    package.  Any failure raises with the path and reason.  Returns
    ``(digest, device, inode)``.

    The confirm/open/prove rule and the chunked hashing come from
    :mod:`confirmed`, shared with import; only the export wording is
    assembled here.
    """
    try:
        confirmed_status = confirmed.inspect_regular(name, dir_fd=directory_fd)
        descriptor = confirmed.open_without_follow(name, dir_fd=directory_fd)
        # prove_opened_regular closes the descriptor on rejection; on
        # success it also clears the open's temporary O_NONBLOCK, so the
        # descriptor hash_descriptor streams is an ordinary blocking one.
        opened = confirmed.prove_opened_regular(descriptor, confirmed_status)
    except confirmed.ConfirmationError as failure:
        if failure.stage is confirmed.ConfirmStage.OPEN:
            raise _scan_open_error(
                failure, directory_fd, name, path, confirmed_status
            ) from failure
        raise _scan_file_error(failure, path) from failure
    try:
        digest = confirmed.hash_descriptor(descriptor)
    except OSError as error:
        raise ExportError(f"cannot read source file {path}: {error}") from error
    # The descriptor was proved to be the confirmed regular file, so
    # the identity the bytes carry is the opened object's own.
    return digest, opened.st_dev, opened.st_ino


def _scan_open_error(
    failure: confirmed.ConfirmationError,
    directory_fd: int,
    name: str,
    path: Path,
    confirmed_status: os.stat_result,
) -> ExportError:
    """Translate a failed pinned open, naming a swap the kernel refused.

    The open itself failing (something other than the ``ELOOP`` already
    classified as a symlink) usually means a plain access error, but it
    can also be the kernel refusing a swapped non-regular object (a
    Unix socket, for instance, fails with ``ENXIO`` before it could be
    fstat()ed through a descriptor).  Inspect the name once more,
    relative to the same pinned directory: if it no longer is the
    confirmed regular file, report the replacement and what sits there
    now instead of a generic open failure; otherwise keep the
    underlying error.
    """
    if failure.reason is not confirmed.ConfirmReason.ACCESS_ERROR:
        return _scan_file_error(failure, path)
    try:
        now = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError:
        now = None
    if now is None or (
        stat.S_ISREG(now.st_mode)
        and (now.st_dev, now.st_ino)
        == (confirmed_status.st_dev, confirmed_status.st_ino)
    ):
        return ExportError(f"cannot read source file {path}: {failure.error}")
    if stat.S_ISLNK(now.st_mode):
        return ExportError(
            f"source file changed before it was read: "
            f"{path} is now a symbolic link"
        )
    return _scan_file_changed(path, now)


def _scan_file_changed(
    path: Path, status: os.stat_result | None
) -> ExportError:
    """Fail a scan read naming the path that changed and its new occupant.

    The walk typed this name as a regular file; whatever occupies it at
    the pinned open is therefore a replacement, even when it carries
    identical content.  The message says so in terms of the object the
    kernel reported (another regular file, a named pipe, a symbolic
    link, ...), so a FIFO swapped into the confirm-to-open gap is
    reported as a replaced source rather than as a read problem.
    """
    kind = _confirmed_object_kind(status)
    if kind == "regular file":
        detail = "now names a different regular file than the one confirmed"
    else:
        detail = f"is now a {kind}, not the confirmed regular file"
    return ExportError(
        f"source file changed before it was read: {path} {detail}"
    )


def _scan_file_error(
    failure: confirmed.ConfirmationError, path: Path
) -> ExportError:
    """Translate a shared confirmation failure into the scan's wording.

    The walk had just typed this name as a regular file, so a symlink,
    a named pipe or any other object that occupies it while the open is
    pinned is a source that changed before it was read: the export fails
    naming the path and the new occupant, without ever waiting on the
    object or hashing its bytes.  Entries that were already
    non-regular when the walk first listed them never reach this point
    — they are skipped before any read is attempted.
    """
    if failure.reason is confirmed.ConfirmReason.NOW_SYMLINK:
        return ExportError(
            f"source file changed before it was read: "
            f"{path} is now a symbolic link"
        )
    if failure.stage is confirmed.ConfirmStage.INSPECT:
        if failure.reason is confirmed.ConfirmReason.ACCESS_ERROR:
            return ExportError(
                f"cannot inspect entry {path}: {failure.error}"
            )
        # A non-regular object at a name the walk had just typed as a
        # regular file: the entry changed before the pinned open.
        return _scan_file_changed(path, failure.status)
    if failure.reason is confirmed.ConfirmReason.WRONG_IDENTITY:
        # The non-blocking open let a swapped non-regular carrier
        # through (a named pipe, even with no writer at all) or the
        # opened descriptor names a different regular inode than the
        # one inspected; the descriptor's own fstat says which.
        return _scan_file_changed(path, failure.status)
    return ExportError(f"cannot read source file {path}: {failure.error}")


def _assert_target_name_free(target: Path) -> None:
    """Reject an occupied target name without following a final symlink.

    ``lstat`` inspects the name itself, so a symlink counts as occupied
    even when it points at a location that does not exist.  Only a proven
    ``ENOENT`` means the name is free; any other inspection error fails
    the export too, rather than risking a write at a name that cannot be
    shown to be free.
    """
    try:
        os.lstat(target)
    except FileNotFoundError:
        return
    except OSError as error:
        raise ExportError(f"cannot inspect target {target}: {error}") from error
    raise ExportError(f"target already exists: {target}")


def _publish_package(temp_path: Path, target: Path) -> None:
    """Publish the finished package without ever replacing the target name.

    The package is complete and source-verified by the time this runs.
    The name is re-checked and then claimed with a single ``link(2)``,
    which creates a new directory entry and fails with ``EEXIST`` when the
    name is occupied by anything at all — a regular file, a directory or
    a symlink, including one whose target does not exist.  A file another
    program creates at the last instant — even an empty one or one
    byte-identical to the package — is therefore never replaced, unlinked
    or written through.  The package lives in the same directory as the
    target, so the link is same-filesystem and a reader can only ever see
    the complete file; this run's temporary link is removed afterwards,
    leaving the occupant untouched on conflict.
    """
    _assert_target_name_free(target)
    try:
        os.link(temp_path, target)
    except FileExistsError:
        # The name was claimed in the gap after the check: link refused
        # it atomically, so the occupant — file or symlink of any kind —
        # is exactly as the other program left it.
        raise ExportError(f"target already exists: {target}") from None
    except OSError as error:
        raise ExportError(f"cannot write export package: {error}") from error
    # The target now carries the finished package under its own link.
    # Drop the temporary link; if even that fails the package is already
    # published, so the export still succeeds.
    try:
        os.unlink(temp_path)
    except OSError:
        pass


def _write_package(
    target: Path,
    classes: dict[str, str],
    members: list[dict[str, Any]],
    manifest: dict[str, Any],
    resolver: dict[str, tuple[Path, int, int]] | None = None,
    source_anchor: _SourceAnchor | None = None,
) -> None:
    """Write the ZIP atomically: lock the target name, stream to a temp file.

    The target is never overwritten: an occupied target is rejected both
    when the export starts and once the package is finished, and the
    finished package is claimed with a non-replacing ``link`` rather than
    a rename, so a file or symlink (even a dangling one) that another
    program creates while the images are copied is preserved and the
    export fails naming the target.  Streaming to a temp file until the
    whole package has been written, fsynced and source-verified means a
    failure or interruption leaves neither an incomplete target nor this
    run's temp file.  An flock on a lock file next to the target
    serializes concurrent exports to the same path.
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
        _assert_target_name_free(target)

        fd, temp_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=parent
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                _write_zip(
                    stream,
                    classes,
                    members,
                    manifest,
                    resolver,
                    source_anchor=source_anchor,
                )
                stream.flush()
                os.fsync(stream.fileno())
            if source_anchor is not None:
                # Final gate before the package is published: every
                # directory binding confirmed during the lookup must
                # still hold.  A directory turned into a symlink while
                # the last entry was streamed is caught here and the
                # temp file is discarded rather than published.
                source_anchor.verify_all()
            # Claim the name only now that the package is complete and
            # every source check passed; link refuses a name occupied at
            # the last instant instead of replacing its occupant.
            _publish_package(temp_path, target)
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
    source_anchor: _SourceAnchor | None = None,
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
            _write_sample(
                archive,
                arc_name,
                member,
                resolver,
                source_anchor=source_anchor,
            )
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
    source_anchor: _SourceAnchor | None = None,
) -> None:
    digest = member["sha256"]
    hasher = hashlib.sha256()
    if resolver is not None:
        # Source-directory export: read from the matched copy only.  The
        # plan's recorded source path is deliberately not consulted, so a
        # still-usable original location cannot paper over a missing
        # match.  First prove every ancestor directory is still the
        # directory confirmed during the lookup (anchored to descriptors,
        # not path strings), so a real directory replaced by a symlink
        # after the lookup cannot lead to the bytes below.  The opened
        # file itself is then pinned to the regular inode confirmed at
        # lookup time, so a file swap after that confirmation cannot be
        # copied in its place.
        try:
            source, expected_device, expected_inode = resolver[digest]
        except KeyError:
            raise ExportError(
                f"sample {digest} in set {member['set']}: no file with "
                f"matching content found under the source directory"
            ) from None
        if source_anchor is not None:
            source_anchor.verify_file(source)
        source_stream, source_size = _open_resolved_source(
            source, digest, expected_device, expected_inode
        )
    else:
        source = Path(member["source"])
        source_stream, source_size = _open_recorded_source(member, digest)

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
            f"sample {digest}: source file content changed or is corrupt "
            f"while reading {source}: expected digest {digest}, got {actual}"
        )


def _open_recorded_source(
    member: dict[str, Any], digest: str
) -> tuple[Any, int]:
    """Open the plan-recorded source, pinned to one confirmed regular file.

    The export reads the sample bytes from the path saved in the plan.
    That path may be a symbolic link when the export starts; as for
    single-file import the link is followed once and the actual file it
    names at that moment is the object confirmed for this read.  From
    that point on the read is pinned with the same
    inspect/open-no-follow/fstat-prove rule as a source-directory copy:

    * the concrete target is lstat()ed and must be a regular file;
    * it is opened with ``O_NOFOLLOW`` and ``O_NONBLOCK``, so a name
      swapped in the gap — a symlink or a named pipe with no writer —
      can neither redirect nor block the open;
    * the descriptor is fstat()ed and must be the exact regular file the
      inspection named, so another regular file renamed onto the path in
      that gap — even one with identical content, size and modification
      time — is rejected rather than read, and a named pipe or other
      non-regular object is rejected before any byte is waited on.

    A stream is returned only for the proved descriptor, together with
    that confirmed file's size; blocking semantics are restored before
    streaming, so large sources are still read in fixed chunks.
    """
    recorded = member["source"]
    display = Path(recorded)
    # Follow a symlink given (or recorded) at the start exactly once:
    # the confirmed object is the actual file the link names now.  A
    # strict resolution also turns a vanished path or an unsearchable
    # component into a plain access failure before anything is opened.
    try:
        target = os.path.realpath(recorded, strict=True)
    except OSError as error:
        raise ExportError(
            f"sample {digest}: cannot open recorded source file "
            f"{display}: {error}"
        ) from error
    try:
        opened_file = confirmed.confirm_and_open_regular(target)
    except confirmed.ConfirmationError as failure:
        raise _recorded_source_error(failure, display, digest) from failure
    descriptor = opened_file.descriptor
    try:
        return os.fdopen(descriptor, "rb"), opened_file.opened.st_size
    except OSError as error:
        os.close(descriptor)
        raise ExportError(
            f"sample {digest}: cannot read source file {display}: {error}"
        ) from error


def _confirmed_object_kind(status: os.stat_result | None) -> str:
    """Human word for what occupies a confirmed path."""
    if status is None:
        return "non-regular object"
    mode = status.st_mode
    if stat.S_ISLNK(mode):
        return "symbolic link"
    if stat.S_ISREG(mode):
        return "regular file"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISFIFO(mode):
        return "named pipe"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISCHR(mode):
        return "character device"
    if stat.S_ISBLK(mode):
        return "block device"
    return "non-regular file"


def _recorded_source_error(
    failure: confirmed.ConfirmationError, source: Path, digest: str
) -> ExportError:
    """Translate a recorded-source confirmation failure into export wording.

    Every message carries the sample's full digest, the path recorded in
    the plan and the concrete reason.  An object occupying the path at
    the first inspection that simply is not regular keeps the plain
    “not a regular file” wording (optionally naming what it is).  An
    object that replaced the confirmed regular file between that
    inspection and the proved open — another regular file, even a
    byte-identical one, a final symbolic link, or a named pipe caught
    without blocking — says the recorded source was replaced before it
    was read, naming what sits there now.  Underlying access failures
    keep their ``OSError`` text.
    """
    if failure.reason is confirmed.ConfirmReason.NOT_REGULAR:
        kind = _confirmed_object_kind(failure.status)
        return ExportError(
            f"sample {digest}: source is not a regular file ({kind}): {source}"
        )
    if failure.reason is confirmed.ConfirmReason.NOW_SYMLINK:
        return ExportError(
            f"sample {digest}: recorded source was replaced before it was "
            f"read: {source} is now a symbolic link"
        )
    if failure.reason is confirmed.ConfirmReason.WRONG_IDENTITY:
        kind = _confirmed_object_kind(failure.status)
        if kind == "regular file":
            detail = "now names a different regular file than the one confirmed"
        else:
            detail = f"is now a {kind}, not the confirmed regular file"
        return ExportError(
            f"sample {digest}: recorded source was replaced before it was "
            f"read: {source} {detail}"
        )
    action = {
        confirmed.ConfirmStage.INSPECT: "stat",
        confirmed.ConfirmStage.OPEN: "open",
        confirmed.ConfirmStage.PROVE: "read",
    }.get(failure.stage, "open")
    detail = failure.error if failure.error is not None else "unknown error"
    return ExportError(
        f"sample {digest}: cannot {action} source file {source}: {detail}"
    )


def _open_resolved_source(
    source: Path, digest: str, expected_device: int, expected_inode: int
) -> tuple[Any, int]:
    """Open the lookup-confirmed copy, pinned to its confirmed inode.

    The selected path can be remapped in the gap between the directory
    lookup and this copy.  The path is lstat()ed without following
    symlinks and must still be a regular file carrying the device/inode
    recorded at lookup; it is opened with ``O_NOFOLLOW`` and the
    descriptor is fstat()ed and proved to be that exact regular file, so
    the decision covers the object that is actually read, not just one
    pre-open look at the path.  A stream is returned only then, together
    with that confirmed file's size.

    A different regular file renamed onto the path — even one with
    identical content, size and modification time — or a symlink — even
    one pointing at the original file — fails with the sample's full
    digest, the selected source path and the reason.  Once the
    descriptor is open this function is done: unlike import, export does
    not re-check the path after the read, so the path being occupied by
    another file cannot stop the already pinned original from being
    read.
    """
    try:
        opened_file = confirmed.confirm_and_open_regular(
            source, expected=(expected_device, expected_inode)
        )
    except confirmed.ConfirmationError as failure:
        raise _resolved_source_error(failure, source, digest) from failure
    descriptor = opened_file.descriptor
    try:
        return os.fdopen(descriptor, "rb"), opened_file.opened.st_size
    except OSError as error:
        os.close(descriptor)
        raise ExportError(
            f"sample {digest}: cannot read source file {source}: {error}"
        ) from error


def _resolved_source_error(
    failure: confirmed.ConfirmationError, source: Path, digest: str
) -> ExportError:
    """Translate a shared confirmation failure into the export wording.

    Replacements (a different object at the selected path, at the
    inspection or when the descriptor is proved) say “replaced after
    lookup”; a final symlink additionally says so.  Plain access
    failures keep their underlying ``OSError`` and “cannot read/stat”
    wording, so an unreadable or vanished selected copy reports exactly
    what it did before.
    """
    replaced = (
        failure.reason is confirmed.ConfirmReason.WRONG_IDENTITY
        or failure.reason is confirmed.ConfirmReason.NOW_SYMLINK
    )
    if replaced:
        if failure.reason is confirmed.ConfirmReason.NOW_SYMLINK:
            return ExportError(
                f"sample {digest}: source file was replaced after lookup: "
                f"{source} is now a symlink"
            )
        return ExportError(
            f"sample {digest}: source file was replaced after lookup: {source}"
        )
    if failure.reason is confirmed.ConfirmReason.NOT_REGULAR:
        return ExportError(
            f"sample {digest}: source is not a regular file: {source}"
        )
    action = "stat" if failure.stage is confirmed.ConfirmStage.INSPECT else "read"
    return ExportError(
        f"sample {digest}: cannot {action} source file {source}: {failure.error}"
    )


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
