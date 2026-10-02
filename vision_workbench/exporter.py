"""Portable offline export of a saved split plan.

An export package is a self-contained ZIP: one directory per set
(``train``/``validation``/``test``), each containing the same numbered
ImageFolder class directories, plus a UTF-8 JSON manifest.  Samples are
stored under ``<set>/<class>/<full-sha256><source-extension>``.

The export is driven solely by the saved plan: its sample identities,
labels, set assignments and recorded source paths.  Later imports, label
changes or undos never affect an exported package, and exporting never
writes to the workspace.  With ``source_dir``, sample content is read
from the files found under that directory (matched by content digest)
instead of the recorded source paths — useful after the original files
were moved or renamed — while everything the plan decides (identities,
labels, sets, package-internal names) stays unchanged.

ZIP bytes are deterministic for a given plan, options and source content:
entry metadata uses fixed values, entries are written in a fixed order and
source modification times are never consulted.
"""

from __future__ import annotations

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


def export_split(
    store: Any,
    plan_name: str,
    target: Path,
    skip_unlabeled: bool = False,
    source_dir: Path | None = None,
) -> dict[str, Any]:
    """Export a saved split plan to a new ZIP at ``target``.

    When ``source_dir`` is given, sample content is read from the files
    found under that directory (matched by content digest) instead of the
    source paths recorded in the plan; the plan still decides sample
    identities, labels, set assignments and package-internal names.

    Returns a result payload describing the plan, the number of samples
    actually exported, the class distribution per set and the per-set skip
    counts.  Raises :class:`ExportError` (a ``ValueError``) on any failure;
    the target path is never created in that case.
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

    if source_dir is not None:
        root = _check_source_dir(Path(source_dir))
        if exported:
            index = _index_source_dir(root)
            for member in exported:
                found = index.get(member["sha256"])
                if found is None:
                    raise ExportError(
                        f"sample {member['sha256']}: no file with matching "
                        f"content found under source directory {root}"
                    )
                member["read_from"] = found

    classes = _build_classes(exported)
    manifest = _build_manifest(plan, classes, exported, skipped_counts)
    distributions = _distributions(exported)

    _write_package(target, classes, exported, manifest)

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


def _check_source_dir(root: Path) -> Path:
    """Validate the ``--source-dir`` argument; return it unchanged."""
    try:
        stat_result = os.stat(root)
    except OSError as error:
        raise ExportError(
            f"cannot access source directory {root}: {error}"
        ) from error
    if not stat.S_ISDIR(stat_result.st_mode):
        raise ExportError(f"source directory {root}: not a directory")
    return root


def _index_source_dir(root: Path) -> dict[str, Path]:
    """Map content digest to one file per digest found under ``root``.

    Every regular file in the directory tree is hashed; symlinks and other
    non-regular files are skipped and symlinked directories are never
    entered.  When several files share a digest, the one whose path
    relative to ``root`` (with ``/`` separators) sorts first by Unicode
    code point wins, so the choice does not depend on file names beyond
    that ordering.  Any directory that cannot be scanned or any regular
    file that cannot be read in full fails the whole export.
    """
    index: dict[str, Path] = {}
    relative: dict[str, str] = {}

    def visit(directory: Path) -> None:
        try:
            with os.scandir(directory) as entries:
                entry_list = list(entries)
        except OSError as error:
            raise ExportError(
                f"cannot scan source directory {directory}: {error}"
            ) from error
        for entry in entry_list:
            entry_path = Path(entry.path)
            try:
                entry_stat = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise ExportError(
                    f"cannot scan source directory {directory}: "
                    f"cannot stat {entry_path}: {error}"
                ) from error
            if stat.S_ISLNK(entry_stat.st_mode):
                continue
            if stat.S_ISDIR(entry_stat.st_mode):
                visit(entry_path)
            elif stat.S_ISREG(entry_stat.st_mode):
                digest = _hash_file(entry_path)
                rel = entry_path.relative_to(root).as_posix()
                if digest not in relative or rel < relative[digest]:
                    relative[digest] = rel
                    index[digest] = entry_path
            # Anything else (devices, sockets, ...) is skipped.

    visit(root)
    return index


def _hash_file(path: Path) -> str:
    """Full SHA-256 of a regular file; any read failure aborts the export."""
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(_CHUNK_SIZE)
                if not chunk:
                    break
                hasher.update(chunk)
    except OSError as error:
        raise ExportError(
            f"cannot read file {path} under source directory: {error}"
        ) from error
    return hasher.hexdigest()


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


def _write_package(
    target: Path,
    classes: dict[str, str],
    members: list[dict[str, Any]],
    manifest: dict[str, Any],
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
                _write_zip(stream, classes, members, manifest)
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
            _write_sample(archive, arc_name, member)
        archive.writestr(_file_info("manifest.json"), _manifest_bytes(manifest))


def _directory_entries(classes: dict[str, str]) -> set[str]:
    entries = {f"{set_name}/" for set_name in SET_NAMES}
    for class_directory in classes.values():
        for set_name in SET_NAMES:
            entries.add(f"{set_name}/{class_directory}/")
    return entries


def _write_sample(
    archive: zipfile.ZipFile, arc_name: str, member: dict[str, Any]
) -> None:
    source = Path(member.get("read_from") or member["source"])
    digest = member["sha256"]
    try:
        stat_result = os.stat(source)
    except OSError as error:
        raise ExportError(f"sample {digest}: cannot stat source file {source}: {error}") from error
    if not stat.S_ISREG(stat_result.st_mode):
        raise ExportError(f"sample {digest}: source is not a regular file: {source}")

    info = _file_info(arc_name)
    hasher = hashlib.sha256()
    try:
        source_stream = source.open("rb")
    except OSError as error:
        raise ExportError(f"sample {digest}: cannot read source file {source}: {error}") from error
    try:
        with source_stream, archive.open(info, "w") as target_stream:
            while True:
                chunk = source_stream.read(_CHUNK_SIZE)
                if not chunk:
                    break
                hasher.update(chunk)
                target_stream.write(chunk)
    except OSError as error:
        raise ExportError(f"sample {digest}: cannot read source file {source}: {error}") from error

    actual = hasher.hexdigest()
    if actual != digest:
        raise ExportError(
            f"sample {digest}: source file content changed or is corrupt: "
            f"expected digest {digest}, got {actual}"
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
