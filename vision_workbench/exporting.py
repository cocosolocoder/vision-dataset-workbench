"""Offline export of a saved split plan into a classification data package.

The produced ZIP contains three set directories (``train``, ``validation``,
``test``), each laid out as an ImageFolder tree of category directories,
plus a UTF-8 JSON manifest describing the plan and every packaged sample.

Everything in the archive is derived solely from the saved plan: later
imports, batch label updates and undos never affect an export of an older
plan, and an export never writes back into the workspace.

The archive bytes are deterministic for the same plan, options and source
file contents.  In particular they do not depend on workspace location,
target path, file modification times or the export time.  Source media is
streamed one sample at a time, so memory use does not grow with the total
number of media bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable

from .splits import SET_NAMES

MANIFEST_ENTRY = "manifest.json"
PACKAGE_SCHEMA_VERSION = 1
_CHUNK_SIZE = 1024 * 1024


class ExportError(ValueError):
    """An export request failed (bad plan, unreadable source, bad target)."""


@dataclass(frozen=True)
class PlannedSample:
    digest: str
    label: str  # "" means unlabeled
    source: str


@dataclass(frozen=True)
class ExportPlan:
    name: str
    seed: int
    ratios: dict[str, str]
    members: dict[str, list[PlannedSample]]


def parse_saved_plan(data: Any) -> ExportPlan:
    """Validate and normalize a loaded split-plan JSON payload for export.

    Only the fields export depends on are examined; damaged or inconsistent
    plans raise :class:`ExportError` rather than producing a partial
    package.
    """
    if not isinstance(data, dict):
        raise ExportError("Split plan record is malformed")
    if data.get("schema_version") != 1:
        raise ExportError("Unsupported split plan schema version")
    name = data.get("name")
    if not isinstance(name, str):
        raise ExportError("Split plan record is missing its name")
    seed = data.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ExportError("Split plan record is missing its seed")
    ratios = data.get("ratios")
    if not isinstance(ratios, dict):
        raise ExportError("Split plan record is missing its ratios")
    ratio_map: dict[str, str] = {}
    for set_name in SET_NAMES:
        value = ratios.get(set_name)
        if not isinstance(value, str):
            raise ExportError(f"Split plan record is missing ratio for {set_name}")
        ratio_map[set_name] = value

    sets = data.get("sets")
    if not isinstance(sets, dict):
        raise ExportError("Split plan record is missing its set assignments")

    members: dict[str, list[PlannedSample]] = {}
    seen: set[str] = set()
    for set_name in SET_NAMES:
        set_payload = sets.get(set_name)
        if not isinstance(set_payload, dict):
            raise ExportError(f"Split plan record is missing set {set_name}")
        raw_members = set_payload.get("members")
        if not isinstance(raw_members, list):
            raise ExportError(f"Split plan record is missing members for {set_name}")
        planned: list[PlannedSample] = []
        for member in raw_members:
            if not isinstance(member, dict):
                raise ExportError(f"Split plan record has a bad member in {set_name}")
            digest = member.get("sha256")
            label = member.get("label")
            source = member.get("source")
            if (
                not isinstance(digest, str)
                or not digest
                or not isinstance(label, str)
                or not isinstance(source, str)
            ):
                raise ExportError(f"Split plan record has a bad member in {set_name}")
            if digest in seen:
                raise ExportError(
                    f"Split plan record lists {digest} in more than one set"
                )
            seen.add(digest)
            planned.append(PlannedSample(digest=digest, label=label, source=source))
        members[set_name] = planned
    return ExportPlan(name=name, seed=seed, ratios=ratio_map, members=members)


def build_category_map(labels: list[str]) -> dict[str, str]:
    """Assign each distinct label a unique, safe category directory name.

    Labels are arbitrary user strings (Chinese, separators, dots, trailing
    spaces).  Directories stay distinct as exact strings *and* after case
    folding (so labels differing only in case cannot collide on a
    case-insensitive filesystem), never equal a DOS device name, and never
    equal ``.``/``..`` or contain a path separator.  Collisions get a
    content-derived suffix; the mapping is a pure function of the sorted
    label set, hence stable across workspaces and runs.
    """
    result: dict[str, str] = {}
    used: set[str] = set()
    for label in sorted(set(labels)):
        candidate = _safe_dir_name(label)
        if candidate in used or candidate.casefold() in used or _is_reserved(candidate):
            digest = hashlib.sha256(
                b"vision-workbench-category\x00" + label.encode("utf-8")
            ).hexdigest()
            stem = candidate if candidate != "_" else "cat"
            length = 10
            candidate = f"{stem}-{digest[:length]}"
            # The suffixed name could in principle meet an earlier entry
            # (or a reserved device name); lengthen the digest until unique.
            while (
                candidate in used
                or candidate.casefold() in used
                or _is_reserved(candidate)
            ):
                length += 2
                candidate = f"{stem}-{digest[:length]}"
        used.add(candidate)
        used.add(candidate.casefold())
        result[label] = candidate
    return result


_DOS_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def _is_reserved(name: str) -> bool:
    return name.split(".", 1)[0].casefold() in _DOS_RESERVED


def safe_extension(source: str) -> str:
    """Return the source file's extension, constrained to stay in its set.

    The extension is taken from the basename only and reduced to a small
    charset, so it can never contain a path separator (including a
    backslash, which Windows extraction treats as one) or control bytes.
    Empty when the source name has no extension.
    """
    extension = os.path.splitext(os.path.basename(source))[1]
    cleaned = "".join(
        char
        for char in extension
        if char.isascii() and (char.isalnum() or char in "._-")
    )
    # splitext keeps the leading dot; names like ".jpg" yield no extension.
    if not cleaned.startswith(".") or cleaned in (".", ".."):
        return ""
    return cleaned[:32]


def _safe_dir_name(label: str) -> str:
    """Best-effort readable directory name for one label.

    Returned names contain no path separators or control characters, are
    not ``.``/``..``, and have their surrounding spaces/trailing dots
    trimmed (Windows would strip those anyway).  A label with nothing
    representable left maps to the reserved placeholder ``_`` (which gets a
    disambiguating suffix from the caller if more than one maps there).
    """
    chars: list[str] = []
    for char in label:
        category = unicodedata.category(char)
        if (
            char in '/\\:*?"<>|'
            or category.startswith("C")  # control/format/surrogate
            or char == "\u007f"
        ):
            chars.append("_")
        else:
            chars.append(char)
    name = "".join(chars).strip().rstrip(".").strip()
    return name if name else "_"


def export_package(
    plan: ExportPlan,
    target: Path | str,
    *,
    skip_unlabeled: bool = False,
) -> dict[str, Any]:
    """Export ``plan`` to a ZIP file at ``target``.

    Returns a JSON-serializable summary naming the plan, the total number
    of samples actually written, the per-set category distribution and the
    number of skipped unlabeled samples.

    Raises :class:`ExportError` when the target already exists, any sample
    is unlabeled without ``skip_unlabeled``, a source file is missing, is
    not a regular file or cannot be read, or its bytes do not hash to the
    planned digest.  No target file is created in any of those cases.
    """
    target = Path(target)

    # Fast, pre-work rejection of an existing target (including a symlink,
    # even a broken one).  The final hard-link install is the authoritative
    # atomic check.
    if target.exists() or target.is_symlink():
        raise ExportError(f"export target already exists: {target}")

    all_samples = [
        sample for set_name in SET_NAMES for sample in plan.members[set_name]
    ]

    unlabeled = sorted(
        sample.digest for sample in all_samples if sample.label == ""
    )
    if unlabeled and not skip_unlabeled:
        raise ExportError(
            "plan contains unlabeled samples; refusing to export "
            f"(first: {unlabeled[0]}); use --skip-unlabeled to skip them"
        )

    included: dict[str, list[PlannedSample]] = {}
    skipped_counts: dict[str, int] = {}
    for set_name in SET_NAMES:
        members = plan.members[set_name]
        included[set_name] = [sample for sample in members if sample.label != ""]
        skipped_counts[set_name] = len(members) - len(included[set_name])

    labels = sorted(
        {
            sample.label
            for samples in included.values()
            for sample in samples
        }
    )
    category_by_label = build_category_map(labels)

    # Verify every source and hash its bytes before the target is created,
    # so a missing or mismatched file never produces an archive.  Bytes are
    # re-hashed while streaming, so a file changing between this pass and
    # the write cannot pass off altered content under the old digest.
    sizes: dict[str, int] = {}
    for set_name in SET_NAMES:
        for sample in included[set_name]:
            sizes[sample.digest] = _verify_source(sample)

    manifest_payload = _build_manifest(
        plan, included, category_by_label, skipped_counts
    )
    manifest_bytes = (
        json.dumps(manifest_payload, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n"
    ).encode("utf-8")

    entries: list[tuple[str, str, Any]] = [(MANIFEST_ENTRY, "manifest", manifest_bytes)]
    for set_name in SET_NAMES:
        entries.append((f"{set_name}/", "dir", None))
    for label in labels:
        category = category_by_label[label]
        for set_name in SET_NAMES:
            entries.append((f"{set_name}/{category}/", "dir", None))
    for set_name in SET_NAMES:
        for sample in included[set_name]:
            entries.append((_sample_arcname(set_name, category_by_label, sample), "sample", sample))
    ordered = [entries[0], *sorted(entries[1:], key=lambda entry: entry[0])]

    def write_archive(stream: BinaryIO) -> None:
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for arcname, kind, payload in ordered:
                info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3  # Unix
                info.compress_type = zipfile.ZIP_DEFLATED
                if kind == "dir":
                    info.external_attr = (0o755 << 16) | 0x10  # MS-DOS directory flag
                    archive.writestr(info, b"")
                elif kind == "manifest":
                    info.external_attr = 0o644 << 16
                    archive.writestr(info, payload)
                else:
                    info.external_attr = 0o644 << 16
                    # The temp file is a regular seekable file, so zipfile
                    # rewrites the local header with the real sizes on
                    # close; media itself is still written one chunk at a
                    # time instead of being buffered in memory.
                    size, digest = _stream_member(archive, info, payload)
                    if size != sizes[payload.digest] or digest != payload.digest:
                        raise ExportError(
                            f"source for {payload.digest} changed while reading; "
                            "export aborted"
                        )

    try:
        _install_target(target, write_archive)
    except ExportError:
        raise
    except OSError as error:
        raise ExportError(f"cannot write export archive: {error}") from error

    distributions: dict[str, dict[str, int]] = {}
    for set_name in SET_NAMES:
        counts = {label: 0 for label in labels}
        for sample in included[set_name]:
            counts[sample.label] += 1
        distributions[set_name] = {label: counts[label] for label in labels}

    total = sum(len(samples) for samples in included.values())
    return {
        "plan": plan.name,
        "target": os.fspath(target),
        "exported": total,
        "sets": {
            set_name: {
                "exported": len(included[set_name]),
                "distribution": distributions[set_name],
                "skipped": skipped_counts[set_name],
            }
            for set_name in SET_NAMES
        },
        "skipped": dict(skipped_counts),
    }


def _sample_arcname(
    set_name: str, category_by_label: dict[str, str], sample: PlannedSample
) -> str:
    return (
        f"{set_name}/{category_by_label[sample.label]}/"
        f"{sample.digest}{safe_extension(sample.source)}"
    )


def _verify_source(sample: PlannedSample) -> int:
    """Open the saved source once and prove its bytes match the digest."""
    path = Path(sample.source)
    try:
        # Follow links: a symlink to a regular file is still a readable
        # regular file, while a broken link is "does not exist" and a link
        # to a directory/FIFO is "not a regular file".
        info = path.stat()
    except OSError as error:
        raise ExportError(
            f"source file for {sample.digest} does not exist: {sample.source}"
        ) from error
    if not stat.S_ISREG(info.st_mode):
        raise ExportError(
            f"source for {sample.digest} is not a regular file: {sample.source}"
        )

    hasher = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(_CHUNK_SIZE)
                if not chunk:
                    break
                hasher.update(chunk)
                size += len(chunk)
    except OSError as error:
        raise ExportError(f"cannot read source for {sample.digest}: {error}") from error
    if hasher.hexdigest() != sample.digest:
        raise ExportError(
            f"source content for {sample.digest} does not match its saved digest "
            f"(actual: {hasher.hexdigest()})"
        )
    return size


def _stream_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    sample: PlannedSample,
    force_zip64: bool = False,
) -> tuple[int, str]:
    """Write one source file into the archive without buffering it fully."""
    hasher = hashlib.sha256()
    size = 0
    try:
        with archive.open(info, "w", force_zip64=force_zip64) as member, Path(
            sample.source
        ).open("rb") as stream:
            while True:
                chunk = stream.read(_CHUNK_SIZE)
                if not chunk:
                    break
                hasher.update(chunk)
                size += len(chunk)
                member.write(chunk)
    except OSError as error:
        raise ExportError(
            f"cannot read source for {sample.digest}: {error}"
        ) from error
    return size, hasher.hexdigest()


def _build_manifest(
    plan: ExportPlan,
    included: dict[str, list[PlannedSample]],
    category_by_label: dict[str, str],
    skipped_counts: dict[str, int],
) -> dict[str, Any]:
    classes = sorted(category_by_label.values())
    class_index = {directory: index for index, directory in enumerate(classes)}
    labels_by_directory = {
        category_by_label[label]: label for label in category_by_label
    }

    samples: list[dict[str, Any]] = []
    set_summaries: list[dict[str, Any]] = []
    for set_name in SET_NAMES:
        counts = {directory: 0 for directory in classes}
        for sample in included[set_name]:
            counts[category_by_label[sample.label]] += 1
            samples.append(
                {
                    "path": _sample_arcname(set_name, category_by_label, sample),
                    "set": set_name,
                    "sha256": sample.digest,
                }
            )
        set_summaries.append(
            {
                "name": set_name,
                "ratio": plan.ratios[set_name],
                "samples": len(included[set_name]),
                "skipped_unlabeled": skipped_counts[set_name],
                # Counts keyed by the original label; zero entries mark
                # classes with no sample in this set.
                "distribution": {
                    labels_by_directory[directory]: counts[directory]
                    for directory in classes
                },
            }
        )
    samples.sort(key=lambda item: item["path"])

    return {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "plan": plan.name,
        "seed": plan.seed,
        "ratios": {set_name: plan.ratios[set_name] for set_name in SET_NAMES},
        "sets": set_summaries,
        # One shared class table for every set: directory name, numeric
        # index (ImageFolder-style) and the original label it encodes.
        "classes": [
            {
                "index": class_index[directory],
                "directory": directory,
                "label": labels_by_directory[directory],
            }
            for directory in classes
        ],
        "skipped_unlabeled": {
            set_name: skipped_counts[set_name] for set_name in SET_NAMES
        },
        "samples_total": len(samples),
        "samples": samples,
    }


def _install_target(target: Path, write: Callable[[BinaryIO], None]) -> None:
    """Write to a uniquely-named temp file, then atomically link it in.

    The destination only ever appears once the archive is complete and
    fsynced: ``os.link`` refuses to overwrite, so when two exports race for
    one target exactly one link succeeds.  A failure or interruption leaves
    only a removable temp file, never a partial target, and stale temps
    have unique names so a re-run when no target exists always succeeds.
    """
    parent = target.parent if str(target.parent) else Path(".")
    parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        dir=parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            raise ExportError(f"export target already exists: {target}") from None
        _fsync_directory(parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    temporary.unlink(missing_ok=True)


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
