"""Batch label-update submission, validation and history.

A batch file is a UTF-8 JSON object of the form::

    {
      "batch": "non-empty batch number",
      "changes": [
        {"sha256": "<full SHA-256 digest>", "old": <string or null>,
         "new": <string or null>}
      ]
    }

``null`` and the empty string both mean "unlabeled"; any other string is
taken literally (including Chinese text and a class named ``unlabeled``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

BATCH_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BatchResolution:
    """One record's verdict, produced before anything is written.

    ``current`` is the label read once during verification and ``changed``
    is whether ``new`` differs from it.  A resolution never references the
    manifest item or its revision, so the decision stays independent of
    the later workspace save.
    """

    sha256: str
    old: str | None
    new: str | None
    current: str | None
    changed: bool


class BatchError(ValueError):
    """A batch submission or undo request is invalid."""


def normalize_label(label: Any) -> str | None:
    """Map the two unlabeled spellings to ``None``; keep strings literal.

    Raises :class:`BatchError` for anything that is not a string or ``None``.
    """
    if label is None or label == "":
        return None
    if isinstance(label, str):
        return label
    raise BatchError(f"label must be a string or null, got {label!r}")


def render_label(label: str | None) -> str:
    """Human-readable rendering for messages."""
    return "unlabeled" if label is None else label


def parse_batch_file(data: Any) -> tuple[str, list[dict[str, Any]]]:
    """Validate a parsed batch payload; return ``(number, records)``.

    Each record is ``{"sha256": str, "old": str | None, "new": str | None}``
    with labels normalized (``None`` for unlabeled).  Structural problems,
    an empty batch number and duplicate digests are all rejected here.
    """
    if not isinstance(data, dict):
        raise BatchError("batch file must contain a JSON object")
    number = data.get("batch")
    if not isinstance(number, str) or not number.strip():
        raise BatchError("batch number must be a non-empty string")
    changes = data.get("changes")
    if not isinstance(changes, list):
        raise BatchError("batch file must contain a 'changes' list")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, change in enumerate(changes):
        location = f"change #{index + 1}"
        if not isinstance(change, dict):
            raise BatchError(
                f"{location}: must be an object with 'sha256', 'old' and 'new'"
            )
        if not all(key in change for key in ("sha256", "old", "new")):
            raise BatchError(
                f"{location}: must contain 'sha256', 'old' and 'new' keys"
            )
        digest = change["sha256"]
        if not isinstance(digest, str) or not digest:
            raise BatchError(f"{location}: 'sha256' must be a non-empty string")
        if digest in seen:
            raise BatchError(f"{location}: duplicate sample {digest}")
        seen.add(digest)
        old = normalize_label(change["old"])
        new = normalize_label(change["new"])
        records.append({"sha256": digest, "old": old, "new": new})
    return number, records


def content_key(records: list[dict[str, Any]]) -> frozenset[tuple[str, str | None, str | None]]:
    """Order- and spelling-insensitive identity of a batch's content.

    Two submissions with the same number are considered the same batch
    when their record sets compare equal here: list order does not matter
    and ``null`` vs ``""`` does not matter.
    """
    return frozenset(
        (record["sha256"], record["old"], record["new"]) for record in records
    )


# ---------------------------------------------------------------------------
# Submission rules
#
# Each stage is a pure function of parsed records and the state passed to
# it; none of them mutate the manifest or the history.  The stages line up
# with the conditions that reject a submission, in their original order:
#
#   1. reject_unknown_samples  — records naming no registered sample
#   2. resolve_re_submission   — same-number history: replay or conflict
#   3. verify_records          — one current-label pass: old-label
#                                mismatches reject the whole batch; the
#                                surviving resolutions also decide which
#                                records actually change
#   4. apply_resolutions       — the only mutating stage (store layer)
#   5. build_history_entry / result_payload — result/history assembly
#
# "Current label" is read exactly once for the whole submission (stage 3),
# so verification and the change decision never process it twice.
# ---------------------------------------------------------------------------

# A current-label lookup: full digest -> normalized current label (or
# ``None`` when the digest names no registered sample).
CurrentLabels = Callable[[str], str | None]


def reject_unknown_samples(
    records: list[dict[str, Any]], known: Callable[[str], bool]
) -> None:
    """Reject the batch if any record names a sample that is not registered.

    The first unknown record is reported, in submission order.
    """
    for record in records:
        if not known(record["sha256"]):
            raise BatchError(f"sample not found: {record['sha256']}")


def resolve_re_submission(
    number: str,
    records: list[dict[str, Any]],
    existing: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Handle a batch number already present in history.

    Equal content (per :func:`content_key`) replays the first submission's
    result without touching the manifest — even when samples have since
    changed — while different content conflicts.  Returns the replay
    result, or ``None`` when the number is free.
    """
    if existing is None:
        return None
    if content_key(existing["records"]) == content_key(records):
        return replay_result(existing)
    raise BatchError(
        f"batch number {number!r} is already used with different content"
    )


def verify_records(
    records: list[dict[str, Any]], current_label: CurrentLabels
) -> list[BatchResolution]:
    """Verify every expected old label and classify each record, in one pass.

    Each record's current label is looked up once and used for both
    decisions: a current label differing from the expected old rejects the
    whole batch; otherwise the record is classified as changed only when
    the target label differs from that same current label.  No manifest
    item is modified here.
    """
    resolutions: list[BatchResolution] = []
    for record in records:
        current = current_label(record["sha256"])
        if current != record["old"]:
            raise BatchError(
                f"change for {record['sha256']}: expected old label "
                f"{render_label(record['old'])!r} but current label is "
                f"{render_label(current)!r}"
            )
        resolutions.append(
            BatchResolution(
                sha256=record["sha256"],
                old=record["old"],
                new=record["new"],
                current=current,
                changed=current != record["new"],
            )
        )
    return resolutions


def apply_resolutions(
    resolutions: list[BatchResolution],
    items: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Mutate manifest items for the resolved records and build history rows.

    This is the only stage that changes samples: records flagged
    ``changed`` receive their target label and a new revision; no-op
    records keep both.  Every row additionally records the revision after
    the batch, which later undos pin against.  The pre-change revision is
    read from each item here, once, rather than during verification.
    """
    applied: list[dict[str, Any]] = []
    for resolution in resolutions:
        item = items[resolution.sha256]
        revision = item_revision(item)
        if resolution.changed:
            item["label"] = resolution.new
            revision += 1
            item["rev"] = revision
        applied.append(
            {
                "sha256": resolution.sha256,
                "old": resolution.old,
                "new": resolution.new,
                "changed": resolution.changed,
                "rev": revision,
            }
        )
    return applied


def build_history_entry(
    number: str, applied_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Assemble the history row for a batch with at least one change."""
    return {
        "batch": number,
        "records": applied_records,
        "changed_count": sum(1 for record in applied_records if record["changed"]),
        "undone": False,
        "undone_at": None,
    }


def result_payload(
    number: str, status: str, applied_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build the user-facing result, keeping the original count meanings.

    ``changed`` counts records that actually changed a label,
    ``unchanged`` counts target-equals-current records and ``total``
    counts every submitted record.
    """
    changed = sum(1 for record in applied_records if record["changed"])
    return {
        "batch": number,
        "status": status,
        "changed": changed,
        "unchanged": len(applied_records) - changed,
        "total": len(applied_records),
    }


def no_changes_result(number: str, total: int) -> dict[str, Any]:
    """Result for a batch whose records all match their current labels."""
    return {
        "batch": number,
        "status": "no-changes",
        "changed": 0,
        "unchanged": total,
        "total": total,
    }


def replay_result(entry: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the original result from a stored history entry."""
    return result_payload(entry["batch"], "already-applied", entry["records"])


def item_revision(item: dict[str, Any]) -> int:
    """Read a sample's label revision, treating malformed stored values as 0."""
    value = item.get("rev", 0)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def describe_history_revision_problem(value: Any, *, present: bool) -> str | None:
    """Describe why a history record's ``rev`` is not a usable revision.

    Every batch-history sample record must carry ``rev`` as a genuine
    non-negative integer: the revision pinned for later undos.  Booleans
    (``True == 1``), decimals (``1.0 == 1``), numeric strings (``"1"``),
    ``null`` and a missing field are all rejected rather than coerced, so
    they can never compare equal to a sample's current revision.  Returns
    ``None`` when the value is valid.
    """
    if not present:
        return "missing 'rev' field"
    if isinstance(value, bool):
        return f"'rev' must be a non-negative integer, got boolean {value!r}"
    if not isinstance(value, int):
        return f"'rev' must be a non-negative integer, got {value!r}"
    if value < 0:
        return f"'rev' must be a non-negative integer, got negative {value}"
    return None


def require_intact_history_revisions(entry: Mapping[str, Any]) -> None:
    """Reject a batch entry whose records lack a sound pinned ``rev``.

    Every record — changed or not — must carry ``rev`` as a genuine
    non-negative integer, since undo pins each changed sample to the
    revision it had at the end of the batch.  This is integrity damage,
    unlike the *manifest's* optional revision (see
    :func:`item_revision`, which keeps the legacy missing-means-zero
    behaviour): a bad row is reported with the batch number, the sample's
    full digest and the exact ``rev`` problem, and nothing is repaired.

    Callers must run this before any success-shaped short-circuit (such as
    an already-undone batch), so corruption can never be masked by a
    repeated-undo success.
    """
    number = entry.get("batch")
    records = entry.get("records")
    if not isinstance(records, list):
        # validate_history() already rules this out; defend direct callers.
        raise BatchError(
            f"Batch history is corrupted: batch {number!r}: missing records"
        )
    for record in records:
        if not isinstance(record, dict):
            raise BatchError(
                f"Batch history is corrupted: batch {number!r}: malformed record"
            )
        problem = describe_history_revision_problem(
            record.get("rev"), present="rev" in record
        )
        if problem is not None:
            raise BatchError(
                f"Batch history is corrupted: batch {number!r}, "
                f"sample {record.get('sha256')}: {problem}"
            )


def empty_history() -> dict[str, Any]:
    return {"schema_version": BATCH_SCHEMA_VERSION, "batches": []}


def validate_history(data: Any) -> dict[str, Any]:
    """Validate a loaded batch history payload; return it unchanged."""
    if not isinstance(data, dict) or data.get("schema_version") != BATCH_SCHEMA_VERSION:
        raise BatchError("Unsupported batch history")
    batches = data.get("batches")
    if not isinstance(batches, list):
        raise BatchError("Malformed batch history")
    for entry in batches:
        if not isinstance(entry, dict):
            raise BatchError("Malformed batch history entry")
        if not isinstance(entry.get("batch"), str) or not entry["batch"].strip():
            raise BatchError("Malformed batch history entry: missing batch number")
        if not isinstance(entry.get("records"), list):
            raise BatchError("Malformed batch history entry: missing records")
        changed_count = entry.get("changed_count")
        if not isinstance(changed_count, int) or isinstance(changed_count, bool):
            raise BatchError("Malformed batch history entry: bad changed_count")
        if not isinstance(entry.get("undone"), bool):
            raise BatchError("Malformed batch history entry: bad undone flag")
        for record in entry["records"]:
            if not isinstance(record, dict):
                raise BatchError("Malformed batch history record")
            if not isinstance(record.get("sha256"), str):
                raise BatchError("Malformed batch history record: bad digest")
            for key in ("old", "new"):
                value = record.get(key)
                if value is not None and not isinstance(value, str):
                    raise BatchError("Malformed batch history record: bad label")
            if not isinstance(record.get("changed"), bool):
                raise BatchError("Malformed batch history record: bad changed flag")
    return data
