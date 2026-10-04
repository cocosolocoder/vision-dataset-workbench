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
from typing import Any

BATCH_SCHEMA_VERSION = 1


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
# Submission rules (pure: no workspace I/O, no mutation)
#
# These functions decide *whether* a submission may be committed and *what*
# a commit would change.  The store is responsible only for supplying the
# current manifest state and for applying the returned plan and persisting
# it, so the rejection rules stay independent of the journal and lock.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordPlan:
    """One record's outcome decided against the current manifest.

    ``current`` is the sample's normalized current label; ``changed`` says
    whether applying the record would rewrite that sample's label.  The
    record is rejected unless ``current == record["old"]``.
    """

    record: dict[str, Any]
    current: str | None
    changed: bool


def reject_unknown_samples(
    records: list[dict[str, Any]], known_digests: Any
) -> None:
    """Reject the whole batch if any record names an unregistered sample.

    ``known_digests`` is any container of the digests currently in the
    manifest.  Records are checked in file order, so the first unknown
    digest is the one named in the error.
    """
    for record in records:
        if record["sha256"] not in known_digests:
            raise BatchError(f"sample not found: {record['sha256']}")


def check_number_available(
    number: str,
    records: list[dict[str, Any]],
    existing: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Apply the same-number rule.

    Return the earlier entry when ``existing`` carries the same content so
    the caller can replay its result, raise on a number used with different
    content, and return ``None`` when the number is free.  An undone batch
    keeps its number, so neither case can re-activate it.
    """
    if existing is None:
        return None
    if content_key(existing["records"]) == content_key(records):
        return existing
    raise BatchError(
        f"batch number {number!r} is already used with different content"
    )


def plan_records(
    records: list[dict[str, Any]], current_labels: dict[str, str | None]
) -> list[RecordPlan]:
    """Check every record's expected old label once and decide its outcome.

    Each current label is normalized and compared a single time: the same
    comparison establishes both the old-label precondition (``current``
    must equal ``old`` or the batch is rejected with the record located)
    and whether the record actually changes the sample (``current``
    differs from ``new``).  Nothing is mutated here.
    """
    plans: list[RecordPlan] = []
    for record in records:
        current = current_labels[record["sha256"]]
        if current != record["old"]:
            raise BatchError(
                f"change for {record['sha256']}: expected old label "
                f"{render_label(record['old'])!r} but current label is "
                f"{render_label(current)!r}"
            )
        plans.append(
            RecordPlan(record=record, current=current, changed=current != record["new"])
        )
    return plans


def build_history_entry(
    number: str, outcomes: list[tuple[RecordPlan, int]]
) -> dict[str, Any]:
    """Assemble the history entry for a verified batch that changes labels.

    ``outcomes`` pairs each planned record with the label revision its
    sample has right after this batch (the prior revision plus one for
    changed records, the unchanged revision for no-ops).  The entry shape
    is the on-disk format and must stay compatible with workspaces saved
    by earlier versions.
    """
    return {
        "batch": number,
        "records": [
            {
                "sha256": plan.record["sha256"],
                "old": plan.record["old"],
                "new": plan.record["new"],
                "changed": plan.changed,
                "rev": revision,
            }
            for plan, revision in outcomes
        ],
        "changed_count": sum(1 for plan, _ in outcomes if plan.changed),
        "undone": False,
        "undone_at": None,
    }


def result_no_changes(number: str, total: int) -> dict[str, Any]:
    """Result for a batch whose records change nothing (also an empty list)."""
    return {
        "batch": number,
        "status": "no-changes",
        "changed": 0,
        "unchanged": total,
        "total": total,
    }


def result_applied(number: str, plans: list[RecordPlan]) -> dict[str, Any]:
    """Result for a freshly committed batch."""
    changed = sum(1 for plan in plans if plan.changed)
    return {
        "batch": number,
        "status": "applied",
        "changed": changed,
        "unchanged": len(plans) - changed,
        "total": len(plans),
    }


def replay_result(entry: dict[str, Any]) -> dict[str, Any]:
    """Result returned when an identical batch number is submitted again."""
    changed = sum(1 for record in entry["records"] if record["changed"])
    return {
        "batch": entry["batch"],
        "status": "already-applied",
        "changed": changed,
        "unchanged": len(entry["records"]) - changed,
        "total": len(entry["records"]),
    }


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
