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
