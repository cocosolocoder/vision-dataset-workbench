"""Batch label modifications.

A batch is a client-submitted JSON document naming a batch id and a list
of label changes; every change identifies one imported sample by its full
SHA-256 digest and states the label it expects to find there plus the
label it wants instead.  ``None`` and the empty string both mean
"unlabeled" (matching :func:`vision_workbench.splits.category_key`), any
other string is taken literally, including a real class named
``"unlabeled"``.

Batches are validated *before* anything is written: an unknown digest, a
duplicated digest, a bad label value, a malformed document, or an expected
label that does not match the sample's current label rejects the whole
batch without touching any sample.  Resubmitting an already-applied batch
under the same id and with the same content returns the original result;
the same id with different content is an id conflict.

Every batch that actually changed a label leaves an undo marker
(``labels_last_batch``) on each changed sample.  A batch can only be
undone while none of those samples has been touched again, so undoing is
an all-or-nothing restoration.
"""

from __future__ import annotations

import hashlib
from typing import Any

# Manifest schema that first records label-batch history.  Schema 1
# manifests (older workspaces) upgrade transparently on the next write.
LABEL_SCHEMA_VERSION = 2


class BatchError(ValueError):
    """A batch document is invalid or cannot be applied/undone.

    ``code`` names the rejection category and ``records`` points at the
    entries responsible (indices into the submitted change list and/or
    digests), so callers can report where the problem is.
    """

    def __init__(
        self,
        message: str,
        code: str = "invalid_batch",
        records: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.records = records or []


def normalize_label(label: Any) -> str | None:
    """Map a label value to its stored form.

    ``None`` and ``""`` both collapse to ``None`` ("unlabeled"); every
    other string is kept verbatim.  Anything else is rejected.
    """
    if label is None:
        return None
    if isinstance(label, str):
        return label or None
    raise BatchError(
        f"Label must be a string or null, got {type(label).__name__}",
        code="invalid_label",
    )


def labels_equal(left: Any, right: Any) -> bool:
    """Whether two stored-or-submitted labels describe the same category.

    The two unlabeled spellings (``None`` and ``""``) compare equal.
    """
    return normalize_label(left) == normalize_label(right)


def content_fingerprint(changes: list[dict[str, Any]]) -> str:
    """Order- and unlabeled-spelling-independent fingerprint of changes.

    Reordering the change list or writing ``null`` where the original had
    ``""`` (or vice versa) does not change batch content.
    """
    normalized = sorted(
        (
            change["sha256"],
            normalize_label(change.get("old_label")) or "",
            normalize_label(change.get("new_label")) or "",
        )
        for change in changes
    )
    joined = "\n".join(f"{d}\x1f{o}\x1f{n}" for d, o, n in normalized)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def parse_batch_document(payload: Any) -> tuple[str, list[dict[str, Any]]]:
    """Validate the structure of a submitted batch document.

    Returns ``(batch_id, changes)`` where each change is normalized to
    ``{"sha256", "old_label", "new_label"}`` with labels in stored form.
    Raises :class:`BatchError` with record locations on any structural or
    value problem; nothing past the first problem is reported.
    """
    if not isinstance(payload, dict):
        raise BatchError(
            "Batch document must be a JSON object", code="invalid_document"
        )
    batch_id = payload.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id.strip():
        raise BatchError(
            "Batch id must be a non-empty string", code="invalid_batch_id"
        )

    changes = payload.get("changes")
    if not isinstance(changes, list):
        raise BatchError(
            "Batch document needs a 'changes' list", code="invalid_document"
        )

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, change in enumerate(changes):
        location = {"index": index}
        if not isinstance(change, dict):
            raise BatchError(
                f"Change #{index} must be an object",
                code="invalid_change",
                records=[location],
            )
        digest = change.get("sha256")
        if not isinstance(digest, str) or not digest:
            raise BatchError(
                f"Change #{index} needs a non-empty 'sha256' string",
                code="invalid_digest",
                records=[location],
            )
        location["sha256"] = digest
        if digest in seen:
            raise BatchError(
                f"Sample {digest} appears more than once in the batch",
                code="duplicate_digest",
                records=[location],
            )
        if "old_label" not in change or "new_label" not in change:
            raise BatchError(
                f"Change #{index} needs both 'old_label' and 'new_label'",
                code="invalid_change",
                records=[location],
            )
        try:
            old_label = normalize_label(change["old_label"])
            new_label = normalize_label(change["new_label"])
        except BatchError as error:
            raise BatchError(
                f"{error} (change #{index})",
                code=error.code,
                records=[location],
            ) from None
        seen.add(digest)
        normalized.append(
            {
                "sha256": digest,
                "old_label": old_label,
                "new_label": new_label,
            }
        )
    return batch_id, normalized
