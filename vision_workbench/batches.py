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
from typing import Any, Callable, Iterator, Mapping

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


# ---------------------------------------------------------------------------
# Undo-time integrity checks
#
# Every check targets exactly one batch entry and judges its whole record
# list before the caller may restore a label or take an already-undone
# short-circuit.  The checks share a single reading of the target entry
# (:func:`_history_check_context`) and a single way of turning a finding
# into the standard corruption message (:func:`_corrupted`), so the four
# damage kinds never re-derive the batch number, re-walk the list,
# re-test that a row is an object or re-assemble the same batch/sample
# error context on their own.
#
# The kinds keep a fixed, whole-batch precedence:
#
#   1. a sample digest appearing more than once in the target batch;
#   2. an illegal or missing pinned history revision (``rev``);
#   3. a missing before/after label (``old``/``new``);
#   4. a ``changed`` flag contradicting the labels saved on its own row.
#
# Each kind is one complete pass over every record — unchanged rows and
# rows behind restorable ones included — so when several kinds are
# damaged at once the earliest kind in the list wins even if a later
# kind's damage sits on an earlier record, instead of a row-by-row walk
# reporting whichever problem it meets first.  Within one kind the
# damaged records are still reported in record order.
# ---------------------------------------------------------------------------


def _history_check_context(entry: Mapping[str, Any]) -> tuple[Any, list[Any]]:
    """Return the target batch's ``(number, records)`` once for every check.

    ``validate_history`` already guarantees a records list, but a check
    can be handed a hand-built entry directly, so a missing or non-list
    ``records`` field is reported here once instead of defended in every
    check separately.
    """
    number = entry.get("batch")
    records = entry.get("records")
    if not isinstance(records, list):
        raise BatchError(
            f"Batch history is corrupted: batch {number!r}: missing records"
        )
    return number, records


def _corrupted(number: Any, detail: str) -> BatchError:
    """Build the standard target-batch corruption error for ``detail``."""
    return BatchError(f"Batch history is corrupted: batch {number!r}, {detail}")


def _iter_duplicate_errors(
    number: Any, records: list[Any]
) -> Iterator[BatchError]:
    """Yield the non-object-row error and every repeated-digest error, in order.

    Undo pins every stored record to a sample content digest, so a digest
    may occur at most once in the target batch; the content digest alone
    decides, independently of the labels, the before/after labels and the
    ``changed`` flag.  A repeat names both 1-based record positions,
    keeping the first occurrence when the same digest appears yet again.
    """
    seen: dict[Any, int] = {}
    for position, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            yield BatchError(
                f"Batch history is corrupted: batch {number!r}: malformed record"
            )
            continue
        digest = record.get("sha256")
        if digest in seen:
            yield _corrupted(
                number,
                f"sample {digest} appears more than once, at record "
                f"#{seen[digest]} and record #{position}",
            )
            # Keep the first position, so a third row still names #1.
            continue
        seen[digest] = position


def _iter_record_errors(
    number: Any,
    records: list[Any],
    describe: Callable[[Mapping[str, Any], int], str | None],
) -> Iterator[BatchError]:
    """Yield one corruption error per damaged record, in record order.

    ``describe`` receives an object record and its 1-based position and
    returns that check's damage detail (batch/sample framing added here),
    or ``None`` when the record is sound.  A non-object row is the same
    damage for every check, so it is detected here once rather than in
    each check.
    """
    for position, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            yield BatchError(
                f"Batch history is corrupted: batch {number!r}: malformed record"
            )
            continue
        problem = describe(record, position)
        if problem is not None:
            yield _corrupted(number, problem)


def _raise_first(errors: Iterator[BatchError]) -> None:
    """Raise the first record-order damage from a check's error iterator."""
    for error in errors:
        raise error


def require_intact_history_records(entry: Mapping[str, Any]) -> None:
    """Reject a batch entry whose records are not one row per sample.

    A duplicated row — even a byte-for-byte copy, or a copy of a record
    that never changed a label — would restore the same sample more than
    once, bump its label revision repeatedly and inflate the restored
    count, so it is batch-history corruption, never a row to merge or
    de-duplicate.  The whole list is scanned, so a restorable row at the
    front can never mask a later duplicate.  See
    :func:`require_intact_history` for the checks' shared contract.
    """
    number, records = _history_check_context(entry)
    _raise_first(_iter_duplicate_errors(number, records))


def _revision_damage(record: Mapping[str, Any], position: int) -> str | None:
    """Detail for one record's illegal or missing pinned ``rev``."""
    problem = describe_history_revision_problem(
        record.get("rev"), present="rev" in record
    )
    if problem is None:
        return None
    return f"sample {record.get('sha256')}: {problem}"


def require_intact_history_revisions(entry: Mapping[str, Any]) -> None:
    """Reject a batch entry whose records lack a sound pinned ``rev``.

    Every record — changed or not — must carry ``rev`` as a genuine
    non-negative integer, since undo pins each changed sample to the
    revision it had at the end of the batch; unlike the manifest's
    optional revision, a bad or missing row is integrity damage, never a
    coerced zero.  See :func:`require_intact_history` for the checks'
    shared contract.
    """
    number, records = _history_check_context(entry)
    _raise_first(_iter_record_errors(number, records, _revision_damage))


def describe_missing_history_labels(record: Mapping[str, Any]) -> str | None:
    """Describe which of a history record's before/after labels are absent.

    Every batch-history sample record must spell out both ``old`` (the
    label before the batch) and ``new`` (the label after it).  Only key
    presence is judged here: an explicitly saved ``null`` or empty string
    is a recorded unlabeled label, never a missing field.  Returns
    ``None`` when both keys are present.
    """
    missing = [key for key in ("old", "new") if key not in record]
    if not missing:
        return None
    if len(missing) == 2:
        return "missing 'old' and 'new' labels"
    return f"missing {missing[0]!r} label"


def _missing_labels_damage(record: Mapping[str, Any], position: int) -> str | None:
    """Detail for one record's missing before/after label, with its position."""
    problem = describe_missing_history_labels(record)
    if problem is None:
        return None
    return (
        f"sample {record.get('sha256')}, record #{position}: {problem}"
    )


def require_intact_history_labels(entry: Mapping[str, Any]) -> None:
    """Reject a batch entry whose records lack a saved before/after label.

    Undo restores and counts samples purely from what the target batch's
    history rows say, so every row must explicitly carry both ``old`` and
    ``new``; a missing key is corruption, never an unlabeled-to-unlabeled
    no-op and never something guessed from the current label.  An
    explicitly saved ``null`` or empty string stays legal (both mean
    unlabeled), as does a literal class named ``unlabeled``.  The error
    names the batch number, the sample's full digest, the record's
    1-based position and which label is absent.  See
    :func:`require_intact_history` for the checks' shared contract.
    """
    number, records = _history_check_context(entry)
    _raise_first(_iter_record_errors(number, records, _missing_labels_damage))


def history_labels_equal(old: Any, new: Any) -> bool:
    """Whether two stored history labels describe the same category.

    The decision uses the labels saved in the history record itself, never
    a sample's current label: ``null`` and the empty string both mean
    unlabeled and are therefore equal, while every other value is compared
    as the exact original string — case, surrounding whitespace, Chinese
    text and path separators all keep their meaning, and a real class
    literally named ``unlabeled`` never equals unlabeled.
    """
    return normalize_label(old) == normalize_label(new)


def describe_history_change_problem(record: Mapping[str, Any]) -> str | None:
    """Describe why a history record's ``changed`` flag contradicts old/new.

    The flag must agree with the before/after labels saved on that same
    record: differing labels require ``changed: true`` and equal labels
    require ``changed: false``.  Returns ``None`` when the flag is
    consistent.  The stored labels are validated upstream as strings or
    ``None`` and the flag as a boolean, so no type guessing happens here.
    """
    actually_changed = not history_labels_equal(record["old"], record["new"])
    old_label = normalize_label(record["old"])
    new_label = normalize_label(record["new"])
    if actually_changed and not record["changed"]:
        return (
            f"'changed' is false but the before/after labels differ: "
            f"{render_label(old_label)!r} -> {render_label(new_label)!r}"
        )
    if not actually_changed and record["changed"]:
        return (
            f"'changed' is true but the before/after labels are identical: "
            f"{render_label(old_label)!r}"
        )
    return None


def _change_flag_damage(record: Mapping[str, Any], position: int) -> str | None:
    """Detail for one record's ``changed`` flag contradicting its labels."""
    problem = describe_history_change_problem(record)
    if problem is None:
        return None
    return (
        f"sample {record.get('sha256')}, record #{position}: {problem}"
    )


def require_intact_history_changes(entry: Mapping[str, Any]) -> None:
    """Reject a batch entry whose ``changed`` flags belie old/new labels.

    A row marked changed is restored from ``new`` back to ``old`` and
    counted; a row marked unchanged is never restored.  So the flag must
    match the labels saved on that same row, judged from the record's own
    stored labels (never a sample's current label): differing old/new
    require ``changed: true`` and identical old/new require
    ``changed: false`` (``null`` and ``""`` are the same unlabeled
    spelling).  The error names the batch number, the sample's full
    digest, the record's 1-based position and the exact contradiction.
    See :func:`require_intact_history` for the checks' shared contract.
    """
    number, records = _history_check_context(entry)
    _raise_first(_iter_record_errors(number, records, _change_flag_damage))


# The four whole-batch passes in their fixed precedence.  Each pass reads
# the very same (number, records) pair produced once by
# :func:`_history_check_context`; adding a new damage kind means adding
# one describe/pass pair here, not another hand-written scan.
def _history_integrity_passes(
    number: Any, records: list[Any]
) -> list[Iterator[BatchError]]:
    return [
        _iter_duplicate_errors(number, records),
        _iter_record_errors(number, records, _revision_damage),
        _iter_record_errors(number, records, _missing_labels_damage),
        _iter_record_errors(number, records, _change_flag_damage),
    ]


def require_intact_history(entry: Mapping[str, Any]) -> None:
    """Validate one target batch's history before an undo may proceed.

    The entry's whole record list is judged against every damage kind,
    each as a complete pass:

      1. repeated sample digests within the batch,
      2. illegal or missing pinned revisions (``rev``),
      3. missing before/after labels (``old``/``new``),
      4. ``changed`` flags contradicting the labels saved on their rows.

    When several kinds are damaged at once, the kinds are reported in
    exactly that order across the whole batch — a damaged later kind on
    an early record never jumps ahead of an earlier kind whose damage
    sits further back — and within one kind the records keep their stored
    order.  Nothing is repaired, rewritten or restored, and no label is
    written before every record has passed the checks still ahead of it,
    so a restorable row at the front can never mask later damage.

    Callers must run this before any success-shaped short-circuit (an
    already-undone batch included) and before checking whether a
    restorable sample was modified after the batch, so corruption is
    never hidden behind ``already-undone`` or confused with a later
    modification.
    """
    number, records = _history_check_context(entry)
    for errors in _history_integrity_passes(number, records):
        _raise_first(errors)


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
