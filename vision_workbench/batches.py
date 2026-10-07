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

import re
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Sequence

BATCH_SCHEMA_VERSION = 1

# A registered sample identity is a full SHA-256 digest: exactly 64
# lowercase hexadecimal characters — the same spelling the registration
# manifest enforces.  Undo requires every target-batch record to pin a
# sample in that exact spelling, so a truncated or padded digest, an
# uppercase spelling, surrounding or embedded whitespace or any other
# non-hex character can never be accepted by stripping, lowercasing,
# truncating or padding.
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")


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
    and ``null`` vs ``""`` does not matter.  Labels are normalized on both
    sides before comparing, so the equality also holds across the two
    places records come from — freshly parsed records (already normalized
    to ``None``) and history rows loaded from disk, which may have legally
    saved either unlabeled spelling.  Every other string is kept exactly,
    so a real class named ``unlabeled`` never matches an unlabeled record
    and case, surrounding whitespace, Chinese text and path separators
    stay significant.
    """
    return frozenset(
        (
            record["sha256"],
            normalize_label(record["old"]),
            normalize_label(record["new"]),
        )
        for record in records
    )


# ---------------------------------------------------------------------------
# Submission rules
#
# Each stage is a pure function of parsed records and the state passed to
# it; none of them mutate the manifest or the history.  The stages line up
# with the conditions that reject a submission, in their original order:
#
#   1. reject_unknown_samples  — records naming no registered sample
#   2. resolve_re_submission   — same-number history: the number must
#                                identify one entry in the full history
#                                list (require_unique_history_number
#                                refusal), the existing entry must list
#                                each sample once (corruption refusal),
#                                save both labels on every record
#                                (corruption refusal) and carry a changed
#                                flag that matches those labels
#                                (corruption refusal), then replay or
#                                conflict
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

    Integrity is refused before either verdict.  The same-number entry
    must list every sample at most once (see
    :func:`require_intact_history_records`): a content digest appearing on
    a second record is batch-history corruption, and it has to be caught
    here because :func:`content_key` compares record *sets*, so a
    duplicated row would otherwise collapse away and the replay would
    count the sample's label change twice.  The entry must also
    explicitly carry both ``old`` and ``new`` on every record first (see
    :func:`require_intact_history_labels`): a row missing either key is
    corruption too, never an unlabeled label to guess, and it is reported
    before the replay or the conflict verdict — an entry that would
    otherwise conflict is refused for the damage instead.  Explicitly
    saved ``null`` and ``""`` stay ordinary unlabeled labels and take
    part in the comparison as usual.  Finally, every record's ``changed``
    flag must agree with the before/after labels saved on that same
    record (see :func:`require_intact_history_changes`): a replay
    reconstructs the original statistics from those flags, so a row
    flagged unchanged while its labels differ would silently swallow a
    real change, and a row flagged changed on identical labels would
    inflate the changed count — both are corruption, judged from the
    stored labels alone, never from the sample's current label or this
    submission's labels.
    """
    if existing is None:
        return None
    require_intact_history_records(existing)
    require_intact_history_labels(existing)
    require_intact_history_changes(existing)
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
# History integrity checks
#
# Every ``require_intact_history_*`` check scans the same record list of
# the same target batch entry, so the shared mechanics live in two
# helpers: ``_iter_history_records`` reads the list once, rejects a
# missing list or a non-object row and yields ``(position, record)``
# pairs; ``_corruption_prefix`` and ``_record_ref`` assemble the common
# "Batch history is corrupted: batch ..." message parts.  The checks
# themselves stay separate passes and keep their original order on the
# undo path — digest formats, then duplicate digests, then revisions,
# then labels, then changed flags — each reporting its first problem in
# record order.  Re-submission (``resolve_re_submission``) reuses only
# the three checks that verdict needs, in the same order: duplicate
# digests, then labels, then changed flags, all running before the
# same-content replay / different-content conflict decision.  Those are per-entry checks; the
# coarser ``require_unique_history_number`` check — that the submitted
# number identifies one entry at all in the full history list — is
# shared with undo and runs on the submit path even earlier, before the
# entry is selected (see ``find_unique_history_entry`` below).
# ---------------------------------------------------------------------------


def _corruption_prefix(entry: Mapping[str, Any]) -> str:
    """Shared prefix of every batch-history corruption message."""
    return f"Batch history is corrupted: batch {entry.get('batch')!r}"


def _record_ref(record: Mapping[str, Any], position: int | None = None) -> str:
    """Locate a history record by sample digest and optional 1-based position."""
    ref = f"sample {record.get('sha256')}"
    if position is not None:
        ref += f", record #{position}"
    return ref


def _iter_history_records(
    entry: Mapping[str, Any],
) -> Iterator[tuple[int, Mapping[str, Any]]]:
    """Yield ``(position, record)`` for each row of a batch history entry.

    This is the shared front half of every integrity check: the entry
    must carry a record list and every row must be an object.  Positions
    are 1-based, matching the record numbers used in error messages.
    """
    prefix = _corruption_prefix(entry)
    records = entry.get("records")
    if not isinstance(records, list):
        # validate_history() already rules this out; defend direct callers.
        raise BatchError(f"{prefix}: missing records")
    for position, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise BatchError(f"{prefix}: malformed record")
        yield position, record


def describe_history_digest_problem(value: Any) -> str | None:
    """Describe why a history record's ``sha256`` is not a registered identity.

    Every sample undo pins must be named in the exact spelling used at
    registration: a string of exactly 64 lowercase hexadecimal
    characters.  An empty string, any other length, uppercase letters,
    non-hex characters and surrounding or embedded whitespace are all
    rejected with a concrete reason; the value is never normalized into
    acceptance by stripping whitespace, folding case, truncating or
    padding, and no identity is inferred from the current sample list.
    Returns ``None`` when the value is well formed.
    """
    if not isinstance(value, str):
        # validate_history() already rules a missing or non-string
        # digest out; defend direct callers just the same.
        return (
            "it is not a string; expected a string of 64 lowercase "
            "hexadecimal characters"
        )
    if not value:
        return "it is empty; expected 64 lowercase hexadecimal characters"
    if value != value.strip() or any(char in " \t\r\n" for char in value):
        return (
            "it contains surrounding or embedded whitespace; expected 64 "
            "lowercase hexadecimal characters without spaces"
        )
    if len(value) != 64:
        return (
            f"it has {len(value)} characters instead of 64; expected 64 "
            "lowercase hexadecimal characters"
        )
    if _DIGEST_RE.fullmatch(value) is None:
        # Length is exactly 64 here, so what remains is an uppercase
        # letter or a non-hexadecimal character.
        return (
            "it must contain only lowercase hexadecimal characters "
            "(digits 0-9 and letters a-f)"
        )
    return None


def require_intact_history_digests(entry: Mapping[str, Any]) -> None:
    """Reject a batch entry whose records do not pin registered identities.

    Undo restores and counts samples purely from the target batch's
    history rows, so every row's ``sha256`` must name a sample in the
    exact spelling registration uses: a string of exactly 64 lowercase
    hexadecimal characters.  An empty string, a digest of any other
    length, an uppercase spelling, a non-hexadecimal character or
    surrounding or embedded whitespace is batch-history corruption —
    never a value to strip, lowercase, truncate or pad into acceptance,
    and never an identity to guess from the samples currently
    registered: even a spelling whose normalized form would name a known
    sample is refused as saved.  Whether the record actually changed a
    label is irrelevant; its position in the record list is irrelevant
    too.  The error names the batch number, the record's 1-based
    position inside the target batch, the raw saved ``sha256`` value and
    the exact format problem; nothing is repaired, rewritten or restored.

    The whole record list is scanned in order — unchanged rows and rows
    behind restorable ones included — so a well-formed row at the front
    can never mask a malformed one further back, and no sample is
    restored before the damage is found.  This judges format only, not
    registration: a well-formed digest of a sample that has since left
    the manifest is left to undo's ordinary missing-sample rule.

    Callers must run this before any success-shaped short-circuit (such
    as an already-undone batch), so corruption can never be masked by a
    repeated-undo success.
    """
    for position, record in _iter_history_records(entry):
        problem = describe_history_digest_problem(record.get("sha256"))
        if problem is not None:
            raise BatchError(
                f"{_corruption_prefix(entry)}, "
                f"{_record_ref(record, position)}: invalid 'sha256' value "
                f"{record.get('sha256')!r}: {problem}"
            )


def require_intact_history_records(entry: Mapping[str, Any]) -> None:
    """Reject a batch entry whose records are not one row per sample.

    Undo pins every stored record to a sample content digest, so a digest
    may occur at most once in the target batch: a duplicated row — even a
    byte-for-byte copy, or a copy of a record that never changed a label —
    would otherwise restore the same sample more than once, bump its label
    revision repeatedly and inflate the restored count.  Duplicates are
    therefore batch-history corruption, never records to merge or
    de-duplicate.  The decision is independent of the labels, the
    before/after labels and the record's ``changed`` flag: the content
    digest alone decides.

    The whole record list is scanned, so rows preceding the repeat are
    checked too and a restorable row at the front can never mask a later
    duplicate.  On a repeat the error names the batch number, the sample's
    full digest and both 1-based record positions; nothing is repaired.

    Callers must run this before any success-shaped short-circuit (such as
    an already-undone batch or a same-number replay), so corruption can
    never be masked by a repeated-undo success or by a replay whose
    set-based content comparison would collapse the duplicate into one.
    """
    seen: dict[str, int] = {}
    for position, record in _iter_history_records(entry):
        digest = record.get("sha256")
        if digest in seen:
            raise BatchError(
                f"{_corruption_prefix(entry)}, sample {digest} "
                f"appears more than once, at record #{seen[digest]} and "
                f"record #{position}"
            )
        seen[digest] = position


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
    for _position, record in _iter_history_records(entry):
        problem = describe_history_revision_problem(
            record.get("rev"), present="rev" in record
        )
        if problem is not None:
            raise BatchError(
                f"{_corruption_prefix(entry)}, "
                f"{_record_ref(record)}: {problem}"
            )


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


def require_intact_history_labels(entry: Mapping[str, Any]) -> None:
    """Reject a batch entry whose records lack a saved before/after label.

    Undo restores and counts samples purely from what the target batch's
    history rows say: a changed row is restored from ``new`` back to
    ``old``, an unchanged row is left alone.  That is only meaningful
    when every row explicitly carries both labels, so a row missing
    ``old``, ``new`` or both is batch-history corruption — never an
    unlabeled-to-unlabeled no-op to skip, and never something to guess
    from the sample's current label.  The error names the batch number,
    the sample's full SHA-256 digest, the record's 1-based position
    inside the target batch and which of ``old``/``new`` (or both) is
    absent; nothing is repaired, rewritten or restored.

    An explicitly saved ``null`` or empty string stays legal: both mean
    unlabeled, as does a literal class named ``unlabeled`` stay an
    ordinary label.  Only a missing key is damage.

    The whole record list is scanned in order — unchanged rows and rows
    behind restorable ones included — so a complete row at the front can
    never mask a damaged one further back, and no sample is restored
    before the damage is found.

    Callers must run this before any success-shaped short-circuit (such
    as an already-undone batch or a same-number replay), so corruption
    can never be masked by a success verdict or swallowed by a conflict.
    """
    for position, record in _iter_history_records(entry):
        problem = describe_missing_history_labels(record)
        if problem is not None:
            raise BatchError(
                f"{_corruption_prefix(entry)}, "
                f"{_record_ref(record, position)}: {problem}"
            )


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


def require_intact_history_changes(entry: Mapping[str, Any]) -> None:
    """Reject a batch entry whose ``changed`` flags belie old/new labels.

    Undo restores a sample based on what the history row says the batch
    did: a row marked changed is restored from ``new`` back to ``old`` and
    counted, while a row marked unchanged is never restored.  So the flag
    must match the before/after labels saved on that row — judging from the
    record's own stored labels rather than any sample's current label:

    * differing old/new (``null`` and ``""`` being the same unlabeled
      spelling) require ``changed: true``;
    * identical old/new require ``changed: false``.

    A row that claims a change with identical labels would make undo count
    and restore a sample the batch never touched; a row that hides a real
    change would leave the sample at its post-batch label while the batch
    is recorded undone.  Both are batch-history corruption: the error names
    the batch number, the sample's full SHA-256 digest and the record's
    1-based position inside the target batch, together with the exact
    contradiction, and nothing is repaired.

    The whole record list is scanned in order, so a restorable row at the
    front can never mask a contradiction further back.

    Callers must run this before any success-shaped short-circuit (such as
    an already-undone batch or a same-number replay), so corruption can
    never be masked by a repeated-undo success or by a replay whose stored
    flags would miscount the original batch.
    """
    for position, record in _iter_history_records(entry):
        problem = describe_history_change_problem(record)
        if problem is not None:
            raise BatchError(
                f"{_corruption_prefix(entry)}, "
                f"{_record_ref(record, position)}: {problem}"
            )


def history_number_positions(
    number: str, batches: Sequence[Mapping[str, Any]]
) -> list[int]:
    """1-based positions of every history entry whose number is ``number``.

    Matching is on the batch number's saved raw string — no trimming of
    surrounding whitespace and no case folding — so ``"b1"``, ``"B1"``
    and ``" b1"`` never meet.  The whole list is scanned rather than
    stopping at the first hit, since callers must tell a unique number
    from a repeated one.
    """
    return [
        position
        for position, entry in enumerate(batches, start=1)
        if isinstance(entry, Mapping) and entry.get("batch") == number
    ]


def require_unique_history_number(
    number: str, batches: Sequence[Mapping[str, Any]]
) -> None:
    """Reject a number appearing on at least two full-history entries.

    Both re-submission and undo must resolve a batch number to exactly
    one history entry.  A second entry carrying the same number makes the
    number ambiguous no matter what the two entries contain — identical
    record content, different samples, different label changes or
    differing undo states are all refusal, never a reason to pick the
    first, the last or the still-active one, and never something to
    merge, renumber or de-duplicate.  Whether the entries sit next to
    each other or have other batches between them is irrelevant.

    On a repeat the error names the user-given batch number and the two
    1-based positions of the first and second occurrence in the full
    history list; with three or more occurrences only the first two are
    named.  Nothing is repaired.  Callers must run this before any
    same-number verdict (a same-content replay or a different-content
    conflict) and before the already-undone short-circuit: the entries'
    content and undo state can never disambiguate the number, and an
    entry already marked undone leaves it just as ambiguous.
    """
    positions = history_number_positions(number, batches)
    if len(positions) >= 2:
        raise BatchError(
            "Batch history is corrupted: batch number "
            f"{number!r} appears more than once, at batch #{positions[0]} "
            f"and batch #{positions[1]}"
        )


def find_unique_history_entry(
    number: str, batches: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    """Find the single history entry whose number is ``number``.

    An undo target must name exactly one batch.  Matching is on the batch
    number's saved raw string — no trimming of surrounding whitespace and
    no case folding — so the whole list is scanned with exact equality
    rather than stopping at the first hit: a second entry carrying the
    same number, whether adjacent to the first or separated by other
    batches, makes the target ambiguous.  Such a history is corrupted no
    matter what the two entries contain — identical record content,
    different samples, different label changes or differing undo states
    are all refusal, never a reason to pick the first, the last or the
    still-active one, and never something to merge.

    Returns the unique entry, or ``None`` when the number does not occur.
    On a repeat the error names the user-given batch number and the two
    1-based positions of the conflicting entries in the full history
    list, so neither record is undone, preferred or repaired.  Callers
    must run this before every per-entry integrity check and before the
    already-undone short-circuit: even an entry already marked undone
    leaves the number ambiguous and must surface this instead.
    """
    positions = history_number_positions(number, batches)
    if not positions:
        return None
    # Delegate the refusal (and its wording) so submit and undo can never
    # disagree; the positions are already known to number at least one.
    require_unique_history_number(number, batches)
    return batches[positions[0] - 1]


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
