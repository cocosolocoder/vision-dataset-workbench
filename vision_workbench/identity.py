"""The single identity format shared across every stored sample reference.

A sample identity is a full SHA-256 digest: a string of **exactly 64
lowercase hexadecimal characters**.  The registration manifest, saved
split plans and batch history all pin samples in that one spelling, so
the length, case, character-set and whitespace rule is defined here once
and reused at every place an identity is read back from storage:

* :mod:`vision_workbench.manifest` accepts a registration only when its
  ``sha256`` matches :func:`is_hex_identity`;
* :func:`vision_workbench.splits.invalid_digest_reason` and
  :func:`vision_workbench.batches.describe_history_digest_problem` report
  a concrete format reason through :func:`invalid_identity_reason`.

A value is never normalized into acceptance — no stripping whitespace,
folding case, truncating or padding — and no identity is ever inferred
from the samples currently registered: even a spelling whose normalized
form would name a known sample is refused as saved.
"""

from __future__ import annotations

import re
from typing import Any

#: Required length of a full SHA-256 hexadecimal digest.
DIGEST_LENGTH = 64

# A full SHA-256 digest: exactly 64 lowercase hexadecimal characters.
# ``\Z`` (not ``$``) pins the end so a trailing newline cannot sneak in.
_HEX_IDENTITY_RE = re.compile(r"[0-9a-f]{64}\Z")

# Whitespace reported explicitly, kept identical to the historical
# messages: the strip comparison catches any surrounding whitespace and
# these four characters cover whitespace embedded in the middle.
_EMBEDDED_WHITESPACE = " \t\r\n"


def is_hex_identity(value: Any) -> bool:
    """Whether ``value`` is exactly 64 lowercase hexadecimal characters.

    Only a ``str`` matches: ``None``, numbers, booleans and every other
    type are rejected like a malformed string.  The spelling is judged as
    saved — the value is never stripped, case-folded, truncated or padded.
    """
    return isinstance(value, str) and _HEX_IDENTITY_RE.fullmatch(value) is not None


def invalid_identity_reason(value: Any, *, missing_is_null: bool) -> str | None:
    """Return why ``value`` is not a full lowercase SHA-256 identity.

    Returns ``None`` when the value is a string of exactly 64 lowercase
    hexadecimal characters; otherwise a concrete, human-readable reason.
    The accepted spelling is the same one the registration manifest
    enforces, and the raw value is never normalized into acceptance.

    ``missing_is_null`` selects only the wording for non-string values,
    which historically differed between callers:

    * ``True`` (saved split plans) describes ``None`` as a missing or
      null field and names the actual type of any other non-string;
    * ``False`` (batch history) reports every non-string, ``None``
      included, as simply "not a string".

    Every string case — empty, surrounding or embedded whitespace, a
    wrong length, or an uppercase/non-hexadecimal character — uses the
    same wording for both callers.
    """
    if not isinstance(value, str):
        if missing_is_null and value is None:
            return (
                "the field is missing or null; expected a string of 64 "
                "lowercase hexadecimal characters"
            )
        if missing_is_null:
            return (
                f"it is a {type(value).__name__}, not a string; expected a "
                "string of 64 lowercase hexadecimal characters"
            )
        return (
            "it is not a string; expected a string of 64 lowercase "
            "hexadecimal characters"
        )
    if not value:
        return "it is empty; expected 64 lowercase hexadecimal characters"
    if value != value.strip() or any(
        char in _EMBEDDED_WHITESPACE for char in value
    ):
        return (
            "it contains surrounding or embedded whitespace; expected 64 "
            "lowercase hexadecimal characters without spaces"
        )
    if len(value) != DIGEST_LENGTH:
        return (
            f"it has {len(value)} characters instead of 64; expected 64 "
            "lowercase hexadecimal characters"
        )
    if _HEX_IDENTITY_RE.fullmatch(value) is None:
        # Length is exactly 64 here, so every remaining problem is an
        # uppercase letter or a non-hexadecimal character.
        return (
            "it must contain only lowercase hexadecimal characters "
            "(digits 0-9 and letters a-f)"
        )
    return None
