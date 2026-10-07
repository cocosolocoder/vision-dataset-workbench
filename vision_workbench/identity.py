"""The single SHA-256 sample-identity format rule.

A registered sample identity is a string of exactly 64 lowercase
hexadecimal characters.  The registration manifest, saved split plans
and batch-history undo records all enforce that same spelling, so the
checks themselves — their order, the length, the whitespace test and
the hexadecimal alphabet — live here exactly once.  Each feature
classifies a rejected value through :func:`identity_problem` and only
supplies its own error wording around the shared verdict.

A value is never normalized into acceptance: stripping whitespace,
folding case, truncating or padding can never turn a rejected spelling
into a valid identity, and no identity is inferred from the samples
currently registered.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

# A full SHA-256 digest: exactly 64 lowercase hexadecimal characters.
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")


class IdentityProblem(Enum):
    """Why a value is not a valid sample identity.

    Members are listed — and detected by :func:`identity_problem` — in
    check order: a value matching an earlier problem is never also
    judged against a later one.
    """

    NOT_STRING = "not-string"
    EMPTY = "empty"
    WHITESPACE = "whitespace"
    WRONG_LENGTH = "wrong-length"
    BAD_CHARACTERS = "bad-characters"


def identity_problem(value: Any) -> IdentityProblem | None:
    """Classify why ``value`` is not a valid identity, else ``None``.

    A non-string value (including a missing ``None``), an empty string,
    surrounding or embedded whitespace, any length other than 64 and
    any character outside ``0-9a-f`` (uppercase letters included) is
    rejected; only a string of exactly 64 lowercase hexadecimal
    characters returns ``None``.  The value is only inspected, never
    repaired.
    """
    if not isinstance(value, str):
        return IdentityProblem.NOT_STRING
    if not value:
        return IdentityProblem.EMPTY
    if value != value.strip() or any(char in " \t\r\n" for char in value):
        return IdentityProblem.WHITESPACE
    if len(value) != 64:
        return IdentityProblem.WRONG_LENGTH
    if _DIGEST_RE.fullmatch(value) is None:
        # Length is exactly 64 here, so every remaining problem is an
        # uppercase letter or a non-hexadecimal character.
        return IdentityProblem.BAD_CHARACTERS
    return None


def is_valid_identity(value: Any) -> bool:
    """Whether ``value`` is exactly 64 lowercase hexadecimal characters."""
    return identity_problem(value) is None
