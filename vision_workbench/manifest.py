"""Whole-manifest validation for the dataset registration list.

Every operation that uses registered samples reads the manifest through
:func:`validate_manifest` first.  The check covers the complete sample
list before any sample is used, so a missing field, a wrongly typed
field or a duplicate registration anywhere in the list rejects the
whole operation instead of surfacing as a runtime exception on one
record or silently acting on one of two duplicate rows.

The accepted shape is::

    {
      "schema_version": 1,
      "items": [
        {"sha256": "<64 lowercase hexadecimal characters>",
         "source": "<non-empty string>",
         "size": <non-negative integer>,
         "label": <string or null>,
         ...optional fields such as "rev"...}
      ]
    }

Only these fields are validated: optional information that older
manifests omit (such as the label revision) and any extra fields are
left untouched.  Booleans never count as integers for the schema
version or the file size, decimal numbers and numeric strings never
count as sizes, and each digest may appear exactly once.
"""

from __future__ import annotations

import re
from typing import Any

MANIFEST_SCHEMA_VERSION = 1

# A full SHA-256 digest: exactly 64 lowercase hexadecimal characters.
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")


class ManifestError(ValueError):
    """The persisted dataset registration list cannot be used as-is."""


def _is_non_negative_integer(value: Any) -> bool:
    """Whether ``value`` is a genuine non-negative integer.

    Booleans are rejected even though ``bool`` is a subclass of ``int``,
    and floats (including integral values such as ``1.0``) and strings
    (including ``"1"``) are rejected too: the file size must be stored
    as a JSON integer.
    """
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_manifest(data: Any) -> dict[str, Any]:
    """Validate a parsed manifest payload and return it unchanged.

    The overall structure (an object carrying the integer schema
    version 1 and a sample list) is checked first; then every sample is
    checked in list order.  Each sample must be an object with a full
    lowercase SHA-256 digest, a non-empty string source, a non-negative
    integer size and a string-or-null label, and no digest may be
    registered twice.

    The first problem found is reported with enough detail to locate
    it: structural problems name the offending part, sample problems
    name the sample's 1-based position and field, and a duplicate
    digest names the digest and both records' positions.
    """
    if not isinstance(data, dict):
        raise ManifestError("Dataset manifest must be a JSON object")
    version = data.get("schema_version")
    if "schema_version" not in data:
        raise ManifestError("Dataset manifest is missing its schema version")
    if not (
        isinstance(version, int)
        and not isinstance(version, bool)
        and version == MANIFEST_SCHEMA_VERSION
    ):
        raise ManifestError(
            "Dataset manifest schema version must be the integer 1, "
            f"got {version!r}"
        )
    if "items" not in data:
        raise ManifestError("Dataset manifest is missing its sample list ('items')")
    items = data["items"]
    if not isinstance(items, list):
        raise ManifestError(
            "Dataset manifest sample list ('items') must be a list, "
            f"got {type(items).__name__}"
        )

    seen: dict[str, int] = {}
    for position, item in enumerate(items, start=1):
        location = f"Dataset manifest sample #{position}"
        if not isinstance(item, dict):
            raise ManifestError(
                f"{location} must be an object, got {type(item).__name__}"
            )

        if "sha256" not in item:
            raise ManifestError(f"{location} is missing field 'sha256'")
        digest = item["sha256"]
        if not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
            raise ManifestError(
                f"{location} has an invalid 'sha256': must be 64 lowercase "
                f"hexadecimal characters, got {digest!r}"
            )

        if "source" not in item:
            raise ManifestError(f"{location} is missing field 'source'")
        source = item["source"]
        if not isinstance(source, str) or not source:
            raise ManifestError(
                f"{location} has an invalid 'source': must be a non-empty "
                f"string, got {source!r}"
            )

        if "size" not in item:
            raise ManifestError(f"{location} is missing field 'size'")
        size = item["size"]
        if not _is_non_negative_integer(size):
            raise ManifestError(
                f"{location} has an invalid 'size': must be a non-negative "
                "integer (booleans, decimal numbers and numeric strings are "
                f"not accepted), got {size!r}"
            )

        if "label" not in item:
            raise ManifestError(f"{location} is missing field 'label'")
        label = item["label"]
        if label is not None and not isinstance(label, str):
            raise ManifestError(
                f"{location} has an invalid 'label': must be a string or "
                f"null, got {label!r}"
            )

        if digest in seen:
            raise ManifestError(
                f"Dataset manifest lists digest {digest} more than once: "
                f"samples #{seen[digest]} and #{position}"
            )
        seen[digest] = position

    return data
