"""Strict validation of the on-disk sample registration manifest.

Every feature that uses the current manifest — summary, label queries,
batch updates, imports and plan creation — validates the *whole*
registration list before touching any sample.  A manifest is usable only
when it is an object whose ``schema_version`` is the integer ``1`` and
whose ``items`` is a list of complete, well-typed sample records with no
repeated digest.  Booleans and decimals never stand in for integers
(``True == 1`` in Python), numeric strings never stand in for sizes, and
duplicate registrations are reported rather than merged or silently
resolved to one copy.
"""

from __future__ import annotations

from typing import Any

from .identity import is_valid_identity

MANIFEST_SCHEMA_VERSION = 1


class ManifestError(ValueError):
    """The registration manifest is structurally unusable."""


def _is_integer(value: Any) -> bool:
    """Whether ``value`` is a genuine integer.

    Booleans are rejected even though ``bool`` subclasses ``int`` (so
    ``True`` cannot masquerade as version ``1`` or as a byte size), and
    every other non-integer type (floats, including integral ones such as
    ``1.0``, numeric strings, ``None``) is rejected as well.
    """
    return isinstance(value, int) and not isinstance(value, bool)


def validate_manifest(data: Any) -> dict[str, Any]:
    """Validate a parsed manifest payload and return it unchanged.

    Structural problems describe the offending part; a bad sample record
    names its 1-based position in the list and the offending field; a
    digest registered twice names the digest and both record positions.
    The whole list is checked, so valid records preceding a bad one never
    mask a later error.
    """
    if not isinstance(data, dict):
        raise ManifestError(
            "Dataset manifest is corrupted: manifest must be a JSON object"
        )
    version = data.get("schema_version")
    if not _is_integer(version):
        raise ManifestError(
            "Dataset manifest is corrupted: schema_version must be the integer 1"
        )
    if version != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            f"Dataset manifest is corrupted: unsupported schema version {version}"
        )
    items = data.get("items")
    if not isinstance(items, list):
        raise ManifestError(
            "Dataset manifest is corrupted: items must be a JSON array of samples"
        )

    seen: dict[str, int] = {}
    for position, item in enumerate(items, start=1):
        location = f"sample #{position}"
        if not isinstance(item, dict):
            raise ManifestError(
                f"Dataset manifest is corrupted: {location} must be a JSON object"
            )

        digest = item.get("sha256")
        if not isinstance(digest, str):
            raise ManifestError(
                f"Dataset manifest is corrupted: {location} is missing its "
                "string 'sha256' field"
            )
        # The shared identity format rule (exactly 64 lowercase
        # hexadecimal characters, never normalized into acceptance)
        # lives in vision_workbench.identity.
        if not is_valid_identity(digest):
            raise ManifestError(
                f"Dataset manifest is corrupted: {location} has an invalid "
                f"'sha256' field {digest!r}: expected 64 lowercase hexadecimal "
                "characters"
            )

        if "source" not in item:
            raise ManifestError(
                f"Dataset manifest is corrupted: {location} (sha256 {digest}) "
                "is missing its non-empty string 'source' field"
            )
        source = item["source"]
        if not isinstance(source, str):
            raise ManifestError(
                f"Dataset manifest is corrupted: {location} (sha256 {digest}) "
                "has an invalid 'source' field: expected a non-empty string"
            )
        if not source:
            raise ManifestError(
                f"Dataset manifest is corrupted: {location} (sha256 {digest}) "
                "has an invalid 'source' field: the source path must not be empty"
            )

        if "size" not in item:
            raise ManifestError(
                f"Dataset manifest is corrupted: {location} (sha256 {digest}) "
                "is missing its non-negative integer 'size' field"
            )
        size = item["size"]
        if not _is_integer(size):
            raise ManifestError(
                f"Dataset manifest is corrupted: {location} (sha256 {digest}) "
                "has an invalid 'size' field: expected a non-negative integer "
                "(booleans, numeric strings and decimals are not accepted)"
            )
        if size < 0:
            raise ManifestError(
                f"Dataset manifest is corrupted: {location} (sha256 {digest}) "
                f"has an invalid 'size' field: {size} is negative"
            )

        if "label" not in item:
            raise ManifestError(
                f"Dataset manifest is corrupted: {location} (sha256 {digest}) "
                "is missing its 'label' field (expected a string or null)"
            )
        label = item["label"]
        if label is not None and not isinstance(label, str):
            raise ManifestError(
                f"Dataset manifest is corrupted: {location} (sha256 {digest}) "
                "has an invalid 'label' field: expected a string or null"
            )

        if digest in seen:
            raise ManifestError(
                f"Dataset manifest is corrupted: digest {digest} is registered "
                f"twice, at sample #{seen[digest]} and sample #{position}"
            )
        seen[digest] = position

    # Extra fields, both on the manifest and on individual records (for
    # example the optional label revision), are intentionally preserved:
    # only the structural requirements above are enforced.
    return data
