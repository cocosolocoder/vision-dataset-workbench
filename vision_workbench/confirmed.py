"""Confirm one ordinary regular file and open exactly that confirmed file.

This is the single file-confirmation rule shared by every entry point
that turns a path into sample bytes:

* single-file import (``DatasetStore.add``),
* whole-directory import (``DatasetStore.import_directory``),
* ``--source-dir`` export, both when the source tree is looked up and
  when the selected copy is streamed into the package, and
* ordinary (recorded-path) export, when a plan-recorded source is
  streamed into the package.

The rule:

1. The path is inspected *without following a final symlink*
   (``lstat`` / ``fstatat``); the object found there must be a regular
   file.  Callers that arrive through their own non-symlink walk
   (directory import, source-dir export) only ever confirm such paths,
   while single-file import resolves a symlink given as its argument
   before calling in here, so following links at the start stays an
   entry-point decision.
2. The path is opened with ``O_NOFOLLOW`` (relative to a pinned
   directory descriptor when the caller has one), so a symlink swapped
   onto the path after the inspection cannot redirect the open.
3. The opened descriptor is ``fstat()``ed and proved to be a regular
   file carrying the exact ``(device, inode)`` identity that was
   confirmed (or, for the export copy, the identity recorded during the
   lookup): a different regular file renamed onto the path — even one
   with identical content, size and modification time — is rejected
   just like a symlink, even one pointing at the original file.

What differs between entry points deliberately stays outside this
module:

* whether a path occupied by something else *after* a successful open
  still matters — import re-inspects the path once the read is done and
  requires the same file with the same size and mtime, while export
  keeps reading the already pinned descriptor (the path may be
  reoccupied without spoiling the confirmed bytes);
* the public exception type, wording and identifying information.

Failures are therefore reported as structured :class:`ConfirmationError`
reasons; each entry point translates them into its existing public
errors, preserving the path, digest and reason information callers rely
on.
"""

from __future__ import annotations

import errno
import hashlib
import os
import stat
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

# Samples are streamed in fixed chunks, so large files never need memory
# for their whole content; every reader below uses the same chunk size.
READ_CHUNK_SIZE = 1024 * 1024

# Open without following a final symlink; 0 on platforms lacking the flag.
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
# Open in non-blocking mode so a FIFO swapped onto a confirmed path cannot
# stall the open waiting for a writer: on Linux a read-only O_NONBLOCK open
# of a writerless FIFO returns immediately (the descriptor is then proved
# non-regular and closed), and a unix socket fails at the open with ENXIO.
# 0 where the flag is unavailable (regular files are unaffected by it).
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)


class ConfirmStage(str, Enum):
    """Where in the confirm-then-open sequence a failure was detected."""

    INSPECT = "inspect"  # the no-follow lstat()/fstatat() before opening
    OPEN = "open"  # the O_NOFOLLOW open()
    PROVE = "prove"  # the fstat() of the freshly opened descriptor


class ConfirmReason(str, Enum):
    """What went wrong, independent of the caller's public wording."""

    ACCESS_ERROR = "access error"  # an underlying syscall failed; .error set
    NOT_REGULAR = "not a regular file"  # a non-symlink, non-regular object
    NOW_SYMLINK = "is now a symlink"  # the path became a final symlink
    WRONG_IDENTITY = "wrong identity"  # another object occupies the path


@dataclass
class ConfirmationError(Exception):
    """A path could not be confirmed as, or opened as, one stable file.

    ``status`` carries the inspection result when one was obtained, so a
    caller can tell symlinks from other non-regular objects; ``error``
    carries the underlying :class:`OSError` for access failures.
    """

    reason: ConfirmReason
    stage: ConfirmStage
    status: os.stat_result | None = None
    error: OSError | None = None

    def __str__(self) -> str:  # pragma: no cover - debugging only
        return f"{self.reason.value} at {self.stage.value}"


def inspect_regular(
    path: str | os.PathLike[str],
    *,
    dir_fd: int | None = None,
    expected: tuple[int, int] | None = None,
) -> os.stat_result:
    """Inspect ``path`` without following a final symlink and confirm it.

    Returns the status of the confirmed regular file.  When ``dir_fd``
    is given, ``path`` is interpreted relative to that descriptor
    (``fstatat``); otherwise it is ``lstat()``ed directly.

    ``expected`` optionally carries the ``(device, inode)`` the binding
    must still carry — the identity recorded during the source-dir
    lookup when the export copy re-confirms a selected path.  A path
    that is a symlink, is not a regular file, or no longer carries the
    expected identity raises :class:`ConfirmationError` before anything
    is opened.
    """
    try:
        if dir_fd is None:
            status = os.lstat(path)
        else:
            status = os.stat(path, dir_fd=dir_fd, follow_symlinks=False)
    except OSError as error:
        raise ConfirmationError(
            ConfirmReason.ACCESS_ERROR, ConfirmStage.INSPECT, error=error
        ) from error
    if stat.S_ISLNK(status.st_mode):
        raise ConfirmationError(
            ConfirmReason.NOW_SYMLINK, ConfirmStage.INSPECT, status=status
        )
    if not stat.S_ISREG(status.st_mode):
        raise ConfirmationError(
            ConfirmReason.NOT_REGULAR, ConfirmStage.INSPECT, status=status
        )
    if expected is not None and (status.st_dev, status.st_ino) != expected:
        raise ConfirmationError(
            ConfirmReason.WRONG_IDENTITY, ConfirmStage.INSPECT, status=status
        )
    return status


def open_without_follow(
    path: str | os.PathLike[str],
    *,
    dir_fd: int | None = None,
    nonblock: bool = False,
) -> int:
    """Open ``path`` read-only with ``O_NOFOLLOW``, never following a link.

    A final symlink swapped in after the inspection makes the open fail
    with ``ELOOP``, which is reported as :class:`ConfirmReason.NOW_SYMLINK`
    rather than a generic access error; every other open failure carries
    the original :class:`OSError`.  With ``nonblock`` the open also
    carries ``O_NONBLOCK``: a regular file opens just the same, while a
    writerless FIFO opens immediately rather than blocking for a writer
    (the caller then proves the descriptor non-regular) and a unix
    socket fails at the open itself with ``ENXIO``.  The descriptor
    belongs to the caller on success.
    """
    flags = os.O_RDONLY | _O_NOFOLLOW
    if nonblock:
        flags |= _O_NONBLOCK
    try:
        return os.open(path, flags, dir_fd=dir_fd)
    except OSError as error:
        reason = (
            ConfirmReason.NOW_SYMLINK
            if error.errno == errno.ELOOP
            else ConfirmReason.ACCESS_ERROR
        )
        raise ConfirmationError(
            reason, ConfirmStage.OPEN, error=error
        ) from error


def prove_opened_regular(
    descriptor: int, confirmed: os.stat_result
) -> os.stat_result:
    """Prove ``descriptor`` is the confirmed regular file itself.

    The descriptor is ``fstat()``ed and must be a regular file carrying
    ``confirmed``'s ``(device, inode)``; this covers the object that is
    actually read, not just one look at the path before it was opened.
    On any rejection the descriptor is closed here, so no opened file is
    leaked by a failed confirmation.
    """
    try:
        opened = os.fstat(descriptor)
    except OSError as error:
        _close_quietly(descriptor)
        raise ConfirmationError(
            ConfirmReason.ACCESS_ERROR, ConfirmStage.PROVE, error=error
        ) from error
    if not stat.S_ISREG(opened.st_mode) or (
        opened.st_dev,
        opened.st_ino,
    ) != (confirmed.st_dev, confirmed.st_ino):
        _close_quietly(descriptor)
        raise ConfirmationError(
            ConfirmReason.WRONG_IDENTITY, ConfirmStage.PROVE, status=opened
        )
    return opened


class ConfirmedOpen(NamedTuple):
    """A successfully confirmed open.

    ``descriptor`` is the open read-only descriptor (owned by the
    caller), ``confirmed`` is the pre-open no-follow inspection the
    descriptor was proved against, and ``opened`` is the descriptor's
    own ``fstat()`` result — same object identity, but carrying the
    size used while streaming.
    """

    descriptor: int
    confirmed: os.stat_result
    opened: os.stat_result


def confirm_and_open_regular(
    path: str | os.PathLike[str],
    *,
    dir_fd: int | None = None,
    expected: tuple[int, int] | None = None,
    nonblock: bool = False,
) -> ConfirmedOpen:
    """Inspect, open without following links, and prove the descriptor.

    Returns the open descriptor together with the pre-open inspection
    and the descriptor's own status.  The descriptor is owned by the
    caller; every failure path closes it.  ``nonblock`` is passed to
    :func:`open_without_follow` so a FIFO that lands on the path after
    the inspection can never block the open: it opens at once and the
    descriptor proof below rejects it as non-regular.
    """
    confirmed_status = inspect_regular(path, dir_fd=dir_fd, expected=expected)
    descriptor = open_without_follow(path, dir_fd=dir_fd, nonblock=nonblock)
    opened = prove_opened_regular(descriptor, confirmed_status)
    return ConfirmedOpen(descriptor, confirmed_status, opened)


def hash_descriptor(descriptor: int) -> str:
    """Stream ``descriptor`` in chunks and return its full SHA-256 digest.

    The descriptor is wrapped with ``fdopen`` and released by the
    wrapper whether the read succeeds or raises (and closed if the
    wrapping itself fails), so a caller never has to reclaim it.
    """
    hasher = hashlib.sha256()
    try:
        stream = os.fdopen(descriptor, "rb")
    except BaseException:
        _close_quietly(descriptor)
        raise
    with stream:
        while True:
            chunk = stream.read(READ_CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    """Whether two stat results name the same object (device and inode)."""
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def same_identity_size_mtime(
    left: os.stat_result, right: os.stat_result
) -> bool:
    """Whether both results describe the same unchanged file.

    Beyond the object identity this compares the byte size and
    modification time (nanosecond precision), which import requires
    before and after a read so in-place rewrites, appends and truncation
    are rejected as well as replacements.
    """
    return (
        left.st_dev,
        left.st_ino,
        left.st_size,
        left.st_mtime_ns,
    ) == (
        right.st_dev,
        right.st_ino,
        right.st_size,
        right.st_mtime_ns,
    )


def _close_quietly(descriptor: int) -> None:
    try:
        os.close(descriptor)
    except OSError:
        pass
