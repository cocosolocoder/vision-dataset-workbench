"""Pinning a read to one confirmed regular file.

The single rule shared by single-file import, directory import and
``--source-dir`` export: *the bytes actually read must come from the one
regular file that was confirmed*.  The path is inspected without
following a final symlink (``lstat``/``fstatat``), the file is opened
with ``O_NOFOLLOW`` so a symlink swapped onto the path in the meantime
cannot redirect the read, and — for reads that must be pinned to a
specific confirmed inode (both import entry points and the export copy)
— the opened descriptor is ``fstat``\\ ed to prove it carries that
file's identity (device and inode) and is regular, before any stream
reaches a caller.

Each entry point keeps its own error wording, its own read loop and its
own post-open checks, passed in as a :class:`ConfirmationPolicy`:
import re-lstats the path after the read and rejects in-place
identity/size/mtime changes too, while export — once the descriptor is
open and proved — keeps reading the confirmed inode even if another file
later occupies the path.  Those differences live in the callers; this
module owns only the confirm-then-open sequence itself.
"""

from __future__ import annotations

import errno
import os
import stat
from dataclasses import dataclass
from typing import Any, Callable

# Descriptor-relative, no-symlink-follow opens: POSIX and available on
# the platforms this tool runs on (0 where a flag is absent).
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


class ConfirmedFile:
    """A regular file confirmed at a path and pinned through the open.

    ``confirmed_stat`` is the no-follow stat result captured at
    confirmation time: ``st_dev``/``st_ino`` identify the confirmed
    regular inode and ``st_size`` is that confirmed file's size.  The
    binary :attr:`stream` reads from an ``O_RDONLY`` descriptor reached
    without following a final symlink (and, when the caller asked for
    pinning, proved to belong to that exact inode), so bytes can only
    come from the confirmed object even if its path is remapped
    afterwards.  Close the stream — or use the file as a context
    manager — to release the descriptor.
    """

    __slots__ = ("stat", "proven_stat", "stream")

    def __init__(
        self,
        confirmed_stat: os.stat_result,
        descriptor: int,
        proven_stat: os.stat_result | None = None,
    ) -> None:
        self.stat = confirmed_stat
        self.proven_stat = proven_stat
        try:
            self.stream = os.fdopen(descriptor, "rb")
        except BaseException:
            os.close(descriptor)
            raise

    @property
    def identity(self) -> tuple[int, int]:
        """The confirmed ``(device, inode)``."""
        return (self.stat.st_dev, self.stat.st_ino)

    @property
    def size(self) -> int:
        """The confirmed file's size in bytes."""
        return self.stat.st_size

    def close(self) -> None:
        self.stream.close()

    def __enter__(self) -> "ConfirmedFile":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


@dataclass(frozen=True)
class ConfirmationPolicy:
    """How one entry point words and types confirmation failures.

    Each factory receives the display object the entry point uses to
    identify the file in errors (an absolute path, a plan-relative
    candidate path, ...) and builds that entry point's own exception, so
    the shared sequence never dictates wording or exception class.
    ``open_error`` additionally receives the errno from the failed open,
    letting an entry distinguish a symlink swapped in after the stat.
    """

    #: The no-follow stat could not be obtained: ``(display, OSError)``.
    stat_error: Callable[[Any, OSError], Exception]
    #: The ``O_NOFOLLOW`` open failed: ``(display, OSError)``; inspect
    #: ``errno`` (``ELOOP`` marks a symlink swapped in after the stat).
    open_error: Callable[[Any, OSError], Exception]
    #: ``fstat`` of the opened descriptor failed: ``(display, OSError)``.
    fstat_error: Callable[[Any, OSError], Exception]
    #: The object is not a regular file on a first, expectation-less
    #: confirmation (a symlink seen here is reported the same way).
    not_regular_error: Callable[[Any], Exception]
    #: A previously confirmed selection is now a symlink.
    symlink_error: Callable[[Any], Exception]
    #: A previously confirmed selection now carries another identity.
    identity_error: Callable[[Any], Exception]
    #: The opened descriptor is not the confirmed regular inode.
    opened_identity_error: Callable[[Any], Exception]


def confirm_and_open(
    path: Any,
    policy: ConfirmationPolicy,
    *,
    dir_fd: int | None = None,
    expected_identity: tuple[int, int] | None = None,
    display: Any = None,
    prove_opened: bool = True,
) -> ConfirmedFile:
    """Confirm the object at ``path`` and open it pinned to that object.

    ``path`` is inspected without following a final symlink: ``lstat``
    for a plain path, ``fstatat`` relative to ``dir_fd`` for a
    descriptor-relative name.  A selection already confirmed elsewhere
    passes its ``(device, inode)`` as ``expected_identity``; the current
    object must still carry it.  The path is then opened with
    ``O_NOFOLLOW``.

    With ``prove_opened`` (the default, used by both import entry points
    and the export package copy) the descriptor is also ``fstat``\\ ed
    and proved to be a regular file with the confirmed identity — the
    check covers the object that is actually read, not just one look at
    the path before it opens.  The export-time content lookup skips this
    last proof: it records the ``fstatat`` identity itself and the copy
    re-pins that exact inode independently.

    A different regular file renamed onto the path — even one with
    identical content, size and modification time — or a symlink, even
    one pointing at the original file, fails through the policy; so does
    a non-regular object and any stat/open/fstat error.  A descriptor
    obtained before a failure is always closed.
    """
    shown = path if display is None else display
    try:
        if dir_fd is None:
            before = os.lstat(path)
        else:
            before = os.stat(
                path, dir_fd=dir_fd, follow_symlinks=False
            )
    except OSError as error:
        raise policy.stat_error(shown, error) from error

    if stat.S_ISLNK(before.st_mode):
        if expected_identity is not None:
            # A confirmed selection whose path is now a link — even one
            # pointing at the original inode — must not be followed.
            raise policy.symlink_error(shown)
        # A first confirmation never follows a link: the caller asked to
        # read the regular file at this path, not its target.
        raise policy.not_regular_error(shown)
    if not stat.S_ISREG(before.st_mode):
        raise policy.not_regular_error(shown)
    if expected_identity is not None and (
        before.st_dev,
        before.st_ino,
    ) != expected_identity:
        # The path now names another regular file; same content, size or
        # mtime does not make it the confirmed one.
        raise policy.identity_error(shown)

    open_kwargs: dict[str, Any] = {}
    if dir_fd is not None:
        open_kwargs["dir_fd"] = dir_fd
    try:
        # O_NOFOLLOW makes the open itself refuse a symlink swapped onto
        # the path after the no-follow inspection above.
        descriptor = os.open(path, os.O_RDONLY | _O_NOFOLLOW, **open_kwargs)
    except OSError as error:
        raise policy.open_error(shown, error) from error

    if prove_opened:
        try:
            opened = os.fstat(descriptor)
        except OSError as error:
            os.close(descriptor)
            raise policy.fstat_error(shown, error) from error
        # The path may have been remapped in the stat-to-open gap; the
        # descriptor must belong to the confirmed inode or the bytes read
        # would not come from the file this operation agreed to read.
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_dev,
            opened.st_ino,
        ) != (before.st_dev, before.st_ino):
            os.close(descriptor)
            raise policy.opened_identity_error(shown)
    else:
        opened = None
    return ConfirmedFile(before, descriptor, opened)
