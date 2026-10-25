"""Durable publication for the CuTe object cache.

Objects are written to a temporary file, fsynced, renamed into place, and the
directory entry is persisted, so a visible object file is always complete.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import suppress
from pathlib import Path


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        import ctypes as c

        kernel = c.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = (
            c.c_wchar_p,
            c.c_uint32,
            c.c_uint32,
            c.c_void_p,
            c.c_uint32,
            c.c_uint32,
            c.c_void_p,
        )
        kernel.CreateFileW.restype = c.c_void_p
        kernel.FlushFileBuffers.argtypes = (c.c_void_p,)
        kernel.CloseHandle.argtypes = (c.c_void_p,)
        # GENERIC_WRITE, shared read/write/delete, OPEN_EXISTING, BACKUP_SEMANTICS.
        handle = kernel.CreateFileW(str(path), 0x40000000, 7, None, 3, 0x02000000, None)
        if handle == c.c_void_p(-1).value:
            raise c.WinError(c.get_last_error())
        try:
            if not kernel.FlushFileBuffers(handle):
                raise c.WinError(c.get_last_error())
        finally:
            kernel.CloseHandle(handle)
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_durable(path: Path) -> None:
    if path.is_dir():
        return
    _mkdir_durable(path.parent)
    try:
        path.mkdir()
    except FileExistsError:
        if not path.is_dir():
            raise
    # Persist new shard directories as well as files within those shards.
    _fsync_directory(path.parent)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Publish a complete, fsynced file, then persist its directory entry.

    Writers must hold the compiler's per-key lock across object and manifest
    publication. Publish the manifest last; readers reject incomplete pairs.
    """
    _mkdir_durable(path.parent)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = stream.name
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            with suppress(OSError):
                os.unlink(temporary)
