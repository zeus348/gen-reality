"""Publish immutable artifacts in an application-owned directory on Unix."""

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import re
import stat
from uuid import uuid4


MAX_ARTIFACT_BYTES = 512 * 1024 * 1024


class StorageError(ValueError):
    """An artifact cannot be published or verified."""


class ArtifactExistsError(StorageError):
    """The requested artifact already exists and must not be replaced."""


@dataclass(frozen=True)
class StoredArtifact:
    path: str
    sha256: str
    size: int


def _path_parts(relative_path: str) -> list[str]:
    if type(relative_path) is not str or len(relative_path) > 1024:
        raise StorageError("Artifact path must be a relative path of at most 1024 characters.")
    parts = relative_path.split("/")
    if len(parts) > 16 or any(re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9._-]{0,127}",
                                         part) is None for part in parts):
        raise StorageError("Artifact path contains an invalid component.")
    return parts


@contextmanager
def _parent_directory(root: Path, parts: list[str], *, create: bool):
    """Walk directory handles without following symlinks; root is server-owned."""
    directory = None
    try:
        root_parts = Path(root).absolute().parts
        directory = os.open(root_parts[0], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        for part in (*root_parts[1:], *parts[:-1]):
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=directory)
                except FileExistsError:
                    pass
                # Another writer may have created this entry but not synced it.
                os.fsync(directory)
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=directory)
            os.close(directory)
            directory = child
        yield directory
    finally:
        if directory is not None:
            os.close(directory)


def publish_bytes(root: Path, relative_path: str, contents: bytes) -> StoredArtifact:
    """Write, fsync, and link a complete file exactly once; never replace a path.

    The temporary file and target share a directory. A hard link publishes the
    complete bytes atomically and fails if another writer already owns the name.
    The application must keep the root and its ancestors private and trusted.
    """
    parts = _path_parts(relative_path)
    if type(contents) is not bytes or len(contents) > MAX_ARTIFACT_BYTES:
        raise StorageError("Artifact contents must be bytes of at most 512 MiB.")
    try:
        with _parent_directory(root, parts, create=True) as directory:
            temporary = ".pending-" + uuid4().hex
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                                 | os.O_NOFOLLOW, 0o600, dir_fd=directory)
            try:
                with os.fdopen(descriptor, "wb") as file:
                    file.write(contents)
                    file.flush()
                    os.fsync(file.fileno())
                os.link(temporary, parts[-1], src_dir_fd=directory,
                        dst_dir_fd=directory, follow_symlinks=False)
                os.fsync(directory)
            finally:
                os.unlink(temporary, dir_fd=directory)
                os.fsync(directory)
    except FileExistsError:
        raise ArtifactExistsError("Artifact already exists; use a new attempt.") from None
    except OSError:
        raise StorageError("Artifact cannot be published safely.") from None
    return StoredArtifact(relative_path, sha256(contents).hexdigest(), len(contents))


def read_verified(root: Path, artifact: StoredArtifact) -> bytes:
    """Read only a regular file whose size and content match its record."""
    parts = _path_parts(artifact.path)
    if (type(artifact.size) is not int or not 0 <= artifact.size <= MAX_ARTIFACT_BYTES
            or type(artifact.sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", artifact.sha256) is None):
        raise StorageError("Artifact record has an invalid size or SHA-256 hash.")
    try:
        with _parent_directory(root, parts, create=False) as directory:
            descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                 dir_fd=directory)
            with os.fdopen(descriptor, "rb") as file:
                info = os.fstat(file.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size != artifact.size:
                    raise StorageError("Artifact is not a regular file of the recorded size.")
                contents = file.read(artifact.size + 1)
    except OSError:
        raise StorageError("Artifact cannot be read safely.") from None
    if len(contents) != artifact.size or sha256(contents).hexdigest() != artifact.sha256:
        raise StorageError("Artifact content does not match its record.")
    return contents
