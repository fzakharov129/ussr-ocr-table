"""Atomic, versioned checkpoints. Load only trusted local pickle files."""

import hashlib
import os
import pickle
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .exceptions import StorageError
from .validations import StorageChecks


STAGES = ("detection", "ocr", "review", "normalization", "header_review", "structure", "assembly", "export")
FORMAT_VERSION = 1


def book_output_dir(zip_path: str | Path, outputs_dir=None, book_id=None) -> Path:
    """Resolve the same book folder for processing and inspection."""
    book_id = Path(zip_path).stem if book_id is None else book_id
    StorageChecks.book_id(book_id)
    root = Path(__file__).resolve().parent.parent
    return (Path(outputs_dir) if outputs_dir is not None else root / "outputs") / book_id


def source_fingerprint(zip_path: str | Path) -> dict:
    path = Path(zip_path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size": path.stat().st_size}


class BookStorage:
    """All completed stages for one book, plus its final Python result."""

    def __init__(self, output_dir: str | Path, fingerprint: dict):
        self.output_dir = Path(output_dir)
        self.fingerprint = fingerprint

    def path(self, stage: str) -> Path:
        StorageChecks.stage(stage, STAGES)
        if stage == "assembly":
            return self.output_dir / "result.pkl"
        return self.output_dir / "checkpoints" / f"{stage}.pkl"

    @contextmanager
    def locked(self):
        """Reject concurrent writers; retain lock after an abrupt process crash."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        lock = self.output_dir / ".pipeline.lock"
        with StorageChecks.error_context("lock", lock):
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(f"pid={os.getpid()}\n")
            yield self
        finally:
            lock.unlink(missing_ok=True)

    def save(self, stage: str, data) -> Path:
        path = self.path(stage)
        envelope = {
            "format": "ussr-ocr-table-checkpoint",
            "version": FORMAT_VERSION,
            "stage": stage,
            "source": self.fingerprint,
            "data": data,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with StorageChecks.error_context("save", path, stage):
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False,
                ) as stream:
                    temporary = Path(stream.name)
                    pickle.dump(envelope, stream, protocol=pickle.HIGHEST_PROTOCOL)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return path

    def load(self, stage: str):
        path = self.path(stage)
        with StorageChecks.error_context("load", path, stage):
            with path.open("rb") as stream:
                envelope = pickle.load(stream)
                StorageChecks.trailing(stream.read(1))
        StorageChecks.envelope(envelope, stage, self.fingerprint, path, FORMAT_VERSION)
        return envelope["data"]

    def invalidate_from(self, stage: str) -> None:
        """Remove stale snapshots before rerunning this stage, keeping its input."""
        StorageChecks.stage(stage, STAGES, checkpoint_only=False)
        for name in STAGES[STAGES.index(stage):-1]:
            self.path(name).unlink(missing_ok=True)
