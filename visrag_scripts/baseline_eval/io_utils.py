"""Atomic files and stable experiment identities."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import tempfile


def fingerprint(value: object) -> str:
    """Returns a deterministic JSON metadata fingerprint."""
    data = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:20]


def file_digest(path: str | Path) -> str:
    """Hashes contents without loading the whole file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def model_identity(path: str | Path) -> str:
    """Tracks configuration contents and weight file sizes/mtimes."""
    path = Path(path)
    if not (path / "config.json").is_file():
        raise ValueError(f"Missing model config: {path}")
    records = []
    for child in sorted(path.rglob("*")):
        if child.is_file() and ".cache" not in child.parts:
            stat = child.stat()
            row = [str(child.relative_to(path)), stat.st_size, stat.st_mtime_ns]
            if child.suffix in {".json", ".py"}:
                row.append(file_digest(child))
            records.append(row)
    return fingerprint(records)


@contextlib.contextmanager
def atomic_path(path):
    """Replaces the destination only after a successful write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=path.parent, suffix=path.suffix)
    os.close(handle)
    temporary = Path(name)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: object) -> None:
    """Writes JSON atomically."""
    with atomic_path(path) as temporary:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Rebuilds exports, avoiding torn append-only JSON lines."""
    with atomic_path(path) as temporary:
        with temporary.open("w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")


@contextlib.contextmanager
def run_lock(path):
    """Excludes concurrent Linux server runs for the same artifact."""
    import fcntl
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Experiment is already running.") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
