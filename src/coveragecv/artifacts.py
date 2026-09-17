"""Content-addressed artifacts and strict, path-safe integrity verification."""
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import rfc8785

from .schema import DiagnosticError


def canonical(value) -> bytes:
    return rfc8785.dumps(value)


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_child(root: Path, relative: str) -> Path:
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts or "\\" in relative or not relative:
        raise DiagnosticError("UNSAFE_PATH", "expected a relative path within the declared root", path=relative)
    resolved = (root / p).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise DiagnosticError("UNSAFE_PATH", "path or symlink escapes its declared root", path=relative)
    return resolved


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical(value) + b"\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".write-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_jsonl(path: Path, rows):
    path.write_bytes(b"".join(canonical(row) + b"\n" for row in rows))


def publish(staging: Path, output: Path, metadata: dict) -> Path:
    files = {
        p.relative_to(staging).as_posix(): file_digest(p)
        for p in sorted(staging.rglob("*")) if p.is_file()
    }
    identity = {**metadata, "files": files}
    sha = digest(identity)
    write_json(staging / "manifest.json", {**identity, "digest": sha})
    destination = output / sha
    output.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        verify(destination)
        shutil.rmtree(staging)
        return destination
    try:
        os.rename(staging, destination)
    except OSError:
        if not destination.exists():
            raise
        verify(destination)
        shutil.rmtree(staging)
    verify(destination)
    return destination


def stage(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=".staging-", dir=output))


def verify(root: Path) -> dict:
    root = root.resolve()
    manifest = read_json(root / "manifest.json")
    sha = manifest.get("digest")
    identity = {k: v for k, v in manifest.items() if k != "digest"}
    if manifest.get("schema_version") != 1 or digest(identity) != sha:
        raise DiagnosticError("BUNDLE_DIGEST_MISMATCH", "manifest identity is invalid")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    expected = set(manifest["files"]) | {"manifest.json"}
    if actual != expected:
        raise DiagnosticError("BUNDLE_DIGEST_MISMATCH", "artifact file inventory differs from manifest")
    for name, expected_sha in manifest["files"].items():
        p = safe_child(root, name)
        if p.is_symlink() or not p.is_file() or file_digest(p) != expected_sha:
            raise DiagnosticError("BUNDLE_DIGEST_MISMATCH", "artifact payload was modified", path=name)
    return manifest
