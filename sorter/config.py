from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import tomllib
from dataclasses import asdict, dataclass


class SorterError(Exception):
    """An operational error whose message is safe to print."""


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=".sorter-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class Settings:
    query: str = "newer_than:365d -in:chats -in:sent -in:drafts"
    ollama_url: str = "http://127.0.0.1:11434"
    embedding_model: str = "nomic-embed-text:latest"
    chat_model: str = "qwen3.5:0.8b"
    batch_size: int = 16
    context_length: int = 4096
    min_cluster_size: int = 15
    min_samples: int = 5
    similarity_threshold: float = 0.88
    similarity_margin: float = 0.10
    fallback_similarity_floor: float = 0.50
    llm_confidence_threshold: float = 0.85

    @classmethod
    def load(cls, root: Path) -> Settings:
        path = root / "sorter.toml"
        try:
            raw = tomllib.loads(path.read_text()) if path.exists() else {}
            result = cls(**raw)
            for name in ("query", "ollama_url", "embedding_model", "chat_model"):
                if not isinstance(getattr(result, name), str) or not getattr(result, name).strip():
                    raise ValueError(name)
            for name, low, high in (
                ("batch_size", 1, 64),
                ("context_length", 1024, 32768),
                ("min_cluster_size", 2, 1000),
                ("min_samples", 1, 1000),
            ):
                val = getattr(result, name)
                if type(val) is not int or not low <= val <= high:
                    raise ValueError(name)
            for name in (
                "similarity_threshold",
                "similarity_margin",
                "fallback_similarity_floor",
                "llm_confidence_threshold",
            ):
                val = getattr(result, name)
                if type(val) not in (float, int) or not 0 <= val <= 1:
                    raise ValueError(name)
            return result
        except (ValueError, TypeError, OSError) as exc:
            raise SorterError(
                "Invalid sorter.toml. Check its field names, types, and ranges."
            ) from exc

    def fingerprint(self) -> str:
        return digest(asdict(self))

    def example(self) -> str:
        return (
            "\n".join(f"{key} = {json.dumps(value)}" for key, value in asdict(self).items()) + "\n"
        )


class Paths:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.data = self.root / "data"
        self.reports = self.root / "reports"
        self.database = self.data / "mail.db"
        self.credentials = self.root / "credentials.json"
        self.categories = self.root / "categories.yaml"
        self.proposals = self.data / "categories.proposed.yaml"

    def prepare(self):
        private_dir(self.data)
        private_dir(self.reports)

    def token(self, write=False):
        return self.data / ("token-modify.json" if write else "token-readonly.json")


class RunLock:
    def __init__(self, path: Path):
        self.path = path
        self.file = None

    def __enter__(self):
        import fcntl

        self.file = self.path.open("a+")
        self.path.chmod(0o600)
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.file.close()
            raise SorterError("Another sorter command is running for this project.") from exc
        return self

    def __exit__(self, *args):
        self.file.close()
