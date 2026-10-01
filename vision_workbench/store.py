from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STATE_DIRECTORY = ".vision-workbench"
MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class ImportResult:
    digest: str
    added: bool


class DatasetStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.state_directory = self.root / STATE_DIRECTORY
        self.manifest_path = self.state_directory / MANIFEST_NAME

    def initialize(self) -> None:
        self.state_directory.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            self._write({"schema_version": 1, "items": []})

    def add(self, source: Path, label: str | None = None) -> ImportResult:
        self.initialize()
        file_path = source.resolve(strict=True)
        if not file_path.is_file():
            raise ValueError(f"Not a regular file: {source}")
        digest = self._digest(file_path)
        manifest = self._read()
        if any(item["sha256"] == digest for item in manifest["items"]):
            return ImportResult(digest=digest, added=False)
        manifest["items"].append(
            {
                "sha256": digest,
                "source": str(file_path),
                "size": file_path.stat().st_size,
                "label": label,
            }
        )
        manifest["items"].sort(key=lambda item: item["sha256"])
        self._write(manifest)
        return ImportResult(digest=digest, added=True)

    def summary(self) -> dict[str, Any]:
        self.initialize()
        items = self._read()["items"]
        labels = Counter(item["label"] or "unlabeled" for item in items)
        return {
            "items": len(items),
            "bytes": sum(item["size"] for item in items),
            "labels": dict(sorted(labels.items())),
        }

    def items(self) -> list[dict[str, Any]]:
        return self._read()["items"]

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot read dataset manifest: {error}") from error
        if data.get("schema_version") != 1 or not isinstance(data.get("items"), list):
            raise ValueError("Unsupported dataset manifest")
        return data

    def _write(self, data: dict[str, Any]) -> None:
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.manifest_path)

    @staticmethod
    def _digest(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
