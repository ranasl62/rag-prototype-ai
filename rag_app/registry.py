from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from rag_app.config import ensure_dirs, settings


def _read_registry(path: Path) -> List[dict]:
    if not path.exists():
        return []
    entries: List[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entries.append(json.loads(line))
    return entries


def list_documents() -> List[dict]:
    ensure_dirs()
    return _read_registry(settings.registry_file)


def find_by_hash(file_hash: str) -> Optional[dict]:
    for entry in list_documents():
        if entry.get("sha256") == file_hash:
            return entry
    return None


def register_document(entry: Dict) -> None:
    ensure_dirs()
    payload = dict(entry)
    payload["ingested_at"] = datetime.now(timezone.utc).isoformat()
    line = json.dumps(payload, ensure_ascii=True)
    settings.registry_file.write_text(
        (settings.registry_file.read_text(encoding="utf-8") if settings.registry_file.exists() else "")
        + line
        + "\n",
        encoding="utf-8",
    )
