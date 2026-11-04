"""Append-only JSONL audit log.

Stores verdicts, rule names and redacted excerpts - never full prompts. A
security log that faithfully mirrors every request becomes a second copy
of the data it was meant to protect, and a resume pipeline's logs are full
of other people's personal information.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from .models import AuditRecord


class JsonlAuditLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.parent != Path(""):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def __call__(self, record: AuditRecord) -> None:
        line = json.dumps(record.to_log_line(), default=str, sort_keys=True)
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


class MemoryAuditLog:
    """For tests and the benchmark."""

    def __init__(self) -> None:
        self.records: list[AuditRecord] = []

    def __call__(self, record: AuditRecord) -> None:
        self.records.append(record)
