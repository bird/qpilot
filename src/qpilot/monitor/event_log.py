"""Ordered event log for calibration and maintenance events."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from qpilot.enums import PubSubOperation


@dataclass
class LogEntry:
    """A single logged event."""

    timestamp: datetime
    operation: PubSubOperation
    details: dict[str, Any] = field(default_factory=dict)


class EventLog:
    """Bounded event log with query support.

    Stores calibration, maintenance, and chip events in insertion order.
    """

    def __init__(self, max_entries: int = 1000) -> None:
        self._entries: deque[LogEntry] = deque(maxlen=max_entries)

    def append(self, entry: LogEntry) -> None:
        """Add an event to the log."""
        self._entries.append(entry)

    def query(
        self,
        *,
        operation: PubSubOperation | None = None,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[LogEntry]:
        """Query events, optionally filtered by operation type and time."""
        results: list[LogEntry] = []
        for entry in self._entries:
            if operation is not None and entry.operation != operation:
                continue
            if since is not None and entry.timestamp < since:
                continue
            results.append(entry)
            if limit is not None and len(results) >= limit:
                break
        return results

    @property
    def count(self) -> int:
        return len(self._entries)

    def latest(self, n: int = 10) -> list[LogEntry]:
        """Get the N most recent entries."""
        entries = list(self._entries)
        return entries[-n:]
