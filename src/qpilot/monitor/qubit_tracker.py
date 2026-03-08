"""Rolling-window qubit fidelity tracker."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass
class FidelitySample:
    """A single fidelity measurement."""

    fidelity: float
    timestamp: datetime


class QubitTracker:
    """Tracks per-qubit fidelity over a rolling window.

    Thread-safe within a single asyncio event loop (no locks needed).
    """

    def __init__(self, window: int = 100) -> None:
        self._window = window
        self._data: dict[str, deque[FidelitySample]] = {}

    def record(self, qubit_id: str, fidelity: float, timestamp: datetime | None = None) -> None:
        """Record a fidelity measurement for a qubit."""
        ts = timestamp or datetime.now(UTC)
        if qubit_id not in self._data:
            self._data[qubit_id] = deque(maxlen=self._window)
        self._data[qubit_id].append(FidelitySample(fidelity=fidelity, timestamp=ts))

    def latest(self, qubit_id: str) -> float | None:
        """Get the most recent fidelity for a qubit."""
        samples = self._data.get(qubit_id)
        if not samples:
            return None
        return samples[-1].fidelity

    def average(self, qubit_id: str, last_n: int | None = None) -> float | None:
        """Get average fidelity, optionally over the last N samples."""
        samples = self._data.get(qubit_id)
        if not samples:
            return None
        if last_n is not None:
            subset = list(samples)[-last_n:]
        else:
            subset = list(samples)
        return sum(s.fidelity for s in subset) / len(subset)

    def all_qubits(self) -> list[str]:
        """List all tracked qubit IDs."""
        return list(self._data.keys())

    def snapshot(self) -> dict[str, float]:
        """Get latest fidelity for all tracked qubits."""
        result = {}
        for qid, samples in self._data.items():
            if samples:
                result[qid] = samples[-1].fidelity
        return result
