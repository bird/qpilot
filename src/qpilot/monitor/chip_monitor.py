"""Real-time chip health monitoring via PilotOS pub-sub events."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from qpilot.enums import PubSubOperation
from qpilot.models.pubsub import (
    AnyPubSubEvent,
    CalibrationDoneEvent,
    CalibrationStartEvent,
    ChipProtectEvent,
    ChipUpdateEvent,
    ProbeEvent,
    TaskStatusEvent,
)
from qpilot.monitor.event_log import EventLog, LogEntry
from qpilot.monitor.qubit_tracker import QubitTracker
from qpilot.transport.subscriber import PubSubSubscriber

logger = logging.getLogger(__name__)


@dataclass
class ChipState:
    """Mutable snapshot of current chip status."""

    online: bool = False
    protected: bool = False
    last_updated: datetime | None = None
    active_threads: dict[str, dict[str, Any]] = field(default_factory=dict)
    queue_length: int = 0
    empty_threads: int = 0
    total_threads: int = 0
    calibrating: bool = False
    calibration_qubits: list[str] = field(default_factory=list)


class ChipMonitor:
    """Maintains live chip state by subscribing to PilotOS pub-sub events.

    Composes QubitTracker and EventLog for fidelity tracking and event history.
    """

    def __init__(
        self,
        subscriber: PubSubSubscriber,
        *,
        qubit_window: int = 100,
        max_events: int = 1000,
    ) -> None:
        self._subscriber = subscriber
        self._state = ChipState()
        self._qubit_tracker = QubitTracker(window=qubit_window)
        self._event_log = EventLog(max_entries=max_events)
        self._user_callbacks: list[Callable[[PubSubOperation, AnyPubSubEvent], Any]] = []

        # Register handlers for all event types
        subscriber.on(PubSubOperation.PROBE, self._on_probe)
        subscriber.on(PubSubOperation.CHIP_UPDATE, self._on_chip_update)
        subscriber.on(PubSubOperation.CALIBRATION_START, self._on_calibration_start)
        subscriber.on(PubSubOperation.CALIBRATION_DONE, self._on_calibration_done)
        subscriber.on(PubSubOperation.CHIP_PROTECT, self._on_chip_protect)
        subscriber.on(PubSubOperation.TASK_STATUS, self._on_task_status)

    @property
    def state(self) -> ChipState:
        """Current chip state snapshot."""
        return self._state

    @property
    def qubit_tracker(self) -> QubitTracker:
        return self._qubit_tracker

    @property
    def event_log(self) -> EventLog:
        return self._event_log

    def on_event(self, callback: Callable[[PubSubOperation, AnyPubSubEvent], Any]) -> None:
        """Register a callback for all events (for user-level notifications)."""
        self._user_callbacks.append(callback)

    def _notify_users(self, operation: PubSubOperation, event: AnyPubSubEvent) -> None:
        for cb in self._user_callbacks:
            try:
                cb(operation, event)
            except Exception:
                logger.exception("user callback error")

    def _on_probe(self, event: AnyPubSubEvent) -> None:
        assert isinstance(event, ProbeEvent)
        now = datetime.now(UTC)
        self._state.online = event.linked == 1
        self._state.queue_length = event.scheduler.queue_len
        self._state.empty_threads = event.core_status.empty_thread
        self._state.total_threads = event.core_status.thread_num
        self._state.last_updated = now

        # Track active thread details
        self._state.active_threads = {}
        for tid, tinfo in event.core_thread.items():
            self._state.active_threads[tid] = {
                "status": tinfo.status,
                "task_id": tinfo.task_id,
                "user": tinfo.user,
                "use_bits": tinfo.use_bits,
            }

        self._notify_users(PubSubOperation.PROBE, event)

    def _on_chip_update(self, event: AnyPubSubEvent) -> None:
        assert isinstance(event, ChipUpdateEvent)
        self._state.last_updated = datetime.now(UTC)
        self._event_log.append(
            LogEntry(
                timestamp=datetime.now(UTC),
                operation=PubSubOperation.CHIP_UPDATE,
                details={"update_flag": event.update_flag, "time": event.last_update_time},
            )
        )
        self._notify_users(PubSubOperation.CHIP_UPDATE, event)

    def _on_calibration_start(self, event: AnyPubSubEvent) -> None:
        assert isinstance(event, CalibrationStartEvent)
        self._state.calibrating = True
        self._state.calibration_qubits = event.qubits
        self._event_log.append(
            LogEntry(
                timestamp=datetime.now(UTC),
                operation=PubSubOperation.CALIBRATION_START,
                details={
                    "qubits": event.qubits,
                    "couplers": event.couplers,
                    "pairs": event.pairs,
                    "point_label": event.point_label,
                },
            )
        )
        self._notify_users(PubSubOperation.CALIBRATION_START, event)

    def _on_calibration_done(self, event: AnyPubSubEvent) -> None:
        assert isinstance(event, CalibrationDoneEvent)
        self._state.calibrating = False
        self._state.calibration_qubits = []
        self._event_log.append(
            LogEntry(
                timestamp=datetime.now(UTC),
                operation=PubSubOperation.CALIBRATION_DONE,
                details={
                    "qubits": event.qubits,
                    "config_flag": event.config_flag,
                    "point_label": event.point_label,
                },
            )
        )
        self._notify_users(PubSubOperation.CALIBRATION_DONE, event)

    def _on_chip_protect(self, event: AnyPubSubEvent) -> None:
        assert isinstance(event, ChipProtectEvent)
        self._state.protected = event.protect_flag
        self._event_log.append(
            LogEntry(
                timestamp=datetime.now(UTC),
                operation=PubSubOperation.CHIP_PROTECT,
                details={
                    "protect_flag": event.protect_flag,
                    "durative_time": event.durative_time,
                },
            )
        )
        self._notify_users(PubSubOperation.CHIP_PROTECT, event)

    def _on_task_status(self, event: AnyPubSubEvent) -> None:
        assert isinstance(event, TaskStatusEvent)
        self._notify_users(PubSubOperation.TASK_STATUS, event)
