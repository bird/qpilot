"""Tests for the monitoring subsystem."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
import zmq.asyncio

from qpilot.enums import PubSubOperation
from qpilot.monitor.chip_monitor import ChipMonitor
from qpilot.monitor.event_log import EventLog, LogEntry
from qpilot.monitor.qubit_tracker import QubitTracker
from qpilot.transport.subscriber import PubSubSubscriber
from tests.conftest import FakePublisher


class TestQubitTracker:
    def test_record_and_latest(self):
        tracker = QubitTracker(window=10)
        tracker.record("q0", 0.99)
        tracker.record("q0", 0.98)
        assert tracker.latest("q0") == 0.98

    def test_average(self):
        tracker = QubitTracker(window=10)
        tracker.record("q0", 0.90)
        tracker.record("q0", 1.00)
        assert tracker.average("q0") == pytest.approx(0.95)

    def test_average_last_n(self):
        tracker = QubitTracker(window=10)
        tracker.record("q0", 0.80)
        tracker.record("q0", 0.90)
        tracker.record("q0", 1.00)
        assert tracker.average("q0", last_n=2) == pytest.approx(0.95)

    def test_unknown_qubit_returns_none(self):
        tracker = QubitTracker()
        assert tracker.latest("q99") is None
        assert tracker.average("q99") is None

    def test_all_qubits(self):
        tracker = QubitTracker()
        tracker.record("q0", 0.99)
        tracker.record("q1", 0.98)
        assert set(tracker.all_qubits()) == {"q0", "q1"}

    def test_snapshot(self):
        tracker = QubitTracker()
        tracker.record("q0", 0.99)
        tracker.record("q1", 0.95)
        snap = tracker.snapshot()
        assert snap == {"q0": 0.99, "q1": 0.95}

    def test_window_eviction(self):
        tracker = QubitTracker(window=3)
        for i in range(5):
            tracker.record("q0", float(i))
        # Only last 3 should remain
        assert tracker.average("q0") == pytest.approx(3.0)  # (2+3+4)/3


class TestEventLog:
    def test_append_and_query(self):
        log = EventLog()
        entry = LogEntry(
            timestamp=datetime.now(UTC),
            operation=PubSubOperation.CHIP_UPDATE,
            details={"flag": True},
        )
        log.append(entry)
        assert log.count == 1
        results = log.query()
        assert len(results) == 1

    def test_query_by_operation(self):
        log = EventLog()
        log.append(
            LogEntry(
                timestamp=datetime.now(UTC),
                operation=PubSubOperation.CHIP_UPDATE,
            )
        )
        log.append(
            LogEntry(
                timestamp=datetime.now(UTC),
                operation=PubSubOperation.CHIP_PROTECT,
            )
        )
        results = log.query(operation=PubSubOperation.CHIP_UPDATE)
        assert len(results) == 1
        assert results[0].operation == PubSubOperation.CHIP_UPDATE

    def test_max_entries(self):
        log = EventLog(max_entries=5)
        for i in range(10):
            log.append(
                LogEntry(
                    timestamp=datetime.now(UTC),
                    operation=PubSubOperation.PROBE,
                    details={"i": i},
                )
            )
        assert log.count == 5

    def test_latest(self):
        log = EventLog()
        for i in range(5):
            log.append(
                LogEntry(
                    timestamp=datetime.now(UTC),
                    operation=PubSubOperation.PROBE,
                    details={"i": i},
                )
            )
        latest = log.latest(n=2)
        assert len(latest) == 2
        assert latest[-1].details["i"] == 4


class TestChipMonitor:
    @pytest.fixture
    def endpoint(self):
        return "tcp://127.0.0.1:15900"

    @pytest.fixture
    async def zmq_ctx(self):
        ctx = zmq.asyncio.Context()
        yield ctx
        ctx.destroy(linger=0)

    @pytest.fixture
    async def publisher(self, zmq_ctx, endpoint):
        pub = FakePublisher(zmq_ctx, endpoint)
        await pub.start()
        yield pub
        await pub.stop()

    @pytest.fixture
    async def monitor(self, zmq_ctx, endpoint, publisher):
        sub = PubSubSubscriber(zmq_ctx, "127.0.0.1", 15900)
        monitor = ChipMonitor(sub)
        await sub.start()
        await asyncio.sleep(0.1)  # let subscription establish
        yield monitor
        await sub.stop()

    async def test_probe_updates_state(self, monitor, publisher):
        await publisher.publish(
            "probe",
            {
                "inst_status": 1,
                "linked": 1,
                "timestamp": 100.0,
                "scheduler": {"status": "ok", "queue_len": 5},
                "core_status": {"empty_thread": 2, "pause_read": 0, "thread_num": 5},
                "core_thread": {
                    "t0": {
                        "status": "running",
                        "thread_id": "t0",
                        "task_id": "task1",
                        "start_time": 100.0,
                        "user": "admin",
                        "use_bits": ["q0", "q1"],
                    }
                },
            },
        )
        await asyncio.sleep(0.15)

        state = monitor.state
        assert state.online is True
        assert state.queue_length == 5
        assert state.empty_threads == 2
        assert "t0" in state.active_threads

    async def test_calibration_lifecycle(self, monitor, publisher):
        await publisher.publish(
            "calibration_start",
            {
                "config_flag": False,
                "qubits": ["q0", "q1"],
                "couplers": ["c0-1"],
                "pairs": ["q0q1"],
                "discriminators": ["q0_01.bin"],
                "point_label": 2,
            },
        )
        await asyncio.sleep(0.15)

        assert monitor.state.calibrating is True
        assert monitor.state.calibration_qubits == ["q0", "q1"]

        await publisher.publish(
            "calibration_done",
            {
                "config_flag": True,
                "qubits": ["q0", "q1"],
                "couplers": ["c0-1"],
                "pairs": ["q0q1"],
                "discriminators": ["q0", "q1"],
                "point_label": 2,
            },
        )
        await asyncio.sleep(0.15)

        assert monitor.state.calibrating is False
        assert monitor.event_log.count == 2

    async def test_chip_protect(self, monitor, publisher):
        await publisher.publish(
            "chip_protect",
            {
                "ProtectFlag": True,
                "DurativeTime": 10,
                "LastTime": 9999,
            },
        )
        await asyncio.sleep(0.15)

        assert monitor.state.protected is True

    async def test_user_callback(self, monitor, publisher):
        events = []
        monitor.on_event(lambda op, ev: events.append((op, ev)))

        await publisher.publish(
            "chip_update",
            {
                "UpdateFlag": True,
                "LastUpdateTime": 5555,
            },
        )
        await asyncio.sleep(0.15)

        assert len(events) == 1
        assert events[0][0] == PubSubOperation.CHIP_UPDATE
