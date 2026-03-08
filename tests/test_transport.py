"""Tests for ZMQ transport layer — dealer client and pub-sub subscriber."""

from __future__ import annotations

import asyncio

import pytest

from qpilot.models.requests import (
    GetChipConfigRequest,
    GetRBDataRequest,
    HeartbeatRequest,
    TaskStatusRequest,
)
from qpilot.models.responses import (
    GetChipConfigResponse,
    GetRBDataResponse,
    HeartbeatResponse,
    TaskStatusResponse,
)
from qpilot.transport.dealer import DealerClient
from qpilot.transport.subscriber import PubSubSubscriber
from qpilot.enums import PubSubOperation
from qpilot.models.pubsub import ProbeEvent, ChipUpdateEvent

from tests.conftest import FakePublisher, FakeRouter


class TestDealerClient:
    @pytest.fixture
    def endpoint(self):
        return "tcp://127.0.0.1:15700"

    @pytest.fixture
    async def router(self, zmq_ctx, endpoint):
        """ROUTER socket bound to test endpoint."""
        router = FakeRouter(zmq_ctx, endpoint)
        await router.start()
        yield router
        await router.stop()

    @pytest.fixture
    async def dealer(self, zmq_ctx, endpoint, router):
        """DealerClient connected to test router."""
        client = DealerClient(
            zmq_ctx,
            "127.0.0.1",
            15700,
            request_timeout=5.0,
            heartbeat_interval=60.0,  # disable auto-heartbeat during tests
        )
        await client.start()
        yield client
        await client.stop()

    async def test_heartbeat(self, dealer):
        req = HeartbeatRequest(chip_id=72, timestamp=1000)
        resp = await dealer.send_request(req)
        assert isinstance(resp, HeartbeatResponse)
        assert resp.backend == 72

    async def test_task_status(self, dealer):
        req = TaskStatusRequest(task_id="TEST123")
        resp = await dealer.send_request(req)
        assert isinstance(resp, TaskStatusResponse)
        assert resp.task_id == "TEST123"
        assert resp.task_status == 5

    async def test_rb_data(self, dealer):
        req = GetRBDataRequest(chip_id=72)
        resp = await dealer.send_request(req)
        assert isinstance(resp, GetRBDataResponse)
        assert resp.single_gate_fidelity.fidelity == [0.99]

    async def test_chip_config(self, dealer):
        req = GetChipConfigRequest(chip_id=72)
        resp = await dealer.send_request(req)
        assert isinstance(resp, GetChipConfigResponse)
        assert 1 in resp.point_label_list

    async def test_sn_correlation(self, dealer):
        """Multiple concurrent requests should get correct responses."""
        reqs = [HeartbeatRequest(chip_id=72, timestamp=i) for i in range(5)]
        results = await asyncio.gather(*(dealer.send_request(r) for r in reqs))
        assert len(results) == 5
        assert all(isinstance(r, HeartbeatResponse) for r in results)


class TestPubSubSubscriber:
    @pytest.fixture
    def endpoint(self):
        return "tcp://127.0.0.1:15800"

    @pytest.fixture
    async def publisher(self, zmq_ctx, endpoint):
        pub = FakePublisher(zmq_ctx, endpoint)
        await pub.start()
        yield pub
        await pub.stop()

    @pytest.fixture
    async def subscriber(self, zmq_ctx, endpoint, publisher):
        sub = PubSubSubscriber(
            zmq_ctx,
            "127.0.0.1",
            15800,
            topics=[b"simulator_topic"],
        )
        await sub.start()
        # Give ZMQ time to establish subscription
        await asyncio.sleep(0.1)
        yield sub
        await sub.stop()

    async def test_probe_event(self, subscriber, publisher):
        received = []
        subscriber.on(PubSubOperation.PROBE, lambda e: received.append(e))

        await publisher.publish(
            "probe",
            {
                "inst_status": 1,
                "linked": 1,
                "timestamp": 100.0,
                "scheduler": {"status": "ok", "queue_len": 2},
                "core_status": {"empty_thread": 3, "pause_read": 0, "thread_num": 5},
                "core_thread": {},
            },
        )
        await asyncio.sleep(0.15)

        assert len(received) == 1
        assert isinstance(received[0], ProbeEvent)
        assert received[0].linked == 1

    async def test_chip_update_event(self, subscriber, publisher):
        received = []
        subscriber.on(PubSubOperation.CHIP_UPDATE, lambda e: received.append(e))

        await publisher.publish(
            "chip_update",
            {
                "UpdateFlag": True,
                "LastUpdateTime": 9999,
            },
        )
        await asyncio.sleep(0.15)

        assert len(received) == 1
        assert isinstance(received[0], ChipUpdateEvent)
        assert received[0].update_flag is True

    async def test_multiple_handlers(self, subscriber, publisher):
        results_a = []
        results_b = []
        subscriber.on(PubSubOperation.PROBE, lambda e: results_a.append(e))
        subscriber.on(PubSubOperation.PROBE, lambda e: results_b.append(e))

        await publisher.publish(
            "probe",
            {
                "inst_status": 1,
                "linked": 1,
                "timestamp": 100.0,
                "scheduler": {"status": "ok", "queue_len": 0},
                "core_status": {"empty_thread": 5, "pause_read": 0, "thread_num": 5},
                "core_thread": {},
            },
        )
        await asyncio.sleep(0.15)

        assert len(results_a) == 1
        assert len(results_b) == 1

    async def test_bad_handler_doesnt_crash(self, subscriber, publisher):
        """A misbehaving handler should not kill the dispatch loop."""
        good_results = []

        def bad_handler(e):
            raise ValueError("oops")

        subscriber.on(PubSubOperation.PROBE, bad_handler)
        subscriber.on(PubSubOperation.PROBE, lambda e: good_results.append(e))

        await publisher.publish(
            "probe",
            {
                "inst_status": 1,
                "linked": 1,
                "timestamp": 100.0,
                "scheduler": {"status": "ok", "queue_len": 0},
                "core_status": {"empty_thread": 5, "pause_read": 0, "thread_num": 5},
                "core_thread": {},
            },
        )
        await asyncio.sleep(0.15)

        # Good handler still received the event
        assert len(good_results) == 1
