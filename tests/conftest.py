"""Shared test fixtures for qpilot tests."""

from __future__ import annotations

import asyncio
import json

import pytest
import zmq
import zmq.asyncio


@pytest.fixture
def zmq_ctx():
    """Provide a ZMQ async context, destroyed after test."""
    ctx = zmq.asyncio.Context()
    yield ctx
    ctx.destroy(linger=0)


@pytest.fixture
def router_endpoint():
    """Provide a unique inproc endpoint for router-dealer tests."""
    return "inproc://test-router"


@pytest.fixture
def pub_endpoint():
    """Provide a unique inproc endpoint for pub-sub tests."""
    return "inproc://test-pub"


class FakeRouter:
    """A ZMQ ROUTER that echoes or customizes responses for testing."""

    def __init__(self, ctx: zmq.asyncio.Context, endpoint: str) -> None:
        self._ctx = ctx
        self._endpoint = endpoint
        self._socket: zmq.asyncio.Socket | None = None
        self._task: asyncio.Task | None = None
        self._stopping = False
        self._response_overrides: dict[str, dict] = {}

    def set_response(self, msg_type: str, response: dict) -> None:
        """Set a custom response for a given MsgType."""
        self._response_overrides[msg_type] = response

    async def start(self) -> None:
        self._socket = self._ctx.socket(zmq.ROUTER)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(self._endpoint)
        self._stopping = False
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stopping = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._socket:
            self._socket.close()
            self._socket = None

    async def _loop(self) -> None:
        while not self._stopping:
            try:
                frames = await self._socket.recv_multipart()
            except (zmq.ZMQError, asyncio.CancelledError):
                break
            # ROUTER receives: [identity, empty_delim, json_data]
            if len(frames) < 3:
                continue
            identity = frames[0]
            raw = json.loads(frames[-1])
            msg_type = raw.get("MsgType", "")
            sn = raw.get("SN", 0)

            if msg_type in self._response_overrides:
                resp = {**self._response_overrides[msg_type], "SN": sn}
            else:
                resp = self._default_response(msg_type, sn, raw)

            await self._socket.send_multipart([identity, b"", json.dumps(resp).encode()])

    def _default_response(self, msg_type: str, sn: int, raw: dict) -> dict:
        """Generate default echo responses for common message types."""
        ack_map = {
            "MsgTask": "MsgTaskAck",
            "TaskStatus": "TaskStatusAck",
            "MsgHeartbeat": "MsgHeartbeatAck",
            "GetUpdateTime": "GetUpdateTimeAck",
            "GetRBData": "GetRBDataAck",
            "GetChipConfig": "GetChipConfigAck",
            "GetTaskResult": "MsgTaskResult",
            "SetVip": "SetVipAck",
            "ReleaseVip": "ReleaseVipAck",
        }
        ack_type = ack_map.get(msg_type, msg_type + "Ack")
        resp: dict = {"MsgType": ack_type, "SN": sn, "ErrCode": 0, "ErrInfo": ""}

        if msg_type == "MsgHeartbeat":
            resp["backend"] = 72
            resp["TimeStamp"] = raw.get("TimeStamp", 0)
            resp["Topic"] = "test-topic"
        elif msg_type == "TaskStatus":
            resp["TaskId"] = raw.get("TaskId", "")
            resp["TaskStatus"] = 5  # COMPLETED
        elif msg_type == "GetRBData":
            resp["backend"] = 72
            resp["SingleGateCircuitDepth"] = [50]
            resp["DoubleGateCircuitDepth"] = [50]
            resp["SingleGateFidelity"] = {"qubit": ["45"], "fidelity": [0.99]}
            resp["DoubleGateFidelity"] = {"qubitPair": ["45-46"], "fidelity": [0.95]}
        elif msg_type == "GetChipConfig":
            resp["backend"] = 72
            resp["PointLabelList"] = [1, 2]
            resp["ChipConfig"] = {"1": "{}"}
        elif msg_type == "GetTaskResult":
            resp["MsgType"] = "MsgTaskResult"
            resp["TaskId"] = raw.get("TaskId", "")
            resp["Key"] = [["0x0", "0x1"]]
            resp["ProbCount"] = [[500, 500]]
        elif msg_type == "MsgTask":
            pass  # MsgTaskAck with ErrCode=0 is sufficient

        return resp


class FakePublisher:
    """A ZMQ PUB socket that sends test events."""

    def __init__(self, ctx: zmq.asyncio.Context, endpoint: str) -> None:
        self._ctx = ctx
        self._endpoint = endpoint
        self._socket: zmq.asyncio.Socket | None = None

    async def start(self) -> None:
        self._socket = self._ctx.socket(zmq.PUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(self._endpoint)

    async def stop(self) -> None:
        if self._socket:
            self._socket.close()
            self._socket = None

    async def publish(self, operation: str, data: dict, topic: bytes = b"simulator_topic") -> None:
        """Publish a three-frame event message."""
        await self._socket.send_multipart(
            [
                topic,
                operation.encode(),
                json.dumps(data).encode(),
            ]
        )


@pytest.fixture
async def fake_router(zmq_ctx, router_endpoint):
    """Start a FakeRouter and yield it."""
    router = FakeRouter(zmq_ctx, router_endpoint)
    await router.start()
    yield router
    await router.stop()


@pytest.fixture
async def fake_publisher(zmq_ctx, pub_endpoint):
    """Start a FakePublisher and yield it."""
    pub = FakePublisher(zmq_ctx, pub_endpoint)
    await pub.start()
    yield pub
    await pub.stop()
