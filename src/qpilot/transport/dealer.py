"""ZeroMQ DEALER client for request-response communication with PilotOS."""

from __future__ import annotations

import asyncio
import itertools
import logging
import time

import zmq
import zmq.asyncio

from qpilot.exceptions import (
    ReconnectError,
    TimeoutError,
    TransportError,
)
from qpilot.models.base import QPilotMessage
from qpilot.models.requests import HeartbeatRequest
from qpilot.models.responses import AnyResponse, response_adapter

logger = logging.getLogger(__name__)


class DealerClient:
    """Async ZMQ DEALER client with SN-based request-response correlation.

    Uses a single receive loop to dispatch responses to waiting futures
    keyed by sequence number.
    """

    def __init__(
        self,
        ctx: zmq.asyncio.Context,
        host: str = "localhost",
        port: int = 7000,
        *,
        request_timeout: float = 30.0,
        heartbeat_interval: float = 10.0,
    ) -> None:
        self._ctx = ctx
        self._endpoint = f"tcp://{host}:{port}"
        self._request_timeout = request_timeout
        self._heartbeat_interval = heartbeat_interval

        self._sn = itertools.count(1)
        self._pending: dict[int, asyncio.Future[AnyResponse]] = {}
        self._socket: zmq.asyncio.Socket | None = None
        self._recv_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._stopping = False
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self) -> None:
        """Create socket, connect, and start receive + heartbeat loops."""
        self._loop = asyncio.get_running_loop()
        self._stopping = False
        self._create_socket()
        self._recv_task = asyncio.create_task(self._recv_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("DealerClient connected to %s", self._endpoint)

    async def stop(self) -> None:
        """Cancel loops, cancel pending futures, close socket."""
        self._stopping = True
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        self._cancel_all_pending(TransportError("client stopped"))
        self._close_socket()
        logger.info("DealerClient stopped")

    async def send_request(self, request: QPilotMessage) -> AnyResponse:
        """Send a request and await the correlated response by SN."""
        if self._socket is None:
            raise TransportError("not connected")

        sn = next(self._sn)
        payload = request.model_dump(by_alias=True)
        payload["SN"] = sn
        data = _json_bytes(payload)

        fut: asyncio.Future[AnyResponse] = self._loop.create_future()
        self._pending[sn] = fut
        try:
            await self._socket.send_multipart([b"", data])
        except Exception as exc:
            self._pending.pop(sn, None)
            if not fut.done():
                fut.cancel()
            raise TransportError(f"send failed: {exc}") from exc
        try:
            return await asyncio.wait_for(asyncio.shield(fut), self._request_timeout)
        except asyncio.TimeoutError:
            self._pending.pop(sn, None)
            if not fut.done():
                fut.cancel()
            raise TimeoutError(f"request SN={sn} timed out after {self._request_timeout}s")

    async def _recv_loop(self) -> None:
        """Receive responses and resolve pending futures by SN."""
        while not self._stopping:
            try:
                frames = await self._socket.recv_multipart()
            except zmq.ZMQError as exc:
                if exc.errno == zmq.ETERM or self._stopping:
                    break
                logger.warning("recv error: %s", exc)
                continue
            except asyncio.CancelledError:
                break

            # DEALER receives [empty_delimiter, json_bytes] or [json_bytes]
            if len(frames) == 2 and frames[0] == b"":
                raw = frames[1]
            elif len(frames) == 1:
                raw = frames[0]
            else:
                logger.debug("unexpected frame count: %d", len(frames))
                continue

            try:
                response = response_adapter.validate_json(raw)
            except Exception:
                logger.debug("failed to parse response: %s", raw[:200])
                continue

            fut = self._pending.pop(response.sn, None)
            if fut is not None and not fut.done():
                fut.set_result(response)

    async def _heartbeat_loop(self) -> None:
        """Periodically send heartbeats to detect connection loss."""
        while not self._stopping:
            await asyncio.sleep(self._heartbeat_interval)
            if self._stopping:
                break
            try:
                req = HeartbeatRequest(
                    chip_id=72,
                    timestamp=int(time.time() * 1000),
                )
                await self.send_request(req)
            except TimeoutError:
                logger.warning("heartbeat timed out, attempting reconnect")
                await self._reconnect()
            except (TransportError, asyncio.CancelledError):
                break

    async def _reconnect(self) -> None:
        """Close current socket, cancel pending, create new socket."""
        # Cancel receive loop
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass

        self._cancel_all_pending(ReconnectError("connection lost"))
        self._close_socket()
        self._create_socket()
        self._recv_task = asyncio.create_task(self._recv_loop())
        logger.info("DealerClient reconnected to %s", self._endpoint)

    def _create_socket(self) -> None:
        self._socket = self._ctx.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self._endpoint)

    def _close_socket(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def _cancel_all_pending(self, error: Exception) -> None:
        for sn, fut in self._pending.items():
            if not fut.done():
                fut.set_exception(error)
        self._pending.clear()

    async def __aenter__(self) -> DealerClient:
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()


def _json_bytes(obj: dict) -> bytes:
    """Serialize dict to JSON bytes using pydantic's fast JSON."""
    import json

    return json.dumps(obj, separators=(",", ":")).encode()
