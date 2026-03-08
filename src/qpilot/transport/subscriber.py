"""ZeroMQ SUB client for real-time PilotOS event streaming."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Awaitable, Callable

import zmq
import zmq.asyncio

from qpilot.enums import PubSubOperation
from qpilot.models.pubsub import AnyPubSubEvent, pubsub_adapter

logger = logging.getLogger(__name__)

# Type for event callbacks: sync or async
EventCallback = Callable[[AnyPubSubEvent], Awaitable[None] | None]


class PubSubSubscriber:
    """Async ZMQ SUB client that dispatches parsed events to registered callbacks.

    Subscribes to PilotOS pub-sub topics and parses three-frame messages:
    [topic, operation, json_data].
    """

    def __init__(
        self,
        ctx: zmq.asyncio.Context,
        host: str = "localhost",
        port: int = 8000,
        topics: list[bytes] | None = None,
    ) -> None:
        self._ctx = ctx
        self._endpoint = f"tcp://{host}:{port}"
        self._topics = topics or [b"simulator_topic"]
        self._handlers: dict[str, list[EventCallback]] = {}
        self._socket: zmq.asyncio.Socket | None = None
        self._dispatch_task: asyncio.Task | None = None
        self._stopping = False

    def on(self, operation: PubSubOperation | str, callback: EventCallback) -> None:
        """Register a callback for a specific pub-sub operation type."""
        key = str(operation)
        self._handlers.setdefault(key, []).append(callback)

    async def start(self) -> None:
        """Create socket, subscribe to topics, start dispatch loop."""
        self._stopping = False
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        for topic in self._topics:
            self._socket.setsockopt(zmq.SUBSCRIBE, topic)
        self._socket.connect(self._endpoint)
        self._dispatch_task = asyncio.create_task(self._dispatch_loop())
        logger.info("PubSubSubscriber connected to %s", self._endpoint)

    async def stop(self) -> None:
        """Cancel dispatch loop and close socket."""
        self._stopping = True
        if self._dispatch_task:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass
        if self._socket:
            self._socket.close()
            self._socket = None
        logger.info("PubSubSubscriber stopped")

    async def _dispatch_loop(self) -> None:
        """Receive pub-sub frames, parse, and dispatch to handlers."""
        while not self._stopping:
            try:
                frames = await self._socket.recv_multipart()
            except zmq.ZMQError as exc:
                if exc.errno == zmq.ETERM or self._stopping:
                    break
                logger.warning("sub recv error: %s", exc)
                continue
            except asyncio.CancelledError:
                break

            if len(frames) != 3:
                logger.debug("unexpected pub-sub frame count: %d", len(frames))
                continue

            _topic, operation_bytes, json_bytes = frames

            try:
                operation = operation_bytes.decode()
                raw = json.loads(json_bytes)
                raw["operation"] = operation
                event = pubsub_adapter.validate_python(raw)
            except (UnicodeDecodeError, json.JSONDecodeError, Exception) as exc:
                logger.debug("failed to parse pub-sub event: %s", exc)
                continue

            handlers = self._handlers.get(operation, [])
            for cb in handlers:
                try:
                    if inspect.iscoroutinefunction(cb):
                        asyncio.create_task(cb(event))
                    else:
                        cb(event)
                except Exception:
                    logger.exception("handler error for %s", operation)

    async def __aenter__(self) -> PubSubSubscriber:
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()
