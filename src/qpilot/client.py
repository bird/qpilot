"""Top-level QPilot client — unified interface to PilotOS hardware."""

from __future__ import annotations

import logging
import time
from typing import Any

import zmq.asyncio

from qpilot.enums import HARDWARE_PORTS, HardwareType
from qpilot.exceptions import RemoteError
from qpilot.models.requests import (
    GetChipConfigRequest,
    GetRBDataRequest,
    GetTaskResultRequest,
    GetUpdateTimeRequest,
    HeartbeatRequest,
    MsgTaskRequest,
    ReleaseVipRequest,
    SetVipRequest,
    TaskStatusRequest,
)
from qpilot.models.responses import (
    AnyResponse,
    GetChipConfigResponse,
    GetRBDataResponse,
    GetUpdateTimeResponse,
    HeartbeatResponse,
    MsgTaskAck,
    MsgTaskResultResponse,
    ReleaseVipResponse,
    SetVipResponse,
    TaskStatusResponse,
)
from qpilot.monitor.chip_monitor import ChipMonitor
from qpilot.transport.dealer import DealerClient
from qpilot.transport.subscriber import PubSubSubscriber

logger = logging.getLogger(__name__)


class QPilotClient:
    """Unified async client for PilotOS quantum hardware.

    Manages ZMQ transport (DEALER for requests, SUB for events) and
    provides high-level methods for chip interaction.

    Usage::

        async with QPilotClient() as client:
            config = await client.get_chip_config()
            rb = await client.get_rb_data()
            print(client.monitor.state)
    """

    def __init__(
        self,
        host: str = "localhost",
        hardware: HardwareType = HardwareType.SUPERCONDUCTING,
        *,
        request_timeout: float = 30.0,
        heartbeat_interval: float = 10.0,
        topics: list[bytes] | None = None,
        chip_id: int = 72,
    ) -> None:
        ports = HARDWARE_PORTS[hardware]
        self._ctx = zmq.asyncio.Context()
        self._dealer = DealerClient(
            self._ctx,
            host,
            ports.dealer,
            request_timeout=request_timeout,
            heartbeat_interval=heartbeat_interval,
        )
        self._subscriber = PubSubSubscriber(
            self._ctx,
            host,
            ports.pub,
            topics=topics,
        )
        self._monitor = ChipMonitor(self._subscriber)
        self._chip_id = chip_id
        self._host = host
        self._hardware = hardware

    @property
    def monitor(self) -> ChipMonitor:
        """Access the real-time chip monitor."""
        return self._monitor

    @property
    def dealer(self) -> DealerClient:
        """Access the raw DEALER transport (advanced use)."""
        return self._dealer

    # --- High-level API ---

    async def heartbeat(self) -> HeartbeatResponse:
        """Send heartbeat and return response."""
        req = HeartbeatRequest(
            chip_id=self._chip_id,
            timestamp=int(time.time() * 1000),
        )
        resp = await self._dealer.send_request(req)
        assert isinstance(resp, HeartbeatResponse)
        return resp

    async def get_chip_config(self) -> GetChipConfigResponse:
        """Fetch full chip configuration."""
        req = GetChipConfigRequest(chip_id=self._chip_id)
        resp = await self._dealer.send_request(req)
        assert isinstance(resp, GetChipConfigResponse)
        _check_err(resp)
        return resp

    async def get_rb_data(self) -> GetRBDataResponse:
        """Fetch randomized benchmarking fidelity data."""
        req = GetRBDataRequest(chip_id=self._chip_id)
        resp = await self._dealer.send_request(req)
        assert isinstance(resp, GetRBDataResponse)
        _check_err(resp)
        return resp

    async def get_update_time(self) -> GetUpdateTimeResponse:
        """Fetch last calibration update times."""
        req = GetUpdateTimeRequest()
        resp = await self._dealer.send_request(req)
        assert isinstance(resp, GetUpdateTimeResponse)
        _check_err(resp)
        return resp

    async def submit_task(
        self,
        task_id: str,
        circuit: Any,
        shots: int = 1000,
        *,
        priority: int = 0,
        is_experiment: bool = False,
        point_label: int = 128,
    ) -> MsgTaskAck:
        """Submit a task for execution on the QPU."""
        req = MsgTaskRequest(
            task_id=task_id,
            convert_qprog=circuit,
            configure={
                "Shot": shots,
                "TaskPriority": priority,
                "IsExperiment": is_experiment,
                "ClockCycle": None,
                "PointLabel": point_label,
            },
        )
        resp = await self._dealer.send_request(req)
        assert isinstance(resp, MsgTaskAck)
        _check_err(resp)
        return resp

    async def get_task_status(self, task_id: str) -> TaskStatusResponse:
        """Query execution status of a submitted task."""
        req = TaskStatusRequest(task_id=task_id)
        resp = await self._dealer.send_request(req)
        assert isinstance(resp, TaskStatusResponse)
        return resp

    async def get_task_result(self, task_id: str) -> MsgTaskResultResponse:
        """Fetch results of a completed task."""
        req = GetTaskResultRequest(task_id=task_id)
        resp = await self._dealer.send_request(req)
        assert isinstance(resp, MsgTaskResultResponse)
        _check_err(resp)
        return resp

    async def set_vip(self, offset_time: int, exclusive_time: int) -> SetVipResponse:
        """Request exclusive VIP time on the chip."""
        req = SetVipRequest(offset_time=offset_time, exclusive_time=exclusive_time)
        resp = await self._dealer.send_request(req)
        assert isinstance(resp, SetVipResponse)
        _check_err(resp)
        return resp

    async def release_vip(self) -> ReleaseVipResponse:
        """Release previously acquired VIP time."""
        req = ReleaseVipRequest()
        resp = await self._dealer.send_request(req)
        assert isinstance(resp, ReleaseVipResponse)
        _check_err(resp)
        return resp

    # --- Lifecycle ---

    async def start(self) -> None:
        """Start both transports."""
        await self._dealer.start()
        await self._subscriber.start()
        logger.info("QPilotClient started (host=%s, hardware=%s)", self._host, self._hardware)

    async def stop(self) -> None:
        """Stop subscriber first (no more events), then dealer, then destroy context."""
        await self._subscriber.stop()
        await self._dealer.stop()
        self._ctx.destroy(linger=0)
        logger.info("QPilotClient stopped")

    async def __aenter__(self) -> QPilotClient:
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()


def _check_err(resp: AnyResponse) -> None:
    """Raise RemoteError if response contains a non-zero error code."""
    err_code = getattr(resp, "err_code", 0)
    if err_code != 0:
        err_info = getattr(resp, "err_info", "")
        raise RemoteError(err_code, err_info)
