"""Request models for PilotOS Router-Dealer protocol."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict, Field

from qpilot.enums import MsgType
from qpilot.models.base import QPilotMessage


class _RequestBase(QPilotMessage):
    """Base for request messages — forbid extra fields to catch programmer errors."""

    model_config = ConfigDict(
        populate_by_name=True,
        use_enum_values=True,
        extra="forbid",
    )


class MsgTaskRequest(_RequestBase):
    """Submit a task for execution."""

    msg_type: Literal[MsgType.MSG_TASK] = Field(default=MsgType.MSG_TASK, alias="MsgType")
    sn: int = Field(default=0, alias="SN")
    task_id: str = Field(alias="TaskId")
    convert_qprog: Any = Field(alias="ConvertQProg")
    configure: dict[str, Any] = Field(alias="Configure")


class TaskStatusRequest(_RequestBase):
    """Query status of a submitted task."""

    msg_type: Literal[MsgType.TASK_STATUS] = Field(default=MsgType.TASK_STATUS, alias="MsgType")
    sn: int = Field(default=0, alias="SN")
    task_id: str = Field(alias="TaskId")


class HeartbeatRequest(_RequestBase):
    """Send a heartbeat to verify connectivity."""

    msg_type: Literal[MsgType.MSG_HEARTBEAT] = Field(default=MsgType.MSG_HEARTBEAT, alias="MsgType")
    sn: int = Field(default=0, alias="SN")
    chip_id: int = Field(default=72, alias="ChipID")
    timestamp: int = Field(default=0, alias="TimeStamp")


class GetUpdateTimeRequest(_RequestBase):
    """Query last calibration update times."""

    msg_type: Literal[MsgType.GET_UPDATE_TIME] = Field(
        default=MsgType.GET_UPDATE_TIME, alias="MsgType"
    )
    sn: int = Field(default=0, alias="SN")


class GetRBDataRequest(_RequestBase):
    """Query randomized benchmarking fidelity data."""

    msg_type: Literal[MsgType.GET_RB_DATA] = Field(default=MsgType.GET_RB_DATA, alias="MsgType")
    sn: int = Field(default=0, alias="SN")
    chip_id: int = Field(default=72, alias="ChipID")


class GetChipConfigRequest(_RequestBase):
    """Query full chip configuration."""

    msg_type: Literal[MsgType.GET_CHIP_CONFIG] = Field(
        default=MsgType.GET_CHIP_CONFIG, alias="MsgType"
    )
    sn: int = Field(default=0, alias="SN")
    chip_id: int = Field(default=72, alias="ChipID")


class GetTaskResultRequest(_RequestBase):
    """Request result of a completed task."""

    msg_type: Literal[MsgType.GET_TASK_RESULT] = Field(
        default=MsgType.GET_TASK_RESULT, alias="MsgType"
    )
    sn: int = Field(default=0, alias="SN")
    task_id: str = Field(alias="TaskId")


class SetVipRequest(_RequestBase):
    """Request exclusive VIP time on the chip."""

    msg_type: Literal[MsgType.SET_VIP] = Field(default=MsgType.SET_VIP, alias="MsgType")
    sn: int = Field(default=0, alias="SN")
    offset_time: int = Field(alias="OffsetTime")
    exclusive_time: int = Field(alias="ExclusiveTime")


class ReleaseVipRequest(_RequestBase):
    """Release previously acquired VIP time."""

    msg_type: Literal[MsgType.RELEASE_VIP] = Field(default=MsgType.RELEASE_VIP, alias="MsgType")
    sn: int = Field(default=0, alias="SN")
