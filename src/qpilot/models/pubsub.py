"""Pub-Sub event models for PilotOS real-time notifications."""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from qpilot.enums import PubSubOperation
from qpilot.models.base import PubSubEvent


class TaskStatusEvent(PubSubEvent):
    """Real-time task status update."""

    operation: Literal[PubSubOperation.TASK_STATUS] = PubSubOperation.TASK_STATUS
    msg_type: str = Field(default="", alias="MsgType")
    sn: int = Field(default=0, alias="SN")
    task_id: str = Field(default="", alias="TaskId")
    task_status: int = Field(default=0, alias="TaskStatus")


class ChipUpdateEvent(PubSubEvent):
    """Chip configuration has been updated."""

    operation: Literal[PubSubOperation.CHIP_UPDATE] = PubSubOperation.CHIP_UPDATE
    update_flag: bool = Field(default=False, alias="UpdateFlag")
    last_update_time: int = Field(default=0, alias="LastUpdateTime")


class ThreadInfo(BaseModel):
    """Status of a single hardware execution thread."""

    model_config = ConfigDict(extra="ignore")

    status: str = ""
    thread_id: str = ""
    task_id: str = ""
    start_time: float = 0.0
    user: str = ""
    use_bits: list[str] = Field(default_factory=list)


class SchedulerInfo(BaseModel):
    """Scheduler status."""

    model_config = ConfigDict(extra="ignore")

    status: str = ""
    queue_len: int = 0


class CoreStatus(BaseModel):
    """Core execution status."""

    model_config = ConfigDict(extra="ignore")

    empty_thread: int = 0
    pause_read: int = 0
    thread_num: int = 0


class ProbeEvent(PubSubEvent):
    """Real-time qubit resource allocation and thread status."""

    operation: Literal[PubSubOperation.PROBE] = PubSubOperation.PROBE
    inst_status: int = 0
    linked: int = 0
    timestamp: float = 0.0
    scheduler: SchedulerInfo = Field(default_factory=SchedulerInfo)
    core_status: CoreStatus = Field(default_factory=CoreStatus)
    core_thread: dict[str, ThreadInfo] = Field(default_factory=dict)


class CalibrationStartEvent(PubSubEvent):
    """Calibration cycle has started."""

    operation: Literal[PubSubOperation.CALIBRATION_START] = PubSubOperation.CALIBRATION_START
    config_flag: bool = Field(default=False)
    qubits: list[str] = Field(default_factory=list)
    couplers: list[str] = Field(default_factory=list)
    pairs: list[str] = Field(default_factory=list)
    discriminators: list[str] = Field(default_factory=list)
    point_label: int = Field(default=0)


class CalibrationDoneEvent(PubSubEvent):
    """Calibration cycle has completed."""

    operation: Literal[PubSubOperation.CALIBRATION_DONE] = PubSubOperation.CALIBRATION_DONE
    config_flag: bool = Field(default=False)
    qubits: list[str] = Field(default_factory=list)
    couplers: list[str] = Field(default_factory=list)
    pairs: list[str] = Field(default_factory=list)
    discriminators: list[str] = Field(default_factory=list)
    point_label: int = Field(default=0)


class ChipProtectEvent(PubSubEvent):
    """Chip entering or leaving maintenance/protection mode."""

    operation: Literal[PubSubOperation.CHIP_PROTECT] = PubSubOperation.CHIP_PROTECT
    protect_flag: bool = Field(default=False, alias="ProtectFlag")
    durative_time: int = Field(default=0, alias="DurativeTime")
    last_time: int = Field(default=0, alias="LastTime")


# Discriminated union over all pub-sub event types
AnyPubSubEvent = Annotated[
    Union[
        TaskStatusEvent,
        ChipUpdateEvent,
        ProbeEvent,
        CalibrationStartEvent,
        CalibrationDoneEvent,
        ChipProtectEvent,
    ],
    Field(discriminator="operation"),
]

pubsub_adapter: TypeAdapter[AnyPubSubEvent] = TypeAdapter(AnyPubSubEvent)
