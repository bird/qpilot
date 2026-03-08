"""Response models for PilotOS Router-Dealer protocol."""

from __future__ import annotations

from typing import Any, Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from qpilot.enums import MsgType
from qpilot.models.base import QPilotMessage


class _ResponseBase(QPilotMessage):
    """Base for response messages — ignore unknown fields from server."""

    model_config = ConfigDict(
        populate_by_name=True,
        use_enum_values=True,
        extra="ignore",
    )


class MsgTaskAck(_ResponseBase):
    """Acknowledgement of task submission."""

    msg_type: Literal[MsgType.MSG_TASK_ACK] = Field(alias="MsgType")
    err_code: int = Field(default=0, alias="ErrCode")
    err_info: str = Field(default="", alias="ErrInfo")


class TaskStatusResponse(_ResponseBase):
    """Response to task status query."""

    msg_type: Literal[MsgType.TASK_STATUS_ACK] = Field(alias="MsgType")
    task_id: str = Field(alias="TaskId")
    task_status: int = Field(alias="TaskStatus")


class NoteTime(BaseModel):
    """Timing information from task execution."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    compile_time: int = Field(default=0, alias="CompileTime")
    pending_time: int = Field(default=0, alias="PendingTime")
    measure_time: int = Field(default=0, alias="MeasureTime")
    post_process_time: int = Field(default=0, alias="PostProcessTime")


class MsgTaskResultResponse(_ResponseBase):
    """Task execution results with measurement counts."""

    msg_type: Literal[MsgType.MSG_TASK_RESULT] = Field(alias="MsgType")
    task_id: str = Field(alias="TaskId")
    key: list[list[str]] = Field(default_factory=list, alias="Key")
    prob_count: list[list[int]] = Field(default_factory=list, alias="ProbCount")
    note_time: NoteTime | None = Field(default=None, alias="NoteTime")
    err_code: int = Field(default=0, alias="ErrCode")
    err_info: str = Field(default="", alias="ErrInfo")


class HeartbeatResponse(_ResponseBase):
    """Heartbeat acknowledgement with backend info."""

    msg_type: Literal[MsgType.MSG_HEARTBEAT_ACK] = Field(alias="MsgType")
    backend: int = Field(default=0, alias="backend")
    timestamp: int = Field(default=0, alias="TimeStamp")
    topic: str = Field(default="", alias="Topic")


class LastUpdateTimeData(BaseModel):
    """Calibration timestamp data per qubit."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    qubit: list[int] = Field(default_factory=list)
    timestamp: list[int] = Field(default_factory=list, alias="timeStamp")


class GetUpdateTimeResponse(_ResponseBase):
    """Calibration update times response."""

    msg_type: Literal[MsgType.GET_UPDATE_TIME_ACK] = Field(alias="MsgType")
    backend: int = Field(default=0, alias="backend")
    last_update_time: LastUpdateTimeData | None = Field(default=None, alias="LastUpdateTime")
    err_code: int = Field(default=0, alias="ErrCode")
    err_info: str = Field(default="", alias="ErrInfo")


class SingleGateFidelity(BaseModel):
    """Single-qubit gate fidelity data."""

    model_config = ConfigDict(extra="ignore")

    qubit: list[str] = Field(default_factory=list)
    fidelity: list[float] = Field(default_factory=list)


class DoubleGateFidelity(BaseModel):
    """Two-qubit gate fidelity data."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    qubit_pair: list[str] = Field(default_factory=list, alias="qubitPair")
    fidelity: list[float] = Field(default_factory=list)


class GetRBDataResponse(_ResponseBase):
    """Randomized benchmarking data response."""

    msg_type: Literal[MsgType.GET_RB_DATA_ACK] = Field(alias="MsgType")
    backend: int = Field(default=0, alias="backend")
    single_gate_circuit_depth: list[int] = Field(
        default_factory=list, alias="SingleGateCircuitDepth"
    )
    double_gate_circuit_depth: list[int] = Field(
        default_factory=list, alias="DoubleGateCircuitDepth"
    )
    single_gate_fidelity: SingleGateFidelity | None = Field(
        default=None, alias="SingleGateFidelity"
    )
    double_gate_fidelity: DoubleGateFidelity | None = Field(
        default=None, alias="DoubleGateFidelity"
    )
    err_code: int = Field(default=0, alias="ErrCode")
    err_info: str = Field(default="", alias="ErrInfo")


class GetChipConfigResponse(_ResponseBase):
    """Chip configuration response."""

    msg_type: Literal[MsgType.GET_CHIP_CONFIG_ACK] = Field(alias="MsgType")
    backend: int = Field(default=0, alias="backend")
    point_label_list: list[int] = Field(default_factory=list, alias="PointLabelList")
    chip_config: dict[str, Any] = Field(default_factory=dict, alias="ChipConfig")
    err_code: int = Field(default=0, alias="ErrCode")
    err_info: str = Field(default="", alias="ErrInfo")


class SetVipResponse(_ResponseBase):
    """VIP time acquisition response."""

    msg_type: Literal[MsgType.SET_VIP_ACK] = Field(alias="MsgType")
    err_code: int = Field(default=0, alias="ErrCode")
    err_info: str = Field(default="", alias="ErrInfo")


class ReleaseVipResponse(_ResponseBase):
    """VIP time release response."""

    msg_type: Literal[MsgType.RELEASE_VIP_ACK] = Field(alias="MsgType")
    err_code: int = Field(default=0, alias="ErrCode")
    err_info: str = Field(default="", alias="ErrInfo")


# Discriminated union over all response types
AnyResponse = Annotated[
    Union[
        MsgTaskAck,
        TaskStatusResponse,
        MsgTaskResultResponse,
        HeartbeatResponse,
        GetUpdateTimeResponse,
        GetRBDataResponse,
        GetChipConfigResponse,
        SetVipResponse,
        ReleaseVipResponse,
    ],
    Field(discriminator="msg_type"),
]

response_adapter: TypeAdapter[AnyResponse] = TypeAdapter(AnyResponse)
