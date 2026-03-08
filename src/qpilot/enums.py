"""Enumerations and constants for the PilotOS ZMQ protocol."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum


class TaskStatus(IntEnum):
    """Task execution status codes from PilotOS."""

    UNKNOWN = 0
    QUEUED = 1
    RUNNING = 2
    NOT_FOUND = 3
    FAILED = 4
    COMPLETED = 5
    RETRY = 6
    COMPILING = 7
    COMPILED = 8

    @property
    def is_terminal(self) -> bool:
        return self in (self.COMPLETED, self.FAILED, self.NOT_FOUND)


class ErrCode(IntEnum):
    """Error codes from PilotOS responses."""

    NONE = 0
    UNKNOWN = 1
    PARAM_ERROR = 2
    JSON_ERROR = 3
    QUEUE_FULL = 4


class MsgType(StrEnum):
    """Message type identifiers in the Router-Dealer protocol."""

    # Task lifecycle
    MSG_TASK = "MsgTask"
    MSG_TASK_ACK = "MsgTaskAck"
    TASK_STATUS = "TaskStatus"
    TASK_STATUS_ACK = "TaskStatusAck"
    MSG_TASK_RESULT = "MsgTaskResult"
    MSG_TASK_RESULT_ACK = "MsgTaskResultAck"
    GET_TASK_RESULT = "GetTaskResult"

    # Heartbeat
    MSG_HEARTBEAT = "MsgHeartbeat"
    MSG_HEARTBEAT_ACK = "MsgHeartbeatAck"

    # Chip info
    GET_UPDATE_TIME = "GetUpdateTime"
    GET_UPDATE_TIME_ACK = "GetUpdateTimeAck"
    GET_RB_DATA = "GetRBData"
    GET_RB_DATA_ACK = "GetRBDataAck"
    GET_CHIP_CONFIG = "GetChipConfig"
    GET_CHIP_CONFIG_ACK = "GetChipConfigAck"

    # VIP / exclusive time
    SET_VIP = "SetVip"
    SET_VIP_ACK = "SetVipAck"
    RELEASE_VIP = "ReleaseVip"
    RELEASE_VIP_ACK = "ReleaseVipAck"


class PubSubOperation(StrEnum):
    """Operation types in pub-sub event messages."""

    TASK_STATUS = "task_status"
    CHIP_UPDATE = "chip_update"
    PROBE = "probe"
    CALIBRATION_START = "calibration_start"
    CALIBRATION_DONE = "calibration_done"
    CHIP_PROTECT = "chip_protect"


class HardwareType(StrEnum):
    """Supported quantum hardware backend types."""

    SUPERCONDUCTING = "superconducting"
    TRAPPED_ION = "trapped_ion"
    NEUTRAL_ATOM = "neutral_atom"
    PHOTONIC = "photonic"


@dataclass(frozen=True)
class PortPair:
    """ZMQ port pair for a hardware backend."""

    dealer: int
    pub: int


HARDWARE_PORTS: dict[HardwareType, PortPair] = {
    HardwareType.SUPERCONDUCTING: PortPair(dealer=7000, pub=8000),
    HardwareType.TRAPPED_ION: PortPair(dealer=7001, pub=8001),
    HardwareType.NEUTRAL_ATOM: PortPair(dealer=7002, pub=8002),
    HardwareType.PHOTONIC: PortPair(dealer=7003, pub=8003),
}
