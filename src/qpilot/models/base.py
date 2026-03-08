"""Base model classes for PilotOS protocol messages."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from qpilot.enums import MsgType, PubSubOperation


class QPilotMessage(BaseModel):
    """Base class for all Router-Dealer protocol messages."""

    model_config = ConfigDict(
        populate_by_name=True,
        use_enum_values=True,
    )

    msg_type: MsgType = Field(alias="MsgType")
    sn: int = Field(alias="SN")


class PubSubEvent(BaseModel):
    """Base class for all Pub-Sub event payloads."""

    model_config = ConfigDict(
        extra="ignore",
        use_enum_values=True,
    )

    operation: PubSubOperation
