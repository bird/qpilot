"""QPilot exception hierarchy."""


class QPilotError(RuntimeError):
    """Base exception for all qpilot errors."""


class TransportError(QPilotError):
    """ZMQ transport-level error."""


class ConnectionError(TransportError):  # noqa: A001
    """Failed to connect to PilotOS backend."""


class TimeoutError(TransportError):  # noqa: A001
    """Request timed out waiting for response."""


class ReconnectError(TransportError):
    """Connection was lost and pending requests were cancelled."""


class ProtocolError(QPilotError):
    """Protocol-level error in message framing or content."""


class FrameError(ProtocolError):
    """Unexpected ZMQ frame count or empty frame where data expected."""


class DeserializationError(ProtocolError):
    """Failed to parse response JSON or validate against model."""


class RemoteError(QPilotError):
    """Server returned a non-zero error code."""

    def __init__(self, code: int, info: str = ""):
        self.code = code
        self.info = info
        super().__init__(f"ErrCode={code}: {info}" if info else f"ErrCode={code}")


class MonitorError(QPilotError):
    """Error in the monitoring subsystem."""
