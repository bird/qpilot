"""Tests for Pydantic models — serialization, deserialization, discriminated unions."""

from __future__ import annotations

import json


from qpilot.enums import ErrCode, TaskStatus
from qpilot.models.requests import (
    GetRBDataRequest,
    HeartbeatRequest,
    MsgTaskRequest,
    SetVipRequest,
    TaskStatusRequest,
)
from qpilot.models.responses import (
    GetChipConfigResponse,
    GetRBDataResponse,
    HeartbeatResponse,
    MsgTaskAck,
    MsgTaskResultResponse,
    TaskStatusResponse,
    response_adapter,
)
from qpilot.models.pubsub import (
    CalibrationDoneEvent,
    CalibrationStartEvent,
    ChipProtectEvent,
    ChipUpdateEvent,
    ProbeEvent,
    TaskStatusEvent,
    pubsub_adapter,
)


class TestRequestSerialization:
    def test_heartbeat_roundtrip(self):
        req = HeartbeatRequest(chip_id=72, timestamp=1234567890)
        data = req.model_dump(by_alias=True)
        assert data["MsgType"] == "MsgHeartbeat"
        assert data["ChipID"] == 72
        assert data["TimeStamp"] == 1234567890

    def test_msg_task_request(self):
        req = MsgTaskRequest(
            task_id="ABC123",
            convert_qprog=[[{"RPhi": [32, 270.0, 90.0, 0]}]],
            configure={"Shot": 1000, "TaskPriority": 0},
        )
        data = req.model_dump(by_alias=True)
        assert data["MsgType"] == "MsgTask"
        assert data["TaskId"] == "ABC123"
        assert data["Configure"]["Shot"] == 1000

    def test_task_status_request(self):
        req = TaskStatusRequest(task_id="XYZ")
        data = req.model_dump(by_alias=True)
        assert data["MsgType"] == "TaskStatus"
        assert data["TaskId"] == "XYZ"

    def test_get_rb_data_request(self):
        req = GetRBDataRequest(chip_id=99)
        data = req.model_dump(by_alias=True)
        assert data["MsgType"] == "GetRBData"
        assert data["ChipID"] == 99

    def test_set_vip_request(self):
        req = SetVipRequest(offset_time=120, exclusive_time=600)
        data = req.model_dump(by_alias=True)
        assert data["OffsetTime"] == 120
        assert data["ExclusiveTime"] == 600


class TestResponseDeserialization:
    def test_task_ack(self):
        raw = '{"MsgType":"MsgTaskAck","SN":1,"ErrCode":0,"ErrInfo":""}'
        resp = response_adapter.validate_json(raw)
        assert isinstance(resp, MsgTaskAck)
        assert resp.err_code == 0

    def test_heartbeat_ack(self):
        raw = json.dumps(
            {
                "MsgType": "MsgHeartbeatAck",
                "SN": 5,
                "backend": 72,
                "TimeStamp": 999,
                "Topic": "test",
            }
        )
        resp = response_adapter.validate_json(raw)
        assert isinstance(resp, HeartbeatResponse)
        assert resp.backend == 72

    def test_task_status_ack(self):
        raw = json.dumps({"MsgType": "TaskStatusAck", "SN": 10, "TaskId": "ABC", "TaskStatus": 5})
        resp = response_adapter.validate_json(raw)
        assert isinstance(resp, TaskStatusResponse)
        assert resp.task_status == TaskStatus.COMPLETED

    def test_task_result(self):
        raw = json.dumps(
            {
                "MsgType": "MsgTaskResult",
                "SN": 20,
                "TaskId": "ABC",
                "Key": [["0x0", "0x1"]],
                "ProbCount": [[500, 500]],
                "NoteTime": {
                    "CompileTime": 1,
                    "PendingTime": 94,
                    "MeasureTime": 2306,
                    "PostProcessTime": 105,
                },
                "ErrCode": 0,
                "ErrInfo": "",
            }
        )
        resp = response_adapter.validate_json(raw)
        assert isinstance(resp, MsgTaskResultResponse)
        assert resp.prob_count == [[500, 500]]
        assert resp.note_time.measure_time == 2306

    def test_rb_data_response(self):
        raw = json.dumps(
            {
                "MsgType": "GetRBDataAck",
                "SN": 30,
                "backend": 72,
                "SingleGateCircuitDepth": [50, 50],
                "DoubleGateCircuitDepth": [50],
                "SingleGateFidelity": {"qubit": ["45", "46"], "fidelity": [0.99, 0.98]},
                "DoubleGateFidelity": {"qubitPair": ["45-46"], "fidelity": [0.95]},
                "ErrCode": 0,
                "ErrInfo": "",
            }
        )
        resp = response_adapter.validate_json(raw)
        assert isinstance(resp, GetRBDataResponse)
        assert resp.single_gate_fidelity.fidelity == [0.99, 0.98]
        assert resp.double_gate_fidelity.qubit_pair == ["45-46"]

    def test_chip_config_response(self):
        raw = json.dumps(
            {
                "MsgType": "GetChipConfigAck",
                "SN": 40,
                "backend": 72,
                "PointLabelList": [1, 2],
                "ChipConfig": {"1": "{}", "2": "{}"},
                "ErrCode": 0,
                "ErrInfo": "",
            }
        )
        resp = response_adapter.validate_json(raw)
        assert isinstance(resp, GetChipConfigResponse)
        assert resp.point_label_list == [1, 2]

    def test_extra_fields_ignored(self):
        """Server may add new fields — we must not fail."""
        raw = json.dumps(
            {
                "MsgType": "MsgTaskAck",
                "SN": 1,
                "ErrCode": 0,
                "ErrInfo": "",
                "FutureField": "should be ignored",
            }
        )
        resp = response_adapter.validate_json(raw)
        assert isinstance(resp, MsgTaskAck)


class TestPubSubEvents:
    def test_task_status_event(self):
        raw = {
            "operation": "task_status",
            "MsgType": "TaskStatus",
            "SN": 0,
            "TaskId": "ABC123",
            "TaskStatus": 3,
        }
        event = pubsub_adapter.validate_python(raw)
        assert isinstance(event, TaskStatusEvent)
        assert event.task_id == "ABC123"

    def test_chip_update_event(self):
        raw = {
            "operation": "chip_update",
            "UpdateFlag": True,
            "LastUpdateTime": 1705307288685,
        }
        event = pubsub_adapter.validate_python(raw)
        assert isinstance(event, ChipUpdateEvent)
        assert event.update_flag is True

    def test_probe_event(self):
        raw = {
            "operation": "probe",
            "inst_status": 1,
            "linked": 1,
            "timestamp": 1695182448.796,
            "scheduler": {"status": "InitialState", "queue_len": 3},
            "core_status": {"empty_thread": 3, "pause_read": 0, "thread_num": 5},
            "core_thread": {
                "t0": {
                    "status": "waiting",
                    "thread_id": "t0",
                    "task_id": "650a6e6f",
                    "start_time": 1695182447.537,
                    "user": "admin3",
                    "use_bits": ["q56", "q18"],
                },
                "t1": {"status": "ready", "use_bits": []},
            },
        }
        event = pubsub_adapter.validate_python(raw)
        assert isinstance(event, ProbeEvent)
        assert event.core_status.empty_thread == 3
        assert event.core_thread["t0"].use_bits == ["q56", "q18"]

    def test_calibration_start_event(self):
        raw = {
            "operation": "calibration_start",
            "config_flag": False,
            "qubits": ["q0", "q1"],
            "couplers": ["c0-1"],
            "pairs": ["q0q1"],
            "discriminators": ["q0_01.bin"],
            "point_label": 2,
        }
        event = pubsub_adapter.validate_python(raw)
        assert isinstance(event, CalibrationStartEvent)
        assert event.qubits == ["q0", "q1"]

    def test_calibration_done_event(self):
        raw = {
            "operation": "calibration_done",
            "config_flag": True,
            "qubits": ["q0", "q1"],
            "couplers": [],
            "pairs": [],
            "discriminators": ["q0", "q1"],
            "point_label": 2,
        }
        event = pubsub_adapter.validate_python(raw)
        assert isinstance(event, CalibrationDoneEvent)
        assert event.config_flag is True

    def test_chip_protect_event(self):
        raw = {
            "operation": "chip_protect",
            "ProtectFlag": True,
            "DurativeTime": 10,
            "LastTime": 1705307288685,
        }
        event = pubsub_adapter.validate_python(raw)
        assert isinstance(event, ChipProtectEvent)
        assert event.protect_flag is True
        assert event.durative_time == 10


class TestEnums:
    def test_task_status_terminal(self):
        assert TaskStatus.COMPLETED.is_terminal
        assert TaskStatus.FAILED.is_terminal
        assert not TaskStatus.RUNNING.is_terminal
        assert not TaskStatus.QUEUED.is_terminal

    def test_err_code_values(self):
        assert ErrCode.NONE == 0
        assert ErrCode.QUEUE_FULL == 4
