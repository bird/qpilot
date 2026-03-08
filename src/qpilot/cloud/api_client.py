"""Async HTTP client for Origin Quantum Cloud API.

Authentication uses the ``oqcs_auth`` header scheme — the API key obtained
from the Origin Quantum Cloud console is sent as
``Authorization: oqcs_auth=<key>`` on every request.  No separate login
step is required.

Data-fetching endpoints (chip config, queue info) consume **zero** QPU time.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_BASE_URL = "https://qcloud.originqc.com.cn"


class OriginCloudClient:
    """Async client for the Origin Quantum Cloud HTTP API.

    Usage::

        async with OriginCloudClient(api_key="...") as cloud:
            config = await cloud.get_chip_config("72")
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = _BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = (
            api_key
            or os.environ.get("ORIGINQ_API_KEY", "")
            or os.environ.get("originq_cloud_api", "")
        )
        if not self._api_key:
            raise ValueError("API key required (pass api_key= or set ORIGINQ_API_KEY)")
        self._base_url = base_url.rstrip("/")
        self._headers = {
            "Authorization": f"oqcs_auth={self._api_key}",
            "Content-Type": "application/json;charset=UTF-8",
            "Connection": "keep-alive",
        }
        self._client = httpx.AsyncClient(timeout=timeout, headers=self._headers)

    # ------------------------------------------------------------------
    # Data endpoints (zero QPU cost)
    # ------------------------------------------------------------------

    async def get_chip_config(
        self,
        chip_id: str = "72",
    ) -> dict[str, Any]:
        """Fetch full chip configuration (per-qubit fidelity, T1/T2, topology).

        Returns the ``obj`` field of the API response which contains:
        - ``adjJSON``: per-qubit parameters (readout fidelity, T1, T2, gate fidelity, ...)
        - ``gateJSON``: CZ gate fidelities for connected qubit pairs
        - chip-level aggregates (avgT1, avgT2, clops, status, ...)
        """
        return await self._get(
            "/api/taskApi/getFullConfig.json",
            params={"chipId": chip_id},
        )

    async def get_task_detail(
        self,
        task_id: str,
    ) -> dict[str, Any]:
        """Query status and results for a submitted task."""
        return await self._get(
            "/api/taskApi/getTaskDetail.json",
            params={"taskId": task_id},
        )

    async def submit_task(
        self,
        *,
        qprog: str,
        chip_id: str = "72",
        shots: int = 1000,
        task_name: str = "qpilot",
        qubit_num: int = 72,
        cbit_num: int = 72,
        is_mapping: bool = True,
        is_optimization: bool = True,
        specified_block: list[int] | None = None,
    ) -> dict[str, Any]:
        """Submit a quantum circuit for execution on real hardware.

        Parameters
        ----------
        qprog : str
            OriginIR text-format circuit (QINIT/CREG/H/CNOT/MEASURE ...).
        chip_id : str
            Target chip, default ``"72"`` for Wukong.
        shots : int
            Number of measurement repetitions (1000-10000).
        specified_block : list[int], optional
            Pin to specific physical qubits.  When provided, set
            *is_mapping* to ``False`` to prevent the compiler from
            remapping.
        """
        body: dict[str, Any] = {
            "apiKey": self._api_key,
            "code": qprog,
            "codeLen": len(qprog),
            "qubitNum": qubit_num,
            "classicalbitNum": cbit_num,
            "shot": shots,
            "chipId": int(chip_id),
            "measureType": 1,
            "QMachineType": 5,
            "taskFrom": 4,
            "taskName": task_name,
            "isAmend": True,
            "mappingFlag": is_mapping,
            "circuitOptimization": is_optimization,
        }
        if specified_block is not None:
            body["specified_block"] = specified_block
            if is_mapping:
                logger.warning(
                    "specified_block provided but is_mapping=True; "
                    "the compiler may remap qubits away from the pinned block"
                )
        return await self._post("/api/taskApi/submitTask.json", body)

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    async def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send an authenticated GET request."""
        url = f"{self._base_url}{path}"
        resp = await self._client.get(url, params=params)
        resp.raise_for_status()
        return self._unwrap(resp.json())

    async def _post(
        self,
        path: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        """Send an authenticated POST request."""
        url = f"{self._base_url}{path}"
        resp = await self._client.post(url, json=body)
        resp.raise_for_status()
        return self._unwrap(resp.json())

    @staticmethod
    def _unwrap(data: Any) -> dict[str, Any]:
        """Handle Origin API response envelope.

        Successful responses: ``{"success": true, "code": 10000, "obj": {...}}``
        Error responses:      ``{"success": false, "code": NNNNN, "message": "..."}``
        """
        if not isinstance(data, dict):
            return data

        success = data.get("success")
        code = data.get("code")

        if success is False or (code is not None and code != 10000):
            msg = data.get("message") or data.get("msg") or str(data)
            raise RuntimeError(f"API error (code={code}): {msg}")

        if "obj" in data and data["obj"] is not None:
            return data["obj"]
        if "data" in data:
            return data["data"]
        return data

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Verify connectivity (no login needed — auth is per-request)."""
        logger.info("OriginCloudClient ready (%s)", self._base_url)

    async def stop(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> OriginCloudClient:
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()
