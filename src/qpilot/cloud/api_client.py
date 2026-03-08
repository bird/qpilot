"""Async HTTP client for Origin Quantum Cloud API."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_BASE_URL = "https://qcloud.originqc.com.cn"

# Login paths to try in order (API has moved between deployments)
_LOGIN_PATHS = [
    "/api/management/pilotosmachinelogin",
    "/management/pilotosmachinelogin",
]


class OriginCloudClient:
    """Async client for the Origin Quantum Cloud HTTP API.

    Handles authentication, chip config, calibration data, and queue queries.
    No QPU time is consumed by data-fetching endpoints.

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
        self._token: str | None = None
        self._api_prefix = ""  # determined during login
        self._client = httpx.AsyncClient(timeout=timeout)

    async def login(self) -> str:
        """Authenticate and obtain a session token.

        Tries multiple known login paths to handle API changes between deployments.
        """
        last_error = None
        for path in _LOGIN_PATHS:
            try:
                resp = await self._raw_post(path, {"apiKey": self._api_key})
                token = (
                    resp.get("token")
                    or resp.get("Token")
                    or (resp.get("data", {}) or {}).get("token", "")
                )
                if token:
                    self._token = token
                    # Remember which prefix worked
                    self._api_prefix = path.rsplit("/management/", 1)[0]
                    logger.info("Authenticated via %s", path)
                    return token
                last_error = RuntimeError(f"No token in response from {path}: {resp}")
            except Exception as exc:
                last_error = exc
                logger.debug("Login path %s failed: %s", path, exc)
                continue

        raise RuntimeError(f"All login paths failed. Last error: {last_error}")

    async def get_chip_config(
        self,
        chip_id: str = "72",
        label: int = 1,
    ) -> dict[str, Any]:
        """Fetch full chip configuration (fidelity matrices, topology, params)."""
        return await self._post("/management/getchipconfig", {"ChipID": chip_id, "Label": label})

    async def get_queue_info(
        self,
        backend: str = "72",
        max_tasks: int = 500,
    ) -> dict[str, Any]:
        """Query current task queue depth and status."""
        return await self._post(
            "/system/queryBackendQueueTask",
            {"backend": backend, "maxQueryTaskSize": max_tasks},
        )

    async def get_best_qubit_blocks(
        self,
        chip_id: str = "72",
        label: int = 1,
        qubit_num: int = 4,
        block_num: int = 3,
    ) -> dict[str, Any]:
        """Query pre-computed best qubit blocks."""
        return await self._post(
            "/management/query/best_qubit_blocks",
            {"ChipID": chip_id, "Label": label, "QubitNum": qubit_num, "QubitBlockNum": block_num},
        )

    async def get_calibration_timestamps(
        self,
        backend: str = "72",
    ) -> dict[str, Any]:
        """Query last calibration update times."""
        return await self._post("/system/queryChipUpdateTime", {"backend": backend})

    async def _post(
        self,
        path: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        """Send an authenticated POST request."""
        if self._token:
            body = {**body, "token": self._token}
        return await self._raw_post(f"{self._api_prefix}{path}", body)

    async def _raw_post(
        self,
        path: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        """Send a POST request and handle Origin API response wrapping."""
        url = f"{self._base_url}{path}"
        resp = await self._client.post(url, json=body)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, dict) and "code" in data:
            code = data.get("code")
            if code not in (0, "0", None, "success"):
                msg = data.get("message") or data.get("msg") or str(data)
                raise RuntimeError(f"API error (code={code}): {msg}")
            if "data" in data:
                return data["data"]
        return data

    async def start(self) -> None:
        """Login to obtain session token."""
        await self.login()

    async def stop(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> OriginCloudClient:
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()
