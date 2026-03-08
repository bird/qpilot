"""Automated experiment runner with chip-awareness."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from qpilot.characterization.noise_profile import ChipNoiseProfile, NoiseProfiler
from qpilot.client import QPilotClient
from qpilot.enums import TaskStatus
from qpilot.harness.experiment import (
    ContinueExperiment,
    Experiment,
    ExperimentCircuit,
    ExperimentResult,
)

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Runs experiments in an automated loop with chip awareness.

    Handles the full lifecycle: design → submit → poll → collect → analyze.
    """

    def __init__(
        self,
        client: QPilotClient,
        *,
        noise_profile: ChipNoiseProfile | None = None,
        poll_interval: float = 1.0,
        max_poll_attempts: int = 300,
    ) -> None:
        self.client = client
        self._noise_profile = noise_profile
        self._poll_interval = poll_interval
        self._max_poll_attempts = max_poll_attempts
        self.results: list[ExperimentResult] = []

    @property
    def noise_profile(self) -> ChipNoiseProfile | None:
        return self._noise_profile

    @noise_profile.setter
    def noise_profile(self, value: ChipNoiseProfile | None) -> None:
        self._noise_profile = value

    async def refresh_noise_profile(self) -> ChipNoiseProfile:
        """Fetch fresh RB data and rebuild noise profile."""
        rb_data = await self.client.get_rb_data()
        self._noise_profile = NoiseProfiler.from_rb_data(rb_data)
        return self._noise_profile

    async def run_once(self, experiment: Experiment) -> ExperimentResult:
        """Run a single iteration of an experiment.

        1. Get current chip state and noise profile
        2. Call experiment.design() to get circuits
        3. Submit each circuit via ZMQ MsgTask
        4. Poll for completion
        5. Collect results
        6. Call experiment.analyze()
        7. Return ExperimentResult
        """
        t0 = time.monotonic()
        experiment_id = f"{experiment.name}_{uuid.uuid4().hex[:8]}"

        chip_state = self.client.monitor.state
        logger.info("Starting experiment %s", experiment_id)

        # Design phase
        circuits = await experiment.design(chip_state, self._noise_profile)
        if not circuits:
            return ExperimentResult(
                experiment_id=experiment_id,
                circuits_submitted=0,
                circuits_completed=0,
                raw_results=[],
                analysis={"status": "no_circuits"},
                duration_seconds=time.monotonic() - t0,
            )

        # Submit phase
        task_ids = await self._submit_circuits(circuits, experiment_id)

        # Collect phase
        raw_results = await self._collect_results(task_ids)
        completed = sum(1 for r in raw_results if r.get("status") == "completed")

        # Analyze phase
        analysis = await experiment.analyze(raw_results)

        result = ExperimentResult(
            experiment_id=experiment_id,
            circuits_submitted=len(circuits),
            circuits_completed=completed,
            raw_results=raw_results,
            analysis=analysis,
            duration_seconds=time.monotonic() - t0,
        )
        self.results.append(result)
        logger.info(
            "Experiment %s done: %d/%d circuits, %.1fs",
            experiment_id,
            completed,
            len(circuits),
            result.duration_seconds,
        )
        return result

    async def run_loop(
        self,
        experiment: Experiment,
        max_iterations: int = 100,
        pause_during_calibration: bool = True,
        cooldown_seconds: float = 5.0,
    ) -> list[ExperimentResult]:
        """Run experiment in a loop with automatic pausing.

        Pauses during calibration and maintenance windows.
        Catches errors without crashing the loop.
        """
        results: list[ExperimentResult] = []

        for i in range(max_iterations):
            # Check chip availability
            if pause_during_calibration:
                await self._wait_for_chip(timeout=300.0)

            try:
                result = await self.run_once(experiment)
                results.append(result)
            except ContinueExperiment:
                logger.info("Experiment requested continuation (iteration %d)", i)
                continue
            except Exception:
                logger.exception("Experiment iteration %d failed", i)
                if cooldown_seconds > 0:
                    await asyncio.sleep(cooldown_seconds)
                continue

            if cooldown_seconds > 0:
                await asyncio.sleep(cooldown_seconds)

        return results

    async def run_with_vip(
        self,
        experiment: Experiment,
        duration_seconds: int = 600,
        offset_seconds: int = 0,
        **kwargs: Any,
    ) -> list[ExperimentResult]:
        """Run experiment with exclusive VIP chip time.

        Acquires VIP time, runs the experiment loop, then releases.
        """
        await self.client.set_vip(
            offset_time=offset_seconds,
            exclusive_time=duration_seconds,
        )
        logger.info("VIP time acquired for %ds", duration_seconds)
        try:
            return await self.run_loop(experiment, **kwargs)
        finally:
            await self.client.release_vip()
            logger.info("VIP time released")

    async def _submit_circuits(
        self,
        circuits: list[ExperimentCircuit],
        experiment_id: str,
    ) -> list[str]:
        """Submit circuits and return task IDs."""
        task_ids = []
        for i, ec in enumerate(circuits):
            task_id = f"{experiment_id}_c{i}_{uuid.uuid4().hex[:6]}"
            await self.client.submit_task(
                task_id=task_id,
                circuit=ec.circuit,
                shots=ec.shots,
                is_experiment=True,
            )
            task_ids.append(task_id)
        return task_ids

    async def _collect_results(
        self,
        task_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Poll for task completion and collect results."""
        results: list[dict[str, Any]] = []

        for task_id in task_ids:
            result = await self._wait_for_task(task_id)
            results.append(result)

        return results

    async def _wait_for_task(self, task_id: str) -> dict[str, Any]:
        """Poll a single task until terminal state."""
        for _ in range(self._max_poll_attempts):
            status_resp = await self.client.get_task_status(task_id)
            status = TaskStatus(status_resp.task_status)

            if status == TaskStatus.COMPLETED:
                result_resp = await self.client.get_task_result(task_id)
                return {
                    "task_id": task_id,
                    "status": "completed",
                    "key": result_resp.key,
                    "prob_count": result_resp.prob_count,
                    "note_time": {
                        "compile_time": result_resp.note_time.compile_time,
                        "measure_time": result_resp.note_time.measure_time,
                    }
                    if result_resp.note_time
                    else {},
                }
            elif status.is_terminal:
                return {"task_id": task_id, "status": "failed", "task_status": int(status)}

            await asyncio.sleep(self._poll_interval)

        return {"task_id": task_id, "status": "timeout"}

    async def _wait_for_chip(self, timeout: float = 300.0) -> None:
        """Wait until the chip is not calibrating or in maintenance."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.client.monitor.state
            if not state.calibrating and not state.protected:
                return
            logger.info(
                "Chip busy (calibrating=%s, protected=%s), waiting...",
                state.calibrating,
                state.protected,
            )
            await asyncio.sleep(2.0)
        logger.warning("Chip availability timeout after %.0fs", timeout)
