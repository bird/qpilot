"""Calibration-aware experiment scheduling."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from qpilot.harness.experiment import Experiment, ExperimentResult
from qpilot.harness.runner import ExperimentRunner
from qpilot.monitor.chip_monitor import ChipMonitor

logger = logging.getLogger(__name__)


@dataclass
class ScheduledExperiment:
    """An experiment queued for execution."""

    experiment: Experiment
    runner_kwargs: dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # higher = runs first


class ExperimentScheduler:
    """Schedule experiments around calibration windows.

    Maintains a priority queue of experiments and processes them
    when the chip is available.
    """

    def __init__(
        self,
        runner: ExperimentRunner,
        monitor: ChipMonitor,
        *,
        queue_threshold: int = 50,
    ) -> None:
        self._runner = runner
        self._monitor = monitor
        self._queue_threshold = queue_threshold
        self._queue: list[ScheduledExperiment] = []
        self._results: list[ExperimentResult] = []
        self._running = False

    @property
    def results(self) -> list[ExperimentResult]:
        return list(self._results)

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    def submit(
        self,
        experiment: Experiment,
        priority: int = 0,
        **runner_kwargs: Any,
    ) -> None:
        """Add experiment to the scheduling queue."""
        self._queue.append(
            ScheduledExperiment(
                experiment=experiment,
                runner_kwargs=runner_kwargs,
                priority=priority,
            )
        )
        # Sort by priority (highest first)
        self._queue.sort(key=lambda x: x.priority, reverse=True)
        logger.info(
            "Scheduled %s (priority=%d, queue=%d)", experiment.name, priority, len(self._queue)
        )

    async def run(self, max_experiments: int | None = None) -> list[ExperimentResult]:
        """Process experiment queue.

        Runs experiments when the chip is available, automatically pausing
        during calibration and maintenance.
        """
        self._running = True
        processed = 0

        while self._running and self._queue:
            if max_experiments is not None and processed >= max_experiments:
                break

            # Wait for chip availability
            if not self.is_chip_available():
                logger.info("Chip not available, waiting...")
                await asyncio.sleep(2.0)
                continue

            entry = self._queue.pop(0)
            logger.info("Running scheduled experiment: %s", entry.experiment.name)

            try:
                result = await self._runner.run_once(entry.experiment)
                self._results.append(result)
                processed += 1
            except Exception:
                logger.exception("Scheduled experiment %s failed", entry.experiment.name)

        self._running = False
        return list(self._results)

    def stop(self) -> None:
        """Signal the scheduler to stop after the current experiment."""
        self._running = False

    def is_chip_available(self) -> bool:
        """Check if the chip is available for experiments."""
        state = self._monitor.state
        return (
            not state.calibrating
            and not state.protected
            and state.queue_length < self._queue_threshold
        )
