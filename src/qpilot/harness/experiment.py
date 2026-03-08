"""Experiment definition and lifecycle types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from qpilot.characterization.noise_profile import ChipNoiseProfile
from qpilot.monitor.chip_monitor import ChipState


class ContinueExperiment(Exception):
    """Raise from Experiment.analyze() to request another iteration."""


@dataclass
class ExperimentCircuit:
    """A circuit to be submitted as part of an experiment."""

    circuit: list[list[dict[str, Any]]]
    shots: int = 1000
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Result of a single experiment iteration."""

    experiment_id: str
    circuits_submitted: int
    circuits_completed: int
    raw_results: list[dict[str, Any]]
    analysis: dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    duration_seconds: float = 0.0


class Experiment(ABC):
    """Base class for automated experiments.

    Subclass and implement design() and analyze() to create custom experiments.
    The runner calls design() to get circuits, submits them, collects results,
    then calls analyze().
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    async def design(
        self,
        chip_state: ChipState,
        noise_profile: ChipNoiseProfile | None,
    ) -> list[ExperimentCircuit]:
        """Design circuits based on current chip state.

        Called by the runner at the start of each iteration.

        Args:
            chip_state: Current live chip status.
            noise_profile: Current noise profile (if available).

        Returns:
            List of circuits to submit.
        """

    @abstractmethod
    async def analyze(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze results from submitted circuits.

        Called after all circuits complete.
        Raise ContinueExperiment to request another iteration.

        Args:
            results: Raw measurement results from each circuit.

        Returns:
            Analysis dict (stored in ExperimentResult).
        """
