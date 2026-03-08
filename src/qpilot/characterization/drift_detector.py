"""Detect qubit quality drift between calibrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from qpilot.characterization.noise_profile import NoiseProfile


@dataclass
class QubitDrift:
    """Drift metrics for a single qubit."""

    qubit_id: str
    single_gate_delta: float = 0.0  # current - baseline (negative = degraded)
    readout_delta: float = 0.0  # avg readout fidelity change
    composite_delta: float = 0.0  # composite fidelity change
    drifted: bool = False  # exceeds threshold?


@dataclass
class DriftReport:
    """Summary of drift across all profiled qubits."""

    qubit_drifts: dict[str, QubitDrift] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def drifted_qubits(self) -> list[str]:
        """List qubit IDs that have drifted beyond threshold."""
        return [qid for qid, d in self.qubit_drifts.items() if d.drifted]

    @property
    def stable_qubits(self) -> list[str]:
        """List qubit IDs that remain stable."""
        return [qid for qid, d in self.qubit_drifts.items() if not d.drifted]

    @property
    def worst_qubit(self) -> str | None:
        """Return the qubit with the worst composite fidelity drop."""
        if not self.qubit_drifts:
            return None
        return min(
            self.qubit_drifts,
            key=lambda qid: self.qubit_drifts[qid].composite_delta,
        )


class DriftDetector:
    """Compare current fidelity data against historical baselines.

    Flags qubits whose composite fidelity has dropped below a configurable threshold.
    """

    def __init__(
        self,
        *,
        single_gate_threshold: float = 0.02,
        readout_threshold: float = 0.03,
        composite_threshold: float = 0.02,
    ) -> None:
        """
        Args:
            single_gate_threshold: Max allowed drop in single-gate fidelity.
            readout_threshold: Max allowed drop in average readout fidelity.
            composite_threshold: Max allowed drop in composite fidelity.
        """
        self._sg_thresh = single_gate_threshold
        self._ro_thresh = readout_threshold
        self._comp_thresh = composite_threshold

    def check_drift(
        self,
        current: NoiseProfile,
        baseline: NoiseProfile,
    ) -> QubitDrift:
        """Compare current vs baseline for a single qubit."""
        sg_delta = current.single_gate_fidelity - baseline.single_gate_fidelity

        cur_ro_avg = sum(current.readout_fidelity) / 2
        base_ro_avg = sum(baseline.readout_fidelity) / 2
        ro_delta = cur_ro_avg - base_ro_avg

        comp_delta = current.composite_fidelity - baseline.composite_fidelity

        drifted = (
            sg_delta < -self._sg_thresh
            or ro_delta < -self._ro_thresh
            or comp_delta < -self._comp_thresh
        )

        return QubitDrift(
            qubit_id=current.qubit_id,
            single_gate_delta=sg_delta,
            readout_delta=ro_delta,
            composite_delta=comp_delta,
            drifted=drifted,
        )

    def check_chip_drift(
        self,
        current_profiles: dict[str, NoiseProfile],
        baseline_profiles: dict[str, NoiseProfile],
    ) -> DriftReport:
        """Compare all qubits that appear in both current and baseline."""
        report = DriftReport()
        common_qubits = set(current_profiles) & set(baseline_profiles)

        for qid in common_qubits:
            drift = self.check_drift(current_profiles[qid], baseline_profiles[qid])
            report.qubit_drifts[qid] = drift

        return report

    def should_recalibrate(
        self,
        current: NoiseProfile,
        baseline: NoiseProfile,
    ) -> bool:
        """Has this qubit drifted enough to warrant recalibration?"""
        return self.check_drift(current, baseline).drifted
