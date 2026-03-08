"""Per-qubit and chip-level noise profiling.

Builds noise fingerprints from two sources:
1. Fast path: RB data + chip config via ZMQ (no circuits needed).
2. Deep path: Run characterization circuits and analyze results.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from qpilot.models.responses import GetChipConfigResponse, GetRBDataResponse

logger = logging.getLogger(__name__)


@dataclass
class NoiseProfile:
    """Noise fingerprint for a single qubit."""

    qubit_id: str
    single_gate_fidelity: float = 0.0
    two_gate_fidelities: dict[str, float] = field(default_factory=dict)
    readout_fidelity: tuple[float, float] = (1.0, 1.0)  # (P(0|0), P(1|1))
    t1_us: float | None = None
    t2star_us: float | None = None
    last_calibration: datetime | None = None
    last_profiled: datetime | None = None

    @property
    def composite_fidelity(self) -> float:
        """Composite score: geometric mean of single-gate and readout fidelities."""
        p00, p11 = self.readout_fidelity
        readout_avg = (p00 + p11) / 2
        return math.sqrt(self.single_gate_fidelity * readout_avg)


@dataclass
class ChipNoiseProfile:
    """Aggregate noise profile for an entire chip."""

    profiles: dict[str, NoiseProfile] = field(default_factory=dict)
    topology: dict[str, list[str]] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def best_qubits(self, n: int) -> list[str]:
        """Return the n qubits with highest composite fidelity."""
        ranked = sorted(
            self.profiles.values(),
            key=lambda p: p.composite_fidelity,
            reverse=True,
        )
        return [p.qubit_id for p in ranked[:n]]

    def best_connected_subgraph(self, n: int) -> list[str]:
        """Return n connected qubits with highest aggregate composite fidelity.

        Uses greedy expansion: start from the best qubit, then repeatedly add
        the highest-fidelity neighbor until we have n qubits.
        """
        if not self.profiles:
            return []
        if n <= 0:
            return []

        # Start from the single best qubit
        ranked = sorted(
            self.profiles.values(),
            key=lambda p: p.composite_fidelity,
            reverse=True,
        )
        selected = {ranked[0].qubit_id}

        while len(selected) < n:
            best_neighbor = None
            best_score = -1.0
            for qid in list(selected):
                for neighbor in self.topology.get(qid, []):
                    if neighbor in selected or neighbor not in self.profiles:
                        continue
                    score = self.profiles[neighbor].composite_fidelity
                    if score > best_score:
                        best_score = score
                        best_neighbor = neighbor
            if best_neighbor is None:
                break  # no more connected neighbors available
            selected.add(best_neighbor)

        return list(selected)

    def pair_fidelity(self, q1: str, q2: str) -> float | None:
        """Get two-qubit gate fidelity between a pair, if available."""
        profile = self.profiles.get(q1)
        if profile is None:
            return None
        return profile.two_gate_fidelities.get(q2)


class NoiseProfiler:
    """Builds noise profiles from ZMQ data and/or characterization results."""

    @staticmethod
    def from_rb_data(
        rb_data: GetRBDataResponse,
        chip_config: GetChipConfigResponse | None = None,
    ) -> ChipNoiseProfile:
        """Build a noise profile from RB data and optional chip config (fast path).

        This uses pre-existing calibration data — no circuits need to be run.
        """
        now = datetime.now(UTC)
        profiles: dict[str, NoiseProfile] = {}
        topology: dict[str, list[str]] = {}

        # Single-gate fidelities
        if rb_data.single_gate_fidelity:
            for qid, fid in zip(
                rb_data.single_gate_fidelity.qubit,
                rb_data.single_gate_fidelity.fidelity,
            ):
                profiles[qid] = NoiseProfile(
                    qubit_id=qid,
                    single_gate_fidelity=fid,
                    last_profiled=now,
                )

        # Two-gate fidelities and topology
        if rb_data.double_gate_fidelity:
            for pair_str, fid in zip(
                rb_data.double_gate_fidelity.qubit_pair,
                rb_data.double_gate_fidelity.fidelity,
            ):
                parts = pair_str.split("-")
                if len(parts) != 2:
                    continue
                q1, q2 = parts

                # Ensure both profiles exist
                if q1 not in profiles:
                    profiles[q1] = NoiseProfile(qubit_id=q1, last_profiled=now)
                if q2 not in profiles:
                    profiles[q2] = NoiseProfile(qubit_id=q2, last_profiled=now)

                profiles[q1].two_gate_fidelities[q2] = fid
                profiles[q2].two_gate_fidelities[q1] = fid

                topology.setdefault(q1, [])
                topology.setdefault(q2, [])
                if q2 not in topology[q1]:
                    topology[q1].append(q2)
                if q1 not in topology[q2]:
                    topology[q2].append(q1)

        # Extract readout fidelity from chip config if available
        if chip_config and chip_config.chip_config:
            _enrich_from_chip_config(profiles, chip_config)

        return ChipNoiseProfile(profiles=profiles, topology=topology, timestamp=now)

    @staticmethod
    def from_readout_results(
        results: dict[int, tuple[dict[str, int], dict[str, int]]],
        existing: ChipNoiseProfile | None = None,
    ) -> ChipNoiseProfile:
        """Update noise profile with readout characterization results.

        Args:
            results: qubit_index → (prep0_counts, prep1_counts).
                     Each counts dict maps bit string ("0x0"/"0x1") to count.
            existing: Existing profile to update (or create new).
        """
        profile = existing or ChipNoiseProfile()
        now = datetime.now(UTC)

        for q_idx, (prep0, prep1) in results.items():
            qid = str(q_idx)
            total_0 = sum(prep0.values()) or 1
            total_1 = sum(prep1.values()) or 1

            # P(0|0) = fraction of "0" outcomes when preparing |0>
            p00 = prep0.get("0x0", 0) / total_0
            # P(1|1) = fraction of "1" outcomes when preparing |1>
            p11 = prep1.get("0x1", 0) / total_1

            if qid in profile.profiles:
                profile.profiles[qid].readout_fidelity = (p00, p11)
                profile.profiles[qid].last_profiled = now
            else:
                profile.profiles[qid] = NoiseProfile(
                    qubit_id=qid,
                    readout_fidelity=(p00, p11),
                    last_profiled=now,
                )

        return profile

    @staticmethod
    def fit_t1(
        delays: list[int],
        survival_probs: list[float],
    ) -> float | None:
        """Fit T1 from delay vs P(1) data using exponential decay.

        Model: P(1) = A * exp(-t / T1) + B

        Returns T1 in the same units as delays, or None if fit fails.
        """
        if len(delays) < 3 or len(delays) != len(survival_probs):
            return None
        try:
            t = np.array(delays, dtype=float)
            p = np.array(survival_probs, dtype=float)

            # Filter out zero/negative probabilities for log fit
            valid = p > 0
            if valid.sum() < 3:
                return None

            # Simple log-linear fit: ln(P) ≈ -t/T1 + ln(A)
            log_p = np.log(p[valid])
            t_valid = t[valid]

            # Linear regression: log_p = slope * t + intercept
            n = len(t_valid)
            sum_t = t_valid.sum()
            sum_lp = log_p.sum()
            sum_t2 = (t_valid**2).sum()
            sum_tlp = (t_valid * log_p).sum()

            denom = n * sum_t2 - sum_t**2
            if abs(denom) < 1e-15:
                return None

            slope = (n * sum_tlp - sum_t * sum_lp) / denom

            if slope >= 0:
                return None  # Not a decay — data is likely random

            return -1.0 / slope
        except Exception:
            logger.debug("T1 fit failed", exc_info=True)
            return None

    @staticmethod
    def fit_t2star(
        delays: list[int],
        ramsey_probs: list[float],
    ) -> float | None:
        """Fit T2* from Ramsey experiment data.

        Model: P(0) = 0.5 + 0.5 * exp(-t / T2*) * cos(w*t + phi)

        For a simple estimate, we fit the envelope: |P - 0.5| ~ 0.5 * exp(-t/T2*).

        Returns T2* in the same units as delays, or None if fit fails.
        """
        if len(delays) < 3 or len(delays) != len(ramsey_probs):
            return None
        try:
            t = np.array(delays, dtype=float)
            p = np.array(ramsey_probs, dtype=float)

            # Envelope: deviation from 0.5
            envelope = np.abs(p - 0.5)
            valid = envelope > 0.01  # filter noise floor
            if valid.sum() < 3:
                return None

            log_env = np.log(envelope[valid])
            t_valid = t[valid]

            n = len(t_valid)
            sum_t = t_valid.sum()
            sum_le = log_env.sum()
            sum_t2 = (t_valid**2).sum()
            sum_tle = (t_valid * log_env).sum()

            denom = n * sum_t2 - sum_t**2
            if abs(denom) < 1e-15:
                return None

            slope = (n * sum_tle - sum_t * sum_le) / denom

            if slope >= 0:
                return None

            return -1.0 / slope
        except Exception:
            logger.debug("T2* fit failed", exc_info=True)
            return None


def _enrich_from_chip_config(
    profiles: dict[str, NoiseProfile],
    chip_config: GetChipConfigResponse,
) -> None:
    """Extract per-qubit readout fidelity from chip config data.

    ChipConfig values may be JSON strings containing QubitParams with FidelityMat.
    """
    import json

    for _label, config_str in chip_config.chip_config.items():
        try:
            if isinstance(config_str, str):
                config = json.loads(config_str)
            else:
                config = config_str
        except (json.JSONDecodeError, TypeError):
            continue

        if not isinstance(config, dict):
            continue

        # Look for qubit params with fidelity matrices
        qubit_params = config.get("QubitParams", config.get("qubit_params", {}))
        if isinstance(qubit_params, dict):
            for qid, params in qubit_params.items():
                if not isinstance(params, dict):
                    continue
                fidelity_mat = params.get("FidelityMat", params.get("fidelity_mat"))
                if fidelity_mat and isinstance(fidelity_mat, list) and len(fidelity_mat) == 2:
                    try:
                        p00 = float(fidelity_mat[0][0]) if len(fidelity_mat[0]) > 0 else 1.0
                        p11 = float(fidelity_mat[1][1]) if len(fidelity_mat[1]) > 1 else 1.0
                        if qid in profiles:
                            profiles[qid].readout_fidelity = (p00, p11)
                    except (IndexError, TypeError, ValueError):
                        continue
