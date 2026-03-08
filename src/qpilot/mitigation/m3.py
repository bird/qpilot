"""Matrix-free measurement mitigation (M3).

Scalable alternative to full readout mitigation. Instead of building the
complete 2^n × 2^n assignment matrix, only constructs the submatrix
corresponding to bitstrings that actually appear in the measurement results.

This makes mitigation practical for circuits with many qubits where only
a small fraction of possible bitstrings are observed.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class M3Mitigator:
    """Matrix-free measurement mitigation.

    Builds a reduced assignment matrix using only the observed bitstrings,
    solving a much smaller linear system than full readout mitigation.
    """

    def __init__(self, fidelity_matrices: dict[str, list[list[float]]]) -> None:
        """
        Args:
            fidelity_matrices: Per-qubit fidelity matrices keyed by qubit ID.
                Each is a 2x2 list: [[P(0|0), P(0|1)], [P(1|0), P(1|1)]].
        """
        self._fidelity = {qid: np.array(mat, dtype=float) for qid, mat in fidelity_matrices.items()}
        self._qubit_order: list[str] = sorted(self._fidelity.keys())

    @classmethod
    def from_fidelity_params(
        cls,
        fidelity_pairs: dict[str, tuple[float, float]],
    ) -> M3Mitigator:
        """Build from (P(0|0), P(1|1)) pairs.

        Args:
            fidelity_pairs: {qubit_id: (p00, p11)}.
        """
        matrices = {}
        for qid, (p00, p11) in fidelity_pairs.items():
            matrices[qid] = [[p00, 1 - p11], [1 - p00, p11]]
        return cls(matrices)

    def correct(
        self,
        raw_counts: dict[str, int],
        threshold: float = 0.001,
    ) -> dict[str, float]:
        """Apply M3 correction to raw measurement counts.

        1. Identify unique bitstrings in raw_counts.
        2. Build reduced assignment matrix (only observed bitstrings).
        3. Solve the reduced system via least squares.
        4. Clip negatives, renormalize.

        Args:
            raw_counts: Measurement counts {bitstring: count}.
            threshold: Minimum probability to retain in output.
        """
        if not raw_counts:
            return {}

        n_qubits = len(self._qubit_order)
        if n_qubits == 0:
            return _normalize_counts(raw_counts)

        # Collect unique bitstrings and their observed probabilities
        total = sum(raw_counts.values()) or 1
        bitstrings = list(raw_counts.keys())
        indices = [_bitstring_to_int(bs) for bs in bitstrings]
        p_obs = np.array([raw_counts[bs] / total for bs in bitstrings], dtype=float)

        m = len(bitstrings)

        # Build reduced assignment matrix A_red (m × m)
        # A_red[i, j] = probability of measuring bitstring i given true state is bitstring j
        A_red = np.zeros((m, m))
        for i, idx_i in enumerate(indices):
            for j, idx_j in enumerate(indices):
                A_red[i, j] = self._assignment_element(idx_i, idx_j, n_qubits)

        # Solve: A_red @ p_true = p_obs
        try:
            p_true, _, _, _ = np.linalg.lstsq(A_red, p_obs, rcond=None)
        except np.linalg.LinAlgError:
            logger.warning("M3 least-squares failed, returning normalized counts")
            return _normalize_counts(raw_counts)

        # Clip negatives and renormalize
        p_true = np.maximum(p_true, 0.0)
        total_p = p_true.sum()
        if total_p > 0:
            p_true /= total_p

        # Build result dict
        result: dict[str, float] = {}
        for bs, p in zip(bitstrings, p_true):
            if p > threshold:
                result[bs] = float(p)

        return result

    def _assignment_element(self, meas_idx: int, true_idx: int, n_qubits: int) -> float:
        """Compute A[meas_idx, true_idx] = product of per-qubit assignment probs.

        For each qubit q:
            bit_m = bit q of meas_idx
            bit_t = bit q of true_idx
            P(bit_m | bit_t) = fidelity_matrix[bit_m][bit_t]
        """
        prob = 1.0
        for q_pos, qid in enumerate(self._qubit_order):
            mat = self._fidelity[qid]
            shift = n_qubits - 1 - q_pos
            bit_m = (meas_idx >> shift) & 1
            bit_t = (true_idx >> shift) & 1
            prob *= mat[bit_m, bit_t]
        return prob


def _bitstring_to_int(bitstring: str) -> int:
    """Convert a bitstring (hex or binary) to an integer."""
    s = bitstring.strip()
    if s.startswith("0x") or s.startswith("0X"):
        return int(s, 16)
    if s.startswith("0b") or s.startswith("0B"):
        return int(s, 2)
    try:
        return int(s, 2)
    except ValueError:
        return int(s)


def _normalize_counts(raw_counts: dict[str, int]) -> dict[str, float]:
    """Normalize counts to probabilities."""
    total = sum(raw_counts.values()) or 1
    return {k: v / total for k, v in raw_counts.items()}
