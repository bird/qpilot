"""Readout error mitigation using calibration fidelity matrices.

Each qubit has a 2x2 assignment matrix:
    A_q = [[P(0|0), P(0|1)],
           [P(1|0), P(1|1)]]

where P(m|p) is the probability of measuring outcome m when the true state is p.

For n qubits, the full assignment matrix is the Kronecker product:
    A = A_q0 ⊗ A_q1 ⊗ ... ⊗ A_{q(n-1)}

Given observed counts vector p_obs, the corrected distribution is:
    p_true = A^{-1} @ p_obs
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Max qubits for full matrix inversion (2^n × 2^n grows fast)
_FULL_MATRIX_LIMIT = 12


class ReadoutMitigator:
    """Correct measurement results using readout error calibration data.

    Supports two modes:
    - Full matrix: exact correction for ≤12 qubits (builds 2^n × 2^n matrix)
    - Tensored: scalable per-qubit correction (independent qubit assumption)
    """

    def __init__(
        self,
        fidelity_matrices: list[NDArray[np.float64]],
        qubit_ids: list[str],
    ) -> None:
        """
        Args:
            fidelity_matrices: List of 2x2 assignment matrices, one per qubit.
                Each matrix[i][j] = P(measure=i | prepared=j).
            qubit_ids: Corresponding qubit IDs, in the same order.
        """
        if len(fidelity_matrices) != len(qubit_ids):
            raise ValueError("fidelity_matrices and qubit_ids must have same length")
        self._matrices = fidelity_matrices
        self._qubit_ids = qubit_ids
        self._n = len(qubit_ids)

        # Pre-compute inverse of each 2x2 matrix for tensored correction
        self._inv_matrices: list[NDArray[np.float64]] = []
        for mat in fidelity_matrices:
            det = mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]
            if abs(det) < 1e-15:
                logger.warning("Singular fidelity matrix, using identity")
                self._inv_matrices.append(np.eye(2))
            else:
                inv = np.array(
                    [
                        [mat[1, 1] / det, -mat[0, 1] / det],
                        [-mat[1, 0] / det, mat[0, 0] / det],
                    ]
                )
                self._inv_matrices.append(inv)

    @classmethod
    def from_fidelity_params(
        cls,
        fidelity_pairs: list[tuple[float, float]],
        qubit_ids: list[str],
    ) -> ReadoutMitigator:
        """Build from (P(0|0), P(1|1)) pairs.

        Args:
            fidelity_pairs: List of (p00, p11) tuples.
            qubit_ids: Corresponding qubit IDs.
        """
        matrices = []
        for p00, p11 in fidelity_pairs:
            mat = np.array(
                [
                    [p00, 1 - p11],  # [P(0|0), P(0|1)]
                    [1 - p00, p11],  # [P(1|0), P(1|1)]
                ]
            )
            matrices.append(mat)
        return cls(matrices, qubit_ids)

    @classmethod
    def from_chip_config(
        cls,
        chip_config: dict,
        qubit_ids: list[str],
    ) -> ReadoutMitigator:
        """Build from chip config containing per-qubit FidelityMat.

        The chip config is expected to have a "QubitParams" key mapping
        qubit IDs to parameter dicts with "FidelityMat" 2x2 arrays.
        """
        import json

        qubit_params = chip_config.get("QubitParams", chip_config.get("qubit_params", {}))
        if isinstance(qubit_params, str):
            qubit_params = json.loads(qubit_params)

        matrices = []
        for qid in qubit_ids:
            params = qubit_params.get(qid, {})
            if isinstance(params, str):
                params = json.loads(params)
            fmat = params.get("FidelityMat", params.get("fidelity_mat"))
            if fmat and len(fmat) == 2:
                matrices.append(np.array(fmat, dtype=float))
            else:
                # Default to identity (no correction)
                matrices.append(np.eye(2))
        return cls(matrices, qubit_ids)

    @classmethod
    def from_calibration_results(
        cls,
        results_0: dict[str, int],
        results_1: dict[str, int],
        qubit_ids: list[str],
    ) -> ReadoutMitigator:
        """Build from readout characterization experiment results.

        Args:
            results_0: Counts when all qubits prepared in |0>.
            results_1: Counts when all qubits prepared in |1>.
            qubit_ids: Qubit IDs (determines bit ordering).
        """
        n = len(qubit_ids)
        total_0 = sum(results_0.values()) or 1
        total_1 = sum(results_1.values()) or 1

        matrices = []
        for bit_idx in range(n):
            # Count how often this qubit reads 0 vs 1 in each prep state
            p_0given0 = 0.0
            p_1given0 = 0.0
            p_0given1 = 0.0
            p_1given1 = 0.0

            for bitstring, count in results_0.items():
                bit_val = _get_bit(bitstring, bit_idx, n)
                if bit_val == 0:
                    p_0given0 += count
                else:
                    p_1given0 += count

            for bitstring, count in results_1.items():
                bit_val = _get_bit(bitstring, bit_idx, n)
                if bit_val == 0:
                    p_0given1 += count
                else:
                    p_1given1 += count

            p_0given0 /= total_0
            p_1given0 /= total_0
            p_0given1 /= total_1
            p_1given1 /= total_1

            mat = np.array(
                [
                    [p_0given0, p_0given1],
                    [p_1given0, p_1given1],
                ]
            )
            matrices.append(mat)

        return cls(matrices, qubit_ids)

    @property
    def num_qubits(self) -> int:
        return self._n

    def correct(self, raw_counts: dict[str, int]) -> dict[str, float]:
        """Apply full-matrix readout error correction.

        Builds the 2^n × 2^n assignment matrix as the Kronecker product of
        individual qubit matrices, then solves for the corrected distribution.

        For >12 qubits, automatically falls back to tensored correction.
        """
        if self._n > _FULL_MATRIX_LIMIT:
            return self.correct_tensored(raw_counts)

        dim = 1 << self._n
        # Build full assignment matrix via Kronecker product
        A = self._matrices[0]
        for mat in self._matrices[1:]:
            A = np.kron(A, mat)

        # Build observed probability vector
        total = sum(raw_counts.values()) or 1
        p_obs = np.zeros(dim)
        for bitstring, count in raw_counts.items():
            idx = _bitstring_to_index(bitstring, self._n)
            if 0 <= idx < dim:
                p_obs[idx] = count / total

        # Solve: A @ p_true = p_obs → p_true = A^{-1} @ p_obs
        try:
            p_true = np.linalg.solve(A, p_obs)
        except np.linalg.LinAlgError:
            logger.warning("Assignment matrix is singular, using least squares")
            p_true, _, _, _ = np.linalg.lstsq(A, p_obs, rcond=None)

        # Clip negatives and renormalize
        return _to_distribution(p_true, self._n)

    def correct_tensored(self, raw_counts: dict[str, int]) -> dict[str, float]:
        """Apply per-qubit tensored correction.

        Inverts each qubit's 2x2 matrix independently. O(n * 2^n) vs O(4^n).
        Less accurate for correlated readout errors.
        """
        total = sum(raw_counts.values()) or 1
        dim = 1 << self._n
        p_obs = np.zeros(dim)
        for bitstring, count in raw_counts.items():
            idx = _bitstring_to_index(bitstring, self._n)
            if 0 <= idx < dim:
                p_obs[idx] = count / total

        # Apply per-qubit correction
        p_corrected = p_obs.copy()
        for qubit_idx in range(self._n):
            inv = self._inv_matrices[qubit_idx]
            p_corrected = _apply_single_qubit_correction(p_corrected, inv, qubit_idx, self._n)

        return _to_distribution(p_corrected, self._n)


def _bitstring_to_index(bitstring: str, n_qubits: int) -> int:
    """Convert a bitstring (hex or binary) to an integer index."""
    s = bitstring.strip()
    if s.startswith("0x") or s.startswith("0X"):
        return int(s, 16)
    if s.startswith("0b") or s.startswith("0B"):
        return int(s, 2)
    # Try as plain binary string
    try:
        return int(s, 2)
    except ValueError:
        # Try as decimal
        return int(s)


def _index_to_bitstring(idx: int, n_qubits: int) -> str:
    """Convert an index to a hex bitstring."""
    return hex(idx)


def _get_bit(bitstring: str, bit_idx: int, n_qubits: int) -> int:
    """Extract the bit_idx-th bit from a bitstring (MSB first)."""
    val = _bitstring_to_index(bitstring, n_qubits)
    # bit_idx 0 is the MSB (first qubit)
    shift = n_qubits - 1 - bit_idx
    return (val >> shift) & 1


def _apply_single_qubit_correction(
    probs: NDArray[np.float64],
    inv_matrix: NDArray[np.float64],
    qubit_idx: int,
    n_qubits: int,
) -> NDArray[np.float64]:
    """Apply a single-qubit correction matrix to a probability vector.

    For qubit at position qubit_idx, groups states by that qubit's value
    and applies the 2x2 inverse matrix.
    """
    dim = len(probs)
    result = probs.copy()
    stride = 1 << (n_qubits - 1 - qubit_idx)

    for base in range(dim):
        # Check if this qubit is 0 in this index
        if base & stride:
            continue
        idx_0 = base  # qubit = 0
        idx_1 = base | stride  # qubit = 1

        p0 = probs[idx_0]
        p1 = probs[idx_1]

        result[idx_0] = inv_matrix[0, 0] * p0 + inv_matrix[0, 1] * p1
        result[idx_1] = inv_matrix[1, 0] * p0 + inv_matrix[1, 1] * p1

    return result


def _to_distribution(probs: NDArray[np.float64], n_qubits: int) -> dict[str, float]:
    """Clip negatives, renormalize, and convert to {bitstring: probability} dict."""
    probs = np.maximum(probs, 0.0)
    total = probs.sum()
    if total > 0:
        probs = probs / total

    result = {}
    for idx, p in enumerate(probs):
        if p > 1e-10:
            result[_index_to_bitstring(idx, n_qubits)] = float(p)
    return result
