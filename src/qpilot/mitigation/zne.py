"""Zero-noise extrapolation (ZNE) for error mitigation.

Runs the same logical circuit at multiple noise levels via gate folding,
then extrapolates measured expectation values to the zero-noise limit.

Gate folding: to scale noise by factor k, replace gate G with G(G†G)^((k-1)/2).
    k=1: G (original)
    k=3: G G† G
    k=5: G G† G G† G

For the superconducting native gate set:
    RPhi(q, axis, angle)†  = RPhi(q, axis, -angle mod 360)
    CZ†                    = CZ  (self-inverse)
    ECHO†                  = ECHO (self-inverse)
    IDLE: not folded (adds decoherence, not gate noise)
    Measure: not folded
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Type alias for an async executor: takes circuit instructions → returns expectation value
CircuitExecutor = Callable[[list[list[dict[str, Any]]]], Awaitable[float]]


class ZNEMitigator:
    """Zero-noise extrapolation via gate folding and curve fitting."""

    def generate_scaled_circuits(
        self,
        circuit: list[list[dict[str, Any]]],
        scale_factors: list[float] | None = None,
    ) -> list[list[list[dict[str, Any]]]]:
        """Generate noise-scaled versions of a circuit via gate folding.

        Args:
            circuit: Native instruction format (list of instruction layers).
            scale_factors: Noise scale factors (must be odd integers ≥ 1).
                Defaults to [1, 3, 5].

        Returns:
            List of circuits, one per scale factor.
        """
        if scale_factors is None:
            scale_factors = [1.0, 3.0, 5.0]

        scaled_circuits = []
        for factor in scale_factors:
            scaled_circuits.append(self._fold_circuit(circuit, factor))
        return scaled_circuits

    def extrapolate(
        self,
        expectation_values: list[float],
        scale_factors: list[float],
        method: str = "linear",
    ) -> float:
        """Extrapolate to the zero-noise limit.

        Args:
            expectation_values: Measured values at each noise scale.
            scale_factors: Corresponding noise scale factors.
            method: Extrapolation method — "linear", "polynomial", or "exponential".

        Returns:
            Extrapolated zero-noise expectation value.
        """
        x = np.array(scale_factors, dtype=float)
        y = np.array(expectation_values, dtype=float)

        if len(x) != len(y) or len(x) < 2:
            raise ValueError("Need at least 2 data points for extrapolation")

        if method == "linear":
            return self._extrapolate_linear(x, y)
        elif method == "polynomial":
            return self._extrapolate_polynomial(x, y)
        elif method == "exponential":
            return self._extrapolate_exponential(x, y)
        else:
            raise ValueError(f"Unknown extrapolation method: {method}")

    async def mitigate(
        self,
        circuit: list[list[dict[str, Any]]],
        executor: CircuitExecutor,
        scale_factors: list[float] | None = None,
        method: str = "linear",
    ) -> float:
        """Full ZNE pipeline: scale → execute → extrapolate.

        Args:
            circuit: Original circuit in native instruction format.
            executor: Async function that takes a circuit and returns an expectation value.
            scale_factors: Noise scale factors.
            method: Extrapolation method.

        Returns:
            Zero-noise extrapolated expectation value.
        """
        if scale_factors is None:
            scale_factors = [1.0, 3.0, 5.0]

        scaled_circuits = self.generate_scaled_circuits(circuit, scale_factors)
        expectation_values = []
        for sc in scaled_circuits:
            val = await executor(sc)
            expectation_values.append(val)

        return self.extrapolate(expectation_values, scale_factors, method)

    def _fold_circuit(
        self,
        circuit: list[list[dict[str, Any]]],
        scale_factor: float,
    ) -> list[list[dict[str, Any]]]:
        """Apply gate folding to scale noise by the given factor.

        For integer odd factor k, each foldable gate G becomes G(G†G)^((k-1)/2).
        For fractional factors, fold a subset of gates.
        """
        if scale_factor < 1.0:
            raise ValueError("Scale factor must be >= 1")
        if abs(scale_factor - 1.0) < 1e-9:
            return copy.deepcopy(circuit)

        # Collect all foldable gates from all layers
        all_gates: list[tuple[int, int, dict[str, Any]]] = []
        for layer_idx, layer in enumerate(circuit):
            for gate_idx, gate in enumerate(layer):
                if _is_foldable(gate):
                    all_gates.append((layer_idx, gate_idx, gate))

        n_gates = len(all_gates)
        if n_gates == 0:
            return copy.deepcopy(circuit)

        # Number of full folds (G†G pairs per gate) and remainder
        full_folds = int((scale_factor - 1) / 2)
        remainder = scale_factor - 1 - 2 * full_folds
        n_extra = int(round(remainder / 2 * n_gates)) if remainder > 0.01 else 0

        # Build the folded circuit as a flat instruction list
        result_layer: list[dict[str, Any]] = []
        order = 0

        for layer in circuit:
            for gate in layer:
                if not _is_foldable(gate):
                    # Non-foldable gates (Measure, IDLE) pass through
                    updated = _set_order(gate, order)
                    result_layer.append(updated)
                    order += 1
                    continue

                # Original gate
                result_layer.append(_set_order(gate, order))
                order += 1

                # Full folds: G†G repeated
                for _ in range(full_folds):
                    inv = _invert_gate(gate)
                    result_layer.append(_set_order(inv, order))
                    order += 1
                    result_layer.append(_set_order(gate, order))
                    order += 1

        # Apply extra folds to first n_extra foldable gates
        if n_extra > 0:
            extra_gates = []
            count = 0
            for gate in result_layer:
                if _is_foldable(gate) and count < n_extra:
                    inv = _invert_gate(gate)
                    extra_gates.append(_set_order(inv, order))
                    order += 1
                    extra_gates.append(_set_order(gate, order))
                    order += 1
                    count += 1
            result_layer.extend(extra_gates)

        return [result_layer]

    @staticmethod
    def _extrapolate_linear(x: np.ndarray, y: np.ndarray) -> float:
        """Linear fit: y = a + b*x, return a (x=0 intercept)."""
        coeffs = np.polyfit(x, y, 1)
        return float(np.polyval(coeffs, 0.0))

    @staticmethod
    def _extrapolate_polynomial(x: np.ndarray, y: np.ndarray) -> float:
        """Polynomial fit through all points, return f(0)."""
        degree = min(len(x) - 1, 4)  # cap at degree 4
        coeffs = np.polyfit(x, y, degree)
        return float(np.polyval(coeffs, 0.0))

    @staticmethod
    def _extrapolate_exponential(x: np.ndarray, y: np.ndarray) -> float:
        """Exponential fit: y = a * exp(b*x), return a (x=0 value).

        Uses log-linear regression for robustness.
        """
        # Filter positive values for log
        valid = y > 0
        if valid.sum() < 2:
            # Fallback to linear
            return ZNEMitigator._extrapolate_linear(x, y)

        log_y = np.log(y[valid])
        x_valid = x[valid]

        coeffs = np.polyfit(x_valid, log_y, 1)
        # log(y) = log(a) + b*x → y(0) = exp(log(a)) = exp(coeffs[1])
        return float(np.exp(coeffs[1]))


def _is_foldable(gate: dict[str, Any]) -> bool:
    """Check if a gate can be folded (has a well-defined inverse)."""
    name = next(iter(gate))
    return name in ("RPhi", "CZ", "ECHO")


def _invert_gate(gate: dict[str, Any]) -> dict[str, Any]:
    """Compute the inverse of a native gate."""
    name = next(iter(gate))
    params = gate[name]

    if name == "RPhi":
        # RPhi(q, axis, angle, order) → RPhi(q, axis, -angle mod 360, order)
        q, axis, angle, order = params
        return {"RPhi": [q, axis, (360 - angle) % 360, order]}
    elif name == "CZ":
        # CZ is self-inverse
        return copy.deepcopy(gate)
    elif name == "ECHO":
        # ECHO is self-inverse
        return copy.deepcopy(gate)
    else:
        raise ValueError(f"Cannot invert gate: {name}")


def _set_order(gate: dict[str, Any], order: int) -> dict[str, Any]:
    """Create a copy of the gate with an updated order/timing value."""
    name = next(iter(gate))
    params = list(gate[name])

    if name == "RPhi" and len(params) == 4:
        params[3] = order
    elif name == "CZ" and len(params) == 3:
        params[2] = order
    elif name == "ECHO" and len(params) == 2:
        params[1] = order
    elif name == "IDLE" and len(params) == 3:
        params[2] = order
    elif name == "Measure" and len(params) == 2:
        params[1] = order
    else:
        params = list(gate[name])  # pass through unknown

    return {name: params}
