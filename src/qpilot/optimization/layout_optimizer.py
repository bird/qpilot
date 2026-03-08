"""Circuit-to-hardware layout optimization.

Remaps logical qubit indices to physical qubit IDs in native instruction
arrays. Currently a pass-through (no reordering or scheduling).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qpilot.characterization.noise_profile import ChipNoiseProfile
from qpilot.optimization.qubit_selector import QubitMapping


@dataclass
class OptimizedLayout:
    """Result of layout optimization."""

    mapping: QubitMapping
    instructions: list[list[dict[str, Any]]]
    estimated_fidelity: float = 0.0


class LayoutOptimizer:
    """Remap logical qubit indices to physical qubit IDs in instructions."""

    def __init__(self, noise_profile: ChipNoiseProfile) -> None:
        self._profile = noise_profile

    def optimize(
        self,
        mapping: QubitMapping,
        instructions: list[list[dict[str, Any]]],
        logical_qubits: list[int],
    ) -> OptimizedLayout:
        """Remap logical qubit indices to physical qubit IDs in instructions.

        Args:
            mapping: Logical-to-physical qubit mapping.
            instructions: Circuit instructions with logical qubit indices.
            logical_qubits: List of logical qubit indices used in the circuit.

        Returns:
            OptimizedLayout with remapped instructions.
        """
        # Build logical → physical int index mapping
        phys_map: dict[int, int] = {}
        for lq in logical_qubits:
            pq_str = mapping.mapping.get(lq)
            if pq_str is not None:
                # Physical qubit IDs may be strings like "45"; convert to int
                try:
                    phys_map[lq] = int(pq_str)
                except ValueError:
                    phys_map[lq] = lq  # fallback

        remapped = _remap_instructions(instructions, phys_map)
        return OptimizedLayout(
            mapping=mapping,
            instructions=remapped,
            estimated_fidelity=mapping.score,
        )


def _remap_instructions(
    instructions: list[list[dict[str, Any]]],
    qubit_map: dict[int, int],
) -> list[list[dict[str, Any]]]:
    """Remap qubit indices in native instruction arrays."""
    result = []
    for layer in instructions:
        new_layer = []
        for gate in layer:
            new_layer.append(_remap_gate(gate, qubit_map))
        result.append(new_layer)
    return result


def _remap_gate(gate: dict[str, Any], qubit_map: dict[int, int]) -> dict[str, Any]:
    """Remap qubit indices in a single gate instruction."""
    new_gate = {}
    for name, params in gate.items():
        if not isinstance(params, list):
            new_gate[name] = params
            continue

        if name == "RPhi" and len(params) == 4:
            # RPhi(qubit, axis_deg, angle_deg, order)
            new_gate[name] = [
                qubit_map.get(params[0], params[0]),
                params[1],
                params[2],
                params[3],
            ]
        elif name == "CZ" and len(params) == 3:
            # CZ(qubit, ctrl, order)
            new_gate[name] = [
                qubit_map.get(params[0], params[0]),
                qubit_map.get(params[1], params[1]),
                params[2],
            ]
        elif name == "IDLE" and len(params) == 3:
            # IDLE(qubit, delay, order)
            new_gate[name] = [
                qubit_map.get(params[0], params[0]),
                params[1],
                params[2],
            ]
        elif name == "ECHO" and len(params) == 2:
            # ECHO(qubit, order)
            new_gate[name] = [
                qubit_map.get(params[0], params[0]),
                params[1],
            ]
        elif name == "Measure" and len(params) == 2:
            # Measure([qubits...], order)
            qubits = params[0]
            if isinstance(qubits, list):
                new_qubits = [qubit_map.get(q, q) for q in qubits]
            else:
                new_qubits = qubit_map.get(qubits, qubits)
            new_gate[name] = [new_qubits, params[1]]
        else:
            new_gate[name] = params

    return new_gate
