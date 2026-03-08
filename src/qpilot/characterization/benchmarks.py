"""Benchmark circuit generators for the superconducting native instruction set.

Generates circuits as JSON instruction arrays suitable for direct ZMQ MsgTask
submission. Native gate set:

    RPhi(qubit, axis_deg, angle_deg, order)  — single-qubit rotation
    ECHO(qubit, order)                        — echo refocusing pulse
    IDLE(qubit, delay, order)                 — idle with variable delay
    CZ(qubit, ctrl, order)                    — controlled-Z
    Measure([qubits...], order)               — measurement

Clifford decompositions into RPhi (axis_deg, angle_deg):
    I  = identity (no gate)
    X  = RPhi(q, 0, 180, t)
    Y  = RPhi(q, 90, 180, t)
    Z  = RPhi(q, 0, 180, t) RPhi(q, 90, 180, t+1)  [X then Y = iZ, global phase irrelevant]
    H  = RPhi(q, 0, 90, t) RPhi(q, 90, 180, t+1)   [Ry(pi) Rx(pi/2)]
    S  = RPhi(q, 90, 90, t)
    Sd = RPhi(q, 90, 270, t)                         [S-dagger]

The 24-element single-qubit Clifford group is generated from compositions of
H and S (since <H, S> generates the Cliffords).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


# --- Native instruction builders ---


def _rphi(qubit: int, axis_deg: float, angle_deg: float, order: int) -> dict[str, list]:
    return {"RPhi": [qubit, axis_deg, angle_deg, order]}


def _idle(qubit: int, delay: int, order: int) -> dict[str, list]:
    return {"IDLE": [qubit, delay, order]}


def _echo(qubit: int, order: int) -> dict[str, list]:
    return {"ECHO": [qubit, order]}


def _cz(qubit: int, ctrl: int, order: int) -> dict[str, list]:
    return {"CZ": [qubit, ctrl, order]}


def _measure(qubits: list[int], order: int) -> dict[str, list]:
    return {"Measure": [qubits, order]}


# --- Single-qubit Clifford group ---
#
# Each Clifford is represented as a sequence of (axis_deg, angle_deg) pairs
# for RPhi gates. We enumerate all 24 elements via the generators H and S.

# Primitive gate sequences (axis_deg, angle_deg)
_I: list[tuple[float, float]] = []
_X: list[tuple[float, float]] = [(0, 180)]
_Y: list[tuple[float, float]] = [(90, 180)]
_Z: list[tuple[float, float]] = [(0, 180), (90, 180)]
_H: list[tuple[float, float]] = [(0, 90), (90, 180)]
_S: list[tuple[float, float]] = [(90, 90)]
_SD: list[tuple[float, float]] = [(90, 270)]

# The 24 single-qubit Cliffords, each as a list of (axis, angle) RPhi params.
# Generated via: {I, X, Y, Z} x {I, H, S, Sd, HS, HSd, SH, SdH, HSH, HsdH, ...}
# We enumerate them explicitly for correctness.
SINGLE_QUBIT_CLIFFORDS: list[list[tuple[float, float]]] = [
    # --- Identity class (4) ---
    _I,  # C0:  I
    _X,  # C1:  X
    _Y,  # C2:  Y
    _Z,  # C3:  Z (= XY up to phase)
    # --- H class (4) ---
    _H,  # C4:  H
    _H + _X,  # C5:  HX
    _H + _Y,  # C6:  HY
    _H + _Z,  # C7:  HZ
    # --- S class (4) ---
    _S,  # C8:  S
    _S + _X,  # C9:  SX
    _SD,  # C10: Sd
    _SD + _X,  # C11: SdX
    # --- HS class (4) ---
    _H + _S,  # C12: HS
    _H + _S + _X,  # C13: HSX
    _H + _SD,  # C14: HSd
    _H + _SD + _X,  # C15: HSdX
    # --- SH class (4) ---
    _S + _H,  # C16: SH
    _S + _H + _X,  # C17: SHX
    _SD + _H,  # C18: SdH
    _SD + _H + _X,  # C19: SdHX
    # --- HSH class (4) ---
    _H + _S + _H,  # C20: HSH
    _H + _S + _H + _X,  # C21: HSHX
    _H + _SD + _H,  # C22: HSdH
    _H + _SD + _H + _X,  # C23: HSdHX
]


def _compile_clifford(
    qubit: int, gates: list[tuple[float, float]], start_order: int
) -> tuple[list[dict], int]:
    """Compile a Clifford gate sequence into native RPhi instructions.

    Returns (instructions, next_order).
    """
    instrs = []
    t = start_order
    for axis_deg, angle_deg in gates:
        instrs.append(_rphi(qubit, axis_deg, angle_deg, t))
        t += 1
    return instrs, t


@dataclass
class BenchmarkCircuit:
    """A single benchmark circuit ready for ZMQ submission."""

    name: str
    qubits: list[int]
    instructions: list[list[dict[str, Any]]]
    metadata: dict[str, Any] = field(default_factory=dict)


class BenchmarkCircuits:
    """Collection of benchmark circuits from a generation run."""

    def __init__(self, circuits: list[BenchmarkCircuit] | None = None) -> None:
        self.circuits = circuits or []

    def __len__(self) -> int:
        return len(self.circuits)

    def __iter__(self):
        return iter(self.circuits)

    def by_name(self, prefix: str) -> list[BenchmarkCircuit]:
        """Filter circuits whose name starts with prefix."""
        return [c for c in self.circuits if c.name.startswith(prefix)]


# --- Public circuit generators ---


def single_qubit_rb(
    qubit: int,
    depths: list[int] | None = None,
    num_circuits: int = 10,
    seed: int | None = None,
) -> BenchmarkCircuits:
    """Generate single-qubit randomized benchmarking circuits.

    For each depth, generates `num_circuits` random Clifford sequences with
    an inversion gate at the end, so the ideal outcome is always |0>.

    Args:
        qubit: Physical qubit index.
        depths: List of circuit depths (number of Clifford gates).
        num_circuits: Number of random circuits per depth.
        seed: Random seed for reproducibility.

    Returns:
        BenchmarkCircuits containing all generated circuits.
    """
    if depths is None:
        depths = [1, 2, 5, 10, 20, 50, 100]
    rng = random.Random(seed)
    circuits = []

    for depth in depths:
        for seq_idx in range(num_circuits):
            instrs: list[dict] = []
            t = 0

            # Random Clifford sequence
            clifford_indices = [rng.randrange(24) for _ in range(depth)]
            for ci in clifford_indices:
                gates = SINGLE_QUBIT_CLIFFORDS[ci]
                compiled, t = _compile_clifford(qubit, gates, t)
                instrs.extend(compiled)

            # Compute inversion Clifford: we need the inverse of the composed sequence.
            # For RB the inversion gate restores |0>. We append a full inverse sequence.
            # Since we're working with Cliffords, the inverse of each element exists in
            # the group. We reverse and invert each Clifford.
            for ci in reversed(clifford_indices):
                inv_gates = _invert_clifford(SINGLE_QUBIT_CLIFFORDS[ci])
                compiled, t = _compile_clifford(qubit, inv_gates, t)
                instrs.extend(compiled)

            # Measurement
            instrs.append(_measure([qubit], t))

            circuits.append(
                BenchmarkCircuit(
                    name=f"rb1q_d{depth}_s{seq_idx}",
                    qubits=[qubit],
                    instructions=[instrs],
                    metadata={"type": "single_qubit_rb", "depth": depth, "sequence": seq_idx},
                )
            )

    return BenchmarkCircuits(circuits)


def _invert_clifford(gates: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Invert a Clifford gate sequence.

    For RPhi(axis, angle), the inverse is RPhi(axis, 360 - angle).
    The sequence is reversed and each gate individually inverted.
    """
    return [(axis, (360 - angle) % 360) for axis, angle in reversed(gates)]


def two_qubit_rb(
    qubit: int,
    ctrl: int,
    depths: list[int] | None = None,
    num_circuits: int = 10,
    seed: int | None = None,
) -> BenchmarkCircuits:
    """Generate two-qubit randomized benchmarking circuits.

    Uses interleaved random single-qubit Cliffords with CZ gates to benchmark
    the two-qubit gate quality. Each layer is: random_1q(q0), random_1q(q1), CZ.

    Args:
        qubit: First physical qubit.
        ctrl: Second physical qubit (CZ partner).
        depths: Number of CZ layers.
        num_circuits: Number of random circuits per depth.
        seed: Random seed.
    """
    if depths is None:
        depths = [1, 2, 5, 10, 20]
    rng = random.Random(seed)
    circuits = []

    for depth in depths:
        for seq_idx in range(num_circuits):
            instrs: list[dict] = []
            t = 0

            for _ in range(depth):
                # Random single-qubit Cliffords on both qubits
                ci0 = rng.randrange(24)
                ci1 = rng.randrange(24)
                compiled0, t = _compile_clifford(qubit, SINGLE_QUBIT_CLIFFORDS[ci0], t)
                compiled1, t = _compile_clifford(ctrl, SINGLE_QUBIT_CLIFFORDS[ci1], t)
                instrs.extend(compiled0)
                instrs.extend(compiled1)
                # CZ gate
                instrs.append(_cz(qubit, ctrl, t))
                t += 1

            # Measure both qubits
            instrs.append(_measure([qubit, ctrl], t))

            circuits.append(
                BenchmarkCircuit(
                    name=f"rb2q_d{depth}_s{seq_idx}",
                    qubits=[qubit, ctrl],
                    instructions=[instrs],
                    metadata={
                        "type": "two_qubit_rb",
                        "depth": depth,
                        "sequence": seq_idx,
                        "pair": f"{qubit}-{ctrl}",
                    },
                )
            )

    return BenchmarkCircuits(circuits)


def readout_characterization(
    qubits: list[int],
    shots: int = 4000,
) -> BenchmarkCircuits:
    """Generate readout error characterization circuits.

    For each qubit, generates two circuits:
    1. Prepare |0> and measure (no gates before measurement).
    2. Prepare |1> via X gate and measure.

    The deviation from ideal outcomes gives readout error rates:
    - P(1|0): fraction of 1s when preparing |0>
    - P(0|1): fraction of 0s when preparing |1>

    Args:
        qubits: List of physical qubit indices to characterize.
        shots: Suggested shot count (stored in metadata).
    """
    circuits = []

    for q in qubits:
        # Prepare |0>, measure
        circuits.append(
            BenchmarkCircuit(
                name=f"readout_prep0_q{q}",
                qubits=[q],
                instructions=[[_measure([q], 0)]],
                metadata={"type": "readout", "prep_state": 0, "qubit": q, "shots": shots},
            )
        )

        # Prepare |1> (X gate), measure
        circuits.append(
            BenchmarkCircuit(
                name=f"readout_prep1_q{q}",
                qubits=[q],
                instructions=[[_rphi(q, 0, 180, 0), _measure([q], 1)]],
                metadata={"type": "readout", "prep_state": 1, "qubit": q, "shots": shots},
            )
        )

    return BenchmarkCircuits(circuits)


def t1_circuits(
    qubit: int,
    delays: list[int] | None = None,
) -> BenchmarkCircuits:
    """Generate T1 relaxation measurement circuits.

    Prepare |1> via X gate, wait variable delay using IDLE gate, then measure.
    Fit the survival probability P(1) vs delay to an exponential decay to get T1.

    Args:
        qubit: Physical qubit index.
        delays: IDLE delay values in native time units. Defaults to a log-spaced range.
    """
    if delays is None:
        delays = [0, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    circuits = []

    for delay in delays:
        instrs: list[dict] = [_rphi(qubit, 0, 180, 0)]  # X gate: prepare |1>
        t = 1
        if delay > 0:
            instrs.append(_idle(qubit, delay, t))
            t += 1
        instrs.append(_measure([qubit], t))

        circuits.append(
            BenchmarkCircuit(
                name=f"t1_q{qubit}_d{delay}",
                qubits=[qubit],
                instructions=[instrs],
                metadata={"type": "t1", "qubit": qubit, "delay": delay},
            )
        )

    return BenchmarkCircuits(circuits)


def t2star_circuits(
    qubit: int,
    delays: list[int] | None = None,
) -> BenchmarkCircuits:
    """Generate T2* (Ramsey) measurement circuits.

    H gate, variable IDLE delay, H gate, measure. The decay envelope of the
    Ramsey fringes gives T2*.

    H is decomposed as: RPhi(q, 0, 90) then RPhi(q, 90, 180).

    Args:
        qubit: Physical qubit index.
        delays: IDLE delay values. Defaults to a log-spaced range.
    """
    if delays is None:
        delays = [0, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    circuits = []

    for delay in delays:
        instrs: list[dict] = []
        t = 0

        # First H gate
        instrs.append(_rphi(qubit, 0, 90, t))
        t += 1
        instrs.append(_rphi(qubit, 90, 180, t))
        t += 1

        # Variable delay
        if delay > 0:
            instrs.append(_idle(qubit, delay, t))
            t += 1

        # Second H gate
        instrs.append(_rphi(qubit, 0, 90, t))
        t += 1
        instrs.append(_rphi(qubit, 90, 180, t))
        t += 1

        instrs.append(_measure([qubit], t))

        circuits.append(
            BenchmarkCircuit(
                name=f"t2star_q{qubit}_d{delay}",
                qubits=[qubit],
                instructions=[instrs],
                metadata={"type": "t2star", "qubit": qubit, "delay": delay},
            )
        )

    return BenchmarkCircuits(circuits)
