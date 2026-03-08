"""Tests for the optimization module — qubit selection and layout optimization."""

from __future__ import annotations

import pytest

from qpilot.characterization.noise_profile import ChipNoiseProfile, NoiseProfile
from qpilot.optimization.layout_optimizer import LayoutOptimizer, _remap_gate
from qpilot.optimization.qubit_selector import QubitMapping, QubitSelector, ScoringMetric


def _make_chip_profile() -> ChipNoiseProfile:
    """Create a mock 6-qubit chip profile with a linear topology.

    Topology: 45 -- 46 -- 48 -- 52 -- 53 -- 54
    """
    profiles = {
        "45": NoiseProfile(
            qubit_id="45",
            single_gate_fidelity=0.99,
            readout_fidelity=(0.98, 0.97),
            two_gate_fidelities={"46": 0.95},
        ),
        "46": NoiseProfile(
            qubit_id="46",
            single_gate_fidelity=0.98,
            readout_fidelity=(0.97, 0.96),
            two_gate_fidelities={"45": 0.95, "48": 0.93},
        ),
        "48": NoiseProfile(
            qubit_id="48",
            single_gate_fidelity=0.97,
            readout_fidelity=(0.96, 0.95),
            two_gate_fidelities={"46": 0.93, "52": 0.91},
        ),
        "52": NoiseProfile(
            qubit_id="52",
            single_gate_fidelity=0.96,
            readout_fidelity=(0.95, 0.94),
            two_gate_fidelities={"48": 0.91, "53": 0.90},
        ),
        "53": NoiseProfile(
            qubit_id="53",
            single_gate_fidelity=0.95,
            readout_fidelity=(0.94, 0.93),
            two_gate_fidelities={"52": 0.90, "54": 0.89},
        ),
        "54": NoiseProfile(
            qubit_id="54",
            single_gate_fidelity=0.94,
            readout_fidelity=(0.93, 0.92),
            two_gate_fidelities={"53": 0.89},
        ),
    }
    topology = {
        "45": ["46"],
        "46": ["45", "48"],
        "48": ["46", "52"],
        "52": ["48", "53"],
        "53": ["52", "54"],
        "54": ["53"],
    }
    return ChipNoiseProfile(profiles=profiles, topology=topology)


class TestQubitSelectorUnconstrained:
    def test_select_best_n(self):
        profile = _make_chip_profile()
        selector = QubitSelector(profile)
        result = selector.select(num_qubits=2)
        assert len(result.mapping) == 2
        # Should select the 2 qubits with highest composite fidelity
        # which are "45" and "46" (highest single-gate + readout)
        assert "45" in result.physical_qubits
        assert "46" in result.physical_qubits

    def test_select_all(self):
        profile = _make_chip_profile()
        selector = QubitSelector(profile)
        result = selector.select(num_qubits=6)
        assert len(result.mapping) == 6

    def test_select_one(self):
        profile = _make_chip_profile()
        selector = QubitSelector(profile)
        result = selector.select(num_qubits=1)
        assert len(result.mapping) == 1
        assert "45" in result.physical_qubits  # highest fidelity

    def test_too_many_qubits_raises(self):
        profile = _make_chip_profile()
        selector = QubitSelector(profile)
        with pytest.raises(ValueError, match="chip only has"):
            selector.select(num_qubits=100)

    def test_zero_qubits_raises(self):
        profile = _make_chip_profile()
        selector = QubitSelector(profile)
        with pytest.raises(ValueError, match="must be positive"):
            selector.select(num_qubits=0)

    def test_single_gate_metric(self):
        profile = _make_chip_profile()
        selector = QubitSelector(profile)
        result = selector.select(num_qubits=2, metric=ScoringMetric.SINGLE_GATE)
        assert "45" in result.physical_qubits  # 0.99


class TestQubitSelectorConstrained:
    def test_two_qubit_circuit(self):
        """Select 2 connected qubits for a circuit with a CNOT."""
        profile = _make_chip_profile()
        selector = QubitSelector(profile)
        result = selector.select(
            num_qubits=2,
            connectivity=[(0, 1)],
        )
        # The pair must be physically connected
        q0 = result.physical(0)
        q1 = result.physical(1)
        assert q1 in profile.topology[q0]

    def test_three_qubit_linear_circuit(self):
        """Select 3 connected qubits for a linear circuit: 0-1-2."""
        profile = _make_chip_profile()
        selector = QubitSelector(profile)
        result = selector.select(
            num_qubits=3,
            connectivity=[(0, 1), (1, 2)],
        )
        q0 = result.physical(0)
        q1 = result.physical(1)
        q2 = result.physical(2)
        assert q1 in profile.topology[q0]
        assert q2 in profile.topology[q1]

    def test_impossible_connectivity_raises(self):
        """Requesting connectivity that doesn't exist in topology."""
        profile = _make_chip_profile()
        selector = QubitSelector(profile)
        # Request fully-connected 4-qubit subgraph — linear topology can't provide it
        with pytest.raises(ValueError, match="No valid qubit mapping"):
            selector.select(
                num_qubits=4,
                connectivity=[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
            )

    def test_score_includes_pair_fidelity(self):
        """Best mapping should maximize combined qubit + pair fidelity."""
        profile = _make_chip_profile()
        selector = QubitSelector(profile)
        result = selector.select(
            num_qubits=2,
            connectivity=[(0, 1)],
        )
        assert result.score > 0


class TestQubitMapping:
    def test_repr(self):
        m = QubitMapping(mapping={0: "45", 1: "46"}, score=1.5, metric="composite_fidelity")
        assert "L0→45" in repr(m)
        assert "L1→46" in repr(m)

    def test_physical_qubits(self):
        m = QubitMapping(mapping={0: "45", 1: "46"})
        assert set(m.physical_qubits) == {"45", "46"}


class TestLayoutOptimizer:
    def test_remap_rphi(self):
        gate = {"RPhi": [0, 90.0, 180.0, 5]}
        remapped = _remap_gate(gate, {0: 45})
        assert remapped["RPhi"][0] == 45

    def test_remap_cz(self):
        gate = {"CZ": [0, 1, 10]}
        remapped = _remap_gate(gate, {0: 45, 1: 46})
        assert remapped["CZ"] == [45, 46, 10]

    def test_remap_measure(self):
        gate = {"Measure": [[0, 1], 20]}
        remapped = _remap_gate(gate, {0: 45, 1: 46})
        assert remapped["Measure"] == [[45, 46], 20]

    def test_remap_idle(self):
        gate = {"IDLE": [0, 500, 3]}
        remapped = _remap_gate(gate, {0: 45})
        assert remapped["IDLE"] == [45, 500, 3]

    def test_remap_echo(self):
        gate = {"ECHO": [0, 7]}
        remapped = _remap_gate(gate, {0: 45})
        assert remapped["ECHO"] == [45, 7]

    def test_optimize_full(self):
        profile = _make_chip_profile()
        optimizer = LayoutOptimizer(profile)
        mapping = QubitMapping(mapping={0: "45", 1: "46"})
        instructions = [
            [
                {"RPhi": [0, 90.0, 180.0, 0]},
                {"CZ": [0, 1, 1]},
                {"Measure": [[0, 1], 2]},
            ]
        ]
        result = optimizer.optimize(mapping, instructions, logical_qubits=[0, 1])
        instrs = result.instructions[0]
        assert instrs[0]["RPhi"][0] == 45
        assert instrs[1]["CZ"] == [45, 46, 1]
        assert instrs[2]["Measure"] == [[45, 46], 2]

    def test_unmapped_qubit_passthrough(self):
        """Qubits not in the mapping should pass through unchanged."""
        gate = {"RPhi": [99, 0.0, 90.0, 0]}
        remapped = _remap_gate(gate, {0: 45})
        assert remapped["RPhi"][0] == 99  # not remapped
