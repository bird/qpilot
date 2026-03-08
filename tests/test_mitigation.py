"""Tests for error mitigation — readout correction, ZNE, M3."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qpilot.mitigation.readout import (
    ReadoutMitigator,
    _bitstring_to_index,
    _get_bit,
)
from qpilot.mitigation.zne import ZNEMitigator, _invert_gate, _is_foldable, _set_order
from qpilot.mitigation.m3 import M3Mitigator


# ============================================================
# Readout Mitigation
# ============================================================


class TestBitstringUtils:
    def test_hex_to_index(self):
        assert _bitstring_to_index("0x0", 2) == 0
        assert _bitstring_to_index("0x1", 2) == 1
        assert _bitstring_to_index("0x2", 2) == 2
        assert _bitstring_to_index("0x3", 2) == 3

    def test_binary_to_index(self):
        assert _bitstring_to_index("0b00", 2) == 0
        assert _bitstring_to_index("0b11", 2) == 3

    def test_get_bit(self):
        # For 2 qubits, "0x2" = binary "10": bit 0 (MSB) = 1, bit 1 = 0
        assert _get_bit("0x2", 0, 2) == 1
        assert _get_bit("0x2", 1, 2) == 0
        # "0x3" = "11"
        assert _get_bit("0x3", 0, 2) == 1
        assert _get_bit("0x3", 1, 2) == 1


class TestReadoutMitigatorSingleQubit:
    """Test readout mitigation with a single qubit and known noise."""

    def test_perfect_readout(self):
        """Perfect readout should return counts unchanged."""
        mit = ReadoutMitigator.from_fidelity_params([(1.0, 1.0)], ["q0"])
        raw = {"0x0": 600, "0x1": 400}
        corrected = mit.correct(raw)
        assert corrected["0x0"] == pytest.approx(0.6, abs=1e-6)
        assert corrected["0x1"] == pytest.approx(0.4, abs=1e-6)

    def test_noisy_readout_correction(self):
        """Apply known noise, then verify correction recovers true distribution."""
        # True distribution: P(0)=0.8, P(1)=0.2
        p_true = np.array([0.8, 0.2])

        # Fidelity: P(0|0)=0.95, P(1|1)=0.90
        A = np.array([[0.95, 0.10], [0.05, 0.90]])
        p_obs = A @ p_true  # [0.78, 0.22]

        # Simulate observed counts (10000 shots)
        raw = {"0x0": int(p_obs[0] * 10000), "0x1": int(p_obs[1] * 10000)}

        mit = ReadoutMitigator.from_fidelity_params([(0.95, 0.90)], ["q0"])
        corrected = mit.correct(raw)

        assert corrected.get("0x0", 0) == pytest.approx(0.8, abs=0.02)
        assert corrected.get("0x1", 0) == pytest.approx(0.2, abs=0.02)

    def test_tensored_single_qubit(self):
        """Tensored correction on single qubit should match full matrix."""
        p_true = np.array([0.7, 0.3])
        A = np.array([[0.92, 0.08], [0.08, 0.92]])
        p_obs = A @ p_true

        raw = {"0x0": int(p_obs[0] * 10000), "0x1": int(p_obs[1] * 10000)}
        mit = ReadoutMitigator.from_fidelity_params([(0.92, 0.92)], ["q0"])

        full = mit.correct(raw)
        tensored = mit.correct_tensored(raw)

        assert full.get("0x0", 0) == pytest.approx(tensored.get("0x0", 0), abs=0.02)
        assert full.get("0x1", 0) == pytest.approx(tensored.get("0x1", 0), abs=0.02)


class TestReadoutMitigatorTwoQubit:
    """Test readout mitigation with 2 qubits."""

    def _make_noisy_counts(self, p_true, fidelity_params):
        """Generate noisy counts from true distribution and fidelity params."""
        matrices = []
        for p00, p11 in fidelity_params:
            matrices.append(np.array([[p00, 1 - p11], [1 - p00, p11]]))
        A = matrices[0]
        for m in matrices[1:]:
            A = np.kron(A, m)
        p_obs = A @ p_true
        total = 10000
        counts = {}
        for i, p in enumerate(p_obs):
            if p > 0.001:
                counts[hex(i)] = int(round(p * total))
        return counts

    def test_two_qubit_correction(self):
        """Bell state |00⟩+|11⟩ → P(00)=P(11)=0.5"""
        p_true = np.array([0.5, 0.0, 0.0, 0.5])
        fidelity = [(0.95, 0.90), (0.93, 0.88)]
        raw = self._make_noisy_counts(p_true, fidelity)

        mit = ReadoutMitigator.from_fidelity_params(fidelity, ["q0", "q1"])
        corrected = mit.correct(raw)

        assert corrected.get("0x0", 0) == pytest.approx(0.5, abs=0.05)
        assert corrected.get("0x3", 0) == pytest.approx(0.5, abs=0.05)
        # Middle states should be near 0
        assert corrected.get("0x1", 0) < 0.05
        assert corrected.get("0x2", 0) < 0.05

    def test_tensored_two_qubit(self):
        """Tensored correction should give reasonable results for 2 qubits."""
        p_true = np.array([0.5, 0.0, 0.0, 0.5])
        fidelity = [(0.95, 0.90), (0.93, 0.88)]
        raw = self._make_noisy_counts(p_true, fidelity)

        mit = ReadoutMitigator.from_fidelity_params(fidelity, ["q0", "q1"])
        corrected = mit.correct_tensored(raw)

        # Tensored is less accurate but should still be reasonable
        assert corrected.get("0x0", 0) > 0.3
        assert corrected.get("0x3", 0) > 0.3

    def test_missing_bitstrings(self):
        """Counts with missing bitstrings should work fine."""
        mit = ReadoutMitigator.from_fidelity_params([(0.95, 0.90)], ["q0"])
        raw = {"0x0": 1000}  # Only one bitstring
        corrected = mit.correct(raw)
        assert sum(corrected.values()) == pytest.approx(1.0, abs=1e-6)

    def test_non_negativity(self):
        """Corrected probabilities should be non-negative."""
        mit = ReadoutMitigator.from_fidelity_params([(0.8, 0.7), (0.75, 0.65)], ["q0", "q1"])
        raw = {"0x0": 100, "0x1": 50, "0x2": 30, "0x3": 820}
        corrected = mit.correct(raw)
        for v in corrected.values():
            assert v >= 0
        assert sum(corrected.values()) == pytest.approx(1.0, abs=1e-6)


class TestReadoutMitigatorFromCalibration:
    def test_from_calibration_results(self):
        """Build mitigator from characterization experiment results."""
        # Single qubit: prepare |0>, mostly get 0; prepare |1>, mostly get 1
        results_0 = {"0x0": 950, "0x1": 50}
        results_1 = {"0x0": 80, "0x1": 920}
        mit = ReadoutMitigator.from_calibration_results(results_0, results_1, ["q0"])
        assert mit.num_qubits == 1
        # Should be able to correct
        corrected = mit.correct({"0x0": 600, "0x1": 400})
        assert sum(corrected.values()) == pytest.approx(1.0, abs=1e-6)


# ============================================================
# Zero-Noise Extrapolation
# ============================================================


class TestZNEGateFolding:
    def test_scale_1_is_identity(self):
        """Scale factor 1 should return the original circuit unchanged."""
        zne = ZNEMitigator()
        circuit = [[{"RPhi": [0, 90.0, 180.0, 0]}, {"Measure": [[0], 1]}]]
        scaled = zne.generate_scaled_circuits(circuit, [1.0])
        assert len(scaled) == 1
        # Should be a deep copy, structurally equal
        assert scaled[0][0][0]["RPhi"][:3] == [0, 90.0, 180.0]

    def test_scale_3_triples_gates(self):
        """Scale factor 3: G → G G† G. Single gate becomes 3 gates."""
        zne = ZNEMitigator()
        circuit = [[{"RPhi": [0, 90.0, 180.0, 0]}, {"Measure": [[0], 5]}]]
        scaled = zne.generate_scaled_circuits(circuit, [3.0])
        instrs = scaled[0][0]

        # Should have: RPhi, RPhi†, RPhi, Measure = 4 instructions
        rphi_gates = [g for g in instrs if "RPhi" in g]
        measures = [g for g in instrs if "Measure" in g]
        assert len(rphi_gates) == 3
        assert len(measures) == 1

        # First gate: original (angle=180)
        assert rphi_gates[0]["RPhi"][2] == 180.0
        # Second gate: inverse (angle=360-180=180 for X gate, which is self-inverse)
        assert rphi_gates[1]["RPhi"][2] == 180.0
        # Third gate: original again
        assert rphi_gates[2]["RPhi"][2] == 180.0

    def test_scale_3_non_self_inverse(self):
        """Scale factor 3 with a non-self-inverse gate (S gate, angle=90)."""
        zne = ZNEMitigator()
        circuit = [[{"RPhi": [0, 90.0, 90.0, 0]}, {"Measure": [[0], 1]}]]
        scaled = zne.generate_scaled_circuits(circuit, [3.0])
        instrs = scaled[0][0]
        rphi_gates = [g for g in instrs if "RPhi" in g]
        assert len(rphi_gates) == 3
        # Original: angle=90
        assert rphi_gates[0]["RPhi"][2] == 90.0
        # Inverse: angle=270 (360-90)
        assert rphi_gates[1]["RPhi"][2] == 270.0
        # Original again: angle=90
        assert rphi_gates[2]["RPhi"][2] == 90.0

    def test_cz_folding(self):
        """CZ is self-inverse, so CZ† = CZ."""
        zne = ZNEMitigator()
        circuit = [[{"CZ": [0, 1, 0]}, {"Measure": [[0, 1], 1]}]]
        scaled = zne.generate_scaled_circuits(circuit, [3.0])
        instrs = scaled[0][0]
        cz_gates = [g for g in instrs if "CZ" in g]
        assert len(cz_gates) == 3

    def test_idle_not_folded(self):
        """IDLE gates should not be folded."""
        zne = ZNEMitigator()
        circuit = [[{"IDLE": [0, 100, 0]}, {"RPhi": [0, 0, 180, 1]}, {"Measure": [[0], 2]}]]
        scaled = zne.generate_scaled_circuits(circuit, [3.0])
        instrs = scaled[0][0]
        idle_gates = [g for g in instrs if "IDLE" in g]
        rphi_gates = [g for g in instrs if "RPhi" in g]
        assert len(idle_gates) == 1  # not folded
        assert len(rphi_gates) == 3  # folded

    def test_measure_not_folded(self):
        """Measure should not be folded."""
        assert not _is_foldable({"Measure": [[0], 0]})
        assert not _is_foldable({"IDLE": [0, 100, 0]})
        assert _is_foldable({"RPhi": [0, 0, 180, 0]})
        assert _is_foldable({"CZ": [0, 1, 0]})
        assert _is_foldable({"ECHO": [0, 0]})

    def test_order_values_sequential(self):
        """Order values in folded circuit should be sequential."""
        zne = ZNEMitigator()
        circuit = [
            [
                {"RPhi": [0, 0, 90, 0]},
                {"RPhi": [0, 90, 180, 1]},
                {"Measure": [[0], 2]},
            ]
        ]
        scaled = zne.generate_scaled_circuits(circuit, [3.0])
        instrs = scaled[0][0]
        orders = []
        for g in instrs:
            name = next(iter(g))
            params = g[name]
            if name == "RPhi":
                orders.append(params[3])
            elif name == "Measure":
                orders.append(params[1])
        # Should be monotonically increasing
        for i in range(1, len(orders)):
            assert orders[i] > orders[i - 1]

    def test_multiple_scale_factors(self):
        """Generate circuits at multiple scale factors."""
        zne = ZNEMitigator()
        circuit = [[{"RPhi": [0, 0, 180, 0]}, {"Measure": [[0], 1]}]]
        scaled = zne.generate_scaled_circuits(circuit, [1.0, 3.0, 5.0])
        assert len(scaled) == 3


class TestZNEExtrapolation:
    def test_linear_extrapolation(self):
        """Linear: y = 1.0 - 0.1*x, expect y(0) = 1.0"""
        zne = ZNEMitigator()
        scales = [1.0, 3.0, 5.0]
        values = [0.9, 0.7, 0.5]
        result = zne.extrapolate(values, scales, method="linear")
        assert result == pytest.approx(1.0, abs=0.01)

    def test_polynomial_extrapolation(self):
        """Polynomial through 3 points."""
        zne = ZNEMitigator()
        scales = [1.0, 2.0, 3.0]
        # y = 1.0 - 0.05*x^2: y(1)=0.95, y(2)=0.80, y(3)=0.55
        values = [0.95, 0.80, 0.55]
        result = zne.extrapolate(values, scales, method="polynomial")
        assert result == pytest.approx(1.0, abs=0.01)

    def test_exponential_extrapolation(self):
        """Exponential: y = exp(-0.1*x), expect y(0) = 1.0"""
        zne = ZNEMitigator()
        scales = [1.0, 3.0, 5.0]
        values = [math.exp(-0.1), math.exp(-0.3), math.exp(-0.5)]
        result = zne.extrapolate(values, scales, method="exponential")
        assert result == pytest.approx(1.0, abs=0.01)

    def test_insufficient_points_raises(self):
        zne = ZNEMitigator()
        with pytest.raises(ValueError):
            zne.extrapolate([0.5], [1.0], method="linear")

    def test_unknown_method_raises(self):
        zne = ZNEMitigator()
        with pytest.raises(ValueError, match="Unknown"):
            zne.extrapolate([0.5, 0.3], [1.0, 3.0], method="magic")


class TestZNEMitigate:
    async def test_full_pipeline(self):
        """Test the full mitigate pipeline with a mock executor."""
        zne = ZNEMitigator()

        # Mock executor: returns 1.0 - 0.1 * scale_factor
        call_count = 0

        async def mock_executor(circuit):
            nonlocal call_count
            # Count gates to determine noise level
            n_gates = sum(1 for g in circuit[0] if "RPhi" in g)
            call_count += 1
            # Original has 1 gate, scale 3 has 3, scale 5 has 5
            return 1.0 - 0.1 * n_gates

        circuit = [[{"RPhi": [0, 0, 180, 0]}, {"Measure": [[0], 1]}]]
        result = await zne.mitigate(circuit, mock_executor, [1.0, 3.0, 5.0])
        assert call_count == 3
        # Should extrapolate to ~1.0 (zero gates = zero noise)
        assert result == pytest.approx(1.0, abs=0.15)


class TestGateInversion:
    def test_rphi_inversion(self):
        gate = {"RPhi": [0, 90.0, 90.0, 5]}
        inv = _invert_gate(gate)
        assert inv["RPhi"][2] == 270.0  # 360 - 90

    def test_rphi_180_self_inverse(self):
        gate = {"RPhi": [0, 0.0, 180.0, 0]}
        inv = _invert_gate(gate)
        assert inv["RPhi"][2] == 180.0  # 360 - 180 = 180

    def test_cz_self_inverse(self):
        gate = {"CZ": [0, 1, 0]}
        inv = _invert_gate(gate)
        assert inv["CZ"] == [0, 1, 0]

    def test_echo_self_inverse(self):
        gate = {"ECHO": [0, 0]}
        inv = _invert_gate(gate)
        assert inv["ECHO"] == [0, 0]

    def test_set_order(self):
        gate = {"RPhi": [0, 90.0, 180.0, 0]}
        updated = _set_order(gate, 42)
        assert updated["RPhi"][3] == 42
        # Original unchanged
        assert gate["RPhi"][3] == 0


# ============================================================
# M3 Mitigation
# ============================================================


class TestM3Mitigator:
    def test_perfect_readout(self):
        """Perfect fidelity should return normalized counts."""
        m3 = M3Mitigator.from_fidelity_params({"q0": (1.0, 1.0)})
        raw = {"0x0": 700, "0x1": 300}
        corrected = m3.correct(raw)
        assert corrected.get("0x0", 0) == pytest.approx(0.7, abs=0.01)
        assert corrected.get("0x1", 0) == pytest.approx(0.3, abs=0.01)

    def test_noisy_correction(self):
        """M3 should recover true distribution from noisy counts."""
        p_true = np.array([0.8, 0.2])
        A = np.array([[0.95, 0.10], [0.05, 0.90]])
        p_obs = A @ p_true

        raw = {"0x0": int(p_obs[0] * 10000), "0x1": int(p_obs[1] * 10000)}
        m3 = M3Mitigator.from_fidelity_params({"q0": (0.95, 0.90)})
        corrected = m3.correct(raw)

        assert corrected.get("0x0", 0) == pytest.approx(0.8, abs=0.05)
        assert corrected.get("0x1", 0) == pytest.approx(0.2, abs=0.05)

    def test_two_qubit_m3(self):
        """Two-qubit M3 correction."""
        # Bell state: P(00) = P(11) = 0.5
        m3 = M3Mitigator(
            fidelity_matrices={
                "q0": [[0.95, 0.10], [0.05, 0.90]],
                "q1": [[0.93, 0.12], [0.07, 0.88]],
            }
        )

        # Generate noisy counts
        p_true = np.array([0.5, 0.0, 0.0, 0.5])
        A0 = np.array([[0.95, 0.10], [0.05, 0.90]])
        A1 = np.array([[0.93, 0.12], [0.07, 0.88]])
        A_full = np.kron(A0, A1)
        p_obs = A_full @ p_true

        raw = {}
        for i, p in enumerate(p_obs):
            if p > 0.001:
                raw[hex(i)] = int(round(p * 10000))

        corrected = m3.correct(raw)
        assert corrected.get("0x0", 0) == pytest.approx(0.5, abs=0.1)
        assert corrected.get("0x3", 0) == pytest.approx(0.5, abs=0.1)

    def test_non_negativity(self):
        """Corrected probabilities should be non-negative."""
        m3 = M3Mitigator.from_fidelity_params({"q0": (0.8, 0.7)})
        raw = {"0x0": 100, "0x1": 900}
        corrected = m3.correct(raw)
        for v in corrected.values():
            assert v >= 0

    def test_empty_counts(self):
        m3 = M3Mitigator.from_fidelity_params({"q0": (0.95, 0.90)})
        assert m3.correct({}) == {}

    def test_agrees_with_full_readout_on_small_case(self):
        """M3 and full ReadoutMitigator should give similar results for small cases."""
        p00, p11 = 0.95, 0.90
        p_true = np.array([0.6, 0.4])
        A = np.array([[p00, 1 - p11], [1 - p00, p11]])
        p_obs = A @ p_true
        raw = {"0x0": int(p_obs[0] * 10000), "0x1": int(p_obs[1] * 10000)}

        readout_mit = ReadoutMitigator.from_fidelity_params([(p00, p11)], ["q0"])
        m3_mit = M3Mitigator.from_fidelity_params({"q0": (p00, p11)})

        r_corrected = readout_mit.correct(raw)
        m3_corrected = m3_mit.correct(raw)

        assert r_corrected.get("0x0", 0) == pytest.approx(m3_corrected.get("0x0", 0), abs=0.05)
        assert r_corrected.get("0x1", 0) == pytest.approx(m3_corrected.get("0x1", 0), abs=0.05)
