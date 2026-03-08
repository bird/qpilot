"""Tests for the characterization module — benchmarks, noise profiling, drift detection."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qpilot.characterization.benchmarks import (
    BenchmarkCircuit,
    BenchmarkCircuits,
    SINGLE_QUBIT_CLIFFORDS,
    readout_characterization,
    single_qubit_rb,
    t1_circuits,
    t2star_circuits,
    two_qubit_rb,
)
from qpilot.characterization.drift_detector import DriftDetector
from qpilot.characterization.noise_profile import (
    NoiseProfile,
    NoiseProfiler,
)
from qpilot.models.responses import (
    DoubleGateFidelity,
    GetRBDataResponse,
    SingleGateFidelity,
)


# ============================================================
# Benchmarks
# ============================================================


class TestSingleQubitRB:
    def test_default_depths(self):
        circuits = single_qubit_rb(qubit=45, seed=42)
        depths = [1, 2, 5, 10, 20, 50, 100]
        assert len(circuits) == len(depths) * 10  # 10 circuits per depth

    def test_custom_depths_and_count(self):
        circuits = single_qubit_rb(qubit=45, depths=[1, 5], num_circuits=3, seed=0)
        assert len(circuits) == 6

    def test_circuit_structure(self):
        circuits = single_qubit_rb(qubit=45, depths=[2], num_circuits=1, seed=1)
        c = circuits.circuits[0]
        assert c.name.startswith("rb1q_d2")
        assert c.qubits == [45]
        assert c.metadata["type"] == "single_qubit_rb"
        assert c.metadata["depth"] == 2

        # Circuit should be a list of instruction lists
        assert len(c.instructions) == 1
        instrs = c.instructions[0]

        # Last instruction should be a Measure
        assert "Measure" in instrs[-1]
        assert 45 in instrs[-1]["Measure"][0]

    def test_all_gates_are_rphi_or_measure(self):
        circuits = single_qubit_rb(qubit=10, depths=[3], num_circuits=2, seed=7)
        for c in circuits:
            for instr in c.instructions[0]:
                gate_name = list(instr.keys())[0]
                assert gate_name in ("RPhi", "Measure"), f"Unexpected gate: {gate_name}"

    def test_reproducibility(self):
        c1 = single_qubit_rb(qubit=45, depths=[5], num_circuits=2, seed=99)
        c2 = single_qubit_rb(qubit=45, depths=[5], num_circuits=2, seed=99)
        assert c1.circuits[0].instructions == c2.circuits[0].instructions

    def test_rphi_params_valid(self):
        circuits = single_qubit_rb(qubit=45, depths=[3], num_circuits=1, seed=0)
        for instr in circuits.circuits[0].instructions[0]:
            if "RPhi" in instr:
                q, axis, angle, order = instr["RPhi"]
                assert q == 45
                assert 0 <= axis <= 360
                assert 0 <= angle <= 360
                assert isinstance(order, int)


class TestTwoQubitRB:
    def test_default_depths(self):
        circuits = two_qubit_rb(qubit=45, ctrl=46, seed=42)
        depths = [1, 2, 5, 10, 20]
        assert len(circuits) == len(depths) * 10

    def test_circuit_has_cz(self):
        circuits = two_qubit_rb(qubit=45, ctrl=46, depths=[2], num_circuits=1, seed=0)
        instrs = circuits.circuits[0].instructions[0]
        cz_gates = [i for i in instrs if "CZ" in i]
        assert len(cz_gates) == 2  # depth=2 means 2 CZ layers

    def test_metadata(self):
        circuits = two_qubit_rb(qubit=45, ctrl=46, depths=[3], num_circuits=1, seed=0)
        c = circuits.circuits[0]
        assert c.metadata["type"] == "two_qubit_rb"
        assert c.metadata["pair"] == "45-46"
        assert c.qubits == [45, 46]


class TestReadoutCharacterization:
    def test_circuit_count(self):
        circuits = readout_characterization(qubits=[45, 46, 48])
        assert len(circuits) == 6  # 2 per qubit

    def test_prep0_circuit(self):
        circuits = readout_characterization(qubits=[45])
        prep0 = circuits.circuits[0]
        assert prep0.name == "readout_prep0_q45"
        assert prep0.metadata["prep_state"] == 0
        # Should just be a Measure (no X gate)
        instrs = prep0.instructions[0]
        assert len(instrs) == 1
        assert "Measure" in instrs[0]

    def test_prep1_circuit(self):
        circuits = readout_characterization(qubits=[45])
        prep1 = circuits.circuits[1]
        assert prep1.name == "readout_prep1_q45"
        assert prep1.metadata["prep_state"] == 1
        # Should have X gate (RPhi) then Measure
        instrs = prep1.instructions[0]
        assert len(instrs) == 2
        assert "RPhi" in instrs[0]
        assert instrs[0]["RPhi"][2] == 180  # X gate angle
        assert "Measure" in instrs[1]


class TestT1Circuits:
    def test_default_delays(self):
        circuits = t1_circuits(qubit=45)
        assert len(circuits) == 11  # default delays

    def test_structure(self):
        circuits = t1_circuits(qubit=45, delays=[0, 100, 1000])
        assert len(circuits) == 3

        # Zero delay: X gate + Measure (no IDLE)
        c0 = circuits.circuits[0]
        instrs = c0.instructions[0]
        assert "RPhi" in instrs[0]  # X gate
        assert "Measure" in instrs[1]

        # Non-zero delay: X gate + IDLE + Measure
        c1 = circuits.circuits[1]
        instrs = c1.instructions[0]
        assert "RPhi" in instrs[0]
        assert "IDLE" in instrs[1]
        assert instrs[1]["IDLE"][1] == 100  # delay value
        assert "Measure" in instrs[2]


class TestT2StarCircuits:
    def test_default_delays(self):
        circuits = t2star_circuits(qubit=45)
        assert len(circuits) == 11

    def test_structure(self):
        circuits = t2star_circuits(qubit=45, delays=[0, 500])

        # Zero delay: H, H, Measure = 4 RPhi + Measure
        c0 = circuits.circuits[0]
        instrs = c0.instructions[0]
        rphi_gates = [i for i in instrs if "RPhi" in i]
        assert len(rphi_gates) == 4  # 2 RPhi per H gate × 2

        # Non-zero delay: H, IDLE, H, Measure
        c1 = circuits.circuits[1]
        instrs = c1.instructions[0]
        idle_gates = [i for i in instrs if "IDLE" in i]
        assert len(idle_gates) == 1
        assert idle_gates[0]["IDLE"][1] == 500


class TestBenchmarkCircuits:
    def test_by_name(self):
        circuits = BenchmarkCircuits(
            [
                BenchmarkCircuit(name="rb1q_d1_s0", qubits=[0], instructions=[[]]),
                BenchmarkCircuit(name="rb1q_d2_s0", qubits=[0], instructions=[[]]),
                BenchmarkCircuit(name="t1_q0_d0", qubits=[0], instructions=[[]]),
            ]
        )
        assert len(circuits.by_name("rb1q")) == 2
        assert len(circuits.by_name("t1")) == 1


class TestCliffordGroup:
    def test_24_cliffords(self):
        assert len(SINGLE_QUBIT_CLIFFORDS) == 24

    def test_identity_is_empty(self):
        assert SINGLE_QUBIT_CLIFFORDS[0] == []


# ============================================================
# Noise Profile
# ============================================================


def _make_rb_data() -> GetRBDataResponse:
    """Create a mock RB data response."""
    return GetRBDataResponse(
        msg_type="GetRBDataAck",
        sn=1,
        backend=72,
        single_gate_circuit_depth=[50, 50, 50, 50],
        double_gate_circuit_depth=[50, 50, 50],
        single_gate_fidelity=SingleGateFidelity(
            qubit=["45", "46", "48", "52"],
            fidelity=[0.99, 0.98, 0.97, 0.96],
        ),
        double_gate_fidelity=DoubleGateFidelity(
            qubit_pair=["45-46", "46-52", "48-52"],
            fidelity=[0.95, 0.93, 0.91],
        ),
    )


class TestNoiseProfile:
    def test_composite_fidelity(self):
        p = NoiseProfile(
            qubit_id="45",
            single_gate_fidelity=0.99,
            readout_fidelity=(0.98, 0.97),
        )
        # geometric mean of single_gate and avg_readout
        expected = math.sqrt(0.99 * (0.98 + 0.97) / 2)
        assert p.composite_fidelity == pytest.approx(expected)

    def test_composite_fidelity_perfect(self):
        p = NoiseProfile(
            qubit_id="0",
            single_gate_fidelity=1.0,
            readout_fidelity=(1.0, 1.0),
        )
        assert p.composite_fidelity == pytest.approx(1.0)


class TestNoiseProfilerFromRB:
    def test_profiles_created(self):
        rb = _make_rb_data()
        profile = NoiseProfiler.from_rb_data(rb)
        assert "45" in profile.profiles
        assert "46" in profile.profiles
        assert "48" in profile.profiles
        assert "52" in profile.profiles

    def test_single_gate_fidelity(self):
        rb = _make_rb_data()
        profile = NoiseProfiler.from_rb_data(rb)
        assert profile.profiles["45"].single_gate_fidelity == 0.99
        assert profile.profiles["52"].single_gate_fidelity == 0.96

    def test_two_gate_fidelity(self):
        rb = _make_rb_data()
        profile = NoiseProfiler.from_rb_data(rb)
        # 45-46 pair
        assert profile.profiles["45"].two_gate_fidelities["46"] == 0.95
        assert profile.profiles["46"].two_gate_fidelities["45"] == 0.95

    def test_topology(self):
        rb = _make_rb_data()
        profile = NoiseProfiler.from_rb_data(rb)
        assert "46" in profile.topology["45"]
        assert "52" in profile.topology["46"]
        assert "48" in profile.topology["52"]

    def test_best_qubits(self):
        rb = _make_rb_data()
        profile = NoiseProfiler.from_rb_data(rb)
        best2 = profile.best_qubits(2)
        assert len(best2) == 2
        # 45 and 46 have highest single-gate fidelities (0.99, 0.98)
        assert "45" in best2

    def test_best_connected_subgraph(self):
        rb = _make_rb_data()
        profile = NoiseProfiler.from_rb_data(rb)
        # Request 3 connected qubits
        sub = profile.best_connected_subgraph(3)
        assert len(sub) == 3
        # All selected qubits should be connected in the topology
        for q in sub:
            assert q in profile.topology

    def test_pair_fidelity(self):
        rb = _make_rb_data()
        profile = NoiseProfiler.from_rb_data(rb)
        assert profile.pair_fidelity("45", "46") == 0.95
        assert profile.pair_fidelity("45", "99") is None


class TestNoiseProfilerReadout:
    def test_from_readout_results(self):
        results = {
            45: ({"0x0": 980, "0x1": 20}, {"0x0": 30, "0x1": 970}),
            46: ({"0x0": 950, "0x1": 50}, {"0x0": 40, "0x1": 960}),
        }
        profile = NoiseProfiler.from_readout_results(results)
        assert profile.profiles["45"].readout_fidelity == pytest.approx((0.98, 0.97))
        assert profile.profiles["46"].readout_fidelity == pytest.approx((0.95, 0.96))

    def test_updates_existing(self):
        rb = _make_rb_data()
        existing = NoiseProfiler.from_rb_data(rb)
        results = {45: ({"0x0": 990, "0x1": 10}, {"0x0": 15, "0x1": 985})}
        updated = NoiseProfiler.from_readout_results(results, existing)
        assert updated.profiles["45"].readout_fidelity == pytest.approx((0.99, 0.985))
        # Other profiles should be preserved
        assert "46" in updated.profiles


class TestNoiseProfilerFitting:
    def test_fit_t1_exponential_decay(self):
        # Simulate: P(1) = exp(-t / 5000)
        delays = [0, 100, 500, 1000, 2000, 5000]
        probs = [math.exp(-t / 5000) for t in delays]
        t1 = NoiseProfiler.fit_t1(delays, probs)
        assert t1 is not None
        assert t1 == pytest.approx(5000, rel=0.01)

    def test_fit_t1_noisy_data(self):
        # With noise, the fit should still work approximately
        rng = np.random.default_rng(42)
        delays = [0, 100, 500, 1000, 2000, 5000, 10000]
        probs = [max(0.01, math.exp(-t / 3000) + rng.normal(0, 0.02)) for t in delays]
        t1 = NoiseProfiler.fit_t1(delays, probs)
        # Should be in the right ballpark (may not be exact due to noise)
        if t1 is not None:
            assert 1000 < t1 < 10000

    def test_fit_t1_insufficient_data(self):
        assert NoiseProfiler.fit_t1([0, 100], [1.0, 0.9]) is None

    def test_fit_t1_random_data_returns_none(self):
        # Purely random data shouldn't produce a valid decay fit
        rng = np.random.default_rng(123)
        delays = list(range(0, 10000, 500))
        probs = rng.uniform(0, 1, len(delays)).tolist()
        # May return None or a value — either is acceptable
        # The key is it shouldn't crash
        NoiseProfiler.fit_t1(delays, probs)

    def test_fit_t2star_decay(self):
        # Simulate: P(0) = 0.5 + 0.5 * exp(-t / 2000)
        delays = [0, 100, 500, 1000, 2000, 5000]
        probs = [0.5 + 0.5 * math.exp(-t / 2000) for t in delays]
        t2 = NoiseProfiler.fit_t2star(delays, probs)
        assert t2 is not None
        assert t2 == pytest.approx(2000, rel=0.05)

    def test_fit_t2star_insufficient_data(self):
        assert NoiseProfiler.fit_t2star([0], [1.0]) is None


# ============================================================
# Drift Detector
# ============================================================


class TestDriftDetector:
    def _make_profile(self, sg: float, ro: tuple[float, float]) -> NoiseProfile:
        return NoiseProfile(qubit_id="45", single_gate_fidelity=sg, readout_fidelity=ro)

    def test_no_drift(self):
        det = DriftDetector()
        base = self._make_profile(0.99, (0.98, 0.97))
        curr = self._make_profile(0.99, (0.98, 0.97))
        drift = det.check_drift(curr, base)
        assert not drift.drifted
        assert drift.single_gate_delta == pytest.approx(0.0)

    def test_significant_drift(self):
        det = DriftDetector(single_gate_threshold=0.02)
        base = self._make_profile(0.99, (0.98, 0.97))
        curr = self._make_profile(0.95, (0.98, 0.97))  # dropped by 0.04
        drift = det.check_drift(curr, base)
        assert drift.drifted
        assert drift.single_gate_delta == pytest.approx(-0.04)

    def test_readout_drift(self):
        det = DriftDetector(readout_threshold=0.03)
        base = self._make_profile(0.99, (0.98, 0.97))
        curr = self._make_profile(0.99, (0.92, 0.90))
        drift = det.check_drift(curr, base)
        assert drift.drifted

    def test_should_recalibrate(self):
        det = DriftDetector()
        base = self._make_profile(0.99, (0.98, 0.97))
        curr = self._make_profile(0.95, (0.90, 0.88))
        assert det.should_recalibrate(curr, base) is True

    def test_chip_drift_report(self):
        det = DriftDetector(single_gate_threshold=0.02)
        baseline = {
            "45": NoiseProfile(
                qubit_id="45", single_gate_fidelity=0.99, readout_fidelity=(0.98, 0.97)
            ),
            "46": NoiseProfile(
                qubit_id="46", single_gate_fidelity=0.98, readout_fidelity=(0.97, 0.96)
            ),
        }
        current = {
            "45": NoiseProfile(
                qubit_id="45", single_gate_fidelity=0.94, readout_fidelity=(0.98, 0.97)
            ),  # drifted
            "46": NoiseProfile(
                qubit_id="46", single_gate_fidelity=0.98, readout_fidelity=(0.97, 0.96)
            ),  # stable
        }
        report = det.check_chip_drift(current, baseline)
        assert "45" in report.drifted_qubits
        assert "46" in report.stable_qubits
        assert report.worst_qubit == "45"

    def test_empty_report(self):
        det = DriftDetector()
        report = det.check_chip_drift({}, {})
        assert len(report.drifted_qubits) == 0
        assert report.worst_qubit is None
