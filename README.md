# qpilot

Quantum hardware instrumentation for [PilotOS](https://qcloud.originqc.com.cn/en/programming/pilotos).

QPilot talks directly to the ZMQ control protocol exposed by PilotOS, giving
you low-level access to chip monitoring, noise characterization, qubit
selection, and error mitigation on Origin Quantum superconducting hardware.

It complements [pyqpanda3](https://pypi.org/project/pyqpanda3/) — use pyqpanda
for circuit construction, use qpilot for everything between your circuit and
the hardware.

## Install

```bash
pip install qpilot
```

For development:
```bash
git clone https://github.com/bird/qpilot.git
cd qpilot
pip install -e ".[dev]"
```

## Quick start

### Connect and monitor

```python
import asyncio
from qpilot import QPilotClient

async def main():
    async with QPilotClient(host="localhost") as client:
        config = await client.get_chip_config()
        rb = await client.get_rb_data()
        print(client.monitor.state)

asyncio.run(main())
```

### Build a noise profile

```python
from qpilot.characterization import NoiseProfiler

profile = NoiseProfiler.from_rb_data(rb_data, chip_config)
best_5 = profile.best_qubits(5)
subgraph = profile.best_connected_subgraph(4)
```

### Select optimal qubits

```python
from qpilot.optimization import QubitSelector

selector = QubitSelector(profile)
mapping = selector.select(num_qubits=2, connectivity=[(0, 1)])
print(mapping)  # QubitMapping(L0→45, L1→46, score=1.9312)
```

### Mitigate readout errors

```python
from qpilot.mitigation import ReadoutMitigator

mitigator = ReadoutMitigator.from_fidelity_params(
    [(0.97, 0.95), (0.98, 0.96)],
    qubit_ids=["0", "1"],
)
corrected = mitigator.correct({"0x0": 450, "0x1": 50, "0x2": 80, "0x3": 420})
```

### Zero-noise extrapolation

```python
from qpilot.mitigation import ZNEMitigator

zne = ZNEMitigator()
scaled_circuits = zne.generate_scaled_circuits(circuit, [1, 3, 5])
# run each on hardware, collect expectation values...
result = zne.extrapolate([0.85, 0.72, 0.61], [1, 3, 5], method="linear")
```

### Run automated experiments

```python
from qpilot.harness import Experiment, ExperimentCircuit, ExperimentRunner

class MyExperiment(Experiment):
    async def design(self, chip_state, noise_profile):
        return [ExperimentCircuit(circuit=my_circuit, shots=1000)]

    async def analyze(self, results):
        return {"fidelity": compute_fidelity(results)}

runner = ExperimentRunner(client)
result = await runner.run_once(MyExperiment())
```

## Architecture

```
QPilotClient
├── DealerClient          ZMQ DEALER — request/response (port 7000)
├── PubSubSubscriber      ZMQ SUB — real-time events (port 8000)
└── ChipMonitor           Live chip state from pub-sub events
    ├── QubitTracker       Rolling-window fidelity tracking
    └── EventLog           Calibration/maintenance event history

characterization/
├── benchmarks            RB, readout, T1, T2* circuit generators
├── noise_profile         Per-qubit and chip-level noise profiles
└── drift_detector        Fidelity drift detection between calibrations

optimization/
├── qubit_selector        Optimal qubit subset selection
└── layout_optimizer      Logical → physical qubit remapping

mitigation/
├── readout               Full-matrix and tensored readout correction
├── zne                   Gate folding + extrapolation
└── m3                    Matrix-free measurement mitigation

harness/
├── experiment            Experiment ABC + result types
├── runner                Design → submit → poll → collect → analyze
└── scheduler             Priority queue with calibration awareness
```

## Native gate set

QPilot targets the superconducting native instruction set:

| Gate | Format | Notes |
|------|--------|-------|
| RPhi | `{"RPhi": [qubit, axis_deg, angle_deg, order]}` | Single-qubit rotation |
| CZ | `{"CZ": [qubit, ctrl, order]}` | Controlled-Z |
| ECHO | `{"ECHO": [qubit, order]}` | Echo refocusing pulse |
| IDLE | `{"IDLE": [qubit, delay, order]}` | Variable delay |
| Measure | `{"Measure": [[qubits], order]}` | Measurement |

Circuits are JSON arrays of instruction dicts, submitted directly via ZMQ.

## Hardware backends

| Backend | DEALER port | PUB port |
|---------|-------------|----------|
| Superconducting | 7000 | 8000 |
| Trapped ion | 7001 | 8001 |
| Neutral atom | 7002 | 8002 |
| Photonic | 7003 | 8003 |

```python
from qpilot import QPilotClient, HardwareType
client = QPilotClient(host="10.0.0.1", hardware=HardwareType.SUPERCONDUCTING)
```

## Real hardware results

Tested on Origin Quantum's Wukong 72-qubit superconducting chip (2026-03-08).
QPilot fetched live calibration data, selected optimal qubits, ran 21 circuits
at 1000 shots each, and applied error mitigation. Full results in
[`data/wukong72/qpu_report_2026-03-08.md`](data/wukong72/qpu_report_2026-03-08.md).

### Qubit selection impact

QubitSelector picked Q56–Q62 as the best pair (CZ fidelity 0.9885,
composite fidelities 0.947 and 0.966). The worst pair on-chip, Q50–Q51
(CZ fidelity 0.9342), served as the control.

| | Q50–Q51 (worst) | Q56–Q62 (best) |
|--|-----------------|----------------|
| Raw Bell fidelity | 0.599 | 0.852 |
| After readout mitigation | 0.864 | 0.954 |

Qubit selection alone accounts for +0.253 in raw Bell fidelity.

### Readout error mitigation

ReadoutMitigator built calibration matrices from 8 prep-and-measure circuits,
then applied matrix-inverse correction to the Bell state counts.

| Pair | Raw | Mitigated | Improvement |
|------|-----|-----------|-------------|
| Q56–Q62 (best) | 0.852 | 0.954 | +10.2% |
| Q50–Q51 (worst) | 0.599 | 0.864 | +26.5% |

The worst pair benefits more because Q50 has severe readout asymmetry
(P(0|0) = 0.681, P(1|1) = 0.689) — the mitigator corrects for this.

### Zero-noise extrapolation

ZNEMitigator used unitary folding at scales 1x, 3x, 5x on the Bell circuit:

| Scale | Bell Fidelity |
|-------|--------------|
| 1x (raw) | 0.851 |
| 3x | 0.768 |
| 5x | 0.694 |
| Extrapolated (exp) | 0.895 |

### Randomized benchmarking

Single-qubit Clifford RB on the good qubit (Q56) vs the mediocre qubit (Q50):

| Qubit | Depth 1 | Depth 10 | Depth 50 | Fitted gate fidelity |
|-------|---------|----------|----------|---------------------|
| Q56 | 0.972 | 0.854 | 0.708 | 0.984 |
| Q50 | 0.645 | 0.610 | 0.488 | 0.773 |

### Combined gain

Choosing the right qubits and applying readout mitigation together yield
+0.355 over the naive baseline (worst-pair raw 0.599 → best-pair mitigated 0.954).

See [`examples/07_wukong72_demo.py`](examples/07_wukong72_demo.py) for
the full pipeline in a single script.

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

## License

Apache 2.0
