"""Zero-noise extrapolation via gate folding."""

from qpilot.mitigation import ZNEMitigator

zne = ZNEMitigator()

# A simple single-qubit circuit: X gate then measure
circuit = [
    [
        {"RPhi": [0, 0, 180, 0]},
        {"Measure": [[0], 1]},
    ]
]

# Generate noise-scaled circuits (k=1, 3, 5)
scaled = zne.generate_scaled_circuits(circuit, [1, 3, 5])
print(f"Generated {len(scaled)} scaled circuits")
print(f"Original gates: {len(scaled[0][0])}")
print(f"3x folded gates: {len(scaled[1][0])}")
print(f"5x folded gates: {len(scaled[2][0])}")

# Extrapolate from noisy measurements
# (in practice these come from running scaled circuits on hardware)
measured = [0.85, 0.72, 0.61]
zero_noise = zne.extrapolate(measured, [1, 3, 5], method="linear")
print(f"\nMeasured at scale factors [1,3,5]: {measured}")
print(f"Zero-noise extrapolation (linear): {zero_noise:.4f}")
