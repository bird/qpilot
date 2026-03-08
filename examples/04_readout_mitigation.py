"""Apply readout error mitigation to measurement results."""

from qpilot.mitigation import ReadoutMitigator

# Qubit calibration data: (P(0|0), P(1|1))
fidelities = [
    (0.97, 0.95),  # qubit 0
    (0.98, 0.96),  # qubit 1
]

mitigator = ReadoutMitigator.from_fidelity_params(fidelities, ["0", "1"])

# Raw measurement counts from a 2-qubit circuit
raw_counts = {"0x0": 450, "0x1": 50, "0x2": 80, "0x3": 420}

# Full matrix correction
corrected = mitigator.correct(raw_counts)
print("Full matrix correction:")
for bs, prob in sorted(corrected.items()):
    print(f"  {bs}: {prob:.4f}")

# Tensored correction (faster, independent qubit assumption)
corrected_t = mitigator.correct_tensored(raw_counts)
print("\nTensored correction:")
for bs, prob in sorted(corrected_t.items()):
    print(f"  {bs}: {prob:.4f}")
