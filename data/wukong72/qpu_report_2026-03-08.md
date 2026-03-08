# Wukong 72 — QPU Experiment Results (2026-03-08)
Total execution time: 118.6s (21 circuits × 1000 shots)

## B1: Readout Characterization

| Qubit | P(0|0) measured | P(1|1) measured | P(0|0) cloud | P(1|1) cloud |
|-------|----------------|----------------|-------------|-------------|
| Q56 | 0.9870 | 0.8630 | 0.9842 | 0.8130 |
| Q62 | 0.9830 | 0.9210 | 0.9878 | 0.8798 |
| Q50 | 0.6810 | 0.6890 | 0.6754 | 0.6742 |
| Q51 | 0.8040 | 0.7400 | 0.8242 | 0.7316 |

## B2: Bell State — Best vs Worst Pair

### Best (Q56-Q62)

| Run | Raw Fidelity | Mitigated Fidelity | Improvement |
|-----|-------------|-------------------|-------------|
| Run 1 | 0.8520 | 0.9538 | +0.1018 (+10.2%) |
| Run 2 | 0.8500 | 0.9415 | +0.0915 (+9.2%) |

### Worst (Q50-Q51)

| Run | Raw Fidelity | Mitigated Fidelity | Improvement |
|-----|-------------|-------------------|-------------|
| Run 1 | 0.5990 | 0.8639 | +0.2649 (+26.5%) |
| Run 2 | 0.6000 | 0.8782 | +0.2782 (+27.8%) |

## B3: Single-Qubit Randomized Benchmarking

| Qubit | Depth | Survival P(|0>) | Expected (from cloud) |
|-------|-------|----------------|----------------------|
| Q56 | 1 | 0.9720 | 0.9986 |
| Q56 | 10 | 0.8540 | 0.9862 |
| Q56 | 50 | 0.7080 | 0.9346 |
| Q56 | **fit** | **f=0.984441** | gate fid from cloud: 0.9986 |
| Q50 | 1 | 0.6450 | 0.9937 |
| Q50 | 10 | 0.6100 | 0.9406 |
| Q50 | 50 | 0.4880 | 0.7658 |
| Q50 | **fit** | **f=0.773353** | gate fid from cloud: 0.9937 |

## B4: Zero-Noise Extrapolation

| Scale | Bell Fidelity |
|-------|--------------|
| 1x | 0.8510 |
| 3x | 0.7680 |
| 5x | 0.6940 |
| **ZNE linear** | **0.8887** |
| **ZNE exponential** | **0.8953** |

### Summary

- Raw 1x Bell fidelity: 0.8510
- ZNE extrapolated: 0.8953
- Improvement: +0.0443 (+4.4%)

## Overall: QPilot Selection + Mitigation Impact

| | Worst Pair (Q50-Q51) | Best Pair (Q56-Q62) |
|--|---------------------|---------------------|
| Raw Bell fidelity | 0.5990 | 0.8520 |
| After readout mitigation | 0.8639 | 0.9538 |
| Qubit selection gain | — | +0.2530 |
| Mitigation gain (best) | — | +0.1018 |
| Total gain | — | +0.3548 |
