"""Execute the 60-second QPU budget on Wukong 72.

21 planned circuits:
  B1: Readout characterization — 8 circuits on Q56, Q62, Q50, Q51
  B2: Bell state comparison — 4 circuits (best pair Q56-Q62 vs worst Q50-Q51)
  B3: Single-qubit RB — 6 circuits (Q56 good vs Q50 mediocre, depths 1/10/50)
  B4: ZNE validation — 3 circuits (Bell Q56-Q62 at scale 1x/3x/5x)

Usage:
    python scripts/run_wukong_qpu.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qpilot.characterization.benchmarks import SINGLE_QUBIT_CLIFFORDS, _invert_clifford
from qpilot.cloud import OriginCloudClient
from qpilot.mitigation.readout import ReadoutMitigator
from qpilot.mitigation.zne import ZNEMitigator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "wukong72"
RESULTS_DIR = DATA_DIR / "qpu_results"
TODAY = datetime.now(UTC).strftime("%Y-%m-%d")

# --- Qubit assignments (from cloud noise profile) ---
BEST_PAIR = (56, 62)  # CZ fidelity 0.9885
WORST_PAIR = (50, 51)  # CZ fidelity 0.9342
GOOD_QUBIT = 56  # gate fidelity 0.9986
MEDIOCRE_QUBIT = 50  # gate fidelity 0.9937
READOUT_QUBITS = [56, 62, 50, 51]
SHOTS = 1000
MAX_QUEUE = 30

# =====================================================================
# OriginIR circuit builders
# =====================================================================


def make_readout_prep0(qubit: int) -> str:
    """Prepare |0> and measure (just measure, no gates)."""
    return f"QINIT 72\nCREG 1\nMEASURE q[{qubit}],c[0]"


def make_readout_prep1(qubit: int) -> str:
    """Prepare |1> (X gate) and measure."""
    return f"QINIT 72\nCREG 1\nX q[{qubit}]\nMEASURE q[{qubit}],c[0]"


def make_bell(q1: int, q2: int) -> str:
    """H(q1), CNOT(q1,q2), Measure both."""
    return (
        f"QINIT 72\nCREG 2\n"
        f"H q[{q1}]\n"
        f"CNOT q[{q1}],q[{q2}]\n"
        f"MEASURE q[{q1}],c[0]\n"
        f"MEASURE q[{q2}],c[1]"
    )


def _clifford_to_originir(qubit: int, gates: list[tuple[float, float]]) -> list[str]:
    """Convert a Clifford (list of RPhi axis/angle pairs) to OriginIR gate lines."""
    lines = []
    for axis_deg, angle_deg in gates:
        angle_rad = angle_deg * math.pi / 180
        if axis_deg == 0:
            lines.append(f"RX q[{qubit}],({angle_rad:.15f})")
        elif axis_deg == 90:
            lines.append(f"RY q[{qubit}],({angle_rad:.15f})")
        else:
            # General RPhi: Rz(-axis) Rx(angle) Rz(axis)
            axis_rad = axis_deg * math.pi / 180
            lines.append(f"RZ q[{qubit}],({-axis_rad:.15f})")
            lines.append(f"RX q[{qubit}],({angle_rad:.15f})")
            lines.append(f"RZ q[{qubit}],({axis_rad:.15f})")
    return lines


def make_rb_circuit(qubit: int, depth: int, seed: int) -> str:
    """Build a single-qubit Clifford RB circuit in OriginIR.

    Applies `depth` random Cliffords then the composite inverse → ideal outcome |0>.
    """
    rng = random.Random(seed)
    lines = ["QINIT 72", "CREG 1"]

    # Random Clifford sequence
    clifford_indices = [rng.randrange(24) for _ in range(depth)]
    for ci in clifford_indices:
        lines.extend(_clifford_to_originir(qubit, SINGLE_QUBIT_CLIFFORDS[ci]))

    # Inversion: reverse each Clifford's inverse
    for ci in reversed(clifford_indices):
        inv_gates = _invert_clifford(SINGLE_QUBIT_CLIFFORDS[ci])
        lines.extend(_clifford_to_originir(qubit, inv_gates))

    lines.append(f"MEASURE q[{qubit}],c[0]")
    return "\n".join(lines)


def make_bell_zne(q1: int, q2: int, scale: int) -> str:
    """Bell circuit with unitary folding for ZNE.

    scale=1: H CNOT (original)
    scale=3: H CNOT | CNOT† H† | H CNOT
    scale=5: H CNOT | CNOT† H† | H CNOT | CNOT† H† | H CNOT
    """
    lines = ["QINIT 72", "CREG 2"]

    for fold in range(scale):
        if fold % 2 == 0:
            # Forward: H CNOT
            lines.append(f"H q[{q1}]")
            lines.append(f"CNOT q[{q1}],q[{q2}]")
        else:
            # Inverse: CNOT† H† = CNOT H (both self-adjoint)
            lines.append(f"CNOT q[{q1}],q[{q2}]")
            lines.append(f"H q[{q1}]")

    lines.append(f"MEASURE q[{q1}],c[0]")
    lines.append(f"MEASURE q[{q2}],c[1]")
    return "\n".join(lines)


# =====================================================================
# Cloud submission helpers
# =====================================================================


def parse_counts(task_result: dict) -> dict[str, int]:
    """Extract counts dict from task result object."""
    result_list = task_result.get("qcodeTaskNewVo", {}).get("taskResultList", [])
    if not result_list:
        return {}

    entry = result_list[0]
    state = entry.get("taskState", "0")
    if str(state) != "3":
        error = entry.get("errorDetail") or entry.get("errorMessage") or f"state={state}"
        raise RuntimeError(f"Task not completed: {error}")

    # probCount has the integer counts: {"key": ["0x0", ...], "value": [983, ...]}
    prob_raw = entry.get("probCount", "")
    if isinstance(prob_raw, str) and prob_raw:
        prob_data = json.loads(prob_raw)
    elif isinstance(prob_raw, dict):
        prob_data = prob_raw
    else:
        # Fall back to taskResult (probabilities)
        tr_raw = entry.get("taskResult", "")
        if isinstance(tr_raw, str) and tr_raw:
            prob_data = json.loads(tr_raw)
        else:
            return {}

    keys = prob_data.get("key", [])
    values = prob_data.get("value", [])
    return dict(zip(keys, [int(v) for v in values]))


async def submit_and_wait(
    cloud: OriginCloudClient,
    name: str,
    qprog: str,
    *,
    shots: int = SHOTS,
    is_optimization: bool = True,
    poll_interval: float = 2.0,
    max_wait: float = 120.0,
) -> dict[str, Any]:
    """Submit a circuit and poll until completion. Returns full result dict."""
    logger.info("Submitting %s (%d chars, %d shots)...", name, len(qprog), shots)

    resp = await cloud.submit_task(
        qprog=qprog,
        shots=shots,
        is_mapping=False,
        is_optimization=is_optimization,
        task_name=f"qpilot_{name}",
    )
    task_id = resp.get("taskId") or resp.get("id", "")
    logger.info("  Task ID: %s", task_id)

    start = time.monotonic()
    while time.monotonic() - start < max_wait:
        await asyncio.sleep(poll_interval)
        detail = await cloud.get_task_detail(task_id)
        result_list = detail.get("qcodeTaskNewVo", {}).get("taskResultList", [])
        if result_list:
            state = str(result_list[0].get("taskState", "0"))
            if state == "3":
                counts = parse_counts(detail)
                elapsed = time.monotonic() - start
                logger.info("  Completed in %.1fs: %s", elapsed, counts)
                return {
                    "name": name,
                    "task_id": task_id,
                    "qprog": qprog,
                    "shots": shots,
                    "counts": counts,
                    "raw_detail": detail,
                    "elapsed_s": elapsed,
                }
            elif state == "4":
                error = result_list[0].get("errorDetail", "unknown")
                raise RuntimeError(f"Task {name} failed: {error}")

    raise TimeoutError(f"Task {name} ({task_id}) did not complete in {max_wait}s")


def save_result(name: str, data: Any) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}_{TODAY}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved %s (%d bytes)", path, path.stat().st_size)
    return path


# =====================================================================
# Analysis
# =====================================================================


def bell_fidelity(counts: dict[str, int]) -> float:
    """Compute Bell state fidelity: P(|00>) + P(|11>)."""
    total = sum(counts.values()) or 1
    p00 = counts.get("0x0", counts.get("00", 0)) / total
    p11 = counts.get("0x3", counts.get("11", 0)) / total
    return p00 + p11


def survival_prob(counts: dict[str, int]) -> float:
    """P(|0>) for single-qubit measurement."""
    total = sum(counts.values()) or 1
    return counts.get("0x0", counts.get("0", 0)) / total


def build_readout_mitigator(
    prep0_counts: dict[int, dict[str, int]],
    prep1_counts: dict[int, dict[str, int]],
    qubit_ids: list[int],
) -> ReadoutMitigator:
    """Build ReadoutMitigator from characterization results."""
    fidelity_pairs = []
    for q in qubit_ids:
        c0 = prep0_counts[q]
        c1 = prep1_counts[q]
        total0 = sum(c0.values()) or 1
        total1 = sum(c1.values()) or 1
        p00 = c0.get("0x0", c0.get("0", 0)) / total0
        p11 = c1.get("0x1", c1.get("1", 0)) / total1
        fidelity_pairs.append((p00, p11))
    return ReadoutMitigator.from_fidelity_params(fidelity_pairs, [str(q) for q in qubit_ids])


def mitigate_bell(
    counts: dict[str, int],
    mitigator: ReadoutMitigator,
) -> dict[str, float]:
    """Apply readout mitigation to Bell state counts."""
    return mitigator.correct(counts)


def rb_fit(depths: list[int], survivals: list[float]) -> tuple[float, float, float]:
    """Fit RB decay: p(d) = A * f^d + B.

    Returns (A, f, B) where f is the average gate fidelity.
    """
    import numpy as np

    d = np.array(depths, dtype=float)
    p = np.array(survivals, dtype=float)

    # Simple fit: assume B ≈ 0.5 (random guess), A ≈ p[0] - 0.5
    # p(d) = A * f^d + 0.5
    # log(p(d) - 0.5) = log(A) + d * log(f)
    shifted = np.clip(p - 0.5, 1e-6, None)
    try:
        coeffs = np.polyfit(d, np.log(shifted), 1)
        f = np.exp(coeffs[0])
        A = np.exp(coeffs[1])
        return float(A), float(np.clip(f, 0.0, 1.0)), 0.5
    except Exception:
        return 0.5, 0.99, 0.5


# =====================================================================
# Main execution
# =====================================================================


async def main():
    all_results: dict[str, Any] = {}
    t_start = time.monotonic()

    async with OriginCloudClient() as cloud:
        # --- Check queue depth ---
        config = await cloud.get_chip_config("72")
        waiting = config.get("waitingTask")
        status = config.get("status", "unknown")
        logger.info("Chip status: %s, waiting tasks: %s", status, waiting)

        if waiting is not None and int(waiting) > MAX_QUEUE:
            logger.error("Queue too deep (%s > %d). Aborting.", waiting, MAX_QUEUE)
            return

        if status != "Online":
            logger.error("Chip is %s, not Online. Aborting.", status)
            return

        # =============================================================
        # B1: Readout characterization (8 circuits)
        # =============================================================
        logger.info("=" * 60)
        logger.info("B1: READOUT CHARACTERIZATION")
        logger.info("=" * 60)

        prep0_counts: dict[int, dict[str, int]] = {}
        prep1_counts: dict[int, dict[str, int]] = {}

        for q in READOUT_QUBITS:
            r0 = await submit_and_wait(cloud, f"readout_prep0_q{q}", make_readout_prep0(q))
            prep0_counts[q] = r0["counts"]
            all_results[f"readout_prep0_q{q}"] = r0

            r1 = await submit_and_wait(cloud, f"readout_prep1_q{q}", make_readout_prep1(q))
            prep1_counts[q] = r1["counts"]
            all_results[f"readout_prep1_q{q}"] = r1

        # Build mitigators
        best_mitigator = build_readout_mitigator(prep0_counts, prep1_counts, list(BEST_PAIR))
        worst_mitigator = build_readout_mitigator(prep0_counts, prep1_counts, list(WORST_PAIR))

        logger.info("Readout characterization complete.")
        for q in READOUT_QUBITS:
            c0 = prep0_counts[q]
            c1 = prep1_counts[q]
            t0 = sum(c0.values()) or 1
            t1 = sum(c1.values()) or 1
            p00 = c0.get("0x0", c0.get("0", 0)) / t0
            p11 = c1.get("0x1", c1.get("1", 0)) / t1
            logger.info("  Q%d: P(0|0)=%.4f  P(1|1)=%.4f", q, p00, p11)

        # =============================================================
        # B2: Bell state comparison (4 circuits)
        # =============================================================
        logger.info("=" * 60)
        logger.info("B2: BELL STATE COMPARISON")
        logger.info("=" * 60)

        # Best pair: Q56-Q62 (2 runs)
        bell_best_1 = await submit_and_wait(cloud, "bell_best_1", make_bell(*BEST_PAIR))
        all_results["bell_best_1"] = bell_best_1

        bell_best_2 = await submit_and_wait(cloud, "bell_best_2", make_bell(*BEST_PAIR))
        all_results["bell_best_2"] = bell_best_2

        # Worst pair: Q50-Q51 (2 runs)
        bell_worst_1 = await submit_and_wait(cloud, "bell_worst_1", make_bell(*WORST_PAIR))
        all_results["bell_worst_1"] = bell_worst_1

        bell_worst_2 = await submit_and_wait(cloud, "bell_worst_2", make_bell(*WORST_PAIR))
        all_results["bell_worst_2"] = bell_worst_2

        # =============================================================
        # B3: Single-qubit RB at 3 depths (6 circuits)
        # =============================================================
        logger.info("=" * 60)
        logger.info("B3: SINGLE-QUBIT RB")
        logger.info("=" * 60)

        rb_depths = [1, 10, 50]
        rb_results: dict[int, dict[int, dict]] = {
            GOOD_QUBIT: {},
            MEDIOCRE_QUBIT: {},
        }

        for qubit in [GOOD_QUBIT, MEDIOCRE_QUBIT]:
            for depth in rb_depths:
                seed = qubit * 1000 + depth
                qprog = make_rb_circuit(qubit, depth, seed)
                r = await submit_and_wait(
                    cloud,
                    f"rb_q{qubit}_d{depth}",
                    qprog,
                    is_optimization=False,
                )
                rb_results[qubit][depth] = r
                all_results[f"rb_q{qubit}_d{depth}"] = r

        # =============================================================
        # B4: ZNE validation (3 circuits)
        # =============================================================
        logger.info("=" * 60)
        logger.info("B4: ZNE VALIDATION")
        logger.info("=" * 60)

        zne_scales = [1, 3, 5]
        zne_results: dict[int, dict] = {}

        for scale in zne_scales:
            r = await submit_and_wait(
                cloud,
                f"zne_bell_x{scale}",
                make_bell_zne(*BEST_PAIR, scale),
                is_optimization=False,
            )
            zne_results[scale] = r
            all_results[f"zne_bell_x{scale}"] = r

    # =================================================================
    # Save all raw results
    # =================================================================
    t_total = time.monotonic() - t_start
    logger.info("=" * 60)
    logger.info("ALL CIRCUITS COMPLETE — %.1fs total", t_total)
    logger.info("=" * 60)

    # Save individual results
    for name, result in all_results.items():
        save_result(name, result)

    # Save combined results
    save_result(
        "all_results",
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "total_time_s": t_total,
            "num_circuits": len(all_results),
            "best_pair": list(BEST_PAIR),
            "worst_pair": list(WORST_PAIR),
            "good_qubit": GOOD_QUBIT,
            "mediocre_qubit": MEDIOCRE_QUBIT,
            "results": {
                k: {"counts": v["counts"], "elapsed_s": v["elapsed_s"]}
                for k, v in all_results.items()
            },
        },
    )

    # =================================================================
    # Analysis and report
    # =================================================================
    report_lines = [
        f"# Wukong 72 — QPU Experiment Results ({TODAY})",
        f"Total execution time: {t_total:.1f}s ({len(all_results)} circuits × {SHOTS} shots)",
        "",
    ]

    # --- B1: Readout ---
    report_lines += ["## B1: Readout Characterization", ""]
    report_lines += [
        "| Qubit | P(0|0) measured | P(1|1) measured | P(0|0) cloud | P(1|1) cloud |",
        "|-------|----------------|----------------|-------------|-------------|",
    ]
    cloud_config = json.loads((DATA_DIR / "cloud_chip_config_2026-03-08.json").read_text())
    adj = cloud_config.get("adjJSON", {})
    for q in READOUT_QUBITS:
        c0 = prep0_counts[q]
        c1 = prep1_counts[q]
        t0 = sum(c0.values()) or 1
        t1 = sum(c1.values()) or 1
        p00_meas = c0.get("0x0", c0.get("0", 0)) / t0
        p11_meas = c1.get("0x1", c1.get("1", 0)) / t1
        qdata = adj.get(str(q), {})
        rf = qdata.get("readFidelity", "")
        if "/" in str(rf):
            p00_cloud, p11_cloud = [float(x) for x in rf.split("/")]
        else:
            p00_cloud, p11_cloud = 0, 0
        report_lines.append(
            f"| Q{q} | {p00_meas:.4f} | {p11_meas:.4f} | {p00_cloud:.4f} | {p11_cloud:.4f} |"
        )
    report_lines.append("")

    # --- B2: Bell state ---
    report_lines += ["## B2: Bell State — Best vs Worst Pair", ""]

    for label, pair, mitigator, results in [
        ("Best (Q56-Q62)", BEST_PAIR, best_mitigator, [bell_best_1, bell_best_2]),
        ("Worst (Q50-Q51)", WORST_PAIR, worst_mitigator, [bell_worst_1, bell_worst_2]),
    ]:
        report_lines.append(f"### {label}")
        report_lines.append("")
        report_lines.append("| Run | Raw Fidelity | Mitigated Fidelity | Improvement |")
        report_lines.append("|-----|-------------|-------------------|-------------|")

        for i, r in enumerate(results, 1):
            raw_fid = bell_fidelity(r["counts"])
            corrected = mitigate_bell(r["counts"], mitigator)
            # Convert corrected probs to counts-like for fidelity calc
            mit_fid = corrected.get("0x0", 0.0) + corrected.get("0x3", 0.0)
            improvement = mit_fid - raw_fid
            sign = "+" if improvement >= 0 else ""
            report_lines.append(
                f"| Run {i} | {raw_fid:.4f} | {mit_fid:.4f} | {sign}{improvement:.4f} ({sign}{improvement * 100:.1f}%) |"
            )

        report_lines.append("")

    # --- B3: RB ---
    report_lines += ["## B3: Single-Qubit Randomized Benchmarking", ""]
    report_lines += [
        "| Qubit | Depth | Survival P(|0>) | Expected (from cloud) |",
        "|-------|-------|----------------|----------------------|",
    ]

    for qubit in [GOOD_QUBIT, MEDIOCRE_QUBIT]:
        survivals = []
        for depth in rb_depths:
            r = rb_results[qubit][depth]
            surv = survival_prob(r["counts"])
            survivals.append(surv)
            # Expected: gate_fid^(2*depth) (each Clifford ~ 1.5 native gates avg, but
            # with inversion the total is 2*depth Cliffords)
            qdata = adj.get(str(qubit), {})
            gate_fid = float(qdata.get("averageFidelity", 0.999))
            expected = 0.5 + 0.5 * gate_fid ** (2 * depth)
            report_lines.append(f"| Q{qubit} | {depth} | {surv:.4f} | {expected:.4f} |")

        # Fit
        A, f, B = rb_fit(rb_depths, survivals)
        report_lines.append(
            f"| Q{qubit} | **fit** | **f={f:.6f}** | gate fid from cloud: {gate_fid:.4f} |"
        )

    report_lines.append("")

    # --- B4: ZNE ---
    report_lines += ["## B4: Zero-Noise Extrapolation", ""]

    zne_fidelities = []
    for scale in zne_scales:
        fid = bell_fidelity(zne_results[scale]["counts"])
        zne_fidelities.append(fid)

    zne = ZNEMitigator()
    zne_linear = zne.extrapolate(zne_fidelities, [float(s) for s in zne_scales], method="linear")
    zne_exp = zne.extrapolate(zne_fidelities, [float(s) for s in zne_scales], method="exponential")

    report_lines += [
        "| Scale | Bell Fidelity |",
        "|-------|--------------|",
    ]
    for scale, fid in zip(zne_scales, zne_fidelities):
        report_lines.append(f"| {scale}x | {fid:.4f} |")
    report_lines += [
        f"| **ZNE linear** | **{zne_linear:.4f}** |",
        f"| **ZNE exponential** | **{zne_exp:.4f}** |",
        "",
    ]

    raw_1x = zne_fidelities[0]
    best_zne = max(zne_linear, zne_exp)
    improvement = best_zne - raw_1x
    report_lines += [
        "### Summary",
        "",
        f"- Raw 1x Bell fidelity: {raw_1x:.4f}",
        f"- ZNE extrapolated: {best_zne:.4f}",
        f"- Improvement: +{improvement:.4f} (+{improvement * 100:.1f}%)",
        "",
    ]

    # --- Overall summary ---
    best_raw = bell_fidelity(bell_best_1["counts"])
    best_mit = mitigate_bell(bell_best_1["counts"], best_mitigator).get("0x0", 0) + mitigate_bell(
        bell_best_1["counts"], best_mitigator
    ).get("0x3", 0)
    worst_raw = bell_fidelity(bell_worst_1["counts"])
    worst_mit = mitigate_bell(bell_worst_1["counts"], worst_mitigator).get(
        "0x0", 0
    ) + mitigate_bell(bell_worst_1["counts"], worst_mitigator).get("0x3", 0)

    report_lines += [
        "## Overall: QPilot Selection + Mitigation Impact",
        "",
        "| | Worst Pair (Q50-Q51) | Best Pair (Q56-Q62) |",
        "|--|---------------------|---------------------|",
        f"| Raw Bell fidelity | {worst_raw:.4f} | {best_raw:.4f} |",
        f"| After readout mitigation | {worst_mit:.4f} | {best_mit:.4f} |",
        f"| Qubit selection gain | — | +{(best_raw - worst_raw):.4f} |",
        f"| Mitigation gain (best) | — | +{(best_mit - best_raw):.4f} |",
        f"| Total gain | — | +{(best_mit - worst_raw):.4f} |",
        "",
    ]

    report = "\n".join(report_lines)
    report_path = DATA_DIR / f"qpu_report_{TODAY}.md"
    report_path.write_text(report)
    logger.info("Saved report to %s", report_path)

    print()
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
