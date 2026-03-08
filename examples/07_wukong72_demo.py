"""Full pipeline on Wukong 72 real hardware.

Demonstrates the complete qpilot workflow:
  1. Fetch live calibration data from Origin Quantum Cloud
  2. Build a chip noise profile
  3. Select the best qubit pair for a Bell state experiment
  4. Submit the circuit to Wukong 72 (real QPU)
  5. Apply readout error mitigation
  6. Compare raw vs corrected fidelity

Requires:
  - ORIGINQ_API_KEY (or originq_cloud_api) set in environment / .env
  - Wukong 72 chip status "Online"

Usage:
    python examples/07_wukong72_demo.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qpilot.characterization.noise_profile import ChipNoiseProfile, NoiseProfile
from qpilot.cloud import OriginCloudClient
from qpilot.mitigation.readout import ReadoutMitigator
from qpilot.optimization.qubit_selector import QubitSelector

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SHOTS = 1000


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def build_noise_profile(config: dict) -> ChipNoiseProfile:
    """Build a ChipNoiseProfile from a cloud API chip config response."""
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    profiles: dict[str, NoiseProfile] = {}
    topology: dict[str, list[str]] = {}

    for qid, params in config.get("adjJSON", {}).items():
        if not isinstance(params, dict):
            continue
        avg_fid = params.get("averageFidelity", "")
        if avg_fid == "":
            continue

        p = NoiseProfile(qubit_id=str(qid), last_profiled=now)
        try:
            p.single_gate_fidelity = float(avg_fid)
        except (TypeError, ValueError):
            pass

        rf = params.get("readFidelity", "")
        if isinstance(rf, str) and "/" in rf:
            parts = rf.split("/")
            if len(parts) == 2:
                try:
                    p.readout_fidelity = (float(parts[0]), float(parts[1]))
                except (TypeError, ValueError):
                    pass

        for key in ("T1",):
            val = params.get(key, "")
            if val != "":
                try:
                    p.t1_us = float(val)
                except (TypeError, ValueError):
                    pass
        t2 = params.get("T2", params.get("Ts2", ""))
        if t2 != "":
            try:
                p.t2star_us = float(t2)
            except (TypeError, ValueError):
                pass

        profiles[str(qid)] = p

    for pair_key, gate_data in config.get("gateJSON", {}).items():
        parts = pair_key.split("_")
        if len(parts) != 2:
            continue
        q1, q2 = parts
        topology.setdefault(q1, [])
        topology.setdefault(q2, [])
        if q2 not in topology[q1]:
            topology[q1].append(q2)
        if q1 not in topology[q2]:
            topology[q2].append(q1)

        fid = gate_data.get("fidelity", "") if isinstance(gate_data, dict) else ""
        if fid != "":
            try:
                fid_val = float(fid)
                for q in (q1, q2):
                    if q not in profiles:
                        profiles[q] = NoiseProfile(qubit_id=q, last_profiled=now)
                profiles[q1].two_gate_fidelities[q2] = fid_val
                profiles[q2].two_gate_fidelities[q1] = fid_val
            except (TypeError, ValueError):
                pass

    return ChipNoiseProfile(profiles=profiles, topology=topology, timestamp=now)


def make_bell(q1: int, q2: int) -> str:
    """OriginIR Bell state circuit: H(q1) CNOT(q1,q2) MEASURE."""
    return (
        f"QINIT 72\nCREG 2\n"
        f"H q[{q1}]\n"
        f"CNOT q[{q1}],q[{q2}]\n"
        f"MEASURE q[{q1}],c[0]\n"
        f"MEASURE q[{q2}],c[1]"
    )


def make_prep(qubit: int, state: int) -> str:
    """OriginIR |0> or |1> preparation + measurement for readout calibration."""
    lines = ["QINIT 72", "CREG 1"]
    if state == 1:
        lines.append(f"X q[{qubit}]")
    lines.append(f"MEASURE q[{qubit}],c[0]")
    return "\n".join(lines)


def parse_counts(detail: dict) -> dict[str, int]:
    """Extract counts from a task detail response."""
    result_list = detail.get("qcodeTaskNewVo", {}).get("taskResultList", [])
    if not result_list:
        return {}
    entry = result_list[0]
    prob_raw = entry.get("probCount", "")
    if isinstance(prob_raw, str) and prob_raw:
        prob_data = json.loads(prob_raw)
    elif isinstance(prob_raw, dict):
        prob_data = prob_raw
    else:
        return {}
    keys = prob_data.get("key", [])
    values = prob_data.get("value", [])
    return dict(zip(keys, [int(v) for v in values]))


async def submit_and_poll(
    cloud: OriginCloudClient,
    name: str,
    qprog: str,
    shots: int = SHOTS,
) -> dict[str, int]:
    """Submit a circuit and poll until complete. Returns counts dict."""
    logger.info("Submitting %s ...", name)
    resp = await cloud.submit_task(
        qprog=qprog,
        shots=shots,
        is_mapping=False,
        task_name=f"qpilot_{name}",
    )
    task_id = resp.get("taskId") or resp.get("id", "")
    logger.info("  Task %s submitted (id=%s)", name, task_id)

    start = time.monotonic()
    while time.monotonic() - start < 120:
        await asyncio.sleep(2.0)
        detail = await cloud.get_task_detail(task_id)
        result_list = detail.get("qcodeTaskNewVo", {}).get("taskResultList", [])
        if result_list:
            state = str(result_list[0].get("taskState", "0"))
            if state == "3":
                counts = parse_counts(detail)
                logger.info("  %s done in %.1fs: %s", name, time.monotonic() - start, counts)
                return counts
            if state == "4":
                raise RuntimeError(f"Task {name} failed: {result_list[0]}")

    raise TimeoutError(f"Task {name} timed out")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


async def main():
    async with OriginCloudClient() as cloud:
        # ── Step 1: Fetch live calibration ──────────────────────────
        print("=" * 60)
        print("Step 1: Fetching live calibration from Wukong 72")
        print("=" * 60)

        config = await cloud.get_chip_config("72")
        status = config.get("status", "unknown")
        print(f"  Chip status: {status}")
        if status != "Online":
            print(f"  Chip is {status} — cannot run circuits. Exiting.")
            return

        # ── Step 2: Build noise profile ─────────────────────────────
        print("\nStep 2: Building noise profile")
        chip_profile = build_noise_profile(config)
        print(f"  Profiled qubits: {len(chip_profile.profiles)}")
        print(f"  CZ connections:  {sum(len(v) for v in chip_profile.topology.values()) // 2}")

        top5 = chip_profile.best_qubits(5)
        print(f"  Top 5 qubits:    {', '.join(f'Q{q}' for q in top5)}")

        # ── Step 3: Select best qubit pair ──────────────────────────
        print("\nStep 3: Selecting optimal qubit pair for Bell state")
        selector = QubitSelector(chip_profile)
        mapping = selector.select(num_qubits=2, connectivity=[(0, 1)])
        q1 = int(mapping.physical(0))
        q2 = int(mapping.physical(1))
        print(f"  Selected: Q{q1}–Q{q2} (score={mapping.score:.4f})")

        cz_fid = chip_profile.pair_fidelity(str(q1), str(q2))
        p1 = chip_profile.profiles[str(q1)]
        p2 = chip_profile.profiles[str(q2)]
        print(
            f"  Q{q1}: gate={p1.single_gate_fidelity:.4f}, readout=({p1.readout_fidelity[0]:.4f}, {p1.readout_fidelity[1]:.4f})"
        )
        print(
            f"  Q{q2}: gate={p2.single_gate_fidelity:.4f}, readout=({p2.readout_fidelity[0]:.4f}, {p2.readout_fidelity[1]:.4f})"
        )
        print(f"  CZ fidelity: {cz_fid:.4f}" if cz_fid else "  CZ fidelity: N/A")

        # ── Step 4: Readout calibration (4 circuits) ────────────────
        print("\nStep 4: Readout calibration on selected qubits")
        fidelity_pairs = []
        for q in [q1, q2]:
            c0 = await submit_and_poll(cloud, f"cal_prep0_q{q}", make_prep(q, 0))
            c1 = await submit_and_poll(cloud, f"cal_prep1_q{q}", make_prep(q, 1))
            total0 = sum(c0.values()) or 1
            total1 = sum(c1.values()) or 1
            p00 = c0.get("0x0", c0.get("0", 0)) / total0
            p11 = c1.get("0x1", c1.get("1", 0)) / total1
            fidelity_pairs.append((p00, p11))
            print(f"  Q{q}: P(0|0)={p00:.4f}, P(1|1)={p11:.4f}")

        mitigator = ReadoutMitigator.from_fidelity_params(fidelity_pairs, [str(q1), str(q2)])

        # ── Step 5: Run Bell state circuit ──────────────────────────
        print("\nStep 5: Running Bell state on real hardware")
        bell_ir = make_bell(q1, q2)
        print(f"  Circuit:\n    {bell_ir.replace(chr(10), chr(10) + '    ')}")
        counts = await submit_and_poll(cloud, "bell_state", bell_ir)

        # ── Step 6: Analyze — raw vs mitigated ──────────────────────
        print("\nStep 6: Results")
        print("-" * 60)

        total = sum(counts.values()) or 1
        raw_00 = counts.get("0x0", 0) / total
        raw_11 = counts.get("0x3", 0) / total
        raw_fid = raw_00 + raw_11
        print(f"  Raw counts:    {counts}")
        print(f"  Raw fidelity:  P(|00>) + P(|11>) = {raw_00:.4f} + {raw_11:.4f} = {raw_fid:.4f}")

        corrected = mitigator.correct(counts)
        mit_00 = corrected.get("0x0", 0.0)
        mit_11 = corrected.get("0x3", 0.0)
        mit_fid = mit_00 + mit_11
        print(f"\n  Mitigated:     P(|00>) + P(|11>) = {mit_00:.4f} + {mit_11:.4f} = {mit_fid:.4f}")

        improvement = mit_fid - raw_fid
        print(f"  Improvement:   +{improvement:.4f} ({improvement * 100:+.1f}%)")

        print("\n" + "=" * 60)
        print(f"QPilot selected Q{q1}–Q{q2} and improved Bell fidelity")
        print(f"from {raw_fid:.3f} to {mit_fid:.3f} with readout mitigation.")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
