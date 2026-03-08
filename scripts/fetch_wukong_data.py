"""Build a real Wukong 72 noise profile and save as JSON fixtures.

Tries the Origin Cloud HTTP API first. Falls back to the local PilotOS
simulator chip config if the API is unreachable.

Usage:
    python scripts/fetch_wukong_data.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qpilot.characterization.noise_profile import ChipNoiseProfile, NoiseProfile
from qpilot.optimization.qubit_selector import QubitSelector

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "wukong72"
LOCAL_CONFIG = (
    PROJECT_ROOT / "QPilotos-V4.0" / "python_simulator" / "ChipConfig" / "ChipArchConfig_72.json"
)
TODAY = datetime.now(UTC).strftime("%Y-%m-%d")


def save_json(name: str, data: object) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / name
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Saved %s (%d bytes)", path, path.stat().st_size)
    return path


async def try_cloud_fetch() -> dict | None:
    """Try fetching chip config from the Origin Cloud HTTP API."""
    try:
        from qpilot.cloud import OriginCloudClient

        async with OriginCloudClient() as cloud:
            logger.info("Fetching chip config from cloud...")
            config = await cloud.get_chip_config("72")
            save_json(f"cloud_chip_config_{TODAY}.json", config)

            logger.info("Fetching queue info...")
            queue = await cloud.get_queue_info("72")
            save_json(f"queue_info_{TODAY}.json", queue)

            logger.info("Fetching calibration timestamps...")
            cal = await cloud.get_calibration_timestamps("72")
            save_json(f"calibration_times_{TODAY}.json", cal)

            return config
    except Exception as exc:
        logger.warning("Cloud API unavailable (%s), falling back to local config", exc)
        return None


def load_local_config() -> dict:
    """Load chip config from the local PilotOS simulator data."""
    logger.info("Loading local chip config from %s", LOCAL_CONFIG)
    with open(LOCAL_CONFIG) as f:
        raw = json.load(f)
    return raw["QuantumChipArch"]


def build_noise_profile(config: dict) -> tuple[ChipNoiseProfile, dict]:
    """Build a ChipNoiseProfile from raw chip config data."""
    now = datetime.now(UTC)
    profiles: dict[str, NoiseProfile] = {}
    topology: dict[str, list[str]] = {}

    if isinstance(config, str):
        config = json.loads(config)

    # Per-qubit parameters
    qubit_params = _get(config, "QubitParams", "qubit_params")
    if isinstance(qubit_params, str):
        qubit_params = json.loads(qubit_params)

    if isinstance(qubit_params, dict):
        for qid, params in qubit_params.items():
            qid = str(qid)
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except (json.JSONDecodeError, TypeError):
                    continue
            if not isinstance(params, dict):
                continue

            p = NoiseProfile(qubit_id=qid, last_profiled=now)

            fmat = params.get("FidelityMat", params.get("fidelity_mat"))
            if fmat and isinstance(fmat, list) and len(fmat) == 2:
                try:
                    p00 = float(fmat[0][0]) if len(fmat[0]) > 0 else 1.0
                    p11 = float(fmat[1][1]) if len(fmat[1]) > 1 else 1.0
                    p.readout_fidelity = (p00, p11)
                except (IndexError, TypeError, ValueError):
                    pass

            sg = params.get("SingleGateFidelity", params.get("single_gate_fidelity"))
            if sg is not None:
                try:
                    p.single_gate_fidelity = float(sg)
                except (TypeError, ValueError):
                    pass

            t1 = params.get("T1", params.get("t1"))
            if t1 is not None:
                try:
                    p.t1_us = float(t1)
                except (TypeError, ValueError):
                    pass

            t2 = _get(params, "T2star", "T2", "t2star", "t2")
            if t2 is not None:
                try:
                    p.t2star_us = float(t2)
                except (TypeError, ValueError):
                    pass

            profiles[qid] = p

    # Topology from CompensateAngle keys (format: "CZ(q1,q2)")
    compensate = config.get("CompensateAngle", {})
    for pair_key in compensate:
        # Parse "CZ(0,1)" -> ("0", "1")
        inner = pair_key.replace("CZ(", "").replace(")", "")
        parts = inner.split(",")
        if len(parts) != 2:
            continue
        q1, q2 = parts[0].strip(), parts[1].strip()
        topology.setdefault(q1, [])
        topology.setdefault(q2, [])
        if q2 not in topology[q1]:
            topology[q1].append(q2)
        if q1 not in topology[q2]:
            topology[q2].append(q1)

    # Also use AdjMatrix if present
    adj = config.get("AdjMatrix", {})
    if isinstance(adj, dict):
        for q1, neighbors in adj.items():
            q1 = str(q1)
            if isinstance(neighbors, list):
                for q2 in neighbors:
                    q2 = str(q2)
                    topology.setdefault(q1, [])
                    topology.setdefault(q2, [])
                    if q2 not in topology[q1]:
                        topology[q1].append(q2)
                    if q1 not in topology[q2]:
                        topology[q2].append(q1)

    # Top-level RB fidelity data (from cloud API responses)
    sg_fidelity = _get(config, "SingleGateFidelity", "singleGateFidelity")
    if isinstance(sg_fidelity, dict) and "qubit" in sg_fidelity:
        for qid, fid in zip(sg_fidelity["qubit"], sg_fidelity.get("fidelity", [])):
            qid = str(qid)
            if qid not in profiles:
                profiles[qid] = NoiseProfile(qubit_id=qid, last_profiled=now)
            try:
                profiles[qid].single_gate_fidelity = float(fid)
            except (TypeError, ValueError):
                pass

    dg_fidelity = _get(config, "DoubleGateFidelity", "doubleGateFidelity")
    if isinstance(dg_fidelity, dict) and "qubitPair" in dg_fidelity:
        for pair_str, fid in zip(dg_fidelity.get("qubitPair", []), dg_fidelity.get("fidelity", [])):
            parts = str(pair_str).split("-")
            if len(parts) != 2:
                continue
            q1, q2 = parts
            fid_val = float(fid)
            for q in (q1, q2):
                if q not in profiles:
                    profiles[q] = NoiseProfile(qubit_id=q, last_profiled=now)
            profiles[q1].two_gate_fidelities[q2] = fid_val
            profiles[q2].two_gate_fidelities[q1] = fid_val

    # Available qubits
    for q in config.get("AvailableQubits", []):
        qid = str(q)
        if qid not in profiles:
            profiles[qid] = NoiseProfile(qubit_id=qid, last_profiled=now)

    chip_profile = ChipNoiseProfile(profiles=profiles, topology=topology, timestamp=now)

    summary = {
        "chip": "Wukong 72",
        "timestamp": now.isoformat(),
        "num_qubits": len(profiles),
        "num_connections": sum(len(v) for v in topology.values()) // 2,
        "profiles": {},
        "topology": topology,
    }
    for qid, p in sorted(profiles.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
        summary["profiles"][qid] = {
            "single_gate_fidelity": p.single_gate_fidelity,
            "readout_fidelity": list(p.readout_fidelity),
            "composite_fidelity": p.composite_fidelity,
            "two_gate_fidelities": p.two_gate_fidelities,
            "t1_us": p.t1_us,
            "t2star_us": p.t2star_us,
        }

    return chip_profile, summary


def qubit_report(chip_profile: ChipNoiseProfile) -> str:
    """Generate a qubit quality report."""
    lines = [
        f"# Wukong 72 — Qubit Quality Report ({TODAY})",
        "",
        f"Profiled qubits: {len(chip_profile.profiles)}",
        f"CZ connections: {sum(len(v) for v in chip_profile.topology.values()) // 2}",
        "",
        "## Top 10 Qubits by Composite Fidelity",
        "",
        "| Rank | Qubit | Gate | P(0|0) | P(1|1) | Composite | T1 (us) | T2* (us) |",
        "|------|-------|------|--------|--------|-----------|---------|----------|",
    ]

    best = chip_profile.best_qubits(10)
    for rank, qid in enumerate(best, 1):
        p = chip_profile.profiles[qid]
        p00, p11 = p.readout_fidelity
        t1 = f"{p.t1_us:.1f}" if p.t1_us else "—"
        t2 = f"{p.t2star_us:.1f}" if p.t2star_us else "—"
        lines.append(
            f"| {rank} | Q{qid} | {p.single_gate_fidelity:.4f} | "
            f"{p00:.4f} | {p11:.4f} | {p.composite_fidelity:.4f} | {t1} | {t2} |"
        )

    # Best connected subgraphs
    lines += ["", "## Best Connected 4-Qubit Subgraph", ""]
    subgraph = chip_profile.best_connected_subgraph(4)
    if subgraph:
        lines.append(f"Qubits: {', '.join(f'Q{q}' for q in subgraph)}")
        for q in subgraph:
            p = chip_profile.profiles[q]
            lines.append(f"  Q{q}: composite={p.composite_fidelity:.4f}")
    else:
        lines.append("Insufficient connectivity for a 4-qubit subgraph.")

    # Best pairs
    lines += [
        "",
        "## Top 10 Qubit Pairs (by sum of composite + CZ fidelity)",
        "",
        "| Pair | Composite Q1 | Composite Q2 | CZ Fidelity |",
        "|------|-------------|-------------|-------------|",
    ]
    scored_pairs = []
    seen: set[tuple[str, str]] = set()
    for q1, neighbors in chip_profile.topology.items():
        for q2 in neighbors:
            pair = tuple(sorted([q1, q2]))
            if pair in seen:
                continue
            seen.add(pair)
            cz = chip_profile.pair_fidelity(q1, q2) or 0.0
            p1 = chip_profile.profiles.get(q1)
            p2 = chip_profile.profiles.get(q2)
            if p1 and p2:
                score = p1.composite_fidelity + p2.composite_fidelity + cz
                scored_pairs.append(
                    (q1, q2, p1.composite_fidelity, p2.composite_fidelity, cz, score)
                )

    scored_pairs.sort(key=lambda x: x[5], reverse=True)
    for q1, q2, c1, c2, cz, _ in scored_pairs[:10]:
        lines.append(f"| Q{q1}–Q{q2} | {c1:.4f} | {c2:.4f} | {cz:.4f} |")

    # QubitSelector validation
    lines += ["", "## QubitSelector Validation", ""]
    try:
        selector = QubitSelector(chip_profile)
        mapping = selector.select(num_qubits=2, connectivity=[(0, 1)])
        lines.append(f"Best 2-qubit pair (constrained): {mapping}")
        lines.append(f"  Physical qubits: {mapping.physical_qubits}")
        lines.append(f"  Score: {mapping.score:.4f}")
    except Exception as e:
        lines.append(f"QubitSelector failed: {e}")

    return "\n".join(lines) + "\n"


def _get(d: dict, *keys: str) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None


async def main():
    # Try cloud API first, fall back to local config
    cloud_config = await try_cloud_fetch()

    if cloud_config:
        config = cloud_config
        logger.info("Using cloud chip config")
    else:
        config = load_local_config()
        save_json(f"chip_config_{TODAY}.json", config)
        logger.info("Using local chip config (%d keys)", len(config))

    # Build noise profile
    chip_profile, summary = build_noise_profile(config)
    save_json(f"noise_profile_{TODAY}.json", summary)
    save_json("topology.json", summary["topology"])

    available = config.get("AvailableQubits", [])
    if available:
        save_json("available_qubits.json", available)

    # Generate report
    report = qubit_report(chip_profile)
    report_path = DATA_DIR / f"qubit_report_{TODAY}.md"
    report_path.write_text(report)
    logger.info("Saved qubit report to %s", report_path)

    print()
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
