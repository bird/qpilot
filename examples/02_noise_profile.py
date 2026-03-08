"""Build a noise profile from live RB calibration data."""

import asyncio

from qpilot import QPilotClient
from qpilot.characterization import NoiseProfiler


async def main():
    async with QPilotClient(host="localhost") as client:
        rb_data = await client.get_rb_data()
        chip_config = await client.get_chip_config()

        profile = NoiseProfiler.from_rb_data(rb_data, chip_config)

        print(f"Profiled {len(profile.profiles)} qubits")
        for qid in profile.best_qubits(5):
            p = profile.profiles[qid]
            print(
                f"  Q{qid}: gate={p.single_gate_fidelity:.4f}, composite={p.composite_fidelity:.4f}"
            )

        best = profile.best_connected_subgraph(4)
        print(f"Best 4-qubit subgraph: {best}")


if __name__ == "__main__":
    asyncio.run(main())
