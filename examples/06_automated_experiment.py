"""Run an automated experiment with the harness."""

import asyncio

from qpilot import QPilotClient
from qpilot.harness import Experiment, ExperimentCircuit, ExperimentRunner


class BellStateExperiment(Experiment):
    """Prepare a Bell state and measure correlations."""

    async def design(self, chip_state, noise_profile):
        # H on qubit 0, CZ on (0,1), measure both
        return [
            ExperimentCircuit(
                circuit=[
                    [
                        {"RPhi": [0, 0, 90, 0]},
                        {"RPhi": [0, 90, 180, 1]},
                        {"CZ": [0, 1, 2]},
                        {"Measure": [[0, 1], 3]},
                    ]
                ],
                shots=1000,
            )
        ]

    async def analyze(self, results):
        completed = [r for r in results if r.get("status") == "completed"]
        return {"pairs_measured": len(completed)}


async def main():
    async with QPilotClient(host="localhost") as client:
        runner = ExperimentRunner(client)
        exp = BellStateExperiment()

        result = await runner.run_once(exp)
        print(f"Submitted: {result.circuits_submitted}")
        print(f"Completed: {result.circuits_completed}")
        print(f"Duration: {result.duration_seconds:.1f}s")
        print(f"Analysis: {result.analysis}")


if __name__ == "__main__":
    asyncio.run(main())
