"""Select optimal qubits for a 2-qubit circuit."""

import asyncio

from qpilot import QPilotClient
from qpilot.characterization import NoiseProfiler
from qpilot.optimization import QubitSelector


async def main():
    async with QPilotClient(host="localhost") as client:
        rb_data = await client.get_rb_data()
        profile = NoiseProfiler.from_rb_data(rb_data)

        selector = QubitSelector(profile)

        # 2-qubit circuit with one CZ gate between logical qubits 0 and 1
        mapping = selector.select(
            num_qubits=2,
            connectivity=[(0, 1)],
        )
        print(f"Selected: {mapping}")
        print(f"Physical qubits: {mapping.physical_qubits}")


if __name__ == "__main__":
    asyncio.run(main())
