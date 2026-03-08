"""Connect to PilotOS and monitor chip state."""

import asyncio

from qpilot import QPilotClient


async def main():
    async with QPilotClient(host="localhost") as client:
        # Send heartbeat to verify connection
        hb = await client.heartbeat()
        print(f"Connected (backend={hb.backend})")

        # Check live chip state from pub-sub events
        state = client.monitor.state
        print(f"Online: {state.online}")
        print(f"Queue length: {state.queue_length}")
        print(f"Calibrating: {state.calibrating}")

        # Register a callback for all events
        def on_event(operation, event):
            print(f"Event: {operation}")

        client.monitor.on_event(on_event)

        # Keep listening for 10 seconds
        await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(main())
