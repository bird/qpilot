"""Tests for the experimentation harness — runner, scheduler, experiment lifecycle."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


from qpilot.client import QPilotClient
from qpilot.enums import TaskStatus
from qpilot.harness.experiment import (
    ContinueExperiment,
    Experiment,
    ExperimentCircuit,
)
from qpilot.harness.runner import ExperimentRunner
from qpilot.harness.scheduler import ExperimentScheduler
from qpilot.models.responses import (
    MsgTaskAck,
    MsgTaskResultResponse,
    TaskStatusResponse,
)
from qpilot.monitor.chip_monitor import ChipMonitor, ChipState


# --- Mock experiment for testing ---


class SimpleExperiment(Experiment):
    """Trivial experiment that returns a fixed circuit."""

    def __init__(self, n_circuits: int = 1):
        self._n = n_circuits

    async def design(self, chip_state, noise_profile):
        return [
            ExperimentCircuit(
                circuit=[[{"RPhi": [0, 0, 180, 0]}, {"Measure": [[0], 1]}]],
                shots=100,
                metadata={"idx": i},
            )
            for i in range(self._n)
        ]

    async def analyze(self, results):
        return {
            "count": len(results),
            "completed": sum(1 for r in results if r.get("status") == "completed"),
        }


class ContinuingExperiment(Experiment):
    """Experiment that requests continuation on first call."""

    def __init__(self):
        self._call_count = 0

    async def design(self, chip_state, noise_profile):
        return [
            ExperimentCircuit(
                circuit=[[{"RPhi": [0, 0, 180, 0]}, {"Measure": [[0], 1]}]],
                shots=100,
            )
        ]

    async def analyze(self, results):
        self._call_count += 1
        if self._call_count == 1:
            raise ContinueExperiment()
        return {"iterations": self._call_count}


class EmptyExperiment(Experiment):
    """Experiment that produces no circuits."""

    async def design(self, chip_state, noise_profile):
        return []

    async def analyze(self, results):
        return {}


# --- Fixtures ---


def _make_mock_client():
    """Create a mock QPilotClient with all needed methods."""
    client = MagicMock(spec=QPilotClient)
    client.monitor = MagicMock(spec=ChipMonitor)
    client.monitor.state = ChipState(
        online=True, calibrating=False, protected=False, queue_length=0
    )

    # submit_task returns an ack
    client.submit_task = AsyncMock(
        return_value=MsgTaskAck(
            msg_type="MsgTaskAck",
            sn=1,
            err_code=0,
        )
    )

    # get_task_status returns COMPLETED
    client.get_task_status = AsyncMock(
        return_value=TaskStatusResponse(
            msg_type="TaskStatusAck",
            sn=2,
            task_id="test",
            task_status=TaskStatus.COMPLETED,
        )
    )

    # get_task_result returns mock results
    client.get_task_result = AsyncMock(
        return_value=MsgTaskResultResponse(
            msg_type="MsgTaskResult",
            sn=3,
            task_id="test",
            key=[["0x0", "0x1"]],
            prob_count=[[500, 500]],
        )
    )

    # VIP methods
    client.set_vip = AsyncMock()
    client.release_vip = AsyncMock()

    # RB data for noise profile refresh
    client.get_rb_data = AsyncMock()

    return client


# ============================================================
# ExperimentRunner
# ============================================================


class TestExperimentRunnerRunOnce:
    async def test_basic_lifecycle(self):
        """Test the full run_once lifecycle."""
        client = _make_mock_client()
        runner = ExperimentRunner(client, poll_interval=0.01)
        exp = SimpleExperiment(n_circuits=2)

        result = await runner.run_once(exp)

        assert result.circuits_submitted == 2
        assert result.circuits_completed == 2
        assert result.analysis["count"] == 2
        assert result.duration_seconds > 0
        assert client.submit_task.call_count == 2

    async def test_empty_experiment(self):
        """Experiment that designs no circuits."""
        client = _make_mock_client()
        runner = ExperimentRunner(client, poll_interval=0.01)
        exp = EmptyExperiment()

        result = await runner.run_once(exp)

        assert result.circuits_submitted == 0
        assert result.circuits_completed == 0
        assert result.analysis == {"status": "no_circuits"}

    async def test_task_failure(self):
        """Handle task that fails."""
        client = _make_mock_client()
        client.get_task_status = AsyncMock(
            return_value=TaskStatusResponse(
                msg_type="TaskStatusAck",
                sn=2,
                task_id="test",
                task_status=TaskStatus.FAILED,
            )
        )
        runner = ExperimentRunner(client, poll_interval=0.01)
        exp = SimpleExperiment()

        result = await runner.run_once(exp)
        assert result.circuits_completed == 0
        assert result.raw_results[0]["status"] == "failed"

    async def test_task_timeout(self):
        """Handle task that never completes."""
        client = _make_mock_client()
        client.get_task_status = AsyncMock(
            return_value=TaskStatusResponse(
                msg_type="TaskStatusAck",
                sn=2,
                task_id="test",
                task_status=TaskStatus.RUNNING,
            )
        )
        runner = ExperimentRunner(client, poll_interval=0.01, max_poll_attempts=3)
        exp = SimpleExperiment()

        result = await runner.run_once(exp)
        assert result.raw_results[0]["status"] == "timeout"

    async def test_results_accumulated(self):
        """Results should be accumulated across runs."""
        client = _make_mock_client()
        runner = ExperimentRunner(client, poll_interval=0.01)
        exp = SimpleExperiment()

        await runner.run_once(exp)
        await runner.run_once(exp)

        assert len(runner.results) == 2


class TestExperimentRunnerLoop:
    async def test_loop_basic(self):
        """Run a simple loop with 2 iterations."""
        client = _make_mock_client()
        runner = ExperimentRunner(client, poll_interval=0.01)
        exp = SimpleExperiment()

        results = await runner.run_loop(exp, max_iterations=2, cooldown_seconds=0.01)
        assert len(results) == 2

    async def test_loop_handles_errors(self):
        """Loop should continue after errors."""
        client = _make_mock_client()
        call_count = 0

        async def flaky_submit(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("network error")
            return MsgTaskAck(msg_type="MsgTaskAck", sn=1, err_code=0)

        client.submit_task = flaky_submit
        runner = ExperimentRunner(client, poll_interval=0.01)
        exp = SimpleExperiment()

        results = await runner.run_loop(exp, max_iterations=3, cooldown_seconds=0.01)
        # First iteration fails, second and third succeed
        assert len(results) == 2

    async def test_loop_continue_experiment(self):
        """ContinueExperiment should trigger another iteration."""
        client = _make_mock_client()
        runner = ExperimentRunner(client, poll_interval=0.01)
        exp = ContinuingExperiment()

        results = await runner.run_loop(exp, max_iterations=3, cooldown_seconds=0.01)
        # First analyze raises ContinueExperiment (no result), second succeeds
        assert len(results) >= 1


class TestExperimentRunnerVIP:
    async def test_vip_lifecycle(self):
        """VIP mode should acquire and release VIP time."""
        client = _make_mock_client()
        runner = ExperimentRunner(client, poll_interval=0.01)
        exp = SimpleExperiment()

        await runner.run_with_vip(
            exp, duration_seconds=600, max_iterations=1, cooldown_seconds=0.01
        )

        client.set_vip.assert_called_once_with(offset_time=0, exclusive_time=600)
        client.release_vip.assert_called_once()

    async def test_vip_released_on_error(self):
        """VIP time should be released even if experiment fails."""
        client = _make_mock_client()
        client.submit_task = AsyncMock(side_effect=RuntimeError("boom"))
        runner = ExperimentRunner(client, poll_interval=0.01)
        exp = SimpleExperiment()

        await runner.run_with_vip(exp, max_iterations=1, cooldown_seconds=0.01)
        # VIP should still be released
        client.release_vip.assert_called_once()


# ============================================================
# ExperimentScheduler
# ============================================================


class TestExperimentScheduler:
    def _make_scheduler(self):
        client = _make_mock_client()
        runner = ExperimentRunner(client, poll_interval=0.01)
        monitor = client.monitor
        return ExperimentScheduler(runner, monitor), runner

    async def test_submit_and_run(self):
        scheduler, _ = self._make_scheduler()
        exp = SimpleExperiment()
        scheduler.submit(exp)
        assert scheduler.queue_size == 1

        results = await scheduler.run(max_experiments=1)
        assert len(results) == 1
        assert scheduler.queue_size == 0

    async def test_priority_ordering(self):
        scheduler, _ = self._make_scheduler()
        exp_low = SimpleExperiment()
        exp_high = SimpleExperiment()

        scheduler.submit(exp_low, priority=1)
        scheduler.submit(exp_high, priority=10)

        # High priority should be first in queue
        assert scheduler._queue[0].priority == 10

    async def test_chip_unavailable_skips(self):
        """When chip is calibrating, scheduler should wait."""
        scheduler, _ = self._make_scheduler()
        # Set chip as calibrating
        scheduler._monitor.state.calibrating = True

        assert not scheduler.is_chip_available()

    async def test_chip_protected_skips(self):
        scheduler, _ = self._make_scheduler()
        scheduler._monitor.state.protected = True
        assert not scheduler.is_chip_available()

    async def test_queue_full_skips(self):
        scheduler, _ = self._make_scheduler()
        scheduler._monitor.state.queue_length = 100
        assert not scheduler.is_chip_available()

    async def test_stop(self):
        scheduler, _ = self._make_scheduler()
        scheduler.stop()
        assert not scheduler._running

    async def test_empty_queue(self):
        scheduler, _ = self._make_scheduler()
        results = await scheduler.run(max_experiments=5)
        assert len(results) == 0
