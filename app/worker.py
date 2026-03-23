"""Background worker that runs an Experiment on a QThread and streams snapshots."""

from __future__ import annotations

import dataclasses
import threading
import time

import numpy as np
from PySide6.QtCore import QThread, Signal

from src.core.grid import Grid
from src.core.state import SimState
from src.experiments.base import Experiment
from app.snapshot import SimSnapshot


class SimWorker(QThread):
    """Runs a simulation experiment on a background thread and emits snapshots.

    Signals:
        step_ready: Emitted every 10 simulation steps with a :class:`SimSnapshot`.
        run_complete: Emitted when all seeds finish; carries the result_id string.
        error: Emitted if an unhandled exception occurs; carries the error message.
    """

    step_ready: Signal = Signal(object)
    run_complete: Signal = Signal(str)
    error: Signal = Signal(str)

    def __init__(self, experiment: Experiment, params: dict) -> None:
        super().__init__()
        self._experiment = experiment
        self._params = params
        self._step_counter: int = 0
        self._cancel: bool = False
        self.step_delay_ms: int = 0
        # Set initially so the very first frame is always emitted.
        self._frame_consumed: threading.Event = threading.Event()
        self._frame_consumed.set()

    def cancel(self) -> None:
        """Request graceful cancellation; the worker exits at the next on_step call."""
        self._cancel = True

    def on_frame_consumed(self) -> None:
        """Slot: called by canvas after it has finished displaying a frame."""
        self._frame_consumed.set()

    def set_speed(self, delay_ms: int) -> None:
        """Slot: set per-step sleep duration in ms (0 = run as fast as possible)."""
        self.step_delay_ms = delay_ms

    # ------------------------------------------------------------------
    # QThread interface
    # ------------------------------------------------------------------

    # Minimum wall-clock interval between step_ready emissions (~30 fps).
    _EMIT_INTERVAL: float = 1.0 / 30.0

    def run(self) -> None:
        """Entry point for the background thread."""
        try:
            self._step_counter = 0
            last_emit: float = 0.0

            def on_step(state: SimState) -> None:
                nonlocal last_emit
                if self._cancel:
                    raise _Cancelled()
                state.metrics.setdefault("population_history", []).append(len(state.agents))
                state.metrics.setdefault("step_history", []).append(state.step)
                self._step_counter += 1
                now = time.monotonic()
                if now - last_emit >= SimWorker._EMIT_INTERVAL:
                    if self._frame_consumed.is_set():
                        self._frame_consumed.clear()
                        last_emit = now
                        self.step_ready.emit(_build_snapshot(state))
                if self.step_delay_ms > 0:
                    self.msleep(self.step_delay_ms)

            overrides: dict = {"on_step": on_step}
            if "steps" in self._params:
                overrides["steps"] = self._params["steps"]
            if "seeds" in self._params:
                overrides["seeds"] = self._params["seeds"]

            exp = dataclasses.replace(self._experiment, **overrides)
            exp.run()
            self.run_complete.emit(exp.result_id)

        except _Cancelled:
            pass  # clean exit requested by cancel(), no error signal
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


class _Cancelled(BaseException):
    """Raised inside the worker thread to stop the simulation cleanly."""



# ------------------------------------------------------------------
# Snapshot builder
# ------------------------------------------------------------------

def _build_snapshot(state: SimState) -> SimSnapshot:
    """Construct a :class:`SimSnapshot` from the current :class:`SimState`."""
    resources = state.grid.resources / Grid.MAX_RESOURCE
    np.clip(resources, 0.0, 1.0, out=resources)
    resources = resources.astype(np.float32)

    agents = state.agents
    n = len(agents)

    if n > 0:
        xs = np.fromiter((a.x for a in agents), dtype=np.int32, count=n)
        ys = np.fromiter((a.y for a in agents), dtype=np.int32, count=n)
        colors = np.full((n, 3), [1.0, 0.2, 0.2], dtype=np.float32)
        energies = np.fromiter((a.energy for a in agents), dtype=np.float32, count=n)
        dirs = np.array(
            [[a.last_dx, a.last_dy] for a in agents], dtype=np.int8
        )
        traits = np.array(
            [a.policy.traits if hasattr(a.policy, "traits") else [0.0, 0.0, 0.0, 0.0]
             for a in agents],
            dtype=np.float32,
        )
    else:
        xs = np.empty((0,), dtype=np.int32)
        ys = np.empty((0,), dtype=np.int32)
        colors = np.empty((0, 3), dtype=np.float32)
        energies = np.empty((0,), dtype=np.float32)
        dirs = np.empty((0, 2), dtype=np.int8)
        traits = np.empty((0, 4), dtype=np.float32)

    mating = state.scratch.get("mating_events", 0)

    return SimSnapshot(
        step=state.step,
        width=state.grid.width,
        height=state.grid.height,
        resources=resources,
        agent_xs=xs,
        agent_ys=ys,
        agent_colors=colors,
        agent_energies=energies,
        agent_dirs=dirs,
        agent_traits=traits,
        population=n,
        step_metrics={
            "population": n,
            "mating_events": mating,
        },
    )
