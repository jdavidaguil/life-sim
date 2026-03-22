"""Phase-based simulation loop."""

from __future__ import annotations

from typing import Callable, List

from src.core.state import SimState


class SimulationLoop:
    """Runs an ordered list of phase functions over a shared SimState.

    Args:
        phases: Ordered list of phase callables, each with signature
            ``(state: SimState) -> None``.
    """

    def __init__(self, phases: List[Callable[[SimState], None]]) -> None:
        self.phases = phases

    def run(self, state: SimState, steps: int) -> None:
        """Advance the simulation by *steps* iterations.

        Each iteration:
        1. Clears ``state.scratch``.
        2. Calls every phase in order.
        3. Increments ``state.step``.

        Args:
            state: Shared simulation state mutated in place.
            steps: Number of steps to execute.
        """
        for _ in range(steps):
            state.scratch.clear()
            for phase in self.phases:
                phase(state)
            state.step += 1
