"""Simulation state container for the phase-based loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from src.core.grid import Grid


@dataclass
class SimState:
    """Pure data container shared by all phase functions each step.

    Attributes:
        grid: The simulation grid.
        agents: Active agent roster.
        rng: Shared random number generator.
        step: Current step number, starts at 0.
        metrics: Experiment measurements; persists across steps.
        scratch: Within-step communication between phases; cleared at the
            start of every step.
    """

    grid: Grid
    agents: list
    rng: np.random.Generator
    step: int = 0
    metrics: dict = field(default_factory=dict)
    scratch: dict = field(default_factory=dict)
