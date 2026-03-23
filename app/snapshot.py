"""Immutable data container for a single simulation frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimSnapshot:
    """A lightweight, serialisable snapshot of the simulation state at one step.

    Attributes:
        step: The simulation step number this snapshot was captured at.
        width: Grid width in cells.
        height: Grid height in cells.
        resources: Resource levels normalized to [0, 1], shape (height, width) float32.
        agent_xs: Agent column positions, shape (N,) int32.
        agent_ys: Agent row positions, shape (N,) int32.
        agent_colors: Per-agent RGB colour, shape (N, 3) float32.
        population: Number of living agents (equals len(agent_xs)).
        step_metrics: Scalar metrics captured at this step (e.g. population,
            mating_events).
    """

    step: int
    width: int
    height: int
    resources: np.ndarray       # (H, W) float32, values in [0, 1]
    agent_xs: np.ndarray        # (N,) int32
    agent_ys: np.ndarray        # (N,) int32
    agent_colors: np.ndarray    # (N, 3) float32 RGB
    agent_energies: np.ndarray  # (N,) float32 — raw energy per agent
    agent_dirs: np.ndarray      # (N, 2) int8  — (last_dx, last_dy) per agent
    agent_traits: np.ndarray    # (N, 4) float32 — [resource_weight, crowd_sensitivity, noise, energy_awareness]
    population: int
    step_metrics: dict
