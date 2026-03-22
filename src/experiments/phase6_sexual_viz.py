"""Run the Phase 6 sexual reproduction simulation with the desktop renderer.

Usage
-----
    python -m src.experiments.phase6_sexual_viz
    python -m src.experiments.phase6_sexual_viz --steps 2000 --seed 42
    python -m src.experiments.phase6_sexual_viz --mating-cost 3.0 --mating-threshold 10.0

Keyboard shortcuts (same as base renderer):
    p — pause / resume
    q — quit
    e — toggle agent panel between trait colours and energy heatmap
"""

from __future__ import annotations

import argparse
import functools

from src.core.phases import DEFAULT_PHASES
from src.core.phases.reproduce_sexual import (
    reproduce_sexual,
    MATING_ENERGY_COST,
    MATING_ENERGY_THRESHOLD,
)
from src.core.simulation import Simulation
from src.experiments.phase6_sexual import _record_mating_events
from src.viz.renderer import Renderer


def _build_phases(mating_cost: float, mating_threshold: float) -> list:
    """Return DEFAULT_PHASES with reproduce replaced by reproduce_sexual."""
    bound = functools.partial(
        reproduce_sexual,
        mating_cost=mating_cost,
        mating_threshold=mating_threshold,
    )
    bound.__name__ = "reproduce_sexual"  # type: ignore[attr-defined]

    phases = []
    for phase in DEFAULT_PHASES:
        if phase.__name__ == "reproduce":
            phases.append(bound)
        else:
            phases.append(phase)
        # inject mating event recorder immediately after the die phase
        if phase.__name__ == "die":
            phases.append(_record_mating_events)
    return phases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 6 sexual reproduction experiment with live visualisation."
    )
    parser.add_argument("--steps", type=int, default=1000,
                        help="Number of steps to run (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--mating-cost", type=float, default=MATING_ENERGY_COST,
                        help=f"Energy cost per parent per mating (default: {MATING_ENERGY_COST})")
    parser.add_argument("--mating-threshold", type=float, default=MATING_ENERGY_THRESHOLD,
                        help=f"Min energy to mate (default: {MATING_ENERGY_THRESHOLD})")
    parser.add_argument("--delay", type=float, default=0.05,
                        help="Seconds between rendered frames (default: 0.05)")
    args = parser.parse_args()

    phase_list = _build_phases(args.mating_cost, args.mating_threshold)
    label = (
        f"Sexual reproduction  |  "
        f"cost={args.mating_cost}  threshold={args.mating_threshold}  "
        f"seed={args.seed}"
    )
    print(f"Phase list: {[p.__name__ for p in phase_list]}")
    print(f"Config: {label}")
    print()

    sim = Simulation(
        width=50,
        height=50,
        initial_agents=100,
        seed=args.seed,
        policy_mode="richer",
        phases=phase_list,
    )

    renderer = Renderer(delay=args.delay, condition_label=label)
    step = 0

    while step < args.steps and renderer.running:
        sim.step()
        step += 1
        renderer.render(sim, step)

    renderer.close()
    sim.plot_history(block=True)


if __name__ == "__main__":
    main()
