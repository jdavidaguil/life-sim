"""Run a simulation session with the desktop renderer.

Usage:
    python -m tests.run
    python -m tests.run --steps 500
    python -m tests.run --steps 500 --seed 42
"""

import argparse

from src.core.simulation import Simulation
from src.viz.renderer import Renderer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run simulation with desktop visualisation."
    )
    parser.add_argument(
        "--steps", type=int, default=1000,
        help="Number of steps to run (default: 1000)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: random)"
    )
    args = parser.parse_args()

    sim = Simulation(
        width=50, height=50,
        initial_agents=100,
        seed=args.seed,
    )
    renderer = Renderer(delay=0.05)

    for step in range(args.steps):
        if not renderer.running:
            break
        sim.step()
        renderer.render(sim, step)

    renderer.close()
    sim.plot_history(block=True)


if __name__ == "__main__":
    main()
