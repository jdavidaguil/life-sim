"""Run a simulation session with browser-based visualisation.

Usage:
    python -m tests.run
    python -m tests.run --steps 500
    python -m tests.run --steps 500 --seed 42
"""

import argparse
import time

from src.core.simulation import Simulation
import src.viz.server as server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simulation with browser-based visualisation.")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps to run (default: 1000)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: random)")
    args = parser.parse_args()

    sim = Simulation(width=50, height=50, initial_agents=100, seed=args.seed)
    server.start(open_browser=True)

    for step in range(args.steps):
        sim.step()
        server.update_state(sim, step)
        time.sleep(0.05)

    print("Simulation complete.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
