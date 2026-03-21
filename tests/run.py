"""Run a short simulation session.

Usage:
    python -m tests.run
"""

from src.core.simulation import Simulation
from src.viz.renderer import Renderer


def main() -> None:
    sim = Simulation(width=50, height=50, initial_agents=100)
    renderer = Renderer()

    for step in range(200):
        if not renderer.running:
            break
        sim.step()
        renderer.render(sim, step)

    renderer.close()


if __name__ == "__main__":
    main()
