from src.core.grid import Grid
from src.core.state import SimState
from src.core.loop import SimulationLoop
import numpy as np

rng = np.random.default_rng(42)
grid = Grid(50, 50, rng=rng)
state = SimState(grid=grid, agents=[], rng=rng, step=0, metrics={}, scratch={})

def dummy_phase(state: SimState) -> None:
    state.scratch["touched"] = True

loop = SimulationLoop(phases=[dummy_phase])
loop.run(state, steps=3)

assert state.step == 3, f"Expected step=3, got {state.step}"
assert state.scratch.get("touched") == True, "scratch not written by phase"
print("Prompt 1 verification passed")