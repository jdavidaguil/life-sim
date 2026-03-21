# life-sim

A grid-based agent life simulation written in Python.

Agents live on a 2-D grid, consume food that regrows over time, and move
according to one of two competing **policies**:

| Policy | Behaviour |
|--------|-----------|
| **Greedy** | Always moves toward the highest-energy neighbouring cell. |
| **Explorer** | Picks random unexplored cells, favouring novelty over short-term reward. |

Agents gain energy by eating food, lose energy each step, reproduce when their
energy exceeds a threshold, and die when energy reaches zero. Overcrowding on
a single cell incurs an extra energy penalty.

## Project structure

```
src/
  core/
    agent.py        – Agent dataclass and lifecycle logic
    grid.py         – 2-D grid with food regrowth
    policy.py       – GreedyPolicy / ExplorerPolicy
    policies.py     – Policy registry helpers
    simulation.py   – Top-level simulation loop & history tracking
  viz/
    renderer.py     – Terminal / matplotlib visualisation helpers
tests/
  run.py            – Test runner
```

## Quick start

```bash
# (optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate

# no external dependencies beyond NumPy
pip install numpy

# run the built-in tests
python -m tests.run

# run a quick simulation from a Python shell
python - <<'EOF'
from src.core.simulation import Simulation

sim = Simulation(width=50, height=50, initial_agents=100, seed=42)
for _ in range(100):
    sim.step()

h = sim.history
print(f"After 100 steps – greedy: {h['greedy'][-1]}, explorer: {h['explorer'][-1]}")
EOF
```

## Requirements

- Python 3.9+
- NumPy

## License

MIT
