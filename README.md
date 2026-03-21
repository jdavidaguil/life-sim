# life-sim

A grid-based agent life simulation exploring adaptive strategy 
through variation and selection.

Agents live on a 2-D grid with non-uniform regenerating resources.
Each agent carries a continuous **trait vector** that governs its 
movement decisions. Traits are inherited with Gaussian mutation on 
reproduction. Selection pressure determines which trait combinations 
survive.

## Trait vector

| Trait | Description |
|-------|-------------|
| `resource_weight` | Attraction toward resource-rich cells |
| `crowd_sensitivity` | Penalty weight for crowded cells |
| `noise` | Random exploration coefficient |
| `energy_awareness` | Extra resource attraction when energy is low |

## Project structure

```
src/
  core/
    agent.py        – Agent dataclass and lifecycle logic
    grid.py         – 2-D grid with resource regeneration and hotspot drift
    policy.py       – TraitPolicy: continuous trait vector + movement scoring
    simulation.py   – Top-level simulation loop and history tracking
  viz/
    renderer.py     – Matplotlib desktop renderer (RGB trait map + resource panel)
    server.py       – HTTP server for browser-based renderer (port 7777)
    static/
      index.html    – Browser renderer: 4-panel canvas (agents, resources, population, traits)
tests/
  run.py            – Default runner using the desktop renderer
```

## Quick start

```bash
pip install numpy matplotlib

# Run with the Matplotlib desktop renderer
python -m tests.run
python -m tests.run --seed 42 --steps 500

# Run with the browser renderer (open http://localhost:7777)
python -m src.viz.server &
python -m tests.run
```

## Environment configuration

Pass an `env_config` dict to `Simulation` to override grid parameters:

```python
sim = Simulation(
    width=50, height=50, initial_agents=100, seed=42,
    env_config={
        "drift_step": 2,       # hotspot random-walk speed (default: 1)
        "noise_rate": 5.0,     # expected noise events per step (default: 3.0)
        "noise_magnitude": 3.0 # max resource change per event (default: 2.0)
    },
)
```

## Requirements

- Python 3.10+
- NumPy
- Matplotlib

## License

MIT
