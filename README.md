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
    grid.py         – 2-D grid with resource regeneration, hotspot drift, shock tracking
    policy.py       – TraitPolicy: continuous trait vector + movement scoring
    simulation.py   – Top-level simulation loop and history tracking
  viz/
    renderer.py     – Matplotlib desktop renderer with live maps and history panels
    server.py       – HTTP server for browser-based renderer (port 7777)
    static/
      index.html    – Browser renderer: 4-panel canvas (agents, resources, population, traits)
experiments/
  phase2.py         – Headless multi-seed experiment: 4 environmental volatility conditions
  phase3.py         – Richer-perception trait policy vs Phase 2 baseline
  phase4.py         – Neural policy experiment
  phase5.py         – Stateful neural policy with warm start
  phase6_baseline.py – Phase 4 and 5 warm-start rerun on the standard environment
tests/
  run.py            – Default runner using the desktop renderer
```

## Commands

### Interactive simulation

```bash
# Baseline (condition A) — random seed
python -m tests.run

# Specific seed and step count
python -m tests.run --seed 42 --steps 500

# Run a specific environmental condition
python -m tests.run --condition A          # Baseline   (drift_step=1, noise_rate=3.0)
python -m tests.run --condition B          # Fast drift (drift_step=3, noise_rate=3.0)
python -m tests.run --condition C          # Boom/bust  (drift_step=1, noise_rate=8.0)
python -m tests.run --condition D          # Combined   (drift_step=3, noise_rate=8.0)

# Combined
python -m tests.run --condition C --seed 42 --steps 1000
```

Available desktop strategy modes:

```bash
python -m tests.run --policy-mode baseline         --condition B --steps 1000 --seed 42
python -m tests.run --policy-mode richer           --condition B --steps 1000 --seed 42
python -m tests.run --policy-mode neural           --condition B --steps 1000 --seed 42
python -m tests.run --policy-mode phase5           --condition B --steps 1000 --seed 42
python -m tests.run --policy-mode phase6-neural    --condition B --steps 1000 --seed 42
python -m tests.run --policy-mode phase6-stateful  --condition B --steps 1000 --seed 42
```

Strategy mapping:
1. `baseline` — Phase 1 style trait policy
2. `richer` — Phase 3 richer perception
3. `neural` — Phase 4 neural policy
4. `phase5` — Phase 5 stateful warm-start preset
5. `phase6-neural` — Phase 6 neural warm-start on the standard environment
6. `phase6-stateful` — Phase 6 stateful warm-start on the standard environment

Keyboard shortcuts while the window is open:

| Key | Action |
|-----|--------|
| `p` | Pause / resume |
| `e` | Toggle agent panel between trait colours and energy heatmap |
| `m` | Restart with the next policy preset |
| `q` | Quit |

### Phase 2 experiment — 4 conditions × N seeds

```bash
# Default: all 4 conditions, 5 seeds each, 1000 steps, seed base 42
python -m experiments.phase2

# Custom seed base and step count
python -m experiments.phase2 --seed 42 --steps 1000

# More seeds for tighter error bars
python -m experiments.phase2 --seed 42 --steps 1000 --seeds 10
```

Prints a mean ± std comparison table, then opens three Matplotlib figures:
1. **Trait trajectories** — per-condition per-trait mean ± 1σ over time  
2. **End-state distributions** — histograms of trait values at final step (all seeds pooled)  
3. **Population + trait overview** — population count and all-trait means with fill bands

### Later-phase experiment entrypoints

```bash
# Phase 3: richer perception vs Phase 2
python -m experiments.phase3 --steps 1000 --seed 42 --seeds 5

# Phase 4: neural policy
python -m experiments.phase4 --steps 1000 --seed 42 --seeds 5

# Phase 5: stateful neural policy
python -m experiments.phase5 --condition B --steps 1000 --seed 42

# Phase 6: Phase 4 vs Phase 5 warm-start on the standard environment
python -m experiments.phase6_baseline --steps 1000 --seed 42 --seeds 5
```

## Environment configuration

Pass an `env_config` dict to `Simulation` to override grid parameters:

```python
sim = Simulation(
    width=50, height=50, initial_agents=100, seed=42,
    env_config={
        "drift_step":      2,    # hotspot random-walk speed (default: 1)
        "noise_rate":      5.0,  # expected noise events per step (default: 3.0)
        "noise_magnitude": 3.0,  # max resource change per event (default: 2.0)
    },
)
```

## Requirements

```bash
pip install numpy matplotlib
```

- Python 3.10+

## License

MIT
