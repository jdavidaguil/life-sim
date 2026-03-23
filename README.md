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
app/                     – PySide6 workbench (visual runner + results viewer)
  main.py                – Entry point: python -m app.main
  main_window.py         – Root QMainWindow: splitter layout, signal wiring
  worker.py              – SimWorker (QThread): runs Experiment, emits snapshots
  snapshot.py            – SimSnapshot dataclass (one frame of state)
  panels/
    experiment_panel.py  – Experiment picker, seed/step config, progress bar
    simulation_canvas.py – Vispy live canvas: resource heatmap + agent markers
    results_panel.py     – Five-tab results viewer with async chart rendering
src/
  core/
    agent.py             – Agent dataclass
    grid.py              – 50×50 grid with hotspot drift, noise, regeneration
    policy.py            – TraitPolicy (4 floats), NeuralPolicy (224w), StatefulNeuralPolicy (292w)
    state.py             – SimState dataclass (grid, agents, metrics, scratch)
    loop.py              – SimulationLoop: applies ordered phase list each step
    phases/              – Phase functions: move, consume, decay, reproduce, die,
                           regenerate, noise, drift, reproduce_sexual, reproduce_sexual_neural
  experiments/
    base.py              – Experiment dataclass + run() method
    registry.py          – 23 named experiments, get_experiment(), list_experiments()
    phase2_trait_volatility.py  – Trait genome × 4 env conditions
    phase3_richer_perception.py – Richer perception × 4 conditions
    phase4_neural_cold_start.py – Neural cold start × 4 conditions
    phase4_neural_steep.py      – Neural cold start (steep landscape)
    phase5_internal_state.py    – Stateful neural × 4 conditions
    phase6_neural_warm_baseline.py – Neural warm start × 4 conditions
    phase6_sexual.py            – Sexual trait reproduction × conditions
    phase6_neural_sexual.py     – Sexual neural reproduction × conditions
  benchmarking/
    reporter.py          – Load JSON results and render Matplotlib charts
tests/
  ...                    – Unit tests: python -m pytest tests/
```

## Running the workbench

```bash
python -m app.main
```

The workbench opens a three-pane window:

| Pane | Contents |
|------|----------|
| **Left** | Experiment picker with metadata panel, step/seed config, progress bar |
| **Top right** | Live Vispy canvas with colour-mode buttons and a resource/agent legend |
| **Bottom right** | Results viewer: Population History, Final Population, Trait Convergence, Mating Events, Compare tabs |

### Canvas colour modes

| Button | Colours agents by… |
|--------|-------------------|
| **Policy** | Policy type (coral = trait, blue = neural, purple = stateful) |
| **Energy** | Red (low) → Green (high) |
| **Crowd Sens** | Blue intensity = crowd\_sensitivity trait |
| **Noise** | Blue intensity = noise trait |
| **Resource Wt** | Blue intensity = resource\_weight trait |

### Results panel

- **Trait Convergence** requires a run to have been started from the workbench after the current version was installed (per-step trait means are recorded live by the worker and saved to JSON).
- **Compare** tab allows selecting multiple saved result\_ids for cross-experiment bar charts.
- **Export PNG** saves the currently visible tab to a PNG file.

## Experiments

The registry contains 23 named experiments across six research phases:

| Phase | Registry prefix | Policy | Description |
|-------|----------------|--------|-------------|
| 2 | `phase2_trait_*` | trait | 4-float genome × 4 environment conditions |
| 3 | `phase3_richer_*` | richer | Richer perception inputs × 4 conditions |
| 4 | `phase4_neural_*` | neural | Neural cold-start (224w MLP) × 4 conditions |
| 4 | `phase4_neural_steep` | neural | Steep landscape (tight hotspots, no noise) |
| 5 | `phase5_stateful_*` | stateful_warm | 292w MLP + persistent state × 4 conditions |
| 6 | `phase6_warm_*` | neural_warm | Neural warm-start × 4 conditions |
| 6 | `phase6_sexual` | trait | Sexual trait reproduction |
| 6 | `phase6_neural_sexual` | neural | Sexual neural reproduction |

Each experiment runs 5 seeds × 1000 steps by default and saves results to `results/`.

## Environment configuration

Four standard conditions tested across all phases:

| Condition | Hotspot drift | Noise rate | Noise magnitude |
|-----------|--------------|-----------|----------------|
| A — Baseline | 1 cell/step | 3.0 | 2.0 |
| B — Fast drift | 3 cells/step | 3.0 | 2.0 |
| C — Boom/bust | 1 cell/step | 8.0 | 4.0 |
| D — Combined | 3 cells/step | 8.0 | 4.0 |

Pass `env_config` to `Experiment` (or `Grid.__init__`) to override grid parameters:

```python
from src.experiments.base import Experiment

exp = Experiment(
    result_id="my_run",
    env_config={
        "drift_step":       2,    # hotspot random-walk speed (default: 1)
        "noise_rate":       5.0,  # expected noise events per step (default: 3.0)
        "noise_magnitude":  3.0,  # max resource change per event (default: 2.0)
    },
)
exp.run()
```

## Result JSON schema

Each completed seed writes one JSON file to `results/`:

```json
{
  "result_id": "phase2_trait_A",
  "name":      "Trait Genome — Baseline Environment",
  "seed":      42,
  "steps":     1000,
  "env_config": { ... },
  "timestamp": "2026-03-23T00:38:06Z",
  "final_population": 487,
  "metrics": {
    "population_history": [...],
    "step_history":       [...],
    "mean_rw":    [...],
    "mean_cs":    [...],
    "mean_noise": [...],
    "mean_ea":    [...]
  }
}
```

## Requirements

```bash
pip install -r requirements.txt
# numpy, matplotlib, PySide6>=6.6, vispy>=0.14
```

Python 3.10+.

## License

MIT
