# Life-Sim — Research Findings

> **Status:** Phase 6 complete. Phases 1–5 concluded.
> Active development continues — see [What's Next](#whats-next).

---

## What this is

A grid-based agent life simulation exploring how adaptive strategies emerge under selection pressure. Agents live on a 50×50 grid, consume regenerating resources, reproduce when energy exceeds a threshold, and die when energy reaches zero. **No target behavior is specified.** Every strategy that appears must earn its place through selection.

The simulation runs through six phases of increasing representational complexity:

| Phase | Name | Genome |
|-------|------|--------|
| 0 | Discrete Strategies | Fixed policy: Greedy or Explorer |
| 1 | Continuous Traits | 4 floats: `[resource_wt, crowd_sens, noise, energy_aware]` |
| 2 | Dynamic Environments | Same 4 floats × 4 environmental conditions |
| 3 | Richer Perception | Same 4 floats + gradient, density, relative energy inputs |
| 4 | Neural Genome | 224 weights: 18→8 ReLU →8 softmax |
| 5 | Internal State | 292 weights: same + 4-value persistent memory vector |

---

## Environment

Four conditions were tested across all phases:

| Condition | Hotspot Drift | Noise Rate | Noise Magnitude |
|-----------|--------------|------------|-----------------|
| A — Baseline | 1 cell/step | 3.0 | 2.0 |
| B — Fast drift | 3 cells/step | 3.0 | 2.0 |
| C — Boom/bust | 1 cell/step | 8.0 | 4.0 |
| D — Combined | 3 cells/step | 8.0 | 4.0 |

All multi-condition experiments run with 5 seeds × 1000 steps.

---

## Finding 1 — Random movement dominates in stable resource environments

In the Phase 1 baseline, selection consistently drives the trait vector toward:

- **Noise: 0.58** (high randomness)
- **Resource weight: 0.06** (near-zero directed tracking)
- **Crowd sensitivity: 0.10**
- **Energy awareness: 0.11**

This result is robust across all five seeds and reached by step ~700. It persists across Phases 2 and 3 regardless of environment configuration or perception richness.

The Phase 4 neural network (224 weights, no backpropagation) **independently converges to near-uniform directional output** — rediscovering random walk through a completely different mechanism.

**Four mechanisms, one consistent answer.**

### Why this happens

With 4 hotspots on a 50×50 grid (sigma=4.0) and noise events averaging 3 per step, directed movement reaches resources with nearly the same frequency as random movement. The benefit of tracking is near-zero; the cost, in evolutionary terms, is zero too — but selection has no reason to maintain the structure.

### Falsifiable statement

*In spatially stable resource environments with regenerating hotspots, random movement is fitness-equivalent to informed movement at these population densities and resource regeneration rates.*

---

## Finding 2 — Environmental volatility shifts trait attractors

Across Phase 2 conditions, traits respond systematically to environmental parameters:

| Condition | Resource Wt | Crowd Sens | Noise | Energy Aware |
|-----------|------------|------------|-------|--------------|
| A — Baseline | 0.149 ± 0.116 | 0.744 ± 0.092 | 0.589 ± 0.106 | 0.091 ± 0.015 |
| B — Fast drift | 0.211 ± 0.073 | 0.508 ± 0.249 | 0.669 ± 0.077 | 0.147 ± 0.039 |
| C — Boom/bust | 0.178 ± 0.071 | 0.601 ± 0.265 | 0.708 ± 0.083 | 0.124 ± 0.053 |
| D — Combined | 0.262 ± 0.100 | 0.493 ± 0.266 | 0.660 ± 0.041 | 0.184 ± 0.054 |

Key patterns:
- **Fast drift** raises resource weight — predictable hotspot movement rewards tracking
- **Boom/bust** raises noise and energy awareness — unpredictable shocks reward flexibility
- **Combined volatility** produces the highest cross-seed variance — populations under identical conditions reach different attractors
- **Crowd sensitivity is highest in stable environments** — density-dependent competition is the primary selection pressure when resources are predictable

### Falsifiable statement

*The more volatile the environment, the less the population converges to a single behavioral attractor. Environmental volatility is a diversity-maintaining force.*

---

## Finding 3 — Richer perception increases strategy diversity, not mean fitness

Phase 3 added three inputs to the scoring function:
- Resource gradient direction
- Local agent density as a continuous signal
- Relative energy compared to the population mean

**Energy awareness rose in all four conditions** compared to Phase 2. But mean population did not increase. The key effect is behavioral diversification: Phase 3 populations diverge more across seeds than Phase 2. Different seeds reach different trait attractors under identical conditions.

More information enables more viable strategies — it does not select for one better dominant strategy.

### Phase 2 vs Phase 3 — Condition A (Baseline)

| Trait | Phase 2 | Phase 3 |
|-------|---------|---------|
| Resource weight | 0.149 | 0.213 ↑ |
| Crowd sensitivity | 0.744 | 0.665 ↓ |
| Noise | 0.589 | 0.692 ↑ |
| Energy awareness | 0.091 | **0.337 ↑↑** |

### Falsifiable statement

*Increasing perception richness increases the number of viable behavioral strategies the population explores, not the fitness of the best strategy.*

---

## Finding 4 — Neural genomes require functional initialization to evolve

### Cold start (random initialization)

Neural genomes initialized near zero undergo **neutral evolution**:
- Max weight remains below 0.5 after 1000+ steps
- Probe spreads (directional sensitivity) stay at 0.01–0.06 (near-uniform)
- Genome norm grows from ~1.6 to ~3.0 — pure mutation accumulation, not adaptation

This occurs across all tested environment configurations, including steep resource gradients with zero noise.

### Warm start (initialized from Phase 3 linear solution)

Encoding the Phase 3 attractor as initial weights (`rw=2.0`, `cs=1.5`) produces **stabilizing selection**:

| Situation | North probability |
|-----------|-----------------|
| Max resource north, no crowd | **0.364** |
| Max resource north, crowded | 0.169 (drops — crowd avoidance working) |
| Uniform neutral | ~0.125 (near-uniform — correctly uncertain) |

Probe spreads of **0.29–0.31** maintained across all conditions and 1000 steps.

### Warm start population results (standard environment, 5 seeds)

| Condition | Population | Probe Spread |
|-----------|-----------|--------------|
| A — Baseline | 615 ± 63 | 0.295 |
| B — Fast drift | 621 ± 75 | 0.296 |
| C — Boom/bust | 608 ± 44 | 0.292 |
| D — Combined | 573 ± 32 | 0.294 |

### Falsifiable statement

*Selection cannot build functional neural representations from random initialization within 1000 steps in this environment. Selection can maintain functional complexity, but not assemble it. Warm initialization from a functional solution is a prerequisite.*

---

## Finding 5 — Internal state diversifies before it improves fitness

Phase 5 adds a 4-value persistent state vector to the neural genome (292 total weights). After 1000–2000 steps:

- **State vector std: 0.07 → 0.09** — agents diverge into distinct memory patterns
- **State vector mean: ~0.00** — no single memory strategy dominates
- **Probe spreads: ~0.29** — observation pathway (resource tracking, crowd avoidance) is fully preserved

Phase 5 vs Phase 4 at step 1000:

| Condition | P4 Population | P5 Population | P4 Spread | P5 Spread |
|-----------|--------------|--------------|-----------|-----------|
| A — Baseline | 615 ± 63 | 627 ± 25 | 0.295 | 0.295 |
| B — Fast drift | 621 ± 75 | 535 ± 75 | 0.296 | 0.291 |
| C — Boom/bust | 608 ± 44 | 597 ± 50 | 0.292 | 0.296 |
| D — Combined | 573 ± 32 | 492 ± 73 | 0.294 | 0.293 |
| E — Steep | 574 ± 21 | 586 ± 84 | 0.291 | 0.298 |

Phase 5 underperforms Phase 4 in volatile conditions (B, D) due to additional mutation noise from 68 extra weights. In stable conditions it matches or edges ahead.

### Falsifiable statement

*Adding memory to a functional genome produces behavioral diversification before fitness improvement. Selective advantage for memory requires environments where within-episode history is explicitly informative.*

---

## Cross-Phase Summary

The most consistent result across all six phases is the dominance of random movement as a survival strategy. This result emerges independently through:

1. **Phase 1** — Noise trait rises to 0.58 under direct trait selection
2. **Phase 2** — Noise rises in all four environmental conditions
3. **Phase 3** — Noise dominant despite access to gradient, density, and energy signals
4. **Phase 4** — Neural network independently converges to near-uniform directional output

The second consistent result is that selection can maintain functional complexity but cannot assemble it from scratch within these timescales. The path to complex adaptation runs through functional intermediates.

---

## Portable Statements

These findings are stated abstractly, independent of simulation parameters:

1. **Informativeness determines whether perception evolves.** When the benefit of tracking the environment is near-zero, selection does not maintain tracking structures — even when they are available for free.

2. **Volatility is a diversity-maintaining force.** Stable environments drive populations to a single attractor. Volatile environments sustain multiple viable strategies simultaneously.

3. **More inputs expand the strategy space, not the fitness ceiling.** Richer perception increases behavioral variance across populations, not the quality of the best individual strategy.

4. **Selection maintains; it rarely assembles.** Functional complexity requires a starting point. Mutation from random initialization faces a flat fitness landscape that gradient-free selection cannot navigate efficiently at short timescales.

5. **Memory diversifies before it improves.** Adding within-episode history first differentiates agents from each other. Fitness improvement from memory requires an environment that specifically rewards remembering.

---

## What's Next

Open experiments planned:

- **Competition between lineages** — two populations with different genomes competing for the same resources. Does complexity win, or does faster reproduction?
- **Cooperation vs. defection** — agents that can share or defect on the same cell. Under what conditions does cooperation evolve?
- **Predator-prey coevolution** — two agent types evolving simultaneously. Do arms races produce directional progress?
- **Memory-advantaged environments** — alternating resource patches designed so within-episode spatial history provides a real advantage. Does memory finally earn its fitness cost?
- **Extended timescales** — 10,000+ step runs to test whether cold-start neural evolution eventually produces functional representations

---

## Technical Reference

**Grid:** 50×50, 4–6 resource hotspots, Gaussian spread (sigma=3–4), random walk drift, Poisson noise events

**Agent lifecycle:** move → consume → energy decay (0.5 ± 0.1/step) → reproduce if energy > 18.0 → die if energy ≤ 0

**Reproduction:** energy split 50/50, child inherits mutated genome (Gaussian noise, sigma=0.05 per weight)

**Neural architecture (Phase 4):** 18 inputs × 8 hidden (ReLU) × 8 outputs (softmax). Directions: NW, N, NE, W, E, SW, S, SE

**Warm-start encoding:** W1 diagonal = resource weight (2.0), crowd-input diagonal = −crowd sensitivity (−1.5), W2 = identity

**Internal state (Phase 5):** 4-value tanh vector, persists across steps, reset to zero at birth

**Experiments:** `experiments/phase{1..5}.py`, results in `experiments/results/`

---

*Inspired by Conway's Game of Life and Epstein & Axtell's Growing Artificial Societies.*
*Closest prior art: Sugarscape (1996). Original contribution: structured trait genomes, neural genome evolution, internal state.*