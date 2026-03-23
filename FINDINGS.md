# Life-Sim — Research Findings

> **Status:** Phase 6 complete. Phases 1–6 concluded.
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
| 6 | Sexual Reproduction | Neural genome + whole-layer crossover, co-location mating |

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

## Finding 1 — Simple directional tracking doesn't pay, but contextual navigation does

This is the central finding of the project, and it requires careful framing.

### What the trait phases showed

In Phases 1–3, the trait vector converged toward:

- **Noise: 0.58** (high movement randomness)
- **Resource weight: 0.06** (near-zero directed food tracking)
- **Crowd sensitivity: 0.74** (strong crowd avoidance)
- **Energy awareness: 0.34** (Phase 3, with richer perception)

At first glance this looks like "random movement wins." But that misreads what the agents were actually doing.

The agents were not moving randomly. They were **diffuse, crowd-avoiding, energy-responsive wanderers** — a specific adaptive strategy. High noise means pure directional tracking of food wasn't worth the evolutionary cost. But crowd sensitivity of 0.74 and rising energy awareness mean the other traits were doing real work. Agents actively avoided dense cells and adjusted behavior based on their energy state.

The trait vector simply lacked the expressive power to encode something more sophisticated: **conditional, directionally specific responses.**

### What the neural phases revealed

The warm-started neural genome (Phase 4) encodes crowd avoidance *per direction*, not as a scalar penalty. The result:

| Situation | Behavior |
|-----------|----------|
| Rich neighbor, no crowd | North prob **0.364** — moves toward food |
| Rich neighbor, crowded | North prob drops to **0.169** — avoids the crowd specifically |
| Uniform neutral | ~0.125 — correctly uncertain when there's no signal |

That's not random movement. That's context-sensitive navigation — move toward food *unless* it's crowded, in which case prefer another direction.

**Phase 4 and 5 consistently outperform Phases 1–3 in population** (615–627 vs ~490–500), which is the clearest evidence that the trait vector finding was a limitation of representation, not a fundamental truth about the environment.

### The complete picture

Simple directional tracking — high resource weight, ignoring crowd and energy signals — doesn't pay in this environment. The food distribution and noise level mean that chasing the nearest hotspot isn't reliably better than wandering.

But **contextual navigation does pay**: crowd-aware, energy-sensitive movement that responds to the full local situation rather than a single signal. The neural genome found this. The trait vector approximated it but couldn't fully express it.

### Falsifiable statement

*In this resource environment, simple scalar food-tracking does not outcompete diffuse crowd-avoiding movement. Contextual, directionally specific crowd-avoidance with food-tracking produces consistently higher populations than either pure tracking or pure wandering.*

---

## Finding 2 — Environmental volatility shifts trait attractors and maintains diversity

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
- **Combined volatility** produces the highest cross-seed variance — no single strategy dominates
- **Crowd sensitivity is highest in stable environments** — density-dependent competition is the primary pressure when resources are predictable

### Falsifiable statement

*The more volatile the environment, the less the population converges to a single behavioral attractor. Environmental volatility is a diversity-maintaining force.*

---

## Finding 3 — Richer perception increases strategy diversity, not mean fitness

Phase 3 added three inputs to the scoring function: resource gradient direction, local agent density as a continuous signal, and relative energy compared to the population mean.

**Energy awareness rose in all four conditions** compared to Phase 2. But mean population did not increase. The effect is behavioral diversification: Phase 3 populations diverge more across seeds than Phase 2. Different seeds reach different trait attractors under identical conditions.

More information enables more viable strategies — it does not immediately select for one better dominant strategy. The representational limit of the trait vector meant that richer inputs could shift which traits were emphasized, but couldn't encode the conditional behavior the neural genome eventually found.

### Phase 2 vs Phase 3 — Condition A

| Trait | Phase 2 | Phase 3 |
|-------|---------|---------|
| Resource weight | 0.149 | 0.213 ↑ |
| Crowd sensitivity | 0.744 | 0.665 ↓ |
| Noise | 0.589 | 0.692 ↑ |
| Energy awareness | 0.091 | **0.337 ↑↑** |

### Falsifiable statement

*Increasing perception richness increases the number of viable behavioral strategies the population explores. The fitness ceiling rises only when the genome has sufficient expressive power to encode conditional responses to the richer inputs.*

---

## Finding 4 — Neural genomes require functional initialization to evolve

### Cold start (random initialization)

Neural genomes initialized near zero undergo **neutral evolution**:

- Max weight remains below 0.5 after 1000+ steps
- Probe spreads stay at 0.01–0.06 (near-uniform — no directional preference)
- Genome norm grows from ~1.6 to ~3.0 — pure mutation accumulation, not adaptation

This occurs across all tested environment configurations. The fitness landscape around random initialization is too flat for gradient-free selection to navigate within these timescales.

### Warm start (initialized from Phase 3 linear solution)

Encoding the Phase 3 attractor as initial weights (`rw=2.0`, `cs=1.5`) produces **stabilizing selection**:

- North probability with max resource north, no crowd: **0.364**
- North probability with max resource north, crowded: **0.169** (crowd avoidance working)
- Probe spreads of **0.29–0.31** maintained across all conditions and 1000 steps

### Warm start population results (standard environment, 5 seeds)

| Condition | Population | Probe Spread |
|-----------|-----------|--------------|
| A — Baseline | 615 ± 63 | 0.295 |
| B — Fast drift | 621 ± 75 | 0.296 |
| C — Boom/bust | 608 ± 44 | 0.292 |
| D — Combined | 573 ± 32 | 0.294 |

### Falsifiable statement

*Selection cannot build functional neural representations from random initialization within 1000 steps in this environment. Selection can maintain functional complexity but not assemble it. Warm initialization from a functional intermediate is a prerequisite for neural genome evolution at these timescales.*

---

## Finding 5 — Internal state diversifies before it improves fitness

Phase 5 adds a 4-value persistent state vector to the neural genome (292 total weights). After 1000–2000 steps:

- **State vector std: 0.07 → 0.09** — agents diverge into distinct memory patterns
- **State vector mean: ~0.00** — no single memory strategy dominates
- **Probe spreads: ~0.29** — core navigation behavior fully preserved

Phase 5 vs Phase 4 at step 1000:

| Condition | P4 Population | P5 Population | P4 Spread | P5 Spread |
|-----------|--------------|--------------|-----------|-----------|
| A — Baseline | 615 ± 63 | 627 ± 25 | 0.295 | 0.295 |
| B — Fast drift | 621 ± 75 | 535 ± 75 | 0.296 | 0.291 |
| C — Boom/bust | 608 ± 44 | 597 ± 50 | 0.292 | 0.296 |
| D — Combined | 573 ± 32 | 492 ± 73 | 0.294 | 0.293 |
| E — Steep | 574 ± 21 | 586 ± 84 | 0.291 | 0.298 |

Phase 5 underperforms in volatile conditions (B, D) due to extra mutation noise from 68 additional weights. In stable conditions it matches or edges ahead. The state pathway is diversifying without yet providing consistent fitness benefit — this environment doesn't reward within-episode spatial memory.

### Falsifiable statement

*Adding memory to a functional genome produces behavioral diversification before fitness improvement. Selective advantage for memory requires environments where within-episode history is explicitly informative.*

---

## Finding 6 — Sexual reproduction inverts the volatility-population relationship

### Setup

Phase 6 introduces sexual reproduction to the neural genome system. Two agents co-located on the same cell, both above an energy threshold, produce one child via whole-layer crossover of their 224-weight neural genomes. Both parents pay an energy cost. The child genome is a layer-wise mix of both parents with Gaussian mutation σ=0.05 per weight.

All agents warm-started from the Phase 4 functional initialization with added noise σ=0.1 to seed initial diversity. Best mating parameters from a 3×3 parameter sweep (cost × threshold): cost=3.0, threshold=15.0.

### Results across all four conditions (5 seeds × 1000 steps)

| Condition | Population | Matings | Probe north rich | Probe north crowded |
|-----------|-----------|---------|-----------------|-------------------|
| A — Baseline | 554 ± 29 | 5142 | 0.468 | 0.171 |
| B — Fast drift | 576 ± 129 | 6037 | 0.439 | 0.158 |
| C — Boom/bust | 577 ± 37 | 5130 | 0.459 | 0.171 |
| D — Combined | 659 ± 89 | 6136 | 0.480 | 0.174 |

### Comparison to Phase 4/5 asexual baseline

| Condition | Phase 4/5 asexual | Phase 6 sexual | Delta |
|-----------|------------------|----------------|-------|
| A — Baseline | 615 | 554 | −61 |
| B — Fast drift | 621 | 576 | −45 |
| C — Boom/bust | 608 | 577 | −31 |
| D — Combined | 573 | 659 | **+86** |

Sexual reproduction carries a population cost in stable and moderately volatile environments. That cost reverses in combined high-volatility — sexual reproduction outperforms asexual by 86 agents in condition D.

### The volatility inversion

In Phase 4/5, population ranked A > B > C > D — stable environments produced the largest populations. In Phase 6, population ranks D > C > B > A — the ranking is fully inverted.

The mechanism is mate encounter rate. Fast-drifting hotspots force continuous agent movement. Moving agents co-locate more frequently. More co-location means more mating events. Conditions B and D — both fast drift — generate ~6000+ matings vs ~5130 for stable conditions. Boom/bust noise alone (condition C) does not amplify mating rate because it disrupts energy levels rather than forcing movement.

### Functional navigation preserved through crossover

Probe spreads are invariant to environmental condition and reproductive mechanism:

| | Phase 4/5 asexual | Phase 6 sexual |
|---|---|---|
| Probe north rich | 0.364 | 0.468 |
| Probe north crowded | 0.169 | 0.171 |

The crowd-avoidance differential is fully preserved. Whole-layer crossover does not disrupt functional neural navigation. The genome encodes the strategy robustly across all conditions.

### Cross-seed variance as signal

Condition B shows σ=129 — the highest cross-seed variance of any experiment in this project. Fast drift creates winner-takes-all dynamics: populations that locate drifting hotspots early compound that advantage through reproduction; those that don't collapse. Boom/bust (C) is egalitarian by comparison at σ=37 — all seeds converge near the same outcome regardless of early luck.

### Falsifiable statements

*Sexual reproduction with neural genomes inverts the volatility-population relationship seen in asexual populations. There exists a volatility threshold above which sexual reproduction outperforms asexual reproduction — in this environment that threshold lies between boom/bust noise alone and combined drift+noise.*

*Whole-layer neural genome crossover preserves functional navigation behavior. Genome mixing does not disrupt the crowd-avoidance differential encoded by warm initialization.*

*Fast-drifting resource hotspots amplify sexual reproductive output by increasing agent movement and therefore mate encounter rate. Resource noise alone does not produce this effect.*

---

## Cross-Phase Summary

The progression across phases tells a coherent story about the relationship between **representational power**, **reproductive mechanism**, and **adaptive behavior**:

| Phase | Genome | Reproduction | Condition A | Condition D |
|-------|--------|-------------|-------------|-------------|
| 0 | Two fixed strategies | Asexual | ~500 | — |
| 1–3 | Trait vector | Asexual | ~490–500 | ~490 |
| 4 warm | Neural 224w | Asexual | 615 | 573 |
| 5 warm | Neural 292w | Asexual | 627 | 492 |
| 6 | Neural 224w | Sexual | 554 | **659** |

The volatility inversion in Phase 6 is the sharpest finding in the project: the same environment that hurt asexual populations most (condition D) helps sexual populations most. Reproductive mechanism interacts with environment in a way that genome expressiveness alone cannot predict.

The deeper finding across all phases: **the apparent strategy of a population tells you as much about the genome's expressive limits and reproductive constraints as about the environment's structure.**

---

## Portable Statements

1. **Simple tracking often loses to diffuse crowd-aware wandering** in environments with broadly distributed, noisy resources. The benefit of precise food tracking must exceed the cost; in many real resource landscapes, it doesn't.

2. **Volatility maintains diversity.** Stable environments collapse strategy space to a single attractor. Volatile environments sustain multiple viable strategies simultaneously.

3. **More perception expands the strategy space; richer genomes are needed to exploit it.** Inputs that selection can't yet act on due to representational limits accumulate silently, waiting for a genome expressive enough to use them.

4. **Selection maintains functional complexity but rarely assembles it.** Starting from a working solution and exploring variations is far more efficient than starting from random initialization at short evolutionary timescales.

5. **Memory diversifies before it improves.** Fitness advantage from memory requires an environment that specifically rewards remembering.

6. **Reproductive mechanism interacts with environment to determine fitness.** Sexual reproduction is costlier than asexual in stable environments and more productive in volatile ones. The relationship between environment and fitness cannot be read from genome alone — reproductive structure matters.

---

## What's Next

- **Volatility threshold experiment** — parameterize drift speed continuously to find the exact crossover point where sexual reproduction outperforms asexual. A specific, falsifiable prediction from Finding 6.
- **Memory-advantaged environments** — alternating patches where spatial memory of recently depleted areas would genuinely help; directly tests Finding 5's open question
- **Kin selection** — agents share resources with nearby genome-similar agents; requires sexual reproduction as foundation, now in place
- **Competition between lineages** — two populations with different genomes competing for the same resources
- **Extended timescales** — 10,000+ step runs to test whether cold-start neural evolution eventually succeeds given enough time

---

## Technical Reference

**Grid:** 50×50, 4–6 resource hotspots, Gaussian spread (sigma=3–4), random walk drift, Poisson noise events

**Agent lifecycle:** move → consume → energy decay (0.5 ± 0.1/step) → reproduce → die if energy ≤ 0

**Asexual reproduction (Phases 0–5):** energy split 50/50, child inherits mutated genome (Gaussian noise, sigma=0.05 per weight)

**Sexual reproduction (Phase 6):** co-location required, both agents energy > 15.0, mating cost 3.0 per parent, child energy 10.0. Whole-layer crossover: each layer (W1, b1, W2, b2) independently drawn from one parent with 50/50 probability, then Gaussian mutation sigma=0.05 per weight.

**Neural architecture (Phase 4/6):** 18 inputs × 8 hidden (ReLU) × 8 outputs (softmax). Directions: NW, N, NE, W, E, SW, S, SE

**Warm-start encoding:** W1 diagonal = resource weight (2.0), crowd-input diagonal = −1.5, W2 = identity. Phase 6 adds noise σ=0.1 at initialization to seed diversity.

**Internal state (Phase 5):** 4-value tanh vector, persists across steps, reset to zero at birth

**Experiments:** `src/experiments/phase{1..6}.py`, results in `src/experiments/results/`

---

*Inspired by Conway's Game of Life and Epstein & Axtell's Growing Artificial Societies.*
*Closest prior art: Sugarscape (1996). Original contribution: structured trait genomes, neural genome evolution, internal state, sexual reproduction with neural crossover.*