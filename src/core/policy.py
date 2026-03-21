"""Policy module: defines the TraitPolicy genome for agents."""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.agent import Agent
    from src.core.grid import Grid

MUTATION_SIGMA: float = 0.05
NEURAL_MUTATION_SIGMA: float = 0.01
BASELINE_MODE = "baseline"
RICHER_MODE = "richer"
NEURAL_GENOME_SIZE: int = 224  # 18*8+8 + 8*8+8
STATEFUL_STATE_SIZE: int = 4
STATEFUL_GENOME_SIZE: int = 292  # (18+4)*8+8 + 8*(8+4)+12


class TraitPolicy:
    """Holds four continuous traits that govern agent movement decisions.

    Traits:
        resource_weight:   base attraction toward resource-rich cells.
        crowd_sensitivity: penalty weight for crowded cells.
        noise:             random exploration coefficient.
        energy_awareness:  scales extra resource attraction when energy is low.

    Supports two scoring modes:

    `baseline`:
        effective_rw = resource_weight + energy_awareness * (1 - energy / MAX_ENERGY)
        score = effective_rw * resource - crowd_sensitivity * crowding + noise * rng.random()

    `richer`:

        effective_rw = resource_weight + energy_awareness * (1 - energy / MAX_ENERGY)
        gradient     = resource(nx,ny) - resource(agent.x, agent.y)
        local_density = mean occupancy of the 8 cells around the agent
        relative_energy = (agent.energy - pop_mean_energy) / MAX_ENERGY

        score = effective_rw * resource
              - crowd_sensitivity * crowding          # immediate cell crowding
              + noise * rng.random()                  # exploration
              + resource_weight * gradient            # directional resource signal
              - crowd_sensitivity * local_density     # neighbourhood density penalty
              + energy_awareness * relative_energy    # relative-energy drive

    Parameters for decide():
        agent:           the calling Agent instance.
        grid:            Grid providing resource and neighbour queries.
        occupancy:       dict mapping (x, y) -> agent count.
        rng:             numpy random Generator for the noise term.
        pop_mean_energy: population mean energy this step (default 0.0);
                         used to compute the relative-energy term.
    """

    MAX_ENERGY: float = 20.0  # mirrors Agent.INITIAL_ENERGY

    def __init__(
        self,
        traits: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        mode: str = RICHER_MODE,
    ) -> None:
        if mode not in {BASELINE_MODE, RICHER_MODE}:
            raise ValueError(f"unknown policy mode: {mode}")
        self.mode = mode
        if traits is None:
            if rng is None:
                raise ValueError("rng required when traits is None")
            self.traits = np.array(
                [
                    rng.uniform(0.5, 1.5),
                    rng.uniform(0.0, 1.0),
                    rng.uniform(0.0, 0.5),
                    rng.uniform(0.0, 1.0),
                ],
                dtype=np.float32,
            )
        else:
            self.traits = traits

    @property
    def resource_weight(self) -> float:
        return float(self.traits[0])

    @property
    def crowd_sensitivity(self) -> float:
        return float(self.traits[1])

    @property
    def noise(self) -> float:
        return float(self.traits[2])

    @property
    def energy_awareness(self) -> float:
        return float(self.traits[3])

    def decide(
        self,
        agent: "Agent",
        grid: "Grid",
        occupancy: dict,
        rng: np.random.Generator,
        pop_mean_energy: float = 0.0,
    ) -> tuple[int, int]:
        neighbors = grid.get_neighbors(agent.x, agent.y)
        if not neighbors:
            return (0, 0)
        best_score = float("-inf")
        best_nx, best_ny = neighbors[0]
        for nx, ny in neighbors:
            resource = grid.get_resource(nx, ny)
            crowding = occupancy.get((nx, ny), 0)

            # Existing terms
            effective_rw = self.resource_weight + self.energy_awareness * (
                1.0 - agent.energy / self.MAX_ENERGY
            )
            effective_rw = max(0.0, effective_rw)

            if self.mode == BASELINE_MODE:
                score = (
                    effective_rw * resource
                    - self.crowd_sensitivity * crowding
                    + self.noise * float(rng.random())
                )
            else:
                gradient = resource - grid.get_resource(agent.x, agent.y)
                local_density = sum(
                    occupancy.get((agent.x + dx, agent.y + dy), 0)
                    for dy in (-1, 0, 1)
                    for dx in (-1, 0, 1)
                    if not (dx == 0 and dy == 0)
                ) / 8.0
                relative_energy = (
                    agent.energy - pop_mean_energy
                ) / self.MAX_ENERGY

                score = (
                    effective_rw * resource
                    - self.crowd_sensitivity * crowding
                    + self.noise * float(rng.random())
                    + self.resource_weight * gradient
                    - self.crowd_sensitivity * local_density
                    + self.energy_awareness * relative_energy
                )

            if score > best_score:
                best_score = score
                best_nx, best_ny = nx, ny
        return (best_nx - agent.x, best_ny - agent.y)

    def mutate(self, rng: np.random.Generator) -> TraitPolicy:
        new_traits = self.traits + rng.normal(0, MUTATION_SIGMA, size=4).astype(np.float32)
        new_traits = np.maximum(0.0, new_traits)
        return TraitPolicy(traits=new_traits, mode=self.mode)

    def __repr__(self) -> str:
        return (
            f"TraitPolicy(rw={self.resource_weight:.2f}, "
            f"cs={self.crowd_sensitivity:.2f}, "
            f"noise={self.noise:.2f}, "
            f"ea={self.energy_awareness:.2f}, "
            f"mode={self.mode})"
        )


def _make_warm_start_genome(
    rw: float,
    cs: float,
    rng: np.random.Generator,
    noise_scale: float = 0.01,
) -> np.ndarray:
    """Build a genome that implements the Phase 3 linear scorer.

    W1: hidden node i responds to resource[i] - crowd[i]
        W1[i, i]   = rw   (resource direction i -> hidden i)
        W1[i+8, i] = -cs  (crowd direction i -> hidden i)
        All other W1 entries = small noise
    b1: zeros
    W2: identity (hidden i -> output i)
        W2[i, i] = 1.0
        All other W2 entries = small noise
    b2: zeros

    Small Gaussian noise added to all weights so mutation
    has something to work with from the start.
    """
    genome = rng.normal(0.0, noise_scale,
                        size=NEURAL_GENOME_SIZE).astype(np.float32)

    # W1: shape (18, 8), indices 0:144
    W1 = genome[0:144].reshape(18, 8)
    for i in range(8):
        W1[i, i]     = rw    # resource[i] -> hidden[i]
        W1[i + 8, i] = -cs   # crowd[i]    -> hidden[i]
    genome[0:144] = W1.flatten()

    # b1: indices 144:152 — leave as noise

    # W2: shape (8, 8), indices 152:216
    W2 = genome[152:216].reshape(8, 8)
    for i in range(8):
        W2[i, i] = 1.0       # hidden[i] -> output[i]
    genome[152:216] = W2.flatten()

    # b2: indices 216:224 — leave as noise

    return genome


class NeuralPolicy:
    """Neural genome policy — 4 inputs → 8 hidden (ReLU) → 8 outputs (softmax).

    The genome is a flat numpy array of 112 weights:
        W1: shape (18, 8) — input to hidden, indices 0:144
        b1: shape (8,)    — hidden biases,   indices 144:152
        W2: shape (8, 8)  — hidden to output, indices 152:216
        b2: shape (8,)    — output biases,    indices 216:224

    Input vector (18 values):
      indices 0:8   — resource at each neighbor (NW,N,NE,W,E,SW,S,SE)
                    normalized by MAX_RESOURCE (5.0)
      indices 8:16  — crowding at each neighbor
                    normalized by 8.0 (max possible agents)
      index 16      — relative energy / MAX_ENERGY
      index 17      — current resource / MAX_RESOURCE

    Output is a probability distribution over 8 compass directions:
        [NW, N, NE, W, E, SW, S, SE]
        as (dx, dy) deltas:
        [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]

    Initialization: weights drawn from N(0, 0.1) — near-zero so
    selection builds structure from scratch.
    """

    DIRECTIONS: list[tuple[int, int]] = [
        (-1, -1), (0, -1), (1, -1),
        (-1,  0),          (1,  0),
        (-1,  1), (0,  1), (1,  1),
    ]

    mode: str = "neural"

    def __init__(
        self,
        genome: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        warm_start: bool = False,
        rw: float = 2.0,
        cs: float = 1.5,
    ) -> None:
        if genome is not None:
            self.genome = genome.astype(np.float32)
        else:
            if rng is None:
                raise ValueError("rng required when genome is None")
            if warm_start:
                self.genome = _make_warm_start_genome(
                    rw=rw, cs=cs, rng=rng
                )
            else:
                self.genome = rng.normal(
                    0.0, 0.1, size=NEURAL_GENOME_SIZE
                ).astype(np.float32)

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Run forward pass. x shape: (18,). Returns probs shape: (8,)."""
        W1 = self.genome[0:144].reshape(18, 8)
        b1 = self.genome[144:152]
        W2 = self.genome[152:216].reshape(8, 8)
        b2 = self.genome[216:224]
        h = np.maximum(0.0, x @ W1 + b1)
        logits = h @ W2 + b2
        logits -= logits.max()
        exp = np.exp(logits)
        return exp / exp.sum()

    def decide(
        self,
        agent,
        grid,
        occupancy: dict,
        rng: np.random.Generator,
        pop_mean_energy: float = 0.0,
    ) -> tuple[int, int]:
        neighbors = grid.get_neighbors(agent.x, agent.y)
        if not neighbors:
            return (0, 0)

        # Build per-direction input vector (18 values)
        # Fixed direction order matches DIRECTIONS list:
        # NW, N, NE, W, E, SW, S, SE
        resource_inputs = np.zeros(8, dtype=np.float32)
        crowd_inputs    = np.zeros(8, dtype=np.float32)

        for i, (dx, dy) in enumerate(self.DIRECTIONS):
            nx, ny = agent.x + dx, agent.y + dy
            if grid.is_inside(nx, ny):
                resource_inputs[i] = grid.get_resource(nx, ny) / 5.0
                crowd_inputs[i]    = occupancy.get((nx, ny), 0) / 8.0
            # else: leave as 0.0 (wall = empty)

        relative_energy = (agent.energy - pop_mean_energy) / 20.0
        current_res     = grid.get_resource(agent.x, agent.y) / 5.0

        x = np.concatenate([
            resource_inputs,
            crowd_inputs,
            [relative_energy, current_res],
        ]).astype(np.float32)

        probs = self._forward(x)

        # Sample directly over DIRECTIONS — no renormalization needed
        # because we know which directions are valid
        valid_probs = []
        valid_neighbors = []
        for i, (dx, dy) in enumerate(self.DIRECTIONS):
            nx, ny = agent.x + dx, agent.y + dy
            if grid.is_inside(nx, ny):
                valid_probs.append(probs[i])
                valid_neighbors.append((nx, ny))

        if not valid_neighbors:
            return (0, 0)

        # Renormalize over valid neighbors
        valid_probs = np.array(valid_probs, dtype=np.float32)
        valid_probs /= valid_probs.sum()

        # Sample from distribution
        chosen_idx = int(rng.choice(len(valid_neighbors), p=valid_probs))
        nx, ny = valid_neighbors[chosen_idx]
        return (nx - agent.x, ny - agent.y)

    def mutate(self, rng: np.random.Generator) -> "NeuralPolicy":
        new_genome = self.genome + rng.normal(
            0.0, NEURAL_MUTATION_SIGMA,
            size=NEURAL_GENOME_SIZE,
        ).astype(np.float32)
        return NeuralPolicy(genome=new_genome)

    def __repr__(self) -> str:
        return f"NeuralPolicy(genome_size={len(self.genome)}, norm={float(np.linalg.norm(self.genome)):.2f})"


def _make_stateful_warm_start(
    rw: float,
    cs: float,
    rng: np.random.Generator,
    noise_scale: float = 0.01,
) -> np.ndarray:
    """Warm-start genome for StatefulNeuralPolicy.

    Observation weights (first 18 rows of W1) same as
    Phase 4 warm start. State input weights (rows 18:22)
    initialized to zero — state has no meaning yet.
    W2 move weights (columns 0:8) same as Phase 4 identity.
    W2 state weights (columns 8:12) small noise — free to learn.
    """
    genome = rng.normal(
        0.0, noise_scale,
        size=STATEFUL_GENOME_SIZE,
    ).astype(np.float32)

    # W1: shape (22, 8), indices 0:176
    W1 = genome[0:176].reshape(22, 8)
    for i in range(8):
        W1[i, i]     =  rw   # resource[i] -> hidden[i]
        W1[i + 8, i] = -cs   # crowd[i]    -> hidden[i]
    # rows 18:22 (state inputs) stay as small noise
    genome[0:176] = W1.flatten()

    # W2: shape (8, 12), indices 184:280
    W2 = genome[184:280].reshape(8, 12)
    for i in range(8):
        W2[i, i] = 1.0       # hidden[i] -> move[i]
    # columns 8:12 (state outputs) stay as small noise
    genome[184:280] = W2.flatten()

    return genome


class StatefulNeuralPolicy:
    """Neural policy with persistent state vector across steps.

    Genome encodes a recurrent-style network:
        [observations(18) + state(4)] -> hidden(8, ReLU)
        hidden(8) -> move_logits(8, softmax) + new_state(4, tanh)

    Genome layout (292 weights):
        W1: shape (22, 8)  indices 0:176   input+state -> hidden
        b1: shape (8,)     indices 176:184 hidden biases
        W2: shape (8, 12)  indices 184:280 hidden -> move+state
        b2: shape (12,)    indices 280:292 output biases

    Move outputs: indices 0:8 of W2 output (softmax)
    State outputs: indices 8:12 of W2 output (tanh)

    State vector persists across steps, initialized to zeros.
    Directions order: NW, N, NE, W, E, SW, S, SE
    """

    DIRECTIONS: list[tuple[int, int]] = [
        (-1, -1), (0, -1), (1, -1),
        (-1,  0),          (1,  0),
        (-1,  1), (0,  1), (1,  1),
    ]

    mode: str = "stateful"

    def __init__(
        self,
        genome: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        warm_start: bool = False,
        rw: float = 2.0,
        cs: float = 1.5,
    ) -> None:
        if genome is not None:
            self.genome = genome.astype(np.float32)
        else:
            if rng is None:
                raise ValueError("rng required when genome is None")
            if warm_start:
                self.genome = _make_stateful_warm_start(
                    rw=rw, cs=cs, rng=rng
                )
            else:
                self.genome = rng.normal(
                    0.0, 0.1, size=STATEFUL_GENOME_SIZE
                ).astype(np.float32)

        # State vector — zeros at birth, persists across steps
        self.state = np.zeros(STATEFUL_STATE_SIZE, dtype=np.float32)

    def _forward(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass.
        x shape: (18,) observations.
        Returns (move_probs(8,), new_state(4,)).
        """
        W1 = self.genome[0:176].reshape(22, 8)
        b1 = self.genome[176:184]
        W2 = self.genome[184:280].reshape(8, 12)
        b2 = self.genome[280:292]

        # Concatenate observations with current state
        x_full = np.concatenate([x, self.state])  # shape (22,)

        # Hidden layer
        h = np.maximum(0.0, x_full @ W1 + b1)     # ReLU

        # Output layer
        out = h @ W2 + b2                           # shape (12,)

        # Move logits -> softmax
        move_logits = out[:8]
        move_logits -= move_logits.max()
        exp = np.exp(move_logits)
        move_probs = exp / exp.sum()

        # State update -> tanh
        new_state = np.tanh(out[8:12]).astype(np.float32)

        return move_probs, new_state

    def decide(
        self,
        agent,
        grid,
        occupancy: dict,
        rng: np.random.Generator,
        pop_mean_energy: float = 0.0,
    ) -> tuple[int, int]:
        neighbors = grid.get_neighbors(agent.x, agent.y)
        if not neighbors:
            return (0, 0)

        # Build observation vector (18 values) — same as NeuralPolicy
        resource_inputs = np.zeros(8, dtype=np.float32)
        crowd_inputs    = np.zeros(8, dtype=np.float32)
        for i, (dx, dy) in enumerate(self.DIRECTIONS):
            nx, ny = agent.x + dx, agent.y + dy
            if grid.is_inside(nx, ny):
                resource_inputs[i] = grid.get_resource(nx, ny) / 5.0
                crowd_inputs[i]    = occupancy.get((nx, ny), 0) / 8.0
        relative_energy = (agent.energy - pop_mean_energy) / 20.0
        current_res     = grid.get_resource(agent.x, agent.y) / 5.0
        x = np.concatenate([
            resource_inputs, crowd_inputs,
            [relative_energy, current_res],
        ]).astype(np.float32)

        # Forward pass — updates state
        move_probs, new_state = self._forward(x)
        self.state = new_state  # persist state for next step

        # Sample from valid neighbors
        valid_probs = []
        valid_neighbors = []
        for i, (dx, dy) in enumerate(self.DIRECTIONS):
            nx, ny = agent.x + dx, agent.y + dy
            if grid.is_inside(nx, ny):
                valid_probs.append(move_probs[i])
                valid_neighbors.append((nx, ny))

        if not valid_neighbors:
            return (0, 0)

        valid_probs = np.array(valid_probs, dtype=np.float32)
        valid_probs /= valid_probs.sum()
        chosen_idx = int(rng.choice(
            len(valid_neighbors), p=valid_probs
        ))
        nx, ny = valid_neighbors[chosen_idx]
        return (nx - agent.x, ny - agent.y)

    def mutate(
        self, rng: np.random.Generator
    ) -> "StatefulNeuralPolicy":
        new_genome = self.genome + rng.normal(
            0.0, NEURAL_MUTATION_SIGMA,
            size=STATEFUL_GENOME_SIZE,
        ).astype(np.float32)
        child = StatefulNeuralPolicy(genome=new_genome)
        # Child starts with zero state — blank memory at birth
        return child

    def __repr__(self) -> str:
        return (
            f"StatefulNeuralPolicy("
            f"norm={float(np.linalg.norm(self.genome)):.2f}, "
            f"state={np.round(self.state, 2)})"
        )
