"""Whole-layer crossover and noisy warm-start utilities for NeuralPolicy genomes.

Genome layout (224 weights):
    W1  indices   0:144   shape (18, 8)
    b1  indices 144:152   shape  (8,)
    W2  indices 152:216   shape  (8, 8)
    b2  indices 216:224   shape  (8,)
"""

from __future__ import annotations

import numpy as np

from src.core.policy import NeuralPolicy

NEURAL_GENOME_SIZE: int = 224

# (start, stop) slice for each layer in the flat genome.
_LAYER_SLICES: list[tuple[int, int]] = [
    (0,   144),  # W1
    (144, 152),  # b1
    (152, 216),  # W2
    (216, 224),  # b2
]

MUTATION_SIGMA: float = 0.05


def crossover_neural(
    parent_a: NeuralPolicy,
    parent_b: NeuralPolicy,
    rng: np.random.Generator,
) -> NeuralPolicy:
    """Whole-layer uniform crossover with Gaussian mutation.

    For each layer independently a fair coin selects which parent's layer
    is inherited.  Gaussian noise (sigma=0.05) is then added to every weight.
    Neither parent's genome is modified.

    Parameters
    ----------
    parent_a, parent_b:
        ``NeuralPolicy`` instances whose ``.genome`` arrays are 224 floats.
    rng:
        ``numpy.random.Generator`` for coin flips and mutation noise.

    Returns
    -------
    A new ``NeuralPolicy`` built from the crossed-over, mutated genome.
    """
    child_genome = np.empty(NEURAL_GENOME_SIZE, dtype=np.float32)
    for start, stop in _LAYER_SLICES:
        src = parent_a.genome if rng.integers(0, 2) == 0 else parent_b.genome
        child_genome[start:stop] = src[start:stop]
    child_genome += rng.normal(0.0, MUTATION_SIGMA, size=NEURAL_GENOME_SIZE).astype(np.float32)
    return NeuralPolicy(genome=child_genome)


def warm_start_noisy(rng: np.random.Generator, sigma: float = 0.1) -> NeuralPolicy:
    """Return a warm-started ``NeuralPolicy`` with additive Gaussian noise.

    Builds a warm-start genome via ``NeuralPolicy(warm_start=True, rw=2.0,
    cs=1.5)`` then perturbs every weight by ``rng.normal(0, sigma)``.

    Parameters
    ----------
    rng:
        ``numpy.random.Generator`` used for the warm start and the noise.
    sigma:
        Standard deviation of the additive Gaussian noise (default 0.1).

    Returns
    -------
    A new ``NeuralPolicy`` with a noisy warm-started genome.
    """
    base = NeuralPolicy(rng=rng, warm_start=True, rw=2.0, cs=1.5)
    noisy_genome = (
        base.genome
        + rng.normal(0.0, sigma, size=NEURAL_GENOME_SIZE).astype(np.float32)
    )
    return NeuralPolicy(genome=noisy_genome)
