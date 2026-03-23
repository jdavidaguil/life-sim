"""Environmental conditions used across phase experiments.

Defines the Condition dataclass and the canonical four-condition matrix
(A–D) that combines drift speed × noise intensity.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Condition:
    name: str
    drift_step: int
    noise_rate: float
    noise_magnitude: float


CONDITIONS: list[Condition] = [
    Condition("A — Baseline",        drift_step=1, noise_rate=3.0, noise_magnitude=2.0),
    Condition("B — Fast drift",      drift_step=3, noise_rate=3.0, noise_magnitude=2.0),
    Condition("C — Boom/bust",       drift_step=1, noise_rate=8.0, noise_magnitude=4.0),
    Condition("D — Combined",        drift_step=3, noise_rate=8.0, noise_magnitude=4.0),
    Condition("E — Steep landscape", drift_step=1, noise_rate=0.5, noise_magnitude=1.0),
]
