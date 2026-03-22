"""Default simulation phase list in canonical execution order."""

from src.core.phases.move import move
from src.core.phases.consume import consume
from src.core.phases.decay import decay
from src.core.phases.reproduce import reproduce
from src.core.phases.die import die
from src.core.phases.regenerate import regenerate
from src.core.phases.noise import noise
from src.core.phases.drift import drift

DEFAULT_PHASES = [move, consume, decay, reproduce, die, regenerate, noise, drift]

__all__ = [
    "move",
    "consume",
    "decay",
    "reproduce",
    "die",
    "regenerate",
    "noise",
    "drift",
    "DEFAULT_PHASES",
]
