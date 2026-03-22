"""Phase: apply random resource shocks to the grid."""

from __future__ import annotations

from src.core.state import SimState


def noise(state: SimState) -> None:
    """Trigger Poisson-distributed resource noise events on the grid."""
    state.grid.apply_noise()
