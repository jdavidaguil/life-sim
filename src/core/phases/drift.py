"""Phase: advance the hotspot random walk."""

from __future__ import annotations

from src.core.state import SimState


def drift(state: SimState) -> None:
    """Step each resource hotspot one position in its random walk."""
    state.grid.update_hotspots()
