"""Phase: remove agents whose energy has dropped to zero or below."""

from __future__ import annotations

from src.core.state import SimState


def die(state: SimState) -> None:
    """Filter out dead agents from the roster."""
    state.agents = [agent for agent in state.agents if agent.energy > 0]
