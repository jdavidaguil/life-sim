"""Agent module: defines a single actor that moves on the grid."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.policy import Policy


class Agent:
    """An entity that occupies a cell and can act each simulation step.

    Attributes:
        id: Unique integer identifier.
        x: Current column position.
        y: Current row position.
        energy: Remaining energy; agent is removed when this reaches zero.
        alive: Whether the agent is still active.
        last_dx: Column component of the last move taken (0 if none yet).
        last_dy: Row component of the last move taken (0 if none yet).
        policy: Movement strategy; defaults to :class:`~src.core.policy.GreedyPolicy`.
    """

    INITIAL_ENERGY: float = 20.0

    def __init__(self, id: int, x: int, y: int, policy=None) -> None:  # noqa: A002
        """Create an agent at the given grid position.

        Args:
            id: Unique identifier for this agent.
            x: Initial column position.
            y: Initial row position.
            policy: Movement policy instance.  When *None* (default) a
                :class:`~src.core.policy.GreedyPolicy` is used so that
                existing simulation code that sets ``agent.policy`` after
                construction continues to work unchanged.
        """
        # Import here to avoid circular dependency at module load time.
        from src.core.policy import GreedyPolicy

        self.id = id
        self.x = x
        self.y = y
        self.energy: float = self.INITIAL_ENERGY
        self.alive: bool = True
        self.last_dx: int = 0
        self.last_dy: int = 0
        self.policy: "Policy" = policy if policy is not None else GreedyPolicy()

    def move(self, dx: int, dy: int) -> None:
        """Translate the agent's position by (dx, dy).

        No bounds checking is performed here; callers are responsible for
        ensuring the resulting position is valid for the target grid.

        Args:
            dx: Column delta (positive → right).
            dy: Row delta (positive → down).
        """
        self.x += dx
        self.y += dy

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Agent(id={self.id}, x={self.x}, y={self.y}, "
            f"energy={self.energy:.1f}, alive={self.alive})"
        )
