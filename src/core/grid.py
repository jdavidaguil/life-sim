"""Grid module: defines the 2-D spatial structure of the simulation."""

from __future__ import annotations

from typing import Optional

import numpy as np


class Grid:
    """A 2-D grid with spatial hotspots that shape the resource landscape.

    A number of hotspot centres are chosen at construction time.  Each cell's
    initial resource level and per-step regeneration rate decay exponentially
    with distance from the nearest hotspot.  Hotspots drift slowly over time,
    forcing agents to adapt to a shifting resource landscape.

    Attributes:
        width: Number of columns.
        height: Number of rows.
        cells: NumPy array of shape (height, width) holding cell values.
        resources: Float array of shape (height, width) with resource levels.
        regen_rates: Float array of shape (height, width); per-cell regen rate.
        hotspot_xs: Column positions of current hotspot centres.
        hotspot_ys: Row positions of current hotspot centres.
        MAX_RESOURCE: Hard cap on resource level per cell.
        NUM_HOTSPOTS: Number of resource hotspots.
        HOTSPOT_REGEN: Regeneration rate at the centre of a hotspot.
        BASE_REGEN: Minimum regeneration rate far from all hotspots.
        HOTSPOT_SIGMA: Gaussian spread of each hotspot (in grid cells).
        DRIFT_STEP: Maximum single-axis displacement per step in the random walk.
    """

    MAX_RESOURCE: float = 5.0
    NUM_HOTSPOTS: int = 4
    HOTSPOT_REGEN: float = 1.2
    BASE_REGEN: float = 0.005
    HOTSPOT_SIGMA: float = 4.0
    DRIFT_STEP: int = 1          # max single-step displacement per axis (random walk)
    # Environmental noise: Poisson-distributed sudden resource shocks per step.
    NOISE_RATE: float = 3.0      # expected number of noise events per step
    NOISE_MAGNITUDE: float = 2.0 # max absolute resource change per event

    def __init__(
        self,
        width: int,
        height: int,
        rng: Optional[np.random.Generator] = None,
        drift_step: Optional[int] = None,
        noise_rate: Optional[float] = None,
        noise_magnitude: Optional[float] = None,
    ) -> None:
        """Initialise the grid, place hotspots, and fill resources accordingly.

        Args:
            width: Number of columns.
            height: Number of rows.
            rng: NumPy random generator used for all stochastic decisions.
                A fresh default generator is created when *rng* is ``None``.
            drift_step: Override for DRIFT_STEP (max single-step displacement).
            noise_rate: Override for NOISE_RATE (expected noise events per step).
            noise_magnitude: Override for NOISE_MAGNITUDE (max resource change per event).
        """
        if rng is None:
            rng = np.random.default_rng()

        # Allow per-instance override of environment parameters.
        if drift_step is not None:
            self.DRIFT_STEP = drift_step
        if noise_rate is not None:
            self.NOISE_RATE = noise_rate
        if noise_magnitude is not None:
            self.NOISE_MAGNITUDE = noise_magnitude

        self._rng = rng
        self.width = width
        self.height = height

        # Pick initial hotspot centres.
        self.hotspot_xs: np.ndarray = rng.integers(0, width,  size=self.NUM_HOTSPOTS)
        self.hotspot_ys: np.ndarray = rng.integers(0, height, size=self.NUM_HOTSPOTS)

        self._ys, self._xs = np.mgrid[0:height, 0:width]  # cached coord grids
        self.regen_rates: np.ndarray = np.empty((height, width), dtype=np.float32)
        self.resources: np.ndarray = np.empty((height, width), dtype=np.float32)
        self._recompute_influence(seed_resources=True)
        self.last_shocked: list[tuple[int, int]] = []

    def _recompute_influence(self, seed_resources: bool = False) -> None:
        """Rebuild ``regen_rates`` from the current hotspot positions.

        Optionally re-seeds ``resources`` (used at initialisation only).
        During normal drift, existing resource levels are preserved so that
        agents already on a departing hotspot don't instantly lose their food.

        Args:
            seed_resources: When ``True`` initialise resources from the
                influence map.  When ``False`` leave ``resources`` unchanged.
        """
        influence = np.zeros((self.height, self.width), dtype=np.float32)
        sigma2 = 2.0 * self.HOTSPOT_SIGMA ** 2
        for hx, hy in zip(self.hotspot_xs, self.hotspot_ys):
            d2 = (self._xs - hx) ** 2 + (self._ys - hy) ** 2
            influence = np.maximum(influence, np.exp(-d2 / sigma2).astype(np.float32))

        self.regen_rates[:] = (
            self.BASE_REGEN + (self.HOTSPOT_REGEN - self.BASE_REGEN) * influence
        )

        if seed_resources:
            self.resources[:] = np.clip(
                self.MAX_RESOURCE * influence
                + self._rng.uniform(0.0, 0.5, size=(self.height, self.width)),
                0.0,
                self.MAX_RESOURCE,
            ).astype(np.float32)

    def update_hotspots(self) -> None:
        """Advance each hotspot one step in a continuous random walk.

        Every call, each hotspot independently shifts its centre by a vector
        whose x and y components are drawn from ``{-DRIFT_STEP, 0, +DRIFT_STEP}``
        (nine equally likely outcomes).  The new position is clamped to grid
        bounds, and ``regen_rates`` is recomputed only when at least one centre
        actually moved.  ``resources`` is left intact so agents on a departing
        hotspot don't suffer an instant food cliff — the rich area fades
        naturally as its regen rate drops.

        Calling this every simulation step produces smooth, continuous drift
        that keeps clusters from locking onto fixed locations.
        """
        step = self.DRIFT_STEP
        # Draw all deltas at once: shape (NUM_HOTSPOTS, 2), values in {-step,0,+step}.
        deltas = self._rng.integers(-step, step + 1, size=(self.NUM_HOTSPOTS, 2))
        new_xs = np.clip(self.hotspot_xs + deltas[:, 0], 0, self.width  - 1)
        new_ys = np.clip(self.hotspot_ys + deltas[:, 1], 0, self.height - 1)
        if not (np.array_equal(new_xs, self.hotspot_xs)
                and np.array_equal(new_ys, self.hotspot_ys)):
            self.hotspot_xs = new_xs
            self.hotspot_ys = new_ys
            self._recompute_influence(seed_resources=False)

    def is_inside(self, x: int, y: int) -> bool:
        """Return True if (x, y) falls within the grid boundaries."""
        return 0 <= x < self.width and 0 <= y < self.height

    def get_neighbors(self, x: int, y: int) -> list[tuple[int, int]]:
        """Return the valid Moore-neighbourhood coordinates around (x, y).

        The Moore neighbourhood consists of the 8 cells that share a corner or
        an edge with the given cell.  Only coordinates that lie inside the grid
        are returned.

        Args:
            x: Column of the centre cell.
            y: Row of the centre cell.

        Returns:
            A list of ``(nx, ny)`` tuples, excluding (x, y) itself.
        """
        neighbors: list[tuple[int, int]] = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self.is_inside(nx, ny):
                    neighbors.append((nx, ny))
        return neighbors

    def get_resource(self, x: int, y: int) -> float:
        """Return the current resource level at (x, y)."""
        return float(self.resources[y, x])

    def consume_resource(self, x: int, y: int, amount: float) -> float:
        """Remove up to *amount* resource from (x, y) and return what was taken.

        The cell can never go below 0.  The returned value is the actual amount
        consumed, which may be less than *amount* if the cell is nearly empty.

        Args:
            x: Column of the target cell.
            y: Row of the target cell.
            amount: Desired quantity to consume.

        Returns:
            The quantity actually consumed.
        """
        available = float(self.resources[y, x])
        consumed = min(available, amount)
        self.resources[y, x] = available - consumed
        return consumed

    def regenerate(self, pressure: Optional[np.ndarray] = None) -> None:
        """Increase each cell by its individual regen rate, capped at ``MAX_RESOURCE``.

        When *pressure* is provided (an array of agent counts per cell), each
        cell's effective rate is scaled down by ``rate / (1 + num_agents)``.
        Cells grazed by many agents recover much more slowly, pushing agents to
        spread out rather than pile onto the same hotspot indefinitely.

        Args:
            pressure: Optional float/int array of shape ``(height, width)``
                holding the number of agents currently on each cell.  Passing
                ``None`` (the default) uses the base regen rates unchanged.
        """
        if pressure is None:
            effective = self.regen_rates
        else:
            effective = self.regen_rates / (1.0 + pressure)
        self.resources = np.clip(
            self.resources + effective,
            0.0,
            self.MAX_RESOURCE,
        ).astype(np.float32)

    def apply_noise(self) -> None:
        """Apply random resource shocks and record which cells were hit."""
        self.last_shocked = []
        n_events = int(self._rng.poisson(self.NOISE_RATE))
        if n_events == 0:
            return
        xs = self._rng.integers(0, self.width,  size=n_events)
        ys = self._rng.integers(0, self.height, size=n_events)
        deltas = self._rng.uniform(
            -self.NOISE_MAGNITUDE, self.NOISE_MAGNITUDE,
            size=n_events
        ).astype(np.float32)
        for x, y, delta in zip(xs, ys, deltas):
            self.resources[y, x] = float(
                np.clip(self.resources[y, x] + delta, 0.0, self.MAX_RESOURCE)
            )
            self.last_shocked.append((int(x), int(y)))

    def __repr__(self) -> str:  # pragma: no cover
        return f"Grid(width={self.width}, height={self.height})"
