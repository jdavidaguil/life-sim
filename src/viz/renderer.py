"""Renderer module: visualises the simulation state with Matplotlib."""

from __future__ import annotations

import collections
import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.colorbar
import numpy as np

from src.core.agent import Agent
from src.core.simulation import Simulation


class Renderer:
    """Renders the current state of a :class:`~src.core.simulation.Simulation`.

    Interactive mode is enabled on construction so that ``render`` can be
    called repeatedly without blocking the simulation loop.  The figure,
    axes, ``imshow`` image, and colorbar are all created once and reused;
    each frame only updates pixel data via ``self.im.set_data()``.

    A short history of past frames is blended with an exponential decay
    factor to produce a fading-trail effect that reveals movement patterns.

    **Left panel modes** (controlled by ``show_energy``):

    * ``show_energy=False`` *(default)* — RGB trait map.
      Each cell is coloured by the mean traits of agents present:
      R=resource_weight, G=energy_awareness, B=crowd_sensitivity.
      A fading-trail effect blends the last ``trail_length`` frames.
    * ``show_energy=True`` — scalar energy heatmap (plasma colormap).

    Press **e** at runtime to toggle between modes.

    Keyboard shortcuts (active while the plot window is focused):

    * **p** — toggle pause / resume
    * **q** — quit (sets ``running`` to ``False``)
    * **e** — toggle left panel between policy colours and energy heatmap

    Attributes:
        fig: Matplotlib figure reused across frames.
        ax: Left axes — agent fading-trail panel.
        ax_r: Right axes — live resource landscape.
        im: ``AxesImage`` for agent panel (``None`` until first render).
        im_r: ``AxesImage`` for resources (``None`` until first render).
        cbar: Colorbar for agent panel when in energy mode (``None`` otherwise).
        cbar_r: Colorbar for resource panel (``None`` until first render).
        show_energy: When ``True`` the left panel shows energy; when ``False``
            it shows policy colours.
        delay: Seconds to sleep after each rendered frame.
        trail_length: Number of past frames to blend into each display frame.
        decay: Multiplicative weight decay per older frame (0 < decay < 1).
        paused: When ``True`` the renderer spins until unpaused or quit.
        running: Set to ``False`` by pressing **q**; callers should exit their
            loop when this flag is ``False``.
    """

    def __init__(
        self,
        figsize: tuple[int, int] = (12, 6),
        delay: float = 0.1,
        trail_length: int = 8,
        decay: float = 0.7,
        show_energy: bool = False,
    ) -> None:
        """Create the figure and both axes; images are attached on first render.

        Args:
            figsize: Total width and height of the figure in inches.
            delay: Seconds to sleep after each frame.  Set to 0 for max speed.
            trail_length: How many past frames to blend for the trail effect.
            decay: Weight multiplier applied per older frame (0 < decay < 1).
            show_energy: Start in energy mode instead of policy-colour mode.
        """
        plt.ion()
        from matplotlib.gridspec import GridSpec
        self.fig = plt.figure(figsize=figsize)
        # Pre-allocate dedicated colorbar columns so the main axes never
        # resize when a colorbar is added or removed.
        gs = GridSpec(
            1, 4, figure=self.fig,
            width_ratios=[10, 0.4, 10, 0.4],
            wspace=0.35,
        )
        self.ax    = self.fig.add_subplot(gs[0, 0])  # left main
        self.cax   = self.fig.add_subplot(gs[0, 1])  # left colorbar
        self.ax_r  = self.fig.add_subplot(gs[0, 2])  # right main
        self.cax_r = self.fig.add_subplot(gs[0, 3])  # right colorbar
        self.cax.set_visible(False)   # hidden until energy mode is active
        self.delay = delay
        self.trail_length = trail_length
        self.decay = decay
        self.show_energy = show_energy
        self._history: collections.deque[np.ndarray] = collections.deque(
            maxlen=trail_length
        )
        self.im: plt.AxesImage | None = None
        self.im_r: plt.AxesImage | None = None
        self.cbar: matplotlib.colorbar.Colorbar | None = None
        self.cbar_r: matplotlib.colorbar.Colorbar | None = None
        self.paused: bool = False
        self.running: bool = True
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_key(self, event) -> None:  # type: ignore[type-arg]
        """Handle key-press events from the Matplotlib canvas.

        * ``p`` — toggle ``paused``
        * ``q`` — set ``running`` to ``False`` and close the figure
        * ``e`` — toggle left panel between policy colours and energy heatmap
        """
        if event.key == "p":
            self.paused = not self.paused
        elif event.key == "q":
            self.running = False
        elif event.key == "e":
            self.show_energy = not self.show_energy
            # Force recreation of left panel imshow on next render.
            if self.im is not None:
                try:
                    self.im.remove()
                except Exception:
                    pass
                self.im = None
            # Clear the dedicated colorbar axes without touching self.ax.
            if self.cbar is not None:
                self.cax.cla()
                self.cbar = None
            self.cax.set_visible(self.show_energy)
            self._history.clear()

    # ── Frame builders ────────────────────────────────────────────────────────

    def _build_policy_frame(self, simulation: Simulation) -> np.ndarray:
        """Return an RGB array (H×W×3) encoding trait composition per cell.

        R channel: mean resource_weight of agents on cell (normalized to 0-2)
        G channel: mean energy_awareness of agents on cell (normalized to 0-2)
        B channel: mean crowd_sensitivity of agents on cell (normalized to 0-2)
        Black cells have no agents.
        """
        h, w = simulation.height, simulation.width
        frame = np.zeros((h, w, 3), dtype=np.float32)
        counts = np.zeros((h, w), dtype=np.float32)
        for agent in simulation.agents:
            t = agent.policy.traits
            frame[agent.y, agent.x, 0] += t[0]  # resource_weight -> R
            frame[agent.y, agent.x, 1] += t[3]  # energy_awareness -> G
            frame[agent.y, agent.x, 2] += t[1]  # crowd_sensitivity -> B
            counts[agent.y, agent.x] += 1
        mask = counts > 0
        for c in range(3):
            frame[:, :, c][mask] /= counts[mask]
        frame /= 2.0  # normalize: traits sampled up to ~2.0
        return np.clip(frame, 0.0, 1.0)

    def _build_energy_frame(self, simulation: Simulation) -> np.ndarray:
        """Return a scalar (H×W) array of normalised energy values."""
        frame = np.zeros((simulation.height, simulation.width), dtype=np.float32)
        for agent in simulation.agents:
            frame[agent.y, agent.x] = agent.energy / Agent.INITIAL_ENERGY
        return frame

    def _blend_history(self) -> np.ndarray:
        """Blend the frame history with exponential decay into a single frame.

        A sqrt gamma correction is applied to the normalised result so that
        lightly-occupied cells are clearly visible rather than near-black.
        """
        ref = self._history[-1]
        # Purge stale frames from a previous mode whose shape no longer matches.
        compatible = [f for f in self._history if f.shape == ref.shape]
        blended = np.zeros_like(ref, dtype=np.float32)
        n = len(compatible)
        for age, past in enumerate(reversed(compatible)):
            blended += past * (self.decay ** age)
        max_weight = sum(self.decay ** i for i in range(n))
        return np.sqrt(np.clip(blended / max_weight, 0.0, 1.0))

    # ── Main render call ──────────────────────────────────────────────────────

    def render(self, simulation: Simulation, step: int) -> None:
        """Draw the agent panel (left) and resource landscape (right).

        Args:
            simulation: The simulation whose state should be drawn.
            step: The step number displayed in the figure title.
        """
        from src.core.grid import Grid  # local import avoids circular dep

        # ── Build left-panel frame and append to history ───────────────────
        if self.show_energy:
            current_frame = self._build_energy_frame(simulation)
        else:
            current_frame = self._build_policy_frame(simulation)
        self._history.append(current_frame)
        blended = self._blend_history()

        # ── Resource frame ─────────────────────────────────────────────────
        res_norm = (simulation.grid.resources / Grid.MAX_RESOURCE).astype(np.float32)

        # ── Summary stats ──────────────────────────────────────────────────
        population = simulation.agent_count()
        avg_energy = (
            sum(a.energy for a in simulation.agents) / population
            if population > 0 else 0.0
        )

        # ── Left panel: recreate when im is None (first render or mode toggle) ──
        if self.im is None:
            if self.show_energy:
                self.im = self.ax.imshow(
                    blended, origin="upper", cmap="plasma",
                    vmin=0.0, vmax=1.0, interpolation="nearest",
                )
                self.cax.set_visible(True)
                self.cbar = self.fig.colorbar(self.im, cax=self.cax)
                self.cbar.set_label("Agent energy (norm)", fontsize=9)
                self.ax.set_title("Agents — energy", fontsize=10)
            else:
                self.im = self.ax.imshow(
                    blended, origin="upper",
                    vmin=0.0, vmax=1.0, interpolation="nearest",
                )
                self.cax.set_visible(False)
                self.ax.set_title("Agents — traits (R=rw, G=ea, B=cs)", fontsize=10)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
        else:
            self.im.set_data(blended)

        # ── Right panel: create once, then just update data ────────────────
        if self.im_r is None:
            self.im_r = self.ax_r.imshow(
                res_norm, origin="upper", cmap="YlGn",
                vmin=0.0, vmax=1.0, interpolation="nearest",
            )
            self.cbar_r = self.fig.colorbar(self.im_r, cax=self.cax_r)
            self.cbar_r.set_label("Resource (norm)", fontsize=9)
            self.ax_r.set_title("Resources", fontsize=10)
            self.ax_r.set_xticks([])
            self.ax_r.set_yticks([])
        else:
            self.im_r.set_data(res_norm)

        self.fig.suptitle(
            f"Step {step}  |  Pop: {population}  |  "
            f"Avg energy: {avg_energy:.1f}  |  [e] toggle mode",
            fontsize=9,
        )
        self.fig.canvas.draw_idle()
        plt.pause(0.01)

        # Pause loop: spin cheaply until resumed or quit.
        while self.paused and self.running:
            plt.pause(0.05)

        if self.delay > 0:
            time.sleep(self.delay)

    def close(self) -> None:
        """Release the Matplotlib figure resources."""
        plt.close(self.fig)
