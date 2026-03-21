"""Renderer module: three-panel desktop visualisation."""

from __future__ import annotations

import collections
import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colorbar
import numpy as np

from src.core.agent import Agent
from src.core.simulation import Simulation


# Trait display config
TRAIT_KEYS   = ["resource_weight", "crowd_sensitivity", "noise", "energy_awareness"]
TRAIT_SHORT  = ["rw", "cs", "noise", "ea"]
TRAIT_COLORS = ["#ff4444", "#4488ff", "#44cc44", "#ffaa00"]


class Renderer:
    """Three-panel real-time renderer.

    Panels:
        Left:   Agent map — cells colored by dominant trait
                R=resource_weight, G=noise, B=crowd_sensitivity
        Center: Resource map with:
                - Green heatmap of resource levels
                - Hotspot centers marked as white circles
                - Recently shocked cells flashed in yellow
        Right:  Live trait bar chart — horizontal bars showing
                current mean of each trait across all agents

    Keyboard shortcuts:
        p — pause / resume
        q — quit
        e — toggle agent panel between trait colors and energy heatmap
    """

    SHOCK_FLASH_STEPS: int = 3   # how many steps a shock highlight persists

    def __init__(
        self,
        figsize: tuple[int, int] = (16, 5),
        delay: float = 0.05,
        trail_length: int = 6,
        decay: float = 0.7,
        show_energy: bool = False,
        condition_label: str = "",
    ) -> None:
        plt.ion()
        self.fig = plt.figure(figsize=figsize, facecolor="#111111")
        gs = gridspec.GridSpec(
            1, 3, figure=self.fig,
            width_ratios=[10, 10, 5],
            wspace=0.3,
        )
        self.ax_agents   = self.fig.add_subplot(gs[0, 0])
        self.ax_res      = self.fig.add_subplot(gs[0, 1])
        self.ax_traits   = self.fig.add_subplot(gs[0, 2])

        for ax in [self.ax_agents, self.ax_res, self.ax_traits]:
            ax.set_facecolor("#111111")

        self.delay          = delay
        self.trail_length   = trail_length
        self.decay          = decay
        self.show_energy    = show_energy
        self.condition_label = condition_label
        self._history: collections.deque[np.ndarray] = collections.deque(
            maxlen=trail_length
        )
        # Shock flash tracker: dict of (x,y) -> steps_remaining
        self._shock_cells: dict[tuple[int,int], int] = {}

        self.im_agents: plt.AxesImage | None = None
        self.im_res:    plt.AxesImage | None = None
        self._bar_container = None
        self._hotspot_scatter = None
        self._shock_scatter   = None

        self.paused:  bool = False
        self.mode_toggle_requested: bool = False
        self.running: bool = True
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_key(self, event) -> None:
        if event.key == "p":
            self.paused = not self.paused
        elif event.key == "q":
            self.running = False
        elif event.key == "e":
            self.show_energy = not self.show_energy
            if self.im_agents is not None:
                try:
                    self.im_agents.remove()
                except Exception:
                    pass
                self.im_agents = None
            self._history.clear()
        elif event.key == 'm':
            self.mode_toggle_requested = True

    # ── Frame builders ────────────────────────────────────────────────────────

    def _build_trait_frame(self, simulation: Simulation) -> np.ndarray:
        """RGB frame: R=rw, G=noise, B=cs, normalized per trait ceiling."""
        h, w = simulation.height, simulation.width
        frame  = np.zeros((h, w, 3), dtype=np.float32)
        counts = np.zeros((h, w),    dtype=np.float32)
        for agent in simulation.agents:
            t = agent.policy.traits
            frame[agent.y, agent.x, 0] += t[0]   # rw   → R
            frame[agent.y, agent.x, 1] += t[2]   # noise → G
            frame[agent.y, agent.x, 2] += t[1]   # cs   → B
            counts[agent.y, agent.x]   += 1
        mask = counts > 0
        for c in range(3):
            frame[:, :, c][mask] /= counts[mask]
        frame[:, :, 0] /= 1.5   # rw ceiling
        frame[:, :, 1] /= 0.8   # noise ceiling
        frame[:, :, 2] /= 1.0   # cs ceiling
        return np.clip(frame, 0.0, 1.0)

    def _build_energy_frame(self, simulation: Simulation) -> np.ndarray:
        frame = np.zeros((simulation.height, simulation.width),
                         dtype=np.float32)
        for agent in simulation.agents:
            frame[agent.y, agent.x] = agent.energy / Agent.INITIAL_ENERGY
        return frame

    def _blend_history(self) -> np.ndarray:
        ref        = self._history[-1]
        compatible = [f for f in self._history if f.shape == ref.shape]
        blended    = np.zeros_like(ref, dtype=np.float32)
        n          = len(compatible)
        for age, past in enumerate(reversed(compatible)):
            blended += past * (self.decay ** age)
        max_weight = sum(self.decay ** i for i in range(n))
        return np.sqrt(np.clip(blended / max_weight, 0.0, 1.0))

    # ── Main render ───────────────────────────────────────────────────────────

    def render(self, simulation: Simulation, step: int) -> None:
        from src.core.grid import Grid

        # ── Agent panel ───────────────────────────────────────────────────
        if self.show_energy:
            current_frame = self._build_energy_frame(simulation)
        else:
            current_frame = self._build_trait_frame(simulation)
        self._history.append(current_frame)
        blended = self._blend_history()

        if self.im_agents is None:
            cmap = "plasma" if self.show_energy else None
            self.im_agents = self.ax_agents.imshow(
                blended, origin="upper",
                cmap=cmap, vmin=0.0, vmax=1.0,
                interpolation="nearest",
            )
            self.ax_agents.set_xticks([])
            self.ax_agents.set_yticks([])
            title = "Agents — energy" if self.show_energy \
                    else "Agents  (R=rw  G=noise  B=cs)"
            self.ax_agents.set_title(title, color="#dddddd", fontsize=9)
        else:
            self.im_agents.set_data(blended)

        # ── Resource panel ────────────────────────────────────────────────
        res_norm = (simulation.grid.resources /
                    Grid.MAX_RESOURCE).astype(np.float32)

        if self.im_res is None:
            self.im_res = self.ax_res.imshow(
                res_norm, origin="upper", cmap="YlGn",
                vmin=0.0, vmax=1.0, interpolation="nearest",
            )
            self.ax_res.set_xticks([])
            self.ax_res.set_yticks([])
            self.ax_res.set_title("Resources  ◉=hotspot  ✦=shock",
                                  color="#dddddd", fontsize=9)
        else:
            self.im_res.set_data(res_norm)

        # Update shock flash tracker
        for cell in simulation.grid.last_shocked:
            self._shock_cells[cell] = self.SHOCK_FLASH_STEPS
        expired = [c for c, t in self._shock_cells.items() if t <= 0]
        for c in expired:
            del self._shock_cells[c]
        for c in self._shock_cells:
            self._shock_cells[c] -= 1

        # Hotspot markers
        hx = simulation.grid.hotspot_xs
        hy = simulation.grid.hotspot_ys
        if self._hotspot_scatter is not None:
            self._hotspot_scatter.remove()
        self._hotspot_scatter = self.ax_res.scatter(
            hx, hy,
            s=120, facecolors="none", edgecolors="#ffffff",
            linewidths=1.5, zorder=5,
        )

        # Shock flash markers
        if self._shock_scatter is not None:
            self._shock_scatter.remove()
        self._shock_scatter = None
        if self._shock_cells:
            sx = [c[0] for c in self._shock_cells]
            sy = [c[1] for c in self._shock_cells]
            self._shock_scatter = self.ax_res.scatter(
                sx, sy,
                s=40, color="#ffff00", alpha=0.7,
                marker="*", zorder=6,
            )

        # ── Trait bar chart ───────────────────────────────────────────────
        if simulation.agents:
            traits = np.array(
                [a.policy.traits for a in simulation.agents],
                dtype=np.float32,
            )
            means = traits.mean(axis=0)   # [rw, cs, noise, ea]
            stds  = traits.std(axis=0)
        else:
            means = np.zeros(4)
            stds  = np.zeros(4)

        self.ax_traits.cla()
        self.ax_traits.set_facecolor("#111111")
        y_pos = np.arange(4)

        bars = self.ax_traits.barh(
            y_pos, means,
            color=TRAIT_COLORS,
            xerr=stds,
            error_kw=dict(ecolor="#555555", capsize=3),
            height=0.6,
        )
        self.ax_traits.set_xlim(0, 1.5)
        self.ax_traits.set_yticks(y_pos)
        self.ax_traits.set_yticklabels(
            TRAIT_SHORT, color="#aaaaaa", fontsize=9
        )
        self.ax_traits.tick_params(colors="#aaaaaa")
        for spine in self.ax_traits.spines.values():
            spine.set_edgecolor("#444444")
        self.ax_traits.set_title("Trait means", color="#dddddd", fontsize=9)
        self.ax_traits.axvline(x=0.5, color="#333333", lw=0.8, linestyle="--")
        for bar, mean, std in zip(bars, means, stds):
            self.ax_traits.text(
                min(mean + std + 0.05, 1.45),
                bar.get_y() + bar.get_height() / 2,
                f"{mean:.2f}",
                va="center", ha="left",
                color="#aaaaaa", fontsize=8,
            )

        # ── Title ─────────────────────────────────────────────────────────
        population = simulation.agent_count()
        avg_energy = (
            sum(a.energy for a in simulation.agents) / population
            if population > 0 else 0.0
        )
        # Get mode from first agent if available
        mode_str = ""
        if simulation.agents:
            mode_str = simulation.agents[0].policy.mode

        self.fig.suptitle(
            f"Step {step}  |  Pop: {population}  |  "
            f"Avg energy: {avg_energy:.1f}  |  "
            f"Mode: [{mode_str}]  |  "
            f"{self.condition_label}  |  "
            f"[p] pause  [m] toggle mode  [e] toggle view  [q] quit",
            color="#eeeeee", fontsize=9,
        )

        self.fig.canvas.draw_idle()
        plt.pause(0.01)

        while self.paused and self.running:
            plt.pause(0.05)

        if self.delay > 0:
            time.sleep(self.delay)

    def close(self) -> None:
        plt.close(self.fig)
