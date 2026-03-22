"""Renderer module: three-panel desktop visualisation."""

from __future__ import annotations

import collections
import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D

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
        m — restart with the next policy preset
    """

    SHOCK_FLASH_STEPS: int = 3   # how many steps a shock highlight persists

    def __init__(
        self,
        figsize: tuple[int, int] = (14, 9),
        delay: float = 0.05,
        trail_length: int = 6,
        decay: float = 0.7,
        show_energy: bool = False,
        condition_label: str = "",
    ) -> None:
        plt.ion()
        self.fig = plt.figure(figsize=figsize, facecolor="#111111")
        gs = gridspec.GridSpec(
            2, 3, figure=self.fig,
            width_ratios=[10, 10, 5],
            height_ratios=[10, 6],
            wspace=0.3,
            hspace=0.45,
        )
        self.ax_agents   = self.fig.add_subplot(gs[0, 0])
        self.ax_res      = self.fig.add_subplot(gs[0, 1])
        self.ax_traits   = self.fig.add_subplot(gs[0, 2])
        self.ax_pop      = self.fig.add_subplot(gs[1, 0:3])
        self.ax_trait    = None

        for ax in [
            self.ax_agents,
            self.ax_res,
            self.ax_traits,
            self.ax_pop,
        ]:
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

        self.im_agents: AxesImage | None = None
        self.im_res:    AxesImage | None = None
        self._hotspot_scatter = None
        self._shock_scatter   = None
        self._agent_panel_kind: str | None = None

        self.paused:  bool = False
        self.mode_toggle_requested: bool = False
        self.running: bool = True
        self._pop_line: Line2D | None = None
        self._probe_line: Line2D | None = None
        self._probe_ref_line = None
        self._probe_history: list[float] = []
        self._probe_steps: list[int] = []
        self._probe_input: np.ndarray | None = None
        self._trait_lines: dict = {}
        self._history_x: list[int] = []
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
            self._probe_history.clear()
            self._probe_steps.clear()
            self._probe_line = None
            self._probe_ref_line = None
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

    def _build_neural_frame(self, simulation: Simulation) -> np.ndarray:
        frame = np.zeros((simulation.height, simulation.width),
                         dtype=np.float32)
        counts = np.zeros((simulation.height, simulation.width),
                          dtype=np.float32)
        for agent in simulation.agents:
            genome_norm = float(np.linalg.norm(agent.policy.genome)) / 4.0
            frame[agent.y, agent.x] += genome_norm
            counts[agent.y, agent.x] += 1
        mask = counts > 0
        frame[mask] /= counts[mask]
        return np.clip(frame, 0.0, 1.0)

    def _build_stateful_frame(self, simulation: Simulation) -> np.ndarray:
        """RGB frame showing the first three state channels per occupied cell."""
        h, w = simulation.height, simulation.width
        frame = np.zeros((h, w, 3), dtype=np.float32)
        counts = np.zeros((h, w), dtype=np.float32)
        for agent in simulation.agents:
            rgb_state = np.clip(agent.policy.state[:3], -1.0, 1.0)
            frame[agent.y, agent.x] += (rgb_state + 1.0) / 2.0
            counts[agent.y, agent.x] += 1
        mask = counts > 0
        for channel in range(3):
            frame[:, :, channel][mask] /= counts[mask]
        return np.clip(frame, 0.0, 1.0)

    def _blend_history(self) -> np.ndarray:
        ref        = self._history[-1]
        compatible = [f for f in self._history if f.shape == ref.shape]
        blended    = np.zeros_like(ref, dtype=np.float32)
        n          = len(compatible)
        for age, past in enumerate(reversed(compatible)):
            blended += past * (self.decay ** age)
        max_weight = sum(self.decay ** i for i in range(n))
        return np.sqrt(np.clip(blended / max_weight, 0.0, 1.0))

    def _compute_probe_spread(self, simulation: Simulation) -> float | None:
        """Compute directional sensitivity for a fixed test input.

        Uses max resource to north, no crowding anywhere.
        Returns spread = max_prob - min_prob, or None if not applicable.
        """
        from src.core.policy import NeuralPolicy, StatefulNeuralPolicy

        if not simulation.agents:
            return None
        agent = simulation.agents[0]
        policy = agent.policy
        if not isinstance(policy, (NeuralPolicy, StatefulNeuralPolicy)):
            return None
        if self._probe_input is None:
            probe_input = np.zeros(18, dtype=np.float32)
            probe_input[1] = 1.0
            self._probe_input = probe_input
        try:
            result = policy._forward(self._probe_input)
            if isinstance(result, tuple):
                probs = result[0]
            else:
                probs = result
            return float(probs.max() - probs.min())
        except Exception:
            return None

    def _draw_history_panels(self, simulation: Simulation) -> None:
        """Update the bottom two history panels."""
        from src.core.policy import TraitPolicy, NeuralPolicy, StatefulNeuralPolicy

        steps = simulation.history["step"]
        if not steps:
            return

        self._history_x = list(steps)

        ax = self.ax_pop
        total = simulation.history["total"]

        if self._pop_line is None:
            ax.set_facecolor("#111111")
            ax.tick_params(colors="#aaaaaa", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#444444")
            ax.set_ylabel("Agents", color="#aaaaaa", fontsize=8)
            ax.set_title("Population", color="#dddddd", fontsize=9)
            self._pop_line, = ax.plot(
                steps, total, color="#ffffff", lw=1.2
            )
        else:
            self._pop_line.set_data(steps, total)

        ax.set_xlim(0, max(steps[-1] + 10, 50))
        ax.set_ylim(0, max(total) * 1.2 if total else 100)

        agents = simulation.agents
        if not agents:
            return

        sample = agents[0].policy
        is_trait = isinstance(sample, TraitPolicy)
        is_neural = isinstance(sample, (NeuralPolicy, StatefulNeuralPolicy))

        panel_mode = "trait" if is_trait else "probe" if is_neural else "none"
        if getattr(self, "_history_panel_mode", None) != panel_mode:
            if self.ax_trait is not None:
                self.ax_trait.cla()
            self._trait_lines.clear()
            self._probe_line = None
            self._probe_ref_line = None
            self._history_panel_mode = panel_mode

        if is_trait:
            if self.ax_trait is None:
                self.ax_pop.set_position((0.06, 0.05, 0.40, 0.28))
                self.ax_trait = self.fig.add_axes((0.55, 0.05, 0.40, 0.28))
                self.ax_trait.set_facecolor("#111111")
                self.ax_trait.tick_params(colors="#aaaaaa", labelsize=8)
                for spine in self.ax_trait.spines.values():
                    spine.set_edgecolor("#444444")
                self.ax_trait.set_ylabel("Mean trait", color="#aaaaaa", fontsize=8)
                self.ax_trait.set_title("Trait values", color="#dddddd", fontsize=9)
                self.ax_trait.set_ylim(0, 1.1)
            ax2 = self.ax_trait
            rw_vals = simulation.history.get("mean_resource_weight", [])
            cs_vals = simulation.history.get("mean_crowd_sensitivity", [])
            noise_vals = simulation.history.get("mean_noise", [])
            ea_vals = simulation.history.get("mean_energy_awareness", [])

            trait_data = {
                "rw": ("#ff4444", rw_vals),
                "cs": ("#4488ff", cs_vals),
                "noise": ("#44cc44", noise_vals),
                "ea": ("#ffaa00", ea_vals),
            }

            if not self._trait_lines:
                ax2.set_facecolor("#111111")
                ax2.tick_params(colors="#aaaaaa", labelsize=8)
                for spine in ax2.spines.values():
                    spine.set_edgecolor("#444444")
                ax2.set_ylabel("Mean trait", color="#aaaaaa", fontsize=8)
                ax2.set_title("Trait values", color="#dddddd", fontsize=9)
                ax2.set_ylim(0, 1.1)
                for key, (color, vals) in trait_data.items():
                    line, = ax2.plot(steps, vals, color=color, lw=1.0, label=key)
                    self._trait_lines[key] = line
                ax2.legend(
                    facecolor="#222222", edgecolor="#555555",
                    labelcolor="#dddddd", fontsize=7,
                    loc="upper right"
                )
            else:
                for key, (color, vals) in trait_data.items():
                    if key in self._trait_lines and vals:
                        self._trait_lines[key].set_data(steps, vals)

            ax2.set_xlim(0, max(steps[-1] + 10, 50))
        elif is_neural:
            if self.ax_trait is None:
                self.ax_pop.set_position((0.06, 0.05, 0.40, 0.28))
                self.ax_trait = self.fig.add_axes((0.55, 0.05, 0.40, 0.28))

            if simulation.current_step % 10 == 0:
                spread = self._compute_probe_spread(simulation)
                if spread is not None:
                    self._probe_history.append(spread)
                    self._probe_steps.append(simulation.current_step)

            if not self._probe_history:
                return

            ax2 = self.ax_trait
            if ax2 is None:
                return

            if self._probe_line is None:
                ax2.cla()
                ax2.set_facecolor("#111111")
                ax2.tick_params(colors="#aaaaaa", labelsize=8)
                for spine in ax2.spines.values():
                    spine.set_edgecolor("#444444")
                ax2.set_ylabel("Spread", color="#aaaaaa", fontsize=8)
                ax2.set_title(
                    "Directional sensitivity (probe spread)",
                    color="#dddddd", fontsize=9
                )
                self._probe_ref_line = ax2.axhline(
                    y=0.0, color="#e3b341", lw=1.0,
                    linestyle="--", alpha=0.6,
                    label="uniform baseline"
                )
                ax2.legend(
                    facecolor="#222222", edgecolor="#555555",
                    labelcolor="#dddddd", fontsize=7,
                    loc="upper right"
                )
                self._probe_line, = ax2.plot(
                    self._probe_steps,
                    self._probe_history,
                    color="#79c0ff", lw=1.2
                )
            else:
                self._probe_line.set_data(
                    self._probe_steps, self._probe_history
                )

            ax2.set_xlim(0, max(self._probe_steps[-1] + 10, 50))
            ax2.set_ylim(0, max(max(self._probe_history) * 1.2, 0.1))
        else:
            if self.ax_trait is not None:
                self.ax_trait.remove()
                self.ax_trait = None
                self._trait_lines.clear()
            self.ax_pop.set_position((0.125, 0.11, 0.775, 0.28))

    # ── Main render ───────────────────────────────────────────────────────────

    def render(self, simulation: Simulation, step: int) -> None:
        from src.core.grid import Grid

        mode_str = ""
        if simulation.agents:
            mode_str = simulation.agents[0].policy.mode

        # ── Agent panel ───────────────────────────────────────────────────
        if self.show_energy:
            current_frame = self._build_energy_frame(simulation)
            panel_kind = "energy"
            cmap = "plasma"
            title = "Agents — energy"
        elif mode_str == "stateful":
            current_frame = self._build_stateful_frame(simulation)
            panel_kind = "stateful"
            cmap = None
            title = "Agents — Phase 5 state (R=s0 G=s1 B=s2)"
        elif mode_str == "neural":
            current_frame = self._build_neural_frame(simulation)
            panel_kind = "neural"
            cmap = "magma"
            title = "Agents — neural genome norm"
        else:
            current_frame = self._build_trait_frame(simulation)
            panel_kind = "traits"
            cmap = None
            title = "Agents  (R=rw  G=noise  B=cs)"
        self._history.append(current_frame)
        blended = self._blend_history()

        if self.im_agents is None or self._agent_panel_kind != panel_kind:
            if self.im_agents is not None:
                try:
                    self.im_agents.remove()
                except Exception:
                    pass
            self.im_agents = self.ax_agents.imshow(
                blended, origin="upper",
                cmap=cmap, vmin=0.0, vmax=1.0,
                interpolation="nearest",
            )
            self._agent_panel_kind = panel_kind
            self.ax_agents.set_xticks([])
            self.ax_agents.set_yticks([])
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
        if simulation.agents and mode_str not in {"neural", "stateful"}:
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
        self.ax_traits.tick_params(colors="#aaaaaa")
        for spine in self.ax_traits.spines.values():
            spine.set_edgecolor("#444444")

        if mode_str == "neural":
            if simulation.agents:
                norms = np.array(
                    [float(np.linalg.norm(a.policy.genome)) for a in simulation.agents],
                    dtype=np.float32,
                )
                mean_norm = float(norms.mean())
                std_norm = float(norms.std())
            else:
                mean_norm = 0.0
                std_norm = 0.0
            bars = self.ax_traits.barh(
                [0], [mean_norm],
                color="#c77dca",
                xerr=[std_norm],
                error_kw=dict(ecolor="#555555", capsize=3),
                height=0.6,
            )
            self.ax_traits.set_xlim(0, max(1.5, mean_norm + std_norm + 0.5))
            self.ax_traits.set_yticks([0])
            self.ax_traits.set_yticklabels(["gnorm"], color="#aaaaaa", fontsize=9)
            self.ax_traits.set_title("Neural genome norm", color="#dddddd", fontsize=9)
            for bar in bars:
                self.ax_traits.text(
                    mean_norm + std_norm + 0.05,
                    bar.get_y() + bar.get_height() / 2,
                    f"{mean_norm:.2f}",
                    va="center", ha="left",
                    color="#aaaaaa", fontsize=8,
                )
        elif mode_str == "stateful":
            if simulation.agents:
                states = np.array(
                    [a.policy.state for a in simulation.agents],
                    dtype=np.float32,
                )
                state_means = states.mean(axis=0)
                state_stds = states.std(axis=0)
                genome_norm = float(np.mean([
                    float(np.linalg.norm(a.policy.genome))
                    for a in simulation.agents
                ]))
            else:
                state_means = np.zeros(4, dtype=np.float32)
                state_stds = np.zeros(4, dtype=np.float32)
                genome_norm = 0.0

            labels = ["s0", "s1", "s2", "s3"]
            colors = ["#ff6b6b", "#4ecdc4", "#ffe66d", "#5c7cfa"]
            y_pos = np.arange(4)
            bars = self.ax_traits.barh(
                y_pos,
                state_means,
                color=colors,
                xerr=state_stds,
                error_kw=dict(ecolor="#555555", capsize=3),
                height=0.6,
            )
            self.ax_traits.set_xlim(-1.0, 1.0)
            self.ax_traits.set_yticks(y_pos)
            self.ax_traits.set_yticklabels(
                labels,
                color="#aaaaaa",
                fontsize=9,
            )
            self.ax_traits.set_title(
                f"State means | gnorm={genome_norm:.2f}",
                color="#dddddd",
                fontsize=9,
            )
            self.ax_traits.axvline(x=0.0, color="#333333", lw=0.8, linestyle="--")
            for bar, mean in zip(bars, state_means):
                x_text = mean + 0.06 if mean >= 0 else mean - 0.06
                x_text = float(min(max(x_text, -0.92), 0.92))
                self.ax_traits.text(
                    x_text,
                    bar.get_y() + bar.get_height() / 2,
                    f"{mean:.2f}",
                    va="center",
                    ha="left" if mean >= 0 else "right",
                    color="#aaaaaa",
                    fontsize=8,
                )
        else:
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
        self.fig.suptitle(
            f"Step {step}  |  Pop: {population}  |  "
            f"Avg energy: {avg_energy:.1f}  |  "
            f"Mode: [{mode_str}]  |  "
            f"{self.condition_label}  |  "
            f"[p] pause  [m] toggle mode  [e] toggle view  [q] quit",
            color="#eeeeee", fontsize=9,
        )

        self._draw_history_panels(simulation)
        self.fig.canvas.draw_idle()
        plt.pause(0.01)

        while self.paused and self.running:
            plt.pause(0.05)

        if self.delay > 0:
            time.sleep(self.delay)

    def close(self) -> None:
        plt.close(self.fig)
