"""Vispy-based canvas panel: resource heatmap background + agent marker foreground."""

from __future__ import annotations

import numpy as np

import vispy
vispy.use("pyside6")  # must be called before any Canvas is instantiated

import vispy.scene
from vispy.scene.visuals import Image as ImageVisual
from vispy.scene.visuals import Markers as MarkersVisual

from PySide6.QtCore import Signal
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QFrame

from app.snapshot import SimSnapshot

# ── Direction colour lookup ────────────────────────────────────────────────────
# Indexed by (last_dx + 1, last_dy + 1) — values in {0, 1, 2}.
_DIR_LOOKUP = np.full((3, 3, 4), [0.55, 0.55, 0.55, 1.0], dtype=np.float32)
_DIR_LOOKUP[1, 1] = [0.55, 0.55, 0.55, 1.0]  # stationary – grey
_DIR_LOOKUP[2, 1] = [1.00, 0.55, 0.00, 1.0]  # dx=+1 (right) – orange
_DIR_LOOKUP[0, 1] = [0.25, 0.55, 1.00, 1.0]  # dx=-1 (left)  – sky-blue
_DIR_LOOKUP[1, 2] = [0.20, 0.90, 0.30, 1.0]  # dy=+1 (down)  – green
_DIR_LOOKUP[1, 0] = [0.85, 0.20, 0.85, 1.0]  # dy=-1 (up)    – violet

# Sentinel for empty agent sets
_EMPTY_POS = np.array([[-1.0, -1.0, 0.0]], dtype=np.float32)
_EMPTY_COL = np.zeros((1, 4), dtype=np.float32)

_BTN_BASE = (
    "QPushButton {"
    "  background-color: #2a2a2a;"
    "  color: #aaaaaa;"
    "  border: 1px solid #444;"
    "  padding: 3px 10px;"
    "  font-size: 11px;"
    "}"
    "QPushButton:hover { background-color: #3a3a3a; }"
)
_BTN_ACTIVE = (
    "QPushButton {"
    "  background-color: #4a9eff;"
    "  color: #ffffff;"
    "  border: 1px solid #4a9eff;"
    "  padding: 3px 10px;"
    "  font-size: 11px;"
    "  font-weight: bold;"
    "}"
)


class _SceneCanvas(vispy.scene.SceneCanvas):
    """Internal vispy canvas: image layer + markers layer, no Qt chrome."""

    def __init__(self) -> None:
        super().__init__(
            title="",
            bgcolor="#111111",
            app="pyside6",
            show=False,
        )
        self.unfreeze()

        self._view = self.central_widget.add_view()
        self._view.camera = vispy.scene.cameras.PanZoomCamera(aspect=1)
        # Row 0 of the resource array renders at the visual top.
        self._view.camera.flip = (False, True, False)

        # Background: resource heatmap.
        placeholder = np.zeros((50, 50), dtype=np.float32)
        self._image: ImageVisual = ImageVisual(
            placeholder,
            cmap="YlGn",
            clim=(0.0, 1.0),
            parent=self._view.scene,
        )

        # Foreground: agent markers drawn on top, no depth occlusion by image.
        self._markers: MarkersVisual = MarkersVisual(parent=self._view.scene)
        self._markers.set_gl_state("translucent", depth_test=False)
        self._markers.set_data(_EMPTY_POS, face_color=_EMPTY_COL, size=0.0)

        self._grid_w: int = 50
        self._grid_h: int = 50
        self._view.camera.set_range(x=(0, 50), y=(0, 50))

        self.freeze()

    def update_frame(
        self, snapshot: SimSnapshot, face_colors: np.ndarray
    ) -> None:
        """Redraw both layers from *snapshot* using the pre-computed RGBA *face_colors*."""
        if snapshot.width != self._grid_w or snapshot.height != self._grid_h:
            self._grid_w = snapshot.width
            self._grid_h = snapshot.height
            self._view.camera.set_range(
                x=(0, snapshot.width), y=(0, snapshot.height)
            )

        self._image.set_data(snapshot.resources)

        n = snapshot.population
        if n > 0:
            xs = snapshot.agent_xs.astype(np.float32) + 0.5
            ys = snapshot.agent_ys.astype(np.float32) + 0.5
            positions = np.column_stack(
                [xs, ys, np.zeros(n, dtype=np.float32)]
            )
            self._markers.set_data(positions, face_color=face_colors, size=6.0)
        else:
            self._markers.set_data(_EMPTY_POS, face_color=_EMPTY_COL, size=0.0)

        self.update()


class SimulationCanvas(QWidget):
    """Panel combining the vispy scene with a colour-mode toggle bar below.

    Layout::

        ┌────────────────────────────────────────┐
        │  vispy scene (resources + agents)      │
        ├────────────────────────────────────────┤
        │  [Policy] [Energy] [Crowd Sens]        │
        │  [Noise]  [Resource Wt]                │
        └────────────────────────────────────────┘

    Clicking a button immediately redraws agents with the selected colouring.

    Modes:
        * **Policy**      — use the ``agent_colors`` stored in the snapshot
          (set by the worker; currently a fixed coral that can be made
          policy-type-specific).
        * **Energy**      — red (low energy) → green (high energy).
        * **Crowd Sens**  — trait[1]: blue (low) → red (high).
        * **Noise**       — trait[2]: blue (low) → yellow (high).
        * **Resource Wt** — trait[0]: blue (low) → orange (high).

    Embed in a Qt layout directly; no ``.native`` unwrapping needed.
    """

    # (label, internal key)
    _MODES: list[tuple[str, str]] = [
        ("Policy",      "Policy"),
        ("Energy",      "Energy"),
        ("Crowd Sens",  "Crowd Sens"),
        ("Noise",       "Noise"),
        ("Resource Wt", "Resource Wt"),
    ]
    _DEFAULT_MODE = "Policy"

    # Emitted after every completed update_snapshot() so the worker knows the
    # UI has consumed the frame and can send the next one.
    frame_consumed: Signal = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Vispy scene (wrapped in container for overlay support) ─────────────
        self._scene = _SceneCanvas()
        self._scene.native.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        # Container widget: vispy fills it, overlay label sits on top as a
        # raised sibling (avoids QOpenGLWidget child compositing issues).
        self._canvas_container = QWidget()
        self._canvas_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        _cl = QVBoxLayout(self._canvas_container)
        _cl.setContentsMargins(0, 0, 0, 0)
        _cl.setSpacing(0)
        _cl.addWidget(self._scene.native)

        # Overlay label — absolute-positioned child of the container, not
        # managed by any layout so it floats above the GL surface.
        self._step_label = QLabel("", self._canvas_container)
        self._step_label.setStyleSheet(
            "background-color: rgba(0, 0, 0, 180);"
            "color: #ffffff;"
            "padding: 6px;"
            "font-size: 11px;"
        )
        self._step_label.move(6, 6)
        self._step_label.hide()  # hidden until first snapshot arrives

        root.addWidget(self._canvas_container, stretch=1)

        # ── Toggle button bar ──────────────────────────────────────────────────
        btn_bar = QWidget()
        btn_bar.setFixedHeight(32)
        btn_bar.setStyleSheet("background-color: #1a1a1a;")
        btn_layout = QHBoxLayout(btn_bar)
        btn_layout.setContentsMargins(6, 2, 6, 2)
        btn_layout.setSpacing(4)

        self._buttons: dict[str, QPushButton] = {}
        for label, key in self._MODES:
            btn = QPushButton(label)
            btn.setStyleSheet(_BTN_BASE)
            btn.clicked.connect(lambda _checked, k=key: self._on_mode_clicked(k))
            btn_layout.addWidget(btn)
            self._buttons[key] = btn

        btn_layout.addStretch()

        # Status message shown when an auto-switch overrides the Policy mode.
        self._status_label = QLabel("")
        self._status_label.setStyleSheet(
            "color: #ffcc44; font-size: 10px; padding-right: 6px;"
        )
        self._status_label.setVisible(False)
        btn_layout.addWidget(self._status_label)

        root.addWidget(btn_bar)

        # ── Legend strip ───────────────────────────────────────────────────────
        legend_bar = QWidget()
        legend_bar.setFixedHeight(28)
        legend_bar.setStyleSheet("background-color: #161616;")
        legend_layout = QHBoxLayout(legend_bar)
        legend_layout.setContentsMargins(8, 0, 8, 0)
        legend_layout.setSpacing(10)

        # Resource gradient swatch + label (static)
        res_swatch = QFrame()
        res_swatch.setFixedSize(54, 10)
        res_swatch.setStyleSheet(
            "QFrame {"
            "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            "    stop:0 #ffffcc, stop:1 #006837);"
            "  border-radius: 2px;"
            "}"
        )
        res_label = QLabel("Resources: Low \u2192 High")
        res_label.setStyleSheet("color: #888888; font-size: 10px;")

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFixedHeight(14)
        sep.setStyleSheet("color: #3a3a3a;")

        # Agent legend label (dynamic, updates with mode)
        self._legend_label = QLabel()
        self._legend_label.setTextFormat(Qt.TextFormat.RichText)
        self._legend_label.setStyleSheet("font-size: 10px;")

        legend_layout.addWidget(res_swatch)
        legend_layout.addWidget(res_label)
        legend_layout.addWidget(sep)
        legend_layout.addWidget(self._legend_label)
        legend_layout.addStretch()
        root.addWidget(legend_bar)

        # ── State ──────────────────────────────────────────────────────────────
        self._active_mode: str = self._DEFAULT_MODE
        self._last_snapshot: SimSnapshot | None = None
        # Set to True when the user explicitly clicks the Policy button so that
        # the auto-switch is suppressed for that deliberate choice.
        self._suppress_auto_switch: bool = False
        self._buttons[self._DEFAULT_MODE].setStyleSheet(_BTN_ACTIVE)
        self._update_legend(self._DEFAULT_MODE)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_snapshot(self, snapshot: SimSnapshot) -> None:
        """Redraw the canvas from *snapshot*.

        When the active mode is Policy and all agents share the same colour
        (i.e. all have the same policy type), automatically switches to Energy
        mode and shows a status message in the button bar.

        Safe to call from the Qt main thread (e.g. connected to a Signal).
        """
        self._last_snapshot = snapshot

        if (
            self._active_mode == "Policy"
            and not self._suppress_auto_switch
            and snapshot.population > 0
        ):
            if snapshot.all_neural:
                self._set_active_mode("Energy")
                self._status_label.setText(
                    "All agents are Neural — showing Energy instead."
                )
                self._status_label.setVisible(True)
            else:
                self._status_label.setVisible(False)

        face_colors = self._build_face_colors(snapshot)
        self._scene.update_frame(snapshot, face_colors)

        # Update and raise the overlay label so it always sits on top.
        self._step_label.setText(
            f"Step {snapshot.step}  |  Pop: {snapshot.population}"
        )
        self._step_label.adjustSize()
        self._step_label.raise_()
        self._step_label.show()

        self.frame_consumed.emit()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _set_active_mode(self, key: str) -> None:
        """Update ``_active_mode`` and button highlights without triggering a redraw."""
        if key == self._active_mode:
            return
        self._buttons[self._active_mode].setStyleSheet(_BTN_BASE)
        self._active_mode = key
        self._buttons[key].setStyleSheet(_BTN_ACTIVE)

    def _on_mode_clicked(self, key: str) -> None:
        # Hide any auto-switch status message on manual selection.
        self._status_label.setVisible(False)
        # Respect an explicit Policy click — suppress auto-switch until the
        # user navigates away from Policy and back again.
        self._suppress_auto_switch = key == "Policy"
        self._set_active_mode(key)
        self._update_legend(key)
        if self._last_snapshot is not None:
            face_colors = self._build_face_colors(self._last_snapshot)
            self._scene.update_frame(self._last_snapshot, face_colors)

    # ------------------------------------------------------------------
    # Colour computation
    # ------------------------------------------------------------------

    def _update_legend(self, mode: str) -> None:
        """Update the agent legend label to reflect the current colour mode."""
        _LEGEND_HTML: dict[str, str] = {
            "Policy": (
                "<span style='color:#ff4444'>&#9679;</span> Trait&nbsp;&nbsp;"
                "<span style='color:#4488ff'>&#9679;</span> Neural&nbsp;&nbsp;"
                "<span style='color:#9966cc'>&#9679;</span> Stateful"
            ),
            "Energy": (
                "<span style='color:#ff4444'>&#9679;</span> Low energy&nbsp;&nbsp;"
                "<span style='color:#44cc44'>&#9679;</span> High energy"
            ),
            "Crowd Sens": (
                "<span style='color:#4488ff'>&#9679;</span> Low&nbsp;&nbsp;"
                "<span style='color:#ff4444'>&#9679;</span> High"
            ),
            "Noise": (
                "<span style='color:#4488ff'>&#9679;</span> Low&nbsp;&nbsp;"
                "<span style='color:#ffee44'>&#9679;</span> High"
            ),
            "Resource Wt": (
                "<span style='color:#4488ff'>&#9679;</span> Low&nbsp;&nbsp;"
                "<span style='color:#ff9922'>&#9679;</span> High"
            ),
        }
        self._legend_label.setText(_LEGEND_HTML.get(mode, ""))

    def _build_face_colors(self, snapshot: SimSnapshot) -> np.ndarray:
        """Return RGBA face colours for all agents, shape (N, 4) float32."""
        n = snapshot.population
        if n == 0:
            return _EMPTY_COL

        mode = self._active_mode

        if mode == "Policy":
            alpha = np.ones((n, 1), dtype=np.float32)
            return np.concatenate([snapshot.agent_colors, alpha], axis=1).astype(
                np.float32
            )

        if mode == "Energy":
            emax = float(np.max(snapshot.agent_energies))
            if emax < 1e-6:
                emax = 1.0
            t = np.clip(snapshot.agent_energies / emax, 0.0, 1.0).astype(
                np.float32
            )
            r = 1.0 - t
            g = t
            b = np.zeros(n, dtype=np.float32)
            a = np.ones(n, dtype=np.float32)
            return np.column_stack([r, g, b, a]).astype(np.float32)

        # Trait-based modes: map a scalar trait to a two-colour gradient.
        # trait_col: (trait_index, low_rgb, high_rgb)
        _TRAIT_GRADIENTS: dict[str, tuple[int, tuple, tuple]] = {
            "Crowd Sens":  (1, (0.20, 0.40, 1.00), (1.00, 0.20, 0.20)),
            "Noise":       (2, (0.20, 0.40, 1.00), (1.00, 0.90, 0.10)),
            "Resource Wt": (0, (0.20, 0.40, 1.00), (1.00, 0.60, 0.10)),
        }

        if mode in _TRAIT_GRADIENTS:
            idx, lo, hi = _TRAIT_GRADIENTS[mode]
            col = snapshot.agent_traits[:, idx].astype(np.float32)
            cmax, cmin = float(col.max()), float(col.min())
            rng = cmax - cmin
            t = np.clip((col - cmin) / rng if rng > 1e-6 else np.full(n, 0.5, dtype=np.float32), 0.0, 1.0)
            lo_arr = np.array(lo, dtype=np.float32)
            hi_arr = np.array(hi, dtype=np.float32)
            rgb = lo_arr + t[:, None] * (hi_arr - lo_arr)
            a = np.ones((n, 1), dtype=np.float32)
            return np.concatenate([rgb, a], axis=1).astype(np.float32)

        # Fallback — uniform coral
        colors = np.ones((n, 4), dtype=np.float32)
        colors[:, 0] = 0.90
        colors[:, 1] = 0.35
        colors[:, 2] = 0.35
        return colors

