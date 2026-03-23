"""Left-side control panel: experiment selection, parameter editing, run/stop."""

from __future__ import annotations

from collections import defaultdict

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.experiments.registry import get_experiment, list_experiments
from app.snapshot import SimSnapshot


class ExperimentPanel(QWidget):
    """Control panel for selecting and configuring a simulation experiment.

    Signals:
        run_requested: Emitted when the Run button is pressed.  Carries the
            registry key (str) and a dict of parameter overrides.
        stop_requested: Emitted when the Stop button is pressed.
    """

    run_requested: Signal = Signal(str, dict)
    stop_requested: Signal = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(280)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        # ── Title ────────────────────────────────────────────────────────────
        title = QLabel("Life-Sim Workbench")
        title.setStyleSheet("font-size: 13px; font-weight: bold; color: #aaddff;")
        root.addWidget(title)

        # ── Experiment selector ───────────────────────────────────────────────
        root.addWidget(QLabel("Experiment"))
        self._combo = QComboBox()
        self._combo.setModel(self._build_combo_model())
        root.addWidget(self._combo)

        # ── Metadata panel ────────────────────────────────────────────────────
        self._meta_frame = QFrame()
        self._meta_frame.setStyleSheet(
            "QFrame { background: #1e1e2a; border-radius: 4px; }"
        )
        meta_layout = QVBoxLayout(self._meta_frame)
        meta_layout.setContentsMargins(6, 6, 6, 6)
        meta_layout.setSpacing(3)

        self._meta_name = QLabel()
        self._meta_name.setStyleSheet(
            "color: #ffffff; font-weight: bold; font-size: 11px;"
        )
        self._meta_name.setWordWrap(True)
        meta_layout.addWidget(self._meta_name)

        self._meta_desc = QLabel()
        self._meta_desc.setStyleSheet("color: #aaaaaa; font-size: 13px;")
        self._meta_desc.setWordWrap(True)
        self._meta_desc.setMaximumHeight(52)  # ~3 lines
        meta_layout.addWidget(self._meta_desc)

        self._meta_params = QLabel()
        self._meta_params.setStyleSheet("color: #888888; font-size: 13px;")
        self._meta_params.setWordWrap(True)
        meta_layout.addWidget(self._meta_params)

        root.addWidget(self._meta_frame)

        # ── Parameter fields ──────────────────────────────────────────────────
        form = QFormLayout()
        form.setSpacing(6)

        self._steps_spin = QSpinBox()
        self._steps_spin.setRange(10, 100_000)
        self._steps_spin.setSingleStep(100)
        self._steps_spin.setValue(1000)
        form.addRow("Steps", self._steps_spin)

        self._seeds_edit = QLineEdit()
        self._seeds_edit.setPlaceholderText("e.g. 42, 43")
        self._seeds_edit.setText("42")
        form.addRow("Seeds", self._seeds_edit)

        root.addLayout(form)

        self._seeds_error = QLabel("Enter at least one valid integer seed.")
        self._seeds_error.setStyleSheet("color: #ff5555; font-size: 10px;")
        self._seeds_error.setVisible(False)
        root.addWidget(self._seeds_error)

        # ── Run / Stop buttons ────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Run")
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._stop_btn)
        root.addLayout(btn_row)

        # ── Progress bar (hidden until a run begins) ──────────────────────────
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.setFixedHeight(12)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setStyleSheet(
            "QProgressBar { background: #2a2a2a; border: 1px solid #444; border-radius: 3px; }"
            "QProgressBar::chunk { background: #4a9eff; border-radius: 3px; }"
        )
        self._progress_bar.setVisible(False)
        root.addWidget(self._progress_bar)

        # ── Status label ──────────────────────────────────────────────────────
        self._status_label = QLabel("Ready")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #888888; font-size: 11px;")
        root.addWidget(self._status_label)

        root.addStretch()

        # ── Internal connections ───────────────────────────────────────────────
        self._run_btn.clicked.connect(self._on_run_clicked)
        self._stop_btn.clicked.connect(self.stop_requested)
        self._combo.currentIndexChanged.connect(self._on_experiment_changed)
        # Populate metadata for the initial selection (setModel may have already
        # advanced past any leading non-selectable separator without the signal
        # being connected, so trigger the slot explicitly here).
        self._on_experiment_changed(self._combo.currentIndex())

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_run_clicked(self) -> None:
        key = self._current_key()
        if not key:
            return

        seeds = self._parse_seeds(self._seeds_edit.text())
        if not seeds:
            self._seeds_error.setVisible(True)
            return
        self._seeds_error.setVisible(False)

        params: dict = {
            "steps": self._steps_spin.value(),
            "seeds": seeds,
        }
        self._set_running(True)
        self.run_requested.emit(key, params)

    def _on_experiment_changed(self, index: int) -> None:
        """Refresh the metadata panel when the selected experiment changes."""
        key = self._combo.itemData(index, Qt.ItemDataRole.UserRole)
        if not key:
            self._meta_frame.setVisible(False)
            return
        try:
            exp = get_experiment(key)
        except KeyError:
            self._meta_frame.setVisible(False)
            return

        self._meta_name.setText(exp.name)
        self._meta_desc.setText(exp.description)

        # Pre-populate seeds from the experiment's defaults.
        self._seeds_edit.setText(", ".join(str(s) for s in exp.seeds))

        drift = exp.env_config.get("drift_step", 1)
        noise_rate = exp.env_config.get("noise_rate", 3.0)
        self._meta_params.setText(
            f"Steps: {exp.steps} | Seeds: {len(exp.seeds)} | "
            f"Policy: {exp.policy_mode} | "
            f"Env: drift={drift} noise={noise_rate}"
        )
        self._meta_frame.setVisible(True)

    def _parse_seeds(self, text: str) -> list[int]:
        """Parse a comma-separated string of integers; fall back to [42]."""
        seeds: list[int] = []
        for part in text.split(","):
            part = part.strip()
            if part.isdigit():
                seeds.append(int(part))
        return seeds if seeds else [42]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_status(self, snapshot: SimSnapshot) -> None:
        """Update the status label from a live snapshot."""
        self._status_label.setText(
            f"Step: {snapshot.step}  |  Population: {snapshot.population}"
        )

    def set_idle(self) -> None:
        """Reset UI to idle state after a run completes or is stopped."""
        self._set_running(False)
        self._progress_bar.setVisible(False)
        self._progress_bar.setValue(0)
        self._status_label.setText("Done")

    def update_progress(self, current: int, total: int, seed: str) -> None:
        """Update the progress bar; called each time a new seed starts."""
        self._progress_bar.setRange(0, total)
        self._progress_bar.setValue(current)
        self._progress_bar.setVisible(True)
        self._status_label.setText(f"Seed {seed}  ({current + 1}/{total})")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_running(self, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        self._combo.setEnabled(not running)
        self._steps_spin.setEnabled(not running)
        self._seeds_edit.setEnabled(not running)
        if running:
            self._progress_bar.setVisible(False)
            self._progress_bar.setValue(0)
            self._status_label.setText("Running…")

    def _current_key(self) -> str:
        """Return the registry key stored in the current combo selection."""
        return self._combo.currentData(Qt.ItemDataRole.UserRole) or ""

    def _build_combo_model(self) -> QStandardItemModel:
        """Build a QStandardItemModel with phase-group separators."""
        model = QStandardItemModel()

        sep_font = QFont()
        sep_font.setItalic(True)
        sep_color = QColor("#555555")

        groups: dict[str, list[str]] = defaultdict(list)
        for key in list_experiments():
            # Phase prefix is the first token: "phase2", "phase3", etc.
            prefix = key.split("_")[0]
            groups[prefix].append(key)

        for prefix in sorted(groups):
            # Non-selectable group header
            sep = QStandardItem(f"── {prefix} ──")
            sep.setFlags(Qt.ItemFlag(0))
            sep.setFont(sep_font)
            sep.setForeground(sep_color)
            model.appendRow(sep)

            for key in groups[prefix]:
                try:
                    exp = get_experiment(key)
                    display = exp.name or key
                except KeyError:
                    display = key
                item = QStandardItem(display)
                item.setData(key, Qt.ItemDataRole.UserRole)
                model.appendRow(item)

        return model
