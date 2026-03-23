"""Main application window: layout, signal wiring, dark theme."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QSizePolicy,
)

from src.experiments.registry import get_experiment
from app.worker import SimWorker
from app.panels.simulation_canvas import SimulationCanvas
from app.panels.experiment_panel import ExperimentPanel
from app.panels.results_panel import ResultsPanel
from app.snapshot import SimSnapshot

DARK_STYLE = """
    QMainWindow, QWidget { background-color: #1a1a1a; color: #dddddd; }
    QComboBox, QSpinBox, QLineEdit { background-color: #2a2a2a; color: #dddddd; border: 1px solid #444; padding: 4px; }
    QPushButton { background-color: #2a2a2a; color: #dddddd; border: 1px solid #444; padding: 6px 12px; }
    QPushButton:hover { background-color: #3a3a3a; }
    QPushButton:pressed { background-color: #444; }
    QTabWidget::pane { border: 1px solid #444; }
    QTabBar::tab { background-color: #2a2a2a; color: #dddddd; padding: 6px 12px; }
    QTabBar::tab:selected { background-color: #3a3a3a; }
    QSplitter::handle { background-color: #333; }
    QListWidget { background-color: #2a2a2a; color: #dddddd; border: 1px solid #444; }
    QLabel { color: #dddddd; }
"""


class MainWindow(QMainWindow):
    """Top-level application window.

    Layout (horizontal QSplitter):
    ┌─────────────────────────────────────────────────────────┐
    │ ExperimentPanel (280 px)  │  SimulationCanvas (stretch 2) │
    │                           │──────────────────────────────│
    │                           │  ResultsPanel    (stretch 1) │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Life-Sim Research Workbench")
        self.setMinimumSize(1400, 900)

        self._worker: SimWorker | None = None

        # ── Panels ────────────────────────────────────────────────────────────
        self._experiment_panel = ExperimentPanel()
        self._canvas = SimulationCanvas()
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._results_panel = ResultsPanel()

        # ── Speed control bar (sits below the canvas, above the results panel) ─
        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setRange(0, 200)
        self._speed_slider.setValue(0)
        self._speed_slider.setEnabled(False)
        self._speed_slider.setToolTip("0 = max speed, 200 = slowest")
        self._speed_slider.valueChanged.connect(self._on_speed_changed)

        speed_bar = QWidget()
        speed_bar.setFixedHeight(28)
        _sb_layout = QHBoxLayout(speed_bar)
        _sb_layout.setContentsMargins(8, 2, 8, 2)
        _sb_layout.setSpacing(8)
        _sb_layout.addWidget(QLabel("Speed:  Max ←"))
        _sb_layout.addWidget(self._speed_slider, stretch=1)
        _sb_layout.addWidget(QLabel("→ Slow"))

        # Wrap canvas + speed bar together so they move as one in the splitter.
        canvas_wrapper = QWidget()
        _cw_layout = QVBoxLayout(canvas_wrapper)
        _cw_layout.setContentsMargins(0, 0, 0, 0)
        _cw_layout.setSpacing(0)
        _cw_layout.addWidget(self._canvas)
        _cw_layout.addWidget(speed_bar)

        # ── Right vertical splitter (canvas + results) ─────────────────────
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.addWidget(canvas_wrapper)
        right_splitter.addWidget(self._results_panel)
        right_splitter.setStretchFactor(0, 2)
        right_splitter.setStretchFactor(1, 1)

        # ── Horizontal root splitter ──────────────────────────────────────────
        root_splitter = QSplitter(Qt.Orientation.Horizontal)
        root_splitter.addWidget(self._experiment_panel)
        root_splitter.addWidget(right_splitter)
        root_splitter.setStretchFactor(0, 0)
        root_splitter.setStretchFactor(1, 1)

        self.setCentralWidget(root_splitter)

        # ── Signal wiring ─────────────────────────────────────────────────────
        self._experiment_panel.run_requested.connect(self._on_run_requested)
        self._experiment_panel.stop_requested.connect(self._on_stop_requested)
        self._results_panel.status_message.connect(
            lambda msg: self.statusBar().showMessage(msg, 3000)
        )
    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_run_requested(self, name: str, params: dict) -> None:
        """Create a SimWorker for *name* and start it."""
        # Cancel any in-flight worker. Disconnect its signals immediately so
        # stale step_ready / run_complete events don't reach the new session.
        # We do NOT call wait() — that would block the main thread.
        if self._worker is not None and self._worker.isRunning():
            self._disconnect_worker(self._worker)
            self._worker.cancel()

        try:
            experiment = get_experiment(name)
        except KeyError as exc:
            QMessageBox.warning(self, "Unknown Experiment", str(exc))
            self._experiment_panel.set_idle()
            return

        self._worker = SimWorker(experiment, params)
        self._connect_worker(self._worker)
        self._worker.set_speed(self._speed_slider.value())
        self._results_panel.begin_run(params.get("seeds", []))
        self._speed_slider.setEnabled(True)
        self._worker.start()

    def _on_stop_requested(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._disconnect_worker(self._worker)
            self._worker.cancel()
        self._speed_slider.setEnabled(False)
        self._experiment_panel.set_idle()

    def _connect_worker(self, worker: SimWorker) -> None:
        worker.step_ready.connect(self._on_step_ready)
        worker.run_complete.connect(self._on_run_complete)
        worker.error.connect(self._on_worker_error)
        worker.progress.connect(self._experiment_panel.update_progress)
        self._canvas.frame_consumed.connect(worker.on_frame_consumed)

    def _disconnect_worker(self, worker: SimWorker) -> None:
        try:
            worker.step_ready.disconnect(self._on_step_ready)
            worker.run_complete.disconnect(self._on_run_complete)
            worker.error.disconnect(self._on_worker_error)
            worker.progress.disconnect(self._experiment_panel.update_progress)
            self._canvas.frame_consumed.disconnect(worker.on_frame_consumed)
        except RuntimeError:
            pass  # already disconnected

    def _on_step_ready(self, snapshot: SimSnapshot) -> None:
        self._canvas.update_snapshot(snapshot)
        self._experiment_panel.update_status(snapshot)
        self._results_panel.update_step(snapshot)

    def _on_run_complete(self, result_id: str) -> None:
        self._speed_slider.setEnabled(False)
        self._experiment_panel.set_idle()
        self._results_panel.load_results(result_id)

    def _on_worker_error(self, message: str) -> None:
        self._speed_slider.setEnabled(False)
        self._experiment_panel.set_idle()
        QMessageBox.critical(self, "Simulation Error", message)

    def _on_speed_changed(self, value: int) -> None:
        """Relay slider position to the active worker (if any)."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.set_speed(value)
