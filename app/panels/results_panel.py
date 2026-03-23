"""Bottom results panel: tabbed Matplotlib figures rendered from saved JSON results."""

from __future__ import annotations

import time
import traceback

# Minimum seconds between live chart re-renders while a run is in progress.
_CHART_REFRESH_INTERVAL = 2.0

# Backend is configured in app/main.py before this module is ever imported.
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import matplotlib.pyplot as plt

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal as _Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QAbstractItemView,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.benchmarking.reporter import (
    load_experiment_results,
    list_results,
    plot_final_population,
    plot_mating_events,
    plot_population_history,
    plot_trait_convergence,
    plot_comparison,
)

from app.snapshot import SimSnapshot


# ---------------------------------------------------------------------------
# Off-thread chart rendering
# ---------------------------------------------------------------------------

class _ChartSignals(QObject):
    """Carries the (tab, figure) pair from the thread pool back to the main thread."""
    ready = _Signal(object, object)  # (tab: _PlotTab, fig: plt.Figure)


class _ChartJob(QRunnable):
    """Builds a matplotlib Figure on a thread-pool thread."""

    def __init__(self, tab, plot_fn, data, signals: _ChartSignals) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self._tab = tab
        self._fn = plot_fn
        self._data = data
        self._signals = signals

    def run(self) -> None:  # noqa: D102
        try:
            fig = self._fn(self._data)
            self._signals.ready.emit(self._tab, fig)
        except Exception:  # noqa: BLE001
            traceback.print_exc()


class _PlotTab(QWidget):
    """A tab that holds a single Matplotlib FigureCanvasQTAgg."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        # Placeholder empty figure
        self._fig, _ = plt.subplots()
        self._fig.patch.set_facecolor("#111111")
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._layout.addWidget(self._canvas)

    @property
    def figure(self) -> plt.Figure:
        return self._fig

    def set_figure(self, fig: plt.Figure) -> None:
        """Replace the canvas widget with a fresh one bound to *fig*."""
        old_fig = self._fig
        old_canvas = self._canvas
        self._fig = fig
        self._canvas = FigureCanvasQTAgg(fig)
        self._canvas.setMinimumHeight(160)
        self._layout.replaceWidget(old_canvas, self._canvas)
        self._canvas.show()
        old_canvas.hide()
        old_canvas.deleteLater()
        plt.close(old_fig)
        QTimer.singleShot(0, self._canvas.draw)


class _CompareTab(QWidget):
    """Compare tab: multi-select list of result_ids + Refresh button."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        root.addWidget(QLabel("Select experiments to compare:"))

        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self._populate_list()
        root.addWidget(self._list)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self._on_refresh)
        root.addWidget(self._refresh_btn)

        # Canvas for the comparison plot
        self._fig, _ = plt.subplots()
        self._fig.patch.set_facecolor("#111111")
        self._canvas = FigureCanvasQTAgg(self._fig)
        root.addWidget(self._canvas)

    @property
    def figure(self) -> plt.Figure:
        return self._fig

    def _populate_list(self) -> None:
        self._list.clear()
        rows = list_results()
        seen: set[str] = set()
        for row in rows:
            rid = row.get("result_id", "")
            if rid and rid not in seen:
                seen.add(rid)
                self._list.addItem(rid)

    def _on_refresh(self) -> None:
        # Capture selection before _populate_list() calls clear(), which wipes it.
        selected = [item.text() for item in self._list.selectedItems()]
        self._populate_list()
        if not selected:
            return
        fig = plot_comparison(selected)
        old_fig = self._fig
        old_canvas = self._canvas
        self._fig = fig
        self._canvas = FigureCanvasQTAgg(fig)
        self._canvas.setMinimumHeight(160)
        self.layout().replaceWidget(old_canvas, self._canvas)
        self._canvas.show()
        old_canvas.hide()
        old_canvas.deleteLater()
        plt.close(old_fig)
        QTimer.singleShot(0, self._canvas.draw)

    def refresh_list(self) -> None:
        """Called externally after a new run completes."""
        self._populate_list()


class ResultsPanel(QWidget):
    """Tabbed panel that renders reporter plots for a given result_id.

    Tabs:
        Population History, Final Population, Trait Convergence,
        Mating Events, Compare
    """

    status_message: _Signal = _Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 4, 0, 0)

        # Header row: label + Load Latest button + Export PNG button
        header = QHBoxLayout()
        header.addWidget(QLabel("Results"))
        header.addStretch()
        self._export_btn = QPushButton()
        self._export_btn.setText("Export PNG")
        self._export_btn.setMinimumWidth(110)
        self._export_btn.setFixedWidth(110)
        self._export_btn.clicked.connect(self._export_png)
        header.addWidget(self._export_btn)
        self._load_btn = QPushButton("Load Latest")
        self._load_btn.setFixedWidth(100)
        self._load_btn.clicked.connect(self._load_latest)
        header.addWidget(self._load_btn)
        root.addLayout(header)

        self._tabs = QTabWidget()
        root.addWidget(self._tabs)

        self._pop_history_tab = _PlotTab()
        self._final_pop_tab = _PlotTab()
        self._trait_tab = _PlotTab()
        self._mating_tab = _PlotTab()
        self._compare_tab = _CompareTab()

        self._tabs.addTab(self._pop_history_tab, "Population History")
        self._tabs.addTab(self._final_pop_tab, "Final Population")
        self._tabs.addTab(self._trait_tab, "Trait Convergence")
        self._tabs.addTab(self._mating_tab, "Mating Events")
        self._tabs.addTab(self._compare_tab, "Compare")

        # Async chart rendering: one shared signal object + busy flag.
        self._chart_signals = _ChartSignals()
        self._chart_signals.ready.connect(self._on_chart_ready)
        self._chart_busy: bool = False

        # Auto-load the most recent result that exists on disk
        QTimer.singleShot(0, self._load_latest)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def begin_run(self, seeds: list[int]) -> None:
        """Reset live-data buffers when a new run starts."""
        self._live_seeds: list[int] = list(seeds)
        self._live_pop: list[int] = []
        self._live_steps: list[int] = []
        self._live_prev_step: int = -1
        self._live_seed_idx: int = 0
        self._live_seed_starts: list[int] = [0]

    def update_step(self, snapshot: SimSnapshot) -> None:
        """Accumulate one live snapshot and trigger a chart refresh if the
        previous render has already completed."""
        if not hasattr(self, "_live_pop"):
            self.begin_run([])

        # Detect seed boundary: step counter went backwards → new seed started.
        if self._live_steps and snapshot.step <= self._live_prev_step:
            self._live_seed_starts.append(len(self._live_pop))
            self._live_seed_idx += 1

        self._live_prev_step = snapshot.step
        self._live_pop.append(snapshot.population)
        self._live_steps.append(snapshot.step)

        self._render_live_pop_history()

    def _load_latest(self) -> None:
        """Load the most recent result_id found on disk."""
        rows = list_results()
        if rows:
            self.load_results(rows[0]["result_id"])

    def _export_png(self) -> None:
        """Save the currently visible chart tab as a PNG file."""
        tab = self._tabs.currentWidget()
        fig = getattr(tab, "figure", None) or getattr(tab, "_fig", None)
        if fig is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export PNG", "", "PNG Images (*.png)"
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        self.status_message.emit(f"Saved: {path}")

    def load_results(self, result_id: str) -> None:
        """Load saved runs for *result_id* and render each plot tab.

        Tabs whose data is missing (e.g. no trait series) are skipped
        gracefully by the underlying reporter functions (they render a
        "no data" placeholder internally).  When no disk files are found
        the live population-history chart is preserved as-is.
        """
        results = load_experiment_results(result_id)
        if not results:
            # No disk data — do a final live render so the chart is up to date.
            self._render_live_pop_history()
            return

        self._render_tab_async(self._pop_history_tab, plot_population_history, results)
        self._render_tab_async(self._final_pop_tab, plot_final_population, results)
        self._render_tab_async(self._trait_tab, plot_trait_convergence, results)
        self._render_tab_async(self._mating_tab, plot_mating_events, results)

        # Refresh the compare list so the new result_id is available.
        self._compare_tab.refresh_list()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _render_tab_async(self, tab: _PlotTab, plot_fn, results: list[dict]) -> None:
        """Submit a chart render job to the global thread pool.

        The main thread is never blocked — the figure is built on a worker
        thread and swapped in via :meth:`_on_chart_ready` once ready.
        """
        job = _ChartJob(tab, plot_fn, results, self._chart_signals)
        QThreadPool.globalInstance().start(job)

    def _on_chart_ready(self, tab: _PlotTab, fig) -> None:
        """Receive a rendered figure from the thread pool and display it."""
        self._chart_busy = False
        tab.set_figure(fig)

    def _render_live_pop_history(self) -> None:
        """Build and display a Population History chart from in-memory live data.

        Rendering is dispatched to the thread pool so the main thread is never
        blocked.  Concurrent renders are skipped (latest data wins next time).
        """
        if not hasattr(self, "_live_pop") or not self._live_pop:
            return
        if self._chart_busy:
            return  # a render is already in flight; skip this one
        # Slice per-seed segments.
        starts = self._live_seed_starts + [len(self._live_pop)]
        fake_results = []
        for i in range(len(starts) - 1):
            s, e = starts[i], starts[i + 1]
            if e <= s:
                continue
            seed_label = self._live_seeds[i] if i < len(self._live_seeds) else i
            fake_results.append({
                "seed": seed_label,
                "steps": self._live_steps[e - 1],
                "final_population": self._live_pop[e - 1],
                "metrics": {
                    "population_history": list(self._live_pop[s:e]),
                    "step_history": list(self._live_steps[s:e]),
                },
            })
        if fake_results:
            self._chart_busy = True
            self._render_tab_async(self._pop_history_tab, plot_population_history, fake_results)
