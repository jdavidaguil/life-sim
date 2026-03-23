"""Benchmarking reporter: load result JSON files and produce Matplotlib figures."""

from __future__ import annotations

import glob
import json
import os
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── constants ─────────────────────────────────────────────────────────────────

_RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "results"
)

# Dark-theme palette (matches simulation.py's plot_history)
_BG_FIGURE = "#111111"
_BG_AXES   = "#1a1a1a"
_TEXT      = "#dddddd"
_GRID      = "#333333"
_SPINE     = "#444444"
_LEGEND_BG = "#222222"
_LEGEND_ED = "#555555"
_SUBTLE    = "#aaaaaa"

# Per-seed line colours (cycles if more seeds than colours)
_SEED_COLORS = [
    "#ffffff", "#4488ff", "#ff4444", "#44cc44",
    "#ffaa00", "#c77dca", "#4ecdc4", "#ffe66d",
]

# Trait colours — same as simulation.py
_TRAIT_COLORS = {
    "rw":    "#ff4444",
    "cs":    "#4488ff",
    "noise": "#44cc44",
    "ea":    "#ffaa00",
}


# ── internal helpers ──────────────────────────────────────────────────────────

def _style_axes(ax: plt.Axes, ylabel: str = "", xlabel: str = "") -> None:
    """Apply the standard dark theme to a single Axes."""
    ax.set_facecolor(_BG_AXES)
    if ylabel:
        ax.set_ylabel(ylabel, color=_SUBTLE)
    if xlabel:
        ax.set_xlabel(xlabel, color=_SUBTLE)
    ax.tick_params(colors=_SUBTLE)
    ax.grid(color=_GRID, linewidth=0.5, linestyle="--")
    for spine in ax.spines.values():
        spine.set_edgecolor(_SPINE)


def _style_legend(ax: plt.Axes) -> None:
    """Apply dark theme to an axes legend (if present)."""
    leg = ax.get_legend()
    if leg is not None:
        leg.get_frame().set_facecolor(_LEGEND_BG)
        leg.get_frame().set_edgecolor(_LEGEND_ED)
        for text in leg.get_texts():
            text.set_color(_TEXT)


def _make_figure(nrows: int = 1, ncols: int = 1, **kwargs) -> tuple[plt.Figure, Any]:
    """Create a dark-themed figure; returns (fig, axes)."""
    fig, axes = plt.subplots(nrows, ncols, **kwargs)
    fig.patch.set_facecolor(_BG_FIGURE)
    return fig, axes


def _result_id_from(results: list[dict]) -> str:
    return results[0].get("result_id", "") if results else ""


# ── data loading ──────────────────────────────────────────────────────────────

def list_results() -> list[dict]:
    """Return metadata dicts for every JSON file in results/, newest first.

    Each dict contains: ``result_id``, ``seed``, ``steps``, ``timestamp``,
    ``final_population``, ``filepath``.
    """
    pattern = os.path.join(_RESULTS_DIR, "*.json")
    rows: list[dict] = []
    for path in glob.glob(pattern):
        try:
            with open(path) as fh:
                data = json.load(fh)
            rows.append(
                {
                    "result_id":        data.get("result_id", ""),
                    "seed":             data.get("seed"),
                    "steps":            data.get("steps"),
                    "timestamp":        data.get("timestamp", ""),
                    "final_population": data.get("final_population"),
                    "filepath":         path,
                }
            )
        except Exception:
            pass  # skip malformed files silently
    rows.sort(key=lambda r: r["timestamp"], reverse=True)
    return rows


def load_result(filepath: str) -> dict:
    """Load and return a single result JSON file as a dict."""
    with open(filepath) as fh:
        return json.load(fh)


def load_experiment_results(result_id: str) -> list[dict]:
    """Load all saved runs for *result_id*, sorted by seed ascending."""
    rows: list[dict] = []
    pattern = os.path.join(_RESULTS_DIR, f"{result_id}_*.json")
    for path in glob.glob(pattern):
        try:
            rows.append(load_result(path))
        except Exception:
            pass
    rows.sort(key=lambda r: r.get("seed", 0))
    return rows


# ── plot helpers ──────────────────────────────────────────────────────────────

def _population_series(result: dict) -> list[int] | None:
    """Extract per-step population from metrics, or None if not available."""
    metrics = result.get("metrics") or {}
    # Common key names written by recorder hooks
    for key in ("population", "total_population", "pop"):
        val = metrics.get(key)
        if isinstance(val, list) and val:
            return val
    return None


def _trait_series(result: dict, trait_key: str) -> list[float] | None:
    """Return the per-step list for *trait_key* from metrics, or None."""
    metrics = result.get("metrics") or {}
    val = metrics.get(trait_key)
    if isinstance(val, list) and val:
        return val
    return None


# ── public plot functions ─────────────────────────────────────────────────────

def plot_population_history(results: list[dict]) -> plt.Figure:
    """Line chart of population over steps, one line per seed.

    Uses ``metrics["population_history"]`` / ``metrics["step_history"]`` when
    available; falls back to a single final-population point otherwise.
    Title: ``'{result_id} — Population History'``
    """
    rid = results[0].get("result_id", "") if results else ""

    # Keep only most recent result per seed.
    seen: dict = {}
    for r in sorted(results, key=lambda x: x.get("timestamp", "")):
        seen[r.get("seed")] = r
    results = list(seen.values())

    fig, ax = _make_figure(figsize=(10, 5))
    ax.set_title(f"{rid} — Population History", color=_TEXT)

    handles = []
    labels = []

    for i, result in enumerate(results):
        seed = result.get("seed", i)
        color = _SEED_COLORS[i % len(_SEED_COLORS)]
        metrics = result.get("metrics") or {}
        pop_hist = metrics.get("population_history")
        step_hist = metrics.get("step_history")

        if isinstance(pop_hist, list) and pop_hist:
            xs = step_hist if isinstance(step_hist, list) and len(step_hist) == len(pop_hist) else range(len(pop_hist))
            lines = ax.plot(xs, pop_hist, color=color, lw=1.2)
        else:
            # Fall back to single final-population point
            lines = ax.plot(
                [result.get("steps", 0)],
                [result.get("final_population", 0)],
                "o", color=color, ms=5,
            )

        # Always register exactly one legend entry per seed.
        handles.append(lines[0])
        labels.append(f"seed {seed}")

    _style_axes(ax, ylabel="Population", xlabel="Step")
    if handles:
        ax.legend(handles, labels, facecolor=_LEGEND_BG, edgecolor=_LEGEND_ED, labelcolor=_TEXT, fontsize=8)
    _style_legend(ax)
    fig.tight_layout()
    return fig


def plot_final_population(results: list[dict]) -> plt.Figure:
    """Bar chart of final population per seed with a mean line.

    Title: ``'{result_id} — Final Population by Seed'``
    """
    # Keep only most recent result per seed.
    seen: dict = {}
    for r in sorted(results, key=lambda x: x.get("timestamp", "")):
        seen[r.get("seed")] = r
    results = list(seen.values())

    rid = _result_id_from(results)
    fig, ax = _make_figure(figsize=(8, 5))
    ax.set_title(f"{rid} — Final Population by Seed", color=_TEXT)

    seeds = [r.get("seed", i) for i, r in enumerate(results)]
    pops  = [r.get("final_population", 0) for r in results]
    xs    = np.arange(len(seeds))

    bars = ax.bar(xs, pops, color=_SEED_COLORS[1], alpha=0.8, width=0.6)
    for bar, pop in zip(bars, pops):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(pops) * 0.01,
            str(pop),
            ha="center", va="bottom", color=_TEXT, fontsize=8,
        )

    mean_pop = float(np.mean(pops)) if pops else 0.0
    ax.axhline(mean_pop, color="#ffaa00", lw=1.5, linestyle="--",
               label=f"mean = {mean_pop:.1f}")

    ax.set_xticks(xs)
    ax.set_xticklabels([str(s) for s in seeds])
    _style_axes(ax, ylabel="Final population", xlabel="Seed")
    ax.legend(facecolor=_LEGEND_BG, edgecolor=_LEGEND_ED, labelcolor=_TEXT, fontsize=8)
    _style_legend(ax)
    fig.tight_layout()
    return fig


def plot_trait_convergence(results: list[dict]) -> plt.Figure:
    """Four-panel line chart of mean trait values (rw, cs, noise, ea) over steps.

    Only renders data for seeds that actually contain trait metrics.
    Title: ``'{result_id} — Trait Convergence'``
    """
    # Keep only most recent result per seed.
    seen: dict = {}
    for r in sorted(results, key=lambda x: x.get("timestamp", "")):
        seen[r.get("seed")] = r
    results = list(seen.values())

    rid = _result_id_from(results)
    trait_keys  = ["mean_resource_weight", "mean_crowd_sensitivity",
                   "mean_noise", "mean_energy_awareness"]
    short_names = ["rw", "cs", "noise", "ea"]
    ylabels     = ["Resource weight", "Crowd sensitivity", "Noise", "Energy awareness"]

    fig, axes = _make_figure(2, 2, figsize=(12, 8), sharex=False)
    axes_flat = axes.flatten()
    fig.suptitle(f"{rid} — Trait Convergence", color=_TEXT, fontsize=12)

    for col_idx, (tkey, sname, ylabel) in enumerate(
        zip(trait_keys, short_names, ylabels)
    ):
        ax = axes_flat[col_idx]
        ax.set_title(sname, color=_TEXT, fontsize=10)
        plotted = False
        for i, result in enumerate(results):
            series = _trait_series(result, tkey)
            if series is not None:
                seed = result.get("seed", i)
                color = _SEED_COLORS[i % len(_SEED_COLORS)]
                ax.plot(series, color=color, lw=1.0, alpha=0.85,
                        label=f"seed {seed}")
                plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color=_SUBTLE, fontsize=10)
        _style_axes(ax, ylabel=ylabel, xlabel="Step")
        if plotted:
            ax.legend(facecolor=_LEGEND_BG, edgecolor=_LEGEND_ED,
                      labelcolor=_TEXT, fontsize=7)
            _style_legend(ax)

    fig.tight_layout()
    return fig


def plot_mating_events(results: list[dict]) -> plt.Figure:
    """Line chart of mating events per step, one line per seed.

    Only renders data for seeds that contain ``mating_events`` in metrics.
    Title: ``'{result_id} — Mating Events per Step'``
    """
    # Keep only most recent result per seed.
    seen: dict = {}
    for r in sorted(results, key=lambda x: x.get("timestamp", "")):
        seen[r.get("seed")] = r
    results = list(seen.values())

    rid = _result_id_from(results)
    fig, ax = _make_figure(figsize=(10, 5))
    ax.set_title(f"{rid} — Mating Events per Step", color=_TEXT)

    plotted = False
    for i, result in enumerate(results):
        series = _trait_series(result, "mating_events")
        if series is not None:
            seed = result.get("seed", i)
            color = _SEED_COLORS[i % len(_SEED_COLORS)]
            ax.plot(series, color=color, lw=1.2, label=f"seed {seed}")
            plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "no mating_events data", transform=ax.transAxes,
                ha="center", va="center", color=_SUBTLE, fontsize=12)

    _style_axes(ax, ylabel="Mating events", xlabel="Step")
    if plotted:
        ax.legend(facecolor=_LEGEND_BG, edgecolor=_LEGEND_ED,
                  labelcolor=_TEXT, fontsize=8)
        _style_legend(ax)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    return fig


def plot_comparison(result_ids: list[str]) -> plt.Figure:
    """Bar chart comparing mean final population across experiments.

    Error bars show std across seeds.
    Title: ``'Experiment Comparison — Final Population'``
    """
    fig, ax = _make_figure(figsize=(max(6, len(result_ids) * 1.5 + 2), 5))
    ax.set_title("Experiment Comparison — Final Population", color=_TEXT)

    xs    = np.arange(len(result_ids))
    means = []
    stds  = []

    for rid in result_ids:
        runs = load_experiment_results(rid)
        pops = [r.get("final_population", 0) for r in runs] if runs else [0]
        means.append(float(np.mean(pops)))
        stds.append(float(np.std(pops)))

    bars = ax.bar(
        xs, means,
        yerr=stds,
        capsize=5,
        color=[_SEED_COLORS[i % len(_SEED_COLORS)] for i in range(len(result_ids))],
        alpha=0.85,
        width=0.5,
        error_kw={"ecolor": _SUBTLE, "lw": 1.2},
    )
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(means) * 0.015 + std,
            f"{mean:.1f}",
            ha="center", va="bottom", color=_TEXT, fontsize=8,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(result_ids, rotation=15, ha="right")
    _style_axes(ax, ylabel="Mean final population", xlabel="Experiment")
    fig.tight_layout()
    return fig
