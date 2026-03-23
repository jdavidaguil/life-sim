"""Benchmarking package — result loading and standardized figure generation."""

from src.benchmarking.reporter import (
    list_results,
    load_result,
    load_experiment_results,
    plot_population_history,
    plot_final_population,
    plot_trait_convergence,
    plot_mating_events,
    plot_comparison,
)

__all__ = [
    "list_results",
    "load_result",
    "load_experiment_results",
    "plot_population_history",
    "plot_final_population",
    "plot_trait_convergence",
    "plot_mating_events",
    "plot_comparison",
]
