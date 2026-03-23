"""Central registry of named :class:`~src.experiments.base.Experiment` objects.

Each entry is imported from its source module.  If a module fails to import
(missing dependency, no ``EXPERIMENT`` attribute, etc.) it is skipped with a
warning so the registry remains usable even when individual experiments are
broken or incomplete.

Usage::

    from src.experiments.registry import REGISTRY, get_experiment, list_experiments

    exp = get_experiment("phase6_sexual")
    exp.run()
"""

from __future__ import annotations

from src.experiments.base import Experiment

REGISTRY: dict[str, Experiment] = {}

# ── phase6_sexual ─────────────────────────────────────────────────────────────
try:
    from src.experiments.phase6_sexual import EXPERIMENT as _phase6_sexual
    _phase6_sexual.name = "Sexual Trait — All Conditions (canonical)"
    _phase6_sexual.description = "Canonical phase-6 run with sexual selection enabled across all conditions."
    REGISTRY["phase6_sexual"] = _phase6_sexual
except Exception as exc:
    print(f"[registry] WARNING: skipping 'phase6_sexual': {exc}")

# ── phase6_neural_sexual ──────────────────────────────────────────────────────
try:
    from src.experiments.phase6_neural_sexual import EXPERIMENT as _phase6_neural_sexual  # type: ignore[attr-defined]
    _phase6_neural_sexual.name = "Sexual Neural — All Conditions (canonical)"
    _phase6_neural_sexual.description = "Canonical phase-6 run combining neural genomes with sexual selection across all conditions."
    REGISTRY["phase6_neural_sexual"] = _phase6_neural_sexual
except Exception as exc:
    print(f"[registry] WARNING: skipping 'phase6_neural_sexual': {exc}")

# ── phase2_trait_volatility ───────────────────────────────────────────────────
try:
    from src.experiments.phase2_trait_volatility import EXPERIMENTS as _e2tv
    REGISTRY.update(_e2tv)
except Exception as exc:
    print(f"[registry] WARNING: skipping phase2_trait_volatility experiments: {exc}")

# ── phase3_richer_perception ──────────────────────────────────────────────────
try:
    from src.experiments.phase3_richer_perception import EXPERIMENTS as _e3rp
    REGISTRY.update(_e3rp)
except Exception as exc:
    print(f"[registry] WARNING: skipping phase3_richer_perception experiments: {exc}")

# ── phase4_neural_cold_start ──────────────────────────────────────────────────
try:
    from src.experiments.phase4_neural_cold_start import EXPERIMENTS as _e4nc
    REGISTRY.update(_e4nc)
except Exception as exc:
    print(f"[registry] WARNING: skipping phase4_neural_cold_start experiments: {exc}")

# ── phase4_neural_steep ───────────────────────────────────────────────────────
try:
    from src.experiments.phase4_neural_steep import EXPERIMENT as _phase4_neural_steep
    REGISTRY["phase4_neural_steep"] = _phase4_neural_steep
except Exception as exc:
    print(f"[registry] WARNING: skipping 'phase4_neural_steep': {exc}")

# ── phase5_internal_state ─────────────────────────────────────────────────────
try:
    from src.experiments.phase5_internal_state import EXPERIMENTS as _e5is
    REGISTRY.update(_e5is)
except Exception as exc:
    print(f"[registry] WARNING: skipping phase5_internal_state experiments: {exc}")

# ── phase6_neural_warm_baseline ───────────────────────────────────────────────
try:
    from src.experiments.phase6_neural_warm_baseline import EXPERIMENTS as _e6wb
    REGISTRY.update(_e6wb)
except Exception as exc:
    print(f"[registry] WARNING: skipping phase6_neural_warm_baseline experiments: {exc}")

# ── back-fill result_id where not already set ─────────────────────────────────
for _key, _exp in REGISTRY.items():
    if not _exp.result_id:
        _exp.result_id = _key


def list_experiments() -> list[str]:
    """Return a sorted list of registered experiment names."""
    return sorted(REGISTRY)


def get_experiment(name: str) -> Experiment:
    """Return the :class:`Experiment` registered under *name*.

    Raises:
        KeyError: with a message listing available names when *name* is not found.
    """
    if name not in REGISTRY:
        available = ", ".join(list_experiments()) or "<none>"
        raise KeyError(
            f"Unknown experiment {name!r}. Available: {available}"
        )
    return REGISTRY[name]
