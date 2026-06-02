"""pareto-router: predict each model's quality on a query, then route on the
cost-quality frontier. Trained and benchmarked on RouterBench."""
from __future__ import annotations

from .benchmark import BenchmarkReport, RoutePoint, run_benchmark
from .features import Featurizer, TfidfFeaturizer
from .model import RouteDecision, RouterModel
from .predictor import QualityPredictor
from .router import Router

__version__ = "0.1.0"

__all__ = [
    "Router",
    "RouterModel",
    "RouteDecision",
    "QualityPredictor",
    "TfidfFeaturizer",
    "Featurizer",
    "run_benchmark",
    "BenchmarkReport",
    "RoutePoint",
    "__version__",
]


def __getattr__(name):
    # Lazily expose the RouterBench loader so importing the package doesn't pull in
    # the optional [bench] dependencies (pandas / huggingface_hub).
    if name in ("load_routerbench", "RouterBench"):
        from . import data

        return getattr(data, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
