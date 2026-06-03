"""pareto-router: predict each model's quality on a query, then route on the
cost-quality frontier. Benchmarked on current models (SPROUT)."""
from __future__ import annotations

from .benchmark import BenchmarkReport, RoutePoint, run_benchmark
from .features import Featurizer, TfidfFeaturizer
from .model import RouteDecision, RouterModel
from .predictor import QualityPredictor
from .router import Router

__version__ = "0.2.0"

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
    # Lazily expose the dataset loaders so importing the package doesn't pull in the
    # optional [bench] dependencies (pandas / huggingface_hub / pyarrow).
    if name in ("load_sprout", "load_routerbench", "load_frontier", "RoutingDataset", "RouterBench"):
        from . import data

        return getattr(data, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
