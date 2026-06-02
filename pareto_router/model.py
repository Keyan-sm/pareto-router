"""RouterModel: the top-level, trainable, savable router you actually use.

Bundles a featurizer + quality predictor + the cost-aware Router, plus the model
pool and the per-model mean cost (used as the default cost estimate when routing a
live prompt for which the true per-model cost isn't known yet).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .features import TfidfFeaturizer
from .predictor import QualityPredictor
from .router import Router


@dataclass
class RouteDecision:
    model: str
    predicted_quality: float
    est_cost: float
    lam: float
    ranking: List[dict]


class RouterModel:
    """A fitted router: prompt -> model choice."""

    def __init__(self, featurizer, predictor, router, models, mean_cost):
        self.featurizer = featurizer
        self.predictor = predictor
        self.router = router
        self.models = list(models)
        self.mean_cost = np.asarray(mean_cost, dtype=float)

    # --- training ---------------------------------------------------------
    @classmethod
    def fit(cls, data, alpha: float = 10.0, featurizer=None) -> "RouterModel":
        """Fit on a :class:`~pareto_router.data.RouterBench` (or anything with the same
        ``prompts`` / ``quality`` / ``cost`` / ``models`` attributes)."""
        featurizer = featurizer or TfidfFeaturizer()
        X = featurizer.fit_transform(data.prompts)
        predictor = QualityPredictor(alpha=alpha).fit(X, data.quality)
        router = Router.from_costs(data.cost)
        mean_cost = np.asarray(data.cost).mean(axis=0)
        return cls(featurizer, predictor, router, data.models, mean_cost)

    # --- inference --------------------------------------------------------
    def predict_quality(self, prompt: str) -> np.ndarray:
        return self.predictor.predict(self.featurizer.transform([prompt]))[0]

    def route(self, prompt: str, lam: float = 0.5, cost: Optional[np.ndarray] = None) -> RouteDecision:
        """Choose a model for ``prompt``. ``cost`` defaults to the per-model mean cost
        observed in training (a rough live estimate; pass a real per-model cost vector
        when you have one)."""
        quality = self.predict_quality(prompt)
        cost_vec = self.mean_cost if cost is None else np.asarray(cost, dtype=float)
        idx = self.router.select(quality, cost_vec, lam)
        ranking = sorted(
            (
                {"model": m, "predicted_quality": float(quality[j]), "est_cost": float(cost_vec[j])}
                for j, m in enumerate(self.models)
            ),
            key=lambda r: r["predicted_quality"],
            reverse=True,
        )
        return RouteDecision(
            model=self.models[idx],
            predicted_quality=float(quality[idx]),
            est_cost=float(cost_vec[idx]),
            lam=lam,
            ranking=ranking,
        )

    # --- persistence ------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str) -> "RouterModel":
        with open(path, "rb") as fh:
            return pickle.load(fh)
