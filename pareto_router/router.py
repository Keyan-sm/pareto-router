"""The routing decision: given predicted per-model quality and per-model cost, pick a
model by trading quality against cost.

Two modes, both standard in cost-aware routing:

* **lambda trade-off** — maximize ``S = (1 - lam) * quality - lam * cost_norm``.
  ``lam = 0`` is pure quality (route to the highest predicted quality, ignoring price);
  ``lam = 1`` is pure cost (always the cheapest). Sweeping ``lam`` from 0 to 1 traces
  the cost-quality frontier. This is the general trade-off objective R2-Router
  optimizes; R2-Router additionally treats an output-length budget as a control
  variable (out of scope here — see the README "Scope" section).

* **budget** — pick the highest predicted-quality model whose cost is within a budget.

Costs are normalized by a fixed pool scale (set at fit time via :meth:`from_costs`) so
``lam`` is comparable to the [0, 1] quality scale and transfers across queries.
"""
from __future__ import annotations

import numpy as np


class Router:
    """Cost-aware selection over a fixed model pool."""

    def __init__(self, cost_scale: float = 1.0):
        self.cost_scale = float(cost_scale) if cost_scale else 1.0

    @classmethod
    def from_costs(cls, cost_matrix: np.ndarray) -> "Router":
        """Set the normalizer to the most-expensive model's mean cost, so normalized
        costs land roughly in [0, 1] alongside quality."""
        scale = float(np.nanmax(np.asarray(cost_matrix).mean(axis=0)))
        return cls(cost_scale=scale or 1.0)

    def score(self, quality: np.ndarray, cost: np.ndarray, lam: float) -> np.ndarray:
        quality = np.asarray(quality, dtype=float)
        cost = np.asarray(cost, dtype=float)
        return (1.0 - lam) * quality - lam * (cost / self.cost_scale)

    def select(self, quality, cost, lam: float = 0.5) -> int:
        """Index of the chosen model for a single query."""
        return int(np.argmax(self.score(quality, cost, lam)))

    def route(self, quality_matrix, cost_matrix, lam: float = 0.5) -> np.ndarray:
        """Vectorized selection over (n, m) quality and cost matrices -> (n,) indices."""
        return np.argmax(self.score(quality_matrix, cost_matrix, lam), axis=1)

    def select_under_budget(self, quality, cost, budget: float) -> int:
        """Highest-quality model whose cost is within ``budget`` (else the cheapest)."""
        quality = np.asarray(quality, dtype=float)
        cost = np.asarray(cost, dtype=float)
        affordable = np.where(cost <= budget)[0]
        if affordable.size == 0:
            return int(np.argmin(cost))
        return int(affordable[np.argmax(quality[affordable])])
