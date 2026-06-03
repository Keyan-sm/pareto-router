"""Metrics for cost-quality routing on a held-out set.

A router run produces a set of ``(cost, quality)`` points (one per ``lam``). These
helpers summarize that curve: which points are non-dominated, the cheapest way to
reach a target quality, and an area-under-the-curve scalar for comparing routers.
"""
from __future__ import annotations

import numpy as np

# numpy>=2 renames trapz -> trapezoid (and removes trapz in 2.x); support both.
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def pareto_frontier(costs, qualities) -> np.ndarray:
    """Indices of non-dominated points: cheaper *and* at least as good as everything
    to their left once sorted by cost."""
    costs = np.asarray(costs, dtype=float)
    qualities = np.asarray(qualities, dtype=float)
    order = np.argsort(costs)
    keep, best = [], -np.inf
    for i in order:
        if qualities[i] > best:
            keep.append(i)
            best = qualities[i]
    return np.array(keep, dtype=int)


def cost_at_quality(costs, qualities, target: float):
    """Minimum cost among points achieving >= ``target`` quality (None if unreachable)."""
    costs = np.asarray(costs, dtype=float)
    qualities = np.asarray(qualities, dtype=float)
    mask = qualities >= target
    return float(np.min(costs[mask])) if mask.any() else None


def quality_at_cost(costs, qualities, budget: float):
    """Maximum quality among points costing <= ``budget`` (None if none affordable)."""
    costs = np.asarray(costs, dtype=float)
    qualities = np.asarray(qualities, dtype=float)
    mask = costs <= budget
    return float(np.max(qualities[mask])) if mask.any() else None


def frontier_auc(costs, qualities, cost_cap: float) -> float:
    """Area under the achievable-quality curve over cost in [0, cost_cap], normalized
    to [0, 1]. Higher is better: more quality reachable per dollar."""
    costs = np.asarray(costs, dtype=float)
    qualities = np.asarray(qualities, dtype=float)
    order = np.argsort(costs)
    c = np.clip(costs[order], 0, cost_cap)
    q = np.maximum.accumulate(qualities[order])  # best quality reachable at <= each cost
    c = np.concatenate([[0.0], c, [cost_cap]])
    q = np.concatenate([[q[0]], q, [q[-1]]])
    return float(_trapz(q, c) / cost_cap) if cost_cap > 0 else 0.0
