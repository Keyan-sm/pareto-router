import numpy as np

from pareto_router import metrics


def test_cost_at_quality():
    cost = np.array([0.1, 0.2, 0.3])
    quality = np.array([0.5, 0.8, 0.9])
    assert metrics.cost_at_quality(cost, quality, 0.8) == 0.2
    assert metrics.cost_at_quality(cost, quality, 0.95) is None


def test_quality_at_cost():
    cost = np.array([0.1, 0.2, 0.3])
    quality = np.array([0.5, 0.8, 0.9])
    assert metrics.quality_at_cost(cost, quality, 0.25) == 0.8
    assert metrics.quality_at_cost(cost, quality, 0.05) is None


def test_pareto_frontier_drops_dominated_points():
    cost = np.array([0.1, 0.2, 0.3])
    quality = np.array([0.5, 0.4, 0.9])  # middle point is dominated by the first
    assert set(metrics.pareto_frontier(cost, quality).tolist()) == {0, 2}


def test_frontier_auc_rewards_a_better_curve():
    cost = np.array([0.1, 0.5])
    worse = metrics.frontier_auc(cost, np.array([0.4, 0.6]), cost_cap=1.0)
    better = metrics.frontier_auc(cost, np.array([0.6, 0.8]), cost_cap=1.0)
    assert better > worse
