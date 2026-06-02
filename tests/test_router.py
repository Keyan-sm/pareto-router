import numpy as np

from pareto_router import Router


def test_lambda_extremes_pick_quality_then_cost():
    quality = np.array([0.9, 0.6, 0.3])
    cost = np.array([1.0, 0.1, 0.01])
    r = Router(cost_scale=1.0)
    assert r.select(quality, cost, lam=0.0) == 0   # pure quality -> highest quality
    assert r.select(quality, cost, lam=1.0) == 2   # pure cost -> cheapest


def test_higher_lambda_routes_cheaper_on_average():
    rng = np.random.default_rng(0)
    Q = rng.random((50, 4))
    C = rng.random((50, 4))
    r = Router.from_costs(C)
    rows = np.arange(50)
    cheap = C[rows, r.route(Q, C, lam=1.0)].mean()
    pricey = C[rows, r.route(Q, C, lam=0.0)].mean()
    assert cheap <= pricey


def test_from_costs_uses_max_mean_cost_as_scale():
    C = np.array([[1.0, 2.0], [3.0, 4.0]])  # mean per model = [2, 3] -> scale 3
    assert Router.from_costs(C).cost_scale == 3.0


def test_budget_selection_prefers_quality_then_falls_back_to_cheapest():
    quality = np.array([0.9, 0.6, 0.3])
    cost = np.array([1.0, 0.1, 0.01])
    r = Router()
    assert r.select_under_budget(quality, cost, budget=0.5) == 1   # best quality affordable
    assert r.select_under_budget(quality, cost, budget=0.001) == 2  # none affordable -> cheapest
