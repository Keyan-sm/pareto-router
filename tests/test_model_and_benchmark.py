"""End-to-end tests on a tiny synthetic RouterBench (no network, no download).

The toy has two query families ('alpha', 'beta'); model A is best on alpha, model B
on beta, both cheap; model C is mediocre but priced high. A working router learns to
send each family to its cheap specialist, beating any single fixed model.
"""
import os
import tempfile

import numpy as np

from pareto_router import QualityPredictor, RouterModel, TfidfFeaturizer, run_benchmark
from pareto_router.data import RouterBench


def _toy(n_per: int = 60) -> RouterBench:
    prompts, evals, quality, cost = [], [], [], []
    for k in range(n_per):
        prompts.append(f"alpha topic number {k} alpha alpha question")
        evals.append("alpha")
        quality.append([0.9, 0.3, 0.7])
        cost.append([0.0001, 0.0001, 0.001])
    for k in range(n_per):
        prompts.append(f"beta matter number {k} beta beta question")
        evals.append("beta")
        quality.append([0.3, 0.9, 0.7])
        cost.append([0.0001, 0.0001, 0.001])
    return RouterBench(prompts, ["cheap_A", "cheap_B", "exp_C"],
                       np.array(quality), np.array(cost), np.array(evals))


def test_router_model_fit_route_and_persistence():
    data = _toy()
    model = RouterModel.fit(data, alpha=0.1, featurizer=TfidfFeaturizer(min_df=1, max_features=50))
    decision = model.route("alpha topic alpha question", lam=0.2)
    assert decision.model in data.models
    assert 0.0 <= decision.predicted_quality <= 1.0

    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "router.pkl")
    model.save(path)
    reloaded = RouterModel.load(path)
    assert reloaded.route("alpha topic alpha question", lam=0.2).model == decision.model


def test_benchmark_router_beats_single_models_on_toy():
    data = _toy()
    report = run_benchmark(
        data,
        featurizer=TfidfFeaturizer(min_df=1, max_features=50),
        predictor=QualityPredictor(alpha=0.1),
        test_size=0.3,
        seed=0,
    )
    s = report.summary
    assert len(report.router_curve) == 21
    assert report.n_train > 0 and report.n_test > 0
    assert 0.0 <= s["router_max_quality"] <= 1.0
    # Routing each family to its specialist beats the best single model (exp_C avg 0.7).
    assert s["router_max_quality"] > 0.75
    # The router's frontier should be at least as good as the single-model envelope.
    assert s["router_frontier_auc"] >= s["single_model_frontier_auc"] - 1e-9
    assert 0 <= s["dominated_single_models"] <= s["n_single_models"]
