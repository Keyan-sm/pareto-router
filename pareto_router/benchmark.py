"""Reproducible RouterBench evaluation.

Train the router on one split of RouterBench, then evaluate the cost-quality frontier
it traces on *held-out* queries (no leakage), against fixed reference points: every
single model, a random router, and the per-query oracle (the cheapest model that
attains the best available quality — the upper bound any router could reach).

The router makes its decisions from *predicted* quality (so it generalizes to unseen
queries), but realized quality and cost are read from the *true* RouterBench values
for the chosen model. That is the standard way RouterBench-style routers are scored.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from . import metrics
from .data import RoutingDataset
from .features import TfidfFeaturizer
from .predictor import QualityPredictor
from .router import Router


@dataclass
class RoutePoint:
    lam: float
    avg_cost: float
    avg_quality: float


@dataclass
class BenchmarkReport:
    models: List[str]
    router_curve: List[RoutePoint]
    baselines: Dict[str, dict]
    summary: Dict[str, object]
    n_train: int
    n_test: int


def _stratified_split(eval_name, test_size, seed):
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(eval_name))
    try:
        return train_test_split(idx, test_size=test_size, random_state=seed, stratify=eval_name)
    except ValueError:  # some eval set too small to stratify
        return train_test_split(idx, test_size=test_size, random_state=seed)


def run_benchmark(
    data: RoutingDataset,
    lambdas: Optional[List[float]] = None,
    test_size: float = 0.3,
    seed: int = 0,
    featurizer=None,
    predictor=None,
) -> BenchmarkReport:
    if lambdas is None:
        lambdas = [round(x, 3) for x in np.linspace(0.0, 1.0, 21)]

    train_idx, test_idx = _stratified_split(data.eval_name, test_size, seed)
    n_test = len(test_idx)
    rows = np.arange(n_test)

    featurizer = featurizer or TfidfFeaturizer()
    X_train = featurizer.fit_transform([data.prompts[i] for i in train_idx])
    X_test = featurizer.transform([data.prompts[i] for i in test_idx])

    predictor = predictor or QualityPredictor()
    predictor.fit(X_train, data.quality[train_idx])
    q_hat = predictor.predict(X_test)              # predicted quality on held-out queries

    q_true = data.quality[test_idx]
    c_true = data.cost[test_idx]
    router = Router.from_costs(data.cost[train_idx])

    # Router frontier: decide from predicted quality, score with true outcomes.
    curve: List[RoutePoint] = []
    for lam in lambdas:
        choice = router.route(q_hat, c_true, lam)
        curve.append(
            RoutePoint(
                lam=float(lam),
                avg_cost=float(c_true[rows, choice].mean()),
                avg_quality=float(q_true[rows, choice].mean()),
            )
        )

    # Reference points.
    baselines: Dict[str, dict] = {}
    for j, name in enumerate(data.models):
        baselines[name] = {
            "cost": float(c_true[:, j].mean()),
            "quality": float(q_true[:, j].mean()),
            "kind": "single_model",
        }
    rng = np.random.default_rng(seed)
    rand_choice = rng.integers(0, data.n_models, size=n_test)
    baselines["random"] = {
        "cost": float(c_true[rows, rand_choice].mean()),
        "quality": float(q_true[rows, rand_choice].mean()),
        "kind": "baseline",
    }
    # Oracle: per query, cheapest model that reaches the best available quality.
    best_q = q_true.max(axis=1, keepdims=True)
    reachable = q_true >= best_q - 1e-9
    masked_cost = np.where(reachable, c_true, np.inf)
    oracle_choice = masked_cost.argmin(axis=1)
    baselines["oracle"] = {
        "cost": float(c_true[rows, oracle_choice].mean()),
        "quality": float(q_true[rows, oracle_choice].mean()),
        "kind": "oracle",
    }

    summary = _summarize(data.models, curve, baselines, c_true)
    return BenchmarkReport(
        models=data.models,
        router_curve=curve,
        baselines=baselines,
        summary=summary,
        n_train=len(train_idx),
        n_test=n_test,
    )


def _summarize(models, curve, baselines, c_true) -> Dict[str, object]:
    strongest = max(models, key=lambda m: baselines[m]["quality"])
    sq = baselines[strongest]["quality"]
    sc = baselines[strongest]["cost"]
    rc = np.array([p.avg_cost for p in curve])
    rq = np.array([p.avg_quality for p in curve])

    single = [(m, baselines[m]["cost"], baselines[m]["quality"]) for m in models]
    smc = np.array([c for _, c, _ in single])
    smq = np.array([q for _, _, q in single])

    # At each budget (a fraction of the strongest model's cost), compare the router's
    # best achievable quality to the best *single* model you could afford instead.
    vs_best_single = []
    for frac in (0.05, 0.15, 0.30, 0.60, 1.00):
        budget = frac * sc
        affordable = [q for _, c, q in single if c <= budget]
        best_single_q = max(affordable) if affordable else None
        mask = rc <= budget
        router_q = float(rq[mask].max()) if mask.any() else None
        adv = (round(100 * (router_q - best_single_q), 1)
               if router_q is not None and best_single_q is not None else None)
        vs_best_single.append({
            "budget_frac_of_strongest_cost": frac,
            "best_single_quality": best_single_q,
            "router_quality": router_q,
            "router_advantage_pts": adv,
        })

    dominated = int(sum(1 for _, c, q in single if np.any((rc <= c) & (rq >= q))))

    summary: Dict[str, object] = {
        "strongest_model": strongest,
        "strongest_model_quality": sq,
        "strongest_model_cost": sc,
        "router_max_quality": float(rq.max()),
        "router_quality_at_strongest_cost": metrics.quality_at_cost(rc, rq, sc),
        "router_cost_to_match_strongest_quality": metrics.cost_at_quality(rc, rq, sq),
        "router_cost_to_match_95pct_quality": metrics.cost_at_quality(rc, rq, 0.95 * sq),
        "oracle_quality": baselines["oracle"]["quality"],
        "oracle_cost": baselines["oracle"]["cost"],
        # Fair AUC: router frontier vs the envelope of all single models, same cost cap.
        "router_frontier_auc": metrics.frontier_auc(rc, rq, sc),
        "single_model_frontier_auc": metrics.frontier_auc(smc, smq, sc),
        "dominated_single_models": dominated,
        "n_single_models": len(single),
        "router_vs_best_single": vs_best_single,
    }
    cm = summary["router_cost_to_match_strongest_quality"]
    if cm is not None and sc > 0:
        summary["savings_at_strongest_quality_pct"] = round(100 * (sc - cm) / sc, 1)
    c95 = summary["router_cost_to_match_95pct_quality"]
    if c95 is not None and sc > 0:
        summary["savings_at_95pct_quality_pct"] = round(100 * (sc - c95) / sc, 1)
    return summary
