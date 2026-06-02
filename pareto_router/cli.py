"""Command line: ``pareto-router train | route | benchmark``."""
from __future__ import annotations

import argparse
import json


def _cmd_benchmark(args) -> None:
    from .benchmark import run_benchmark
    from .data import load_routerbench

    data = load_routerbench(split=args.split)
    report = run_benchmark(data, test_size=args.test_size, seed=args.seed)
    _print_report(report, verbose=args.verbose)


def _cmd_train(args) -> None:
    from .data import load_routerbench
    from .model import RouterModel

    data = load_routerbench(split=args.split)
    model = RouterModel.fit(data, alpha=args.alpha)
    model.save(args.out)
    print(f"trained on {len(data)} queries x {data.n_models} models -> saved {args.out}")


def _cmd_route(args) -> None:
    from .model import RouterModel

    model = RouterModel.load(args.model)
    decision = model.route(args.prompt, lam=args.lam)
    print(json.dumps(
        {
            "chosen": decision.model,
            "predicted_quality": round(decision.predicted_quality, 4),
            "est_cost": round(decision.est_cost, 6),
            "lam": decision.lam,
            "top3": [
                {"model": r["model"], "predicted_quality": round(r["predicted_quality"], 4)}
                for r in decision.ranking[:3]
            ],
        },
        indent=2,
    ))


def _print_report(report, verbose: bool = False) -> None:
    s = report.summary
    print(f"RouterBench reproduction  |  train={report.n_train}  test={report.n_test}  models={len(report.models)}")
    print(f"strongest single model : {s['strongest_model']}  "
          f"quality={s['strongest_model_quality']:.3f}  cost=${s['strongest_model_cost']:.5f}")
    cm = s.get("router_cost_to_match_strongest_quality")
    if cm is not None:
        print(f"match strongest quality: cost=${cm:.5f}  (savings {s.get('savings_at_strongest_quality_pct')}%)")
    c95 = s.get("router_cost_to_match_95pct_quality")
    if c95 is not None:
        print(f"match 95% of its quality: cost=${c95:.5f}  (savings {s.get('savings_at_95pct_quality_pct')}%)")
    print(f"frontier dominates     : {s['dominated_single_models']}/{s['n_single_models']} single models")
    print(f"frontier AUC (<= strongest cost): router={s['router_frontier_auc']:.4f}  "
          f"single-model envelope={s['single_model_frontier_auc']:.4f}")
    print(f"oracle (upper bound)   : quality={s['oracle_quality']:.3f}  cost=${s['oracle_cost']:.5f}")
    print("\nrouter vs best affordable single model, by budget:")
    print("  budget (xstrongest cost)  best single   router   advantage")
    for row in s["router_vs_best_single"]:
        bs = row["best_single_quality"]; rq = row["router_quality"]; adv = row["router_advantage_pts"]
        bs_s = f"{bs:.3f}" if bs is not None else "  -  "
        rq_s = f"{rq:.3f}" if rq is not None else "  -  "
        adv_s = f"{adv:+.1f} pts" if adv is not None else "  -  "
        print(f"  {row['budget_frac_of_strongest_cost']:<24}  {bs_s:>10}   {rq_s:>6}   {adv_s}")
    if verbose:
        print("\n   lam     cost$     quality")
        for p in report.router_curve:
            print(f"  {p.lam:<5}  {p.avg_cost:.6f}  {p.avg_quality:.4f}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="pareto-router",
                                     description="Cost-aware LLM router, trained and benchmarked on RouterBench.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_bench = sub.add_parser("benchmark", help="Reproduce the RouterBench cost-quality evaluation.")
    p_bench.add_argument("--split", default="0shot", choices=["0shot", "5shot"])
    p_bench.add_argument("--test-size", type=float, default=0.3)
    p_bench.add_argument("--seed", type=int, default=0)
    p_bench.add_argument("-v", "--verbose", action="store_true")

    p_train = sub.add_parser("train", help="Train a router and save it.")
    p_train.add_argument("--split", default="0shot", choices=["0shot", "5shot"])
    p_train.add_argument("--alpha", type=float, default=10.0)
    p_train.add_argument("--out", default="router.routermodel")

    p_route = sub.add_parser("route", help="Route a single prompt with a saved router.")
    p_route.add_argument("prompt")
    p_route.add_argument("--model", default="router.routermodel")
    p_route.add_argument("--lam", type=float, default=0.5)

    args = parser.parse_args(argv)
    {"benchmark": _cmd_benchmark, "train": _cmd_train, "route": _cmd_route}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
