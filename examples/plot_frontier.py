"""Plot the cost-quality frontier on RouterBench and save it to assets/frontier.png.

Run:  pip install 'pareto-router[bench]' matplotlib  &&  python examples/plot_frontier.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pareto_router import load_llmrouterbench, run_benchmark  # noqa: E402


def main() -> None:
    data = load_llmrouterbench()
    report = run_benchmark(data)

    rc = [p.avg_cost for p in report.router_curve]
    rq = [p.avg_quality for p in report.router_curve]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(rc, rq, "-o", ms=3, color="#2563eb", zorder=3,
            label="pareto-router (TF-IDF + Ridge)")

    for name, info in report.baselines.items():
        if info["kind"] == "single_model":
            ax.scatter(info["cost"], info["quality"], color="#9ca3af", s=18, zorder=2)
            ax.annotate(name, (info["cost"], info["quality"]),
                        fontsize=6, color="#6b7280", xytext=(3, 2), textcoords="offset points")

    strongest = report.summary["strongest_model"]
    gi = report.baselines[strongest]
    ax.scatter(gi["cost"], gi["quality"], color="#dc2626", s=42, zorder=4,
               label=f"{strongest} (strongest single)")
    orc = report.baselines["oracle"]
    ax.scatter(orc["cost"], orc["quality"], marker="*", s=170, color="#16a34a", zorder=4,
               label="oracle (upper bound)")

    ax.set_xscale("log")
    ax.set_xlabel("cost  ($ / query, log scale)")
    ax.set_ylabel("quality (accuracy)")
    ax.set_title(f"pareto-router on LLMRouterBench (current models): {report.n_test:,} held-out queries, {len(report.models)} models")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    out = Path(__file__).resolve().parents[1] / "assets" / "frontier.png"
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
