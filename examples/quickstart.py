"""Train a router on RouterBench, then route a few prompts.

Run:  pip install 'pareto-router[bench]'  &&  python examples/quickstart.py
"""
from __future__ import annotations

from pareto_router import RouterModel, load_routerbench

PROMPTS = [
    "What is the capital of France?",
    "Prove that the halting problem is undecidable, step by step.",
    "Extract every date from this contract and return them as JSON.",
]


def main() -> None:
    data = load_routerbench("0shot")
    model = RouterModel.fit(data)  # TF-IDF + multi-output Ridge over the 11-model pool
    for prompt in PROMPTS:
        # lam=0.3 leans toward quality; lam=0.7 leans toward cost.
        decision = model.route(prompt, lam=0.3)
        print(f"\n{prompt!r}")
        print(f"  -> {decision.model}  (predicted quality {decision.predicted_quality:.2f}, "
              f"est. cost ${decision.est_cost:.5f})")


if __name__ == "__main__":
    main()
