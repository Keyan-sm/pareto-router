# pareto-router

**Predict how well each model in your pool will answer a query, then route on the
cost–quality frontier.** A small, dependency-light LLM router you train and benchmark
on real data — not a proxy, and not a research dump.

On **10,950 held-out RouterBench queries** across **11 models**, a `pareto-router`
trained with nothing heavier than TF-IDF + ridge regression:

- **matches GPT-4's quality at 17% lower cost**, and reaches **95% of GPT-4's quality at 80% lower cost**;
- **beats the best single model you could buy at any budget** — e.g. **+12.6 quality points** at a $0.001/query budget;
- **dominates all 11 individual models** on the cost–quality frontier.

![cost–quality frontier on RouterBench](assets/frontier.png)

The blue line is the router as you turn the cost/quality knob (`λ`); every grey dot is
a single model. The router sits above and to the left of all of them. The green star is
the per-query oracle — the headroom a stronger featurizer can still capture (see
[Scope](#scope-and-honest-limitations)).

## Why this exists

The cost gap between LLMs is two orders of magnitude, and no single model is best on
every query. On RouterBench the strongest single model (GPT-4) averages **0.78**
quality, but routing each query to its own best model averages **0.91** (the oracle) —
that gap, at a fraction of the cost, is what a router goes after.

But the open-source options are awkward:

- **LiteLLM** is a *proxy* — a unified API to 100+ models. It doesn't decide *which* model to call; you still do.
- **RouteLLM** is research-grade, focused on binary strong-vs-weak routing, and not a maintained library.
- **vLLM-router** and friends are tied to serving infrastructure.

`pareto-router` is the missing piece: a clean, pip-installable **routing decision**
library — predict per-model quality, pick on the cost–quality frontier — with a
reproducible RouterBench benchmark so you can measure it yourself.

## Install

```bash
pip install pareto-router            # core: numpy + scikit-learn
pip install "pareto-router[bench]"   # + pandas, huggingface_hub  (to load RouterBench)
```

## Use

```python
from pareto_router import RouterModel, load_routerbench

data  = load_routerbench("0shot")        # downloads RouterBench (Hu et al., 2024)
model = RouterModel.fit(data)            # TF-IDF + multi-output ridge over the 11-model pool

decision = model.route("Prove the halting problem is undecidable.", lam=0.3)
print(decision.model, decision.predicted_quality)   # e.g. gpt-4-1106-preview 0.71
```

`lam` is the knob: `lam=0` always picks the highest predicted quality, `lam=1` always
the cheapest, and values in between trace the frontier.

```bash
pareto-router benchmark              # reproduce the RouterBench numbers below
pareto-router train  --out r.pkl     # train and save a router
pareto-router route  "..." --model r.pkl --lam 0.3
```

## How it works

1. **Featurize** the prompt (default: TF-IDF; swap in transformer embeddings via the `Featurizer` protocol).
2. **Predict** each model's quality with one multi-output ridge regression (`predictor.py`).
3. **Route** by maximizing `S = (1 − λ)·quality − λ·cost` over the pool, with cost
   normalized to a fixed pool scale so `λ` is comparable to the [0, 1] quality scale
   (`router.py`). A budget mode (`select_under_budget`) is also provided.

## The benchmark

`pareto-router benchmark` runs the whole thing on RouterBench: a stratified
train/test split (no *quality* leakage — the router decides from **predicted** quality
on held-out queries; per-query cost is taken as known, the standard RouterBench scoring
protocol, see [Scope](#scope-and-honest-limitations)), the router's frontier swept over
`λ`, and fixed reference points.

```
RouterBench reproduction  |  train=25547  test=10950  models=11
strongest single model : gpt-4-1106-preview  quality=0.782  cost=$0.00330
match strongest quality: cost=$0.00274  (savings 16.8%)
match 95% of its quality: cost=$0.00064  (savings 80.5%)
frontier dominates     : 11/11 single models
frontier AUC (<= strongest cost): router=0.7569  single-model envelope=0.6526
oracle (upper bound)   : quality=0.910  cost=$0.00023

router vs best affordable single model, by budget:
  budget (xstrongest cost)  best single   router   advantage
  0.05                           0.549    0.668   +11.9 pts
  0.15                           0.646    0.724    +7.8 pts
  0.3                            0.646    0.772   +12.6 pts
  0.6                            0.646    0.778   +13.2 pts
  1.0                            0.782    0.785    +0.3 pts
```

These are this library's measured numbers on the RouterBench 0-shot split, not figures
quoted from a paper. Reproduce them with `pareto-router benchmark` (≈10 s on a laptop).

## What's faithful to the papers, and what's mine

| Component | Source | Fidelity |
| --- | --- | --- |
| Predict per-model quality, then route | RouteLLM (Ong et al., 2024) | **Faithful** to the paradigm |
| Cost–quality trade-off objective `S=(1−λ)Q−λC` | R2-Router (Xue et al., 2026) | **Faithful** to the general selection form |
| Evaluation on RouterBench | Hu et al., 2024 | **Faithful** — the standard routing benchmark |
| TF-IDF + ridge quality predictor | — | **Mine.** RouteLLM/R2-Router use learned text embeddings; this is the dependency-light baseline. |
| Multi-model frontier + λ knob + budget mode | — | **Mine.** RouteLLM routes binary strong/weak; this routes over the full pool. |
| Length-budget-aware routing | R2-Router | **Not implemented** — see Scope. |

## Scope and honest limitations

- **No length-budget routing (yet).** R2-Router's headline idea is to also choose an
  *output-length budget* per query. That needs per-length quality data (their R2-Bench,
  built on SPROUT with learned embeddings); RouterBench has one quality/cost per
  query–model, so this library can't honestly evaluate it. It's on the roadmap, not in
  the numbers above.
- **The predictor is a baseline.** TF-IDF + ridge is deliberately light. The oracle
  reaches 0.910 vs the router's top of ~0.785 — that gap is the headroom a stronger
  featurizer (e.g. sentence-transformer embeddings, droppable in via the `Featurizer`
  protocol) would capture. The library is built to make that swap a one-liner.
- **Cost at decision time.** The benchmark uses RouterBench's recorded per-query cost;
  for live routing, `RouterModel.route` falls back to per-model mean cost unless you
  pass a real estimate.

## Module map

| Module | Responsibility |
| --- | --- |
| `data.py` | Load/parse RouterBench into quality + cost arrays |
| `features.py` | Featurizers (default TF-IDF; pluggable) |
| `predictor.py` | Multi-output quality regression |
| `router.py` | Cost–quality selection (`λ` trade-off + budget) |
| `metrics.py` | Frontier, cost-at-quality, AUC |
| `benchmark.py` | Train/test RouterBench evaluation |
| `model.py` | `RouterModel`: fit / route / save / load |

## Papers

- **RouterBench** — Hu et al., *A Benchmark for Multi-LLM Routing System*, [arXiv:2403.12031](https://arxiv.org/abs/2403.12031), 2024.
- **RouteLLM** — Ong et al., *Learning to Route LLMs with Preference Data*, [arXiv:2406.18665](https://arxiv.org/abs/2406.18665), 2024.
- **R2-Router** — Xue et al., *A New Paradigm for LLM Routing with Reasoning*, [arXiv:2602.02823](https://arxiv.org/abs/2602.02823), 2026.

## Status

v0.1.0 — tested, CI configured for Python 3.9–3.12, runs offline for everything except
the RouterBench download. Roadmap: transformer-embedding featurizer, length-budget
routing on per-length data, live provider adapters (OpenAI/Anthropic) with token-based
cost estimation.

## License

MIT
