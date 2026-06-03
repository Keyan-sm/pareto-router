# pareto-router

**An LLM router: predict how well each model handles a query, then pick on the
cost-quality frontier.** Small and dependency-light, it implements RouteLLM's
predict-quality-then-route paradigm (Ong et al., 2024) and R2-Router's cost-quality
selection objective (Xue et al., 2026), benchmarked on SPROUT (Somerstep et al., 2025), a
current-model routing dataset.

On 1,949 held-out SPROUT queries across 14 current models (o3-mini, GPT-4o,
Claude-3.5-Sonnet, Llama-3.3-70B, Llama-3.1-405B, ...), a `pareto-router` trained with
TF-IDF and ridge regression:

- matches 95% of o3-mini's quality (the strongest model) at 84% lower cost;
- beats the best single model you can buy at any budget (+12.6 quality points at the cheapest budget tier);
- stays above all 14 individual models on the cost-quality frontier.

![cost-quality frontier on SPROUT](assets/frontier.png)

The blue line traces the router as you turn the cost/quality knob (`λ`). Each grey dot is
one model; the router runs above and to the left of all of them. The green star marks the
per-query oracle, the headroom a stronger featurizer can still reach.

## Why this exists

LLM prices span two orders of magnitude, and no single model wins every query. On SPROUT
the strongest single model (o3-mini) averages 0.91 quality; sending each query to its own
best model averages 0.99. Route well and you capture much of that gap at a fraction of the cost.

The open-source options leave a hole:

- **LiteLLM** is a proxy. It hands you one API to 100+ models, but you still pick which to call.
- **RouteLLM** is research code for binary strong-vs-weak routing, and no one maintains it as a library.
- **vLLM-router** ties routing to serving infrastructure.

`pareto-router` fills the hole: a pip-installable routing-decision library that predicts
per-model quality and picks on the cost-quality frontier, with a benchmark you run yourself.
The pool is swappable. It ships loaders for SPROUT (current models) and RouterBench
(Hu et al., 2024), and any dataset of (prompts, per-model quality, per-model cost) works.

## Install

```bash
pip install pareto-router            # core: numpy + scikit-learn
pip install "pareto-router[bench]"   # + pandas, huggingface_hub, pyarrow  (to load the datasets)
```

## Use

```python
from pareto_router import RouterModel, load_sprout

data  = load_sprout("o3mini")            # current models: o3-mini, GPT-4o, Claude-3.5-Sonnet, ...
model = RouterModel.fit(data)            # TF-IDF + multi-output ridge over the pool

decision = model.route("Prove the halting problem is undecidable.", lam=0.3)
print(decision.model, decision.predicted_quality)   # e.g. openai-gpt-4o 0.74
```

`lam` is the knob. Set it to 0 to always pick the highest predicted quality, 1 to always
pick the cheapest, or anything between to slide along the frontier.

```bash
pareto-router benchmark                          # SPROUT, current models (default)
pareto-router benchmark --benchmark routerbench  # the older 11-model pool
pareto-router train --out r.pkl                  # train and save a router
pareto-router route "..." --model r.pkl --lam 0.3
```

## How it works

1. **Featurize** the prompt. The default uses TF-IDF; you can drop in transformer
   embeddings through the `Featurizer` protocol.
2. **Predict** each model's quality with one multi-output ridge regression (`predictor.py`).
3. **Route** by maximizing `S = (1 − λ)·quality − λ·cost` over the pool. Cost is normalized
   to a fixed pool scale so `λ` lines up with the [0, 1] quality scale (`router.py`).
   `select_under_budget` does the same under a hard cost cap.

## The benchmark

`pareto-router benchmark` runs on SPROUT (CARROT, Somerstep et al., 2025): for each query it
has every model's judge score (quality) and the input/output token counts, from which cost is
computed with a documented per-model price table. The split is stratified by source dataset,
the predictor trains on the train half, and `λ` is swept over the held-out half. The router
decides from predicted quality on queries it never saw.

```
sprout evaluation  |  train=4545  test=1949  models=14
strongest single model : openai-o3-mini  quality=0.908  cost=$0.00556
match strongest quality: cost=$0.00530  (savings 4.7%)
match 95% of its quality: cost=$0.00091  (savings 83.6%)
frontier dominates     : 14/14 single models
frontier AUC (<= strongest cost): router=0.8846  single-model envelope=0.8201
oracle (upper bound)   : quality=0.989  cost=$0.00060

router vs best affordable single model, by budget:
  budget (xstrongest cost)  best single   router   advantage
  0.05                           0.695    0.821   +12.6 pts
  0.15                           0.809    0.860    +5.1 pts
  0.3                            0.809    0.870    +6.1 pts
  0.6                            0.809    0.901    +9.2 pts
  1.0                            0.908    0.911    +0.3 pts
```

Every number above comes from this library on SPROUT-o3mini, not from a paper. Reproduce them
with `pareto-router benchmark` (about 10 s on a laptop). Run `--benchmark routerbench` for the
older 11-model pool; the router code does not change, which is the point of a model-agnostic design.

### Benchmark your own (current) models

Public routing datasets lag model releases: a new frontier (GPT-5.5, Opus/Sonnet 4.8, Gemini
3.1 Pro, ...) ships before any public benchmark scores it. `bench_gen/generate_routing_dataset.py`
builds a fresh `RoutingDataset` for any pool through OpenRouter (one key reaches every provider):
it runs public exact-match tasks through each model, records OpenRouter's real per-call cost,
grades deterministically (no LLM judge), and writes a file `load_frontier` reads.

```bash
export OPENROUTER_API_KEY=sk-or-...
python bench_gen/generate_routing_dataset.py --dry-run --n 60         # validate the pipeline, free
python bench_gen/generate_routing_dataset.py --estimate --n 500       # project the spend first
python bench_gen/generate_routing_dataset.py --n 500 --out frontier.jsonl
```

Then `load_frontier("frontier.jsonl")` feeds the same router, benchmark, and plot.

## Precedent and iteration

| Component | Source | Fidelity |
| --- | --- | --- |
| Predict per-model quality, then route | RouteLLM (Ong et al., 2024) | Faithful to the paradigm |
| Cost-quality trade-off objective `S=(1−λ)Q−λC` | R2-Router (Xue et al., 2026) | Faithful to the general selection form |
| Current-model evaluation data | SPROUT / CARROT (Somerstep et al., 2025) | Used as the benchmark; cost computed from its token counts |
| Older 11-model evaluation | RouterBench (Hu et al., 2024) | Second loader, shows the library is model-agnostic |
| TF-IDF + ridge quality predictor | mine | RouteLLM and R2-Router train text embeddings; this is the dependency-light baseline |
| Multi-model frontier + λ knob + budget mode | mine | RouteLLM routes binary strong/weak; this routes over the full pool |
| Length-budget-aware routing | R2-Router | Not implemented, see Scope |

## Scope

- **Cost from token counts.** SPROUT records input/output token counts, not dollars, so cost
  is computed with a per-model price table (`SPROUT_PRICING`, approximate list prices, editable).
  The frontier is robust to reasonable price changes; pass `pricing=` to use your own rates.
- **The predictor is a baseline.** TF-IDF and ridge stay light on purpose. The oracle reaches
  0.989 against the router's top of ~0.91; that gap is the headroom a stronger featurizer
  (a sentence-transformer through the `Featurizer` protocol) would capture.
- **No length-budget routing yet.** R2-Router also picks an output-length budget per query,
  which needs per-length data (their R2-Bench) that these datasets lack. On the roadmap.
- **Older models available too.** `--benchmark routerbench` runs the 2023 pool; SPROUT's pool
  (o3-mini, GPT-4o, Claude-3.5-Sonnet, Llama-3.3) is current.

## Module map

| Module | Responsibility |
| --- | --- |
| `data.py` | Load SPROUT / RouterBench into quality + cost arrays |
| `features.py` | Featurizers (default TF-IDF; pluggable) |
| `predictor.py` | Multi-output quality regression |
| `router.py` | Cost-quality selection (`λ` trade-off + budget) |
| `metrics.py` | Frontier, cost-at-quality, AUC |
| `benchmark.py` | Train/test evaluation |
| `model.py` | `RouterModel`: fit / route / save / load |

## Papers

- **CARROT (SPROUT)**, Somerstep et al., *A Cost Aware Rate Optimal Router*, [arXiv:2502.03261](https://arxiv.org/abs/2502.03261), 2025.
- **RouteLLM**, Ong et al., *Learning to Route LLMs with Preference Data*, [arXiv:2406.18665](https://arxiv.org/abs/2406.18665), 2024.
- **R2-Router**, Xue et al., *A New Paradigm for LLM Routing with Reasoning*, [arXiv:2602.02823](https://arxiv.org/abs/2602.02823), 2026.
- **RouterBench**, Hu et al., *A Benchmark for Multi-LLM Routing System*, [arXiv:2403.12031](https://arxiv.org/abs/2403.12031), 2024.

## Status

v0.2.0. Tested, CI on Python 3.9 to 3.12, runs offline except for the dataset download.
Default benchmark is SPROUT (current models). Roadmap: a transformer-embedding featurizer,
length-budget routing on per-length data, and live provider adapters with token-based cost
estimation.

## License

MIT
