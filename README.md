# pareto-router

**An LLM router: predict how well each model handles a query, then pick on the
cost-quality frontier.** Small and dependency-light, it implements RouteLLM's
predict-quality-then-route paradigm (Ong et al., 2024) and R2-Router's cost-quality
selection objective (Xue et al., 2026), benchmarked on LLMRouterBench (Li et al., 2026), a
current frontier-model routing dataset.

On 450 held-out LLMRouterBench queries across 12 current models (gpt-5, claude-sonnet-4,
gemini-2.5-pro, qwen3-235b, deepseek-v3.1, kimi-k2, glm-4.6, ...), a `pareto-router` trained
with TF-IDF and ridge regression:

- **matches the strongest model's quality at 98% lower cost** (gemini-2.5-pro: 0.77 at $0.047/query; the router reaches 0.77 at $0.0009);
- at that model's own price it reaches **0.86 vs 0.77**, because no single model wins every query;
- **dominates all 12 models** on the cost-quality frontier.

![cost-quality frontier on LLMRouterBench](assets/frontier.png)

The blue line traces the router as you turn the cost/quality knob (`λ`). Each grey dot is
one model; the router runs above and to the left of all of them. The green star marks the
per-query oracle, the headroom a stronger featurizer can still reach.

## Model-agnostic by design

The router, predictor, and metrics never see a model name. They operate on arrays of
per-query quality and cost, so a model pool is just a dataset. This repo ships four:
**LLMRouterBench** (current frontier, the default), **SPROUT** (Somerstep et al., 2025),
**RouterBench** (Hu et al., 2024), and **`bench_gen`** to generate one for any pool through
OpenRouter. The same router code runs on all of them.

That matters because a public benchmark always trails the newest releases by weeks: by the
time GPT-5.5 or Opus 4.8 has a scored, per-query dataset, there is a newer model. The design
absorbs that. When new models ship you regenerate the dataset (`bench_gen` with one
OpenRouter key reaches every provider), not the router.

## Why this exists

LLM prices span two orders of magnitude, and no single model wins every query. On
LLMRouterBench the strongest single model (gemini-2.5-pro) averages 0.77 quality; sending
each query to its own best model averages 0.98. Route well and you capture much of that gap
at a fraction of the cost.

The open-source options leave a hole:

- **LiteLLM** is a proxy. It hands you one API to 100+ models, but you still pick which to call.
- **RouteLLM** is research code for binary strong-vs-weak routing, and no one maintains it as a library.
- **vLLM-router** ties routing to serving infrastructure.

`pareto-router` fills the hole: a pip-installable routing-decision library that predicts
per-model quality and picks on the cost-quality frontier, with a benchmark you run yourself.

## Install

```bash
pip install pareto-router            # core: numpy + scikit-learn
pip install "pareto-router[bench]"   # + pandas, huggingface_hub, pyarrow  (to load the datasets)
```

## Use

```python
from pareto_router import RouterModel, load_llmrouterbench

data  = load_llmrouterbench()            # current frontier pool (1.28 GB, downloaded once and cached)
model = RouterModel.fit(data)            # TF-IDF + multi-output ridge over the pool

decision = model.route("Prove the halting problem is undecidable.", lam=0.3)
print(decision.model, decision.predicted_quality)   # e.g. qwen3-235b-a22b-2507 0.71
```

`load_sprout("o3mini")` is a lighter alternative pool if you don't want the LLMRouterBench
download. `lam` is the knob: 0 always picks the highest predicted quality, 1 the cheapest.

```bash
pareto-router benchmark                          # LLMRouterBench, current models (default)
pareto-router benchmark --benchmark sprout       # a lighter 2024-2025 pool
pareto-router benchmark --benchmark routerbench  # the older 2023 pool
pareto-router train --out r.pkl
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

`pareto-router benchmark` runs on LLMRouterBench (Li et al., 2026): for each query every
model carries a graded score (quality) and the real per-call cost from the original run. The
default pool is the current frontier subset with full coverage, evaluated on the ArenaHard
family (general, coding, creative, math). The split is stratified by task; the predictor
trains on the train half; `λ` is swept over the held-out half from predicted quality.

```
llmrouterbench evaluation  |  train=1050  test=450  models=12
strongest single model : gemini-2.5-pro  quality=0.770  cost=$0.04749
match strongest quality: cost=$0.00086  (savings 98.2%)
match 95% of its quality: cost=$0.00079  (savings 98.3%)
frontier dominates     : 12/12 single models
frontier AUC (<= strongest cost): router=0.8473  single-model envelope=0.7439
oracle (upper bound)   : quality=0.981  cost=$0.00295

router vs best affordable single model, by budget:
  budget (xstrongest cost)  best single   router   advantage
  0.05                           0.709    0.828   +11.9 pts
  0.15                           0.709    0.837   +12.8 pts
  0.3                            0.751    0.848    +9.7 pts
  0.6                            0.751    0.848    +9.7 pts
  1.0                            0.770    0.858    +8.8 pts
```

Every number above comes from this library on LLMRouterBench, not from a paper. Reproduce
them with `pareto-router benchmark`. The same command with `--benchmark sprout` or
`--benchmark routerbench` runs a different pool, and the router code does not change.

### Benchmark your own (current) models

Public routing datasets lag model releases: a new frontier (GPT-5.5, Opus/Sonnet 4.8, Gemini
3.1 Pro) ships before any public benchmark scores it. `bench_gen/generate_routing_dataset.py`
builds a fresh `RoutingDataset` for any pool through OpenRouter (one key reaches every
provider): it runs public exact-match tasks through each model, records OpenRouter's real
per-call cost, grades deterministically (no LLM judge), and writes a file `load_frontier` reads.

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
| Current frontier evaluation (default) | LLMRouterBench (Li et al., 2026) | Used as the benchmark; real per-call cost |
| Alternative pools | SPROUT (Somerstep et al., 2025), RouterBench (Hu et al., 2024) | Show the design is model-agnostic |
| TF-IDF + ridge quality predictor | mine | RouteLLM and R2-Router train text embeddings; this is the dependency-light baseline |
| Multi-model frontier + λ knob + budget mode | mine | RouteLLM routes binary strong/weak; this routes over the full pool |
| Length-budget-aware routing | R2-Router | Not implemented, see Scope |

## Scope

- **The default pool is a coverage subset.** LLMRouterBench ran 33 models across 21 tasks;
  the default uses the 12-model current-frontier subset with full coverage, on the ArenaHard
  family (1,500 queries). Other pools and tasks are a config away.
- **The predictor is a baseline.** TF-IDF and ridge stay light on purpose. The oracle reaches
  0.98 against the router's top of ~0.86; that gap is the headroom a stronger featurizer
  (a sentence-transformer through the `Featurizer` protocol) would capture.
- **No length-budget routing yet.** R2-Router also picks an output-length budget per query,
  which needs per-length data (their R2-Bench) that these datasets lack. On the roadmap.

## Module map

| Module | Responsibility |
| --- | --- |
| `data.py` | Load LLMRouterBench / SPROUT / RouterBench / a generated file into quality + cost arrays |
| `features.py` | Featurizers (default TF-IDF; pluggable) |
| `predictor.py` | Multi-output quality regression |
| `router.py` | Cost-quality selection (`λ` trade-off + budget) |
| `metrics.py` | Frontier, cost-at-quality, AUC |
| `benchmark.py` | Train/test evaluation |
| `model.py` | `RouterModel`: fit / route / save / load |
| `bench_gen/` | Generate a dataset for any pool via OpenRouter |

## Papers

- **LLMRouterBench**, Li et al., *A Massive Benchmark and Unified Framework for LLM Routing*, [arXiv:2601.07206](https://arxiv.org/abs/2601.07206), 2026.
- **CARROT (SPROUT)**, Somerstep et al., *A Cost Aware Rate Optimal Router*, [arXiv:2502.03261](https://arxiv.org/abs/2502.03261), 2025.
- **RouteLLM**, Ong et al., *Learning to Route LLMs with Preference Data*, [arXiv:2406.18665](https://arxiv.org/abs/2406.18665), 2024.
- **R2-Router**, Xue et al., *A New Paradigm for LLM Routing with Reasoning*, [arXiv:2602.02823](https://arxiv.org/abs/2602.02823), 2026.
- **RouterBench**, Hu et al., *A Benchmark for Multi-LLM Routing System*, [arXiv:2403.12031](https://arxiv.org/abs/2403.12031), 2024.

## Status

v0.2.0. Tested, CI on Python 3.9 to 3.12, runs offline except for the dataset download.
Default benchmark is LLMRouterBench (current frontier models). Roadmap: a transformer-embedding
featurizer, length-budget routing, and live provider adapters with token-based cost estimation.

## License

MIT
