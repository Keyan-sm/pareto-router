"""Load routing datasets into clean arrays for routing experiments.

A routing dataset gives, for each query, every candidate model's answer quality and its
dollar cost. That is all a router needs: features (the prompt) -> a vector of per-model
quality, with a cost matrix to trade off against. The rest of the library never sees a
model name, so any dataset that fills :class:`RoutingDataset` works.

Two loaders ship here:

* :func:`load_sprout` - SPROUT (CARROT, 2025): a current model pool (o3-mini, GPT-4o,
  Claude-3.5-Sonnet, Llama-3.3-70B, Llama-3.1-405B, ...). Quality is the dataset's judge
  score; cost is computed from the input/output token counts SPROUT records, times a
  documented per-model price table (:data:`SPROUT_PRICING`).
* :func:`load_routerbench` - RouterBench (Hu et al., 2024): an older 11-model pool, kept to
  show the library is model-agnostic.

Loaders need the optional ``[bench]`` dependencies (pandas, huggingface_hub, pyarrow); the
core router and predictor do not import them.

References:
- Somerstep et al., "CARROT: A Cost Aware Rate Optimal Router," arXiv:2502.03261, 2025 (SPROUT).
- Hu et al., "RouterBench: A Benchmark for Multi-LLM Routing System," arXiv:2403.12031, 2024.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RoutingDataset:
    """A routing dataset as plain arrays.

    Attributes:
        prompts:   list of query strings, length n.
        models:    list of m model names (the routing pool).
        quality:   (n, m) float array in [0, 1]; quality[i, j] = model j's score on query i.
        cost:      (n, m) float array; cost[i, j] = dollar cost of model j on query i.
        eval_name: (n,) array of the source dataset for each query (for stratified splits).
    """

    prompts: List[str]
    models: List[str]
    quality: np.ndarray
    cost: np.ndarray
    eval_name: np.ndarray

    def __len__(self) -> int:
        return len(self.prompts)

    @property
    def n_models(self) -> int:
        return len(self.models)


# Backward-compatible alias; this used to be the only dataset and was named RouterBench.
RouterBench = RoutingDataset


def _bench_imports():
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download

        return pd, hf_hub_download
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Loading a routing dataset needs the [bench] extra: pip install 'pareto-router[bench]'"
        ) from exc


# =============================================================================
# SPROUT (CARROT) - current model pool
# =============================================================================

SPROUT_REPOS = {"o3mini": "CARROT-LLM-Routing/SPROUT-o3mini", "base": "CARROT-LLM-Routing/SPROUT"}
_SPROUT_META = {"key", "dataset", "dataset_level", "dataset_idx", "prompt", "golden_answer"}

# Approximate list prices, USD per 1M tokens (input, output), ~2026-06. SPROUT records token
# counts rather than dollars, so cost = (in*price_in + out*price_out) / 1e6. Edit to taste;
# the cost-quality frontier is robust to reasonable price changes. Pass `pricing=` to override.
SPROUT_PRICING: Dict[str, Tuple[float, float]] = {
    "openai-o3-mini": (1.10, 4.40),
    "openai-gpt-4o": (2.50, 10.00),
    "openai-gpt-4o-mini": (0.15, 0.60),
    "aws-claude-3-5-sonnet-v1": (3.00, 15.00),
    "aws-titan-text-premier-v1": (0.50, 1.50),
    "wxai-llama-3-405b-instruct": (5.00, 16.00),
    "wxai-llama-3-3-70b-instruct": (0.72, 0.72),
    "wxai-llama-3-1-70b-instruct": (0.72, 0.72),
    "wxai-llama-3-1-8b-instruct": (0.20, 0.20),
    "wxai-llama-3-2-3b-instruct": (0.06, 0.06),
    "wxai-llama-3-2-1b-instruct": (0.04, 0.04),
    "wxai-granite-3-8b-instruct-8k-max-tokens": (0.20, 0.20),
    "wxai-granite-3-2b-instruct-8k-max-tokens": (0.10, 0.10),
    "wxai-mixtral-8x7b-instruct-v01": (0.60, 0.60),
}


def load_sprout(variant: str = "o3mini", split: str = "test",
                cache_dir: Optional[str] = None,
                pricing: Optional[Dict[str, Tuple[float, float]]] = None) -> RoutingDataset:
    """Load SPROUT (CARROT) into a :class:`RoutingDataset`.

    ``variant="o3mini"`` includes OpenAI o3-mini; ``variant="base"`` is the original pool.
    Quality is the per-model judge score; cost is derived from token counts and ``pricing``.
    """
    pd, hf_hub_download = _bench_imports()
    if variant not in SPROUT_REPOS:
        raise ValueError(f"variant must be one of {sorted(SPROUT_REPOS)}; got {variant!r}")
    pricing = pricing or SPROUT_PRICING

    path = hf_hub_download(SPROUT_REPOS[variant], f"data/{split}-00000-of-00001.parquet",
                           repo_type="dataset", cache_dir=cache_dir)
    df = pd.read_parquet(path)
    models = [c for c in df.columns if c not in _SPROUT_META]
    unpriced = [m for m in models if m not in pricing]
    if unpriced:
        raise ValueError(f"No price for {unpriced}; pass pricing= with these models.")

    n, m = len(df), len(models)
    quality = np.full((n, m), np.nan)
    cost = np.full((n, m), np.nan)
    for j, model in enumerate(models):
        price_in, price_out = pricing[model]
        for i, cell in enumerate(df[model].tolist()):
            if not cell:
                continue
            score = cell.get("score")
            if score is None:
                continue
            t_in = cell.get("num_input_tokens") or 0
            t_out = cell.get("num_output_tokens") or 0
            quality[i, j] = float(score)
            cost[i, j] = (t_in * price_in + t_out * price_out) / 1_000_000

    ok = ~(np.isnan(quality).any(axis=1) | np.isnan(cost).any(axis=1))
    prompts = df["prompt"].tolist()
    eval_name = df["dataset"].to_numpy()
    return RoutingDataset(
        prompts=[p for p, keep in zip(prompts, ok) if keep],
        models=models,
        quality=quality[ok],
        cost=cost[ok],
        eval_name=eval_name[ok],
    )


# =============================================================================
# LLMRouterBench - current flagship pool (GPT-5, Claude-4, Gemini-2.5-Pro, ...)
# =============================================================================

LLMROUTERBENCH_REPO = "NPULH/LLMRouterBench"
# The current frontier + strong open-weight subset of the 33-model release. Each record
# already carries a real per-call cost (dollars), so no price table is needed.
LLMROUTERBENCH_FLAGSHIP = [
    "gpt-5", "gpt-5-chat", "claude-sonnet-4", "gemini-2.5-pro", "gemini-2.5-flash",
    "qwen3-235b-a22b-2507", "qwen3-235b-a22b-thinking-2507", "deepseek-v3.1-terminus",
    "deepseek-r1-0528", "deepseek-v3-0324", "kimi-k2-0905", "glm-4.6",
]


def load_llmrouterbench(models: Optional[List[str]] = None,
                        cache_dir: Optional[str] = None) -> RoutingDataset:
    """Load LLMRouterBench (Jan 2026) into a :class:`RoutingDataset`.

    The release is a 1.28 GB archive of ``bench-release/<task>/<model>/*.json`` files;
    each record has a per-query ``score`` and real ``cost``. Queries are aligned across
    models by (task, index). Parsing is cached next to the download, so only the first
    call is slow. ``models`` defaults to the current flagship subset.
    """
    import json
    import os
    import pickle
    import tarfile

    _pd, hf_hub_download = _bench_imports()
    models = list(models or LLMROUTERBENCH_FLAGSHIP)
    path = hf_hub_download(LLMROUTERBENCH_REPO, "bench-release.tar.gz",
                           repo_type="dataset", cache_dir=cache_dir)
    cache = f"{path}.routing-{len(models)}.pkl"
    if os.path.exists(cache):
        with open(cache, "rb") as fh:
            return pickle.load(fh)

    pool = set(models)
    cells: Dict[tuple, Dict[str, tuple]] = {}
    prompts: Dict[tuple, str] = {}
    with tarfile.open(path, "r:gz") as tf:
        for member in tf:
            if not member.name.endswith(".json"):
                continue
            parts = member.name.split("/")  # bench-release/<task>/<model>/<file>.json
            if len(parts) < 4 or parts[2] not in pool:
                continue
            task, model = parts[1], parts[2]
            try:
                payload = json.load(tf.extractfile(member))
            except Exception:  # pragma: no cover - skip a corrupt shard
                continue
            for rec in payload.get("records", []):
                score, cost = rec.get("score"), rec.get("cost")
                if score is None or cost is None:
                    continue
                key = (task, rec.get("index"))
                cells.setdefault(key, {})[model] = (float(score), float(cost))
                prompts.setdefault(key, rec.get("origin_query") or rec.get("prompt") or "")

    keys = sorted(k for k, v in cells.items() if len(v) == len(models))
    n, m = len(keys), len(models)
    quality = np.zeros((n, m))
    cost = np.zeros((n, m))
    for i, key in enumerate(keys):
        for j, model in enumerate(models):
            score, c = cells[key][model]
            quality[i, j] = score
            cost[i, j] = c
    quality = np.clip(quality, 0.0, 1.0)
    dataset = RoutingDataset(
        prompts=[prompts[k] for k in keys],
        models=models,
        quality=quality,
        cost=cost,
        eval_name=np.array([k[0] for k in keys]),
    )
    with open(cache, "wb") as fh:
        pickle.dump(dataset, fh)
    return dataset


# =============================================================================
# RouterBench - older pool, kept to show the library is model-agnostic
# =============================================================================

HF_REPO = "withmartian/routerbench"
SPLITS = {"0shot": "routerbench_0shot.pkl", "5shot": "routerbench_5shot.pkl"}


def load_routerbench(split: str = "0shot", cache_dir: Optional[str] = None) -> RoutingDataset:
    """Download (if needed) and parse a RouterBench split into a :class:`RoutingDataset`."""
    pd, hf_hub_download = _bench_imports()
    if split not in SPLITS:
        raise ValueError(f"split must be one of {sorted(SPLITS)}; got {split!r}")

    path = hf_hub_download(HF_REPO, SPLITS[split], repo_type="dataset", cache_dir=cache_dir)
    df = pd.read_pickle(path)

    cost_cols = [c for c in df.columns if c.endswith("|total_cost")]
    models = [c[: -len("|total_cost")] for c in cost_cols]
    if not models:
        raise ValueError("No '<model>|total_cost' columns found; unexpected RouterBench schema.")

    quality = np.vstack([pd.to_numeric(df[m], errors="coerce").to_numpy() for m in models]).T
    cost = np.vstack([pd.to_numeric(df[f"{m}|total_cost"], errors="coerce").to_numpy() for m in models]).T
    quality = quality.astype(float)
    cost = cost.astype(float)
    prompts = [_join_prompt(p) for p in df["prompt"].tolist()]
    eval_name = df["eval_name"].to_numpy()

    ok = ~(np.isnan(quality).any(axis=1) | np.isnan(cost).any(axis=1))
    return RoutingDataset(
        prompts=[p for p, keep in zip(prompts, ok) if keep],
        models=models,
        quality=quality[ok],
        cost=cost[ok],
        eval_name=eval_name[ok],
    )


def load_frontier(path: str) -> RoutingDataset:
    """Load a dataset built by ``bench_gen/generate_routing_dataset.py`` (JSONL of
    per-query, per-model score + cost). Reads a local file; no Hub download."""
    import json

    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if not rows:
        raise ValueError(f"{path} has no rows")
    models = list(rows[0]["models"].keys())
    n, m = len(rows), len(models)
    quality = np.full((n, m), np.nan)
    cost = np.full((n, m), np.nan)
    for i, row in enumerate(rows):
        for j, model in enumerate(models):
            cell = row["models"].get(model)
            if not cell:
                continue
            quality[i, j] = float(cell["score"])
            cost[i, j] = float(cell["cost"])
    ok = ~(np.isnan(quality).any(axis=1) | np.isnan(cost).any(axis=1))
    prompts = [r["prompt"] for r in rows]
    eval_name = np.array([r.get("dataset", "na") for r in rows])
    return RoutingDataset(
        prompts=[p for p, keep in zip(prompts, ok) if keep],
        models=models,
        quality=quality[ok],
        cost=cost[ok],
        eval_name=eval_name[ok],
    )


def _join_prompt(prompt) -> str:
    """RouterBench prompts are stored as a list (possibly multi-turn); flatten to text."""
    if isinstance(prompt, (list, tuple)):
        return "\n".join(str(x) for x in prompt)
    return str(prompt)
