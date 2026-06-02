"""Load RouterBench (Hu et al., 2024) into clean arrays for routing experiments.

RouterBench provides, for 36,497 queries across 86 eval sets, the per-model quality
score and dollar cost of 11 LLMs, plus the query prompt. That is exactly the
supervision a router needs: features (the prompt) -> a vector of per-model quality,
with a known cost matrix to trade off against.

The dataset is hosted at ``withmartian/routerbench`` on the Hugging Face Hub as
pandas pickles. Loading needs the optional ``[bench]`` dependencies (pandas,
huggingface_hub); the core router and predictor do not import them.

Reference: Hu et al., "RouterBench: A Benchmark for Multi-LLM Routing System,"
arXiv:2403.12031, 2024.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

HF_REPO = "withmartian/routerbench"
SPLITS = {"0shot": "routerbench_0shot.pkl", "5shot": "routerbench_5shot.pkl"}


@dataclass
class RouterBench:
    """A routing dataset as plain arrays.

    Attributes:
        prompts:   list of query strings, length n.
        models:    list of m model names (the routing pool).
        quality:   (n, m) float array in [0, 1]; quality[i, j] = model j's score on query i.
        cost:      (n, m) float array; cost[i, j] = dollar cost of model j on query i.
        eval_name: (n,) array of the source benchmark for each query (for stratified splits).
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


def load_routerbench(split: str = "0shot", cache_dir: Optional[str] = None) -> RouterBench:
    """Download (if needed) and parse a RouterBench split into a :class:`RouterBench`."""
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Loading RouterBench needs the [bench] extra: pip install 'pareto-router[bench]'"
        ) from exc
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

    # Drop any row with missing quality/cost so downstream math is clean.
    ok = ~(np.isnan(quality).any(axis=1) | np.isnan(cost).any(axis=1))
    return RouterBench(
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
