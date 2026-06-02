"""Featurizers turn a prompt into a numeric vector for the quality predictor.

The default is a dependency-light TF-IDF featurizer (scikit-learn): no transformer,
no GPU, no network. It is deliberately the *baseline* featurizer — the ``Featurizer``
protocol lets you drop in transformer/embedding features (e.g. sentence-transformers,
or the Qwen3 embeddings R2-Router uses) without touching the router or predictor.
TF-IDF is the honest, reproducible floor, and it already routes well on RouterBench
(see the README numbers).
"""
from __future__ import annotations

from typing import List, Protocol, runtime_checkable


@runtime_checkable
class Featurizer(Protocol):
    """Anything with fit / transform over a list of prompt strings."""

    def fit(self, prompts: List[str]) -> "Featurizer": ...
    def transform(self, prompts: List[str]): ...


class TfidfFeaturizer:
    """TF-IDF over character-aware word n-grams. Sparse, fast, fully offline."""

    def __init__(self, max_features: int = 30000, ngram_range=(1, 2), min_df: int = 5):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=True,
        )
        self._fitted = False

    def fit(self, prompts: List[str]) -> "TfidfFeaturizer":
        self.vectorizer.fit(prompts)
        self._fitted = True
        return self

    def transform(self, prompts: List[str]):
        if not self._fitted:
            raise RuntimeError("TfidfFeaturizer.transform called before fit().")
        return self.vectorizer.transform(prompts)

    def fit_transform(self, prompts: List[str]):
        matrix = self.vectorizer.fit_transform(prompts)
        self._fitted = True
        return matrix
