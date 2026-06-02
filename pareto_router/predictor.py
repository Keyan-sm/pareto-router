"""Predict each model's quality on a query from its features.

A multi-output ridge regression maps query features -> a vector of predicted quality
scores, one per model in the pool. Ridge is convex, fast, and handles both the sparse
TF-IDF matrix and the multi-output target natively, so training on all of RouterBench
takes seconds on a CPU. Any estimator with scikit-learn ``fit``/``predict`` semantics
and multi-output support can be swapped in.

This is the learned component of the paradigm RouteLLM (Ong et al., 2024) and R2-Router
(Xue et al., 2026) formalize: predict per-model quality, then route. We keep the
predictor transparent and cheap rather than a fine-tuned transformer.
"""
from __future__ import annotations

import numpy as np


class QualityPredictor:
    """Multi-output regression from features to per-model quality in [0, 1]."""

    def __init__(self, alpha: float = 10.0, estimator=None):
        if estimator is None:
            from sklearn.linear_model import Ridge

            # alpha=10 (vs sklearn's default 1.0): the TF-IDF feature space is wide
            # and sparse, so stronger ridge regularization generalizes better — at
            # alpha=1 the held-out router under-shoots; by alpha~10 it plateaus.
            # solver="sparse_cg" is the stable path for sparse design matrices.
            estimator = Ridge(alpha=alpha, solver="sparse_cg")
        self.estimator = estimator
        self._fitted = False

    def fit(self, X, quality: np.ndarray) -> "QualityPredictor":
        """X: (n, d) features; quality: (n, m) target quality per model."""
        # macOS Accelerate BLAS raises spurious divide/overflow flags inside Ridge's
        # sparse intercept matmul even though coefficients stay O(1) and outputs are
        # finite. Silence those benign flags, then assert finiteness rather than ship
        # the noise or hide a real problem.
        with np.errstate(all="ignore"):
            self.estimator.fit(X, quality)
        if not np.all(np.isfinite(self.estimator.coef_)):
            raise FloatingPointError("Predictor produced non-finite coefficients.")
        self._fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Return predicted quality, clipped to the [0, 1] range of the target."""
        if not self._fitted:
            raise RuntimeError("QualityPredictor.predict called before fit().")
        return np.clip(self.estimator.predict(X), 0.0, 1.0)
