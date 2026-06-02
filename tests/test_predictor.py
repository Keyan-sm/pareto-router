import numpy as np
from scipy import sparse

from pareto_router import QualityPredictor


def test_fit_predict_shape_and_clipped_range():
    rng = np.random.default_rng(0)
    X = sparse.csr_matrix(rng.random((40, 12)))
    Q = rng.random((40, 3))
    predictor = QualityPredictor().fit(X, Q)
    P = predictor.predict(X)
    assert P.shape == (40, 3)
    assert P.min() >= 0.0 and P.max() <= 1.0


def test_predict_before_fit_raises():
    try:
        QualityPredictor().predict(np.zeros((1, 3)))
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError when predicting before fit()")


def test_learns_a_simple_signal():
    # Feature 0 drives model A's quality; feature 1 drives model B's.
    rng = np.random.default_rng(1)
    X = rng.random((200, 2))
    Q = np.column_stack([X[:, 0], X[:, 1]])
    predictor = QualityPredictor(alpha=0.1).fit(sparse.csr_matrix(X), Q)
    P = predictor.predict(sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]])))
    assert P[0, 0] > P[0, 1]   # first row: model A predicted higher
    assert P[1, 1] > P[1, 0]   # second row: model B predicted higher
