"""Label-based recalibration baselines (Platt scaling, isotonic regression).

Unlike quantile mapping, these need fraud labels: they fit a monotone
score -> P(fraud) function on a labeled window. A client then keeps fixed
thresholds in probability space across model updates, and the new model
is recalibrated on the most recent labeled window available.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


class Recalibrator:
    """Fit score -> P(fraud) on labeled data; `method` is "platt" or "isotonic".

    Platt is a logistic regression on the logit of the score (the usual form
    for scores already in (0, 1)); isotonic is a monotone step function.
    """

    def __init__(self, method: str):
        if method not in ("platt", "isotonic"):
            raise ValueError(f"unknown recalibration method '{method}'")
        self.method = method
        self._model = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "Recalibrator":
        labels = np.asarray(labels, dtype=int)
        if labels.min() == labels.max():
            raise ValueError("recalibration needs both classes in the labeled window")
        if self.method == "platt":
            self._model = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(scores).reshape(-1, 1), labels)
        else:
            self._model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(scores, labels)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Recalibrator.fit must be called first")
        if self.method == "platt":
            return self._model.predict_proba(_logit(scores).reshape(-1, 1))[:, 1]
        return self._model.predict(np.asarray(scores, dtype=float))


def calibration_month(eval_month: int, label_delay_months: int) -> int:
    """Labeled month used to recalibrate for `eval_month`: the most recent
    month whose labels have matured, given a delay of `label_delay_months`
    (0 = last month's labels are already complete, which production never has)."""
    return eval_month - 1 - label_delay_months
