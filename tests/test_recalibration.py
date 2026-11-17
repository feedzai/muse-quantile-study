import numpy as np
import pytest

from muse.recalibration import Recalibrator, calibration_month


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_recalibrator_is_monotone_and_tracks_prevalence(method):
    rng = np.random.default_rng(0)
    scores = rng.uniform(0.01, 0.99, 20_000)
    labels = rng.uniform(size=scores.size) < scores**3  # true P(fraud) = score^3
    p = Recalibrator(method).fit(scores, labels).predict(np.sort(scores))
    assert np.all(np.diff(p) >= -1e-12)
    assert abs(p.mean() - labels.mean()) < 0.02


def test_calibration_month_applies_label_delay():
    assert calibration_month(5, 0) == 4
    assert calibration_month(5, 2) == 2


def test_recalibrator_needs_both_classes():
    with pytest.raises(ValueError):
        Recalibrator("platt").fit(np.array([0.1, 0.2]), np.array([0, 0]))
