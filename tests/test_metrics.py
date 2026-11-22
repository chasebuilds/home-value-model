import numpy as np
import pytest

from home_value_model.metrics import avm_metrics


def test_perfect_estimates() -> None:
    a = np.array([100.0, 200.0, 300.0])
    m = avm_metrics(a, a.copy())
    assert m["mdape"] == 0 and m["ppe10"] == 1 and m["cod"] == 0
    assert m["median_ratio"] == 1 and m["fsd"] == 0 and m["prd"] == 1


def test_known_values() -> None:
    actual = np.array([100.0, 100.0, 100.0, 100.0])
    est = np.array([95.0, 110.0, 130.0, 100.0])
    m = avm_metrics(actual, est)
    assert m["mdape"] == pytest.approx(0.075)  # errors 0.00, 0.05, 0.10, 0.30: median is 0.075
    assert m["ppe5"] == 0.5
    assert m["ppe10"] == 0.75
    assert m["ppe20"] == 0.75
    assert m["n"] == 4


def test_bias_shows_in_median_ratio() -> None:
    actual = np.full(50, 200.0)
    m = avm_metrics(actual, actual * 1.1)
    assert abs(m["median_ratio"] - 1.1) < 1e-9
    assert m["cod"] < 1e-9
