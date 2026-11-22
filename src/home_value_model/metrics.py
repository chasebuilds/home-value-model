"""AVM industry accuracy metrics. All take sale prices and model estimates in dollars."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Arr = npt.NDArray[np.float64]


def avm_metrics(actual: Arr, estimate: Arr) -> dict[str, float]:
    """Return the usual AVM scorecard.

    - mdape: median absolute percentage error, |estimate - sale| / sale
    - mape: mean of the same
    - ppe5, ppe10, ppe20: share of estimates within 5%, 10%, 20% of the sale price
    - median_ratio: median of estimate / sale (1.0 means no bias)
    - fsd: forecast standard deviation, the std dev of sale / estimate
    - cod: coefficient of dispersion (IAAO), spread of estimate / sale around its median, in %
    - prd: price-related differential (IAAO): mean ratio / price-weighted mean ratio
    """
    actual = np.asarray(actual, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    ape = np.abs(estimate - actual) / actual
    ratio = estimate / actual
    med = float(np.median(ratio))
    return {
        "n": float(len(actual)),
        "mdape": float(np.median(ape)),
        "mape": float(ape.mean()),
        "ppe5": float((ape <= 0.05).mean()),
        "ppe10": float((ape <= 0.10).mean()),
        "ppe20": float((ape <= 0.20).mean()),
        "median_ratio": med,
        "fsd": float(np.std(actual / estimate, ddof=1)),
        "cod": float(100 * np.mean(np.abs(ratio - med)) / med),
        "prd": float(ratio.mean() / (estimate.sum() / actual.sum())),
    }
