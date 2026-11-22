"""Held-out evaluation: metrics per model, confidence-band check, importance, page export."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .data import TARGET
from .metrics import avm_metrics
from .model import AVM, BANDS, permutation_importance


def naive_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Rule of thumb to beat: neighborhood median price per sq ft times living area."""
    ppsf = (train[TARGET] / train["gr_liv_area"]).groupby(train["neighborhood"]).median()
    fallback = float((train[TARGET] / train["gr_liv_area"]).median())
    rate = test["neighborhood"].map(ppsf).fillna(fallback)
    return (rate * test["gr_liv_area"]).to_numpy(dtype=float)


def evaluate(model: AVM, test: pd.DataFrame, train: pd.DataFrame | None = None) -> dict[str, Any]:
    """Score a fitted AVM on later sales it never saw."""
    actual = test[TARGET].to_numpy(dtype=float)
    p = model.log_predictions(test)
    gbm, hed = np.exp(p["gbm"]), np.exp(p["hedonic"])
    blend = np.exp((p["gbm"] + p["hedonic"]) / 2)
    pred = model.predict(test)

    bands: dict[str, Any] = {}
    for b in BANDS:
        mask = (pred["confidence_band"] == b).to_numpy()
        if mask.sum() >= 5:
            m = avm_metrics(actual[mask], pred["estimate"].to_numpy()[mask])
            inside = (actual[mask] >= pred["low"].to_numpy()[mask]) & (
                actual[mask] <= pred["high"].to_numpy()[mask]
            )
            bands[b] = {
                "n": int(mask.sum()),
                "mdape": m["mdape"],
                "ppe10": m["ppe10"],
                "range_coverage": float(inside.mean()),
            }
    inside_all = (actual >= pred["low"].to_numpy()) & (actual <= pred["high"].to_numpy())
    return {
        "train_rows": model.train_rows,
        "train_years": list(model.train_years),
        "test_rows": len(test),
        "test_years": [int(test["yr_sold"].min()), int(test["yr_sold"].max())],
        "hedonic": avm_metrics(actual, hed),
        "gbm": avm_metrics(actual, gbm),
        "blend": avm_metrics(actual, blend),
        "baseline_ppsf": avm_metrics(actual, naive_baseline(train, test))
        if train is not None
        else None,
        "confidence_bands": bands,
        "range_coverage_target": 0.80,
        "range_coverage": float(inside_all.mean()),
        "importance_gbm": permutation_importance(model, test, "gbm"),
        "importance_hedonic": permutation_importance(model, test, "hedonic"),
    }


def page_model(model: AVM) -> dict[str, Any]:
    """Everything the static page needs to run the hedonic model in the browser."""
    co = model.coefficients()
    return {
        "note": "Hedonic model only. Analysis and education, not an appraisal.",
        "valuation_year": 2010,
        "intercept": co["intercept"],
        "numeric": co["numeric"],
        "neighborhood": co["neighborhood"],
        "range_half_width": model.hedonic_half_width,
        "bounds": {k: list(v) for k, v in model.bounds.items()},
        "reference": model.reference,
        "train_rows": model.train_rows,
    }
