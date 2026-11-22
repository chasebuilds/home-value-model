from pathlib import Path

import pandas as pd

from home_value_model.data import TARGET
from home_value_model.metrics import avm_metrics
from home_value_model.model import AVM, permutation_importance
from home_value_model.report import evaluate, page_model

Split = tuple[pd.DataFrame, pd.DataFrame]


def test_beats_naive_on_later_sales(fitted: AVM, split: Split) -> None:
    train, test = split
    pred = fitted.predict(test)
    m = avm_metrics(test[TARGET].to_numpy(), pred["estimate"].to_numpy())
    naive = avm_metrics(test[TARGET].to_numpy(), [train[TARGET].median()] * len(test))
    assert m["mdape"] < naive["mdape"] / 2
    assert m["ppe10"] > 0.45


def test_prediction_columns_and_range(fitted: AVM, split: Split) -> None:
    pred = fitted.predict(split[1])
    assert (pred["low"] < pred["estimate"]).all() and (pred["estimate"] < pred["high"]).all()
    assert pred["confidence_score"].between(0, 100).all()
    assert set(pred["confidence_band"]) <= {"low", "medium", "high"}


def test_confidence_orders_error(fitted: AVM, split: Split) -> None:
    bands = evaluate(fitted, split[1])["confidence_bands"]
    assert bands["high"]["mdape"] < bands["low"]["mdape"]


def test_unseen_neighborhood_is_flagged(fitted: AVM, split: Split) -> None:
    row = split[1].iloc[[0]].copy()
    row["neighborhood"] = "Nowhere"
    out = fitted.predict(row).iloc[0]
    assert any("Nowhere" in f for f in out["flags"])


def test_extreme_home_gets_low_confidence(fitted: AVM, split: Split) -> None:
    row = split[1].iloc[[0]].copy()
    row["gr_liv_area"] = 9000
    row["lot_area"] = 200000
    out = fitted.predict(row).iloc[0]
    assert out["confidence_band"] == "low" and out["flags"]


def test_hedonic_signs_are_sensible(fitted: AVM) -> None:
    num = fitted.coefficients()["numeric"]
    assert num["log_gla"] > 0 and num["qual"] > 0 and num["age"] < 0


def test_drivers_sorted(fitted: AVM, split: Split) -> None:
    d = fitted.drivers(split[1].iloc[[0]])
    mags = [abs(x["effect_pct"]) for x in d]
    assert mags == sorted(mags, reverse=True) and len(d) == 5


def test_permutation_importance_finds_size(fitted: AVM, split: Split) -> None:
    imp = permutation_importance(fitted, split[1], repeats=2)
    assert imp["gr_liv_area"] > 0
    assert list(imp)[0] in {"gr_liv_area", "overall_qual", "neighborhood"}


def test_save_load_roundtrip(fitted: AVM, split: Split, tmp_path: Path) -> None:
    p = tmp_path / "m.joblib"
    fitted.save(p)
    again = AVM.load(p)
    pd.testing.assert_frame_equal(fitted.predict(split[1]), again.predict(split[1]))


def test_page_model_has_everything(fitted: AVM) -> None:
    pm = page_model(fitted)
    assert {"intercept", "numeric", "neighborhood", "range_half_width", "reference"} <= set(pm)
