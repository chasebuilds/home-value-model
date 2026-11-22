"""The AVM: a hedonic log-price regression, a gradient-boosted model, and a confidence layer."""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import NUMERIC, TARGET
from .features import ENGINEERED, engineer
from .metrics import avm_metrics

BANDS = ("low", "medium", "high")
HIGH_SCORE, MEDIUM_SCORE = 67, 33
INTERVAL_QUANTILE = 0.80
DISAGREE_FLAG = 0.15
OOR_PENALTY = 0.05
LABELS = {
    "log_gla": "living area",
    "log_lot": "lot size",
    "qual": "overall quality",
    "cond": "overall condition",
    "age": "age",
    "remod_age": "years since remodel",
    "has_bsmt": "has basement",
    "log_bsmt": "basement size",
    "baths": "bathrooms",
    "bedrooms": "bedrooms",
    "garage_cars": "garage spaces",
    "fireplaces": "fireplaces",
}


def _design(model: Any) -> Pipeline:
    cols = ColumnTransformer(
        [
            ("hood", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["neighborhood"]),
            ("num", "passthrough", ENGINEERED),
        ]
    )
    return Pipeline([("cols", cols), ("model", model)])


def make_hedonic() -> Pipeline:
    return _design(Ridge(alpha=1.0))


def make_gbm(seed: int = 0) -> Pipeline:
    gbm = HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=400,
        max_leaf_nodes=15,
        min_samples_leaf=15,
        l2_regularization=1.0,
        random_state=seed,
    )
    return _design(gbm)


def _error_model(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=120, max_depth=3, min_samples_leaf=20, random_state=seed
    )


class AVM:
    """Fit on a training period, then estimate values with a confidence score and range."""

    def __init__(self, seed: int = 0, folds: int = 5) -> None:
        self.seed = seed
        self.folds = folds
        self.hedonic = make_hedonic()
        self.gbm = make_gbm(seed)
        self.err_model = _error_model(seed)
        self.bounds: dict[str, tuple[float, float]] = {}
        self.hoods: set[str] = set()
        self.risk_sorted = np.array([])
        self.half_width: dict[str, float] = {}
        self.hedonic_half_width = 0.0
        self.reference: dict[str, float] = {}
        self.train_rows = 0
        self.train_years: tuple[int, int] = (0, 0)

    # ---- fitting -------------------------------------------------------------------------
    def fit(self, train: pd.DataFrame) -> AVM:
        x = engineer(train)
        y = np.log(train[TARGET].to_numpy(dtype=float))
        kf = KFold(self.folds, shuffle=True, random_state=self.seed)
        oof_h = cross_val_predict(make_hedonic(), x, y, cv=kf)
        oof_g = cross_val_predict(make_gbm(self.seed), x, y, cv=kf)
        err_x = self._error_features(x, oof_g, oof_h)
        err_y = np.abs(oof_g - y)
        oof_risk = cross_val_predict(_error_model(self.seed), err_x, err_y, cv=kf)

        self.hedonic.fit(x, y)
        self.gbm.fit(x, y)
        self.err_model.fit(err_x, err_y)

        self.bounds = {
            c: (float(x[c].quantile(0.01)), float(x[c].quantile(0.99))) for c in ENGINEERED
        }
        self.hoods = set(x["neighborhood"])
        risk = oof_risk + OOR_PENALTY * self._out_of_range_count(x)
        self.risk_sorted = np.sort(risk)
        bands = self._band(self._score(risk))
        self.half_width = {
            b: float(np.quantile(err_y[bands == b], INTERVAL_QUANTILE)) for b in BANDS
        }
        self.hedonic_half_width = float(np.quantile(np.abs(oof_h - y), INTERVAL_QUANTILE))
        self.reference = {c: float(x[c].median()) for c in ENGINEERED}
        self.train_rows = len(train)
        self.train_years = (int(train["yr_sold"].min()), int(train["yr_sold"].max()))
        return self

    @staticmethod
    def _error_features(x: pd.DataFrame, log_g: Any, log_h: Any) -> pd.DataFrame:
        f = x[ENGINEERED].copy()
        f["disagreement"] = np.abs(np.asarray(log_g) - np.asarray(log_h))
        return f

    def _out_of_range_count(self, x: pd.DataFrame) -> np.ndarray:
        n = np.zeros(len(x))
        for c, (lo, hi) in self.bounds.items():
            n += ((x[c] < lo) | (x[c] > hi)).to_numpy()
        n += (~x["neighborhood"].isin(self.hoods)).to_numpy() * 2
        return n

    def _score(self, risk: np.ndarray) -> np.ndarray:
        """Confidence 0-100: share of training homes that looked riskier than this one."""
        rank = np.searchsorted(self.risk_sorted, risk, side="left") / len(self.risk_sorted)
        return np.round(100 * (1 - rank))

    @staticmethod
    def _band(scores: np.ndarray) -> np.ndarray:
        high = scores >= HIGH_SCORE
        return np.where(high, "high", np.where(scores >= MEDIUM_SCORE, "medium", "low"))

    # ---- prediction ----------------------------------------------------------------------
    def log_predictions(
        self, df: pd.DataFrame, valuation_year: int | None = None
    ) -> dict[str, Any]:
        x = engineer(df, valuation_year)
        return {"x": x, "gbm": self.gbm.predict(x), "hedonic": self.hedonic.predict(x)}

    def predict(self, df: pd.DataFrame, valuation_year: int | None = None) -> pd.DataFrame:
        p = self.log_predictions(df, valuation_year)
        x, lg, lh = p["x"], p["gbm"], p["hedonic"]
        err = self.err_model.predict(self._error_features(x, lg, lh))
        score = self._score(np.asarray(err) + OOR_PENALTY * self._out_of_range_count(x))
        band = self._band(score)
        hw = np.array([self.half_width[b] for b in band])
        out = pd.DataFrame(index=df.index)
        out["estimate"] = np.exp(lg).round(-2)
        out["low"] = np.exp(lg - hw).round(-2)
        out["high"] = np.exp(lg + hw).round(-2)
        out["hedonic_estimate"] = np.exp(lh).round(-2)
        out["confidence_score"] = score.astype(int)
        out["confidence_band"] = band
        out["flags"] = [self._flags(x.iloc[i], lg[i], lh[i]) for i in range(len(x))]
        return out

    def _flags(self, row: pd.Series, lg: float, lh: float) -> list[str]:
        flags = []
        if row["neighborhood"] not in self.hoods:
            flags.append(f"neighborhood '{row['neighborhood']}' was not in the training data")
        for c, (lo, hi) in self.bounds.items():
            if row[c] < lo or row[c] > hi:
                flags.append(f"{LABELS[c]} is outside the range of most training homes")
        if abs(lg - lh) > DISAGREE_FLAG:
            flags.append("the two models disagree by more than 15%")
        return flags

    # ---- explanation ---------------------------------------------------------------------
    def coefficients(self) -> dict[str, Any]:
        names = list(self.hedonic.named_steps["cols"].get_feature_names_out())
        ridge = self.hedonic.named_steps["model"]
        pairs = list(zip(names, ridge.coef_, strict=True))
        return {
            "intercept": float(ridge.intercept_),
            "numeric": {
                n.removeprefix("num__"): float(c) for n, c in pairs if n.startswith("num__")
            },
            "neighborhood": {
                n.removeprefix("hood__neighborhood_"): float(c)
                for n, c in pairs
                if n.startswith("hood__")
            },
        }

    def drivers(self, subject: pd.DataFrame, top: int = 5) -> list[dict[str, Any]]:
        """Hedonic price effects vs a typical training home, as a percent each."""
        x = engineer(subject).iloc[0]
        co = self.coefficients()
        out: list[dict[str, Any]] = []
        for c in ENGINEERED:
            eff = co["numeric"][c] * (x[c] - self.reference[c])
            out.append({"feature": LABELS[c], "effect_pct": round(100 * float(np.expm1(eff)), 1)})
        mean_hood = float(np.mean(list(co["neighborhood"].values())))
        h = co["neighborhood"].get(x["neighborhood"], mean_hood) - mean_hood
        out.append({"feature": "neighborhood", "effect_pct": round(100 * float(np.expm1(h)), 1)})
        out.sort(key=lambda d: -abs(float(d["effect_pct"])))
        return out[:top]

    def save(self, path: Any) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: Any) -> AVM:
        obj = joblib.load(path)
        if not isinstance(obj, AVM):
            raise TypeError("not an AVM model file")
        return obj


def permutation_importance(
    model: AVM, test: pd.DataFrame, which: str = "gbm", repeats: int = 5, seed: int = 0
) -> dict[str, float]:
    """Rise in median absolute % error when one raw input is shuffled across homes."""
    rng = np.random.default_rng(seed)
    actual = test[TARGET].to_numpy(dtype=float)

    def mdape(frame: pd.DataFrame) -> float:
        est = np.exp(model.log_predictions(frame)[which])
        return avm_metrics(actual, est)["mdape"]

    base = mdape(test)
    result = {}
    for col in [*NUMERIC, "neighborhood"]:
        rises = []
        for _ in range(repeats):
            shuffled = test.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            rises.append(mdape(shuffled) - base)
        result[col] = float(np.mean(rises))
    return dict(sorted(result.items(), key=lambda kv: -kv[1]))
