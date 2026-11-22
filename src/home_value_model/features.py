"""Feature engineering shared by both models, so the web page can mirror it exactly."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import NUMERIC

DEFAULT_VALUATION_YEAR = 2010

# Numeric model inputs after engineering, in order. Neighborhood is added as dummies.
ENGINEERED = [
    "log_gla",
    "log_lot",
    "qual",
    "cond",
    "age",
    "remod_age",
    "has_bsmt",
    "log_bsmt",
    "baths",
    "bedrooms",
    "garage_cars",
    "fireplaces",
]

REQUIRED = [*NUMERIC, "neighborhood"]


def engineer(df: pd.DataFrame, valuation_year: int | None = None) -> pd.DataFrame:
    """Turn raw subject fields into model inputs.

    `age` is measured to the sale year (or `valuation_year` for a subject with no sale).
    """
    year = df.get("yr_sold", valuation_year or DEFAULT_VALUATION_YEAR)
    bsmt = df["total_bsmt_sf"].astype(float)
    out = pd.DataFrame(index=df.index)
    out["log_gla"] = np.log(df["gr_liv_area"].astype(float))
    out["log_lot"] = np.log(df["lot_area"].astype(float))
    out["qual"] = df["overall_qual"].astype(float)
    out["cond"] = df["overall_cond"].astype(float)
    out["age"] = (year - df["year_built"]).clip(lower=0).astype(float)
    out["remod_age"] = (year - df["year_remod"]).clip(lower=0).astype(float)
    out["has_bsmt"] = (bsmt > 0).astype(float)
    out["log_bsmt"] = np.log1p(bsmt)
    out["baths"] = df["full_bath"].astype(float) + 0.5 * df["half_bath"].astype(float)
    out["bedrooms"] = df["bedrooms"].astype(float)
    out["garage_cars"] = df["garage_cars"].astype(float)
    out["fireplaces"] = df["fireplaces"].astype(float)
    out["neighborhood"] = df["neighborhood"].astype(str)
    return out
