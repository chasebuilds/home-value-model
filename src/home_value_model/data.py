"""Data loading: the Ames, Iowa housing data (downloaded at run time) and a seeded synthetic set."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

AMES_URL = "https://jse.amstat.org/v19n3/decock/AmesHousing.txt"
DEFAULT_CACHE = Path.home() / ".cache" / "home-value-model"

NUMERIC = [
    "gr_liv_area",
    "lot_area",
    "overall_qual",
    "overall_cond",
    "year_built",
    "year_remod",
    "total_bsmt_sf",
    "full_bath",
    "half_bath",
    "bedrooms",
    "garage_cars",
    "fireplaces",
]
CATEGORICAL = ["neighborhood"]
TIME = ["yr_sold", "mo_sold"]
TARGET = "sale_price"

_AMES_COLUMNS = {
    "Gr Liv Area": "gr_liv_area",
    "Lot Area": "lot_area",
    "Overall Qual": "overall_qual",
    "Overall Cond": "overall_cond",
    "Year Built": "year_built",
    "Year Remod/Add": "year_remod",
    "Total Bsmt SF": "total_bsmt_sf",
    "Full Bath": "full_bath",
    "Half Bath": "half_bath",
    "Bedroom AbvGr": "bedrooms",
    "Garage Cars": "garage_cars",
    "Fireplaces": "fireplaces",
    "Neighborhood": "neighborhood",
    "Yr Sold": "yr_sold",
    "Mo Sold": "mo_sold",
    "SalePrice": TARGET,
}


def fetch_ames(cache_dir: Path = DEFAULT_CACHE) -> Path:
    """Download the raw Ames file into the user cache (never into the repo)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "AmesHousing.txt"
    if not path.exists():
        req = urllib.request.Request(AMES_URL, headers={"User-Agent": "home-value-model"})
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            path.write_bytes(resp.read())
    return path


def load_ames(path: Path | None = None) -> pd.DataFrame:
    """Load Ames, keep normal arms-length single-family sales, drop the parcel ID.

    Filters follow the data's author (De Cock 2011): normal sale condition only, and
    the five homes above 4,000 sq ft of living area are removed.
    """
    raw = pd.read_csv(path or fetch_ames(), sep="\t")
    raw = raw[(raw["Sale Condition"] == "Normal") & (raw["Bldg Type"] == "1Fam")]
    raw = raw[raw["Gr Liv Area"] <= 4000]
    df = raw[list(_AMES_COLUMNS)].rename(columns=_AMES_COLUMNS).copy()
    for col in ("total_bsmt_sf", "garage_cars"):
        df[col] = df[col].fillna(0)
    df = df.dropna()
    return df.reset_index(drop=True)


def synthetic(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Seeded fake homes with a known price process. Nothing here is a real property."""
    rng = np.random.default_rng(seed)
    hoods = [f"Zone{c}" for c in "ABCDEFGH"]
    hood_effect = dict(zip(hoods, np.linspace(-0.25, 0.30, len(hoods)), strict=True))
    hood = rng.choice(hoods, n, p=[0.06, 0.10, 0.18, 0.20, 0.18, 0.14, 0.09, 0.05])
    gla = np.clip(rng.normal(1500, 450, n), 600, 3800).round()
    lot = np.clip(rng.lognormal(9.1, 0.35, n), 2500, 40000).round()
    qual = np.clip(np.round(rng.normal(6, 1.4, n)), 1, 10)
    cond = np.clip(np.round(rng.normal(5.5, 1.1, n)), 1, 9)
    yr_sold = rng.integers(2006, 2011, n)
    mo_sold = rng.integers(1, 13, n)
    built = np.clip(yr_sold - rng.gamma(2.0, 18.0, n).round(), 1880, yr_sold)
    remod = np.maximum(built, built + rng.integers(0, 30, n)).clip(max=yr_sold)
    bsmt = np.where(rng.random(n) < 0.9, gla * rng.uniform(0.4, 1.0, n), 0).round()
    full = np.clip(np.round(gla / 900 + rng.normal(0, 0.5, n)), 1, 4)
    half = rng.integers(0, 3, n)
    beds = np.clip(np.round(gla / 550 + rng.normal(0, 0.6, n)), 1, 6)
    garage = rng.choice([0, 1, 2, 3], n, p=[0.05, 0.2, 0.65, 0.1])
    fire = rng.choice([0, 1, 2], n, p=[0.5, 0.4, 0.1])
    log_p = (
        6.4
        + 0.62 * np.log(gla)
        + 0.10 * np.log(lot)
        + 0.085 * qual
        + 0.03 * cond
        - 0.0016 * (yr_sold - built)
        + 0.0012 * (remod - built)
        + 0.00008 * bsmt
        + 0.03 * full
        + 0.015 * half
        + 0.045 * garage
        + 0.03 * fire
        + np.array([hood_effect[h] for h in hood])
        + rng.normal(0, 0.09, n)
    )
    return pd.DataFrame(
        {
            "gr_liv_area": gla,
            "lot_area": lot,
            "overall_qual": qual,
            "overall_cond": cond,
            "year_built": built,
            "year_remod": remod,
            "total_bsmt_sf": bsmt,
            "full_bath": full,
            "half_bath": half,
            "bedrooms": beds,
            "garage_cars": garage,
            "fireplaces": fire,
            "neighborhood": hood,
            "yr_sold": yr_sold,
            "mo_sold": mo_sold,
            TARGET: np.exp(log_p).round(-2),
        }
    )


def time_split(df: pd.DataFrame, test_from: int = 2009) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on sales before `test_from`, test on that year and later (no random shuffle)."""
    train = df[df["yr_sold"] < test_from].reset_index(drop=True)
    test = df[df["yr_sold"] >= test_from].reset_index(drop=True)
    return train, test
