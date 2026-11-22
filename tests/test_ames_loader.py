from pathlib import Path

import pandas as pd

from home_value_model.data import _AMES_COLUMNS, load_ames


def _fake_ames(path: Path) -> None:
    """A tiny made-up file in the Ames column layout (no real records)."""
    base = {
        "Gr Liv Area": 1500, "Lot Area": 8000, "Overall Qual": 6, "Overall Cond": 5,
        "Year Built": 1990, "Year Remod/Add": 2000, "Total Bsmt SF": 800, "Full Bath": 2,
        "Half Bath": 0, "Bedroom AbvGr": 3, "Garage Cars": 2, "Fireplaces": 1,
        "Neighborhood": "NAmes", "Yr Sold": 2008, "Mo Sold": 6, "SalePrice": 180000,
        "Sale Condition": "Normal", "Bldg Type": "1Fam",
    }  # fmt: skip
    rows = [
        dict(base),
        {**base, "Sale Condition": "Family"},
        {**base, "Bldg Type": "Duplex"},
        {**base, "Gr Liv Area": 5000},
        {**base, "Total Bsmt SF": None, "Garage Cars": None},
    ]
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def test_loader_filters_and_renames(tmp_path: Path) -> None:
    p = tmp_path / "AmesHousing.txt"
    _fake_ames(p)
    df = load_ames(p)
    assert len(df) == 2  # normal, single-family, under 4,000 sq ft; missing basement/garage -> 0
    assert set(df.columns) == set(_AMES_COLUMNS.values())
    assert "PID" not in df.columns
    assert df["total_bsmt_sf"].tolist() == [800, 0]
