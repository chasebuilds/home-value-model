import pandas as pd

from home_value_model.data import NUMERIC, TARGET, synthetic, time_split
from home_value_model.features import ENGINEERED, engineer


def test_synthetic_is_seeded() -> None:
    a, b, c = synthetic(200, 1), synthetic(200, 1), synthetic(200, 2)
    pd.testing.assert_frame_equal(a, b)
    assert not a[TARGET].equals(c[TARGET])


def test_synthetic_looks_sane(sample: pd.DataFrame) -> None:
    assert len(sample) == 600
    assert sample[TARGET].between(20_000, 2_000_000).all()
    assert (sample["year_remod"] >= sample["year_built"]).all()
    assert set(NUMERIC) <= set(sample.columns)


def test_time_split_never_leaks(sample: pd.DataFrame) -> None:
    train, test = time_split(sample, 2009)
    assert train["yr_sold"].max() < 2009 <= test["yr_sold"].min()
    assert len(train) + len(test) == len(sample)


def test_engineer_columns_and_age(sample: pd.DataFrame) -> None:
    x = engineer(sample)
    assert list(x.columns) == [*ENGINEERED, "neighborhood"]
    assert (x["age"] >= 0).all()
    row = sample.iloc[[0]].drop(columns=["yr_sold"])
    assert engineer(row, 2010)["age"].iloc[0] == max(0, 2010 - row["year_built"].iloc[0])
