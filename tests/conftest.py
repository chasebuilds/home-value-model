from pathlib import Path

import pandas as pd
import pytest

from home_value_model.data import time_split
from home_value_model.model import AVM

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample_synthetic.csv"


@pytest.fixture(scope="session")
def sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE)


@pytest.fixture(scope="session")
def split(sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return time_split(sample)


@pytest.fixture(scope="session")
def fitted(split: tuple[pd.DataFrame, pd.DataFrame]) -> AVM:
    return AVM(seed=0).fit(split[0])
