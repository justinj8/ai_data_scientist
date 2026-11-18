import json
from pathlib import Path

import pandas as pd
import pytest

SAMPLE = Path(__file__).resolve().parents[3] / "sample_data" / "out"


@pytest.fixture(scope="session")
def customers():
    return pd.read_csv(SAMPLE / "customers.csv")


@pytest.fixture(scope="session")
def orders():
    return pd.read_csv(SAMPLE / "orders.csv", parse_dates=["order_date"])


@pytest.fixture(scope="session")
def truth():
    return json.loads((SAMPLE / "ground_truth.json").read_text())
