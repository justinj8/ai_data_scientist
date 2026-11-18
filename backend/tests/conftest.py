import pytest
from fastapi.testclient import TestClient

from adsci.app.main import create_app
from adsci.config import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data")


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c
