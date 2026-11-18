from pathlib import Path

import pytest

from adsci.app.sessions import SessionManager
from adsci.tools.base import ToolContext

SAMPLE = Path(__file__).resolve().parents[3] / "sample_data" / "out"


@pytest.fixture
def session(settings):
    mgr = SessionManager(settings)
    s = mgr.create()
    s.files.add_file(SAMPLE / "orders.csv")
    s.files.add_file(SAMPLE / "customers.csv")
    yield s
    mgr.close(s.id)


@pytest.fixture
def ctx(session, settings):
    return ToolContext(session=session, settings=settings)


@pytest.fixture
def live_ctx(live_app):
    """A ToolContext whose session lives in a running app, so the kernel's sql() can reach it."""
    app, _ = live_app
    s = app.state.sessions.create()
    s.files.add_file(SAMPLE / "orders.csv")
    yield ToolContext(session=s, settings=app.state.settings)
