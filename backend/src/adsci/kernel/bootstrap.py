"""Helpers that run *inside* the analysis kernel (and in exported notebooks).

The kernel never receives database credentials. ``sql()`` asks the backend to run
the query through the same read-only guard, over a localhost endpoint protected
by a per-session token, and receives the result as Parquet.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from collections.abc import Callable

TABLE_MIME = "application/vnd.adsci.table+json"
PLOTLY_MIME = "application/vnd.plotly.v1+json"


class SQLError(Exception):
    """The backend rejected or failed to run a query."""


def make_sql_client(base_url: str, session_id: str, token: str) -> Callable[..., object]:
    import pandas as pd

    endpoint = f"{base_url.rstrip('/')}/internal/sessions/{session_id}/sql"

    def sql(query: str, source: str = "files") -> pd.DataFrame:
        """Run a read-only SQL query against a named data source and return a DataFrame."""
        body = json.dumps({"query": query, "source": source}).encode()
        req = urllib.request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "X-ADSCI-Token": token},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return pd.read_parquet(io.BytesIO(resp.read()))
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read().decode()).get("detail", str(e))
            except Exception:  # noqa: BLE001
                detail = str(e)
            raise SQLError(detail) from None

    return sql


def _table_payload(df, preview_rows: int) -> dict:
    import pandas as pd

    if isinstance(df, pd.Series):
        df = df.to_frame()
    head = df.head(preview_rows)
    if not isinstance(head.index, pd.RangeIndex):
        head = head.reset_index()
    records = json.loads(head.to_json(orient="split", date_format="iso", default_handler=str))
    return {"columns": [str(c) for c in records["columns"]], "rows": records["data"], "total_rows": int(len(df))}


def setup_display(preview_rows: int = 50) -> None:
    """Make DataFrames and Plotly figures render as structured outputs the UI understands."""
    import pandas as pd
    import plotly.io as pio
    from IPython import get_ipython
    from IPython.core.formatters import BaseFormatter

    ip = get_ipython()
    if ip is None:
        return

    class TableFormatter(BaseFormatter):
        format_type = TABLE_MIME
        print_method = "_adsci_table_"  # never present; we register by type instead
        _return_type = dict

    formatter = TableFormatter(parent=ip.display_formatter)
    ip.display_formatter.formatters[TABLE_MIME] = formatter
    formatter.for_type(pd.DataFrame, lambda df: _table_payload(df, preview_rows))
    formatter.for_type(pd.Series, lambda s: _table_payload(s, preview_rows))
    pio.renderers.default = "plotly_mimetype"


def show(fig) -> None:
    """Display a Plotly figure in the analysis timeline."""
    fig.show()


KERNEL_STARTUP = """
import os as _os
from adsci.kernel.bootstrap import make_sql_client as _msc, setup_display as _sd, show, SQLError
sql = _msc(_os.environ["ADSCI_INTERNAL_URL"], _os.environ["ADSCI_SESSION_ID"], _os.environ["ADSCI_TOKEN"])
_sd(int(_os.environ.get("ADSCI_PREVIEW_ROWS", "50")))
import sys as _sys
if _sys.platform.startswith("linux"):
    import resource as _r
    _limit = int(_os.environ.get("ADSCI_KERNEL_MEMORY_BYTES", str(2 * 1024 ** 3)))
    try:
        _r.setrlimit(_r.RLIMIT_AS, (_limit, _limit))
    except (ValueError, OSError):
        pass
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import statsmodels.api as sm
try:
    from adsci.analytics import *  # noqa: F401,F403
except ImportError:
    pass
"""
