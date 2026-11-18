"""Record every executed step as a runnable Jupyter notebook."""

from __future__ import annotations

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook, new_output

from adsci.kernel.bootstrap import PLOTLY_MIME
from adsci.kernel.manager import ImageOutput, Output, PlotlyOutput, TableOutput, TextOutput


def _table_text(o: TableOutput) -> str:
    lines = ["\t".join(o.columns)] + ["\t".join("" if v is None else str(v) for v in r) for r in o.rows]
    if o.total_rows > len(o.rows):
        lines.append(f"... ({o.total_rows} rows)")
    return "\n".join(lines)


def _nb_outputs(outputs: list[Output]) -> list:
    nb_out = []
    for i, o in enumerate(outputs, start=1):
        if isinstance(o, TextOutput):
            nb_out.append(new_output("stream", name="stdout", text=o.text))
        elif isinstance(o, TableOutput):
            nb_out.append(new_output("execute_result", data={"text/plain": _table_text(o)}, execution_count=i))
        elif isinstance(o, PlotlyOutput):
            nb_out.append(new_output("display_data", data={PLOTLY_MIME: o.figure, "text/plain": "<Figure>"}))
        elif isinstance(o, ImageOutput):
            nb_out.append(new_output("display_data", data={"image/png": o.png_b64, "text/plain": "<Figure>"}))
    return nb_out


class NotebookRecorder:
    """Accumulates cells. Connection credentials never appear: sources are referenced by name."""

    def __init__(self, session_id: str, base_url: str, token: str, sources: list[str] | None = None) -> None:
        self.session_id = session_id
        self.base_url = base_url
        self.token = token
        self.sources = list(sources or [])
        self.cells: list = []

    def add_markdown(self, text: str) -> None:
        self.cells.append(new_markdown_cell(text))

    def add_python(self, code: str, outputs: list[Output]) -> None:
        self.cells.append(new_code_cell(code, outputs=_nb_outputs(outputs)))

    def add_sql(self, query: str, source: str, outputs: list[Output]) -> None:
        code = f'df = sql("""{query}""", source="{source}")\ndf'
        self.cells.append(new_code_cell(code, outputs=_nb_outputs(outputs)))

    def to_ipynb(self) -> str:
        setup = (
            "# Setup: reconnect to the AI Data Scientist session that produced this notebook.\n"
            "# The session token only works on this machine while that session is open.\n"
            "from adsci.kernel.bootstrap import make_sql_client, setup_display, show\n"
            "import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go\n"
            "import scipy.stats as stats, statsmodels.api as sm\n"
            "from adsci.analytics import *\n"
            f'sql = make_sql_client("{self.base_url}", session_id="{self.session_id}", token="{self.token}")\n'
            "setup_display()"
        )
        nb = new_notebook(
            cells=[
                new_markdown_cell(
                    "# AI Data Scientist session\n"
                    f"Data sources: {', '.join(f'`{s}`' for s in self.sources) or 'none'}."
                ),
                new_code_cell(setup),
                *self.cells,
            ],
            metadata={
                "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                "adsci": {"session_id": self.session_id, "sources": self.sources},
            },
        )
        return nbformat.writes(nb)
