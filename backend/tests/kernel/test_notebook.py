import json

import nbformat

from adsci.kernel.manager import PlotlyOutput, TableOutput, TextOutput
from adsci.kernel.notebook import NotebookRecorder


def test_notebook_roundtrip_and_no_secrets():
    rec = NotebookRecorder(session_id="abc", base_url="http://127.0.0.1:8000", token="tok", sources=["files", "shop"])
    rec.add_markdown("## Question\nWhich region sells most?")
    rec.add_sql("select region, sum(revenue) from orders group by 1", "files",
                [TableOutput(columns=["region", "sum"], rows=[["West", 1.0]], total_rows=1)])
    rec.add_python("print(1)", [TextOutput(text="1\n"), PlotlyOutput(figure={"data": [], "layout": {}})])
    raw = rec.to_ipynb()
    nb = nbformat.reads(raw, as_version=4)
    nbformat.validate(nb)
    sources = [c.source for c in nb.cells]
    assert any('sql("""select region' in s and 'source="files"' in s for s in sources)
    assert "hunter2" not in raw
    assert json.loads(raw)["metadata"]["adsci"]["sources"] == ["files", "shop"]
