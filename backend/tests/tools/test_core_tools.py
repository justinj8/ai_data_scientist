from adsci.data.base import DataRef
from adsci.tools.base import default_registry


async def test_run_sql_returns_table_and_code(ctx, session):
    reg = default_registry()
    q = "select region, count(*) as n from orders group by region order by region"
    out = await reg.execute("run_sql", {"source": "files", "query": q, "purpose": "count by region"}, ctx)
    assert not out.is_error, out.summary
    assert out.code == q and out.language == "sql"
    table = out.ui[0]
    assert table.type == "table" and table.total_rows == 4 and table.columns == ["region", "n"]
    assert "West" in out.summary
    assert len(session.notebook.cells) == 1


async def test_run_sql_unsafe_is_error(ctx):
    out = await default_registry().execute(
        "run_sql", {"source": "files", "query": "drop table orders", "purpose": "x"}, ctx
    )
    assert out.is_error and "read-only" in out.summary


async def test_run_sql_unknown_source_lists_available(ctx):
    out = await default_registry().execute("run_sql", {"source": "nope", "query": "select 1", "purpose": "x"}, ctx)
    assert out.is_error and "files" in out.summary


async def test_run_python_chart_and_error(live_ctx):
    ctx, session = live_ctx, live_ctx.session
    reg = default_registry()
    out = await reg.execute(
        "run_python",
        {"code": "df = sql('select region, sum(revenue) as r from orders group by 1')\n"
                 "show(px.bar(df, x='region', y='r'))\nprint(len(df))", "purpose": "chart"},
        ctx,
    )
    assert not out.is_error, out.summary
    kinds = [b.type for b in out.ui]
    assert "chart" in kinds and "text" in kinds and "4" in out.summary
    err = await reg.execute("run_python", {"code": "undefined_name", "purpose": "x"}, ctx)
    assert err.is_error and "NameError" in err.summary and err.ui[-1].type == "error"
    assert len(session.notebook.cells) == 2


async def test_create_chart_aggregates_like_sql(ctx, session):
    reg = default_registry()
    out = await reg.execute(
        "create_chart",
        {
            "data": DataRef(source="files", sql="select region, revenue from orders").model_dump(),
            "kind": "bar", "x": "region", "y": "revenue", "agg": "sum", "title": "Revenue by region",
        },
        ctx,
    )
    assert not out.is_error, out.summary
    chart = out.ui[0]
    assert chart.type == "chart" and chart.title == "Revenue by region"
    expected = session.files.query("select sum(revenue) as s from orders")["s"][0]
    total = sum(chart.figure["data"][0]["y"])
    assert abs(total - expected) < 1e-3  # analytic loads are NOT capped at the 10k preview limit
    assert len(session.notebook.cells) == 1
