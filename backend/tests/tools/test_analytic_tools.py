from adsci.tools.base import default_registry

CUSTOMERS = {"source": "files", "sql": "select * from customers"}


async def _run(ctx, name, args):
    out = await default_registry().execute(name, args, ctx)
    assert not out.is_error, out.summary
    return out


async def test_profile_tool(ctx_customers):
    ctx = ctx_customers
    out = await _run(ctx, "profile_dataset", {"data": CUSTOMERS, "target": "churned"})
    kinds = [b.type for b in out.ui]
    assert kinds[0] == "metrics" and "table" in kinds and "chart" in kinds
    assert "customer_id" in out.summary and "id_like" in out.summary
    assert out.data["n_rows"] == 5000
    assert "profile_frame(" in out.code and len(ctx.session.notebook.cells) == 1


async def test_stats_tool_group_comparison(ctx):
    out = await _run(ctx, "statistical_test", {
        "data": {"source": "files", "sql": "select plan, monthly_spend from customers"},
        "kind": "compare_groups", "value": "monthly_spend", "group": "plan",
    })
    assert out.data["p_value"] < 0.001 and "effect" in out.summary.lower()
    assert any(b.type == "chart" for b in out.ui)
    assert "compare_groups(" in out.code


async def test_stats_tool_missing_args_is_helpful_error(ctx):
    out = await default_registry().execute("statistical_test", {
        "data": CUSTOMERS, "kind": "compare_groups", "value": "monthly_spend"}, ctx)
    assert out.is_error and "group" in out.summary


async def test_train_model_tool(ctx):
    out = await _run(ctx, "train_model", {"data": CUSTOMERS, "target": "churned"})
    assert out.data["task"] == "classification"
    assert "baseline" in out.summary and "customer_id" in out.summary
    assert any(b.type == "chart" for b in out.ui) and "train_and_evaluate(" in out.code


DAILY = {"source": "files", "sql": "select order_date, sum(revenue) as revenue from orders group by 1"}


async def test_forecast_tool(ctx):
    out = await _run(ctx, "forecast", {"data": DAILY, "date_col": "order_date", "value_col": "revenue", "horizon": 14})
    assert len(out.data["forecast"]) == 14 and "MAPE" in out.summary
    assert any(b.type == "chart" for b in out.ui) and "forecast_series(" in out.code


async def test_anomaly_tool(ctx):
    out = await _run(ctx, "detect_anomalies", {"data": DAILY, "date_col": "order_date", "value_col": "revenue"})
    assert "2024-11-05" in out.summary and "detect_anomalies(" in out.code


async def test_segment_tool(ctx):
    out = await _run(ctx, "segment", {"data": CUSTOMERS, "features": ["tenure_months", "monthly_spend", "nps"]})
    assert out.data["k"] >= 2 and "silhouette" in out.summary.lower()
    assert "labels" not in out.data  # per-row labels are too big for grounding data
