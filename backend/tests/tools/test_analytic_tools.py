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
