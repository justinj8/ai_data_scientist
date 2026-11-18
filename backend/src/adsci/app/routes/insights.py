"""Dataset overview, the insights board, question history and exports."""

from __future__ import annotations

import asyncio
import secrets
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlglot import exp

from adsci.app.overview import build_overview
from adsci.exports.report import build_report
from adsci.kernel.notebook import NotebookRecorder
from adsci.tools.base import UIBlock

router = APIRouter(prefix="/api")
OVERVIEW_ROWS = 200_000


class Insight(BaseModel):
    id: str = Field(default_factory=lambda: secrets.token_hex(6))
    question_id: str | None = None
    title: str
    block: UIBlock
    note: str | None = None
    order: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class NewInsight(BaseModel):
    question_id: str | None = None
    title: str = Field(min_length=1, max_length=200)
    block: UIBlock
    note: str | None = None


class InsightPatch(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=200)
    note: str | None = None
    order: int | None = Field(None, ge=0)


def _session(request: Request, sid: str):
    return request.app.state.sessions.get(sid)


@router.get("/sessions/{sid}/overview")
async def overview(request: Request, sid: str, source: str, table: str) -> dict:
    s = _session(request, sid)
    try:
        src = s.get_source(source)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e.args[0])) from None
    if table not in {t.name for t in src.schema()}:
        raise HTTPException(status_code=404, detail=f"No table '{table}' in source '{source}'.")
    sql = exp.select("*").from_(exp.to_table(table, dialect=src.dialect)).sql(dialect=src.dialect)
    df = await asyncio.to_thread(src.query, sql, OVERVIEW_ROWS)
    out = await asyncio.to_thread(build_overview, df)
    out["sampled"] = len(df) >= OVERVIEW_ROWS
    return out


@router.get("/sessions/{sid}/insights")
def list_insights(request: Request, sid: str) -> dict:
    s = _session(request, sid)
    return {"insights": [i.model_dump(mode="json") for i in sorted(s.insights, key=lambda i: i.order)]}


@router.post("/sessions/{sid}/insights")
def add_insight(request: Request, sid: str, body: NewInsight) -> dict:
    s = _session(request, sid)
    ins = Insight(**body.model_dump(), order=len(s.insights))
    s.insights.append(ins)
    return ins.model_dump(mode="json")


@router.patch("/sessions/{sid}/insights/{iid}")
def patch_insight(request: Request, sid: str, iid: str, body: InsightPatch) -> dict:
    s = _session(request, sid)
    items = sorted(s.insights, key=lambda i: i.order)
    ins = next((i for i in items if i.id == iid), None)
    if ins is None:
        raise HTTPException(status_code=404, detail="Insight not found.")
    if body.title is not None:
        ins.title = body.title
    if body.note is not None:
        ins.note = body.note
    if body.order is not None:
        items.remove(ins)
        items.insert(min(body.order, len(items)), ins)
        for n, i in enumerate(items):
            i.order = n
    return ins.model_dump(mode="json")


@router.delete("/sessions/{sid}/insights/{iid}", status_code=204)
def delete_insight(request: Request, sid: str, iid: str) -> Response:
    s = _session(request, sid)
    before = len(s.insights)
    s.insights = [i for i in s.insights if i.id != iid]
    if len(s.insights) == before:
        raise HTTPException(status_code=404, detail="Insight not found.")
    for n, i in enumerate(sorted(s.insights, key=lambda i: i.order)):
        i.order = n
    return Response(status_code=204)


@router.get("/sessions/{sid}/questions")
def questions(request: Request, sid: str) -> dict:
    return {"questions": [q.model_dump(mode="json") for q in _session(request, sid).questions]}


@router.get("/sessions/{sid}/export/notebook")
def export_notebook(request: Request, sid: str) -> Response:
    s = _session(request, sid)
    nb = s.notebook or NotebookRecorder(s.id, s.settings.internal_base_url, s.internal_token, list(s.sources))
    return Response(nb.to_ipynb(), media_type="application/x-ipynb+json",
                    headers={"Content-Disposition": 'attachment; filename="ai-data-scientist-analysis.ipynb"'})


@router.get("/sessions/{sid}/export/report")
def export_report(request: Request, sid: str) -> Response:
    return Response(build_report(_session(request, sid)), media_type="text/markdown; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="ai-data-scientist-report.md"'})
