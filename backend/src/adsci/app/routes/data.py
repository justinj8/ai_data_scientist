"""Sessions, file uploads, database connections, sample data and schema."""

from __future__ import annotations

import re
import shutil
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel, Field

from adsci.app.sessions import FILES_SOURCE, Session
from adsci.data.base import LoadReport, TableInfo
from adsci.data.databases import ConnectionConfig, ConnectionFailed, SQLAlchemyDataSource
from adsci.samples import ensure_samples

router = APIRouter(prefix="/api")
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
ALLOWED_SUFFIXES = {".csv", ".xlsx"}
SAMPLE_SQLITE = "retail_sqlite"


def _session(request: Request, sid: str) -> Session:
    return request.app.state.sessions.get(sid)


class SourceOut(BaseModel):
    name: str
    kind: str
    dialect: str
    tables: list[TableInfo]


class NewConnection(BaseModel):
    name: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")]
    config: ConnectionConfig


@router.post("/sessions")
def create_session(request: Request) -> dict[str, str]:
    return {"session_id": request.app.state.sessions.create().id}


@router.get("/sessions/{sid}/schema")
def get_schema(request: Request, sid: str) -> dict[str, list[SourceOut]]:
    s = _session(request, sid)
    out = []
    for name, src in s.sources.items():
        out.append(SourceOut(name=name, kind=src.kind, dialect=src.dialect, tables=src.schema()))
    return {"sources": out}


@router.post("/sessions/{sid}/files")
def upload_file(request: Request, sid: str, file: UploadFile = File(...)) -> dict[str, list[LoadReport]]:
    s = _session(request, sid)
    name = re.sub(r"[^\w.\- ]", "_", file.filename or "upload")
    suffix = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail="Only .csv and .xlsx files are supported.")
    dest = s.dir / "uploads" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with dest.open("wb") as out:
        while chunk := file.file.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File is larger than 200 MB.")
            out.write(chunk)
    try:
        reports = s.files.add_file(dest)
    except Exception as e:  # noqa: BLE001 - surface loader errors to the user
        raise HTTPException(status_code=400, detail=f"Could not read {name}: {e}") from None
    return {"tables": reports}


@router.post("/connections/test")
def test_connection(request: Request, cfg: ConnectionConfig) -> dict[str, object]:
    try:
        src = SQLAlchemyDataSource("test", cfg, request.app.state.settings)
        try:
            src.test_connection()
        finally:
            src.close()
    except ConnectionFailed as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "error": None}


@router.post("/sessions/{sid}/connections")
def add_connection(request: Request, sid: str, body: NewConnection) -> dict[str, object]:
    s = _session(request, sid)
    if body.name in s.sources:
        raise HTTPException(status_code=409, detail=f"A source named '{body.name}' already exists.")
    try:
        src = SQLAlchemyDataSource(body.name, body.config, s.settings)
        src.test_connection()
        tables = src.schema()
    except ConnectionFailed as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    s.sources[body.name] = src
    return {"name": body.name, "tables": tables}


@router.delete("/sessions/{sid}/sources/{name}", status_code=204)
def delete_source(request: Request, sid: str, name: str) -> Response:
    s = _session(request, sid)
    src = s.sources.pop(name, None)
    if src is None:
        raise HTTPException(status_code=404, detail=f"No source named '{name}'.")
    src.close()
    return Response(status_code=204)


@router.post("/sessions/{sid}/samples/{dataset}")
def load_sample(request: Request, sid: str, dataset: str) -> dict[str, object]:
    if dataset != "retail":
        raise HTTPException(status_code=404, detail="Unknown sample dataset. Available: retail.")
    s = _session(request, sid)
    sample_dir = ensure_samples(s.settings)
    existing = {t.name for t in s.files.schema()} if FILES_SOURCE in s.sources else set()
    for fname in ("orders.csv", "customers.csv"):
        if fname.removesuffix(".csv") not in existing:
            s.files.add_file(sample_dir / fname)
    if SAMPLE_SQLITE not in s.sources:
        db_copy = s.dir / "retail.db"
        shutil.copy(sample_dir / "retail.db", db_copy)
        cfg = ConnectionConfig(engine="sqlite", database=str(db_copy))
        s.sources[SAMPLE_SQLITE] = SQLAlchemyDataSource(SAMPLE_SQLITE, cfg, s.settings)
    return {"sources": [FILES_SOURCE, SAMPLE_SQLITE]}
