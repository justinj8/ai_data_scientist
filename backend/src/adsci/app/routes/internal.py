"""Localhost-only endpoint the analysis kernel uses to run guarded SQL."""

from __future__ import annotations

import hmac
import io

from fastapi import APIRouter, Header, HTTPException, Request, Response
from pydantic import BaseModel

from adsci.data.sql_guard import UnsafeSQLError

router = APIRouter(prefix="/internal")
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


class SQLRequest(BaseModel):
    query: str
    source: str = "files"


@router.post("/sessions/{sid}/sql")
def kernel_sql(request: Request, sid: str, body: SQLRequest, x_adsci_token: str = Header("")) -> Response:
    if request.client is None or request.client.host not in LOCAL_HOSTS:
        raise HTTPException(status_code=403, detail="Forbidden.")
    s = request.app.state.sessions.get(sid)
    if not hmac.compare_digest(x_adsci_token.encode(), s.internal_token.encode()):
        raise HTTPException(status_code=403, detail="Invalid session token.")
    try:
        df = s.get_source(body.source).query(body.query)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e.args[0])) from None
    except UnsafeSQLError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:  # noqa: BLE001 - database errors go back to the analyst's code
        raise HTTPException(status_code=400, detail=f"Query failed: {str(e).splitlines()[0]}") from None
    buf = io.BytesIO()
    df.columns = [str(c) for c in df.columns]
    df.to_parquet(buf, index=False)
    return Response(content=buf.getvalue(), media_type="application/vnd.apache.parquet")
