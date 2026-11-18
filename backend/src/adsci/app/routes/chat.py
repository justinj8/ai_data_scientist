"""Ask a question: the agent's work streams back as Server-Sent Events."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, StringConstraints
from sse_starlette.sse import EventSourceResponse

from adsci.agent.events import ErrorEvent
from adsci.llm.base import ProviderError

router = APIRouter(prefix="/api")


class AskBody(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)]
    model: str | None = None  # "opus" | "sonnet" | "gpt" | "grok" | a model id; default: opus


@router.post("/sessions/{sid}/ask")
async def ask(request: Request, sid: str, body: AskBody) -> EventSourceResponse:
    session = request.app.state.sessions.get(sid)
    if session.lock.locked():
        raise HTTPException(status_code=409, detail="This session is already answering a question.")
    orchestrator = request.app.state.orchestrator

    async def events():
        async with session.lock:
            try:
                adapter, model = await request.app.state.models.resolve(body.model)
            except ProviderError as e:
                err = ErrorEvent(message=e.message, user_action=e.user_action)
                yield {"event": err.type, "data": err.model_dump_json()}
                return
            async for event in orchestrator.run(session, body.question, adapter, model):
                yield {"event": event.type, "data": event.model_dump_json()}

    return EventSourceResponse(events(), ping=15)
