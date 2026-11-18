"""Events streamed to the UI (as Server-Sent Events) while the agent works on a question."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from adsci.llm.base import UserAction
from adsci.llm.types import ModelInfo, Usage
from adsci.tools.base import UIBlock


class ModelEvent(BaseModel):
    type: Literal["model"] = "model"
    question_id: str
    model: ModelInfo
    billed_to: str | None = None


class TextDeltaEvent(BaseModel):
    """Live text of the current model turn; committed later as a plan, note or answer."""

    type: Literal["text_delta"] = "text_delta"
    text: str


class PlanEvent(BaseModel):
    type: Literal["plan"] = "plan"
    text: str


class NoteEvent(BaseModel):
    type: Literal["note"] = "note"
    text: str


class StepStartEvent(BaseModel):
    type: Literal["step_start"] = "step_start"
    step_id: str
    tool: str
    purpose: str | None = None
    code: str | None = None
    language: Literal["sql", "python"] | None = None
    args: dict


class StepOutputEvent(BaseModel):
    type: Literal["step_output"] = "step_output"
    step_id: str
    blocks: list[UIBlock]
    is_error: bool
    duration_s: float
    code: str | None = None
    language: Literal["sql", "python"] | None = None


class AnswerEvent(BaseModel):
    type: Literal["answer"] = "answer"
    text: str
    revised: bool = False


class VerificationEvent(BaseModel):
    type: Literal["verification"] = "verification"
    status: Literal["verified", "warnings"]
    ungrounded: list[str] = []
    issues: list[str] = []


class UsageEvent(BaseModel):
    type: Literal["usage"] = "usage"
    usage: Usage
    model_id: str


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    question_id: str
    answer: str
    steps: int


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str
    user_action: UserAction = "none"


AgentEvent = Annotated[
    ModelEvent | TextDeltaEvent | PlanEvent | NoteEvent | StepStartEvent | StepOutputEvent | AnswerEvent
    | VerificationEvent | UsageEvent | DoneEvent | ErrorEvent,
    Field(discriminator="type"),
]
