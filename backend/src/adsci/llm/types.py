"""Provider-neutral LLM types. Adapters convert these to and from each provider's wire format."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolCallPart(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict


class ToolResultPart(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    content: str
    is_error: bool = False


Part = Annotated[TextPart | ToolCallPart | ToolResultPart, Field(discriminator="type")]


class Message(BaseModel):
    """A conversation turn. Tool results travel in ``user`` messages."""

    role: Literal["user", "assistant"]
    parts: list[Part]
    # The provider's raw content for this turn (e.g. Anthropic thinking blocks with signatures, OpenAI reasoning
    # items). Replayed verbatim to the same provider; other providers use ``parts``.
    provider_state: dict | None = None

    @property
    def text(self) -> str:
        return "".join(p.text for p in self.parts if isinstance(p, TextPart))

    @property
    def tool_calls(self) -> list[ToolCallPart]:
        return [p for p in self.parts if isinstance(p, ToolCallPart)]


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict


BilledTo = Literal["api_key", "chatgpt_plan", "supergrok_plan"]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    billed_to: BilledTo = "api_key"


class TextDelta(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class ToolCallEvent(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    call: ToolCallPart


class TurnEnd(BaseModel):
    type: Literal["turn_end"] = "turn_end"
    provider_state: dict | None = None
    stop_reason: Literal["end_turn", "tool_use", "max_tokens", "refusal"]
    usage: Usage


StreamEvent = TextDelta | ToolCallEvent | TurnEnd

Provider = Literal["anthropic", "openai", "xai", "fake"]
Family = Literal["opus", "sonnet", "gpt", "grok", "other"]


class ModelInfo(BaseModel):
    id: str
    provider: Provider
    family: Family
    display_name: str
    created: datetime | None = None
    supports_tools: bool = True
