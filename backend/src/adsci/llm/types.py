"""Provider-neutral LLM types."""

from __future__ import annotations

from pydantic import BaseModel


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict
