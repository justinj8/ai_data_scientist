"""Answer verification. (Numeric grounding and the critic pass are added in the next task.)"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from adsci.config import Settings


class VerificationResult(BaseModel):
    status: Literal["verified", "warnings"] = "verified"
    ungrounded: list[str] = []
    issues: list[str] = []


class Verifier:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def verify(self, question, answer, outputs, adapter, model) -> VerificationResult:
        return VerificationResult()
