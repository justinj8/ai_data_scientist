"""The agent loop: plan, call tools until the model answers, verify the answer, and stream it all."""

from __future__ import annotations

import secrets
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from pydantic import BaseModel

from adsci.agent.events import (
    AnswerEvent,
    DoneEvent,
    ErrorEvent,
    ModelEvent,
    NoteEvent,
    PlanEvent,
    StepOutputEvent,
    StepStartEvent,
    TextDeltaEvent,
    UsageEvent,
    VerificationEvent,
)
from adsci.agent.history import compact_history
from adsci.agent.prompts import build_system_prompt
from adsci.agent.verifier import VerificationResult, Verifier
from adsci.config import Settings
from adsci.llm.base import LLMAdapter, ProviderError
from adsci.llm.types import (
    Message,
    ModelInfo,
    TextDelta,
    TextPart,
    ToolCallEvent,
    ToolCallPart,
    ToolResultPart,
    TurnEnd,
    Usage,
)
from adsci.tools.base import ToolContext, ToolOutput, ToolRegistry

STEP_LIMIT_NOTE = (
    "You have reached the step limit for this question. Do not call more tools. "
    "Answer now with what you have found, and say what remains uncertain."
)
REVISION_STEPS = 3


class StepRecord(BaseModel):
    step_id: str
    tool: str
    purpose: str | None = None
    code: str | None = None
    language: str | None = None
    blocks: list = []
    is_error: bool = False


class QuestionRecord(BaseModel):
    id: str
    question: str
    model_id: str
    plan: str | None = None
    steps: list[StepRecord] = []
    answer: str = ""
    verification: VerificationResult | None = None


@dataclass
class _Run:
    messages: list[Message]
    outputs: list[ToolOutput] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    steps: int = 0
    answer: str = ""
    record: QuestionRecord | None = None
    plan_seen: bool = False


def _add_usage(total: Usage, u: Usage) -> Usage:
    cost = None if total.cost_usd is None and u.cost_usd is None else (total.cost_usd or 0) + (u.cost_usd or 0)
    return Usage(input_tokens=total.input_tokens + u.input_tokens, output_tokens=total.output_tokens + u.output_tokens,
                 cost_usd=cost, billed_to=u.billed_to)


class Orchestrator:
    def __init__(self, registry: ToolRegistry, verifier: Verifier, settings: Settings) -> None:
        self.registry = registry
        self.verifier = verifier
        self.settings = settings

    async def run(self, session, question: str, adapter: LLMAdapter, model: ModelInfo) -> AsyncIterator:
        qid = secrets.token_hex(6)
        billed = getattr(adapter, "billed_to", "api_key")
        yield ModelEvent(question_id=qid, model=model, billed_to=billed)
        history = compact_history(session.history, self.settings.history_char_budget)
        run = _Run(messages=history + [Message(role="user", parts=[TextPart(text=question)])])
        run.record = QuestionRecord(id=qid, question=question, model_id=model.id)
        ctx = ToolContext(session=session, settings=self.settings)
        tools = self.registry.specs()
        system = build_system_prompt(session, tools)
        ctx.notebook.add_markdown(f"## {question}")
        try:
            async for event in self._loop(run, ctx, adapter, model, system, tools, self.settings.max_agent_steps):
                yield event
            if run.answer is None:
                return
            yield AnswerEvent(text=run.answer)
            result = await self.verifier.verify(question, run.answer, run.outputs, adapter, model)
            if result.status == "warnings":
                feedback = self._revision_request(result)
                run.messages.append(Message(role="user", parts=[TextPart(text=feedback)]))
                async for event in self._loop(run, ctx, adapter, model, system, tools,
                                              run.steps + REVISION_STEPS):
                    yield event
                if run.answer is None:
                    return
                yield AnswerEvent(text=run.answer, revised=True)
                result = await self.verifier.verify(question, run.answer, run.outputs, adapter, model)
        except ProviderError as e:
            yield ErrorEvent(message=e.message, user_action=e.user_action)
            return
        except Exception as e:  # noqa: BLE001 - the stream must end with an event the UI understands
            yield ErrorEvent(message=f"Something went wrong while answering: {type(e).__name__}: {e}",
                             user_action="retry")
            return
        yield VerificationEvent(status=result.status, ungrounded=result.ungrounded, issues=result.issues,
                                notes=result.notes)
        yield UsageEvent(usage=run.usage, model_id=model.id)
        run.record.answer = run.answer
        run.record.verification = result
        ctx.notebook.add_markdown(run.answer)
        session.history = run.messages
        session.questions.append(run.record)
        session.model_id = model.id
        yield DoneEvent(question_id=qid, answer=run.answer, steps=run.steps)

    @staticmethod
    def _revision_request(result: VerificationResult) -> str:
        lines = ["A reviewer checked your answer against the tool results and found problems:"]
        lines += [f"- The number {n} does not appear in any tool result." for n in result.ungrounded]
        lines += [f"- {i}" for i in result.issues]
        lines.append(
            "Revise the answer. Verify or remove each flagged number (run a tool if needed) and fix the issues. "
            "Reply with the complete revised answer."
        )
        return "\n".join(lines)

    async def _loop(self, run: _Run, ctx: ToolContext, adapter: LLMAdapter, model: ModelInfo, system: str,
                    tools, max_steps: int) -> AsyncIterator:
        """Run model turns until the model answers. Sets run.answer (None after a reported error)."""
        limit_noted = False
        while True:
            text_parts: list[str] = []
            calls: list[ToolCallPart] = []
            end: TurnEnd | None = None
            async for ev in adapter.stream(model=model.id, system=system, messages=run.messages, tools=tools):
                if isinstance(ev, TextDelta):
                    text_parts.append(ev.text)
                    yield TextDeltaEvent(text=ev.text)
                elif isinstance(ev, ToolCallEvent):
                    calls.append(ev.call)
                elif isinstance(ev, TurnEnd):
                    end = ev
            text = "".join(text_parts).strip()
            if end is not None:
                run.usage = _add_usage(run.usage, end.usage)
            run.messages.append(Message(
                role="assistant",
                parts=([TextPart(text=text)] if text else []) + calls,
                provider_state=end.provider_state if end else None,
            ))

            if end is not None and end.stop_reason == "refusal":
                run.answer = None
                yield ErrorEvent(message="The model declined to answer this request. Try rephrasing it or "
                                         "switching to another model.", user_action="switch_model")
                return
            if not calls:
                if not text:
                    run.answer = None
                    yield ErrorEvent(message="The model returned an empty response. Please try again.",
                                     user_action="retry")
                    return
                run.answer = text
                return
            if run.steps >= max_steps:
                run.answer = (text + "\n\n" if text else "") + (
                    "I reached the step limit for this question before finishing the analysis. "
                    "The results above are partial; ask a narrower follow-up to continue."
                )
                # Close the dangling tool calls so the history stays valid for the next question.
                run.messages.append(Message(role="user", parts=[
                    ToolResultPart(tool_call_id=c.id, content="Not run: step limit reached.", is_error=True)
                    for c in calls
                ]))
                return

            if text:
                if not run.plan_seen:
                    run.plan_seen = True
                    run.record.plan = text
                    yield PlanEvent(text=text)
                else:
                    yield NoteEvent(text=text)
            results: list[ToolResultPart] = []
            for c in calls:
                if run.steps >= max_steps:
                    results.append(ToolResultPart(tool_call_id=c.id, content="Not run: step limit reached.",
                                                  is_error=True))
                    continue
                run.steps += 1
                step_id = f"{run.record.id}-{run.steps}"
                args = c.arguments
                pre_code, lang = ((args.get("query"), "sql") if c.name == "run_sql"
                                  else (args.get("code"), "python") if c.name == "run_python" else (None, None))
                yield StepStartEvent(step_id=step_id, tool=c.name, purpose=args.get("purpose"), code=pre_code,
                                     language=lang, args=args)
                t0 = time.monotonic()
                out = await self.registry.execute(c.name, args, ctx)
                run.outputs.append(out)
                yield StepOutputEvent(step_id=step_id, blocks=out.ui, is_error=out.is_error,
                                      duration_s=round(time.monotonic() - t0, 3), code=out.code, language=out.language)
                run.record.steps.append(StepRecord(step_id=step_id, tool=c.name, purpose=args.get("purpose"),
                                                   code=out.code, language=out.language,
                                                   blocks=[b.model_dump() for b in out.ui], is_error=out.is_error))
                results.append(ToolResultPart(tool_call_id=c.id, content=out.summary, is_error=out.is_error))
            parts: list = list(results)
            if run.steps >= max_steps and not limit_noted:
                parts.append(TextPart(text=STEP_LIMIT_NOTE))
                limit_noted = True
            run.messages.append(Message(role="user", parts=parts))
