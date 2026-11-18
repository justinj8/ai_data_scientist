import { describe, expect, it } from "vitest";
import type { AgentEvent, QuestionRecord } from "../../api/types";
import { fromRecord, newQuestion, questionReducer } from "./reducer";

const model = { id: "claude-opus-5-5", provider: "anthropic" as const, family: "opus", display_name: "Claude Opus 5.5", supports_tools: true };

describe("questionReducer", () => {
  it("builds the timeline from a full event sequence with a revision", () => {
    const events: AgentEvent[] = [
      { type: "model", question_id: "q1", model, billed_to: "api_key" },
      { type: "text_delta", text: "Plan: " },
      { type: "text_delta", text: "count" },
      { type: "plan", text: "Plan: count" },
      { type: "step_start", step_id: "s1", tool: "run_sql", purpose: "count", code: "select 1", language: "sql", args: {} },
      { type: "step_output", step_id: "s1", blocks: [{ type: "text", text: "1", format: "plain" }], is_error: false, duration_s: 0.2, code: "select 1", language: "sql" },
      { type: "note", text: "Now testing." },
      { type: "answer", text: "There are 9 rows.", revised: false },
      { type: "text_delta", text: "There are" },
      { type: "answer", text: "There is 1 row.", revised: true },
      { type: "verification", status: "verified", ungrounded: [], issues: [], notes: [] },
      { type: "usage", usage: { input_tokens: 10, output_tokens: 5, cost_usd: 0.01, billed_to: "api_key" }, model_id: "claude-opus-5-5" },
      { type: "done", question_id: "q1", answer: "There is 1 row.", steps: 1 },
    ];
    const s = events.reduce(questionReducer, newQuestion("local-1", "How many rows?"));
    expect(s.status).toBe("done");
    expect(s.serverId).toBe("q1");
    expect(s.plan).toBe("Plan: count");
    expect(s.live).toBe("");
    expect(s.items.map((i) => i.kind)).toEqual(["step", "note"]);
    const step = s.items[0];
    expect(step.kind === "step" && step.output?.duration_s).toBe(0.2);
    expect(s.answer).toEqual({ text: "There is 1 row.", revised: true });
    expect(s.verification?.status).toBe("verified");
    expect(s.usage?.cost_usd).toBe(0.01);
  });

  it("records errors with their suggested action", () => {
    const s = questionReducer(newQuestion("l", "q"), { type: "error", message: "rate limit", user_action: "switch_model" });
    expect(s.status).toBe("error");
    expect(s.error).toEqual({ message: "rate limit", action: "switch_model" });
  });

  it("converts a stored question record", () => {
    const rec: QuestionRecord = {
      id: "q9", question: "Q?", model_id: "gpt-6.2-astra", plan: "Plan", answer: "A",
      steps: [{ step_id: "s", tool: "run_sql", code: "select 1", language: "sql", blocks: [], is_error: false }],
      verification: { status: "warnings", ungrounded: ["31%"], issues: [], notes: [] },
    };
    const s = fromRecord(rec);
    expect(s.status).toBe("done");
    expect(s.items).toHaveLength(1);
    expect(s.verification?.ungrounded).toEqual(["31%"]);
  });
});
