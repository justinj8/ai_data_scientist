import type { AgentEvent, ModelInfo, QuestionRecord, UIBlock, Usage, UserAction } from "../../api/types";

export type StepItem = {
  kind: "step";
  step_id: string;
  tool: string;
  purpose?: string | null;
  code?: string | null;
  language?: string | null;
  output?: { blocks: UIBlock[]; is_error: boolean; duration_s: number };
};
export type NoteItem = { kind: "note"; text: string };
export type TimelineItem = StepItem | NoteItem;

export type QuestionView = {
  localId: string;
  serverId?: string;
  question: string;
  status: "running" | "done" | "error";
  model?: ModelInfo;
  modelId?: string;
  billedTo?: string | null;
  plan?: string;
  items: TimelineItem[];
  live: string;
  answer?: { text: string; revised: boolean };
  verification?: { status: "verified" | "warnings"; ungrounded: string[]; issues: string[]; notes: string[] };
  usage?: Usage;
  error?: { message: string; action: UserAction };
};

export function newQuestion(localId: string, question: string): QuestionView {
  return { localId, question, status: "running", items: [], live: "" };
}

export function questionReducer(s: QuestionView, e: AgentEvent): QuestionView {
  switch (e.type) {
    case "model":
      return { ...s, serverId: e.question_id, model: e.model, modelId: e.model.id, billedTo: e.billed_to };
    case "text_delta":
      return { ...s, live: s.live + e.text };
    case "plan":
      return { ...s, plan: e.text, live: "" };
    case "note":
      return { ...s, items: [...s.items, { kind: "note", text: e.text }], live: "" };
    case "step_start":
      return {
        ...s,
        live: "",
        items: [...s.items, { kind: "step", step_id: e.step_id, tool: e.tool, purpose: e.purpose, code: e.code, language: e.language }],
      };
    case "step_output":
      return {
        ...s,
        items: s.items.map((it) =>
          it.kind === "step" && it.step_id === e.step_id
            ? { ...it, code: e.code ?? it.code, language: e.language ?? it.language,
                output: { blocks: e.blocks, is_error: e.is_error, duration_s: e.duration_s } }
            : it,
        ),
      };
    case "answer":
      return { ...s, answer: { text: e.text, revised: e.revised }, live: "", verification: undefined };
    case "verification":
      return { ...s, verification: { status: e.status, ungrounded: e.ungrounded, issues: e.issues, notes: e.notes } };
    case "usage":
      return { ...s, usage: e.usage };
    case "done":
      return { ...s, status: "done", live: "" };
    case "error":
      return { ...s, status: "error", live: "", error: { message: e.message, action: e.user_action } };
  }
}

export function fromRecord(r: QuestionRecord): QuestionView {
  return {
    localId: r.id,
    serverId: r.id,
    question: r.question,
    status: "done",
    modelId: r.model_id,
    plan: r.plan ?? undefined,
    items: r.steps.map((st) => ({
      kind: "step" as const,
      step_id: st.step_id,
      tool: st.tool,
      purpose: st.purpose,
      code: st.code,
      language: st.language,
      output: { blocks: st.blocks, is_error: st.is_error, duration_s: 0 },
    })),
    live: "",
    answer: { text: r.answer, revised: false },
    verification: r.verification ?? undefined,
  };
}
