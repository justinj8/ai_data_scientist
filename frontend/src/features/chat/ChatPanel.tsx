import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowUp, Square } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api } from "../../api/client";
import { streamAsk } from "../../api/sse";
import type { UIBlock } from "../../api/types";
import { starterQuestions } from "./labels";
import { QuestionThread } from "./QuestionThread";
import { fromRecord, newQuestion, questionReducer, type QuestionView } from "./reducer";

export function ChatPanel({
  sessionId,
  model,
  onOpenSettings,
}: {
  sessionId: string;
  model: string;
  onOpenSettings: () => void;
  onModelChange?: (m: string) => void;
}) {
  const qc = useQueryClient();
  const [questions, setQuestions] = useState<QuestionView[]>([]);
  const [draft, setDraft] = useState("");
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);
  const schema = useQuery({ queryKey: ["schema", sessionId], queryFn: () => api.schema(sessionId) });
  const running = questions.some((q) => q.status === "running");

  useEffect(() => {
    api.questions(sessionId).then((r) => setQuestions(r.questions.map(fromRecord))).catch(() => {});
  }, [sessionId]);

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [questions]);

  const pin = useMutation({
    mutationFn: (p: { block: UIBlock; title: string; qid?: string }) => api.pin(sessionId, p.title, p.block, p.qid),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["insights", sessionId] }),
  });

  async function ask(text: string) {
    const question = text.trim();
    if (!question || running) return;
    const localId = crypto.randomUUID();
    setQuestions((qs) => [...qs, newQuestion(localId, question)]);
    setDraft("");
    const controller = new AbortController();
    abort.current = controller;
    const update = (fn: (q: QuestionView) => QuestionView) =>
      setQuestions((qs) => qs.map((q) => (q.localId === localId ? fn(q) : q)));
    try {
      await streamAsk(sessionId, { question, model }, (e) => update((q) => questionReducer(q, e)), controller.signal);
      update((q) => (q.status === "running" ? { ...q, status: "error", error: { message: "The connection closed before the answer finished.", action: "retry" } } : q));
    } catch (e) {
      const stopped = controller.signal.aborted;
      update((q) => ({ ...q, status: "error", live: "", error: { message: stopped ? "Stopped." : (e as Error).message, action: "retry" } }));
    } finally {
      abort.current = null;
      void qc.invalidateQueries({ queryKey: ["models"] });
    }
  }

  const sources = schema.data?.sources ?? [];
  const starters = starterQuestions(sources);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-5 sm:px-8">
        <div className="mx-auto flex max-w-[860px] flex-col gap-5">
          {questions.length === 0 && (
            <div className="pt-[8vh]">
              <h2 className="text-2xl font-semibold tracking-tight">Ask a question about your data</h2>
              <p className="mt-1 max-w-[60ch] text-ink-2">
                The assistant plans an analysis, runs SQL, Python, statistics and models, and checks every number before
                it answers. Each step's code is shown so you can verify it.
              </p>
              {sources.length === 0 ? (
                <p className="mt-4 text-sm text-ink-3">Start by uploading a file or loading the sample data on the left.</p>
              ) : (
                <ul className="mt-5 grid gap-2 sm:grid-cols-2">
                  {starters.map((s) => (
                    <li key={s}>
                      <button type="button" onClick={() => ask(s)}
                        className="w-full rounded-lg border border-rule bg-surface px-3 py-2.5 text-left text-sm hover:border-accent">
                        {s}
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
          {questions.map((q) => (
            <QuestionThread key={q.localId} q={q}
              onPin={(block, title, qid) => pin.mutate({ block, title, qid })}
              onRetry={() => ask(q.question)} onOpenSettings={onOpenSettings} />
          ))}
          <div ref={bottom} />
        </div>
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          void ask(draft);
        }}
        className="border-t border-rule bg-surface px-4 py-3 sm:px-8"
      >
        <div className="mx-auto flex max-w-[860px] items-end gap-2">
          <label htmlFor="question" className="sr-only">Question</label>
          <textarea
            id="question"
            rows={2}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void ask(draft);
              }
            }}
            placeholder={sources.length ? "Ask about your data, e.g. what predicts churn?" : "Load data first, then ask a question"}
            className="min-h-[44px] flex-1 resize-none rounded-lg border border-rule bg-paper px-3 py-2 text-[15px] placeholder:text-ink-3 focus:border-accent focus:outline-none"
          />
          {running ? (
            <button type="button" onClick={() => abort.current?.abort()} aria-label="Stop"
              className="flex size-10 items-center justify-center rounded-lg border border-rule hover:bg-sunken">
              <Square size={15} />
            </button>
          ) : (
            <button type="submit" disabled={!draft.trim()} aria-label="Ask"
              className="flex size-10 items-center justify-center rounded-lg bg-accent text-white disabled:opacity-40 dark:text-[#0b1a1d]">
              <ArrowUp size={18} />
            </button>
          )}
        </div>
      </form>
    </div>
  );
}
