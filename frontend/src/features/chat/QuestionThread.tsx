import { Pin, RotateCcw, Settings } from "lucide-react";
import type { UIBlock } from "../../api/types";
import { Markdown } from "../../components/Markdown";
import type { QuestionView } from "./reducer";
import { StepCard } from "./StepCard";
import { VerificationBadge } from "./VerificationBadge";

function usageText(q: QuestionView): string | null {
  if (!q.usage) return null;
  const model = q.modelId ?? "";
  if (q.usage.billed_to === "chatgpt_plan") return `${model}, using your ChatGPT plan`;
  if (q.usage.billed_to === "supergrok_plan") return `${model}, using your SuperGrok plan`;
  const tokens = (q.usage.input_tokens + q.usage.output_tokens).toLocaleString();
  const cost = q.usage.cost_usd != null ? `$${q.usage.cost_usd.toFixed(q.usage.cost_usd < 0.1 ? 3 : 2)}` : "cost unknown";
  return `${model}, ${tokens} tokens, ${cost}`;
}

export function QuestionThread({
  q,
  onPin,
  onRetry,
  onOpenSettings,
}: {
  q: QuestionView;
  onPin: (block: UIBlock, title: string, questionId?: string) => void;
  onRetry: () => void;
  onOpenSettings: () => void;
}) {
  let stepNo = 0;
  const running = q.status === "running";
  return (
    <article className="rounded-xl border border-rule bg-surface/95 px-5 py-4 shadow-[0_1px_0_var(--rule)]">
      <header className="mb-4 flex flex-wrap items-baseline gap-x-3">
        <h3 className="text-[17px] font-semibold leading-snug">{q.question}</h3>
        {q.modelId && <span className="text-xs text-ink-3">{q.modelId}</span>}
      </header>

      <ol className="relative border-l-2 border-rule/70 [&>li]:-ml-[13px]">
        {q.plan && (
          <li className="relative pb-5 pl-9">
            <span aria-hidden className="absolute left-0 top-0.5 size-6 rounded-full border-2 border-rule bg-sunken" />
            <h4 className="text-sm font-semibold">Plan</h4>
            <div className="text-sm text-ink-2"><Markdown text={q.plan.replace(/^plan:\s*/i, "")} /></div>
          </li>
        )}
        {q.items.map((it, i) =>
          it.kind === "step" ? (
            <StepCard key={it.step_id} step={it} index={++stepNo} onPin={(b, t) => onPin(b, t, q.serverId)} />
          ) : (
            <li key={`n${i}`} className="relative pb-5 pl-9 text-sm text-ink-2">
              <span aria-hidden className="absolute left-[7px] top-1.5 size-2.5 rounded-full bg-rule" />
              <Markdown text={it.text} />
            </li>
          ),
        )}
        {running && (
          <li className="relative pb-5 pl-9 text-sm text-ink-3" aria-live="polite">
            <span aria-hidden className="absolute left-[5px] top-1 size-3.5 animate-pulse rounded-full bg-accent/60" />
            {q.live ? <span className="whitespace-pre-wrap">{q.live.slice(-600)}</span> : "Thinking…"}
          </li>
        )}
      </ol>

      {q.error && (
        <div role="alert" className="mt-2 rounded-lg border border-bad/30 bg-bad-soft px-3 py-2 text-sm text-bad">
          <p>{q.error.message}</p>
          <div className="mt-2 flex gap-3">
            {(q.error.action === "update_credentials" || q.error.action === "switch_model") && (
              <button type="button" onClick={onOpenSettings} className="inline-flex items-center gap-1 font-medium">
                <Settings size={14} /> {q.error.action === "switch_model" ? "Choose another model" : "Open settings"}
              </button>
            )}
            {(q.error.action === "retry" || q.error.action === "switch_model") && (
              <button type="button" onClick={onRetry} className="inline-flex items-center gap-1 font-medium">
                <RotateCcw size={14} /> Try again
              </button>
            )}
          </div>
        </div>
      )}

      {q.answer && (
        <section className="mt-1 space-y-3" aria-label="Answer">
          {q.verification && <VerificationBadge v={q.verification} />}
          {q.answer.revised && <p className="text-xs text-ink-3">Revised after verification.</p>}
          <Markdown text={q.answer.text} />
          <div className="flex flex-wrap items-center gap-3 border-t border-rule pt-2 text-xs text-ink-3">
            <button type="button" className="inline-flex items-center gap-1 hover:text-accent"
              onClick={() => onPin({ type: "text", text: q.answer!.text, format: "markdown" }, q.question, q.serverId)}>
              <Pin size={12} /> Pin answer
            </button>
            {usageText(q) && <span>{usageText(q)}</span>}
          </div>
        </section>
      )}
    </article>
  );
}
