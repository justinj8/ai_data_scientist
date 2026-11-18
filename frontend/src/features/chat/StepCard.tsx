import { AlertTriangle, ChevronRight } from "lucide-react";
import { useState } from "react";
import type { UIBlock } from "../../api/types";
import { BlockView } from "../../components/BlockView";
import { CodeBlock } from "../../components/CodeBlock";
import { TOOL_LABELS } from "./labels";
import type { StepItem } from "./reducer";

export function StepCard({ step, index, onPin }: { step: StepItem; index: number; onPin: (b: UIBlock, title: string) => void }) {
  const running = !step.output;
  const failed = step.output?.is_error;
  // SQL is short and the point of the step, so it is shown; longer generated code starts collapsed.
  const [showCode, setShowCode] = useState(step.language === "sql");
  const title = step.purpose || TOOL_LABELS[step.tool] || step.tool;
  return (
    <li className="relative pb-6 pl-9">
      <span
        aria-hidden
        className={`absolute left-0 top-0 flex size-6 items-center justify-center rounded-full border-2 text-xs font-semibold tabular ${
          failed ? "border-bad bg-bad-soft text-bad" : running ? "border-accent bg-surface text-accent" : "border-accent bg-accent text-white dark:text-[#0b1a1d]"
        }`}
      >
        {failed ? <AlertTriangle size={12} /> : index}
      </span>
      <div className="flex flex-wrap items-baseline gap-x-2">
        <h4 className="text-sm font-semibold">{title}</h4>
        <span className="text-xs text-ink-3">{TOOL_LABELS[step.tool] ?? step.tool}</span>
        {running && <span className="text-xs text-accent">running…</span>}
        {step.output && step.output.duration_s > 0 && <span className="tabular text-xs text-ink-3">{step.output.duration_s.toFixed(1)}s</span>}
      </div>
      {step.code && (
        <div className="mt-1.5">
          <button type="button" onClick={() => setShowCode((v) => !v)} aria-expanded={showCode}
            className="inline-flex items-center gap-1 text-xs text-ink-3 hover:text-ink">
            <ChevronRight size={12} className={showCode ? "rotate-90" : ""} />
            {showCode ? "Hide" : "Show"} {step.language === "sql" ? "SQL" : "code"}
          </button>
          {showCode && <div className="mt-1"><CodeBlock code={step.code} language={step.language ?? "python"} /></div>}
        </div>
      )}
      {step.output && (
        <div className="mt-2 space-y-3">
          {step.output.blocks.map((b, i) => (
            <BlockView key={i} block={b} compact onPin={(blk) => onPin(blk, title)} />
          ))}
        </div>
      )}
    </li>
  );
}
