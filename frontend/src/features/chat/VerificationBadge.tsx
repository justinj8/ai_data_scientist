import { ShieldAlert, ShieldCheck } from "lucide-react";
import type { QuestionView } from "./reducer";

/** The stamp at the end of the timeline: were the answer's numbers found in the analysis results? */
export function VerificationBadge({ v }: { v: NonNullable<QuestionView["verification"]> }) {
  const ok = v.status === "verified";
  const problems = [...v.ungrounded.map((n) => `${n} doesn't appear in any result.`), ...v.issues];
  return (
    <div className={`flex items-start gap-2.5 rounded-lg border px-3 py-2 text-sm ${ok ? "border-ok/40 bg-ok-soft" : "border-warn/40 bg-warn-soft"}`}>
      {ok ? <ShieldCheck size={18} className="mt-0.5 shrink-0 text-ok" /> : <ShieldAlert size={18} className="mt-0.5 shrink-0 text-warn" />}
      <div>
        <p className={`font-semibold ${ok ? "text-ok" : "text-warn"}`}>{ok ? "Verified" : "Needs review"}</p>
        <p className="text-ink-2">
          {ok
            ? "Every number in this answer was found in the analysis results, and a reviewer model checked the reasoning."
            : "A check found possible problems. Treat these points with care:"}
        </p>
        {!ok && (
          <ul className="mt-1 list-disc pl-4 text-ink-2">
            {problems.map((p) => <li key={p}>{p}</li>)}
          </ul>
        )}
        {v.notes.map((n) => <p key={n} className="mt-1 text-xs text-ink-3">{n}</p>)}
      </div>
    </div>
  );
}
