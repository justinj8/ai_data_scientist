import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FileDown, NotebookText } from "lucide-react";
import { useState } from "react";
import type { TableTarget } from "../../App";
import { api } from "../../api/client";
import { InsightCard } from "./InsightCard";
import { OverviewView } from "./OverviewView";

export function DashboardPanel({ sessionId, target }: { sessionId: string; target: TableTarget | null }) {
  const qc = useQueryClient();
  const [tab, setTab] = useState<"overview" | "board">("overview");
  const insights = useQuery({ queryKey: ["insights", sessionId], queryFn: () => api.insights(sessionId) });
  const refresh = () => qc.invalidateQueries({ queryKey: ["insights", sessionId] });
  const update = useMutation({
    mutationFn: (p: { id: string; patch: { title?: string; note?: string; order?: number } }) =>
      api.updateInsight(sessionId, p.id, p.patch),
    onSuccess: refresh,
  });
  const remove = useMutation({ mutationFn: (id: string) => api.unpin(sessionId, id), onSuccess: refresh });
  const items = insights.data?.insights ?? [];

  const tabClass = (t: string) =>
    `px-3 py-2.5 text-sm ${tab === t ? "border-b-2 border-accent font-medium text-ink" : "text-ink-3 hover:text-ink"}`;

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center border-b border-rule px-2" role="tablist">
        <button type="button" role="tab" aria-selected={tab === "overview"} className={tabClass("overview")} onClick={() => setTab("overview")}>
          Overview
        </button>
        <button type="button" role="tab" aria-selected={tab === "board"} className={tabClass("board")} onClick={() => setTab("board")}>
          Board{items.length ? ` (${items.length})` : ""}
        </button>
        <div className="ml-auto flex items-center gap-1 pr-1">
          <a href={api.notebookUrl(sessionId)} download className="inline-flex items-center gap-1 rounded px-2 py-1 text-xs text-ink-2 hover:bg-sunken"
            title="Download every step as a runnable Jupyter notebook">
            <NotebookText size={14} /> Notebook
          </a>
          <a href={api.reportUrl(sessionId)} download className="inline-flex items-center gap-1 rounded px-2 py-1 text-xs text-ink-2 hover:bg-sunken"
            title="Download questions, answers and pinned insights as Markdown">
            <FileDown size={14} /> Report
          </a>
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto">
        {tab === "overview" ? (
          <OverviewView sessionId={sessionId} target={target} />
        ) : items.length === 0 ? (
          <p className="p-4 text-sm text-ink-3">
            Pin charts, tables and answers from the analysis to build a board you can export with the report.
          </p>
        ) : (
          <div className="space-y-3 p-3">
            {items.map((ins, i) => (
              <InsightCard key={ins.id} insight={ins} isFirst={i === 0} isLast={i === items.length - 1}
                onUpdate={(patch) => update.mutate({ id: ins.id, patch })}
                onMove={(d) => update.mutate({ id: ins.id, patch: { order: i + d } })}
                onRemove={() => remove.mutate(ins.id)} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
