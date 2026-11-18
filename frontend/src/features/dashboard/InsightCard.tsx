import { ArrowDown, ArrowUp, Download, Trash2 } from "lucide-react";
import { useRef, useState } from "react";
import type { Insight } from "../../api/types";
import { BlockView } from "../../components/BlockView";
import { downloadChartPng } from "../../components/PlotlyChart";

export function InsightCard({
  insight,
  isFirst,
  isLast,
  onUpdate,
  onMove,
  onRemove,
}: {
  insight: Insight;
  isFirst: boolean;
  isLast: boolean;
  onUpdate: (patch: { title?: string; note?: string }) => void;
  onMove: (delta: -1 | 1) => void;
  onRemove: () => void;
}) {
  const [title, setTitle] = useState(insight.title);
  const [note, setNote] = useState(insight.note ?? "");
  const body = useRef<HTMLDivElement>(null);
  const iconBtn = "rounded p-1 text-ink-3 hover:bg-sunken hover:text-ink disabled:opacity-30";
  return (
    <article className="rounded-lg border border-rule bg-surface p-3">
      <div className="flex items-start gap-1">
        <label className="sr-only" htmlFor={`title-${insight.id}`}>Title</label>
        <input id={`title-${insight.id}`} value={title} onChange={(e) => setTitle(e.target.value)}
          onBlur={() => title.trim() && title !== insight.title && onUpdate({ title: title.trim() })}
          className="min-w-0 flex-1 rounded bg-transparent px-1 py-0.5 text-sm font-semibold hover:bg-sunken focus:bg-sunken focus:outline-none" />
        {insight.block.type === "chart" && (
          <button type="button" className={iconBtn} aria-label="Download chart as PNG"
            onClick={() => body.current && downloadChartPng(body.current, title)}>
            <Download size={14} />
          </button>
        )}
        <button type="button" className={iconBtn} disabled={isFirst} onClick={() => onMove(-1)} aria-label="Move up"><ArrowUp size={14} /></button>
        <button type="button" className={iconBtn} disabled={isLast} onClick={() => onMove(1)} aria-label="Move down"><ArrowDown size={14} /></button>
        <button type="button" className={`${iconBtn} hover:text-bad`} onClick={onRemove} aria-label="Remove from board"><Trash2 size={14} /></button>
      </div>
      <div ref={body} className="mt-2">
        <BlockView block={insight.block} compact />
      </div>
      <label className="sr-only" htmlFor={`note-${insight.id}`}>Note</label>
      <textarea id={`note-${insight.id}`} value={note} onChange={(e) => setNote(e.target.value)} rows={1}
        onBlur={() => note !== (insight.note ?? "") && onUpdate({ note })}
        placeholder="Add a note"
        className="mt-2 w-full resize-y rounded bg-transparent px-1 py-0.5 text-sm text-ink-2 placeholder:text-ink-3 hover:bg-sunken focus:bg-sunken focus:outline-none" />
    </article>
  );
}
