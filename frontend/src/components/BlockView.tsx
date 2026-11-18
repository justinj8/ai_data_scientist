import { Pin } from "lucide-react";
import type { UIBlock } from "../api/types";
import { DataTable } from "./DataTable";
import { Markdown } from "./Markdown";
import { PlotlyChart } from "./PlotlyChart";

type Props = { block: UIBlock; onPin?: (block: UIBlock) => void; compact?: boolean };

/** Renders one tool output block (table, chart, metrics, text, image or error). */
export function BlockView({ block, onPin, compact }: Props) {
  const pin = onPin && (block.type === "table" || block.type === "chart" || block.type === "metrics") && (
    <button
      type="button"
      onClick={() => onPin(block)}
      className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-xs text-ink-3 hover:bg-sunken hover:text-accent"
    >
      <Pin size={12} /> Pin to board
    </button>
  );
  switch (block.type) {
    case "table":
      return (
        <div>
          <DataTable block={block} maxHeight={compact ? 220 : 320} />
          {pin}
        </div>
      );
    case "chart":
      return (
        <div className="rounded-md border border-rule bg-surface p-1">
          <PlotlyChart figure={block.figure} height={compact ? 240 : 320} />
          {pin && <div className="px-2 pb-1">{pin}</div>}
        </div>
      );
    case "metrics":
      return (
        <div>
          <dl className="grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-4">
            {block.items.map((m) => (
              <div key={m.label}>
                <dt className="text-xs text-ink-3">{m.label}</dt>
                <dd className="tabular text-[15px] font-semibold">{m.value}</dd>
                {m.detail && <dd className="text-xs text-ink-3">{m.detail}</dd>}
              </div>
            ))}
          </dl>
          {pin}
        </div>
      );
    case "text":
      return block.format === "markdown" ? (
        <div className="text-sm">
          <Markdown text={block.text} />
        </div>
      ) : (
        <pre className="max-h-64 overflow-auto whitespace-pre-wrap rounded-md bg-sunken px-3 py-2 font-mono text-[12.5px]">
          {block.text}
        </pre>
      );
    case "image":
      return <img src={`data:image/png;base64,${block.png_b64}`} alt="Chart from Python code" className="max-w-full" />;
    case "error":
      return (
        <div className="rounded-md border border-bad/30 bg-bad-soft px-3 py-2 text-sm text-bad">
          <p className="font-medium">{block.message}</p>
          {block.detail && (
            <details className="mt-1">
              <summary className="cursor-pointer text-xs">Traceback</summary>
              <pre className="mt-1 overflow-x-auto whitespace-pre-wrap font-mono text-xs">{block.detail}</pre>
            </details>
          )}
        </div>
      );
  }
}
