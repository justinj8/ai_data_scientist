import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Calendar,
  ChevronRight,
  Database,
  FileSpreadsheet,
  Hash,
  Loader2,
  Plus,
  ToggleLeft,
  Trash2,
  Type,
  Upload,
} from "lucide-react";
import { useRef, useState } from "react";
import type { TableTarget } from "../../App";
import { api } from "../../api/client";
import type { LoadReport, SourceInfo } from "../../api/types";
import { secondaryButton } from "../../components/Dialog";
import { ConnectionDialog } from "./ConnectionDialog";

function typeIcon(dtype: string) {
  const t = dtype.toLowerCase();
  if (/int|float|double|decimal|numeric|real|number|bigint/.test(t)) return <Hash size={12} aria-label="number" />;
  if (/date|time/.test(t)) return <Calendar size={12} aria-label="date" />;
  if (/bool/.test(t)) return <ToggleLeft size={12} aria-label="boolean" />;
  return <Type size={12} aria-label="text" />;
}

export function DataPanel({
  sessionId,
  selected,
  onSelect,
}: {
  sessionId: string;
  selected: TableTarget | null;
  onSelect: (t: TableTarget) => void;
}) {
  const qc = useQueryClient();
  const fileInput = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [connectOpen, setConnectOpen] = useState(false);
  const [reports, setReports] = useState<LoadReport[]>([]);
  const schema = useQuery({ queryKey: ["schema", sessionId], queryFn: () => api.schema(sessionId) });
  const refresh = () => qc.invalidateQueries({ queryKey: ["schema", sessionId] });

  const upload = useMutation({
    mutationFn: async (files: File[]) => {
      const out: LoadReport[] = [];
      for (const f of files) out.push(...(await api.upload(sessionId, f)).tables);
      return out;
    },
    onSuccess: (r) => {
      setReports(r);
      void refresh();
      if (r[0]) onSelect({ source: "files", table: r[0].table });
    },
  });
  const sample = useMutation({
    mutationFn: () => api.loadSample(sessionId),
    onSuccess: () => {
      void refresh();
      onSelect({ source: "files", table: "orders" });
    },
  });
  const remove = useMutation({ mutationFn: (name: string) => api.removeSource(sessionId, name), onSuccess: refresh });

  const sources: SourceInfo[] = schema.data?.sources ?? [];
  const warnings = reports.flatMap((r) => r.warnings.map((w) => `${r.table}: ${w}`));

  return (
    <div className="flex flex-col gap-4 p-4">
      <div>
        <h2 className="text-sm font-semibold">Data</h2>
        <p className="text-xs text-ink-3">Upload CSV or Excel files, or connect live databases.</p>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          upload.mutate(Array.from(e.dataTransfer.files));
        }}
        className={`rounded-lg border border-dashed px-3 py-4 text-center ${dragging ? "border-accent bg-accent-soft" : "border-rule"}`}
      >
        <input ref={fileInput} type="file" accept=".csv,.xlsx" multiple hidden
          onChange={(e) => e.target.files && upload.mutate(Array.from(e.target.files))} />
        <button type="button" onClick={() => fileInput.current?.click()} className="inline-flex items-center gap-1.5 text-sm font-medium text-accent">
          {upload.isPending ? <Loader2 size={15} className="animate-spin" /> : <Upload size={15} />}
          {upload.isPending ? "Loading…" : "Upload CSV or Excel"}
        </button>
        <p className="mt-0.5 text-xs text-ink-3">or drop files here. Each Excel sheet becomes a table.</p>
      </div>
      {upload.error && <p className="text-xs text-bad" role="alert">{upload.error.message}</p>}
      {warnings.length > 0 && (
        <ul className="space-y-0.5 text-xs text-warn">
          {warnings.map((w) => <li key={w}>{w}</li>)}
        </ul>
      )}

      <button type="button" className={secondaryButton} onClick={() => setConnectOpen(true)}>
        <Plus size={15} /> Connect a database
      </button>

      {sources.length === 0 && !schema.isLoading && (
        <div className="rounded-lg bg-sunken p-3 text-sm">
          <p className="text-ink-2">No data yet. Try the sample retail data: three years of orders, customers with churn, and a live SQLite database.</p>
          <button type="button" onClick={() => sample.mutate()} disabled={sample.isPending}
            className="mt-2 inline-flex items-center gap-1.5 font-medium text-accent">
            {sample.isPending && <Loader2 size={14} className="animate-spin" />} Load sample data
          </button>
        </div>
      )}

      <div className="space-y-3">
        {sources.map((src) => (
          <div key={src.name}>
            <div className="flex items-center gap-1.5">
              {src.kind === "file" ? <FileSpreadsheet size={15} className="text-ink-3" /> : <Database size={15} className="text-ink-3" />}
              <span className="text-sm font-medium">{src.name}</span>
              <span className="text-xs text-ink-3">{src.dialect}</span>
              <button type="button" onClick={() => remove.mutate(src.name)} className="ml-auto rounded p-1 text-ink-3 hover:text-bad"
                aria-label={`Remove ${src.name}`}>
                <Trash2 size={13} />
              </button>
            </div>
            <ul className="mt-1 space-y-0.5 border-l border-rule pl-2">
              {src.tables.map((t) => {
                const active = selected?.source === src.name && selected.table === t.name;
                return (
                  <li key={t.name}>
                    <details>
                      <summary className={`flex cursor-pointer list-none items-center gap-1 rounded px-1.5 py-1 text-sm ${active ? "bg-accent-soft" : "hover:bg-sunken"}`}>
                        <ChevronRight size={13} className="shrink-0 text-ink-3 transition-transform [details[open]_&]:rotate-90" />
                        <button type="button" className="truncate text-left" title={t.original_name ?? t.name}
                          onClick={(e) => {
                            e.preventDefault();
                            onSelect({ source: src.name, table: t.name });
                          }}>
                          {t.name}
                        </button>
                        {t.row_count != null && <span className="tabular ml-auto text-xs text-ink-3">{t.row_count.toLocaleString()}</span>}
                      </summary>
                      <ul className="mb-1 ml-5 space-y-0.5">
                        {t.columns.map((c) => (
                          <li key={c.name} className="flex items-center gap-1.5 text-xs text-ink-2" title={c.original_name ? `Original: ${c.original_name}` : c.dtype}>
                            <span className="text-ink-3">{typeIcon(c.dtype)}</span>
                            <span className="truncate">{c.name}</span>
                            <span className="ml-auto shrink-0 text-ink-3">{c.dtype.toLowerCase()}</span>
                          </li>
                        ))}
                      </ul>
                    </details>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </div>

      <ConnectionDialog open={connectOpen} onClose={() => setConnectOpen(false)} sessionId={sessionId}
        taken={sources.map((s) => s.name)} />
    </div>
  );
}
