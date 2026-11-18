import { useQuery } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import type { TableTarget } from "../../App";
import { api } from "../../api/client";
import { PlotlyChart } from "../../components/PlotlyChart";

const SEVERE = new Set(["high_missing", "possible_leakage", "duplicates"]);

export function OverviewView({ sessionId, target }: { sessionId: string; target: TableTarget | null }) {
  const q = useQuery({
    queryKey: ["overview", sessionId, target?.source, target?.table],
    queryFn: () => api.overview(sessionId, target!.source, target!.table),
    enabled: !!target,
    staleTime: 5 * 60_000,
  });
  if (!target) {
    return <p className="p-4 text-sm text-ink-3">Select a table on the left to see its overview: size, data quality and distributions.</p>;
  }
  if (q.isLoading) {
    return <p className="flex items-center gap-2 p-4 text-sm text-ink-3"><Loader2 size={15} className="animate-spin" /> Profiling {target.table}…</p>;
  }
  if (q.error) return <p className="p-4 text-sm text-bad">{q.error.message}</p>;
  const o = q.data!;
  return (
    <div className="space-y-5 p-4">
      <div>
        <h3 className="text-base font-semibold">{target.table}</h3>
        <p className="text-xs text-ink-3">in {target.source}{o.sampled ? ", profiled on the first 200,000 rows" : ""}</p>
      </div>
      <dl className="grid grid-cols-2 gap-x-4 gap-y-3">
        {o.metrics.map((m) => (
          <div key={m.label} className={m.label === "Date range" ? "col-span-2" : undefined}>
            <dt className="text-xs text-ink-3">{m.label}</dt>
            <dd className="tabular font-semibold">{m.value}</dd>
          </div>
        ))}
      </dl>
      {o.warnings.length > 0 && (
        <section>
          <h4 className="mb-1.5 text-sm font-semibold">Data quality</h4>
          <ul className="space-y-1.5 text-sm">
            {o.warnings.map((w, i) => (
              <li key={i} className="flex gap-2">
                <span aria-hidden className={`mt-1.5 size-2 shrink-0 rounded-full ${SEVERE.has(w.kind) ? "bg-bad" : w.kind === "id_like" ? "bg-ink-3" : "bg-warn"}`} />
                <span className="text-ink-2">{w.message}</span>
              </li>
            ))}
          </ul>
        </section>
      )}
      {o.distributions.length > 0 && (
        <section>
          <h4 className="mb-1 text-sm font-semibold">Distributions</h4>
          <div className="space-y-1">
            {o.distributions.map((c, i) => <PlotlyChart key={i} figure={c.figure} height={220} />)}
          </div>
        </section>
      )}
      {o.correlation && (
        <section>
          <h4 className="mb-1 text-sm font-semibold">Correlations</h4>
          <PlotlyChart figure={o.correlation.figure} height={380} />
        </section>
      )}
    </div>
  );
}
