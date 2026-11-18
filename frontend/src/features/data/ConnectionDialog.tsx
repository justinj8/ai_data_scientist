import { useMutation, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Loader2, XCircle } from "lucide-react";
import { useMemo, useState } from "react";
import { api } from "../../api/client";
import type { Engine } from "../../api/types";
import { Dialog, inputClass, primaryButton, secondaryButton } from "../../components/Dialog";
import { buildConfig, ENGINES, suggestName } from "./connectionForm";

export function ConnectionDialog({
  open,
  onClose,
  sessionId,
  taken,
}: {
  open: boolean;
  onClose: () => void;
  sessionId: string;
  taken: string[];
}) {
  const qc = useQueryClient();
  const [engine, setEngine] = useState<Engine>("postgresql");
  const [values, setValues] = useState<Record<string, string>>({});
  const [useUrl, setUseUrl] = useState(false);
  const [name, setName] = useState("");
  const [test, setTest] = useState<{ ok: boolean; error: string | null } | null>(null);
  const spec = ENGINES.find((e) => e.engine === engine)!;
  const config = useMemo(() => buildConfig(engine, values, useUrl), [engine, values, useUrl]);
  const finalName = name || suggestName(engine, values, taken);

  const testMut = useMutation({ mutationFn: () => api.testConnection(config), onSuccess: setTest });
  const connect = useMutation({
    mutationFn: () => api.addConnection(sessionId, finalName, config),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["schema", sessionId] });
      setValues({});
      setName("");
      setTest(null);
      onClose();
    },
  });

  const set = (k: string, v: string) => {
    setValues((s) => ({ ...s, [k]: v }));
    setTest(null);
  };

  return (
    <Dialog open={open} onClose={onClose} title="Connect a database">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          connect.mutate();
        }}
        className="space-y-4"
      >
        <fieldset>
          <legend className="mb-2 text-sm font-medium">Database</legend>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            {ENGINES.map((e) => (
              <button
                key={e.engine}
                type="button"
                onClick={() => {
                  setEngine(e.engine);
                  setTest(null);
                }}
                aria-pressed={engine === e.engine}
                className={`rounded-md border px-3 py-2 text-left text-sm ${engine === e.engine ? "border-accent bg-accent-soft font-medium" : "border-rule hover:bg-sunken"}`}
              >
                {e.label}
              </button>
            ))}
          </div>
        </fieldset>

        <label className="flex items-center gap-2 text-sm text-ink-2">
          <input type="checkbox" checked={useUrl} onChange={(e) => setUseUrl(e.target.checked)} />
          Use a connection URL instead
        </label>

        {useUrl ? (
          <label className="block text-sm">
            <span className="mb-1 block">Connection URL</span>
            <input className={inputClass} type="password" autoComplete="off" placeholder={spec.urlExample}
              value={values.url ?? ""} onChange={(e) => set("url", e.target.value)} />
          </label>
        ) : (
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            {spec.fields.map((f) => (
              <label key={f.key} className={`block text-sm ${f.type === "file" || f.key === "database" && engine === "sqlite" ? "sm:col-span-2" : ""}`}>
                {f.type === "checkbox" ? (
                  <span className="flex items-center gap-2 pt-6">
                    <input type="checkbox" checked={values[f.key] === "true"} onChange={(e) => set(f.key, String(e.target.checked))} />
                    {f.label}
                  </span>
                ) : (
                  <>
                    <span className="mb-1 block">{f.label}</span>
                    {f.type === "file" ? (
                      <input type="file" accept="application/json,.json" className="text-sm"
                        onChange={async (e) => {
                          const file = e.target.files?.[0];
                          if (file) set(f.key, await file.text());
                        }} />
                    ) : (
                      <input
                        className={inputClass}
                        type={f.type === "password" ? "password" : f.type === "number" ? "number" : "text"}
                        autoComplete="off"
                        placeholder={f.placeholder ?? (f.key === "port" && spec.defaultPort ? String(spec.defaultPort) : undefined)}
                        value={values[f.key] ?? ""}
                        onChange={(e) => set(f.key, e.target.value)}
                      />
                    )}
                    {f.hint && <span className="mt-1 block text-xs text-ink-3">{f.hint}</span>}
                  </>
                )}
              </label>
            ))}
          </div>
        )}

        <label className="block text-sm">
          <span className="mb-1 block">Name in this session</span>
          <input className={inputClass} value={name} placeholder={finalName}
            onChange={(e) => setName(e.target.value.toLowerCase().replace(/[^a-z0-9_]/g, "_"))} />
          <span className="mt-1 block text-xs text-ink-3">The assistant refers to this source by name. Credentials stay in memory on your machine.</span>
        </label>

        {test && (
          <p className={`flex items-start gap-1.5 text-sm ${test.ok ? "text-ok" : "text-bad"}`} role="status">
            {test.ok ? <CheckCircle2 size={16} className="mt-0.5 shrink-0" /> : <XCircle size={16} className="mt-0.5 shrink-0" />}
            {test.ok ? "Connection works." : test.error}
          </p>
        )}
        {connect.error && <p className="text-sm text-bad" role="alert">{connect.error.message}</p>}

        <div className="flex justify-end gap-2 border-t border-rule pt-4">
          <button type="button" className={secondaryButton} onClick={() => testMut.mutate()} disabled={testMut.isPending}>
            {testMut.isPending && <Loader2 size={14} className="animate-spin" />} Test connection
          </button>
          <button type="submit" className={primaryButton} disabled={connect.isPending}>
            {connect.isPending && <Loader2 size={14} className="animate-spin" />} Connect
          </button>
        </div>
      </form>
    </Dialog>
  );
}
