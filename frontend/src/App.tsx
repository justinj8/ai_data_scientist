import { Moon, Settings, Sun } from "lucide-react";
import { useState } from "react";
import { ChatPanel } from "./features/chat/ChatPanel";
import { ModelPicker } from "./features/chat/ModelPicker";
import { DashboardPanel } from "./features/dashboard/DashboardPanel";
import { DataPanel } from "./features/data/DataPanel";
import { SettingsDialog } from "./features/settings/SettingsDialog";
import { useSession } from "./hooks/useSession";
import { useTheme } from "./hooks/useTheme";

export type TableTarget = { source: string; table: string };
type MobileTab = "data" | "analysis" | "dashboard";
const MODEL_KEY = "adsci-model";

function savedModel(): string {
  try {
    return localStorage.getItem(MODEL_KEY) ?? "opus";
  } catch {
    return "opus";
  }
}

export default function App() {
  const { sessionId, error } = useSession();
  const { theme, toggle } = useTheme();
  const [model, setModel] = useState(savedModel);
  const [target, setTarget] = useState<TableTarget | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [tab, setTab] = useState<MobileTab>("analysis");

  const chooseModel = (m: string) => {
    setModel(m);
    try {
      localStorage.setItem(MODEL_KEY, m);
    } catch {
      /* storage unavailable */
    }
  };

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center gap-3 border-b border-rule bg-surface px-4 py-2.5">
        <img src="/favicon.svg" alt="" className="size-6" />
        <h1 className="text-[15px] font-semibold tracking-tight">AI Data Scientist</h1>
        <div className="ml-auto flex items-center gap-2">
          <ModelPicker value={model} onChange={chooseModel} onOpenSettings={() => setSettingsOpen(true)} />
          <button
            type="button"
            onClick={() => setSettingsOpen(true)}
            className="inline-flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-sm text-ink-2 hover:bg-sunken"
          >
            <Settings size={16} /> <span className="hidden sm:inline">Settings</span>
          </button>
          <button
            type="button"
            onClick={toggle}
            className="rounded-md p-1.5 text-ink-2 hover:bg-sunken"
            aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
          >
            {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
          </button>
        </div>
      </header>

      <nav className="flex border-b border-rule bg-surface lg:hidden" aria-label="Panels">
        {(["data", "analysis", "dashboard"] as const).map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => setTab(t)}
            aria-current={tab === t}
            className={`flex-1 py-2 text-sm capitalize ${tab === t ? "border-b-2 border-accent font-medium text-ink" : "text-ink-3"}`}
          >
            {t}
          </button>
        ))}
      </nav>

      {error && (
        <div role="alert" className="bg-bad-soft px-4 py-2 text-sm text-bad">
          Can't reach the backend: {error} Start it with <code className="font-mono">uv run adsci</code> and reload.
        </div>
      )}

      {sessionId && (
        <main className="grid min-h-0 flex-1 lg:grid-cols-[280px_minmax(0,1fr)_440px]">
          <section aria-label="Data" className={`min-h-0 overflow-y-auto border-r border-rule bg-surface ${tab === "data" ? "" : "hidden lg:block"}`}>
            <DataPanel sessionId={sessionId} selected={target} onSelect={(t) => { setTarget(t); setTab("dashboard"); }} />
          </section>
          <section aria-label="Analysis" className={`graph-paper flex min-h-0 flex-col ${tab === "analysis" ? "" : "hidden lg:flex"}`}>
            <ChatPanel sessionId={sessionId} model={model} onOpenSettings={() => setSettingsOpen(true)} onModelChange={chooseModel} />
          </section>
          <section aria-label="Dashboard" className={`min-h-0 overflow-y-auto border-l border-rule bg-surface ${tab === "dashboard" ? "" : "hidden lg:block"}`}>
            <DashboardPanel sessionId={sessionId} target={target} />
          </section>
        </main>
      )}

      <SettingsDialog open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </div>
  );
}
