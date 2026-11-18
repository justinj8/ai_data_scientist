import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ExternalLink, Loader2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api } from "../../api/client";
import type { Provider, ProviderStatus } from "../../api/types";
import { inputClass, primaryButton, secondaryButton } from "../../components/Dialog";

const NAMES: Record<Provider, { title: string; models: string; keyUrl: string; keyHint: string }> = {
  anthropic: { title: "Anthropic", models: "Claude Opus and Sonnet", keyUrl: "https://console.anthropic.com/settings/keys", keyHint: "sk-ant-…" },
  openai: { title: "OpenAI", models: "GPT models", keyUrl: "https://platform.openai.com/api-keys", keyHint: "sk-…" },
  xai: { title: "xAI", models: "Grok models", keyUrl: "https://console.x.ai", keyHint: "xai-…" },
};

function statusLine(s: ProviderStatus): string {
  if (!s.configured) return "Not connected";
  if (s.kind === "chatgpt_plan") return `Using your ChatGPT plan${s.account_label ? ` (${s.account_label})` : ""}`;
  if (s.kind === "supergrok_plan") return "Using your SuperGrok plan";
  if (s.from_env) return `Using the API key from the environment (${s.masked})`;
  return `Connected with API key ${s.masked}`;
}

export function ProviderCard({ provider, status }: { provider: Provider; status: ProviderStatus }) {
  const qc = useQueryClient();
  const [key, setKey] = useState("");
  const [device, setDevice] = useState<{ uri: string; code: string } | null>(null);
  const [signInError, setSignInError] = useState<string | null>(null);
  const [waiting, setWaiting] = useState(false);
  const poll = useRef<number | null>(null);
  const info = NAMES[provider];

  const refresh = () =>
    Promise.all(["credentials", "models", "models-all"].map((k) => qc.invalidateQueries({ queryKey: [k] })));
  useEffect(() => () => { if (poll.current) window.clearInterval(poll.current); }, []);

  const save = useMutation({ mutationFn: () => api.setApiKey(provider, key), onSuccess: () => { setKey(""); void refresh(); } });
  const removeKey = useMutation({ mutationFn: () => api.removeCredentials(provider, "api_key"), onSuccess: () => void refresh() });
  const signOut = useMutation({ mutationFn: () => api.logout(provider), onSuccess: () => void refresh() });

  function stopPolling() {
    if (poll.current) window.clearInterval(poll.current);
    poll.current = null;
    setWaiting(false);
  }

  async function chatgpt() {
    setSignInError(null);
    try {
      const { authorize_url } = await api.chatgptStart();
      window.open(authorize_url, "_blank", "noopener");
      setWaiting(true);
      const started = Date.now();
      poll.current = window.setInterval(async () => {
        const st = (await api.credentials()).providers.openai;
        if (st.has_subscription) { stopPolling(); void refresh(); }
        else if (Date.now() - started > 5 * 60_000) stopPolling();
      }, 2000);
    } catch (e) {
      setSignInError((e as Error).message);
    }
  }

  async function grok() {
    setSignInError(null);
    try {
      const d = await api.grokStart();
      setDevice({ uri: d.verification_uri_complete ?? d.verification_uri, code: d.user_code });
      setWaiting(true);
      poll.current = window.setInterval(async () => {
        const { status: st } = await api.grokStatus();
        if (st === "pending") return;
        stopPolling();
        setDevice(null);
        if (st === "approved") void refresh();
        else setSignInError(st === "denied" ? "Sign-in was denied." : "The sign-in code expired. Try again.");
      }, Math.max(1, d.interval) * 1000);
    } catch (e) {
      setSignInError((e as Error).message);
    }
  }

  return (
    <section className="rounded-lg border border-rule p-4" aria-labelledby={`p-${provider}`}>
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 id={`p-${provider}`} className="font-semibold">
          {info.title} <span className="font-normal text-ink-3">for {info.models}</span>
        </h3>
        <span className={`text-sm ${status.configured ? "text-ok" : "text-ink-3"}`}>{statusLine(status)}</span>
      </div>

      <form className="mt-3 flex gap-2" onSubmit={(e) => { e.preventDefault(); if (key.trim()) save.mutate(); }}>
        <label htmlFor={`key-${provider}`} className="sr-only">{info.title} API key</label>
        <input id={`key-${provider}`} type="password" autoComplete="off" value={key} onChange={(e) => setKey(e.target.value)}
          placeholder={status.has_api_key ? "Replace API key" : `Paste API key (${info.keyHint})`} className={inputClass} />
        <button type="submit" className={primaryButton} disabled={!key.trim() || save.isPending}>
          {save.isPending && <Loader2 size={14} className="animate-spin" />} Save key
        </button>
      </form>
      <div className="mt-1.5 flex flex-wrap gap-x-4 text-xs">
        <a href={info.keyUrl} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-accent">
          Get an API key <ExternalLink size={11} />
        </a>
        {status.has_api_key && (
          <button type="button" className="text-ink-3 hover:text-bad" onClick={() => removeKey.mutate()}>Remove API key</button>
        )}
      </div>
      {save.error && <p className="mt-1.5 text-sm text-bad" role="alert">{save.error.message}</p>}

      <div className="mt-4 border-t border-rule pt-3 text-sm">
        {provider === "anthropic" && <p className="text-ink-3">{status.subscription_note}</p>}

        {provider === "openai" && (status.has_subscription ? (
          <div className="flex flex-wrap items-center gap-3">
            <span>Signed in with ChatGPT{status.account_label ? ` as ${status.account_label}` : ""}.</span>
            <a href="https://chatgpt.com/settings/usage" target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-accent">
              Manage usage <ExternalLink size={11} />
            </a>
            <button type="button" className="text-ink-3 hover:text-bad" onClick={() => signOut.mutate()}>Sign out</button>
          </div>
        ) : (
          <div>
            <p className="mb-2 text-ink-3">Have ChatGPT Plus or Pro? Use your plan instead of a paid API key.</p>
            <button type="button" className={secondaryButton} onClick={chatgpt} disabled={waiting}>
              {waiting && <Loader2 size={14} className="animate-spin" />} Continue with ChatGPT
            </button>
            {waiting && <p className="mt-1.5 text-xs text-ink-3">Finish signing in in the new tab. This updates automatically.</p>}
          </div>
        ))}

        {provider === "xai" && (status.has_subscription ? (
          <div className="flex items-center gap-3">
            <span>Signed in with SuperGrok (experimental).</span>
            <button type="button" className="text-ink-3 hover:text-bad" onClick={() => signOut.mutate()}>Sign out</button>
          </div>
        ) : status.subscription_available ? (
          <div>
            <button type="button" className={secondaryButton} onClick={grok} disabled={waiting}>
              {waiting && <Loader2 size={14} className="animate-spin" />} Sign in with Grok (experimental)
            </button>
            {device && (
              <p className="mt-2">
                Open <a href={device.uri} target="_blank" rel="noreferrer" className="text-accent underline">{device.uri}</a> and
                enter <strong className="font-mono">{device.code}</strong>.
              </p>
            )}
            <p className="mt-1.5 text-xs text-ink-3">{status.subscription_note}</p>
          </div>
        ) : (
          <p className="text-ink-3">{status.subscription_note}</p>
        ))}
        {signInError && <p className="mt-1.5 text-bad" role="alert">{signInError}</p>}
      </div>
    </section>
  );
}
